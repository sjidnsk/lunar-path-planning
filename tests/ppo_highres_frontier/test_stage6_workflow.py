"""Stage 6 final eval、artifact、verifier、runner 与 claim boundary。"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import pickle
import stat
import subprocess
import threading
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_ppo_stage6_standard.py"
CONFIG = ROOT / "configs/ppo_highres_frontier_stage6_v1.json"


def _module():
    return importlib.import_module("lunar_exploration_ppo.workflows.stage6")


def _make_directory_reparse(link: Path, target: Path) -> None:
    if os.name != "nt":
        os.symlink(target, link, target_is_directory=True)
        return
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        pytest.fail(
            "Windows junction creation failed: "
            f"{completed.stdout} {completed.stderr}"
        )


def _make_file_symlink(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=False)
    except OSError as exc:
        pytest.skip(f"file symlink creation is unavailable: {exc}")


def test_final_evaluation_plan_has_exact_counts_shared_scenarios_and_theta() -> None:
    module = _module()
    assert module.FINAL_EVALUATION_METHODS == (
        "ppo_policy",
        "random_valid_frontier",
        "nearest_frontier",
        "max_potential_gain_frontier",
        "gain_over_cost_frontier",
    )
    scenarios = {
        "test": tuple(f"test/scenario-{index:04d}" for index in range(64)),
        "unseen": tuple(f"unseen/scenario-{index:04d}" for index in range(64)),
    }

    plan = module.build_final_evaluation_plan(scenarios)

    assert len(plan) == 640
    for split in ("test", "unseen"):
        split_jobs = [job for job in plan if job.split == split]
        assert len(split_jobs) == 320
        for method in module.FINAL_EVALUATION_METHODS:
            jobs = [job for job in split_jobs if job.method == method]
            assert len(jobs) == 64
            assert [job.scenario_id for job in jobs] == list(scenarios[split])
        by_episode: dict[int, set[tuple[str, int]]] = {}
        for job in split_jobs:
            by_episode.setdefault(job.episode_index, set()).add(
                (job.scenario_id, job.evaluation_seed)
            )
        assert all(len(shared) == 1 for shared in by_episode.values())
    assert all(
        job.theta_source == (
            "policy_theta_mu/v1"
            if job.method == "ppo_policy"
            else "candidate_recommended_theta/v1"
        )
        for job in plan
    )


def test_eval_only_hash_guard_and_no_advantage_do_not_block_machine_pass() -> None:
    module = _module()
    frozen = {
        "checkpoint_sha256": "a" * 64,
        "policy_state_sha256": "b" * 64,
        "optimizer_state_sha256": "c" * 64,
        "config_sha256": "d" * 64,
    }
    assert module.assert_eval_only_hashes_unchanged(frozen, dict(frozen)) == frozen
    with pytest.raises(module.Stage6WorkflowError, match="mutated"):
        module.assert_eval_only_hashes_unchanged(
            frozen,
            {**frozen, "policy_state_sha256": "e" * 64},
        )

    no_advantage = module.decide_performance_advantage(
        ppo_ci95_low=0.50,
        gain_over_cost_ci95_high=0.50,
    )
    assert no_advantage.established is False
    assert no_advantage.claim == "未建立性能优势"
    acceptance = module.build_machine_acceptance(
        system_gates_passed=True,
        performance=no_advantage,
    )
    assert acceptance["state"] == "awaiting_independent_review"
    assert acceptance["machine_passed"] is True
    assert acceptance["performance_advantage_established"] is False

    advantage = module.decide_performance_advantage(
        ppo_ci95_low=0.5000001,
        gain_over_cost_ci95_high=0.50,
    )
    assert advantage.established is True


def test_state_journal_is_append_only_hash_chained_and_tamper_evident(
    tmp_path: Path,
) -> None:
    from test_stage6_machine_preflight import (
        _environment_sha256,
        _valid_environment_identity,
    )

    module = _module()
    journal = module.Stage6StateJournal(tmp_path / "s6/job-state.jsonl")
    environment_identity = _valid_environment_identity()
    reviewed_tree = "1" * 40
    bindings = {
        "config_sha256": "1" * 64,
        "source_set_sha256": "2" * 64,
        "prospective_tree_sha256": hashlib.sha256(
            reviewed_tree.encode("ascii")
        ).hexdigest(),
        "data_sha256": "4" * 64,
        "environment_identity": environment_identity,
        "environment_sha256": _environment_sha256(environment_identity),
        "stage5_gate_sha256": "5" * 64,
        "formal_run_id": FORMAL_STAGE6_RUN_ID,
        "changed_path_set_sha256": "7" * 64,
        "review_authorization_record_sha256": "8" * 64,
        "authorization_file_sha256": "9" * 64,
        "review_identity_sha256": "a" * 64,
        "reviewed_prospective_git_tree": reviewed_tree,
        "frozen_diff_sha256": "b" * 64,
        "spec_review_sha256": "c" * 64,
        "quality_review_sha256": "d" * 64,
        "checkpoint_sha256": "6" * 64,
    }
    first = journal.append("preflight", bindings)
    second = journal.append("seed_20260716_initializing", bindings)
    records = journal.verify()

    assert len(records) == 2
    assert second["previous_record_hash"] == first["record_hash"]
    assert records[-1]["record_hash"] == second["record_hash"]

    payload = journal.path.read_text(encoding="utf-8")
    journal.path.write_text(
        payload.replace("seed_20260716_initializing", "seed_20260717_initializing"),
        encoding="utf-8",
    )
    with pytest.raises(module.Stage6WorkflowError, match="hash chain"):
        journal.verify()


def test_state_journal_verify_avoids_naked_target_path_read_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_stage6_machine_preflight import (
        _environment_sha256,
        _valid_environment_identity,
    )

    module = _module()
    path = tmp_path / "s6/job-state.jsonl"
    journal = module.Stage6StateJournal(path)
    environment_identity = _valid_environment_identity()
    reviewed_tree = "1" * 40
    bindings = {
        "config_sha256": "1" * 64,
        "source_set_sha256": "2" * 64,
        "prospective_tree_sha256": hashlib.sha256(
            reviewed_tree.encode("ascii")
        ).hexdigest(),
        "data_sha256": "4" * 64,
        "environment_identity": environment_identity,
        "environment_sha256": _environment_sha256(environment_identity),
        "stage5_gate_sha256": "5" * 64,
        "formal_run_id": FORMAL_STAGE6_RUN_ID,
        "changed_path_set_sha256": "7" * 64,
        "review_authorization_record_sha256": "8" * 64,
        "authorization_file_sha256": "9" * 64,
        "review_identity_sha256": "a" * 64,
        "reviewed_prospective_git_tree": reviewed_tree,
        "frozen_diff_sha256": "b" * 64,
        "spec_review_sha256": "c" * 64,
        "quality_review_sha256": "d" * 64,
        "checkpoint_sha256": "6" * 64,
    }
    journal.append("preflight", bindings)
    original_read_bytes = Path.read_bytes

    def reject_target_read(self: Path) -> bytes:
        if self == path:
            raise AssertionError("state journal used naked Path.read_bytes")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", reject_target_read)
    assert journal.verify()[0]["state"] == "preflight"


def _write_manifest_fixture(
    module,
    stage: Path,
    *,
    execution_identity: dict[str, object] | None = None,
    verified_review: dict[str, object] | None = None,
) -> None:
    stage.mkdir(parents=True)
    for relative in module.STAGE6_MANIFEST_BOUND_ARTIFACTS:
        path = stage / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((relative + "\n").encode("utf-8"))
    children = {
        "checkpoints/index.jsonl": b'{"receipt":1}\n',
        "checkpoints/seed-20260716/latest.json": b'{"latest":1}\n',
        "checkpoints/seed-20260716/update-00000010/checkpoint.pt": b"checkpoint",
        "checkpoints/seed-20260716/update-00000010/manifest.json": b'{"manifest":1}\n',
        "checkpoints/seed-20260716/update-00000010/complete.json": b'{"complete":1}\n',
        "episode-traces/final-test-ppo_policy.jsonl": b'{"episode":1}\n',
        "episode-traces/final-test-ppo_policy.summary.json": b'{"summary":1}\n',
        "episode-traces/final-test-ppo_policy.commit.json": b'{"commit":1}\n',
        "preflight/audit.json": b'{"passed":true}\n',
    }
    for relative, payload in children.items():
        path = stage / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    if execution_identity is None or verified_review is None:
        identity, verified = _review_identity_fixture(module)
    else:
        identity = dict(execution_identity)
        verified = dict(verified_review)
    immutable = {
        "config_sha256": identity["config_sha256"],
        "source_set_sha256": identity["source_set_sha256"],
        "prospective_tree_sha256": identity["prospective_tree_sha256"],
        "data_sha256": identity["data_sha256"],
        "environment_identity": identity["environment_identity"],
        "environment_sha256": identity["environment_sha256"],
        "stage5_gate_sha256": "d" * 64,
        "formal_run_id": verified["formal_run_id"],
        "changed_path_set_sha256": verified["changed_path_set_sha256"],
        "review_authorization_record_sha256": hashlib.sha256(
            module.ArtifactStore.canonical_json_bytes(verified)
        ).hexdigest(),
        "authorization_file_sha256": verified["authorization_file_sha256"],
        "review_identity_sha256": verified["review_identity_sha256"],
        "reviewed_prospective_git_tree": verified[
            "reviewed_prospective_git_tree"
        ],
        "frozen_diff_sha256": verified["frozen_diff_sha256"],
        "spec_review_sha256": verified["spec_review_sha256"],
        "quality_review_sha256": verified["quality_review_sha256"],
    }
    (stage / "lineage_audit.json").write_bytes(
        module.ArtifactStore.canonical_json_bytes(
            {
                "schema_version": "stage6_lineage_audit/v2",
                "execution_identity": identity,
                "immutable_bindings": immutable,
                "verified_review_authorization": verified,
            }
        )
    )
    (stage / "summary.json").write_bytes(
        module.ArtifactStore.canonical_json_bytes(
            {
                "schema_version": "stage6_summary/v1",
                "machine_passed": True,
                "state": "awaiting_independent_review",
            }
        )
    )
    (stage / "routing.json").write_bytes(
        module.ArtifactStore.canonical_json_bytes(
            {
                "schema_version": "stage6_routing/v1",
                "machine_passed": True,
                "route": "awaiting_independent_review",
            }
        )
    )
    stage_stat = stage.lstat()
    evidence_files = []
    for relative in ("config.json", "phase-state.jsonl", "resource_audit.jsonl"):
        path = stage / relative
        payload = path.read_bytes()
        metadata = path.lstat()
        evidence_files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "identity": {
                    "device": int(metadata.st_dev),
                    "inode": int(metadata.st_ino),
                    "file_type": int(stat.S_IFMT(metadata.st_mode)),
                    "size_bytes": int(metadata.st_size),
                    "file_attributes": int(
                        getattr(metadata, "st_file_attributes", 0)
                    ),
                    "link_count": int(metadata.st_nlink),
                },
            }
        )
    evidence_directories = [
        {
            "path": ".",
            "identity": {
                "device": int(stage_stat.st_dev),
                "inode": int(stage_stat.st_ino),
                "mode": int(stage_stat.st_mode),
                "file_attributes": int(
                    getattr(stage_stat, "st_file_attributes", 0)
                ),
                "link_count": int(stage_stat.st_nlink),
            },
            "members": [
                {"name": relative, "kind": "file"}
                for relative in (
                    "config.json",
                    "phase-state.jsonl",
                    "resource_audit.jsonl",
                )
            ],
        }
    ]
    evidence_graph = {
        "files": evidence_files,
        "directories": evidence_directories,
    }
    evidence_binding = {
        "schema_version": "stage6_preterminal_evidence_binding/v1",
        "graph_sha256": hashlib.sha256(
            module.ArtifactStore.canonical_json_bytes(evidence_graph)
        ).hexdigest(),
        **evidence_graph,
    }
    semantic = {
        "schema_version": "stage6_machine_acceptance_verification/v2",
        "passed": True,
        "state": "awaiting_independent_review",
        "final_evaluation_count": 10,
        "final_episode_count": 640,
        "checkpoint_receipt_count": 100,
        "global_best_policy_state_sha256": "8" * 64,
        "evidence_binding": evidence_binding,
    }
    receipt_immutable = copy.deepcopy(immutable)
    receipt_immutable["environment_sha256"] = hashlib.sha256(
        module.ArtifactStore.canonical_json_bytes(
            receipt_immutable["environment_identity"]
        )
    ).hexdigest()
    global_checkpoint = {
        "schema_version": "stage6_global_best/v1",
        "record": {
            "seed": 20260716,
            "update": 100,
            "success_rate_under_fixed_step_budget": 0.75,
            "mean_final_coverage": 0.625,
            "checkpoint_ref": "checkpoints/seed-20260716/update-00000100",
        },
        "transaction_key": "seed:20260716:update:100",
        "checkpoint_sha256": "6" * 64,
        "complete_marker_sha256": "7" * 64,
        "policy_state_sha256": "8" * 64,
    }
    terminal_artifacts = []
    for relative in (
        "summary.json",
        "routing.json",
        "standard_training_report.md",
        "standard_eval_report.md",
        "report.md",
    ):
        payload = (stage / relative).read_bytes()
        terminal_artifacts.append(
            {
                "path": relative,
                "utf8": payload.decode("utf-8"),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    resource_payload = (stage / "resource_audit.jsonl").read_bytes()
    phase_payload = (stage / "phase-state.jsonl").read_bytes()
    (stage / "preterminal_acceptance.json").write_bytes(
        module.ArtifactStore.canonical_json_bytes(
            {
                "schema_version": "stage6_preterminal_acceptance/v2",
                "semantic_verification": semantic,
                "preterminal_evidence_binding": evidence_binding,
                "immutable_bindings": receipt_immutable,
                "global_checkpoint_identity": global_checkpoint,
                "terminal_artifacts": terminal_artifacts,
                "preterminal_artifact_graph": [
                    {
                        "path": row["path"],
                        "sha256": row["sha256"],
                        "size_bytes": row["size_bytes"],
                    }
                    for row in evidence_files
                ],
                "preterminal_directory_graph": evidence_directories,
                "resource_audit_prefix": {
                    "sha256": hashlib.sha256(resource_payload).hexdigest(),
                    "size_bytes": len(resource_payload),
                },
                "phase_state_prefix": {
                    "sha256": hashlib.sha256(phase_payload).hexdigest(),
                    "size_bytes": len(phase_payload),
                },
            }
        )
    )


def _write_authorized_manifest(module, stage: Path, active: dict[str, object]) -> Path:
    return module.write_stage6_manifest(
        stage_root=stage,
        repo_root=ROOT,
        execution_capability=active["capability"],
        run_lease=active["lease"],
    )


def test_single_seed_manifest_children_are_allowed(tmp_path: Path) -> None:
    module = _module()
    allowed = (
        "checkpoints/seed-20260716/latest.json",
        "checkpoints/seed-20260716/update-00000010/checkpoint.pt",
        "checkpoints/seed-20260716/update-00000010/manifest.json",
        "checkpoints/seed-20260716/update-00000010/complete.json",
        "episode-traces/validation-seed-20260716-update-010.jsonl",
    )

    assert all(module._is_allowed_stage6_child(relative) for relative in allowed)
    stage = tmp_path / "single-seed/s6"
    _write_manifest_fixture(module, stage)
    validation_trace = stage / allowed[-1]
    validation_trace.parent.mkdir(parents=True, exist_ok=True)
    validation_trace.write_bytes(b'{"episode":1}\n')
    assert set(allowed).issubset(module._stage6_manifest_relative_paths(stage))


def test_manifest_graph_optionally_binds_complete_source_repair_chain(
    tmp_path: Path,
) -> None:
    module = _module()
    stage = tmp_path / FORMAL_STAGE6_RUN_ID / "s6"
    _write_manifest_fixture(module, stage)
    for relative in (
        "source-repair-amendment.json",
        "source-repair-continuation.json",
        "source-repair-supplement.json",
        "source-repair-closure.json",
        "source-repair-frontier-recovery.json",
        "source-repair-sensor-acceleration.json",
    ):
        (stage / relative).write_bytes((relative + "\n").encode("utf-8"))

    paths = set(module._stage6_manifest_relative_paths(stage))

    assert {
        "source-repair-amendment.json",
        "source-repair-continuation.json",
        "source-repair-supplement.json",
        "source-repair-closure.json",
        "source-repair-frontier-recovery.json",
        "source-repair-sensor-acceleration.json",
    }.issubset(paths)


def _mark_manifest_fixture_as_planning_warm_start(
    module,
    stage: Path,
) -> None:
    lineage_path = stage / "lineage_audit.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    warm_artifact = {
        "schema_version": "stage6_planning_u74_warm_start/v1",
        "mode": "child_lineage_warm_start_not_exact_resume/v1",
    }
    warm_payload = module.ArtifactStore.canonical_json_bytes(warm_artifact)
    lineage["schema_version"] = "stage6_lineage_audit/v3"
    lineage["planning_warm_start"] = {
        "artifact": warm_artifact,
        "artifact_sha256": hashlib.sha256(warm_payload).hexdigest(),
        "artifact_size_bytes": len(warm_payload),
    }
    lineage_path.write_bytes(
        module.ArtifactStore.canonical_json_bytes(lineage)
    )


def test_manifest_graph_binds_planning_child_repair_as_independent_profile_artifact(
    tmp_path: Path,
) -> None:
    module = _module()
    stage = tmp_path / FORMAL_STAGE6_RUN_ID / "s6"
    _write_manifest_fixture(module, stage)
    _mark_manifest_fixture_as_planning_warm_start(module, stage)
    child_payload = module.ArtifactStore.canonical_json_bytes(
        {
            "schema_version": "stage6_planning_child_source_repair/v1",
            "canonical_sha256": "a" * 64,
        }
    )
    (stage / "planning-child-source-repair.json").write_bytes(
        child_payload
    )

    paths = module._stage6_manifest_relative_paths(stage)
    entries = module._build_stage6_manifest_entries(stage, paths)
    child_rows = [
        row
        for row in entries
        if row["path"] == "planning-child-source-repair.json"
    ]

    assert len(child_rows) == 1
    assert child_rows[0] == {
        "path": "planning-child-source-repair.json",
        "sha256": hashlib.sha256(child_payload).hexdigest(),
        "size_bytes": len(child_payload),
    }
    assert (
        "planning-child-source-repair.json"
        not in module.STAGE6_OPTIONAL_MANIFEST_BOUND_ARTIFACTS
    )


def test_manifest_graph_rejects_planning_child_without_warm_start(
    tmp_path: Path,
) -> None:
    module = _module()
    stage = tmp_path / FORMAL_STAGE6_RUN_ID / "s6"
    _write_manifest_fixture(module, stage)
    (stage / "planning-child-source-repair.json").write_bytes(b"{}\n")

    with pytest.raises(
        module.Stage6WorkflowError,
        match="planning child.*warm-start",
    ):
        module._stage6_manifest_relative_paths(stage)


def test_manifest_graph_rejects_classic_and_planning_child_repair_together(
    tmp_path: Path,
) -> None:
    module = _module()
    stage = tmp_path / FORMAL_STAGE6_RUN_ID / "s6"
    _write_manifest_fixture(module, stage)
    _mark_manifest_fixture_as_planning_warm_start(module, stage)
    (stage / "planning-child-source-repair.json").write_bytes(b"{}\n")
    (stage / "source-repair-amendment.json").write_bytes(b"{}\n")

    with pytest.raises(
        module.Stage6WorkflowError,
        match="planning child.*classic|classic.*planning child",
    ):
        module._stage6_manifest_relative_paths(stage)


def test_terminal_evidence_graph_binds_planning_child_and_rejects_mixed_profile(
    tmp_path: Path,
) -> None:
    module = _module()
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    stage = tmp_path / FORMAL_STAGE6_RUN_ID / "s6"
    stage.mkdir(parents=True)
    for relative in ("resource_audit.jsonl", "phase-state.jsonl"):
        (stage / relative).write_bytes(b"{}\n")
    warm_artifact = {
        "schema_version": "stage6_planning_u74_warm_start/v1",
        "mode": "child_lineage_warm_start_not_exact_resume/v1",
    }
    warm_payload = module.ArtifactStore.canonical_json_bytes(warm_artifact)
    lineage_path = stage / "lineage_audit.json"
    lineage_path.write_bytes(
        module.ArtifactStore.canonical_json_bytes(
            {
                "schema_version": "stage6_lineage_audit/v3",
                "planning_warm_start": {
                    "artifact": warm_artifact,
                    "artifact_sha256": hashlib.sha256(
                        warm_payload
                    ).hexdigest(),
                    "artifact_size_bytes": len(warm_payload),
                },
            }
        )
    )
    child_payload = module.ArtifactStore.canonical_json_bytes(
        {
            "schema_version": "stage6_planning_child_source_repair/v1",
            "canonical_sha256": "b" * 64,
        }
    )
    child_path = stage / "planning-child-source-repair.json"
    child_path.write_bytes(child_payload)
    u84_path = (
        stage
        / "checkpoints/seed-20260716/update-00000084/checkpoint.pt"
    )
    u84_path.parent.mkdir(parents=True)
    u84_path.write_bytes(b"immutable-u84-checkpoint")
    lineage_before = lineage_path.read_bytes()
    u84_before = u84_path.read_bytes()

    with stage6_terminal_recovery._capture_preterminal_evidence_handle(
        stage
    ) as evidence:
        child_rows = [
            row
            for row in evidence.graph_snapshot()
            if row["path"] == "planning-child-source-repair.json"
        ]
        assert child_rows == [
            {
                "path": "planning-child-source-repair.json",
                "sha256": hashlib.sha256(child_payload).hexdigest(),
                "size_bytes": len(child_payload),
            }
        ]
        with pytest.raises(OSError):
            child_path.write_bytes(child_payload + b"tamper")
        evidence.require_current("planning child terminal evidence")

    assert lineage_path.read_bytes() == lineage_before
    assert u84_path.read_bytes() == u84_before

    (stage / "source-repair-amendment.json").write_bytes(b"{}\n")
    with pytest.raises(
        stage6_terminal_recovery.TerminalRecoveryError,
        match="planning child.*classic|classic.*planning child",
    ):
        stage6_terminal_recovery._capture_preterminal_evidence_handle(stage)


def test_terminal_evidence_graph_rejects_planning_child_without_warm_start(
    tmp_path: Path,
) -> None:
    module = _module()
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    stage = tmp_path / FORMAL_STAGE6_RUN_ID / "s6"
    stage.mkdir(parents=True)
    for relative in ("resource_audit.jsonl", "phase-state.jsonl"):
        (stage / relative).write_bytes(b"{}\n")
    (stage / "lineage_audit.json").write_bytes(
        module.ArtifactStore.canonical_json_bytes(
            {"schema_version": "stage6_lineage_audit/v2"}
        )
    )
    (stage / "planning-child-source-repair.json").write_bytes(b"{}\n")

    with pytest.raises(
        stage6_terminal_recovery.TerminalRecoveryError,
        match="planning child.*warm-start",
    ):
        stage6_terminal_recovery._capture_preterminal_evidence_handle(stage)


def test_machine_verifier_wires_planning_child_dual_lineage_contract() -> None:
    module = _module()
    source = inspect.getsource(
        module._verify_stage6_machine_acceptance_from_bound_graph
    )

    assert "load_planning_child_source_repair_artifact(" not in source
    assert "planning_child_recovery_capability" in source
    assert "_validate_planning_child_machine_context(" in source
    assert "_validate_planning_child_resource_boundary(" in source
    assert "_stage6_planning_checkpoint_lineage_for_update(" in source
    assert "planning_child_source_repair_binding" in source
    assert "journal_binding_epochs" in source


def test_terminal_receipt_verifier_wires_planning_child_identity_split() -> None:
    module = _module()
    source = inspect.getsource(module.verify_stage6_machine_acceptance)

    assert "load_planning_child_source_repair_artifact(" not in source
    assert "planning_child_recovery_capability" in source
    assert "_validate_planning_child_machine_context(" in source
    assert "_require_planning_child_recovery_binding(" in source
    assert "historical_immutable_bindings" in source
    assert "current_immutable_bindings" in source


def _task5_jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _task5_write_json(path: Path, value: object) -> bytes:
    payload = _module().ArtifactStore.canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _task5_collection_audit(
    *,
    policy_sha256: str,
    update: int,
) -> dict[str, object]:
    return {
        "schema_version": "stage4_spawn_collector/v2",
        "worker_pids": list(range(30_001, 30_009)),
        "worker_start_methods": ["spawn"] * 8,
        "per_env_trainable_counts": [128] * 8,
        "diagnostic_reset_counts": [0] * 8,
        "trainable_transition_count": 1024,
        "inference_pids": [os.getpid()],
        "inference_batch_count": 128,
        "policy_device": "cuda:0",
        "policy_state_sha256": policy_sha256,
        "snapshot_sha256": [
            hashlib.sha256(
                f"task5:{update}:snapshot:{index}".encode("ascii")
            ).hexdigest()
            for index in range(1024)
        ],
        "terminal_transition_count": 0,
        "reset_diagnostics": [],
        "planner_failure_counts": {
            "endpoint_physical_unsafe": 0,
            "endpoint_unknown_buffer_unsafe": 0,
            "path_physical_unsafe": 0,
            "path_unknown_buffer_unsafe": 0,
            "planner_no_path": 0,
        },
    }


def _task5_review_record(
    module: object,
    *,
    execution_identity: dict[str, object],
    formal_run_id: str,
    authorization_payload: bytes,
) -> dict[str, object]:
    _, template = _review_identity_fixture(module)
    record = dict(template)
    record.update(
        {
            "formal_run_id": formal_run_id,
            "authorization_file_sha256": hashlib.sha256(
                authorization_payload
            ).hexdigest(),
            "authorization_file_size_bytes": len(authorization_payload),
            "review_identity_sha256": hashlib.sha256(
                module.ArtifactStore.canonical_json_bytes(
                    execution_identity
                )
            ).hexdigest(),
            "reviewed_prospective_git_tree": execution_identity[
                "prospective_git_tree"
            ],
            "prospective_tree_sha256": execution_identity[
                "prospective_tree_sha256"
            ],
            "changed_path_set_sha256": execution_identity[
                "changed_path_set_sha256"
            ],
            "source_set_sha256": execution_identity[
                "source_set_sha256"
            ],
            "config_sha256": execution_identity["config_sha256"],
            "data_sha256": execution_identity["data_sha256"],
            "environment_sha256": execution_identity[
                "environment_sha256"
            ],
        }
    )
    return record


def _task5_origin_immutable(
    *,
    identity: dict[str, object],
    review: dict[str, object],
    stage5_gate_sha256: str,
) -> dict[str, object]:
    from lunar_exploration_ppo.ppo import standard_training

    result = {
        key: identity[key]
        for key in (
            "config_sha256",
            "source_set_sha256",
            "prospective_tree_sha256",
            "data_sha256",
            "environment_identity",
            "environment_sha256",
        )
    }
    result["stage5_gate_sha256"] = stage5_gate_sha256
    result.update(
        standard_training._review_authorization_immutable_bindings(review)
    )
    return result


def _build_task5_planning_child_machine_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_startup_only_resource_segment_two: bool = False,
    retained_checkpoint_updates: tuple[int, ...] = (84, 90, 100),
) -> SimpleNamespace:
    """Build one isolated child graph while keeping Task5-critical paths real."""

    from dataclasses import replace

    import torch

    from lunar_exploration_ppo.configs.stage6 import SafetyContract
    from lunar_exploration_ppo.ppo import (
        checkpoint as checkpoint_module,
        checkpoint_retention,
        standard_training,
        trainer as trainer_module,
    )
    from lunar_exploration_ppo.workflows import (
        stage6_planning_child_source_repair as child_module,
        stage6_planning_warm_start as warm_module,
        stage6_terminal_recovery,
    )
    from test_stage6_planning_child_source_repair import (
        FORMAL_RUN_ID,
        SEED,
        _build,
        _build_fake_planner_runtime_identity,
        _fixture,
        _rehash_artifact,
    )
    from test_stage6_planning_warm_start import _artifact as _warm_artifact

    module = _module()
    fixture = _fixture(tmp_path)
    stage = fixture.stage_root

    effective_config_bytes = warm_module.planning_effective_config_bytes(
        CONFIG.read_bytes()
    )
    effective = warm_module.parse_planning_effective_config_bytes(
        effective_config_bytes
    )
    config = effective.base_config
    (stage / "config.json").write_bytes(effective_config_bytes)
    current_identity = module.stage6_execution_identity(
        repo_root=ROOT,
        config_path=CONFIG,
        effective_config_bytes=effective_config_bytes,
    )

    authority = SimpleNamespace(
        identity={
            "gate_sha256": config.stage5_authority.gate_sha256,
            "commit_sha256": config.stage5_authority.commit_sha256,
            "commit_tree": config.stage5_authority.commit_tree,
            "review_sha256": config.stage5_authority.review_sha256,
            "manifest_sha256": config.stage5_authority.manifest_sha256,
            "checkpoint_sha256": config.stage5_authority.checkpoint_sha256,
            "policy_state_sha256": (
                config.stage5_authority.policy_state_sha256
            ),
            "performance_advantage_established": False,
        },
        require_current=lambda label: None,
    )
    monkeypatch.setattr(
        module,
        "verify_frozen_stage5_authority",
        lambda **kwargs: authority,
    )

    origin_identity = copy.deepcopy(current_identity)
    origin_tree = "0" * 40
    origin_identity.update(
        {
            "formal_run_id": FORMAL_RUN_ID,
            "prospective_git_tree": origin_tree,
            "prospective_tree_sha256": hashlib.sha256(
                origin_tree.encode("ascii")
            ).hexdigest(),
            "changed_path_set_sha256": "1" * 64,
            "source_set_sha256": "2" * 64,
        }
    )
    origin_authorization_path = fixture.warm_start_path.parent / (
        "launch-authorization.json"
    )
    origin_authorization_payload = module.ArtifactStore.canonical_json_bytes(
        {
            "authorized": True,
            "formal_run_id": FORMAL_RUN_ID,
            "fixture": "task5-origin",
        }
    )
    origin_authorization_path.write_bytes(origin_authorization_payload)
    origin_review = _task5_review_record(
        module,
        execution_identity=origin_identity,
        formal_run_id=FORMAL_RUN_ID,
        authorization_payload=origin_authorization_payload,
    )
    origin_immutable = _task5_origin_immutable(
        identity=origin_identity,
        review=origin_review,
        stage5_gate_sha256=config.stage5_authority.gate_sha256,
    )

    current_authorization_payload = module.ArtifactStore.canonical_json_bytes(
        {
            "authorized": True,
            "formal_run_id": FORMAL_RUN_ID,
            "fixture": "task5-current",
        }
    )
    fixture.current_authorization_path.write_bytes(
        current_authorization_payload
    )
    current_review = _task5_review_record(
        module,
        execution_identity=current_identity,
        formal_run_id=FORMAL_RUN_ID,
        authorization_payload=current_authorization_payload,
    )
    current_immutable = module._stage6_current_immutable_bindings(
        execution_identity=current_identity,
        verified_review_authorization=current_review,
        stage5_authority=authority,
    )

    warm_artifact = _warm_artifact()
    warm_artifact["child"] = {
        **warm_artifact["child"],
        "run_id": FORMAL_RUN_ID,
        "seed": SEED,
        "effective_config_sha256": hashlib.sha256(
            effective_config_bytes
        ).hexdigest(),
    }
    warm_artifact["bindings"] = (
        warm_module.planning_warm_start_expected_bindings_from_record(
            repo_root=ROOT,
            review_authorization_record=origin_review,
        )
    )
    warm_payload = (
        json.dumps(
            warm_artifact,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    fixture.warm_start_path.write_bytes(warm_payload)
    origin_lineage = {
        "schema_version": "stage6_lineage_audit/v3",
        "execution_identity": origin_identity,
        "immutable_bindings": origin_immutable,
        "verified_review_authorization": origin_review,
        "planning_warm_start": {
            "artifact": warm_artifact,
            "artifact_sha256": hashlib.sha256(warm_payload).hexdigest(),
            "artifact_size_bytes": len(warm_payload),
        },
    }
    origin_lineage_payload = _task5_write_json(
        stage / "lineage_audit.json",
        origin_lineage,
    )

    transactions = warm_module.build_planning_child_transactions(config)

    def digest(update: int, label: str) -> str:
        return hashlib.sha256(
            f"task5:{update}:{label}".encode("utf-8")
        ).hexdigest()

    checkpoint_root = (
        stage
        / "checkpoints"
        / f"seed-{SEED}"
        / "update-00000084"
    )
    checkpoint_payload = (checkpoint_root / "checkpoint.pt").read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_payload).hexdigest()
    u84_policy_sha256 = digest(84, "policy")
    u84_manifest_payload = _task5_write_json(
        checkpoint_root / "manifest.json",
        {
            "schema_version": "stage6_checkpoint_manifest/v1",
            "update_step": 84,
            "checkpoint": {
                "path": "checkpoint.pt",
                "size_bytes": len(checkpoint_payload),
                "sha256": checkpoint_sha256,
            },
            "policy_state_sha256": u84_policy_sha256,
            "config_sha256": hashlib.sha256(
                effective_config_bytes
            ).hexdigest(),
            "lineage_sha256": hashlib.sha256(
                module.ArtifactStore.canonical_json_bytes(
                    fixture.u84_lineage
                )
            ).hexdigest(),
        },
    )
    u84_complete_payload = _task5_write_json(
        checkpoint_root / "complete.json",
        {
            "schema_version": "stage6_checkpoint_complete/v1",
            "update_step": 84,
            "checkpoint_sha256": checkpoint_sha256,
            "manifest_sha256": hashlib.sha256(
                u84_manifest_payload
            ).hexdigest(),
        },
    )

    if 84 not in retained_checkpoint_updates or not set(
        retained_checkpoint_updates
    ).issubset({84, 90, 100}):
        raise AssertionError("Task5 retained checkpoint fixture is invalid")
    retained_payloads: dict[int, tuple[bytes, bytes, bytes]] = {
        84: (
            checkpoint_payload,
            u84_manifest_payload,
            u84_complete_payload,
        )
    }
    for update in (90, 100):
        if update not in retained_checkpoint_updates:
            continue
        directory = (
            stage
            / "checkpoints"
            / f"seed-{SEED}"
            / f"update-{update:08d}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        payloads = (
            f"task5-checkpoint-{update}\n".encode("ascii"),
            b"{}\n",
            f"task5-complete-{update}\n".encode("ascii"),
        )
        for name, payload in zip(
            ("checkpoint.pt", "manifest.json", "complete.json"),
            payloads,
            strict=True,
        ):
            (directory / name).write_bytes(payload)
        retained_payloads[update] = payloads

    checkpoints: list[standard_training.StandardTransactionCheckpoint] = []
    for transaction in transactions:
        update = transaction.update
        if update in retained_payloads:
            checkpoint_bytes, _, complete_bytes = retained_payloads[update]
            checkpoint_value = hashlib.sha256(checkpoint_bytes).hexdigest()
            complete_value = hashlib.sha256(complete_bytes).hexdigest()
        else:
            checkpoint_value = digest(update, "checkpoint")
            complete_value = digest(update, "complete")
        checkpoints.append(
            standard_training.StandardTransactionCheckpoint(
                transaction_key=transaction.key,
                seed=transaction.seed,
                update=update,
                checkpoint_sha256=checkpoint_value,
                complete_marker_sha256=complete_value,
                policy_state_sha256=(
                    u84_policy_sha256
                    if update == 84
                    else digest(update, "policy")
                ),
            )
        )
    checkpoint_by_update = {
        checkpoint.update: checkpoint for checkpoint in checkpoints
    }

    receipt_path = stage / "checkpoints" / "index.jsonl"
    receipt_path.write_bytes(b"")
    receipt_index = standard_training.CheckpointReceiptIndex(receipt_path)
    job_path = stage / "job-state.jsonl"
    job_path.write_bytes(b"")
    job_journal = module.Stage6StateJournal(job_path)
    for transaction, checkpoint in zip(
        transactions[:10],
        checkpoints[:10],
        strict=True,
    ):
        assert receipt_index.append_once(checkpoint) is True
        bindings = {
            **origin_immutable,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
        }
        for state in transaction.commit_states:
            job_journal.append(state, bindings)

    def training_row(
        transaction: object,
        checkpoint: object,
        policy_before: str,
    ) -> dict[str, object]:
        update = int(transaction.update)
        validation = (
            {
                "episode_count": 16,
                "success_rate_under_fixed_step_budget": {
                    80: 0.70,
                    90: 0.90,
                    100: 0.80,
                }.get(update, 0.0),
                "mean_final_coverage": {
                    80: 0.70,
                    90: 0.90,
                    100: 0.80,
                }.get(update, 0.0),
            }
            if transaction.validation_episodes == 16
            else {}
        )
        math_evidence = {
            "evidence_sha256": digest(update, "math-evidence"),
            "snapshot_list_sha256": digest(update, "snapshot-list"),
        }
        return {
            "transaction_key": transaction.key,
            "seed": transaction.seed,
            "update": update,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "policy_state_sha256": checkpoint.policy_state_sha256,
            "update_metrics": {
                "update_step": update,
                "policy_state_sha256_before": policy_before,
                "policy_state_sha256_after": (
                    checkpoint.policy_state_sha256
                ),
                "optimizer_steps": 1,
                "math_evidence": math_evidence,
            },
            "validation": validation,
            "collection_audit": _task5_collection_audit(
                policy_sha256=policy_before,
                update=update,
            ),
        }

    training_rows: list[dict[str, object]] = []
    policy_before = warm_module.PARENT_POLICY_STATE_SHA256
    for transaction, checkpoint in zip(
        transactions,
        checkpoints,
        strict=True,
    ):
        training_rows.append(
            training_row(transaction, checkpoint, policy_before)
        )
        policy_before = checkpoint.policy_state_sha256
    training_prefix = _task5_jsonl_bytes(training_rows[:10])
    (stage / "training_metrics.jsonl").write_bytes(training_prefix)

    validation_rows = []
    best: dict[str, object] = {}
    for update in (80, 90, 100):
        transaction = next(
            item for item in transactions if item.update == update
        )
        training = training_rows[transaction.sequence]
        result = dict(training["validation"])
        best = standard_training.select_validation_checkpoint_best(
            seed=SEED,
            update=update,
            validation_metrics=result,
            previous_best=best,
        )
        validation_rows.append(
            {
                "transaction_key": transaction.key,
                "seed": SEED,
                "update": update,
                "episode_count": 16,
                "result": result,
                "eval_isolation": {"passed": True},
                "best_record": dict(best),
            }
        )
    validation_prefix = _task5_jsonl_bytes(validation_rows[:1])
    (stage / "validation_metrics.jsonl").write_bytes(validation_prefix)

    resource_path = stage / "resource_audit.jsonl"
    resource_rows = [
        json.loads(line)
        for line in resource_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in resource_rows:
        if (
            row.get("kind") == "update"
            and row.get("phase") == "accepted"
        ):
            checkpoint = checkpoint_by_update[int(row["update"])]
            row["checkpoint"] = {
                "seed": SEED,
                "update": checkpoint.update,
                "transaction_key": checkpoint.transaction_key,
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "complete_marker_sha256": (
                    checkpoint.complete_marker_sha256
                ),
                "policy_state_sha256": checkpoint.policy_state_sha256,
            }
    resource_prefix = _task5_jsonl_bytes(resource_rows)
    if include_startup_only_resource_segment_two:
        startup_segment_id = "0123456789ab4def8123456789abcdef"
        resource_prefix += _task5_jsonl_bytes(
            [
                {
                    "schema_version": (
                        "stage6_resource_lifecycle_segment/v1"
                    ),
                    "kind": "resource_lifecycle_segment",
                    "phase": "segment_start",
                    "segment_id": startup_segment_id,
                    "segment_index": 2,
                    "root_pid": 4242,
                    "first_sample": {
                        "rss_source": (
                            "process_tree_lifecycle_peak_current_sum/v1"
                        ),
                        "rss_root_pid": 4242,
                        "rss_sample_count": 1,
                        "rss_bytes": 1001,
                        "passed": True,
                    },
                    "prior_resource_log": {
                        "size_bytes": len(resource_prefix),
                        "sha256": hashlib.sha256(
                            resource_prefix
                        ).hexdigest(),
                    },
                }
            ]
        )
    resource_path.write_bytes(resource_prefix)

    planner_args = _build_fake_planner_runtime_identity(tmp_path)
    _, package_root, resolved_symbols, expected_paths = planner_args

    def current_planner_identity() -> dict[str, object]:
        return child_module.build_legacy_path_planner_runtime_identity(
            package_root=package_root,
            distribution_version="0.1.0",
            resolved_symbol_paths=resolved_symbols,
            loaded_module_names=(),
        )

    assert tuple(
        row["path"] for row in current_planner_identity()["paths"]
    ) == expected_paths
    monkeypatch.setattr(
        child_module,
        "resolve_legacy_path_planner_runtime_identity",
        current_planner_identity,
    )

    builder_identity = {
        **current_identity,
        "formal_run_id": FORMAL_RUN_ID,
    }
    builder_review = _task5_review_record(
        module,
        execution_identity=builder_identity,
        formal_run_id=FORMAL_RUN_ID,
        authorization_payload=current_authorization_payload,
    )
    builder_immutable = module._stage6_current_immutable_bindings(
        execution_identity=builder_identity,
        verified_review_authorization=builder_review,
        stage5_authority=authority,
    )
    builder_fixture = replace(
        fixture,
        current_execution_identity=builder_identity,
        current_verified_review_authorization=builder_review,
        current_immutable_bindings=builder_immutable,
        origin_lineage_bytes=origin_lineage_payload,
        u84_checkpoint_sha256=checkpoint_sha256,
        u84_manifest_sha256=hashlib.sha256(
            u84_manifest_payload
        ).hexdigest(),
        u84_complete_sha256=hashlib.sha256(
            u84_complete_payload
        ).hexdigest(),
        u84_policy_state_sha256=u84_policy_sha256,
    )
    child_artifact = _build(builder_fixture)
    child_artifact["current"]["execution_identity"] = current_identity
    child_artifact["current"]["execution_identity_sha256"] = hashlib.sha256(
        module.ArtifactStore.canonical_json_bytes(current_identity)
    ).hexdigest()
    child_artifact["current"][
        "verified_review_authorization"
    ] = current_review
    child_artifact["current"]["immutable_bindings"] = current_immutable
    child_artifact = _rehash_artifact(child_artifact)
    child_path = stage / "planning-child-source-repair.json"
    child_payload = module.ArtifactStore.canonical_json_bytes(
        child_artifact
    )
    child_path.write_bytes(child_payload)
    continuation_path = (
        stage / "planning-child-source-repair-continuation.json"
    )
    continuation_payload = module.ArtifactStore.canonical_json_bytes(
        {
            "schema_version": (
                "task1_planning_child_recovery_fixture/v1"
            ),
            "parent_artifact_sha256": hashlib.sha256(
                child_payload
            ).hexdigest(),
        }
    )
    continuation_path.write_bytes(continuation_payload)

    for transaction, checkpoint in zip(
        transactions[10:],
        checkpoints[10:],
        strict=True,
    ):
        assert receipt_index.append_once(checkpoint) is True
        bindings = {
            **current_immutable,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
        }
        for state in transaction.commit_states:
            job_journal.append(state, bindings)
    (stage / "training_metrics.jsonl").write_bytes(
        training_prefix + _task5_jsonl_bytes(training_rows[10:])
    )
    (stage / "validation_metrics.jsonl").write_bytes(
        validation_prefix + _task5_jsonl_bytes(validation_rows[1:])
    )

    segment_id = "abcdef0123454789abcdef0123456789"
    completed_segment_index = (
        3 if include_startup_only_resource_segment_two else 2
    )

    def snapshot(sample: int) -> dict[str, object]:
        return {
            "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
            "rss_root_pid": 5252,
            "rss_sample_count": sample,
            "rss_bytes": 1000 + sample,
            "passed": True,
        }

    tail_resource_rows: list[dict[str, object]] = [
        {
            "schema_version": "stage6_resource_lifecycle_segment/v1",
            "kind": "resource_lifecycle_segment",
            "phase": "segment_start",
            "segment_id": segment_id,
            "segment_index": completed_segment_index,
            "root_pid": 5252,
            "first_sample": snapshot(1),
            "prior_resource_log": {
                "size_bytes": len(resource_prefix),
                "sha256": hashlib.sha256(resource_prefix).hexdigest(),
            },
        }
    ]
    for transaction in transactions[10:]:
        checkpoint = checkpoint_by_update[transaction.update]
        attempt = 2 if transaction.update == 85 else 1
        pre = snapshot(2 * transaction.sequence + 2)
        post = snapshot(2 * transaction.sequence + 3)
        common = {
            "kind": "update",
            "seed": SEED,
            "update": transaction.update,
            "transaction_key": transaction.key,
            "attempt": attempt,
            "segment_id": segment_id,
            "segment_index": completed_segment_index,
        }
        tail_resource_rows.extend(
            [
                {
                    **common,
                    "schema_version": "stage6_resource_attempt/v1",
                    "phase": "pre",
                    "accepted": False,
                    "resource": pre,
                },
                {
                    **common,
                    "schema_version": "stage6_resource_attempt/v1",
                    "phase": "post",
                    "accepted": False,
                    "resource": post,
                },
                {
                    **common,
                    "schema_version": "stage6_resource_acceptance/v2",
                    "phase": "accepted",
                    "accepted": True,
                    "pre": pre,
                    "post": post,
                    "checkpoint": {
                        "seed": SEED,
                        "update": transaction.update,
                        "transaction_key": transaction.key,
                        "checkpoint_sha256": (
                            checkpoint.checkpoint_sha256
                        ),
                        "complete_marker_sha256": (
                            checkpoint.complete_marker_sha256
                        ),
                        "policy_state_sha256": (
                            checkpoint.policy_state_sha256
                        ),
                    },
                },
            ]
        )
    resource_path.write_bytes(
        resource_prefix + _task5_jsonl_bytes(tail_resource_rows)
    )

    receipts = receipt_index.verify()
    global_transaction = next(
        item for item in transactions if item.update == 90
    )
    global_checkpoint = checkpoint_by_update[90]
    global_record = next(
        row["best_record"]
        for row in validation_rows
        if row["update"] == 90
    )
    global_best = {
        "schema_version": "stage6_global_best/v1",
        "record": global_record,
        "transaction_key": global_transaction.key,
        "checkpoint_sha256": global_checkpoint.checkpoint_sha256,
        "complete_marker_sha256": (
            global_checkpoint.complete_marker_sha256
        ),
        "policy_state_sha256": global_checkpoint.policy_state_sha256,
    }
    _task5_write_json(stage / "global-best.json", global_best)

    acceptance_profile = (
        warm_module.build_planning_child_acceptance_profile()
    )
    receipt_bytes = receipt_path.read_bytes()
    _task5_write_json(
        stage / "standard_checkpoint_manifest.json",
        {
            "schema_version": "stage6_checkpoint_manifest/v2",
            "receipt_count": 26,
            "receipt_index": {
                "path": "checkpoints/index.jsonl",
                "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                "size_bytes": len(receipt_bytes),
            },
            "global_best": global_best,
            "receipts": list(receipts),
            "acceptance_profile": acceptance_profile,
        },
    )

    phase_path = stage / "phase-state.jsonl"
    phase_path.write_bytes(b"")
    phase_journal = module.Stage6StateJournal(phase_path)
    for index, state in enumerate(
        (
            "preflight",
            "global_best_frozen",
            "final_test_running",
            "final_unseen_running",
            "baselines_running",
        )
    ):
        immutable = origin_immutable if index == 0 else current_immutable
        phase_journal.append(
            state,
            {
                **immutable,
                "checkpoint_sha256": (
                    config.stage5_authority.checkpoint_sha256
                    if index == 0
                    else global_checkpoint.checkpoint_sha256
                ),
            },
        )

    final_result = {
        "episode_count": 64,
        "bootstrap_audit": {
            "metrics": {
                "success_rate_under_fixed_step_budget": {
                    "ci95_low": 0.80,
                    "ci95_high": 0.90,
                }
            }
        },
    }
    metrics = [
        {
            "transaction_key": f"final:{split}:{method}",
            "kind": "final_evaluation",
            "split": split,
            "method": method,
            "result": final_result,
        }
        for split in ("test", "unseen")
        for method in module.FINAL_EVALUATION_METHODS
    ]
    (stage / "metrics.jsonl").write_bytes(_task5_jsonl_bytes(metrics))

    from lunar_exploration_ppo.ppo.collector import PLANNER_FAILURE_REASONS

    aggregate = {
        "fairness_audit": {
            "schema_version": "stage6_fairness_audit/v1",
            "passed": True,
            "eval_only_isolation": {"passed": True},
        },
        "leakage_audit": {
            "schema_version": "stage6_leakage_audit/v1",
            "passed": True,
        },
        "failure_audit": {
            "schema_version": "stage6_failure_audit/v3",
            "passed": True,
            "planner_failure_counts": {
                reason: 0 for reason in PLANNER_FAILURE_REASONS
            },
            "reset_scan_audit": {
                "scan_order": [
                    "reset_local_safety",
                    "reset_exploration",
                ],
                "dual_scan_episode_count": 640,
                "passed": True,
            },
        },
        "comparison_csv": b"method,value\nfixture,1\n",
        "coverage_curves_csv": b"method,step,value\nfixture,1,1\n",
    }
    for name, key in (
        ("fairness_audit.json", "fairness_audit"),
        ("leakage_audit.json", "leakage_audit"),
        ("failure_audit.json", "failure_audit"),
    ):
        _task5_write_json(stage / name, aggregate[key])
    (stage / "standard_baseline_comparison.csv").write_bytes(
        aggregate["comparison_csv"]
    )
    (stage / "standard_coverage_curves.csv").write_bytes(
        aggregate["coverage_curves_csv"]
    )
    _task5_write_json(
        stage / "scenario_split_audit.json",
        {
            "schema_version": "stage6_scenario_split_audit/v1",
            "catalog_sha256": current_identity["catalog_sha256"],
            "spatial_audit": {"parent_cross_split_count": 0},
        },
    )

    checkpoint_transaction = transactions[-1]
    checkpoint_audit = [
        {
            "transaction_key": checkpoint_transaction.key,
            "passed": True,
        }
    ]
    (stage / "checkpoint_audit.jsonl").write_bytes(
        _task5_jsonl_bytes(checkpoint_audit)
    )
    u100_training = training_rows[-1]
    collection = dict(u100_training["collection_audit"])
    evidence = dict(u100_training["update_metrics"]["math_evidence"])
    math_payload = {
        **{
            name: True
            for name in (
                "observation_finite",
                "action_finite",
                "logprob_finite",
                "value_finite",
                "advantage_finite",
                "return_finite",
                "ratio_finite",
                "loss_finite",
                "kl_finite",
                "grad_finite",
            )
        },
        "mask_violation_count": 0,
        "snapshot_mismatch_count": 0,
        "stale_policy_transition_count": 0,
        "initial_ratio_max_abs_error": 0.0,
        "device": config.device,
        "allowed_initial_ratio_tolerance": (
            1e-5 if config.device == "cuda" else 1e-6
        ),
        "compute_dtype": "float32",
        "joint_logprob": True,
        "joint_logprob_factorization_max_abs_error": 0.0,
        "grad_post_clip_norm_max": 0.1,
        "evidence_source": "ppo_update_math_evidence/v1",
        "evidence_sha256": evidence["evidence_sha256"],
        "sample_count": 1024,
        "snapshot_list_sha256": evidence[
            "snapshot_list_sha256"
        ],
        "initial_forward_sample_count": 1024,
        "forward_sample_count": 1024,
        "loss_sample_count": 1024,
        "gradient_step_count": 1,
    }
    math_row = {
        "transaction_key": checkpoint_transaction.key,
        "seed": SEED,
        "update": 100,
        "schema_version": "stage6_math_audit/v1",
        "passed": True,
        "collection_audit": collection,
        "policy_state_sha256_before": u100_training[
            "update_metrics"
        ]["policy_state_sha256_before"],
        "policy_state_sha256_after": u100_training[
            "update_metrics"
        ]["policy_state_sha256_after"],
        "ppo_math_evidence": evidence,
        **math_payload,
    }
    (stage / "math_audit.jsonl").write_bytes(
        _task5_jsonl_bytes([math_row])
    )

    traces = stage / "episode-traces"
    traces.mkdir(parents=True, exist_ok=True)
    for split in ("test", "unseen"):
        for method in module.FINAL_EVALUATION_METHODS:
            stem = f"final-{split}-{method}"
            for suffix in (".jsonl", ".summary.json", ".commit.json"):
                (traces / f"{stem}{suffix}").write_bytes(b"{}\n")
    preflight = stage / "preflight"
    preflight.mkdir(parents=True, exist_ok=True)
    (preflight / "audit.json").write_bytes(b"{}\n")

    loaded_context, child_sha256 = (
        child_module.load_planning_child_source_repair_artifact(
            child_path,
            stage_root=stage,
            current_execution_identity=current_identity,
            current_verified_review_authorization=current_review,
            current_immutable_bindings=current_immutable,
            require_exact_prefix=False,
        )
    )
    assert child_sha256 == hashlib.sha256(child_payload).hexdigest()
    from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
        PlanningChildLineageEpoch,
        PlanningChildRecoveryCapability,
        PlanningChildResumeCursor,
    )

    continuation_sha256 = hashlib.sha256(
        continuation_payload
    ).hexdigest()
    input_snapshot_sha256 = hashlib.sha256(
        child_payload + continuation_payload
    ).hexdigest()
    current_identity_sha256 = hashlib.sha256(
        module.ArtifactStore.canonical_json_bytes(current_identity)
    ).hexdigest()
    current_immutable_sha256 = hashlib.sha256(
        module.ArtifactStore.canonical_json_bytes(current_immutable)
    ).hexdigest()
    journal_prefixes = {
        relative: {
            "path": relative,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "line_count": payload.count(b"\n"),
        }
        for relative, payload in {
            relative: (stage / relative).read_bytes()
            for relative in (
                "checkpoints/index.jsonl",
                "job-state.jsonl",
                "resource_audit.jsonl",
                "training_metrics.jsonl",
                "validation_metrics.jsonl",
            )
        }.items()
    }
    safety_contract = SafetyContract.from_stage6_config(config)
    warm_context = warm_module.validate_planning_warm_start_artifact(
        warm_artifact,
        expected_child_run_id=FORMAL_RUN_ID,
        expected_child_config_bytes=effective_config_bytes,
    )
    warm_sha256 = hashlib.sha256(warm_payload).hexdigest()

    def checkpoint_lineage_base(
        bindings: Mapping[str, object],
        *,
        update: int,
    ) -> dict[str, object]:
        return {
            **module._stage6_checkpoint_lineage_base(
                config=config,
                bindings=bindings,
                safety_contract=safety_contract,
            ),
            **warm_context.checkpoint_lineage_for_update(
                update,
                warm_start_artifact_sha256=warm_sha256,
            ),
        }

    parent_checkpoint_lineage = (
        loaded_context.checkpoint_lineage_for_update(85)
    )
    checkpoint_lineage_templates = {
        "origin": checkpoint_lineage_base(
            origin_immutable,
            update=75,
        ),
        "parent": {
            **checkpoint_lineage_base(
                current_immutable,
                update=85,
            ),
            "planning_child_source_repair": (
                parent_checkpoint_lineage
            ),
        },
        "continuation": {
            **checkpoint_lineage_base(
                current_immutable,
                update=86,
            ),
            "planning_child_source_repair": {
                "schema_version": (
                    "stage6_planning_child_source_repair_"
                    "continuation_lineage/v1"
                ),
                "artifact_sha256": continuation_sha256,
                "parent": parent_checkpoint_lineage,
                "first_continuation_update": 86,
                "current_execution_identity_sha256": (
                    current_identity_sha256
                ),
            },
        },
    }
    capability_acceptance = {
        "schema_version": (
            "stage6_planning_child_recovery_capability_acceptance/v1"
        ),
        "formal_run_id": FORMAL_RUN_ID,
        "seed": SEED,
        "parent_artifact_sha256": child_sha256,
        "continuation_artifact_sha256": continuation_sha256,
        "input_snapshot_sha256": input_snapshot_sha256,
        "terminal_complete": True,
        "resume_cursor": None,
        "lineage_epochs": [
            {
                "first_update": 75,
                "last_update": 84,
                "artifact_sha256": child_sha256,
                "execution_identity_sha256": hashlib.sha256(
                    module.ArtifactStore.canonical_json_bytes(
                        origin_identity
                    )
                ).hexdigest(),
            },
            {
                "first_update": 85,
                "last_update": 85,
                "artifact_sha256": child_sha256,
                "execution_identity_sha256": (
                    loaded_context.current_execution_identity_sha256
                ),
            },
            {
                "first_update": 86,
                "last_update": None,
                "artifact_sha256": continuation_sha256,
                "execution_identity_sha256": current_identity_sha256,
            },
        ],
        "checkpoint_lineage_templates": (
            checkpoint_lineage_templates
        ),
        "journal_binding_epochs": [
            [transactions[0].key, origin_immutable],
            [transactions[10].key, current_immutable],
            [transactions[11].key, current_immutable],
        ],
        "current_execution_identity_sha256": (
            current_identity_sha256
        ),
        "current_immutable_bindings_sha256": (
            current_immutable_sha256
        ),
        "current_verified_review_authorization": current_review,
        "origin_verified_review_authorization": origin_review,
        "protected_checkpoint_updates": [84],
        "accepted_anchor": {
            "last_accepted_update": 100,
        },
        "journal_prefixes": journal_prefixes,
    }
    recovery_capability = PlanningChildRecoveryCapability(
        formal_run_id=FORMAL_RUN_ID,
        seed=SEED,
        stage_root=stage,
        parent_artifact_sha256=child_sha256,
        continuation_artifact_sha256=continuation_sha256,
        input_snapshot_sha256=input_snapshot_sha256,
        capability_sha256=hashlib.sha256(
            b"task1-planning-child-machine-fixture"
            + child_payload
            + continuation_payload
        ).hexdigest(),
        resume_cursor=None,
        lineage_epochs=(
            PlanningChildLineageEpoch(
                75,
                84,
                child_sha256,
                capability_acceptance["lineage_epochs"][0][
                    "execution_identity_sha256"
                ],
            ),
            PlanningChildLineageEpoch(
                85,
                85,
                child_sha256,
                loaded_context.current_execution_identity_sha256,
            ),
            PlanningChildLineageEpoch(
                86,
                None,
                continuation_sha256,
                current_identity_sha256,
            ),
        ),
        protected_checkpoint_updates=(84,),
        input_pin_requests=(),
        acceptance_binding=capability_acceptance,
        terminal_complete=True,
    )
    performance = module.decide_performance_advantage(
        ppo_ci95_low=0.80,
        gain_over_cost_ci95_high=0.90,
    )
    acceptance = standard_training.build_stage6_acceptance_artifacts(
        global_best=global_record,
        performance_advantage_established=performance.established,
        performance_claim=performance.claim,
        ppo_ci95_low=performance.ppo_ci95_low,
        gain_over_cost_ci95_high=performance.gain_over_cost_ci95_high,
        final_evaluation_count=10,
        final_episode_count=640,
        checkpoint_receipt_count=26,
        immutable_bindings=current_immutable,
        source_repair_binding=None,
        planning_child_source_repair_binding=(
            recovery_capability.evidence_binding()
        ),
        acceptance_profile=acceptance_profile,
    )

    parent = warm_module.VerifiedParentU74(
        parent_update=74,
        checkpoint_sha256=warm_module.PARENT_CHECKPOINT_SHA256,
        manifest_sha256=warm_module.PARENT_MANIFEST_SHA256,
        complete_sha256=warm_module.PARENT_COMPLETE_SHA256,
        policy_state_sha256=warm_module.PARENT_POLICY_STATE_SHA256,
        config_sha256=warm_module.PARENT_CONFIG_SHA256,
        lineage_sha256=warm_module.PARENT_LINEAGE_SHA256,
        resource_accepted=True,
        u75_attempt1_discarded=True,
        checkpoint_payload={},
    )
    monkeypatch.setattr(
        warm_module,
        "verify_parent_u74_bundle",
        lambda path: parent,
    )
    monkeypatch.setattr(
        standard_training,
        "verify_standard_final_evaluation_artifacts_from_bytes",
        lambda **kwargs: {"result": final_result},
    )
    monkeypatch.setattr(
        standard_training,
        "build_standard_final_aggregate_artifacts",
        lambda *args, **kwargs: aggregate,
    )
    monkeypatch.setattr(
        standard_training,
        "validate_stage6_resource_audit",
        lambda rows, schedule, require_terminal: {
            "update_accepted_count": len(schedule),
            "final_accepted_count": 10,
        },
    )

    def validate_collection(
        value: object,
        *,
        expected_device: str,
        expected_policy_sha256: str,
        require_dual_scan: bool,
    ) -> dict[str, object]:
        del expected_device, require_dual_scan
        assert isinstance(value, dict)
        assert value["policy_state_sha256"] == expected_policy_sha256
        return dict(value)

    monkeypatch.setattr(
        standard_training,
        "validate_standard_collection_audit",
        validate_collection,
    )
    monkeypatch.setattr(
        standard_training,
        "validate_checkpoint_replay_audit",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        standard_training,
        "validate_math_audit",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        trainer_module,
        "validate_ppo_math_evidence",
        lambda value, **kwargs: dict(value),
    )
    monkeypatch.setattr(
        checkpoint_retention,
        "verify_retained_checkpoint_snapshot",
        lambda **kwargs: None,
    )

    checkpoint_lineages: dict[int, dict[str, object]] = {}

    def inspect_checkpoint(**kwargs: object) -> SimpleNamespace:
        update = int(kwargs["update_step"])
        checkpoint_lineages[update] = dict(kwargs["expected_lineage"])
        checkpoint_payload = bytes(kwargs["checkpoint_bytes"])
        manifest_payload = bytes(kwargs["manifest_bytes"])
        manifest = json.loads(manifest_payload.decode("utf-8"))
        checkpoint = checkpoint_by_update[update]
        return SimpleNamespace(
            update_step=update,
            checkpoint_sha256=hashlib.sha256(
                checkpoint_payload
            ).hexdigest(),
            manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
            policy_state_sha256=manifest.get(
                "policy_state_sha256",
                checkpoint.policy_state_sha256,
            ),
        )

    monkeypatch.setattr(
        checkpoint_module,
        "inspect_complete_checkpoint_snapshot",
        inspect_checkpoint,
    )

    def load_checkpoint(**kwargs: object) -> SimpleNamespace:
        update = int(kwargs["update_step"])
        checkpoint_lineages[update] = dict(kwargs["expected_lineage"])
        checkpoint = checkpoint_by_update[update]
        return SimpleNamespace(
            checkpoint_sha256=checkpoint.checkpoint_sha256,
            policy_state_sha256=checkpoint.policy_state_sha256,
            best_record=global_record,
            config_sha256=current_immutable["config_sha256"],
            lineage=dict(kwargs["expected_lineage"]),
            safety_contract=kwargs[
                "expected_safety_contract"
            ].to_dict(),
        )

    monkeypatch.setattr(
        checkpoint_module,
        "load_complete_checkpoint_snapshot",
        load_checkpoint,
    )
    policy = torch.nn.Linear(1, 1)
    monkeypatch.setattr(
        module,
        "load_stage4_policy_for_standard",
        lambda **kwargs: policy,
    )
    monkeypatch.setattr(
        trainer_module,
        "policy_state_sha256",
        lambda candidate: global_checkpoint.policy_state_sha256,
    )

    critical_calls = {
        "context": 0,
        "resource": 0,
        "acceptance": 0,
    }
    original_context = module._validate_planning_child_machine_context
    original_resource = module._validate_planning_child_resource_boundary
    original_acceptance = standard_training.build_stage6_acceptance_artifacts

    def context_wrapper(**kwargs: object) -> dict[str, object]:
        critical_calls["context"] += 1
        return original_context(**kwargs)

    def resource_wrapper(**kwargs: object) -> dict[str, object]:
        critical_calls["resource"] += 1
        return original_resource(**kwargs)

    def acceptance_wrapper(**kwargs: object) -> dict[str, object]:
        critical_calls["acceptance"] += 1
        return original_acceptance(**kwargs)

    monkeypatch.setattr(
        module,
        "_validate_planning_child_machine_context",
        context_wrapper,
    )
    monkeypatch.setattr(
        module,
        "_validate_planning_child_resource_boundary",
        resource_wrapper,
    )
    monkeypatch.setattr(
        standard_training,
        "build_stage6_acceptance_artifacts",
        acceptance_wrapper,
    )

    return SimpleNamespace(
        module=module,
        terminal_recovery=stage6_terminal_recovery,
        authority=authority,
        stage=stage,
        config=config,
        current_identity=current_identity,
        current_review=current_review,
        current_immutable=current_immutable,
        origin_immutable=origin_immutable,
        effective_config_bytes=effective_config_bytes,
        warm_start_path=fixture.warm_start_path,
        child_path=child_path,
        continuation_path=continuation_path,
        child_payload=child_payload,
        continuation_payload=continuation_payload,
        loaded_context=loaded_context,
        recovery_capability=recovery_capability,
        planner_root=package_root,
        transactions=transactions,
        receipts=receipts,
        checkpoint_by_update=checkpoint_by_update,
        global_best=global_best,
        acceptance=acceptance,
        critical_calls=critical_calls,
        checkpoint_lineages=checkpoint_lineages,
        safety_contract=safety_contract,
    )


def _task5_expected_checkpoint_lineage(
    fixture: SimpleNamespace,
    *,
    update: int,
    context: object | None = None,
) -> dict[str, object]:
    capability = (
        fixture.recovery_capability
        if context is None
        else context
    )
    return fixture.module._stage6_planning_checkpoint_lineage_for_update(
        update=update,
        current_checkpoint_lineage={},
        historical_checkpoint_lineage={},
        planning_warm_start_context=None,
        planning_warm_start_sha256="",
        planning_child_source_repair=capability,
    )


def _task5_complete_checkpoint_bundle(
    fixture: SimpleNamespace,
    *,
    update: int,
    context: object | None = None,
    checkpoint_record_extra: Mapping[str, object] | None = None,
    checkpoint_record_remove: tuple[str, ...] = (),
    production_training_row: Mapping[str, object] | None = None,
    prior_policy_state_sha256: str | None = None,
) -> SimpleNamespace:
    import io

    import torch

    from lunar_exploration_ppo.ppo import checkpoint as checkpoint_module

    lineage = _task5_expected_checkpoint_lineage(
        fixture,
        update=update,
        context=context,
    )
    policy = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-4)
    manager = checkpoint_module.CheckpointManager(
        fixture.stage.parent / f"r4-bundle-{update:03d}",
        schema_version=fixture.config.checkpoint.schema_version,
    )
    receipt = manager.save_complete(
        policy=policy,
        optimizer=optimizer,
        update_step=update,
        normalization_stats={},
        scenario_sampler_state={"workers": list(range(8))},
        vector_env_states=[
            {"worker": worker, "update": update} for worker in range(8)
        ],
        best_record={},
        versions={
            "observation_schema_version": (
                fixture.config.observation_schema_version
            ),
            "action_space_version": fixture.config.action_space_version,
            "network_architecture_version": (
                fixture.config.network_architecture_version
            ),
            "reward_version": fixture.config.reward_version,
            "frontier_version": "stage2_observed_frontier_top_m/v1",
            "planner_version": fixture.config.planner_version,
        },
        top_m_config={
            "frontier_top_m": fixture.config.scale.frontier_top_m,
            "selection": "stage2_observed_frontier_top_m/v1",
        },
        scale_profile=fixture.config.scale.profile,
        training_config={
            "rollout": fixture.config.rollout.model_dump(mode="json"),
            "ppo": fixture.config.ppo.model_dump(mode="json"),
            "training": fixture.config.training.model_dump(mode="json"),
            "safety": fixture.safety_contract.to_dict(),
        },
        config_sha256=fixture.current_immutable["config_sha256"],
        lineage=lineage,
        eval_metrics={},
        safety_contract=fixture.safety_contract,
    )
    checkpoint_payload = receipt.checkpoint_path.read_bytes()
    manifest_payload = receipt.manifest_path.read_bytes()
    complete_payload = receipt.complete_marker_path.read_bytes()

    effective_checkpoint_extra = dict(checkpoint_record_extra or {})
    if production_training_row is not None:
        from lunar_exploration_ppo.ppo.trainer import (
            _seal_math_evidence,
            snapshot_hash_list_sha256,
        )

        assert isinstance(prior_policy_state_sha256, str)
        training_row = copy.deepcopy(dict(production_training_row))
        manifest = json.loads(manifest_payload.decode("utf-8"))
        policy_after = manifest["policy_state_sha256"]
        training_row["update_metrics"][
            "policy_state_sha256_before"
        ] = prior_policy_state_sha256
        training_row["update_metrics"][
            "policy_state_sha256_after"
        ] = policy_after
        training_row["collection_audit"][
            "policy_state_sha256"
        ] = prior_policy_state_sha256
        update_metrics = training_row["update_metrics"]
        update_metrics.update(
            {
                "initial_ratio_max_abs_error": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "approx_kl": 0.0,
                "grad_pre_clip_norm_max": 0.5,
                "grad_post_clip_norm_max": 0.5,
            }
        )
        optimizer_steps = int(update_metrics["optimizer_steps"])
        sample_count = int(
            training_row["collection_audit"][
                "trainable_transition_count"
            ]
        )
        math_evidence = {
            "schema_version": "ppo_update_math_evidence/v1",
            "sample_count": sample_count,
            "snapshot_list_sha256": snapshot_hash_list_sha256(
                training_row["collection_audit"][
                    "snapshot_sha256"
                ]
            ),
            "batch_policy_state_sha256": (
                prior_policy_state_sha256
            ),
            "observation_finite": True,
            "action_finite": True,
            "old_logprob_finite": True,
            "old_value_finite": True,
            "advantage_finite": True,
            "return_finite": True,
            "mask_violation_count": 0,
            "snapshot_mismatch_count": 0,
            "stale_policy_transition_count": 0,
            "old_joint_logprob_factorization_max_abs_error": 0.0,
            "observed_compute_dtypes": ["float32"],
            "new_logprob_finite": True,
            "new_value_finite": True,
            "ratio_finite": True,
            "loss_finite": True,
            "kl_finite": True,
            "grad_finite": True,
            "joint_logprob_factorization_max_abs_error": 0.0,
            "initial_forward_sample_count": sample_count,
            "forward_sample_count": sample_count * 2,
            "loss_sample_count": sample_count,
            "gradient_step_count": optimizer_steps,
        }
        training_row["update_metrics"]["math_evidence"] = dict(
            _seal_math_evidence(math_evidence)
        )
        effective_checkpoint_extra["eval_metrics"] = {
            "validation": dict(training_row["validation"]),
            "update": dict(training_row["update_metrics"]),
            "eval_isolation": {"passed": True},
            "collection_audit": dict(
                training_row["collection_audit"]
            ),
        }
    checkpoint_record_extra = effective_checkpoint_extra or None
    if checkpoint_record_extra is not None or checkpoint_record_remove:
        checkpoint_record = checkpoint_module.safe_load_checkpoint_payload(
            checkpoint_payload
        )
        assert isinstance(checkpoint_record, dict)
        checkpoint_record = dict(checkpoint_record)
        for name in checkpoint_record_remove:
            checkpoint_record.pop(name)
        if checkpoint_record_extra is not None:
            checkpoint_record.update(checkpoint_record_extra)
        checkpoint_buffer = io.BytesIO()
        torch.save(checkpoint_record, checkpoint_buffer)
        checkpoint_payload = checkpoint_buffer.getvalue()
        checkpoint_sha256 = hashlib.sha256(checkpoint_payload).hexdigest()
        manifest = json.loads(manifest_payload.decode("utf-8"))
        manifest["checkpoint"].update(
            {
                "size_bytes": len(checkpoint_payload),
                "sha256": checkpoint_sha256,
            }
        )
        manifest_payload = fixture.module.ArtifactStore.canonical_json_bytes(
            manifest
        )
        complete = json.loads(complete_payload.decode("utf-8"))
        complete.update(
            {
                "checkpoint_sha256": checkpoint_sha256,
                "manifest_sha256": hashlib.sha256(
                    manifest_payload
                ).hexdigest(),
            }
        )
        complete_payload = fixture.module.ArtifactStore.canonical_json_bytes(
            complete
        )

    manifest = json.loads(manifest_payload.decode("utf-8"))
    return SimpleNamespace(
        checkpoint_payload=checkpoint_payload,
        manifest_payload=manifest_payload,
        complete_payload=complete_payload,
        checkpoint_sha256=hashlib.sha256(checkpoint_payload).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        complete_sha256=hashlib.sha256(complete_payload).hexdigest(),
        policy_state_sha256=manifest["policy_state_sha256"],
        lineage=lineage,
    )


def _prepare_task5_accepted_u85_restart_tail(
    fixture: SimpleNamespace,
    *,
    checkpoint_record_extra: Mapping[str, object] | None = None,
    checkpoint_record_remove: tuple[str, ...] = (),
) -> SimpleNamespace:
    """Keep the frozen U75..U84 bytes and append one real accepted U85."""

    from lunar_exploration_ppo.ppo import standard_training

    artifact = json.loads(fixture.child_path.read_text(encoding="utf-8"))
    prefixes = artifact["append_only_prefixes"]
    relatives = tuple(prefixes)
    full_training_rows = [
        json.loads(line)
        for line in (
            fixture.stage / "training_metrics.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    frozen: dict[str, bytes] = {}
    for relative in relatives:
        payload = (fixture.stage / relative).read_bytes()
        prefix_size = int(prefixes[relative]["prefix_size_bytes"])
        frozen[relative] = payload[:prefix_size]
        (fixture.stage / relative).write_bytes(frozen[relative])

    transaction = fixture.transactions[10]
    assert transaction.update == 85
    checkpoint_root = (
        fixture.stage
        / "checkpoints"
        / f"seed-{transaction.seed}"
        / "update-00000085"
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    bundle = _task5_complete_checkpoint_bundle(
        fixture,
        update=transaction.update,
        checkpoint_record_extra=checkpoint_record_extra,
        checkpoint_record_remove=checkpoint_record_remove,
        production_training_row=full_training_rows[10],
        prior_policy_state_sha256=(
            fixture.receipts[9]["policy_state_sha256"]
        ),
    )
    for name, payload in (
        ("checkpoint.pt", bundle.checkpoint_payload),
        ("manifest.json", bundle.manifest_payload),
        ("complete.json", bundle.complete_payload),
    ):
        (checkpoint_root / name).write_bytes(payload)
    checkpoint = standard_training.StandardTransactionCheckpoint(
        transaction_key=transaction.key,
        seed=transaction.seed,
        update=transaction.update,
        checkpoint_sha256=bundle.checkpoint_sha256,
        complete_marker_sha256=bundle.complete_sha256,
        policy_state_sha256=bundle.policy_state_sha256,
    )
    receipt_index = standard_training.CheckpointReceiptIndex(
        fixture.stage / "checkpoints" / "index.jsonl"
    )
    assert receipt_index.append_once(checkpoint) is True

    journal = fixture.module.Stage6StateJournal(
        fixture.stage / "job-state.jsonl"
    )
    bindings = {
        **fixture.current_immutable,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
    }
    for state in transaction.commit_states:
        journal.append(state, bindings)

    from lunar_exploration_ppo.ppo.checkpoint import (
        safe_load_checkpoint_payload,
    )

    checkpoint_record = safe_load_checkpoint_payload(
        bundle.checkpoint_payload
    )
    eval_metrics = checkpoint_record["eval_metrics"]
    training_row = {
        "transaction_key": transaction.key,
        "seed": transaction.seed,
        "update": transaction.update,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "policy_state_sha256": checkpoint.policy_state_sha256,
        "update_metrics": dict(eval_metrics["update"]),
        "validation": dict(eval_metrics["validation"]),
        "collection_audit": dict(eval_metrics["collection_audit"]),
    }
    (fixture.stage / "training_metrics.jsonl").write_bytes(
        frozen["training_metrics.jsonl"]
        + _task5_jsonl_bytes([training_row])
    )

    resource_path = fixture.stage / "resource_audit.jsonl"
    resource_payload = frozen["resource_audit.jsonl"]
    segment_ids = {
        2: "0123456789ab4def8123456789abcde2",
        3: "0123456789ab4def8123456789abcde3",
        4: "0123456789ab4def8123456789abcde4",
    }
    for segment_index, segment_id in segment_ids.items():
        segment_start = {
            "schema_version": "stage6_resource_lifecycle_segment/v1",
            "kind": "resource_lifecycle_segment",
            "phase": "segment_start",
            "segment_id": segment_id,
            "segment_index": segment_index,
            "root_pid": 5200 + segment_index,
            "first_sample": {
                "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
                "rss_root_pid": 5200 + segment_index,
                "rss_sample_count": 1,
                "rss_bytes": 1000 + segment_index,
                "passed": True,
            },
            "prior_resource_log": {
                "size_bytes": len(resource_payload),
                "sha256": hashlib.sha256(resource_payload).hexdigest(),
            },
        }
        resource_payload += _task5_jsonl_bytes([segment_start])
    pre = {
        "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
        "rss_root_pid": 5204,
        "rss_sample_count": 2,
        "rss_bytes": 1100,
        "passed": True,
    }
    post = {
        **pre,
        "rss_sample_count": 3,
        "rss_bytes": 1200,
    }
    common = {
        "kind": "update",
        "seed": transaction.seed,
        "update": transaction.update,
        "transaction_key": transaction.key,
        "attempt": 2,
        "segment_id": segment_ids[4],
        "segment_index": 4,
    }
    resource_payload += _task5_jsonl_bytes(
        [
            {
                **common,
                "schema_version": "stage6_resource_attempt/v1",
                "phase": "pre",
                "accepted": False,
                "resource": pre,
            },
            {
                **common,
                "schema_version": "stage6_resource_attempt/v1",
                "phase": "post",
                "accepted": False,
                "resource": post,
            },
            {
                **common,
                "schema_version": "stage6_resource_acceptance/v2",
                "phase": "accepted",
                "accepted": True,
                "pre": pre,
                "post": post,
                "checkpoint": {
                    "seed": transaction.seed,
                    "update": transaction.update,
                    "transaction_key": transaction.key,
                    "checkpoint_sha256": checkpoint.checkpoint_sha256,
                    "complete_marker_sha256": (
                        checkpoint.complete_marker_sha256
                    ),
                    "policy_state_sha256": checkpoint.policy_state_sha256,
                },
            },
        ]
    )
    resource_path.write_bytes(resource_payload)

    valid = {
        relative: (fixture.stage / relative).read_bytes()
        for relative in relatives
    }
    return SimpleNamespace(
        frozen=frozen,
        valid=valid,
        checkpoint=checkpoint,
        full_training_rows=full_training_rows,
    )


def _load_task5_child_for_workflow(
    fixture: SimpleNamespace,
) -> object:
    identity = getattr(
        fixture,
        "successor_identity",
        fixture.current_identity,
    )
    review = getattr(
        fixture,
        "successor_review",
        fixture.current_review,
    )
    handle = SimpleNamespace(
        canonical_record=lambda: copy.deepcopy(review)
    )
    return fixture.module._load_planning_child_recovery_capability_for_workflow(
        path=fixture.child_path,
        continuation_path=getattr(
            fixture,
            "continuation_path",
            None,
        ),
        run_id=fixture.stage.parent.name,
        execution_identity=identity,
        review_authorization_handle=handle,
        stage5_authority=fixture.authority,
    )


def _patch_task5_stage6_identity_without_rasterio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    def identity(**kwargs: object) -> dict[str, object]:
        effective = kwargs.get("effective_config_bytes")
        assert isinstance(effective, bytes)
        prospective_tree = "3" * 40
        environment_identity = {
            "schema_version": "stage6_environment_identity/v1",
            "python_version": "3.12.0",
            "python_implementation": "CPython",
            "os_name": "nt",
            "platform_system": "Windows",
            "platform_machine": "AMD64",
            "numpy_version": "2.0.0",
            "torch_version": "2.0.0",
            "torch_cuda_version": "12.0",
            "cudnn_version": 9000,
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_current_device": 0,
            "cuda_device_name": "Task5 GPU",
            "cuda_compute_capability": [8, 0],
            "cuda_total_vram_bytes": 1,
            "compute_dtype": "float32",
            "amp_enabled": False,
        }
        return {
            "schema_version": "stage6_execution_identity/v1",
            "base_commit": module.STAGE5_COMMIT,
            "head_commit": module.STAGE5_COMMIT,
            "real_index_empty": True,
            "config_sha256": hashlib.sha256(effective).hexdigest(),
            "source_set_sha256": "4" * 64,
            "prospective_git_tree": prospective_tree,
            "prospective_tree_sha256": hashlib.sha256(
                prospective_tree.encode("ascii")
            ).hexdigest(),
            "changed_path_set_sha256": "5" * 64,
            "data_sha256": "6" * 64,
            "environment_identity": environment_identity,
            "environment_sha256": module.stage6_environment_sha256(
                environment_identity
            ),
            "catalog_sha256": "7" * 64,
        }

    monkeypatch.setattr(module, "stage6_execution_identity", identity)


def _append_task5_resource_segment(
    fixture: SimpleNamespace,
    *,
    expected_index: int,
) -> dict[str, object]:
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        append_resource_segment_start,
    )

    row = append_resource_segment_start(
        fixture.stage / "resource_audit.jsonl",
        first_sample={
            "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
            "rss_root_pid": os.getpid(),
            "rss_sample_count": 1,
            "rss_bytes": 1300,
            "passed": True,
        },
    )
    assert row["segment_index"] == expected_index
    return row


def _append_task5_resource_pre(
    fixture: SimpleNamespace,
    *,
    segment: Mapping[str, object],
    transaction: object,
    attempt: int,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        bind_resource_row_to_segment,
    )

    DurableJsonl(fixture.stage / "resource_audit.jsonl").append(
        bind_resource_row_to_segment(
            {
                "schema_version": "stage6_resource_attempt/v1",
                "kind": "update",
                "seed": transaction.seed,
                "update": transaction.update,
                "transaction_key": transaction.key,
                "attempt": attempt,
                "phase": "pre",
                "accepted": False,
                "resource": {
                    "rss_source": (
                        "process_tree_lifecycle_peak_current_sum/v1"
                    ),
                    "rss_root_pid": os.getpid(),
                    "rss_sample_count": 2,
                    "rss_bytes": 1400,
                    "passed": True,
                },
            },
            segment,
        )
    )


def _append_task5_u86_crash_suffix(
    fixture: SimpleNamespace,
    tail: SimpleNamespace,
    *,
    boundary: str,
    transaction_update: int = 86,
    segment: Mapping[str, object] | None = None,
    prior_policy_state_sha256: str | None = None,
    resource_sample_count: int = 2,
    resource_rss_bytes: int = 1400,
) -> SimpleNamespace:
    from lunar_exploration_ppo.ppo import standard_training
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        bind_resource_row_to_segment,
    )

    boundaries = (
        "pre",
        "post",
        "accepted",
        "checkpoint",
        "receipt",
        "journal",
        "training",
    )
    if boundary not in boundaries:
        raise AssertionError(f"unknown U86 crash boundary: {boundary}")
    boundary_index = boundaries.index(boundary)
    context = _load_task5_child_for_workflow(fixture)
    transaction = next(
        item
        for item in fixture.transactions
        if item.update == transaction_update
    )
    if segment is None:
        segment = _append_task5_resource_segment(
            fixture,
            expected_index=5,
        )
    checkpoint_root = (
        fixture.stage
        / "checkpoints"
        / f"seed-{transaction.seed}"
        / f"update-{transaction.update:08d}"
    )
    pending_checkpoint_root = checkpoint_root.parent / (
        f".pending-{checkpoint_root.name}-{'1' * 32}"
    )
    prior_policy = (
        tail.checkpoint.policy_state_sha256
        if prior_policy_state_sha256 is None
        else prior_policy_state_sha256
    )
    original_training_row = tail.full_training_rows[
        transaction.sequence
    ]
    required_math_metrics = {
        "initial_ratio_max_abs_error",
        "policy_loss",
        "value_loss",
        "approx_kl",
        "grad_pre_clip_norm_max",
        "grad_post_clip_norm_max",
        "optimizer_steps",
    }
    template_training_row = next(
        (
            row
            for row in tail.full_training_rows
            if required_math_metrics.issubset(
                row["update_metrics"]
            )
        ),
        original_training_row,
    )
    production_training_row = copy.deepcopy(
        template_training_row
    )
    production_training_row.update(
        {
            "transaction_key": transaction.key,
            "seed": transaction.seed,
            "update": transaction.update,
            "validation": copy.deepcopy(
                original_training_row["validation"]
            ),
        }
    )
    production_training_row["update_metrics"][
        "update_step"
    ] = transaction.update
    production_best_record: dict[str, object] | None = None
    if transaction.validation_episodes == 16:
        validation_trace_payload = _task5_jsonl_bytes(
            [
                {
                    "episode_index": episode_index,
                    "seed": transaction.seed,
                    "update": transaction.update,
                }
                for episode_index in range(
                    transaction.validation_episodes
                )
            ]
        )
        production_training_row["validation"][
            "validation_trace_binding"
        ] = standard_training._validation_trace_binding_from_bytes(
            validation_trace_payload
        )
        pending_trace = (
            fixture.stage.parent
            / "stage6-attempts"
            / (
                f"validation-seed-{transaction.seed}-"
                f"update-{transaction.update:03d}-"
                "attempt-001.pending.jsonl"
            )
        )
        pending_trace.parent.mkdir(parents=True, exist_ok=True)
        pending_trace.write_bytes(validation_trace_payload)
        existing_validation_rows = [
            json.loads(line)
            for line in (
                fixture.stage / "validation_metrics.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        production_best_record = (
            standard_training.select_validation_checkpoint_best(
                seed=transaction.seed,
                update=transaction.update,
                validation_metrics=(
                    production_training_row["validation"]
                ),
                previous_best=dict(
                    existing_validation_rows[-1]["best_record"]
                ),
            )
        )
    bundle = _task5_complete_checkpoint_bundle(
        fixture,
        update=transaction.update,
        context=context,
        checkpoint_record_extra=(
            None
            if production_best_record is None
            else {"best_record": production_best_record}
        ),
        production_training_row=production_training_row,
        prior_policy_state_sha256=prior_policy,
    )
    checkpoint = standard_training.StandardTransactionCheckpoint(
        transaction_key=transaction.key,
        seed=transaction.seed,
        update=transaction.update,
        checkpoint_sha256=bundle.checkpoint_sha256,
        complete_marker_sha256=bundle.complete_sha256,
        policy_state_sha256=bundle.policy_state_sha256,
    )
    pre = {
        "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
        "rss_root_pid": os.getpid(),
        "rss_sample_count": resource_sample_count,
        "rss_bytes": resource_rss_bytes,
        "passed": True,
    }
    post = {
        **pre,
        "rss_sample_count": resource_sample_count + 1,
        "rss_bytes": resource_rss_bytes + 50,
    }
    identity = {
        "kind": "update",
        "seed": transaction.seed,
        "update": transaction.update,
        "transaction_key": transaction.key,
        "attempt": 1,
    }
    ledger = DurableJsonl(fixture.stage / "resource_audit.jsonl")
    ledger.append(
        bind_resource_row_to_segment(
            {
                "schema_version": "stage6_resource_attempt/v1",
                **identity,
                "phase": "pre",
                "accepted": False,
                "resource": pre,
            },
            segment,
        )
    )
    if boundary_index >= boundaries.index("post"):
        ledger.append(
            bind_resource_row_to_segment(
                {
                    "schema_version": "stage6_resource_attempt/v1",
                    **identity,
                    "phase": "post",
                    "accepted": False,
                    "resource": post,
                },
                segment,
            )
        )
    if boundary_index >= boundaries.index("accepted"):
        pending_checkpoint_root.mkdir(parents=True, exist_ok=True)
        for name, payload in (
            ("checkpoint.pt", bundle.checkpoint_payload),
            ("manifest.json", bundle.manifest_payload),
            ("complete.json", bundle.complete_payload),
        ):
            (pending_checkpoint_root / name).write_bytes(payload)
        ledger.append(
            bind_resource_row_to_segment(
                {
                    "schema_version": (
                        "stage6_resource_acceptance/v2"
                    ),
                    **identity,
                    "phase": "accepted",
                    "accepted": True,
                    "pre": pre,
                    "post": post,
                    "checkpoint": {
                        "transaction_key": transaction.key,
                        "seed": transaction.seed,
                        "update": transaction.update,
                        "checkpoint_sha256": (
                            checkpoint.checkpoint_sha256
                        ),
                        "complete_marker_sha256": (
                            checkpoint.complete_marker_sha256
                        ),
                        "policy_state_sha256": (
                            checkpoint.policy_state_sha256
                        ),
                    },
                },
                segment,
            )
        )
    if boundary_index >= boundaries.index("checkpoint"):
        pending_checkpoint_root.rename(checkpoint_root)
    if boundary_index >= boundaries.index("receipt"):
        receipt_index = standard_training.CheckpointReceiptIndex(
            fixture.stage / "checkpoints" / "index.jsonl"
        )
        assert receipt_index.append_once(checkpoint) is True
    if boundary_index >= boundaries.index("journal"):
        journal = fixture.module.Stage6StateJournal(
            fixture.stage / "job-state.jsonl"
        )
        bindings = {
            **fixture.current_immutable,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
        }
        for state in transaction.commit_states:
            journal.append(state, bindings)
    if boundary_index >= boundaries.index("training"):
        from lunar_exploration_ppo.ppo.checkpoint import (
            safe_load_checkpoint_payload,
        )

        checkpoint_record = safe_load_checkpoint_payload(
            bundle.checkpoint_payload
        )
        eval_metrics = checkpoint_record["eval_metrics"]
        training_row = {
            "transaction_key": transaction.key,
            "seed": transaction.seed,
            "update": transaction.update,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "policy_state_sha256": checkpoint.policy_state_sha256,
            "update_metrics": dict(eval_metrics["update"]),
            "validation": dict(eval_metrics["validation"]),
            "collection_audit": dict(
                eval_metrics["collection_audit"]
            ),
        }
        training_path = fixture.stage / "training_metrics.jsonl"
        training_path.write_bytes(
            training_path.read_bytes()
            + _task5_jsonl_bytes([training_row])
        )
        if transaction.validation_episodes == 16:
            validation_path = (
                fixture.stage / "validation_metrics.jsonl"
            )
            validation_rows = [
                json.loads(line)
                for line in validation_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            result = dict(training_row["validation"])
            best_record = production_best_record
            assert best_record is not None
            validation_path.write_bytes(
                validation_path.read_bytes()
                + _task5_jsonl_bytes(
                    [
                        {
                            "transaction_key": transaction.key,
                            "seed": transaction.seed,
                            "update": transaction.update,
                            "episode_count": 16,
                            "result": result,
                            "eval_isolation": {"passed": True},
                            "best_record": dict(best_record),
                        }
                    ]
                )
            )
        if standard_training.audit_required(transaction.update):
            math_audit = (
                standard_training._math_audit_from_completed_update(
                    training_row["update_metrics"],
                    collection_audit=training_row[
                        "collection_audit"
                    ],
                    device=fixture.config.device,
                    compute_dtype=fixture.config.ppo.compute_dtype,
                    require_dual_scan=True,
                )
            )
            standard_training.append_transaction_metric_once(
                fixture.stage / "math_audit.jsonl",
                {
                    "transaction_key": transaction.key,
                    "seed": transaction.seed,
                    "update": transaction.update,
                    **math_audit,
                },
            )
    return SimpleNamespace(
        checkpoint=checkpoint,
        checkpoint_root=checkpoint_root,
        pending_checkpoint_root=pending_checkpoint_root,
        segment=segment,
        boundary=boundary,
    )


def _append_task5_accepted_u86_restart_tail(
    fixture: SimpleNamespace,
    tail: SimpleNamespace,
) -> SimpleNamespace:
    from lunar_exploration_ppo.ppo import standard_training
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        bind_resource_row_to_segment,
    )

    capability = _load_task5_child_for_workflow(fixture)
    transaction = fixture.transactions[11]
    assert transaction.update == 86
    segment = _append_task5_resource_segment(
        fixture,
        expected_index=5,
    )
    checkpoint_root = (
        fixture.stage
        / "checkpoints"
        / f"seed-{transaction.seed}"
        / f"update-{transaction.update:08d}"
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    bundle = _task5_complete_checkpoint_bundle(
        fixture,
        update=transaction.update,
        context=capability,
    )
    for name, payload in (
        ("checkpoint.pt", bundle.checkpoint_payload),
        ("manifest.json", bundle.manifest_payload),
        ("complete.json", bundle.complete_payload),
    ):
        (checkpoint_root / name).write_bytes(payload)
    checkpoint = standard_training.StandardTransactionCheckpoint(
        transaction_key=transaction.key,
        seed=transaction.seed,
        update=transaction.update,
        checkpoint_sha256=bundle.checkpoint_sha256,
        complete_marker_sha256=bundle.complete_sha256,
        policy_state_sha256=bundle.policy_state_sha256,
    )
    receipt_index = standard_training.CheckpointReceiptIndex(
        fixture.stage / "checkpoints" / "index.jsonl"
    )
    assert receipt_index.append_once(checkpoint) is True
    journal = fixture.module.Stage6StateJournal(
        fixture.stage / "job-state.jsonl"
    )
    acceptance = capability.evidence_binding().get(
        "acceptance_binding"
    )
    assert isinstance(acceptance, Mapping)
    epochs = acceptance.get(
        "journal_binding_epochs"
    )
    assert isinstance(epochs, list) and epochs
    current_epoch = epochs[-1]
    assert (
        isinstance(current_epoch, list)
        and len(current_epoch) == 2
        and isinstance(current_epoch[1], Mapping)
    )
    assert dict(current_epoch[1]) == fixture.current_immutable
    bindings = {
        **dict(current_epoch[1]),
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
    }
    for state in transaction.commit_states:
        journal.append(state, bindings)

    training_row = copy.deepcopy(tail.full_training_rows[11])
    prior_policy = tail.checkpoint.policy_state_sha256
    training_row["checkpoint_sha256"] = checkpoint.checkpoint_sha256
    training_row["policy_state_sha256"] = checkpoint.policy_state_sha256
    training_row["update_metrics"]["policy_state_sha256_before"] = prior_policy
    training_row["update_metrics"]["policy_state_sha256_after"] = (
        checkpoint.policy_state_sha256
    )
    training_row["collection_audit"]["policy_state_sha256"] = prior_policy
    training_path = fixture.stage / "training_metrics.jsonl"
    training_path.write_bytes(
        training_path.read_bytes() + _task5_jsonl_bytes([training_row])
    )

    root_pid = os.getpid()
    pre = {
        "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
        "rss_root_pid": root_pid,
        "rss_sample_count": 2,
        "rss_bytes": 1400,
        "passed": True,
    }
    post = {
        **pre,
        "rss_sample_count": 3,
        "rss_bytes": 1500,
    }
    identity = {
        "kind": "update",
        "seed": transaction.seed,
        "update": transaction.update,
        "transaction_key": transaction.key,
    }
    ledger = DurableJsonl(fixture.stage / "resource_audit.jsonl")
    for phase, resource in (("pre", pre), ("post", post)):
        ledger.append(
            bind_resource_row_to_segment(
                {
                    "schema_version": "stage6_resource_attempt/v1",
                    **identity,
                    "attempt": 1,
                    "phase": phase,
                    "accepted": False,
                    "resource": resource,
                },
                segment,
            )
        )
    ledger.append(
        bind_resource_row_to_segment(
            {
                "schema_version": "stage6_resource_acceptance/v2",
                **identity,
                "attempt": 1,
                "phase": "accepted",
                "accepted": True,
                "pre": pre,
                "post": post,
                "checkpoint": {
                    "transaction_key": transaction.key,
                    "seed": transaction.seed,
                    "update": transaction.update,
                    "checkpoint_sha256": checkpoint.checkpoint_sha256,
                    "complete_marker_sha256": (
                        checkpoint.complete_marker_sha256
                    ),
                    "policy_state_sha256": checkpoint.policy_state_sha256,
                },
            },
            segment,
        )
    )
    return SimpleNamespace(
        checkpoint=checkpoint,
        checkpoint_root=checkpoint_root,
        segment=segment,
    )


def _append_task5_accepted_through_u90_restart_tail(
    fixture: SimpleNamespace,
    tail: SimpleNamespace,
    *,
    final_update: int = 90,
) -> SimpleNamespace:
    from lunar_exploration_ppo.ppo import standard_training
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        bind_resource_row_to_segment,
    )

    context = _load_task5_child_for_workflow(fixture)
    segment = _append_task5_resource_segment(
        fixture,
        expected_index=5,
    )
    receipt_index = standard_training.CheckpointReceiptIndex(
        fixture.stage / "checkpoints" / "index.jsonl"
    )
    journal = fixture.module.Stage6StateJournal(
        fixture.stage / "job-state.jsonl"
    )
    training_path = fixture.stage / "training_metrics.jsonl"
    resource_ledger = DurableJsonl(fixture.stage / "resource_audit.jsonl")
    validation_path = fixture.stage / "validation_metrics.jsonl"
    validation_rows = [
        json.loads(line)
        for line in validation_path.read_text(encoding="utf-8").splitlines()
    ]
    best_record = dict(validation_rows[-1]["best_record"])
    prior_policy = tail.checkpoint.policy_state_sha256
    latest_checkpoint = None
    latest_root = None

    if final_update not in (90, 99, 100):
        raise AssertionError(
            "Task5 successor tail must stop at U90, U99, or U100"
        )
    final_sequence = next(
        transaction.sequence
        for transaction in fixture.transactions
        if transaction.update == final_update
    )
    for offset, transaction in enumerate(
        fixture.transactions[11 : final_sequence + 1],
        start=0,
    ):
        if transaction.update == 90:
            checkpoint = fixture.checkpoint_by_update[90]
            latest_root = (
                fixture.stage
                / "checkpoints"
                / f"seed-{transaction.seed}"
                / f"update-{transaction.update:08d}"
            )
        elif transaction.update == final_update:
            bundle = _task5_complete_checkpoint_bundle(
                fixture,
                update=transaction.update,
                context=context,
            )
            latest_root = (
                fixture.stage
                / "checkpoints"
                / f"seed-{transaction.seed}"
                / f"update-{transaction.update:08d}"
            )
            latest_root.mkdir(parents=True, exist_ok=True)
            for name, payload in (
                ("checkpoint.pt", bundle.checkpoint_payload),
                ("manifest.json", bundle.manifest_payload),
                ("complete.json", bundle.complete_payload),
            ):
                (latest_root / name).write_bytes(payload)
            checkpoint = standard_training.StandardTransactionCheckpoint(
                transaction_key=transaction.key,
                seed=transaction.seed,
                update=transaction.update,
                checkpoint_sha256=bundle.checkpoint_sha256,
                complete_marker_sha256=bundle.complete_sha256,
                policy_state_sha256=bundle.policy_state_sha256,
            )
        else:
            checkpoint = fixture.checkpoint_by_update[transaction.update]
        assert receipt_index.append_once(checkpoint) is True
        bindings = {
            **fixture.current_immutable,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
        }
        for state in transaction.commit_states:
            journal.append(state, bindings)

        training_row = copy.deepcopy(
            tail.full_training_rows[transaction.sequence]
        )
        training_row["checkpoint_sha256"] = checkpoint.checkpoint_sha256
        training_row["policy_state_sha256"] = checkpoint.policy_state_sha256
        training_row["update_metrics"][
            "policy_state_sha256_before"
        ] = prior_policy
        training_row["update_metrics"][
            "policy_state_sha256_after"
        ] = checkpoint.policy_state_sha256
        training_row["collection_audit"][
            "policy_state_sha256"
        ] = prior_policy
        training_path.write_bytes(
            training_path.read_bytes()
            + _task5_jsonl_bytes([training_row])
        )

        sample_count = 2 + offset * 2
        pre = {
            "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
            "rss_root_pid": int(segment["root_pid"]),
            "rss_sample_count": sample_count,
            "rss_bytes": 1400 + offset * 100,
            "passed": True,
        }
        post = {
            **pre,
            "rss_sample_count": sample_count + 1,
            "rss_bytes": int(pre["rss_bytes"]) + 50,
        }
        identity = {
            "kind": "update",
            "seed": transaction.seed,
            "update": transaction.update,
            "transaction_key": transaction.key,
            "attempt": 1,
        }
        for phase, resource in (("pre", pre), ("post", post)):
            resource_ledger.append(
                bind_resource_row_to_segment(
                    {
                        "schema_version": "stage6_resource_attempt/v1",
                        **identity,
                        "phase": phase,
                        "accepted": False,
                        "resource": resource,
                    },
                    segment,
                )
            )
        resource_ledger.append(
            bind_resource_row_to_segment(
                {
                    "schema_version": "stage6_resource_acceptance/v2",
                    **identity,
                    "phase": "accepted",
                    "accepted": True,
                    "pre": pre,
                    "post": post,
                    "checkpoint": {
                        "transaction_key": transaction.key,
                        "seed": transaction.seed,
                        "update": transaction.update,
                        "checkpoint_sha256": checkpoint.checkpoint_sha256,
                        "complete_marker_sha256": (
                            checkpoint.complete_marker_sha256
                        ),
                        "policy_state_sha256": (
                            checkpoint.policy_state_sha256
                        ),
                    },
                },
                segment,
            )
        )
        if transaction.validation_episodes == 16:
            result = dict(training_row["validation"])
            best_record = (
                standard_training.select_validation_checkpoint_best(
                    seed=transaction.seed,
                    update=transaction.update,
                    validation_metrics=result,
                    previous_best=best_record,
                )
            )
            validation_path.write_bytes(
                validation_path.read_bytes()
                + _task5_jsonl_bytes(
                    [
                        {
                            "transaction_key": transaction.key,
                            "seed": transaction.seed,
                            "update": transaction.update,
                            "episode_count": 16,
                            "result": result,
                            "eval_isolation": {"passed": True},
                            "best_record": dict(best_record),
                        }
                    ]
                )
            )
        prior_policy = checkpoint.policy_state_sha256
        latest_checkpoint = checkpoint

    assert latest_checkpoint is not None
    assert latest_root is not None
    if final_update == 99:
        for audit_name in (
            "math_audit.jsonl",
            "checkpoint_audit.jsonl",
        ):
            audit_path = fixture.stage / audit_name
            if not audit_path.is_file():
                continue
            audit_rows = [
                json.loads(line)
                for line in audit_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            audit_path.write_bytes(
                _task5_jsonl_bytes(
                    [
                        row
                        for row in audit_rows
                        if (
                            type(row.get("update")) is not int
                            or int(row["update"]) <= final_update
                        )
                    ]
                )
            )
    if final_update == 100:
        receipt_payload = (
            fixture.stage / "checkpoints" / "index.jsonl"
        ).read_bytes()
        receipt_rows = receipt_index.verify()
        checkpoint_manifest_path = (
            fixture.stage / "standard_checkpoint_manifest.json"
        )
        checkpoint_manifest = json.loads(
            checkpoint_manifest_path.read_text(encoding="utf-8")
        )
        checkpoint_manifest.update(
            {
                "receipt_count": len(receipt_rows),
                "receipt_index": {
                    "path": "checkpoints/index.jsonl",
                    "sha256": hashlib.sha256(
                        receipt_payload
                    ).hexdigest(),
                    "size_bytes": len(receipt_payload),
                },
                "receipts": list(receipt_rows),
                "global_best": fixture.global_best,
            }
        )
        _task5_write_json(
            checkpoint_manifest_path,
            checkpoint_manifest,
        )
        phase_path = fixture.stage / "phase-state.jsonl"
        phase_rows = fixture.module.Stage6StateJournal.verify_snapshot_bytes(
            phase_path.read_bytes()
        )
        phase_path.write_bytes(b"")
        phase_journal = fixture.module.Stage6StateJournal(phase_path)
        for index, row in enumerate(phase_rows):
            bindings = dict(row["bindings"])
            if index > 0:
                bindings = {
                    **fixture.current_immutable,
                    "checkpoint_sha256": fixture.global_best[
                        "checkpoint_sha256"
                    ],
                }
            phase_journal.append(str(row["state"]), bindings)
        training_rows = [
            json.loads(line)
            for line in (
                fixture.stage / "training_metrics.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        final_training = next(
            row for row in training_rows if row["update"] == 100
        )
        math_path = fixture.stage / "math_audit.jsonl"
        math_rows = [
            json.loads(line)
            for line in math_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        for row in math_rows:
            if row["update"] != 100:
                continue
            math_evidence = final_training["update_metrics"][
                "math_evidence"
            ]
            row.update(
                {
                    "collection_audit": final_training[
                        "collection_audit"
                    ],
                    "policy_state_sha256_before": final_training[
                        "update_metrics"
                    ]["policy_state_sha256_before"],
                    "policy_state_sha256_after": final_training[
                        "update_metrics"
                    ]["policy_state_sha256_after"],
                    "ppo_math_evidence": math_evidence,
                    "evidence_sha256": math_evidence[
                        "evidence_sha256"
                    ],
                    "snapshot_list_sha256": math_evidence[
                        "snapshot_list_sha256"
                    ],
                }
            )
        math_path.write_bytes(_task5_jsonl_bytes(math_rows))
    return SimpleNamespace(
        checkpoint=latest_checkpoint,
        checkpoint_root=latest_root,
        segment=segment,
    )


# Dynamic restart-tail coverage moved to
# test_stage6_planning_child_recovery.py.  Workflow tests below consume only
# the already-issued immutable recovery capability.


def _run_task5_planning_child_semantic_verifier(
    fixture: SimpleNamespace,
) -> dict[str, object]:
    terminal_recovery = fixture.terminal_recovery
    with terminal_recovery._capture_preterminal_evidence_handle(
        fixture.stage
    ) as evidence:
        result = (
            fixture.module._verify_stage6_machine_acceptance_from_bound_graph(
                stage_root=fixture.stage,
                repo_root=ROOT,
                require_terminal=False,
                summary_override=fixture.acceptance["summary"],
                routing_override=fixture.acceptance["routing"],
                reports_override=fixture.acceptance["reports"],
                evidence_handle=evidence,
                planning_child_recovery_capability=(
                    fixture.recovery_capability
                ),
            )
        )
        binding = evidence.binding
        graph = evidence.graph_snapshot()
        directories = evidence.directory_snapshot()
        resource_payload = evidence.read_bytes(
            "resource_audit.jsonl",
            label="Task5 resource prefix",
        )
        phase_payload = evidence.read_bytes(
            "phase-state.jsonl",
            label="Task5 phase prefix",
        )
    fixture.preterminal_evidence = {
        "binding": binding,
        "graph": graph,
        "directories": directories,
        "resource_payload": resource_payload,
        "phase_payload": phase_payload,
    }
    fixture.semantic_result = result
    return result


def _finalize_task5_planning_child_terminal_fixture(
    fixture: SimpleNamespace,
    *,
    validate_context: bool = True,
    terminal_row_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Commit an isolated receipt-bound terminal graph without formal APIs."""

    from lunar_exploration_ppo.utils.resource_lifecycle import (
        build_bound_terminal_resource_evidence,
    )

    module = fixture.module
    recovery = fixture.terminal_recovery
    stage = fixture.stage
    captured = fixture.preterminal_evidence
    semantic = recovery._bind_planning_child_semantic_result(
        {
            "schema_version": recovery.SEMANTIC_SCHEMA,
            **fixture.semantic_result,
            "evidence_binding": captured["binding"],
        },
        capability=fixture.recovery_capability,
    )
    terminal_artifacts = {
        "summary.json": module.ArtifactStore.canonical_json_bytes(
            fixture.acceptance["summary"]
        ),
        "routing.json": module.ArtifactStore.canonical_json_bytes(
            fixture.acceptance["routing"]
        ),
        **fixture.acceptance["reports"],
    }
    artifact_rows = recovery._validate_terminal_artifacts(
        terminal_artifacts
    )
    receipt = {
        "schema_version": recovery.RECEIPT_SCHEMA,
        "semantic_verification": semantic,
        "preterminal_evidence_binding": captured["binding"],
        "immutable_bindings": fixture.current_immutable,
        "global_checkpoint_identity": fixture.global_best,
        "terminal_artifacts": artifact_rows,
        "preterminal_artifact_graph": captured["graph"],
        "preterminal_directory_graph": captured["directories"],
        "resource_audit_prefix": recovery._identity(
            captured["resource_payload"]
        ),
        "phase_state_prefix": recovery._identity(
            captured["phase_payload"]
        ),
    }
    receipt_payload = recovery._canonical_json(receipt)
    validated_receipt, _, receipt_identity = (
        recovery._validate_receipt_payload(receipt_payload)
    )
    assert validated_receipt == receipt
    (stage / recovery.RECEIPT_NAME).write_bytes(receipt_payload)
    for relative, payload in terminal_artifacts.items():
        (stage / relative).write_bytes(payload)

    resource_path = stage / recovery.RESOURCE_NAME
    resource_prefix = resource_path.read_bytes()
    assert resource_prefix == captured["resource_payload"]
    resource_rows = recovery._strict_jsonl(
        resource_prefix,
        label="Task5 terminal resource prefix",
    )
    last_segment = next(
        row
        for row in reversed(resource_rows)
        if row.get("kind") == "resource_lifecycle_segment"
    )
    resource_snapshots = [
        snapshot
        for row in resource_rows
        for snapshot in (
            row.get("first_sample"),
            row.get("resource"),
            row.get("pre"),
            row.get("post"),
        )
        if isinstance(snapshot, Mapping)
    ]
    latest_rss_bytes = max(
        int(snapshot["rss_bytes"])
        for snapshot in resource_snapshots
    )
    latest_sample_count = max(
        int(snapshot["rss_sample_count"])
        for snapshot in resource_snapshots
    )
    terminal_resource = {
        "d_free_bytes": 200 * 1024**3,
        "rss_bytes": latest_rss_bytes + 100,
        "peak_vram_bytes": 4096,
        "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
        "rss_root_pid": int(last_segment["root_pid"]),
        "rss_sample_count": latest_sample_count + 1,
        "rss_latest_process_count": 9,
        "rss_peak_process_count": 9,
        "warnings": [],
        "hard_stops": [],
        "passed": True,
    }
    terminal_row = (
        build_bound_terminal_resource_evidence(
            resource_rows,
            terminal_resource=terminal_resource,
            preterminal_acceptance=receipt_identity,
        )
        if terminal_row_override is None
        else {
            **dict(terminal_row_override),
            "preterminal_acceptance": receipt_identity,
        }
    )
    resource_path.write_bytes(
        resource_prefix + recovery._canonical_row(terminal_row)
    )

    phase_path = stage / recovery.PHASE_NAME
    assert phase_path.read_bytes() == captured["phase_payload"]
    success_bindings = {
        **fixture.current_immutable,
        "checkpoint_sha256": fixture.global_best["checkpoint_sha256"],
        "preterminal_acceptance_sha256": receipt_identity["sha256"],
        "preterminal_evidence_graph_sha256": captured["binding"][
            "graph_sha256"
        ],
    }
    phase_journal = module.Stage6StateJournal(phase_path)
    phase_journal.append("machine_passed", success_bindings)
    phase_journal.append(
        "awaiting_independent_review",
        success_bindings,
    )

    manifest = {
        "schema_version": "stage6_sha256_manifest/v3",
        "source_identity": module.stage6_source_identity(ROOT),
        "verified_review_authorization": (
            module._stage6_manifest_review_authorization(
                stage,
                repo_root=ROOT,
                planning_child_recovery_capability=(
                    fixture.recovery_capability
                ),
            )
        ),
        "preterminal_evidence_binding": captured["binding"],
        "artifacts": module._build_stage6_manifest_entries(
            stage,
            module._stage6_manifest_relative_paths(stage),
        ),
    }
    manifest_payload = _task5_write_json(
        stage / recovery.MANIFEST_NAME,
        manifest,
    )
    module.verify_stage6_manifest(
        stage_root=stage,
        repo_root=ROOT,
        planning_child_recovery_capability=(
            fixture.recovery_capability
        ),
    ).require_current()
    if validate_context:
        with recovery._capture_terminal_evidence_handle(stage) as evidence:
            recovery._load_terminal_context(
                stage,
                manifest_verifier=lambda candidate: {
                    "sha256": hashlib.sha256(manifest_payload).hexdigest(),
                    "size_bytes": len(manifest_payload),
                },
                evidence_handle=evidence,
                planning_child_capability=(
                    fixture.recovery_capability
                ),
            )
    fixture.receipt = receipt
    fixture.receipt_identity = receipt_identity
    fixture.terminal_artifacts = terminal_artifacts
    fixture.manifest_payload = manifest_payload
    return {
        "receipt_identity": receipt_identity,
        "terminal_resource": terminal_row,
        "manifest_identity": {
            "sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "size_bytes": len(manifest_payload),
        },
    }


def test_planning_child_machine_verifiers_execute_production_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_task5_planning_child_machine_fixture(
        tmp_path,
        monkeypatch,
    )

    result = _run_task5_planning_child_semantic_verifier(fixture)

    assert result == {
        "passed": True,
        "state": "awaiting_independent_review",
        "final_evaluation_count": 10,
        "final_episode_count": 640,
        "checkpoint_receipt_count": 26,
        "global_best_policy_state_sha256": fixture.global_best[
            "policy_state_sha256"
        ],
    }
    assert fixture.acceptance["summary"][
        "parent_semantics_update_count"
    ] == 74
    assert fixture.acceptance["summary"][
        "child_new_semantics_update_count"
    ] == 26
    assert fixture.acceptance["summary"]["checkpoint_receipt_count"] == 26
    validation_rows = [
        json.loads(line)
        for line in (
            fixture.stage / "validation_metrics.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["update"] for row in validation_rows] == [80, 90, 100]
    assert "source_repair" not in fixture.acceptance["summary"]
    assert (
        fixture.acceptance["summary"][
            "planning_child_recovery"
        ]["capability_sha256"]
        == fixture.recovery_capability.capability_sha256
    )
    assert fixture.critical_calls == {
        "context": 1,
        "resource": 1,
        "acceptance": 1,
    }
    assert fixture.checkpoint_lineages[84]["checkpoint_update"] == 84
    assert (
        "planning_child_source_repair"
        not in fixture.checkpoint_lineages[84]
    )
    for update in (90, 100):
        assert (
            fixture.checkpoint_lineages[update][
                "planning_child_source_repair"
            ]["schema_version"]
            == (
                "stage6_planning_child_source_repair_"
                "continuation_lineage/v1"
            )
        )
    job_rows = fixture.module.Stage6StateJournal.verify_snapshot_bytes(
        (fixture.stage / "job-state.jsonl").read_bytes()
    )
    u80_state = f"seed_20260716_training_update_80"
    u90_state = f"seed_20260716_training_update_90"
    u80 = next(row for row in job_rows if row["state"] == u80_state)
    u90 = next(row for row in job_rows if row["state"] == u90_state)
    assert {
        key: u80["bindings"][key]
        for key in fixture.origin_immutable
    } == fixture.origin_immutable
    assert {
        key: u90["bindings"][key]
        for key in fixture.current_immutable
    } == fixture.current_immutable


def test_planning_child_recovery_capability_binds_current_resource_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_task5_planning_child_machine_fixture(
        tmp_path,
        monkeypatch,
        include_startup_only_resource_segment_two=True,
    )
    result = _run_task5_planning_child_semantic_verifier(fixture)
    capability = result._planning_child_recovery_capability
    payload = (fixture.stage / "resource_audit.jsonl").read_bytes()
    binding = capability.acceptance_binding["journal_prefixes"][
        "resource_audit.jsonl"
    ]

    assert capability is fixture.recovery_capability
    assert binding == {
        "path": "resource_audit.jsonl",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "line_count": payload.count(b"\n"),
    }


def test_planning_child_machine_rejects_capability_review_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    fixture = _build_task5_planning_child_machine_fixture(
        tmp_path,
        monkeypatch,
    )
    acceptance = copy.deepcopy(
        dict(fixture.recovery_capability.acceptance_binding)
    )
    acceptance["current_verified_review_authorization"] = {
        **fixture.current_review,
        "quality_review_sha256": "f" * 64,
    }
    tampered = replace(
        fixture.recovery_capability,
        acceptance_binding=acceptance,
    )
    lineage = json.loads(
        (fixture.stage / "lineage_audit.json").read_text(
            encoding="utf-8"
        )
    )

    with pytest.raises(
        fixture.module.Stage6WorkflowError,
        match="acceptance binding",
    ):
        fixture.module._validate_planning_child_machine_context(
            capability=tampered,
            lineage_audit=lineage,
            current_execution_identity=fixture.current_identity,
            current_verified_review_authorization=(
                fixture.current_review
            ),
            current_immutable_bindings=fixture.current_immutable,
        )


@pytest.mark.parametrize(
    "drift",
    (
        "origin_execution_identity",
        "origin_immutable_bindings",
        "current_final_lineage_epoch",
    ),
)
def test_planning_child_machine_context_binds_capability_epoch_endpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    from dataclasses import replace

    fixture = _build_task5_planning_child_machine_fixture(
        tmp_path,
        monkeypatch,
    )
    capability = fixture.recovery_capability
    lineage = json.loads(
        (fixture.stage / "lineage_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if drift == "origin_execution_identity":
        lineage["execution_identity"] = {
            **lineage["execution_identity"],
            "source_set_sha256": "f" * 64,
        }
    elif drift == "origin_immutable_bindings":
        lineage["immutable_bindings"] = {
            **lineage["immutable_bindings"],
            "source_set_sha256": "e" * 64,
        }
    else:
        final_epoch = replace(
            capability.lineage_epochs[-1],
            execution_identity_sha256="d" * 64,
        )
        capability = replace(
            capability,
            lineage_epochs=(
                *capability.lineage_epochs[:-1],
                final_epoch,
            ),
        )

    with pytest.raises(
        fixture.module.Stage6WorkflowError,
        match="acceptance binding|capability drifted",
    ):
        fixture.module._validate_planning_child_machine_context(
            capability=capability,
            lineage_audit=lineage,
            current_execution_identity=fixture.current_identity,
            current_verified_review_authorization=(
                fixture.current_review
            ),
            current_immutable_bindings=fixture.current_immutable,
        )


def _task5_terminal_resource(
    *,
    sample_count: int,
    rss_bytes: int,
) -> dict[str, object]:
    return {
        "d_free_bytes": 200 * 1024**3,
        "rss_bytes": rss_bytes,
        "peak_vram_bytes": 4096,
        "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
        "rss_root_pid": os.getpid(),
        "rss_sample_count": sample_count,
        "rss_latest_process_count": 1,
        "rss_peak_process_count": 1,
        "warnings": [],
        "hard_stops": [],
        "passed": True,
    }


def test_planning_child_recovery_authorization_uses_issued_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows import (
        stage6_planning_child_recovery as recovery,
    )
    from test_stage6_planning_child_source_repair_continuation import (
        _continuation_preview,
        _publish_successor_chain,
    )

    (
        fixture,
        tail,
        identity,
        review,
        immutable,
        _authorization_path,
        artifact,
    ) = _continuation_preview(tmp_path, monkeypatch)
    continuation_path = _publish_successor_chain(
        fixture,
        identity=identity,
        review=review,
        immutable=immutable,
        artifact=artifact,
    )
    capability = recovery.issue_planning_child_recovery_capability(
        stage_root=fixture.stage,
        parent_artifact_path=fixture.child_path,
        continuation_artifact_path=continuation_path,
        current_execution_identity=identity,
        current_verified_review_authorization=review,
        current_immutable_bindings=immutable,
    )
    fixture.current_identity = identity
    fixture.current_review = review
    fixture.current_immutable = immutable
    fixture.recovery_capability = capability
    monkeypatch.setattr(
        fixture.module,
        "stage6_execution_identity",
        lambda **_kwargs: copy.deepcopy(identity),
    )

    _append_task5_accepted_through_u90_restart_tail(
        fixture,
        tail,
        final_update=100,
    )
    with fixture.terminal_recovery._capture_preterminal_evidence_handle(
        fixture.stage
    ) as evidence:
        fixture.preterminal_evidence = {
            "binding": evidence.binding,
            "graph": evidence.graph_snapshot(),
            "directories": evidence.directory_snapshot(),
            "resource_payload": evidence.read_bytes(
                "resource_audit.jsonl",
                label="R2 I1 resource prefix",
            ),
            "phase_payload": evidence.read_bytes(
                "phase-state.jsonl",
                label="R2 I1 phase prefix",
            ),
        }
    fixture.semantic_result = {
        "passed": True,
        "state": "awaiting_independent_review",
        "final_evaluation_count": 10,
        "final_episode_count": 640,
        "checkpoint_receipt_count": 26,
        "global_best_policy_state_sha256": fixture.global_best[
            "policy_state_sha256"
        ],
    }
    _finalize_task5_planning_child_terminal_fixture(
        fixture,
        validate_context=False,
    )
    review_handle = _FakeReviewAuthorizationHandle(record=review)

    with fixture.terminal_recovery._capture_terminal_evidence_handle(
        fixture.stage
    ) as evidence_handle:
        evidence_type = type(evidence_handle)
        read_bytes = evidence_type.read_bytes

        def forbid_planning_child_reopen(
            handle: object,
            relative_path: str,
            *,
            label: str,
        ) -> bytes:
            if relative_path in {
                "planning-child-source-repair.json",
                "planning-child-source-repair-continuation.json",
            }:
                raise AssertionError(
                    "authorization consumer reopened planning-child inputs"
                )
            return read_bytes(
                handle,
                relative_path,
                label=label,
            )

        monkeypatch.setattr(
            evidence_type,
            "read_bytes",
            forbid_planning_child_reopen,
        )
        fixture.module._verify_stage6_recovery_authorization_bindings(
            stage_root=fixture.stage,
            repo_root=ROOT,
            config_path=CONFIG,
            run_id=fixture.stage.parent.name,
            stage5_gate_path=tmp_path / "unused-stage5-gate.json",
            review_authorization_handle=review_handle,
            evidence_handle=evidence_handle,
            planning_child_recovery_capability=capability,
        )


def test_planning_child_validated_capability_drives_receipt_and_terminal_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_task5_planning_child_machine_fixture(
        tmp_path,
        monkeypatch,
    )
    terminal_artifacts = {
        "summary.json": fixture.module.ArtifactStore.canonical_json_bytes(
            fixture.acceptance["summary"]
        ),
        "routing.json": fixture.module.ArtifactStore.canonical_json_bytes(
            fixture.acceptance["routing"]
        ),
        **fixture.acceptance["reports"],
    }
    semantic = fixture.module.verify_stage6_pending_acceptance(
        stage_root=fixture.stage,
        repo_root=ROOT,
        summary=fixture.acceptance["summary"],
        routing=fixture.acceptance["routing"],
        reports=fixture.acceptance["reports"],
        planning_child_recovery_capability=(
            fixture.recovery_capability
        ),
    )
    capability = fixture.terminal_recovery._semantic_planning_child_capability(
        semantic
    )
    assert capability is not None
    assert capability is fixture.recovery_capability

    with _active_execution_capability(
        tmp_path / "task5-recovery-authority",
        monkeypatch,
        run_root=fixture.stage.parent,
        effective_config_bytes=fixture.effective_config_bytes,
        planning_warm_start_path=fixture.warm_start_path,
        planning_child_source_repair_path=fixture.child_path,
        formal_run_id=fixture.stage.parent.name,
    ) as active:
        with pytest.raises(
            fixture.terminal_recovery.TerminalRecoveryError,
            match="planning child|capability|validated",
        ):
            fixture.terminal_recovery.write_stage6_preterminal_acceptance(
                stage_root=fixture.stage,
                semantic_result=dict(semantic),
                immutable_bindings=fixture.current_immutable,
                global_checkpoint_identity=fixture.global_best,
                terminal_artifacts=terminal_artifacts,
                execution_capability=active["capability"],
            )

        receipt_identity = (
            fixture.terminal_recovery.write_stage6_preterminal_acceptance(
                stage_root=fixture.stage,
                semantic_result=semantic,
                immutable_bindings=fixture.current_immutable,
                global_checkpoint_identity=fixture.global_best,
                terminal_artifacts=terminal_artifacts,
                execution_capability=active["capability"],
            )
        )
        assert fixture.terminal_recovery.detect_stage6_terminal_recovery(
            stage_root=fixture.stage,
        )["status"] == "invalid"
        assert fixture.terminal_recovery.detect_stage6_terminal_recovery(
            stage_root=fixture.stage,
            planning_child_recovery_capability=capability,
        )["status"] == "valid_preterminal_recovery"

        fixture.terminal_recovery.append_stage6_recovery_resource_segment(
            stage_root=fixture.stage,
            first_sample=_task5_terminal_resource(
                sample_count=1,
                rss_bytes=2048,
            ),
            execution_capability=active["capability"],
            planning_child_recovery_capability=capability,
        )
        terminal = (
            fixture.terminal_recovery.append_stage6_recovery_resource_terminal(
                stage_root=fixture.stage,
                terminal_resource=_task5_terminal_resource(
                    sample_count=2,
                    rss_bytes=4096,
                ),
                execution_capability=active["capability"],
                planning_child_recovery_capability=capability,
            )
        )
        assert terminal["preterminal_acceptance"] == receipt_identity

        def commit_manifest(candidate: Path) -> dict[str, object]:
            manifest_path = candidate / fixture.terminal_recovery.MANIFEST_NAME
            if not manifest_path.exists():
                payload = _task5_write_json(
                    manifest_path,
                    {
                        "schema_version": "stage6_sha256_manifest/v3",
                        "source_identity": fixture.module.stage6_source_identity(
                            ROOT
                        ),
                        "verified_review_authorization": (
                            fixture.module._stage6_lineage_review_authorization(
                                candidate,
                                repo_root=ROOT,
                            )
                        ),
                        "preterminal_evidence_binding": semantic[
                            "evidence_binding"
                        ],
                        "artifacts": (
                            fixture.module._build_stage6_manifest_entries(
                                candidate,
                                fixture.module._stage6_manifest_relative_paths(
                                    candidate
                                ),
                            )
                        ),
                    },
                )
            else:
                payload = manifest_path.read_bytes()
            return {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }

        committed = fixture.terminal_recovery.recover_stage6_terminal_commit(
            stage_root=fixture.stage,
            manifest_committer=commit_manifest,
            execution_capability=active["capability"],
            planning_child_recovery_capability=capability,
        )

    assert committed["receipt_identity"] == receipt_identity
    assert fixture.terminal_recovery.detect_stage6_terminal_recovery(
        stage_root=fixture.stage,
        manifest_verifier=commit_manifest,
        planning_child_recovery_capability=capability,
    )["status"] == "valid_terminal_recovery"


def test_planning_child_validated_capability_rejects_other_incomplete_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_task5_planning_child_machine_fixture(
        tmp_path,
        monkeypatch,
    )
    result = _run_task5_planning_child_semantic_verifier(fixture)
    capability = result._planning_child_recovery_capability
    resource_path = fixture.stage / "resource_audit.jsonl"
    rows = [
        json.loads(line)
        for line in resource_path.read_text(encoding="utf-8").splitlines()
    ]
    u86_pre = copy.deepcopy(
        next(
            row
            for row in rows
            if row.get("update") == 86 and row.get("phase") == "pre"
        )
    )
    last_accepted = next(
        row for row in reversed(rows) if row.get("phase") == "accepted"
    )
    u86_pre["attempt"] = 2
    u86_pre["resource"] = {
        **last_accepted["post"],
        "rss_sample_count": int(
            last_accepted["post"]["rss_sample_count"]
        )
        + 1,
        "rss_bytes": int(last_accepted["post"]["rss_bytes"]) + 1,
    }
    resource_path.write_bytes(
        resource_path.read_bytes() + _task5_jsonl_bytes([u86_pre])
    )
    with fixture.terminal_recovery._capture_preterminal_evidence_handle(
        fixture.stage
    ) as evidence:
        semantic = fixture.terminal_recovery._bind_planning_child_semantic_result(
            {
                "schema_version": fixture.terminal_recovery.SEMANTIC_SCHEMA,
                **result,
                "evidence_binding": evidence.binding,
            },
            capability=capability,
        )
    terminal_artifacts = {
        "summary.json": fixture.module.ArtifactStore.canonical_json_bytes(
            fixture.acceptance["summary"]
        ),
        "routing.json": fixture.module.ArtifactStore.canonical_json_bytes(
            fixture.acceptance["routing"]
        ),
        **fixture.acceptance["reports"],
    }
    with _active_execution_capability(
        tmp_path / "task5-incomplete-authority",
        monkeypatch,
        run_root=fixture.stage.parent,
        effective_config_bytes=fixture.effective_config_bytes,
        planning_warm_start_path=fixture.warm_start_path,
        planning_child_source_repair_path=fixture.child_path,
        formal_run_id=fixture.stage.parent.name,
    ) as active:
        with pytest.raises(
            fixture.terminal_recovery.TerminalRecoveryError,
            match=(
                "completed resource prefix|incomplete|resource lifecycle|"
                "U85 attempt1 resource prefix"
            ),
        ):
            fixture.terminal_recovery.write_stage6_preterminal_acceptance(
                stage_root=fixture.stage,
                semantic_result=semantic,
                immutable_bindings=fixture.current_immutable,
                global_checkpoint_identity=fixture.global_best,
                terminal_artifacts=terminal_artifacts,
                execution_capability=active["capability"],
            )

    assert not (
        fixture.stage / fixture.terminal_recovery.RECEIPT_NAME
    ).exists()


@pytest.mark.parametrize(
    "drift",
    (
        "missing_child",
        "child_bytes",
        "current_identity",
        "current_review",
        "current_immutable",
        "u86_before_u85",
    ),
)
def test_planning_child_machine_verifier_rejects_bound_graph_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    from test_stage6_planning_child_source_repair import _rehash_artifact

    fixture = _build_task5_planning_child_machine_fixture(
        tmp_path,
        monkeypatch,
    )
    if drift == "missing_child":
        fixture.child_path.rename(tmp_path / "removed-child-artifact.json")
    elif drift == "child_bytes":
        fixture.child_path.write_bytes(fixture.child_payload + b"\n")
    elif drift in {
        "current_identity",
        "current_review",
        "current_immutable",
    }:
        artifact = json.loads(
            fixture.child_path.read_text(encoding="utf-8")
        )
        current = artifact["current"]
        if drift == "current_identity":
            current["execution_identity"]["source_set_sha256"] = "d" * 64
            current["execution_identity_sha256"] = hashlib.sha256(
                fixture.module.ArtifactStore.canonical_json_bytes(
                    current["execution_identity"]
                )
            ).hexdigest()
        elif drift == "current_review":
            current["verified_review_authorization"][
                "quality_review_sha256"
            ] = "d" * 64
        else:
            current["immutable_bindings"]["source_set_sha256"] = "d" * 64
        fixture.child_path.write_bytes(
            fixture.module.ArtifactStore.canonical_json_bytes(
                _rehash_artifact(artifact)
            )
        )
    else:
        resource_path = fixture.stage / "resource_audit.jsonl"
        rows = [
            json.loads(line)
            for line in resource_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        segment2_index = next(
            index
            for index, row in enumerate(rows)
            if (
                row.get("phase") == "segment_start"
                and row.get("segment_index") == 2
            )
        )
        u86_pre = next(
            row
            for row in rows
            if (
                row.get("update") == 86
                and row.get("phase") == "pre"
            )
        )
        rows.insert(segment2_index + 1, dict(u86_pre))
        resource_path.write_bytes(_task5_jsonl_bytes(rows))

    with pytest.raises(
        fixture.module.Stage6WorkflowError,
        match=(
            "planning child|warm-start|source-repair|resource|"
            "identity|authorization|evidence|binding"
        ),
    ):
        _run_task5_planning_child_semantic_verifier(fixture)
    if drift == "u86_before_u85":
        assert fixture.critical_calls["resource"] == 1


def test_planning_child_machine_verifier_rejects_rehashed_inflated_resource_prefix_line_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_task5_planning_child_machine_fixture(
        tmp_path,
        monkeypatch,
    )
    from dataclasses import replace

    acceptance = copy.deepcopy(
        dict(fixture.recovery_capability.acceptance_binding)
    )
    binding = acceptance["journal_prefixes"][
        "resource_audit.jsonl"
    ]
    binding["line_count"] += 1
    fixture.recovery_capability = replace(
        fixture.recovery_capability,
        acceptance_binding=acceptance,
    )

    with pytest.raises(
        fixture.module.Stage6WorkflowError,
        match="planning child|resource|prefix|line count",
    ):
        _run_task5_planning_child_semantic_verifier(fixture)

    assert fixture.critical_calls["context"] == 1
    assert fixture.critical_calls["resource"] == 1


def test_planning_child_machine_verifier_rejects_rehashed_deflated_resource_prefix_line_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_task5_planning_child_machine_fixture(
        tmp_path,
        monkeypatch,
    )
    from dataclasses import replace

    acceptance = copy.deepcopy(
        dict(fixture.recovery_capability.acceptance_binding)
    )
    binding = acceptance["journal_prefixes"][
        "resource_audit.jsonl"
    ]
    binding["line_count"] -= 1
    fixture.recovery_capability = replace(
        fixture.recovery_capability,
        acceptance_binding=acceptance,
    )

    with pytest.raises(
        fixture.module.Stage6WorkflowError,
        match="planning child|resource|prefix|line count",
    ):
        _run_task5_planning_child_semantic_verifier(fixture)

    assert fixture.critical_calls["context"] == 1
    assert fixture.critical_calls["resource"] == 1


def test_stage6_source_sets_bind_the_complete_sensor_cache_stack() -> None:
    module = _module()

    assert module.STAGE6_OPTIONAL_MANIFEST_BOUND_ARTIFACTS[-2:] == (
        "source-repair-frontier-recovery.json",
        "source-repair-sensor-acceleration.json",
    )
    assert {
        "docs/superpowers/plans/2026-07-22-ppo-stage6-sensor-hotpath-acceleration.md",
        "docs/superpowers/plans/2026-07-22-ppo-stage6-coverable-mask-cache.md",
        "docs/superpowers/specs/2026-07-22-ppo-stage6-sensor-hotpath-acceleration-design-addendum.md",
        "docs/superpowers/specs/2026-07-22-ppo-stage6-coverable-mask-cache-design-addendum.md",
        "scripts/benchmark_ppo_stage6_sensor_hotpath.py",
        "scripts/prewarm_ppo_stage6_coverage_cache.py",
        "src/lunar_exploration_ppo/env/action_execution.py",
        "src/lunar_exploration_ppo/env/coverage_cache.py",
        "src/lunar_exploration_ppo/env/sensor_model.py",
        "src/lunar_exploration_ppo/workflows/stage6_coverage_cache.py",
    } <= set(module.STAGE6_PRODUCTION_SOURCE_PATHS)
    assert {
        "tests/ppo_highres_frontier/test_stage6_coverage_cache.py",
        "tests/ppo_highres_frontier/test_stage6_coverage_cache_workflow.py",
        "tests/ppo_highres_frontier/test_stage6_sensor_acceleration.py",
    } <= set(module.STAGE6_TEST_SOURCE_PATHS)
    assert "tests/ppo_highres_frontier/test_stage1_smoke_env_r1.py" not in module.STAGE6_TEST_SOURCE_PATHS
    assert len(module.STAGE6_TEST_SOURCE_PATHS) == 30
    assert len(module.STAGE6_SOURCE_PATHS) == len(set(module.STAGE6_SOURCE_PATHS))


def test_stage6_input_pin_requests_bind_planner_runtime_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore

    module = _module()
    package_root = tmp_path / "editable" / "src" / "path_planner"
    source_text = {
        "__init__.py": "PACKAGE = True\n",
        "platform.py": "PLATFORM = True\n",
        "core/__init__.py": "CORE = True\n",
        "core/models.py": "MODELS = True\n",
        "postprocess/__init__.py": "POSTPROCESS = True\n",
        "regions/__init__.py": "REGIONS = True\n",
        "search/__init__.py": "SEARCH = True\n",
        "search/astar.py": "ASTAR = True\n",
        "trajectory/__init__.py": "TRAJECTORY = True\n",
    }
    rows: list[dict[str, object]] = []
    for relative, text in sorted(source_text.items()):
        path = package_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = text.encode("utf-8")
        path.write_bytes(payload)
        rows.append(
            {
                "path": relative,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    planner_identity = {
        "schema_version": (
            "stage6_path_planner_legacy_runtime_source_set/v1"
        ),
        "distribution_name": "path-planner",
        "distribution_version": "0.1.0",
        "package_root": package_root.resolve().as_posix(),
        "source_set_sha256": hashlib.sha256(
            ArtifactStore.canonical_json_bytes(rows)
        ).hexdigest(),
        "paths": rows,
        "resolved_symbols": {
            "AStarPlanner": "search/astar.py",
            "Cell": "core/models.py",
            "CostGrid": "core/models.py",
            "GridSpec": "core/models.py",
            "NeighborPolicy": "core/models.py",
            "PlanRequest": "core/models.py",
        },
        "forbidden_loaded_prefix": "path_planner.v2",
    }
    stage = tmp_path / "out" / "run" / "s6"
    stage.mkdir(parents=True)
    amendment = {
        "schema_version": "stage6_planning_child_source_repair/v1",
        "mode": "same_run_exact_resume_after_reviewed_source_repair/v1",
        "current": {"planner_runtime_identity": planner_identity},
        "immutable_inputs": {},
        "append_only_prefixes": {},
    }
    amendment["canonical_sha256"] = hashlib.sha256(
        ArtifactStore.canonical_json_bytes(amendment)
    ).hexdigest()
    amendment_path = stage / "planning-child-source-repair.json"
    amendment_path.write_bytes(ArtifactStore.canonical_json_bytes(amendment))
    child_repair_module = importlib.import_module(
        "lunar_exploration_ppo.workflows."
        "stage6_planning_child_source_repair"
    )
    monkeypatch.setattr(
        child_repair_module,
        "resolve_legacy_path_planner_runtime_identity",
        lambda: planner_identity,
        raising=False,
    )

    requests = module._stage6_input_pin_requests(
        repo_root=tmp_path / "repo",
        config_path=tmp_path / "config.json",
        stage5_authority=SimpleNamespace(snapshots=()),
        review_authorization_handle=SimpleNamespace(evidence_paths=()),
        planning_child_source_repair_path=amendment_path,
    )

    request_paths = {path for _label, path in requests}
    assert {
        (package_root / str(row["path"])).resolve() for row in rows
    }.issubset(request_paths)


def test_source_repair_loader_forwards_and_binds_ordinal6_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    source_repair = importlib.import_module(
        "lunar_exploration_ppo.workflows.stage6_source_repair"
    )
    stage = tmp_path / FORMAL_STAGE6_RUN_ID / "s6"
    stage.mkdir(parents=True)
    current_identity = {"source_set_sha256": "a" * 64}
    current_review = {"formal_run_id": FORMAL_STAGE6_RUN_ID}
    current_immutable = {"coverage_cache_manifest_sha256": "b" * 64}
    value = {
        "current": {
            "execution_identity": current_identity,
            "verified_review_authorization": current_review,
            "immutable_bindings": current_immutable,
        }
    }
    payloads = tuple(
        module.ArtifactStore.canonical_json_bytes({**value, "ordinal": ordinal})
        for ordinal in range(1, 7)
    )
    captured: dict[str, object] = {}

    def fake_load(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            amendment_sha256=hashlib.sha256(payloads[0]).hexdigest(),
            amendment_size_bytes=len(payloads[0]),
            continuation_sha256=hashlib.sha256(payloads[1]).hexdigest(),
            continuation_size_bytes=len(payloads[1]),
            supplement_sha256=hashlib.sha256(payloads[2]).hexdigest(),
            supplement_size_bytes=len(payloads[2]),
            closure_sha256=hashlib.sha256(payloads[3]).hexdigest(),
            closure_size_bytes=len(payloads[3]),
            frontier_recovery_sha256=hashlib.sha256(payloads[4]).hexdigest(),
            frontier_recovery_size_bytes=len(payloads[4]),
            sensor_acceleration_sha256=hashlib.sha256(payloads[5]).hexdigest(),
            sensor_acceleration_size_bytes=len(payloads[5]),
        )

    monkeypatch.setattr(source_repair, "load_stage6_source_repair_context", fake_load)

    context = module._load_stage6_source_repair_for_current_identity(
        stage=stage,
        current_execution_identity=current_identity,
        amendment_payload=payloads[0],
        continuation_payload=payloads[1],
        supplement_payload=payloads[2],
        closure_payload=payloads[3],
        frontier_recovery_payload=payloads[4],
        sensor_acceleration_payload=payloads[5],
    )

    assert context.sensor_acceleration_sha256 == hashlib.sha256(
        payloads[5]
    ).hexdigest()
    assert captured["current_verified_review_authorization"] == current_review
    assert captured["current_immutable_bindings"] == current_immutable


def test_lineage_authorization_forwards_ordinal6_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    stage = tmp_path / FORMAL_STAGE6_RUN_ID / "s6"
    stage.mkdir(parents=True)
    verified = {
        "formal_run_id": FORMAL_STAGE6_RUN_ID,
        "changed_path_set_sha256": "1" * 64,
        "authorization_file_sha256": "2" * 64,
        "review_identity_sha256": "3" * 64,
        "reviewed_prospective_git_tree": "4" * 40,
        "frozen_diff_sha256": "5" * 64,
        "spec_review_sha256": "6" * 64,
        "quality_review_sha256": "7" * 64,
        "prospective_tree_sha256": "8" * 64,
        "source_set_sha256": "9" * 64,
        "config_sha256": "a" * 64,
        "data_sha256": "b" * 64,
        "environment_sha256": "c" * 64,
    }
    immutable = {
        **verified,
        "review_authorization_record_sha256": module._canonical_sha256(verified),
    }
    immutable.pop("reviewed_prospective_git_tree")
    immutable["reviewed_prospective_git_tree"] = verified[
        "reviewed_prospective_git_tree"
    ]
    origin_identity = {"source_set_sha256": "d" * 64}
    lineage_payload = module.ArtifactStore.canonical_json_bytes(
        {
            "schema_version": "stage6_lineage_audit/v2",
            "execution_identity": origin_identity,
            "immutable_bindings": immutable,
            "verified_review_authorization": verified,
        }
    )
    payloads = tuple(f"ordinal-{ordinal}".encode() for ordinal in range(1, 7))
    for name, payload in zip(
        module.STAGE6_OPTIONAL_MANIFEST_BOUND_ARTIFACTS,
        payloads,
        strict=True,
    ):
        (stage / name).write_bytes(payload)
    captured: dict[str, object] = {}
    current_review = {"formal_run_id": FORMAL_STAGE6_RUN_ID, "current": True}

    monkeypatch.setattr(
        module,
        "validate_stage6_verified_review_authorization",
        lambda *args, **kwargs: dict(verified),
    )
    monkeypatch.setattr(
        module,
        "stage6_execution_identity",
        lambda **kwargs: {"source_set_sha256": "e" * 64},
    )

    def fake_load(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            origin_execution_identity=origin_identity,
            origin_immutable_bindings=immutable,
            origin_verified_review_authorization=verified,
            current_verified_review_authorization=current_review,
        )

    monkeypatch.setattr(
        module,
        "_load_stage6_source_repair_for_current_identity",
        fake_load,
    )

    result = module._stage6_lineage_review_authorization(
        stage,
        payload=lineage_payload,
        repo_root=tmp_path,
        source_repair_amendment_payload=payloads[0],
        source_repair_continuation_payload=payloads[1],
        source_repair_supplement_payload=payloads[2],
        source_repair_closure_payload=payloads[3],
        source_repair_frontier_recovery_payload=payloads[4],
        source_repair_sensor_acceleration_payload=payloads[5],
    )

    assert result == current_review
    assert captured["sensor_acceleration_payload"] == payloads[5]


@pytest.mark.parametrize("seed", (20260717, 20260718, 20260719, 20260720))
@pytest.mark.parametrize(
    "relative_templates",
    (
        ("checkpoints/seed-{seed}/latest.json",),
        (
            "checkpoints/seed-{seed}/update-00000010/checkpoint.pt",
            "checkpoints/seed-{seed}/update-00000010/manifest.json",
            "checkpoints/seed-{seed}/update-00000010/complete.json",
        ),
        ("episode-traces/validation-seed-{seed}-update-010.jsonl",),
    ),
)
def test_extra_seed_manifest_children_are_rejected(
    tmp_path: Path,
    seed: int,
    relative_templates: tuple[str, ...],
) -> None:
    module = _module()
    relatives = tuple(template.format(seed=seed) for template in relative_templates)
    assert all(
        not module._is_allowed_stage6_child(relative) for relative in relatives
    )

    for index, relative in enumerate(relatives):
        stage = tmp_path / f"case-{index}/s6"
        _write_manifest_fixture(module, stage)
        child = stage / relative
        child.parent.mkdir(parents=True, exist_ok=True)
        child.write_bytes(b"extra-seed-artifact")
        with pytest.raises(module.Stage6WorkflowError, match="unknown child"):
            module._stage6_manifest_relative_paths(stage)


def test_manifest_binds_required_artifacts_current_code_and_rejects_authority_tamper(
    tmp_path: Path,
    active_execution_authority_factory,
) -> None:
    module = _module()
    stage = tmp_path / FORMAL_STAGE6_RUN_ID / "s6"
    active = active_execution_authority_factory(run_root=stage.parent)
    _write_manifest_fixture(
        module,
        stage,
        execution_identity=active["identity"],
        verified_review=active["verified"],
    )
    _write_authorized_manifest(module, stage, active)

    handle = module.verify_stage6_manifest(stage_root=stage, repo_root=ROOT)
    assert handle.source_identity == module.stage6_source_identity(ROOT)
    handle.require_current()
    manifest_path = stage / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(
        (stage / "preterminal_acceptance.json").read_text(encoding="utf-8")
    )
    lineage = json.loads(
        (stage / "lineage_audit.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "stage6_sha256_manifest/v3"
    assert manifest["preterminal_evidence_binding"] == receipt[
        "preterminal_evidence_binding"
    ]
    assert manifest["verified_review_authorization"] == lineage[
        "verified_review_authorization"
    ]
    assert "preterminal_acceptance.json" in {
        row["path"] for row in manifest["artifacts"]
    }
    manifest_before_verify = manifest_path.read_bytes()
    first_identity = module.write_or_verify_stage6_manifest(
        stage,
        repo_root=ROOT,
        execution_capability=active["capability"],
        run_lease=active["lease"],
    )
    second_identity = module.write_or_verify_stage6_manifest(
        stage,
        repo_root=ROOT,
        execution_capability=active["capability"],
        run_lease=active["lease"],
    )
    assert first_identity == second_identity == {
        "sha256": hashlib.sha256(manifest_before_verify).hexdigest(),
        "size_bytes": len(manifest_before_verify),
    }
    assert manifest_path.read_bytes() == manifest_before_verify

    drifted_manifest = dict(manifest)
    drifted_review = dict(drifted_manifest["verified_review_authorization"])
    drifted_review["quality_review_sha256"] = "f" * 64
    drifted_manifest["verified_review_authorization"] = drifted_review
    manifest_path.write_bytes(module.ArtifactStore.canonical_json_bytes(drifted_manifest))
    with pytest.raises(module.Stage6WorkflowError, match="review.*authorization"):
        module.verify_stage6_manifest(stage_root=stage, repo_root=ROOT)
    manifest_path.write_bytes(module.ArtifactStore.canonical_json_bytes(manifest))

    authority = stage / "review.json"
    authority.write_text("{}\n", encoding="utf-8")
    with pytest.raises(module.Stage6WorkflowError, match="authority"):
        module.verify_stage6_manifest(stage_root=stage, repo_root=ROOT)
    authority.unlink()

    report = stage / "standard_eval_report.md"
    report.write_bytes(report.read_bytes() + b"tamper\n")
    with pytest.raises(module.Stage6WorkflowError, match="manifest"):
        module.verify_stage6_manifest(stage_root=stage, repo_root=ROOT)

    report.write_bytes(("standard_eval_report.md\n").encode("utf-8"))
    (stage / "episode-traces/unknown.bin").write_bytes(b"unknown")
    with pytest.raises(module.Stage6WorkflowError, match="unknown"):
        module.verify_stage6_manifest(stage_root=stage, repo_root=ROOT)


def test_manifest_publication_rejects_old_incomplete_receipt(
    tmp_path: Path,
    active_execution_authority_factory,
) -> None:
    module = _module()
    stage = tmp_path / FORMAL_STAGE6_RUN_ID / "s6"
    active = active_execution_authority_factory(run_root=stage.parent)
    _write_manifest_fixture(
        module,
        stage,
        execution_identity=active["identity"],
        verified_review=active["verified"],
    )
    receipt_path = stage / "preterminal_acceptance.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["schema_version"] = "stage6_preterminal_acceptance/v1"
    del receipt["preterminal_evidence_binding"]
    receipt["semantic_verification"]["schema_version"] = (
        "stage6_machine_acceptance_verification/v1"
    )
    del receipt["semantic_verification"]["evidence_binding"]
    receipt_path.write_bytes(module.ArtifactStore.canonical_json_bytes(receipt))

    with pytest.raises(
        module.Stage6WorkflowError,
        match="receipt.*incomplete|receipt.*drifted",
    ):
        _write_authorized_manifest(module, stage, active)
    assert not (stage / "manifest.json").exists()


@pytest.mark.parametrize("target_location", ("inside", "outside"))
def test_manifest_identity_rejects_file_symlink_targets(
    tmp_path: Path,
    target_location: str,
) -> None:
    module = _module()
    stage = tmp_path / "file-link/run/s6"
    stage.mkdir(parents=True)
    target = (
        stage / "plain-target.bin"
        if target_location == "inside"
        else tmp_path / "outside-target.bin"
    )
    target.write_bytes(b"same-payload")
    linked = stage / "linked.bin"
    _make_file_symlink(linked, target)

    with pytest.raises(module.Stage6WorkflowError, match="link|reparse"):
        module._FileIdentity.capture(linked, base=stage)


@pytest.mark.parametrize("target_location", ("inside", "outside"))
def test_manifest_identity_rejects_parent_directory_reparse_targets(
    tmp_path: Path,
    target_location: str,
) -> None:
    module = _module()
    stage = tmp_path / "directory-link/run/s6"
    stage.mkdir(parents=True)
    target = (
        stage / "plain-target"
        if target_location == "inside"
        else tmp_path / "outside-target"
    )
    target.mkdir()
    (target / "artifact.bin").write_bytes(b"same-payload")
    linked = stage / "linked"
    _make_directory_reparse(linked, target)

    with pytest.raises(module.Stage6WorkflowError, match="link|reparse"):
        module._FileIdentity.capture(linked / "artifact.bin", base=stage)


def test_manifest_identity_rejects_ordinary_file_swap_during_secure_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    path_security = importlib.import_module(
        "lunar_exploration_ppo.utils.path_security"
    )
    stage = tmp_path / "ordinary-swap/run/s6"
    stage.mkdir(parents=True)
    artifact = stage / "artifact.bin"
    replacement = stage / "replacement.bin"
    backup = stage / "artifact.backup"
    trusted_payload = b"trusted-payload"
    artifact.write_bytes(trusted_payload)
    replacement.write_bytes(b"attacker-payload")
    original_read_bytes = Path.read_bytes
    events: list[str] = []

    def swap_hook(event: str, path: Path, descriptor: int | None = None) -> None:
        del descriptor
        if Path(path) != artifact:
            return
        events.append(event)
        if event == "after_path_identity":
            artifact.replace(backup)
            replacement.replace(artifact)
        elif event == "after_descriptor_read":
            artifact.replace(replacement)
            backup.replace(artifact)

    def legacy_read_bytes(path: Path) -> bytes:
        if Path(path) != artifact:
            return original_read_bytes(path)
        swap_hook("after_path_identity", artifact)
        payload = original_read_bytes(artifact)
        swap_hook("after_descriptor_read", artifact)
        return payload

    monkeypatch.setattr(
        path_security,
        "_secure_read_event",
        swap_hook,
        raising=False,
    )
    monkeypatch.setattr(Path, "read_bytes", legacy_read_bytes)

    with pytest.raises(module.Stage6WorkflowError, match="changed|identity"):
        module._FileIdentity.capture(artifact, base=stage)

    assert events == ["after_path_identity", "after_descriptor_read"]
    assert original_read_bytes(artifact) == trusted_payload


@pytest.mark.parametrize("target_location", ("inside", "outside"))
def test_manifest_identity_rejects_hard_link_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_location: str,
) -> None:
    module = _module()
    path_security = importlib.import_module(
        "lunar_exploration_ppo.utils.path_security"
    )
    stage = tmp_path / "hard-link/run/s6"
    stage.mkdir(parents=True)
    target = (
        stage / "plain-target.bin"
        if target_location == "inside"
        else tmp_path / "outside-target.bin"
    )
    target.write_bytes(b"same-payload")
    linked = stage / "linked.bin"
    try:
        os.link(target, linked)
    except (NotImplementedError, OSError):
        linked.write_bytes(target.read_bytes())
        monkeypatch.setattr(
            path_security,
            "_stat_link_count",
            lambda metadata: 2,
            raising=False,
        )

    with pytest.raises(
        module.Stage6WorkflowError,
        match="hard link|link count|alias",
    ):
        module._FileIdentity.capture(linked, base=stage)


def test_manifest_traversal_rejects_nested_directory_reparse(
    tmp_path: Path,
) -> None:
    module = _module()
    stage = tmp_path / "nested-link/run/s6"
    _write_manifest_fixture(module, stage)
    audit = stage / "preflight/audit.json"
    audit.unlink()
    audit.parent.rmdir()
    target = tmp_path / "nested-link-target"
    target.mkdir()
    (target / "audit.json").write_bytes(b'{"passed":true}\n')
    _make_directory_reparse(stage / "preflight", target)

    with pytest.raises(module.Stage6WorkflowError, match="link|reparse"):
        module._stage6_manifest_relative_paths(stage)


def test_manifest_traversal_rejects_top_level_file_reparse_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    stage = tmp_path / "top-file-reparse/run/s6"
    _write_manifest_fixture(module, stage)
    flagged = stage / "report.md"
    original = module._is_link_or_reparse
    monkeypatch.setattr(
        module,
        "_is_link_or_reparse",
        lambda path: Path(path) == flagged or original(Path(path)),
    )

    with pytest.raises(module.Stage6WorkflowError, match="link|reparse"):
        module._stage6_manifest_relative_paths(stage)


@pytest.mark.parametrize("target_location", ("inside", "outside"))
def test_manifest_snapshot_rejects_capture_after_directory_alias_swap(
    tmp_path: Path,
    target_location: str,
) -> None:
    module = _module()
    stage = tmp_path / "alias-swap/run/s6"
    original_parent = stage / "plain"
    original_parent.mkdir(parents=True)
    artifact = original_parent / "artifact.bin"
    artifact.write_bytes(b"same-payload")
    snapshot = module._FileIdentity.capture(artifact, base=stage)

    artifact.unlink()
    original_parent.rmdir()
    target = (
        stage / "replacement"
        if target_location == "inside"
        else tmp_path / "outside-replacement"
    )
    target.mkdir()
    (target / "artifact.bin").write_bytes(b"same-payload")
    _make_directory_reparse(original_parent, target)

    with pytest.raises(module.Stage6WorkflowError, match="link|reparse|changed"):
        snapshot.require_current("manifest alias swap")


def test_manifest_handle_rejects_member_inserted_after_capture(
    tmp_path: Path,
    active_execution_authority_factory,
) -> None:
    module = _module()
    stage = tmp_path / FORMAL_STAGE6_RUN_ID / "s6"
    active = active_execution_authority_factory(run_root=stage.parent)
    _write_manifest_fixture(
        module,
        stage,
        execution_identity=active["identity"],
        verified_review=active["verified"],
    )
    _write_authorized_manifest(module, stage, active)
    handle = module.verify_stage6_manifest(stage_root=stage, repo_root=ROOT)

    (stage / "episode-traces/unknown.bin").write_bytes(b"unknown")

    with pytest.raises(module.Stage6WorkflowError, match="member|graph|unknown"):
        handle.require_current()


def test_manifest_rejects_duplicate_file_id_across_allowed_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_execution_authority_factory,
) -> None:
    module = _module()
    path_security = importlib.import_module(
        "lunar_exploration_ppo.utils.path_security"
    )
    stage = tmp_path / FORMAL_STAGE6_RUN_ID / "s6"
    active = active_execution_authority_factory(run_root=stage.parent)
    _write_manifest_fixture(
        module,
        stage,
        execution_identity=active["identity"],
        verified_review=active["verified"],
    )
    first = stage / "report.md"
    second = stage / "standard_training_report.md"
    second.unlink()
    try:
        os.link(first, second)
    except (NotImplementedError, OSError):
        second.write_bytes(first.read_bytes())
        monkeypatch.setattr(
            path_security,
            "_stat_file_id",
            lambda metadata: (101, 202),
            raising=False,
        )
    else:
        monkeypatch.setattr(
            path_security,
            "_stat_link_count",
            lambda metadata: 1,
            raising=False,
        )

    with pytest.raises(
        module.Stage6WorkflowError,
        match="File ID|file identity|alias|duplicate",
    ):
        _write_authorized_manifest(module, stage, active)


def test_receipt_bound_v3_terminal_manifest_is_tamper_safe_and_idempotent(
    tmp_path: Path,
    active_execution_authority_factory,
) -> None:
    module = _module()
    from test_stage6_terminal_recovery import (
        ABSENT_HASH,
        PRETERMINAL_STATES,
        _global_checkpoint_identity,
        _phase_row,
        _resource,
        _semantic_result,
    )
    from test_stage6_machine_preflight import _valid_environment_identity
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        append_resource_segment_start,
    )
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    stage = tmp_path / FORMAL_STAGE6_RUN_ID / "s6"
    active = active_execution_authority_factory(
        run_root=stage.parent,
        environment_identity=_valid_environment_identity(),
    )
    _write_manifest_fixture(
        module,
        stage,
        execution_identity=active["identity"],
        verified_review=active["verified"],
    )
    (stage / "preterminal_acceptance.json").unlink()
    manifest_path = stage / "manifest.json"
    resource_path = stage / "resource_audit.jsonl"
    lineage = json.loads((stage / "lineage_audit.json").read_text(encoding="utf-8"))
    immutable = lineage["immutable_bindings"]
    environment_sha256 = hashlib.sha256(
        module.ArtifactStore.canonical_json_bytes(immutable["environment_identity"])
    ).hexdigest()
    lineage["execution_identity"]["environment_sha256"] = environment_sha256
    lineage["verified_review_authorization"][
        "environment_sha256"
    ] = environment_sha256
    lineage["verified_review_authorization"]["review_identity_sha256"] = (
        hashlib.sha256(
            module.ArtifactStore.canonical_json_bytes(lineage["execution_identity"])
        ).hexdigest()
    )
    immutable["environment_sha256"] = environment_sha256
    immutable["review_identity_sha256"] = lineage[
        "verified_review_authorization"
    ]["review_identity_sha256"]
    immutable["review_authorization_record_sha256"] = hashlib.sha256(
        module.ArtifactStore.canonical_json_bytes(
            lineage["verified_review_authorization"]
        )
    ).hexdigest()
    (stage / "lineage_audit.json").write_bytes(
        module.ArtifactStore.canonical_json_bytes(lineage)
    )
    global_checkpoint = _global_checkpoint_identity()
    phase_rows = []
    previous = ABSENT_HASH
    for index, state in enumerate(PRETERMINAL_STATES):
        bindings = {
            **immutable,
            "checkpoint_sha256": (
                "9" * 64
                if index == 0
                else global_checkpoint["checkpoint_sha256"]
            ),
        }
        row = _phase_row(state, previous, bindings)
        phase_rows.append(row)
        previous = row["record_hash"]
    (stage / "phase-state.jsonl").write_bytes(
        b"".join(
            (
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            for row in phase_rows
        )
    )
    resource_path.unlink()
    append_resource_segment_start(
        resource_path,
        first_sample=_resource(
            root_pid=os.getpid(),
            sample_count=1,
            rss_bytes=1024,
        ),
    )
    terminal_artifacts = {
        "summary.json": module.ArtifactStore.canonical_json_bytes(
            {
                "schema_version": "stage6_summary/v1",
                "machine_passed": True,
                "state": "awaiting_independent_review",
            }
        ),
        "routing.json": module.ArtifactStore.canonical_json_bytes(
            {
                "schema_version": "stage6_routing/v1",
                "machine_passed": True,
                "route": "awaiting_independent_review",
            }
        ),
        "standard_training_report.md": b"# training\n",
        "standard_eval_report.md": b"# evaluation\n",
        "report.md": b"# Stage 6\n",
    }
    for relative in terminal_artifacts:
        (stage / relative).unlink()
    with stage6_terminal_recovery._capture_preterminal_evidence_handle(
        stage
    ) as evidence:
        semantic_result = _semantic_result(evidence.binding)
    receipt_identity = stage6_terminal_recovery.write_stage6_preterminal_acceptance(
        stage_root=stage,
        semantic_result=semantic_result,
        immutable_bindings=immutable,
        global_checkpoint_identity=global_checkpoint,
        terminal_artifacts=terminal_artifacts,
        execution_capability=active["capability"],
    )
    terminal = stage6_terminal_recovery.append_stage6_recovery_resource_terminal(
        stage_root=stage,
        terminal_resource=_resource(
            root_pid=os.getpid(),
            sample_count=2,
            rss_bytes=2048,
        ),
        execution_capability=active["capability"],
    )
    assert terminal["preterminal_acceptance"] == receipt_identity
    stage6_terminal_recovery.recover_stage6_terminal_commit(
        stage_root=stage,
        manifest_committer=lambda candidate: module.write_or_verify_stage6_manifest(
            candidate,
            repo_root=ROOT,
        ),
        execution_capability=active["capability"],
    )
    terminal_bound_manifest = manifest_path.read_bytes()

    report = stage / "report.md"
    original_report = report.read_bytes()
    report.write_bytes(original_report + b"tamper\n")
    with pytest.raises(module.Stage6WorkflowError, match="manifest|rebind"):
        module.rebind_stage6_manifest_for_terminal_resource(
            stage_root=stage,
            repo_root=ROOT,
        )
    report.write_bytes(original_report)

    module.rebind_stage6_manifest_for_terminal_resource(
        stage_root=stage,
        repo_root=ROOT,
    )
    rebound_manifest = manifest_path.read_bytes()
    assert rebound_manifest == terminal_bound_manifest
    module.verify_stage6_manifest(stage_root=stage, repo_root=ROOT).require_current()

    module.rebind_stage6_manifest_for_terminal_resource(
        stage_root=stage,
        repo_root=ROOT,
    )
    assert manifest_path.read_bytes() == rebound_manifest

    original_resource = resource_path.read_bytes()
    rows = [
        json.loads(line)
        for line in original_resource.decode("utf-8").splitlines()
    ]
    rows[-1]["schema_version"] = "stage6_terminal_resource_evidence/v2"
    resource_path.write_bytes(
        b"".join(
            (
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            for row in rows
        )
    )
    manifest_path.unlink()
    _write_authorized_manifest(module, stage, active)
    with pytest.raises(module.Stage6WorkflowError, match="terminal|v3|lifecycle"):
        module.rebind_stage6_manifest_for_terminal_resource(
            stage_root=stage,
            repo_root=ROOT,
        )


def test_stage6_source_paths_bind_approved_stage6a_package_exactly() -> None:
    module = _module()
    repaired_frontier = "src/lunar_exploration_ppo/env/frontier.py"
    repaired_frontier_test = "tests/ppo_highres_frontier/test_stage2_frontier.py"
    approved_production_additions = (
        "docs/superpowers/plans/2026-07-10-ppo-highres-frontier-map-exploration.md",
        "docs/superpowers/plans/2026-07-16-ppo-stage6-single-seed-pipeline.md",
        "docs/superpowers/specs/2026-07-09-ppo-highres-frontier-map-exploration-design.md",
        "docs/superpowers/specs/2026-07-16-ppo-stage6-single-seed-pipeline-design-addendum.md",
        "scripts/create_ppo_stage6_source_repair_amendment.py",
        "src/lunar_exploration_ppo/env/scenario_catalog.py",
        "src/lunar_exploration_ppo/utils/artifact_io.py",
        "src/lunar_exploration_ppo/utils/stage6_input_pinning.py",
        "src/lunar_exploration_ppo/utils/resource_lifecycle.py",
        "src/lunar_exploration_ppo/workflows/stage6_review_authorization.py",
        "src/lunar_exploration_ppo/workflows/stage6_source_repair.py",
        "src/lunar_exploration_ppo/workflows/stage6_terminal_recovery.py",
    )
    approved_test_additions = (
        "tests/ppo_highres_frontier/test_foundation.py",
        "tests/ppo_highres_frontier/test_stage2_catalog.py",
        "tests/ppo_highres_frontier/test_stage6_input_pinning.py",
        "tests/ppo_highres_frontier/test_stage6_resource_lifecycle.py",
        "tests/ppo_highres_frontier/test_stage6_review_authorization.py",
        "tests/ppo_highres_frontier/test_stage6_source_repair.py",
        "tests/ppo_highres_frontier/test_stage6_terminal_recovery.py",
    )

    assert tuple(
        path
        for path in module.STAGE6_PRODUCTION_SOURCE_PATHS
        if path in approved_production_additions
    ) == approved_production_additions
    assert tuple(
        path
        for path in module.STAGE6_TEST_SOURCE_PATHS
        if path in approved_test_additions
    ) == approved_test_additions
    assert module.STAGE6_SOURCE_PATHS == (
        *module.STAGE6_PRODUCTION_SOURCE_PATHS,
        *module.STAGE6_TEST_SOURCE_PATHS,
    )
    assert module.STAGE6_PRODUCTION_SOURCE_PATHS.count(repaired_frontier) == 1
    assert module.STAGE6_TEST_SOURCE_PATHS.count(repaired_frontier_test) == 1
    source_identity = module.stage6_source_identity(ROOT)
    rows = {row["path"]: row for row in source_identity["paths"]}
    assert [row["path"] for row in source_identity["paths"]].count(
        repaired_frontier
    ) == 1
    assert [row["path"] for row in source_identity["paths"]].count(
        repaired_frontier_test
    ) == 1
    foundation_relative = "tests/ppo_highres_frontier/test_foundation.py"
    foundation_payload = (ROOT / foundation_relative).read_bytes()
    assert rows[foundation_relative] == {
        "path": foundation_relative,
        "sha256": hashlib.sha256(foundation_payload).hexdigest(),
        "size_bytes": len(foundation_payload),
    }


def test_scheme_b_recovery_documents_are_bound_once_in_source_identity(
) -> None:
    module = _module()
    plan_path = (
        "docs/superpowers/plans/"
        "2026-07-26-ppo-stage6-planning-child-recovery-"
        "capability-consolidation.md"
    )
    spec_path = (
        "docs/superpowers/specs/"
        "2026-07-26-ppo-stage6-planning-child-recovery-"
        "capability-consolidation-design.md"
    )

    assert module.STAGE6_PRODUCTION_SOURCE_PATHS.count(plan_path) == 1
    assert module.STAGE6_PRODUCTION_SOURCE_PATHS.count(spec_path) == 1
    assert module.STAGE6_PRODUCTION_SOURCE_PATHS[
        module.STAGE6_PRODUCTION_SOURCE_PATHS.index(
            "docs/superpowers/plans/"
            "2026-07-25-ppo-stage6-planning-child-source-repair.md"
        )
        + 1
    ] == plan_path
    assert module.STAGE6_PRODUCTION_SOURCE_PATHS[
        module.STAGE6_PRODUCTION_SOURCE_PATHS.index(
            "docs/superpowers/specs/"
            "2026-07-25-ppo-stage6-planning-child-source-repair-design.md"
        )
        + 1
    ] == spec_path

    rows = {
        row["path"]: row
        for row in module.stage6_source_identity(ROOT)["paths"]
        if row["path"] in {plan_path, spec_path}
    }
    assert set(rows) == {plan_path, spec_path}
    for relative_path in (plan_path, spec_path):
        payload = (ROOT / relative_path).read_bytes()
        assert rows[relative_path] == {
            "path": relative_path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }


def test_stage6_frontier_oracle_is_registered_once_in_source_identity_and_pins() -> None:
    module = _module()
    frontier_oracle = "src/lunar_exploration_ppo/env/frontier_oracle.py"

    assert module.STAGE6_PRODUCTION_SOURCE_PATHS.count(frontier_oracle) == 1
    assert module.STAGE6_SOURCE_PATHS.count(frontier_oracle) == 1
    assert (
        module.STAGE6_PRODUCTION_SOURCE_PATHS.index(
            "src/lunar_exploration_ppo/env/frontier.py"
        )
        < module.STAGE6_PRODUCTION_SOURCE_PATHS.index(frontier_oracle)
        < module.STAGE6_PRODUCTION_SOURCE_PATHS.index(
            "src/lunar_exploration_ppo/env/map_state.py"
        )
    )

    payload = (ROOT / frontier_oracle).read_bytes()
    source_rows = [
        row
        for row in module.stage6_source_identity(ROOT)["paths"]
        if row["path"] == frontier_oracle
    ]
    assert source_rows == [
        {
            "path": frontier_oracle,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    ]

    requests = module._stage6_input_pin_requests(
        repo_root=ROOT,
        config_path=CONFIG,
        stage5_authority=SimpleNamespace(snapshots=()),
        review_authorization_handle=SimpleNamespace(evidence_paths=()),
    )
    assert [label for label, _path in requests].count(
        f"source:{frontier_oracle}"
    ) == 1
    assert [Path(path) for _label, path in requests].count(
        (ROOT / frontier_oracle).resolve()
    ) == 1


def test_stage6_prospective_execution_identity_includes_frontier_oracle_once() -> None:
    module = _module()
    frontier_oracle = "src/lunar_exploration_ppo/env/frontier_oracle.py"

    identity = module.stage6_execution_identity(
        repo_root=ROOT,
        config_path=CONFIG,
    )

    assert identity["changed_paths"].count(frontier_oracle) == 1


def test_stage5_and_review_authority_paths_are_all_requested_for_input_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    repo = tmp_path / "repo"
    for relative in module.STAGE6_SOURCE_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
    config = repo / "configs/ppo_highres_frontier_stage6_v1.json"
    dem = tmp_path / "dem.tif"
    slope = tmp_path / "slope.tif"
    dem.write_bytes(b"dem")
    slope.write_bytes(b"slope")
    coverage_cache_manifest = tmp_path / "coverage-cache-manifest.json"
    coverage_cache_manifest.write_bytes(b"{}\n")
    monkeypatch.setattr(
        module,
        "STAGE6_DATA_INPUT_PATHS",
        (("dem", dem), ("slope_provenance", slope)),
    )
    monkeypatch.setattr(
        module,
        "STAGE6_COVERAGE_CACHE_MANIFEST_PATH",
        coverage_cache_manifest,
        raising=False,
    )

    class Snapshot:
        def __init__(self, path: Path) -> None:
            self.path = path

    stage5_paths = tuple(tmp_path / f"stage5-{index}.bin" for index in range(6))
    for path in stage5_paths:
        path.write_bytes(path.name.encode("utf-8"))
    authority = module.FrozenStage5AuthorityHandle(
        identity={},
        snapshots=tuple(
            (f"member-{index}", Snapshot(path))
            for index, path in enumerate(stage5_paths)
        ),
        stage5_root=tmp_path,
        repo_root=repo,
    )
    review_paths = tuple(tmp_path / f"review-{index}.json" for index in range(5))
    for path in review_paths:
        path.write_bytes(b"{}\n")

    class ReviewHandle:
        evidence_paths = review_paths

    requests = module._stage6_input_pin_requests(
        repo_root=repo,
        config_path=config,
        stage5_authority=authority,
        review_authorization_handle=ReviewHandle(),
    )

    labels = [label for label, _path in requests]
    requested_paths = [Path(path) for _label, path in requests]
    assert labels.count("config:canonical") == 1
    assert labels.count("coverage-cache:manifest") == 1
    assert requested_paths.count(coverage_cache_manifest.resolve()) == 1
    assert {
        label.removeprefix("source:")
        for label in labels
        if label.startswith("source:")
    } == set(module.STAGE6_SOURCE_PATHS)
    for relative in (
        "src/lunar_exploration_ppo/env/frontier.py",
        "tests/ppo_highres_frontier/test_stage2_frontier.py",
    ):
        assert labels.count(f"source:{relative}") == 1
        assert requested_paths.count((repo / relative).resolve()) == 1
    assert config.resolve() in requested_paths
    assert {dem.resolve(), slope.resolve()}.issubset(requested_paths)
    assert set(stage5_paths).issubset(requested_paths)
    assert set(review_paths).issubset(requested_paths)
    assert authority.evidence_paths == stage5_paths


def test_execution_identity_uses_exact_dirty_paths_and_temporary_git_index() -> None:
    module = _module()

    identity = module.stage6_execution_identity(
        repo_root=ROOT,
        config_path=CONFIG,
    )

    assert identity["schema_version"] == "stage6_execution_identity/v1"
    assert identity["base_commit"] == module.STAGE5_COMMIT
    assert identity["head_commit"] == module.STAGE5_COMMIT
    assert identity["changed_paths"] == sorted(module.STAGE6_SOURCE_PATHS)
    assert identity["changed_paths"].count(
        "src/lunar_exploration_ppo/env/frontier.py"
    ) == 1
    assert identity["changed_paths"].count(
        "tests/ppo_highres_frontier/test_stage2_frontier.py"
    ) == 1
    assert identity["real_index_empty"] is True
    assert len(identity["prospective_git_tree"]) == 40
    assert identity["prospective_tree_sha256"] == hashlib.sha256(
        identity["prospective_git_tree"].encode("ascii")
    ).hexdigest()
    assert identity["source_set_sha256"] == module.stage6_source_identity(ROOT)[
        "source_set_sha256"
    ]
    assert identity["config_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert len(identity["data_sha256"]) == 64
    assert len(identity["catalog_sha256"]) == 64
    assert identity["environment_identity"] == module.stage6_environment_identity()
    assert identity["environment_sha256"] == module.stage6_environment_sha256(
        identity["environment_identity"]
    )


def test_reviewed_source_tree_ignores_unrelated_head_and_worktree_drift(
    tmp_path: Path,
) -> None:
    module = _module()
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout.strip()

    git("init")
    git("config", "user.email", "stage6-test@example.invalid")
    git("config", "user.name", "Stage6 Test")
    reviewed_a = repo / "reviewed" / "a.py"
    reviewed_b = repo / "reviewed" / "b.py"
    reviewed_a.parent.mkdir()
    reviewed_a.write_text("VALUE = 'base-a'\n", encoding="utf-8")
    reviewed_b.write_text("VALUE = 'base-b'\n", encoding="utf-8")
    unrelated = repo / "unrelated.txt"
    unrelated.write_text("base\n", encoding="utf-8")
    git("add", "--", ".")
    git("commit", "-m", "base")
    base_commit = git("rev-parse", "HEAD")

    reviewed_a.write_text("VALUE = 'reviewed-a'\n", encoding="utf-8")
    unrelated.write_text("committed unrelated change\n", encoding="utf-8")
    git("add", "--", "unrelated.txt")
    git("commit", "-m", "unrelated head advance")
    advanced_head = git("rev-parse", "HEAD")
    assert advanced_head != base_commit

    first = module._stage6_reviewed_source_git_identity(
        repo,
        base_commit=base_commit,
        reviewed_paths=("reviewed/a.py", "reviewed/b.py"),
    )
    unrelated.write_text("dirty unrelated change\n", encoding="utf-8")
    second = module._stage6_reviewed_source_git_identity(
        repo,
        base_commit=base_commit,
        reviewed_paths=("reviewed/a.py", "reviewed/b.py"),
    )

    assert first["diagnostic_head_commit"] == advanced_head
    assert first["prospective_git_tree"] == second["prospective_git_tree"]
    assert first["changed_paths"] == ["reviewed/a.py"]
    assert first["real_index_empty"] is True

    reviewed_b.write_text("VALUE = 'reviewed-b'\n", encoding="utf-8")
    third = module._stage6_reviewed_source_git_identity(
        repo,
        base_commit=base_commit,
        reviewed_paths=("reviewed/a.py", "reviewed/b.py"),
    )
    assert third["prospective_git_tree"] != first["prospective_git_tree"]
    assert third["changed_paths"] == ["reviewed/a.py", "reviewed/b.py"]


def test_runner_is_thin_production_only_and_package_imports_are_isolated() -> None:
    module = _module()
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert imported <= {
        "__future__",
        "argparse",
        "json",
        "pathlib",
        "lunar_exploration_ppo.workflows.stage6",
    }
    lowered = source.lower()
    assert "sys.path" not in lowered
    assert "model_explorer" not in lowered
    assert "scripts.xunce" not in lowered
    assert "fixture" not in lowered
    assert "fake" not in lowered
    assert "--force" not in lowered
    assert "--skip" not in lowered
    assert "--output-root" not in lowered
    assert "_run_stage6_workflow_for_test" not in source
    assert "run_stage6_workflow" in source

    parameters = inspect.signature(module.run_stage6_workflow).parameters
    assert tuple(parameters) == (
        "config_path",
        "run_id",
        "stage5_gate_path",
        "review_authorization_path",
        "planning_warm_start_path",
        "planning_child_source_repair_path",
        "planning_child_source_repair_continuation_path",
    )
    assert tuple(
        inspect.signature(module.run_stage6_machine_preflight).parameters
    ) == ("config_path", "run_id", "stage5_gate_path")
    assert tuple(
        inspect.signature(
            module.verify_stage6_machine_preflight_for_resume
        ).parameters
    ) == ("config_path", "run_id", "stage5_gate_path")
    assert set(module.STAGE6_SOURCE_PATHS) == {
        *module.STAGE6_PRODUCTION_SOURCE_PATHS,
        *module.STAGE6_TEST_SOURCE_PATHS,
    }
    assert {
        "src/lunar_exploration_ppo/eval/standard.py",
        "src/lunar_exploration_ppo/ppo/checkpoint.py",
        "src/lunar_exploration_ppo/ppo/collector.py",
        "src/lunar_exploration_ppo/utils/artifact_io.py",
        "src/lunar_exploration_ppo/workflows/stage6_planning_child_source_repair.py",
        "scripts/create_ppo_stage6_planning_child_source_repair.py",
        "tests/ppo_highres_frontier/test_stage4_checkpoint.py",
        "tests/ppo_highres_frontier/test_stage4_collector.py",
        "tests/ppo_highres_frontier/test_stage6_planning_child_source_repair.py",
        "tests/ppo_highres_frontier/test_stage6_planning_child_source_repair_cli.py",
        "tests/ppo_highres_frontier/test_stage6_execution.py",
        "tests/ppo_highres_frontier/test_stage6_machine_preflight.py",
        "tests/ppo_highres_frontier/test_stage6_resource_lifecycle.py",
        "tests/ppo_highres_frontier/test_stage6_review_authorization.py",
        "tests/ppo_highres_frontier/test_stage6_standard_eval.py",
        "tests/ppo_highres_frontier/test_stage6_terminal_recovery.py",
    }.issubset(module.STAGE6_SOURCE_PATHS)
    for relative in (
        path for path in module.STAGE6_PRODUCTION_SOURCE_PATHS if path.endswith(".py")
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "sys.path" not in text
        assert "model_explorer" not in text
        assert "scripts.xunce" not in text


FORMAL_STAGE6_RUN_ID = "s6-standard-single-r1-20260717T010203Z"
STANDALONE_PREFLIGHT_RUN_ID = (
    "s6-standard-single-preflight-r1-20260717T010203Z"
)
HISTORICAL_STAGE6_RUN_IDS = (
    "s6-standard-r1-20260716T035116Z",
    "s6-standard-r2-20260716T133254Z",
    "s6-standard-single-r1-20260718T062833Z",
)


def test_failed_formal_r1_is_a_historical_immutable_run_id() -> None:
    module = _module()
    failed_run_id = "s6-standard-single-r1-20260718T062833Z"

    assert failed_run_id in HISTORICAL_STAGE6_RUN_IDS
    with pytest.raises(module.Stage6WorkflowError, match="historical.*immutable"):
        module.validate_stage6_formal_run_id(failed_run_id)


def _write_minimal_handle_capture_fixture(stage: Path) -> None:
    """Supply mutable-prefix files when detector/result behavior is stubbed."""

    stage.mkdir(parents=True, exist_ok=True)
    for relative in ("resource_audit.jsonl", "phase-state.jsonl"):
        path = stage / relative
        if not path.exists():
            path.write_bytes(b"{}\n")


class _FakeReviewAuthorizationHandle:
    def __init__(
        self,
        *,
        events: list[str] | None = None,
        fail_label: str | None = None,
        record: dict[str, object] | None = None,
        evidence_records: tuple[object, ...] = (),
    ) -> None:
        self.events = events if events is not None else []
        self.fail_label = fail_label
        self._record = record or {
            "schema_version": "stage6_verified_review_launch_authorization/v1",
            "formal_run_id": FORMAL_STAGE6_RUN_ID,
        }
        self.evidence_records = evidence_records
        self.evidence_paths = tuple(record.path for record in evidence_records)

    @property
    def record(self) -> dict[str, object]:
        return json.loads(json.dumps(self._record, sort_keys=True))

    def canonical_record(self) -> dict[str, object]:
        return self.record

    def require_current(self, label: str) -> None:
        self.events.append(f"review:{label}")
        if label == self.fail_label:
            from lunar_exploration_ppo.workflows.stage6_review_authorization import (
                Stage6ReviewAuthorizationError,
            )

            raise Stage6ReviewAuthorizationError(f"injected drift at {label}")


class _FakeInputPin:
    def __init__(
        self,
        *,
        events: list[str] | None = None,
        fail_label: str | None = None,
        _issuer: object | None = None,
        resources: list[object] | None = None,
        guards: list[object] | None = None,
    ) -> None:
        del _issuer
        self.events = events if events is not None else []
        self.fail_label = fail_label
        self._resources = list(resources or [])
        self._guards = list(guards or [])
        self.records = tuple(
            SimpleNamespace(
                labels=tuple(resource.labels),
                path=resource.path,
                sha256=resource.sha256,
                size_bytes=resource.size_bytes,
                identity=resource.identity,
            )
            for resource in self._resources
        )
        self.paths = tuple(record.path for record in self.records)
        self.closed = False

    def bind_requests(
        self,
        requests: tuple[tuple[str, Path], ...],
        *,
        bindings: dict[str, tuple[str, int]],
    ) -> None:
        grouped: list[tuple[list[str], Path]] = []
        indexes: dict[str, int] = {}
        for label, raw_path in requests:
            path = Path(os.path.abspath(raw_path))
            key = os.path.normcase(os.fspath(path))
            index = indexes.get(key)
            if index is None:
                indexes[key] = len(grouped)
                grouped.append(([label], path))
            else:
                grouped[index][0].append(label)
        records: list[object] = []
        for index, (labels, path) in enumerate(grouped):
            identities = {bindings[label] for label in labels}
            if len(identities) != 1:
                raise AssertionError(f"conflicting fake pin bindings for {labels}")
            digest, size_bytes = identities.pop()
            records.append(
                SimpleNamespace(
                    labels=tuple(labels),
                    path=path,
                    sha256=digest,
                    size_bytes=size_bytes,
                    identity=(1, index + 1, 1, size_bytes),
                )
            )
        self.records = tuple(records)
        self.paths = tuple(record.path for record in self.records)

    def __enter__(self):
        self.events.append("pin:enter")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()
        self.events.append("pin:exit")

    def require_current(self, label: str, *, rehash: bool = False) -> None:
        if self.closed:
            from lunar_exploration_ppo.utils.stage6_input_pinning import (
                Stage6InputPinError,
            )

            raise Stage6InputPinError("injected closed input pin")
        suffix = ":rehash" if rehash else ""
        self.events.append(f"pin:{label}{suffix}")
        if label == self.fail_label:
            from lunar_exploration_ppo.utils.stage6_input_pinning import (
                Stage6InputPinError,
            )

            raise Stage6InputPinError(f"injected input drift at {label}")

    def require_active_issuance(self, label: str = "input pin") -> None:
        if self.closed:
            raise RuntimeError(f"{label} closed")

    def read_bytes(self, path: str | Path, *, label: str) -> bytes:
        self.require_current(label, rehash=True)
        target = Path(os.path.abspath(path))
        matches = [
            resource
            for resource in self._resources
            if Path(resource.path) == target
        ]
        if len(matches) != 1:
            raise RuntimeError(f"{label} fake pinned path drifted")
        resource = matches[0]
        original_offset = os.lseek(resource.descriptor, 0, os.SEEK_CUR)
        os.lseek(resource.descriptor, 0, os.SEEK_SET)
        try:
            chunks: list[bytes] = []
            while True:
                chunk = os.read(resource.descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.lseek(resource.descriptor, original_offset, os.SEEK_SET)
        payload = b"".join(chunks)
        if (
            hashlib.sha256(payload).hexdigest() != resource.sha256
            or len(payload) != resource.size_bytes
        ):
            raise RuntimeError(f"{label} fake pinned bytes changed")
        return payload

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._resources or self._guards:
            from lunar_exploration_ppo.utils import stage6_input_pinning

            stage6_input_pinning._close_resources(
                self._resources,
                self._guards,
            )
            self._resources = []
            self._guards = []


class _FakeStage5Authority:
    def __init__(
        self,
        identity: dict[str, object] | None = None,
        *,
        snapshots: tuple[tuple[str, object], ...] = (),
        fail_label: str | None = None,
    ) -> None:
        self.identity = identity or {"gate_sha256": "a" * 64}
        self.snapshots = snapshots
        self.evidence_paths = tuple(snapshot.path for _name, snapshot in snapshots)
        self.fail_label = fail_label
        self.require_current_calls: list[str] = []

    def require_current(self, label: str = "authority") -> None:
        self.require_current_calls.append(label)
        if label == self.fail_label:
            raise RuntimeError(f"injected Stage 5 drift at {label}")


def _allow_review_authorization(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[dict[str, object]] | None = None,
    *,
    tmp_path: Path,
    handle: _FakeReviewAuthorizationHandle | None = None,
    input_pin: _FakeInputPin | None = None,
) -> _FakeReviewAuthorizationHandle:
    from lunar_exploration_ppo.workflows import stage6_review_authorization

    authority = _build_execution_authority(
        tmp_path / "workflow-authority",
        monkeypatch,
    )
    baseline_handle = authority["review_handle"]
    assert isinstance(baseline_handle, _FakeReviewAuthorizationHandle)
    selected = handle or baseline_handle
    if handle is not None:
        selected._record = copy.deepcopy(baseline_handle._record)
        selected.evidence_records = baseline_handle.evidence_records
        selected.evidence_paths = baseline_handle.evidence_paths
    baseline_pin = authority["pin"]
    assert isinstance(baseline_pin, _FakeInputPin)
    selected_pin = input_pin or baseline_pin
    if input_pin is not None:
        selected_pin.records = baseline_pin.records
        selected_pin.paths = baseline_pin.paths
        selected_pin.closed = False
    stage5_handle = authority["stage5_handle"]
    execution_identity = authority["identity"]

    def verify(**kwargs: object) -> _FakeReviewAuthorizationHandle:
        if calls is not None:
            calls.append(dict(kwargs))
        selected._record["formal_run_id"] = kwargs["expected_run_id"]
        return selected

    monkeypatch.setattr(
        stage6_review_authorization,
        "verify_stage6_review_launch_authorization",
        verify,
    )
    module = _module()
    monkeypatch.setattr(
        module,
        "verify_frozen_stage5_authority",
        lambda **kwargs: stage5_handle,
    )
    def acquire_pin(**kwargs: object) -> _FakeInputPin:
        del kwargs
        if input_pin is not None:
            return selected_pin
        fresh = _FakeInputPin()
        fresh.records = baseline_pin.records
        fresh.paths = baseline_pin.paths
        return fresh

    monkeypatch.setattr(
        module,
        "_acquire_stage6_workflow_input_pin",
        acquire_pin,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_validate_stage6_pinned_execution_identity",
        lambda **kwargs: (
            kwargs["input_pin"].require_current(
                "Stage 6 input pin acquisition complete"
            )
            or execution_identity
        ),
        raising=False,
    )
    selected._stage5_authority = stage5_handle
    selected._execution_identity = execution_identity
    selected._input_pin = selected_pin
    return selected


def test_public_workflow_pins_all_inputs_before_config_and_output_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    from lunar_exploration_ppo.configs import stage6 as stage6_config
    from lunar_exploration_ppo.workflows import stage6_review_authorization

    events: list[str] = []
    handle = _FakeReviewAuthorizationHandle(events=events)
    pin = _FakeInputPin(events=events)

    def verify(**kwargs):
        events.append("review:initial-verify")
        handle._record["formal_run_id"] = kwargs["expected_run_id"]
        return handle

    class Config:
        output_root = "D:/xunce/out/ppo_frontier"

    monkeypatch.setattr(
        stage6_review_authorization,
        "verify_stage6_review_launch_authorization",
        verify,
    )
    monkeypatch.setattr(
        module,
        "verify_frozen_stage5_authority",
        lambda **kwargs: events.append("authority:verify") or _FakeStage5Authority(),
    )
    monkeypatch.setattr(
        module,
        "_acquire_stage6_workflow_input_pin",
        lambda **kwargs: events.append("pin:acquire") or pin,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_validate_stage6_pinned_execution_identity",
        lambda **kwargs: events.append("identity:validate"),
        raising=False,
    )
    monkeypatch.setattr(
        stage6_config,
        "load_stage6_config",
        lambda path: events.append("config:load") or Config(),
    )
    expected = module.Stage6WorkflowResult(
        run_id=FORMAL_STAGE6_RUN_ID,
        stage_root=tmp_path / "unused",
        summary={},
        routing={},
    )

    def at_root(**kwargs):
        events.append("workflow:at-root")
        assert kwargs["input_pin"] is pin
        assert isinstance(kwargs["stage5_authority"], _FakeStage5Authority)
        return expected

    monkeypatch.setattr(module, "_run_stage6_workflow_at_root", at_root)

    result = module.run_stage6_workflow(
        config_path=CONFIG,
        run_id=FORMAL_STAGE6_RUN_ID,
        stage5_gate_path=tmp_path / "gate.json",
        review_authorization_path=tmp_path / "launch-authorization.json",
    )

    assert result == expected
    assert events == [
        "review:initial-verify",
        "review:Stage 6 review authorization before input pin acquisition",
        "authority:verify",
        "pin:acquire",
        "pin:enter",
        "identity:validate",
        "config:load",
        "workflow:at-root",
        "pin:Stage 6 input pin before final workflow return:rehash",
        "pin:exit",
    ]


def test_planning_child_workflow_issues_one_recovery_capability_per_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    from lunar_exploration_ppo.workflows import (
        stage6_planning_child_recovery,
    )

    run_root = (
        tmp_path / "s6-standard-single-r1-20260724T000124Z"
    )
    stage = run_root / "s6"
    stage.mkdir(parents=True)
    parent = stage / "planning-child-source-repair.json"
    continuation = (
        stage / "planning-child-source-repair-continuation.json"
    )
    capability = object()
    calls: list[dict[str, object]] = []
    review = {"formal_run_id": run_root.name}
    immutable = {"formal_run_id": run_root.name}
    handle = SimpleNamespace(canonical_record=lambda: review)

    monkeypatch.setattr(
        module,
        "validate_stage6_verified_review_authorization",
        lambda *_args, **_kwargs: review,
    )
    monkeypatch.setattr(
        module,
        "_stage6_current_immutable_bindings",
        lambda **_kwargs: immutable,
    )
    monkeypatch.setattr(
        stage6_planning_child_recovery,
        "issue_planning_child_recovery_capability",
        lambda **kwargs: calls.append(dict(kwargs)) or capability,
    )
    result = module._load_planning_child_recovery_capability_for_workflow(
        path=parent,
        continuation_path=continuation,
        run_id=run_root.name,
        execution_identity={"identity": True},
        review_authorization_handle=handle,
        stage5_authority=SimpleNamespace(),
    )

    assert result is capability
    assert len(calls) == 1
    assert calls[0]["parent_artifact_path"] == parent
    assert calls[0]["continuation_artifact_path"] == continuation


def test_planning_child_workflow_passes_only_recovery_capability() -> None:
    module = _module()

    assert hasattr(
        module,
        "_load_planning_child_recovery_capability_for_workflow",
    )
    assert not hasattr(
        module,
        "_load_planning_child_source_repair_for_workflow",
    )
    for function in (
        module.run_stage6_workflow,
        module._run_stage6_workflow_for_test,
        module._run_stage6_workflow_at_root,
        module._run_stage6_workflow_locked,
    ):
        source = inspect.getsource(function)
        assert "planning_child_source_repair_context" not in source
        assert "planning_child_source_repair_sha256" not in source


def test_planning_child_machine_verifier_uses_capability_acceptance_binding(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery
    from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
        PlanningChildLineageEpoch,
        PlanningChildRecoveryCapability,
        PlanningChildResumeCursor,
    )

    acceptance = {"schema_version": "task1-machine-acceptance"}
    capability = PlanningChildRecoveryCapability(
        formal_run_id="s6-standard-single-r1-20260724T000124Z",
        seed=20260716,
        stage_root=tmp_path / "s6",
        parent_artifact_sha256="a" * 64,
        continuation_artifact_sha256="b" * 64,
        input_snapshot_sha256="c" * 64,
        capability_sha256="d" * 64,
        resume_cursor=PlanningChildResumeCursor(
            last_accepted_update=85,
            next_update=86,
            next_attempt=1,
            next_transaction_key="0011:20260716:update:086",
            next_resource_segment_index=5,
            latest_complete_checkpoint_update=85,
            pending_pre_attempt=None,
        ),
        lineage_epochs=(
            PlanningChildLineageEpoch(75, 84, "1" * 64, "2" * 64),
            PlanningChildLineageEpoch(85, 85, "a" * 64, "3" * 64),
            PlanningChildLineageEpoch(86, None, "b" * 64, "4" * 64),
        ),
        protected_checkpoint_updates=(84, 85),
        input_pin_requests=(),
        acceptance_binding=acceptance,
    )
    semantic = stage6_terminal_recovery._bind_planning_child_semantic_result(
        {"passed": True},
        capability=capability,
    )

    assert semantic["planning_child_recovery"] == {
        "schema_version": (
            "stage6_planning_child_recovery_semantic_binding/v1"
        ),
        "capability_sha256": capability.capability_sha256,
        "input_snapshot_sha256": capability.input_snapshot_sha256,
        "parent_artifact_sha256": capability.parent_artifact_sha256,
        "continuation_artifact_sha256": (
            capability.continuation_artifact_sha256
        ),
        "acceptance_binding": acceptance,
    }
    assert (
        stage6_terminal_recovery._semantic_planning_child_capability(
            semantic
        )
        is capability
    )


@pytest.mark.parametrize("workflow_kind", ["public", "test"])
def test_planning_child_workflow_uses_origin_warm_start_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workflow_kind: str,
) -> None:
    module = _module()
    from lunar_exploration_ppo.configs import stage6 as stage6_config
    from lunar_exploration_ppo.workflows import (
        stage6_planning_warm_start,
        stage6_review_authorization,
    )

    events: list[str] = []
    handle = _FakeReviewAuthorizationHandle(events=events)
    pin = _FakeInputPin(events=events)
    origin_review = {"formal_run_id": FORMAL_STAGE6_RUN_ID}
    child_context = SimpleNamespace(
        parent_artifact_sha256="c" * 64,
        acceptance_binding={
            "origin_verified_review_authorization": origin_review,
        },
    )
    warm_context = SimpleNamespace()
    identity = {"schema_version": "stage6_execution_identity/v1"}

    def verify(**kwargs):
        events.append("review:initial-verify")
        handle._record["formal_run_id"] = kwargs["expected_run_id"]
        return handle

    class Config:
        output_root = "D:/xunce/out/ppo_frontier"

    monkeypatch.setattr(
        stage6_planning_warm_start,
        "planning_effective_config_bytes",
        lambda payload: b"effective-config",
    )
    monkeypatch.setattr(
        stage6_review_authorization,
        "verify_stage6_review_launch_authorization",
        verify,
    )
    monkeypatch.setattr(
        module,
        "verify_frozen_stage5_authority",
        lambda **kwargs: events.append("authority:verify")
        or _FakeStage5Authority(),
    )
    monkeypatch.setattr(
        module,
        "_acquire_stage6_workflow_input_pin",
        lambda **kwargs: events.append("pin:acquire") or pin,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_validate_stage6_pinned_execution_identity",
        lambda **kwargs: events.append("identity:validate") or identity,
        raising=False,
    )
    monkeypatch.setattr(
        stage6_config,
        "load_stage6_config",
        lambda path: events.append("config:load") or Config(),
    )
    monkeypatch.setattr(
        module,
        "_load_planning_child_recovery_capability_for_workflow",
        lambda **kwargs: events.append("child:load") or child_context,
    )

    def reject_current_warm_bindings(**kwargs):
        raise stage6_planning_warm_start.Stage6PlanningWarmStartError(
            "warm-start evidence binding drifted"
        )

    monkeypatch.setattr(
        stage6_planning_warm_start,
        "planning_warm_start_expected_bindings",
        reject_current_warm_bindings,
    )

    def origin_warm_bindings(**kwargs):
        assert kwargs["review_authorization_record"] is origin_review
        events.append("warm:origin-bindings")
        return {"origin": "bound"}

    monkeypatch.setattr(
        stage6_planning_warm_start,
        "planning_warm_start_expected_bindings_from_record",
        origin_warm_bindings,
    )

    def load_warm(path, **kwargs):
        assert kwargs["expected_bindings"] == {"origin": "bound"}
        events.append("warm:load")
        return warm_context, "w" * 64

    monkeypatch.setattr(
        stage6_planning_warm_start,
        "load_planning_warm_start_artifact",
        load_warm,
    )
    monkeypatch.setattr(
        stage6_planning_warm_start,
        "verify_parent_u74_bundle",
        lambda *args, **kwargs: SimpleNamespace(resource_accepted=True),
    )
    expected = module.Stage6WorkflowResult(
        run_id=FORMAL_STAGE6_RUN_ID,
        stage_root=tmp_path / "unused",
        summary={},
        routing={},
    )

    def at_root(**kwargs):
        assert kwargs["planning_child_recovery_capability"] is child_context
        assert kwargs["planning_warm_start_context"] is warm_context
        events.append("workflow:at-root")
        return expected

    monkeypatch.setattr(module, "_run_stage6_workflow_at_root", at_root)

    workflow = (
        module.run_stage6_workflow
        if workflow_kind == "public"
        else module._run_stage6_workflow_for_test
    )
    kwargs = {
        "config_path": CONFIG,
        "run_id": FORMAL_STAGE6_RUN_ID,
        "stage5_gate_path": tmp_path / "gate.json",
        "review_authorization_path": tmp_path / "launch-authorization.json",
        "planning_warm_start_path": tmp_path / "planning-warm-start.json",
        "planning_child_source_repair_path": (
            tmp_path / "planning-child-source-repair.json"
        ),
        "planning_child_source_repair_continuation_path": (
            tmp_path
            / "planning-child-source-repair-continuation.json"
        ),
    }
    if workflow_kind == "test":
        kwargs["base_output_root"] = tmp_path / "output"
    result = workflow(**kwargs)

    assert result is expected
    assert events.index("child:load") < events.index("warm:origin-bindings")
    assert events.index("warm:origin-bindings") < events.index("warm:load")
    assert "workflow:at-root" in events


def test_pinned_identity_mismatch_fails_before_config_backend_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    module = _module()
    from lunar_exploration_ppo.configs import stage6 as stage6_config

    _allow_review_authorization(monkeypatch, tmp_path=tmp_path)
    pin = _FakeInputPin()
    monkeypatch.setattr(
        module,
        "_acquire_stage6_workflow_input_pin",
        lambda **kwargs: pin,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_validate_stage6_pinned_execution_identity",
        lambda **kwargs: (_ for _ in ()).throw(
            module.Stage6WorkflowError("pinned execution identity mismatch")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        stage6_config,
        "load_stage6_config",
        lambda path: pytest.fail("identity mismatch loaded config"),
    )
    monkeypatch.setattr(
        module,
        "_run_stage6_workflow_at_root",
        lambda **kwargs: pytest.fail("identity mismatch created workflow output"),
    )
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "lunar_exploration_ppo.ppo.standard_training":
            raise AssertionError("identity mismatch imported training backend")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    run_id = FORMAL_STAGE6_RUN_ID
    run_root = Path("D:/xunce/out/ppo_frontier") / run_id
    run_root_existed = run_root.exists()

    with pytest.raises(module.Stage6WorkflowError, match="pinned execution identity"):
        module.run_stage6_workflow(
            config_path=CONFIG,
            run_id=run_id,
            stage5_gate_path=tmp_path / "gate.json",
            review_authorization_path=tmp_path / "launch-authorization.json",
        )

    assert run_root.exists() is run_root_existed


def _install_stage6_new_run_fakes(
    module,
    monkeypatch: pytest.MonkeyPatch,
    *,
    events: list[str],
) -> None:
    from lunar_exploration_ppo.ppo import standard_training
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    monkeypatch.setattr(
        stage6_terminal_recovery,
        "detect_stage6_terminal_recovery",
        lambda **kwargs: {"status": "no_terminal", "reason": ""},
    )

    def preflight(**kwargs):
        events.append("preflight")
        stage = (
            Path(kwargs["base_output_root"])
            / kwargs["run_id"]
            / "s6/preflight"
        )
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "audit.json").write_bytes(b"{}\n")
        return {"passed": True}

    monkeypatch.setattr(module, "_run_stage6_machine_preflight", preflight)

    def execute(**kwargs):
        events.append("execute")
        assert kwargs["execution_capability"] is not None
        stage = Path(kwargs["run_root"]) / "s6"
        return {
            "stage_root": str(stage),
            "summary": {"state": "training"},
            "routing": {"route": "resume"},
        }

    monkeypatch.setattr(standard_training, "execute_standard_training", execute)


def _review_identity_fixture(module) -> tuple[dict[str, object], dict[str, object]]:
    tree = "1" * 40
    identity = {
        "schema_version": "stage6_execution_identity/v1",
        "base_commit": module.STAGE5_COMMIT,
        "head_commit": module.STAGE5_COMMIT,
        "prospective_git_tree": tree,
        "prospective_tree_sha256": hashlib.sha256(tree.encode("ascii")).hexdigest(),
        "changed_paths": sorted(module.STAGE6_SOURCE_PATHS),
        "changed_path_set_sha256": "2" * 64,
        "real_index_empty": True,
        "config_sha256": "3" * 64,
        "source_set_sha256": "4" * 64,
        "data_sha256": "5" * 64,
        "catalog_sha256": "6" * 64,
        "environment_identity": {"schema_version": "test-environment/v1"},
        "environment_sha256": "7" * 64,
        "source_identity": {"schema_version": "test-source/v1"},
        "data_identity": {"schema_version": "test-data/v1"},
    }
    verified = {
        "schema_version": "stage6_verified_review_launch_authorization/v1",
        "stage_id": module.STAGE6_STAGE_ID,
        "formal_run_id": FORMAL_STAGE6_RUN_ID,
        "single_seed_scope": {
            "scope_kind": "single_seed_system_closure/v1",
            "seed": 20260716,
            "updates": 100,
        },
        "authorized": True,
        "base_commit": module.STAGE5_COMMIT,
        "authorization_file_sha256": "8" * 64,
        "authorization_file_size_bytes": 100,
        "review_identity_sha256": hashlib.sha256(
            module.ArtifactStore.canonical_json_bytes(identity)
        ).hexdigest(),
        "reviewed_prospective_git_tree": tree,
        "prospective_tree_sha256": identity["prospective_tree_sha256"],
        "changed_path_set_sha256": identity["changed_path_set_sha256"],
        "source_set_sha256": identity["source_set_sha256"],
        "config_sha256": identity["config_sha256"],
        "data_sha256": identity["data_sha256"],
        "environment_sha256": identity["environment_sha256"],
        "frozen_diff_sha256": "a" * 64,
        "frozen_diff_size_bytes": 200,
        "spec_review_sha256": "b" * 64,
        "quality_review_sha256": "c" * 64,
    }
    return identity, verified


def test_formal_runner_cli_reports_fixed_ordinal6_cache_without_override(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location("stage6_ordinal6_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    calls: list[dict[str, object]] = []

    def fake_run(**kwargs: object) -> object:
        calls.append(kwargs)
        return SimpleNamespace(
            run_id=kwargs["run_id"],
            stage_root=tmp_path / "s6",
            summary={"source_repair_summary_schema": "stage6_source_repair_summary/v6"},
            routing={"route": "dry-test"},
        )

    monkeypatch.setattr(runner, "run_stage6_workflow", fake_run)
    assert runner.main(
        [
            "--run-id",
            FORMAL_STAGE6_RUN_ID,
            "--review-authorization",
            str(tmp_path / "authorization.json"),
            "--planning-warm-start",
            str(tmp_path / "planning-warm-start.json"),
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["coverage_cache"] == {
        "runtime_mode": "persistent_exact_manifest_read_only/v1",
        "manifest_path": (
            "D:/xunce/review/s6-coverable-cache-r1-20260722T144932/"
            "coverage-cache-manifest.json"
        ),
        "manifest_sha256": (
            "675b923b64cee4cde90d5728550a7f939c41cfbc56e9c237268078b59641081c"
        ),
        "manifest_size_bytes": 519284,
        "cache_root": (
            "D:/xunce/cache/ppo_frontier/"
            "s6-standard-single-r1-20260718T220434Z/coverage-v1"
        ),
        "entry_set_sha256": (
            "c543a277154b41d14bd1a194d6c6e4f2b43075d5b93096c50d1952ee44c43558"
        ),
        "formal_audit_path": (
            "D:/xunce/review/s6-coverable-cache-r1-20260722T144932/"
            "formal-audit.json"
        ),
        "formal_audit_sha256": (
            "d7c12737da4c9a43353e02347b9c9d6abd527a78f7a3c6675e9164b6ae5b0068"
        ),
        "formal_audit_size_bytes": 4416,
    }
    assert len(calls) == 1
    option_strings = {
        option
        for action in runner.build_parser()._actions
        for option in action.option_strings
    }
    assert "--coverage-cache-manifest" not in option_strings
    assert "--coverage-cache-root" not in option_strings


def _stage5_identity_for_capability(config) -> dict[str, object]:
    authority = config.stage5_authority
    return {
        "schema_version": "stage6_stage5_authority_audit/v1",
        "verified": True,
        "authorized_stage": authority.authorized_stage,
        "run_id": "s5-task6-fix-r1-20260714T223702Z",
        "commit_sha256": authority.commit_sha256,
        "commit_tree": authority.commit_tree,
        "gate_path": authority.gate_path,
        "gate_sha256": authority.gate_sha256,
        "review_sha256": authority.review_sha256,
        "manifest_sha256": authority.manifest_sha256,
        "checkpoint_sha256": authority.checkpoint_sha256,
        "policy_state_sha256": authority.policy_state_sha256,
        "performance_advantage_established": False,
        "history_states": [
            "machine_passed",
            "awaiting_independent_review",
            "awaiting_human_approval",
            "approved",
            "next_stage",
        ],
    }


def _build_execution_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    repo_root: Path = ROOT,
    config_path: Path = CONFIG,
    run_root: Path | None = None,
    stage5_overrides: dict[str, object] | None = None,
    pinned_config_sha256: str | None = None,
    pin_overrides: dict[str, tuple[str, int]] | None = None,
    request_drift: str | None = None,
    pin_kind: str = "valid",
    environment_identity: dict[str, object] | None = None,
    effective_config_bytes: bytes | None = None,
    planning_warm_start_path: Path | None = None,
    planning_child_source_repair_path: Path | None = None,
    execution_identity_override: Mapping[str, object] | None = None,
    verified_review_override: Mapping[str, object] | None = None,
    review_authorization_path: Path | None = None,
    formal_run_id: str = FORMAL_STAGE6_RUN_ID,
):
    module = _module()
    from lunar_exploration_ppo.configs.stage6 import load_stage6_config
    from lunar_exploration_ppo.utils import stage6_input_pinning
    from lunar_exploration_ppo.workflows import stage6_review_authorization

    repo = repo_root.resolve()
    config_file = config_path.resolve()
    config = load_stage6_config(config_file)
    config_payload = module.ArtifactStore.canonical_json_bytes(
        config.model_dump(mode="json")
    )
    base_config_digest = hashlib.sha256(config_payload).hexdigest()
    config_digest = hashlib.sha256(
        effective_config_bytes
        if effective_config_bytes is not None
        else config_payload
    ).hexdigest()
    identity, verified = _review_identity_fixture(module)
    verified["formal_run_id"] = formal_run_id
    identity["config_sha256"] = config_digest
    if environment_identity is not None:
        identity["environment_identity"] = json.loads(
            json.dumps(environment_identity, sort_keys=True)
        )
        identity["environment_sha256"] = module.stage6_environment_sha256(
            identity["environment_identity"]
        )
    source_rows: list[dict[str, object]] = []
    pin_bindings: dict[str, tuple[str, int]] = {
        "config:canonical": (base_config_digest, len(config_payload)),
        "coverage-cache:manifest": (
            module.STAGE6_COVERAGE_CACHE_MANIFEST_SHA256,
            module.STAGE6_COVERAGE_CACHE_MANIFEST_SIZE_BYTES,
        ),
    }
    for relative in module.STAGE6_SOURCE_PATHS:
        label = f"source:{relative}"
        if (repo / relative).resolve() == config_file:
            digest, size_bytes = base_config_digest, len(config_payload)
        else:
            payload = label.encode("utf-8")
            digest, size_bytes = hashlib.sha256(payload).hexdigest(), len(payload)
        source_rows.append(
            {"path": relative, "sha256": digest, "size_bytes": size_bytes}
        )
        pin_bindings[label] = (digest, size_bytes)
    source_identity = {
        "schema_version": "stage6_current_source_set/v1",
        "source_set_sha256": "3" * 64,
        "paths": source_rows,
    }
    identity["source_identity"] = source_identity
    identity["source_set_sha256"] = source_identity["source_set_sha256"]

    data_rows: list[dict[str, object]] = []
    for label, raw_path in module.STAGE6_DATA_INPUT_PATHS:
        pin_label = f"data:{label}"
        payload = pin_label.encode("utf-8")
        digest, size_bytes = hashlib.sha256(payload).hexdigest(), len(payload)
        path = Path(os.path.abspath(raw_path))
        data_rows.append(
            {
                "label": label,
                "path": path.as_posix(),
                "sha256": digest,
                "size_bytes": size_bytes,
            }
        )
        pin_bindings[pin_label] = (digest, size_bytes)
    data_identity = {
        "schema_version": "stage6_data_catalog_identity/v1",
        "files": data_rows,
        "catalog_sha256": "6" * 64,
    }
    identity["data_identity"] = data_identity
    identity["data_sha256"] = module._canonical_sha256(data_identity)
    identity["catalog_sha256"] = data_identity["catalog_sha256"]
    if execution_identity_override is not None:
        identity = copy.deepcopy(dict(execution_identity_override))
        for row in identity["source_identity"]["paths"]:
            pin_bindings[f"source:{row['path']}"] = (
                str(row["sha256"]),
                int(row["size_bytes"]),
            )
        for row in identity["data_identity"]["files"]:
            pin_bindings[f"data:{row['label']}"] = (
                str(row["sha256"]),
                int(row["size_bytes"]),
            )

    stage5_identity = _stage5_identity_for_capability(config)
    if stage5_overrides:
        stage5_identity.update(stage5_overrides)
    stage5_snapshots: list[tuple[str, object]] = []
    for name in (
        "gate",
        "approval",
        "review",
        "manifest",
        "checkpoint",
        "checkpoint_manifest",
    ):
        label = f"stage5-authority:{name}"
        path = (
            Path(os.path.abspath(str(stage5_identity["gate_path"])))
            if name == "gate"
            else Path(os.path.abspath(tmp_path / "stage5-evidence" / f"{name}.json"))
        )
        identity_field = {
            "gate": "gate_sha256",
            "review": "review_sha256",
            "manifest": "manifest_sha256",
            "checkpoint": "checkpoint_sha256",
        }.get(name)
        if identity_field is None:
            payload = label.encode("utf-8")
            digest = hashlib.sha256(payload).hexdigest()
        else:
            digest = str(stage5_identity[identity_field])
        size_bytes = len(label.encode("utf-8"))
        snapshot = SimpleNamespace(
            path=path,
            sha256=digest,
            size_bytes=size_bytes,
        )
        stage5_snapshots.append((name, snapshot))
        pin_bindings[label] = (digest, size_bytes)
    stage5_handle = _FakeStage5Authority(
        stage5_identity,
        snapshots=tuple(stage5_snapshots),
    )

    if verified_review_override is not None:
        verified = copy.deepcopy(dict(verified_review_override))
    verified["config_sha256"] = identity["config_sha256"]
    verified["source_set_sha256"] = identity["source_set_sha256"]
    verified["data_sha256"] = identity["data_sha256"]
    verified["environment_sha256"] = identity["environment_sha256"]
    verified["review_identity_sha256"] = hashlib.sha256(
        module.ArtifactStore.canonical_json_bytes(identity)
    ).hexdigest()
    review_evidence: list[object] = []
    for index, name in enumerate(
        (
            "launch-authorization.json",
            "execution-identity.json",
            "prospective.diff",
            "spec-review.json",
            "quality-review.json",
        ),
        start=1,
    ):
        label = f"review-authorization:{index}:{name}"
        expected_digest = (
            verified["authorization_file_sha256"],
            verified["review_identity_sha256"],
            verified["frozen_diff_sha256"],
            verified["spec_review_sha256"],
            verified["quality_review_sha256"],
        )[index - 1]
        expected_size = (
            verified["authorization_file_size_bytes"],
            len(label.encode("utf-8")),
            verified["frozen_diff_size_bytes"],
            len(label.encode("utf-8")),
            len(label.encode("utf-8")),
        )[index - 1]
        record = SimpleNamespace(
            path=(
                Path(os.path.abspath(review_authorization_path))
                if index == 1 and review_authorization_path is not None
                else Path(
                    os.path.abspath(tmp_path / "review-evidence" / name)
                )
            ),
            sha256=expected_digest,
            size_bytes=expected_size,
        )
        review_evidence.append(record)
        pin_bindings[label] = (record.sha256, record.size_bytes)

    review_handle = _FakeReviewAuthorizationHandle(
        record=verified,
        evidence_records=tuple(review_evidence),
    )
    requests = module._stage6_input_pin_requests(
        repo_root=repo,
        config_path=config_file,
        stage5_authority=stage5_handle,
        review_authorization_handle=review_handle,
        planning_warm_start_path=planning_warm_start_path,
        planning_child_source_repair_path=(
            planning_child_source_repair_path
        ),
    )
    if planning_warm_start_path is not None:
        warm_payload = planning_warm_start_path.read_bytes()
        pin_bindings["planning-warm-start:artifact"] = (
            hashlib.sha256(warm_payload).hexdigest(),
            len(warm_payload),
        )
        for label, path in requests:
            if not label.startswith("planning-warm-start:parent:"):
                continue
            parent_payload = Path(path).read_bytes()
            pin_bindings[label] = (
                hashlib.sha256(parent_payload).hexdigest(),
                len(parent_payload),
            )
    if planning_child_source_repair_path is not None:
        for label, path in requests:
            if not label.startswith("planning-child-source-repair:"):
                continue
            child_input_payload = Path(path).read_bytes()
            pin_bindings[label] = (
                hashlib.sha256(child_input_payload).hexdigest(),
                len(child_input_payload),
            )
    if request_drift == "missing":
        requests = requests[:-1]
    elif request_drift == "extra":
        extra_label = "source:unexpected-extra.py"
        requests = (
            *requests,
            (extra_label, Path(os.path.abspath(tmp_path / "unexpected-extra.py"))),
        )
        payload = extra_label.encode("utf-8")
        pin_bindings[extra_label] = (
            hashlib.sha256(payload).hexdigest(),
            len(payload),
        )
    elif request_drift == "alias":
        requests = tuple(
            (
                label,
                Path(os.path.abspath(tmp_path / "aliased-dem.bin")),
            )
            if label == "data:dem"
            else (label, path)
            for label, path in requests
        )
    elif request_drift is not None:
        raise AssertionError(f"unknown request drift: {request_drift}")
    pin: object = _FakeInputPin()
    if pinned_config_sha256 is not None:
        for label, path in requests:
            if Path(path) == config_file:
                pin_bindings[label] = (pinned_config_sha256, len(config_payload))
    if pin_overrides:
        pin_bindings.update(pin_overrides)
    if pin_kind == "valid":
        pin.bind_requests(requests, bindings=pin_bindings)
    elif pin_kind == "empty":
        pass
    elif pin_kind == "forged":
        pin = object()
    else:
        raise AssertionError(f"unknown pin kind: {pin_kind}")
    monkeypatch.setattr(
        stage6_review_authorization,
        "Stage6ReviewAuthorizationHandle",
        _FakeReviewAuthorizationHandle,
    )
    monkeypatch.setattr(
        stage6_input_pinning,
        "Stage6InputPin",
        _FakeInputPin,
    )
    monkeypatch.setattr(
        module,
        "FrozenStage5AuthorityHandle",
        _FakeStage5Authority,
    )
    selected_run_root = (
        run_root.resolve()
        if run_root is not None
        else (tmp_path / formal_run_id).resolve()
    )
    return {
        "config": config,
        "identity": identity,
        "pin": pin,
        "requests": requests,
        "review_handle": review_handle,
        "run_root": selected_run_root,
        "repo_root": repo,
        "config_path": config_file,
        "stage5_handle": stage5_handle,
        "verified": verified,
    }


@contextmanager
def _active_execution_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    repo_root: Path = ROOT,
    config_path: Path = CONFIG,
    run_root: Path | None = None,
    stage5_overrides: dict[str, object] | None = None,
    pinned_config_sha256: str | None = None,
    pin_overrides: dict[str, tuple[str, int]] | None = None,
    request_drift: str | None = None,
    pin_kind: str = "valid",
    environment_identity: dict[str, object] | None = None,
    effective_config_bytes: bytes | None = None,
    planning_warm_start_path: Path | None = None,
    planning_child_source_repair_path: Path | None = None,
    execution_identity_override: Mapping[str, object] | None = None,
    verified_review_override: Mapping[str, object] | None = None,
    review_authorization_path: Path | None = None,
    formal_run_id: str = FORMAL_STAGE6_RUN_ID,
):
    module = _module()
    from lunar_exploration_ppo.utils.durable_jsonl import RunLease

    active = _build_execution_authority(
        tmp_path,
        monkeypatch,
        repo_root=repo_root,
        config_path=config_path,
        run_root=run_root,
        stage5_overrides=stage5_overrides,
        pinned_config_sha256=pinned_config_sha256,
        pin_overrides=pin_overrides,
        request_drift=request_drift,
        pin_kind=pin_kind,
        environment_identity=environment_identity,
        effective_config_bytes=effective_config_bytes,
        planning_warm_start_path=planning_warm_start_path,
        planning_child_source_repair_path=(
            planning_child_source_repair_path
        ),
        execution_identity_override=execution_identity_override,
        verified_review_override=verified_review_override,
        review_authorization_path=review_authorization_path,
        formal_run_id=formal_run_id,
    )
    lease = RunLease(Path(active["run_root"]) / ".stage6.lease")
    with lease:
        with module._stage6_execution_capability_scope(
            config=active["config"],
            config_path=active["config_path"],
            formal_run_id=formal_run_id,
            run_root=active["run_root"],
            stage_root=Path(active["run_root"]) / "s6",
            repo_root=active["repo_root"],
            execution_identity=active["identity"],
            expected_input_requests=active["requests"],
            review_authorization_handle=active["review_handle"],
            input_pin=active["pin"],
            stage5_authority=active["stage5_handle"],
            run_lease=lease,
            effective_config_bytes=effective_config_bytes,
            planning_warm_start_path=planning_warm_start_path,
            planning_child_source_repair_path=(
                planning_child_source_repair_path
            ),
        ) as capability:
            yield {**active, "capability": capability, "lease": lease}


@pytest.fixture
def active_execution_authority_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    with ExitStack() as stack:
        counter = 0

        def activate(
            *,
            run_root: Path | None = None,
            **authority_kwargs: object,
        ) -> dict[str, object]:
            nonlocal counter
            counter += 1
            authority_root = tmp_path / f"workflow-capability-{counter:02d}"
            selected_run_root = run_root or (
                authority_root / FORMAL_STAGE6_RUN_ID
            )
            return stack.enter_context(
                _active_execution_capability(
                    authority_root,
                    monkeypatch,
                    run_root=selected_run_root,
                    **authority_kwargs,
                )
            )

        yield activate


@pytest.mark.parametrize("authority_kind", ("missing", "forged", "wrong_lease"))
@pytest.mark.parametrize(
    "publication_entry",
    ("write_stage6_manifest", "write_or_verify_stage6_manifest"),
)
def test_manifest_publication_rejects_invalid_execution_capability_without_side_effects(
    tmp_path: Path,
    active_execution_authority_factory,
    authority_kind: str,
    publication_entry: str,
) -> None:
    module = _module()
    stage = tmp_path / FORMAL_STAGE6_RUN_ID / "s6"
    _write_manifest_fixture(module, stage)
    active = active_execution_authority_factory(run_root=stage.parent)
    capability = (
        active["capability"]
        if authority_kind == "wrong_lease"
        else None if authority_kind == "missing" else object()
    )
    run_lease = object() if authority_kind == "wrong_lease" else active["lease"]

    with pytest.raises(module.Stage6WorkflowError, match="capability|RunLease"):
        getattr(module, publication_entry)(
            stage_root=stage,
            repo_root=ROOT,
            execution_capability=capability,
            run_lease=run_lease,
        )

    assert not (stage / "manifest.json").exists()


@pytest.mark.parametrize(
    "publication_entry",
    ("write_stage6_manifest", "write_or_verify_stage6_manifest"),
)
def test_manifest_publication_rejects_revoked_capability_without_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication_entry: str,
) -> None:
    module = _module()
    stage = tmp_path / FORMAL_STAGE6_RUN_ID / "s6"
    _write_manifest_fixture(module, stage)
    with _active_execution_capability(
        tmp_path / "revoked-manifest-authority",
        monkeypatch,
        run_root=stage.parent,
    ) as active:
        revoked_capability = active["capability"]
        revoked_lease = active["lease"]

    with pytest.raises(module.Stage6WorkflowError, match="capability"):
        getattr(module, publication_entry)(
            stage_root=stage,
            repo_root=ROOT,
            execution_capability=revoked_capability,
            run_lease=revoked_lease,
        )

    assert not (stage / "manifest.json").exists()


@pytest.mark.parametrize(
    "publication_entry",
    ("write_stage6_manifest", "write_or_verify_stage6_manifest"),
)
def test_manifest_publication_accepts_exact_capability_and_same_run_lease(
    tmp_path: Path,
    active_execution_authority_factory,
    publication_entry: str,
) -> None:
    module = _module()
    stage = tmp_path / FORMAL_STAGE6_RUN_ID / "s6"
    active = active_execution_authority_factory(run_root=stage.parent)
    _write_manifest_fixture(
        module,
        stage,
        execution_identity=active["identity"],
        verified_review=active["verified"],
    )

    result = getattr(module, publication_entry)(
        stage_root=stage,
        repo_root=ROOT,
        execution_capability=active["capability"],
        run_lease=active["lease"],
    )

    path = stage / "manifest.json"
    assert path.is_file()
    if publication_entry == "write_stage6_manifest":
        assert result == path
    else:
        assert result == {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }


def _formal_preflight_kwargs(
    active: dict[str, object],
    *,
    execution_capability: object,
    run_lease: object | None = None,
) -> dict[str, object]:
    return {
        "config_path": active["config_path"],
        "run_id": FORMAL_STAGE6_RUN_ID,
        "stage5_gate_path": Path(active["run_root"]) / "stage5-gate.json",
        "base_output_root": Path(active["run_root"]).parent,
        "require_canonical_output": False,
        "config": active["config"],
        "execution_identity": active["identity"],
        "review_authorization_handle": active["review_handle"],
        "input_pin": active["pin"],
        "stage5_authority": active["stage5_handle"],
        "run_lease": active["lease"] if run_lease is None else run_lease,
        "execution_capability": execution_capability,
    }


@pytest.mark.parametrize("authority_kind", ("missing", "forged", "wrong_lease"))
def test_formal_preflight_rejects_invalid_capability_or_lease_before_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_execution_authority_factory,
    authority_kind: str,
) -> None:
    module = _module()
    active = active_execution_authority_factory(
        run_root=tmp_path / FORMAL_STAGE6_RUN_ID
    )
    marker = tmp_path / f"formal-preflight-{authority_kind}.side-effect"

    def forbidden_authority_verification(**kwargs: object) -> object:
        del kwargs
        marker.write_bytes(b"formal preflight body was entered")
        raise AssertionError("formal preflight guard was bypassed")

    monkeypatch.setattr(
        module,
        "verify_frozen_stage5_authority",
        forbidden_authority_verification,
    )
    capability = (
        active["capability"]
        if authority_kind == "wrong_lease"
        else None if authority_kind == "missing" else object()
    )
    lease = object() if authority_kind == "wrong_lease" else None

    with pytest.raises(module.Stage6WorkflowError, match="capability|RunLease"):
        module._run_stage6_machine_preflight(
            **_formal_preflight_kwargs(
                active,
                execution_capability=capability,
                run_lease=lease,
            )
        )

    assert not marker.exists()


def test_formal_preflight_rejects_revoked_capability_before_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    marker = tmp_path / "revoked-formal-preflight.side-effect"
    with _active_execution_capability(
        tmp_path / "revoked-preflight-authority",
        monkeypatch,
        run_root=tmp_path / FORMAL_STAGE6_RUN_ID,
    ) as active:
        revoked = dict(active)

    def forbidden_authority_verification(**kwargs: object) -> object:
        del kwargs
        marker.write_bytes(b"revoked capability entered preflight")
        raise AssertionError("revoked capability guard was bypassed")

    monkeypatch.setattr(
        module,
        "verify_frozen_stage5_authority",
        forbidden_authority_verification,
    )
    with pytest.raises(module.Stage6WorkflowError, match="capability"):
        module._run_stage6_machine_preflight(
            **_formal_preflight_kwargs(
                revoked,
                execution_capability=revoked["capability"],
            )
        )

    assert not marker.exists()


def test_formal_preflight_operation_blocks_capability_revoke_for_full_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    body_entered = threading.Event()
    release_body = threading.Event()
    preflight_done = threading.Event()
    revoke_done = threading.Event()
    errors: list[BaseException] = []

    class InjectedPreflightStop(RuntimeError):
        pass

    with _active_execution_capability(
        tmp_path / "blocking-preflight-authority",
        monkeypatch,
        run_root=tmp_path / FORMAL_STAGE6_RUN_ID,
    ) as active:
        capability = active["capability"]

        def blocked_authority_verification(**kwargs: object) -> object:
            del kwargs
            body_entered.set()
            if not release_body.wait(5.0):
                raise AssertionError("formal preflight release timed out")
            raise InjectedPreflightStop("stop after operation probe")

        monkeypatch.setattr(
            module,
            "verify_frozen_stage5_authority",
            blocked_authority_verification,
        )

        def run_preflight() -> None:
            try:
                module._run_stage6_machine_preflight(
                    **_formal_preflight_kwargs(
                        active,
                        execution_capability=capability,
                    )
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)
            finally:
                preflight_done.set()

        def revoke() -> None:
            try:
                module._revoke_stage6_execution_capability(capability)
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)
            finally:
                revoke_done.set()

        preflight_thread = threading.Thread(target=run_preflight)
        preflight_thread.start()
        assert body_entered.wait(5.0)
        revoke_thread = threading.Thread(target=revoke)
        revoke_thread.start()
        try:
            assert not revoke_done.wait(0.05)
        finally:
            release_body.set()
            assert preflight_done.wait(5.0)
            assert revoke_done.wait(5.0)
            preflight_thread.join(timeout=1.0)
            revoke_thread.join(timeout=1.0)

    assert len(errors) == 1
    assert isinstance(errors[0], InjectedPreflightStop)


def test_private_preflight_seam_only_allows_standalone_namespace_without_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    authority_calls: list[str] = []

    class InjectedStandaloneStop(RuntimeError):
        pass

    def stop_after_namespace_check(**kwargs: object) -> object:
        del kwargs
        authority_calls.append("verify")
        raise InjectedStandaloneStop("standalone namespace reached body")

    monkeypatch.setattr(
        module,
        "verify_frozen_stage5_authority",
        stop_after_namespace_check,
    )
    with pytest.raises(module.Stage6WorkflowError, match="preflight run id|formal"):
        module._run_stage6_machine_preflight_for_test(
            config_path=CONFIG,
            run_id=FORMAL_STAGE6_RUN_ID,
            stage5_gate_path=tmp_path / "gate.json",
            base_output_root=tmp_path,
        )
    assert authority_calls == []

    with pytest.raises(InjectedStandaloneStop, match="standalone namespace"):
        module._run_stage6_machine_preflight_for_test(
            config_path=CONFIG,
            run_id=STANDALONE_PREFLIGHT_RUN_ID,
            stage5_gate_path=tmp_path / "gate.json",
            base_output_root=tmp_path,
        )
    assert authority_calls == ["verify"]


def test_capability_issuance_rejects_stage5_identity_drift_from_frozen_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        _module().Stage6WorkflowError,
        match="Stage 5.*binding|Stage 5.*drift",
    ):
        with _active_execution_capability(
            tmp_path,
            monkeypatch,
            stage5_overrides={"gate_sha256": "f" * 64},
        ):
            pass


def test_capability_issuance_rejects_config_pin_digest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        _module().Stage6WorkflowError,
        match="input pin.*config|config.*pin|input pin.*binding",
    ):
        with _active_execution_capability(
            tmp_path,
            monkeypatch,
            pinned_config_sha256="f" * 64,
        ):
            pass


@pytest.mark.parametrize(
    "label",
    (
        "source:docs/ppo-highres-frontier-stage6.md",
        "data:dem",
        "stage5-authority:approval",
        "review-authorization:3:prospective.diff",
    ),
)
def test_capability_issuance_cross_binds_every_pinned_evidence_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
) -> None:
    with pytest.raises(
        _module().Stage6WorkflowError,
        match="input pin.*binding|pinned.*identity|pin.*drift",
    ):
        with _active_execution_capability(
            tmp_path,
            monkeypatch,
            pin_overrides={label: ("f" * 64, 999)},
        ):
            pass


@pytest.mark.parametrize("request_drift", ("missing", "extra", "alias"))
def test_capability_issuance_rejects_non_authoritative_pin_request_sets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request_drift: str,
) -> None:
    with pytest.raises(
        _module().Stage6WorkflowError,
        match="input pin.*request|input pin.*binding|exact path set",
    ):
        with _active_execution_capability(
            tmp_path,
            monkeypatch,
            request_drift=request_drift,
        ):
            pass


@pytest.mark.parametrize("pin_kind", ("empty", "forged"))
def test_capability_issuance_rejects_empty_or_forged_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pin_kind: str,
) -> None:
    with pytest.raises(
        _module().Stage6WorkflowError,
        match="Stage6InputPin|input pin.*empty|input pin.*invalid",
    ):
        with _active_execution_capability(
            tmp_path,
            monkeypatch,
            pin_kind=pin_kind,
        ):
            pass


def test_execution_capability_hot_path_has_only_lightweight_current_checks() -> None:
    module = _module()

    def dotted_name(node: ast.AST) -> str:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))

    capability_tree = ast.parse(
        inspect.getsource(module._require_stage6_execution_capability)
    )
    capability_calls = {
        dotted_name(node.func)
        for node in ast.walk(capability_tree)
        if isinstance(node, ast.Call)
    }
    assert capability_calls == {
        "Stage6WorkflowError",
        "_capability_config_payload",
        "_capability_json_payload",
        "_stage6_capability_state",
        "_stage6_pin_record_snapshot",
        "isinstance",
        "label.strip",
        "lexical_absolute",
        "state.input_pin.require_current",
        "state.review_authorization_handle.canonical_record",
        "state.run_lease.require_current",
        "type",
    }
    pin_calls = [
        node
        for node in ast.walk(capability_tree)
        if isinstance(node, ast.Call)
        and dotted_name(node.func) == "state.input_pin.require_current"
    ]
    assert len(pin_calls) == 1
    assert len(pin_calls[0].keywords) == 1
    rehash_keyword = pin_calls[0].keywords[0]
    assert rehash_keyword.arg == "rehash"
    assert isinstance(rehash_keyword.value, ast.Name)
    assert rehash_keyword.value.id == "rehash_inputs"
    assert (
        inspect.signature(
            module._require_stage6_execution_capability
        ).parameters["rehash_inputs"].default
        is False
    )

    snapshot_tree = ast.parse(inspect.getsource(module._stage6_pin_record_snapshot))
    snapshot_calls = {
        dotted_name(node.func)
        for node in ast.walk(snapshot_tree)
        if isinstance(node, ast.Call)
    }
    assert snapshot_calls == {
        "Stage6WorkflowError",
        "_normalized_capability_input_requests",
        "actual_groups.append",
        "any",
        "digest.lower",
        "input_pin.require_active_issuance",
        "isinstance",
        "len",
        "lexical_absolute",
        "os.fspath",
        "os.path.normcase",
        "path_keys.add",
        "set",
        "snapshot.append",
        "tuple",
        "type",
    }


def test_execution_capability_hot_path_596_checks_avoid_full_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    check_labels = [f"capability hot-path poll {index:03d}" for index in range(596)]

    with _active_execution_capability(tmp_path, monkeypatch) as active:
        review_handle = active["review_handle"]
        stage5_handle = active["stage5_handle"]
        input_pin = active["pin"]
        run_lease = active["lease"]

        lease_checks: list[None] = []
        pin_checks: list[tuple[str, bool]] = []
        review_record_checks: list[None] = []
        payload_reads: list[str] = []
        expensive_calls: list[str] = []
        original_lease_require_current = run_lease.require_current
        original_pin_require_current = input_pin.require_current
        original_canonical_record = review_handle.canonical_record

        def track_lease_current() -> None:
            lease_checks.append(None)
            original_lease_require_current()

        def track_pin_current(label: str, *, rehash: bool = False) -> None:
            pin_checks.append((label, rehash))
            original_pin_require_current(label, rehash=rehash)

        def track_review_record() -> dict[str, object]:
            review_record_checks.append(None)
            return original_canonical_record()

        def reject_payload_read(*args: object, **kwargs: object) -> bytes:
            del args, kwargs
            payload_reads.append("input-pin")
            raise AssertionError("capability hot path read a pinned payload")

        def reject_expensive_call(name: str):
            def reject(*args: object, **kwargs: object) -> object:
                del args, kwargs
                expensive_calls.append(name)
                raise AssertionError(f"capability hot path called {name}")

            return reject

        monkeypatch.setattr(run_lease, "require_current", track_lease_current)
        monkeypatch.setattr(input_pin, "require_current", track_pin_current)
        monkeypatch.setattr(input_pin, "read_bytes", reject_payload_read)
        monkeypatch.setattr(review_handle, "canonical_record", track_review_record)
        for name in (
            "stage6_execution_identity",
            "stage6_source_identity",
            "_verify_git_identity",
            "secure_read_bytes",
        ):
            monkeypatch.setattr(module, name, reject_expensive_call(name))

        for label in check_labels:
            module._require_stage6_execution_capability(
                active["capability"],
                label=label,
            )

        assert review_handle.events == [
            "review:Stage 6 execution capability issuance"
        ]
        assert stage5_handle.require_current_calls == [
            "Stage 6 execution capability issuance"
        ]
        assert len(lease_checks) == len(check_labels)
        assert pin_checks == [(label, False) for label in check_labels]
        assert len(review_record_checks) == len(check_labels)
        assert payload_reads == []
        assert expensive_calls == []


def test_execution_capability_is_opaque_root_bound_and_revoked_on_scope_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    revoked: object | None = None
    with _active_execution_capability(tmp_path, monkeypatch) as active:
        capability = active["capability"]
        revoked = capability
        module._require_stage6_execution_capability(
            capability,
            label="valid capability",
            formal_run_id=FORMAL_STAGE6_RUN_ID,
            run_root=active["run_root"],
            stage_root=active["run_root"] / "s6",
            repo_root=ROOT,
            config_path=CONFIG,
        )
        for operation in (
            lambda: copy.copy(capability),
            lambda: copy.deepcopy(capability),
            lambda: pickle.dumps(capability),
        ):
            with pytest.raises(TypeError, match="copied|serialized"):
                operation()
        with pytest.raises(module.Stage6WorkflowError, match="run root.*drift"):
            module._require_stage6_execution_capability(
                capability,
                label="wrong run root",
                run_root=tmp_path / "wrong-run",
            )
        with pytest.raises(module.Stage6WorkflowError, match="stage root.*drift"):
            module._require_stage6_execution_capability(
                capability,
                label="wrong stage root",
                stage_root=active["run_root"] / "wrong-stage",
            )

    assert revoked is not None
    with pytest.raises(module.Stage6WorkflowError, match="revoked|inactive"):
        module._require_stage6_execution_capability(
            revoked,
            label="after capability scope",
        )


def test_execution_operation_blocks_revoke_and_rejects_new_work_while_revoking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    operation_entered = threading.Event()
    release_operation = threading.Event()
    operation_done = threading.Event()
    revoke_done = threading.Event()
    errors: list[BaseException] = []

    with _active_execution_capability(tmp_path, monkeypatch) as active:
        capability = active["capability"]

        def run_operation() -> None:
            try:
                with module._stage6_execution_operation(
                    capability,
                    label="concurrent protected write",
                    stage_root=Path(active["run_root"]) / "s6",
                ):
                    operation_entered.set()
                    if not release_operation.wait(5.0):
                        raise AssertionError("operation release timed out")
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)
            finally:
                operation_done.set()

        def revoke() -> None:
            try:
                module._revoke_stage6_execution_capability(capability)
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)
            finally:
                revoke_done.set()

        operation_thread = threading.Thread(target=run_operation)
        operation_thread.start()
        assert operation_entered.wait(5.0)

        revoke_thread = threading.Thread(target=revoke)
        revoke_thread.start()
        for _ in range(100):
            with module._STAGE6_EXECUTION_CAPABILITY_LOCK:
                state = module._STAGE6_EXECUTION_CAPABILITY_REGISTRY.get(capability)
                if state is not None and state.revoking is True:
                    break
            threading.Event().wait(0.01)
        else:
            pytest.fail("capability never entered revoking state")

        assert not revoke_done.wait(0.05)
        with pytest.raises(module.Stage6WorkflowError, match="revoking|inactive"):
            with module._stage6_execution_operation(
                capability,
                label="new work after revoke began",
            ):
                pytest.fail("revoking capability admitted new work")

        release_operation.set()
        assert operation_done.wait(5.0)
        assert revoke_done.wait(5.0)
        operation_thread.join(timeout=1.0)
        revoke_thread.join(timeout=1.0)
        assert errors == []


def test_execution_operation_preserves_primary_error_when_exit_revalidation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    with _active_execution_capability(tmp_path, monkeypatch) as active:
        with pytest.raises(RuntimeError, match="primary mutation failure") as caught:
            with module._stage6_execution_operation(
                active["capability"],
                label="primary-error-preservation",
            ):
                active["pin"].fail_label = "primary-error-preservation exit"
                raise RuntimeError("primary mutation failure")

        notes = tuple(getattr(caught.value, "__notes__", ()))
        assert len(notes) == 1
        assert "capability close validation failure" in notes[0]
        active["pin"].fail_label = None


@pytest.mark.parametrize(
    "drift",
    ("review-record", "stage5-identity", "pin-record", "pin-current", "lease"),
)
def test_execution_capability_hot_path_fails_closed_on_bound_authority_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    module = _module()
    with _active_execution_capability(tmp_path, monkeypatch) as active:
        if drift == "review-record":
            active["review_handle"]._record["quality_review_sha256"] = "f" * 64
        elif drift == "stage5-identity":
            active["stage5_handle"].identity["gate_sha256"] = "f" * 64
        elif drift == "pin-record":
            active["pin"].records[0].sha256 = "f" * 64
        elif drift == "pin-current":
            active["pin"].fail_label = "live authority drift"
        elif drift == "lease":
            active["lease"].release()
        else:  # pragma: no cover - parametrization is closed
            raise AssertionError(drift)

        with pytest.raises(
            module.Stage6WorkflowError,
            match="current|drift|changed|inactive|released|revoked",
        ):
            module._require_stage6_execution_capability(
                active["capability"],
                label="live authority drift",
            )


def test_same_active_capability_routes_through_workflow_backend_and_all_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    from lunar_exploration_ppo.ppo import standard_training
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    run_root = tmp_path / FORMAL_STAGE6_RUN_ID
    with _active_execution_capability(
        tmp_path / "full-route-authority",
        monkeypatch,
        run_root=run_root,
    ) as active:
        capability = active["capability"]
        status = {"value": "no_terminal"}
        seen: list[tuple[str, object]] = []

        monkeypatch.setattr(
            stage6_terminal_recovery,
            "detect_stage6_terminal_recovery",
            lambda **kwargs: {
                "status": status["value"],
                "reason": "",
                "receipt_identity": {"sha256": "a" * 64, "size_bytes": 1},
                "phase_states": stage6_terminal_recovery.COMPLETE_STATES,
            },
        )
        monkeypatch.setattr(
            module,
            "_verify_stage6_recovery_authorization_bindings",
            lambda **kwargs: None,
        )
        recovered = module.Stage6WorkflowResult(
            run_id=FORMAL_STAGE6_RUN_ID,
            stage_root=run_root / "s6",
            summary={"state": "awaiting_independent_review"},
            routing={"route": "awaiting_independent_review"},
        )
        monkeypatch.setattr(
            module,
            "_stage6_recovery_workflow_result",
            lambda **kwargs: recovered,
        )

        def preflight(**kwargs):
            stage = Path(kwargs["base_output_root"]) / kwargs["run_id"] / "s6"
            (stage / "preflight").mkdir(parents=True, exist_ok=True)
            (stage / "preflight/audit.json").write_bytes(b"{}\n")
            return {"passed": True}

        monkeypatch.setattr(module, "_run_stage6_machine_preflight", preflight)

        def private_execute(**kwargs):
            current = kwargs["execution_capability"]
            seen.append(("training-wrapper", current))
            backend = object.__new__(standard_training.StandardProductionBackend)
            backend._execution_capability = current
            backend.config = kwargs["config"]
            backend.run_root = Path(kwargs["run_root"]).resolve()
            backend.stage_root = backend.run_root / "s6"
            backend.repo_root = Path(kwargs["repo_root"]).resolve()
            backend.stage5_authority = dict(kwargs["stage5_authority"])
            backend._require_execution_capability_current(
                "same-capability backend route"
            )
            seen.append(("production-backend", backend._execution_capability))
            stage6_terminal_recovery._require_recovery_execution_capability(
                backend._execution_capability,
                stage_root=backend.stage_root,
                label="same-capability backend internal recovery route",
            )
            seen.append(
                ("backend-internal-recovery", backend._execution_capability)
            )
            return {
                "stage_root": str(backend.stage_root),
                "summary": {"state": "training"},
                "routing": {"route": "resume"},
            }

        monkeypatch.setattr(
            standard_training,
            "_execute_standard_training",
            private_execute,
        )

        def preterminal_recovery(**kwargs):
            current = kwargs["execution_capability"]
            module._require_stage6_execution_capability(
                current,
                label="same-capability preterminal route",
                stage_root=kwargs["stage_root"],
            )
            seen.append(("preterminal-recovery", current))
            status["value"] = "valid_terminal_recovery"
            return {"phase_states": stage6_terminal_recovery.COMPLETE_STATES}

        monkeypatch.setattr(
            module,
            "_recover_stage6_preterminal_receipt",
            preterminal_recovery,
        )

        def terminal_recovery(
            *,
            stage_root,
            manifest_committer,
            execution_capability,
            planning_child_recovery_capability=None,
        ):
            del manifest_committer
            assert planning_child_recovery_capability is None
            stage6_terminal_recovery._require_recovery_execution_capability(
                execution_capability,
                stage_root=stage_root,
                label="same-capability terminal-only route",
            )
            seen.append(("terminal-recovery", execution_capability))
            return {"phase_states": stage6_terminal_recovery.COMPLETE_STATES}

        monkeypatch.setattr(
            stage6_terminal_recovery,
            "recover_stage6_terminal_commit",
            terminal_recovery,
        )

        arguments = {
            "config": active["config"],
            "config_path": CONFIG,
            "run_id": FORMAL_STAGE6_RUN_ID,
            "stage5_gate_path": tmp_path / "stage5-gate.json",
            "base": tmp_path,
            "run_root": run_root,
            "repo": ROOT,
            "run_preexisted": False,
            "review_authorization_handle": active["review_handle"],
            "stage5_authority": active["stage5_handle"],
            "input_pin": active["pin"],
            "execution_identity": active["identity"],
            "execution_capability": capability,
        }
        trained = module._run_stage6_workflow_locked(**arguments)
        assert trained.stage_root == run_root / "s6"
        _write_minimal_handle_capture_fixture(run_root / "s6")

        status["value"] = "valid_preterminal_recovery"
        assert module._run_stage6_workflow_locked(
            **{**arguments, "run_preexisted": True}
        ) == recovered

        status["value"] = "valid_terminal_recovery"
        assert module._run_stage6_workflow_locked(
            **{**arguments, "run_preexisted": True}
        ) == recovered

        assert [label for label, _value in seen] == [
            "training-wrapper",
            "production-backend",
            "backend-internal-recovery",
            "preterminal-recovery",
            "terminal-recovery",
        ]
        assert all(value is capability for _label, value in seen)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("formal_run_id", "s6-standard-single-r1-20260717T010204Z"),
        ("base_commit", "f" * 40),
        ("reviewed_prospective_git_tree", "e" * 40),
        ("changed_path_set_sha256", "d" * 64),
        ("source_set_sha256", "d" * 64),
        ("config_sha256", "d" * 64),
        ("data_sha256", "d" * 64),
        ("environment_sha256", "d" * 64),
    ),
)
def test_verified_review_record_exactly_binds_current_execution_identity(
    field: str,
    value: object,
) -> None:
    module = _module()
    identity, verified = _review_identity_fixture(module)
    assert module.validate_stage6_verified_review_authorization(
        verified,
        execution_identity=identity,
        formal_run_id=FORMAL_STAGE6_RUN_ID,
    ) == verified

    drifted = dict(verified)
    drifted[field] = value
    with pytest.raises(module.Stage6WorkflowError, match="review.*authorization"):
        module.validate_stage6_verified_review_authorization(
            drifted,
            execution_identity=identity,
            formal_run_id=FORMAL_STAGE6_RUN_ID,
        )


def test_formal_parser_requires_review_authorization_and_rejects_output_root_option() -> None:
    spec = importlib.util.spec_from_file_location("stage6_formal_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(["--run-id", FORMAL_STAGE6_RUN_ID])

    review_authorization = Path("D:/xunce/review/launch-authorization.json")
    planning_warm_start = Path("D:/xunce/review/planning-warm-start.json")
    parsed = runner.build_parser().parse_args(
        [
            "--run-id",
            FORMAL_STAGE6_RUN_ID,
            "--review-authorization",
            str(review_authorization),
            "--planning-warm-start",
            str(planning_warm_start),
        ]
    )
    assert parsed.run_id == FORMAL_STAGE6_RUN_ID
    assert parsed.review_authorization == review_authorization
    assert parsed.planning_warm_start == planning_warm_start
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(
            [
                "--run-id",
                FORMAL_STAGE6_RUN_ID,
                "--review-authorization",
                str(review_authorization),
                "--planning-warm-start",
                str(planning_warm_start),
                "--output-root",
                "D:/alternate",
            ]
        )


def test_formal_documentation_requires_run_bound_review_authorization_and_r1_rollover() -> None:
    stage6_doc = (ROOT / "docs/ppo-highres-frontier-stage6.md").read_text(
        encoding="utf-8"
    )
    single_seed_plan = (
        ROOT
        / "docs/superpowers/plans/2026-07-16-ppo-stage6-single-seed-pipeline.md"
    ).read_text(encoding="utf-8")

    for document in (stage6_doc, single_seed_plan):
        normalized = " ".join(document.split())
        assert "--review-authorization" in normalized
        assert "formal_run_id" in normalized
        assert "同一外部 authorization" in normalized
        assert "R1 是不可变的失败证据" in normalized
        assert "Update 4 的 checkpoint、optimizer、policy、RNG 或 vector state 均不会被携带" in normalized
        assert "新的正式 run 从 Update 1 重新开始" in normalized
        assert "只有用户明确要求才追加" in normalized


@pytest.mark.parametrize(
    "run_id",
    (
        *HISTORICAL_STAGE6_RUN_IDS,
        "canonical-run",
        "arbitrary-fresh-id",
        "s6-standard-single-r2-20260717T010203Z",
        STANDALONE_PREFLIGHT_RUN_ID,
        "s6-standard-single-r1-20260230T010203Z",
        "s6-standard-single-r1-20260717T246000Z",
        "s6-standard-single-r1-20260717T010203z",
    ),
)
def test_formal_parser_rejects_noncanonical_run_ids_without_filesystem_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_id: str,
) -> None:
    spec = importlib.util.spec_from_file_location("stage6_formal_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    workflow = _module()
    base = tmp_path / "out"
    target = base / run_id
    sentinel = target / "immutable-evidence.bin"
    if run_id in HISTORICAL_STAGE6_RUN_IDS:
        target.mkdir(parents=True)
        sentinel.write_bytes(b"immutable-cli-evidence\x00\xff")
    before = sentinel.read_bytes() if sentinel.exists() else None
    monkeypatch.setattr(workflow, "CANONICAL_STAGE6_OUTPUT_ROOT", base)

    with pytest.raises(SystemExit):
        runner.main(["--run-id", run_id])

    if before is None:
        assert not target.exists()
    else:
        assert sentinel.read_bytes() == before
        assert {child.name for child in target.iterdir()} == {
            "immutable-evidence.bin"
        }
    assert not (target / ".stage6.lease").exists()


@pytest.mark.parametrize(
    ("run_id", "preexisting"),
    (
        *((run_id, True) for run_id in HISTORICAL_STAGE6_RUN_IDS),
        ("canonical-run", False),
        ("arbitrary-fresh-id", False),
        ("s6-standard-single-r1-20260230T010203Z", False),
        (STANDALONE_PREFLIGHT_RUN_ID, False),
    ),
)
def test_public_workflow_rejects_invalid_run_id_before_config_or_filesystem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_id: str,
    preexisting: bool,
) -> None:
    module = _module()
    from lunar_exploration_ppo.configs import stage6 as stage6_config

    base = tmp_path / "out"
    target = base / run_id
    sentinel = target / "immutable-evidence.bin"
    if preexisting:
        target.mkdir(parents=True)
        sentinel.write_bytes(b"immutable-stage6-evidence\x00\xff")
    before = sentinel.read_bytes() if preexisting else None
    config_loads: list[Path] = []
    authorization_calls: list[dict[str, object]] = []

    def reject_config_load(path: str | Path):
        config_loads.append(Path(path))
        raise AssertionError("run-id rejection must precede config loading")

    monkeypatch.setattr(module, "CANONICAL_STAGE6_OUTPUT_ROOT", base)
    monkeypatch.setattr(stage6_config, "load_stage6_config", reject_config_load)
    from lunar_exploration_ppo.workflows import stage6_review_authorization

    monkeypatch.setattr(
        stage6_review_authorization,
        "verify_stage6_review_launch_authorization",
        lambda **kwargs: authorization_calls.append(dict(kwargs)),
    )

    with pytest.raises(module.Stage6WorkflowError, match="run id|historical|formal"):
        module.run_stage6_workflow(
            config_path=tmp_path / "must-not-be-read.json",
            run_id=run_id,
            stage5_gate_path=tmp_path / "must-not-be-read-gate.json",
            review_authorization_path=tmp_path / "must-not-be-read-auth.json",
        )

    assert config_loads == []
    assert authorization_calls == []
    if preexisting:
        assert sentinel.read_bytes() == before
        assert {child.name for child in target.iterdir()} == {
            "immutable-evidence.bin"
        }
    else:
        assert not target.exists()
    assert not (target / ".stage6.lease").exists()


def test_public_workflow_verifies_review_authorization_before_config_or_filesystem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    from lunar_exploration_ppo.configs import stage6 as stage6_config
    from lunar_exploration_ppo.workflows import stage6_review_authorization

    base = tmp_path / "out"
    target = base / FORMAL_STAGE6_RUN_ID
    events: list[str] = []

    def reject_authorization(**kwargs: object) -> dict[str, object]:
        events.append("authorization")
        assert kwargs["expected_run_id"] == FORMAL_STAGE6_RUN_ID
        raise stage6_review_authorization.Stage6ReviewAuthorizationError(
            "reviewed source byte drift"
        )

    def reject_config_load(path: str | Path):
        del path
        events.append("config")
        raise AssertionError("authorization rejection must precede config loading")

    monkeypatch.setattr(module, "CANONICAL_STAGE6_OUTPUT_ROOT", base)
    monkeypatch.setattr(
        stage6_review_authorization,
        "verify_stage6_review_launch_authorization",
        reject_authorization,
    )
    monkeypatch.setattr(stage6_config, "load_stage6_config", reject_config_load)

    with pytest.raises(module.Stage6WorkflowError, match="review.*authorization"):
        module.run_stage6_workflow(
            config_path=tmp_path / "must-not-be-read.json",
            run_id=FORMAL_STAGE6_RUN_ID,
            stage5_gate_path=tmp_path / "must-not-be-read-gate.json",
            review_authorization_path=tmp_path / "drifted-auth.json",
        )

    assert events == ["authorization"]
    assert not target.exists()
    assert not (target / ".stage6.lease").exists()


def test_public_workflow_rejects_authorization_for_another_legal_run_id_without_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    from lunar_exploration_ppo.configs import stage6 as stage6_config
    from lunar_exploration_ppo.workflows import stage6_review_authorization

    base = tmp_path / "out"
    target = base / FORMAL_STAGE6_RUN_ID
    other_formal_run_id = "s6-standard-single-r1-20260717T010204Z"
    events: list[str] = []

    def reject_other_run(**kwargs: object) -> dict[str, object]:
        events.append("authorization")
        assert kwargs["expected_run_id"] == FORMAL_STAGE6_RUN_ID
        raise stage6_review_authorization.Stage6ReviewAuthorizationError(
            f"authorization formal run id is {other_formal_run_id}"
        )

    def reject_config_load(path: str | Path):
        del path
        raise AssertionError("cross-run rejection must precede config loading")

    monkeypatch.setattr(module, "CANONICAL_STAGE6_OUTPUT_ROOT", base)
    monkeypatch.setattr(
        stage6_review_authorization,
        "verify_stage6_review_launch_authorization",
        reject_other_run,
    )
    monkeypatch.setattr(stage6_config, "load_stage6_config", reject_config_load)

    with pytest.raises(module.Stage6WorkflowError, match="review.*authorization"):
        module.run_stage6_workflow(
            config_path=tmp_path / "must-not-be-read.json",
            run_id=FORMAL_STAGE6_RUN_ID,
            stage5_gate_path=tmp_path / "must-not-be-read-gate.json",
            review_authorization_path=tmp_path / "other-run-auth.json",
        )

    assert events == ["authorization"]
    assert not target.exists()
    assert not (target / ".stage6.lease").exists()


def test_standalone_preflight_uses_distinct_strict_run_id_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    calls: list[dict[str, object]] = []

    def capture(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"passed": True}

    monkeypatch.setattr(module, "_run_stage6_machine_preflight", capture)

    assert module.run_stage6_machine_preflight(
        config_path=tmp_path / "config.json",
        run_id=STANDALONE_PREFLIGHT_RUN_ID,
        stage5_gate_path=tmp_path / "gate.json",
    ) == {"passed": True}
    assert calls[-1]["run_id"] == STANDALONE_PREFLIGHT_RUN_ID

    for run_id in (
        FORMAL_STAGE6_RUN_ID,
        *HISTORICAL_STAGE6_RUN_IDS,
        "s6-standard-single-preflight-r1-20260230T010203Z",
    ):
        with pytest.raises(
            module.Stage6WorkflowError,
            match="preflight run id|historical",
        ):
            module.run_stage6_machine_preflight(
                config_path=tmp_path / "config.json",
                run_id=run_id,
                stage5_gate_path=tmp_path / "gate.json",
            )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "run_id",
    (*HISTORICAL_STAGE6_RUN_IDS, STANDALONE_PREFLIGHT_RUN_ID, "canonical-run"),
)
def test_public_resume_preflight_rejects_nonformal_id_before_config_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_id: str,
) -> None:
    module = _module()
    from lunar_exploration_ppo.configs import stage6 as stage6_config

    config_loads: list[Path] = []

    def reject_config_load(path: str | Path):
        config_loads.append(Path(path))
        raise AssertionError("run-id rejection must precede config loading")

    monkeypatch.setattr(stage6_config, "load_stage6_config", reject_config_load)

    with pytest.raises(module.Stage6WorkflowError, match="run id|historical|formal"):
        module.verify_stage6_machine_preflight_for_resume(
            config_path=tmp_path / "must-not-be-read.json",
            run_id=run_id,
            stage5_gate_path=tmp_path / "must-not-be-read-gate.json",
        )
    assert config_loads == []


def test_canonical_output_root_check_normalizes_but_rejects_other_d_roots() -> None:
    module = _module()

    assert module._require_canonical_stage6_output_root(
        "D:/xunce/out/ppo_frontier/."
    ) == module._require_canonical_stage6_output_root(
        module.CANONICAL_STAGE6_OUTPUT_ROOT
    )
    with pytest.raises(module.Stage6WorkflowError, match="canonical"):
        module._require_canonical_stage6_output_root("D:/xunce/out/other")


def test_stage6_new_public_execution_apis_are_exported() -> None:
    module = _module()
    from lunar_exploration_ppo.ppo import standard_training

    assert {
        "FinalEvaluationJob",
        "rebind_stage6_manifest_for_terminal_resource",
        "validate_stage6_machine_preflight_audit",
        "verify_stage6_preterminal_acceptance",
    }.issubset(module.__all__)
    assert {
        "SAMPLER_SEED_DERIVATION_VERSION",
        "EvalIsolationSnapshot",
        "StandardUpdateResult",
        "append_transaction_metric_once",
        "build_standard_final_aggregate_artifacts",
        "derive_standard_sampler_seeds",
        "reconcile_checkpointed_resume",
        "run_eval_only_transaction",
        "run_standard_training_schedule",
        "run_standard_update_transaction",
        "select_validation_checkpoint_best",
        "validate_checkpoint_replay_audit",
        "validate_stage6_resource_audit",
        "validate_standard_collection_audit",
        "verify_standard_final_evaluation_artifacts",
    }.issubset(standard_training.__all__)


@pytest.mark.parametrize(
    ("field", "threshold"),
    (
        ("rss_bytes", 20 * 1024**3),
        ("peak_vram_bytes", int(10.1 * 1024**3)),
    ),
)
def test_machine_preflight_rejects_resource_value_at_exact_hard_limit(
    field: str,
    threshold: int,
) -> None:
    from test_stage6_machine_preflight import _valid_audit

    module = _module()
    audit = _valid_audit()
    audit["resources"][field] = threshold  # type: ignore[index]

    with pytest.raises(module.Stage6WorkflowError, match="resource"):
        module.validate_stage6_machine_preflight_audit(audit)


def test_workflow_preflights_new_root_and_reuses_verified_audit_for_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    from lunar_exploration_ppo.ppo import standard_training
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    authorization_calls: list[dict[str, object]] = []
    events: list[str] = []
    issued_capabilities: list[object] = []
    review_handle = _FakeReviewAuthorizationHandle(events=events)
    _allow_review_authorization(
        monkeypatch,
        authorization_calls,
        tmp_path=tmp_path,
        handle=review_handle,
    )

    def preflight(**kwargs):
        events.append("preflight:new")
        audit = (
            Path(kwargs["base_output_root"])
            / kwargs["run_id"]
            / "s6/preflight/audit.json"
        )
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_bytes(b"{}\n")
        return {"passed": True}

    def verify_resume(**kwargs):
        events.append("preflight:resume")
        audit = (
            Path(kwargs["base_output_root"])
            / kwargs["run_id"]
            / "s6/preflight/audit.json"
        )
        assert audit.read_bytes() == b"{}\n"
        return {"passed": True}

    monkeypatch.setattr(module, "_run_stage6_machine_preflight", preflight)
    monkeypatch.setattr(
        module,
        "_verify_stage6_machine_preflight_for_resume_at_root",
        verify_resume,
        raising=False,
    )

    def execute(**kwargs):
        events.append("execute")
        capability = kwargs["execution_capability"]
        issued_capabilities.append(capability)
        standard_training._require_standard_execution_capability(
            capability,
            label="workflow-to-training integration boundary",
            config=kwargs["config"],
            run_root=kwargs["run_root"],
            repo_root=kwargs["repo_root"],
            stage5_authority=kwargs["stage5_authority"],
            verified_review_authorization=kwargs[
                "verified_review_authorization"
            ],
        )
        stage6_terminal_recovery._require_recovery_execution_capability(
            capability,
            stage_root=Path(kwargs["run_root"]) / "s6",
            label="training-to-terminal-recovery integration boundary",
        )
        assert type(kwargs["verified_review_authorization"]) is dict
        assert kwargs["verified_review_authorization"] == review_handle.canonical_record()
        assert kwargs["verified_review_authorization"] is not review_handle
        stage = Path(kwargs["run_root"]) / "s6"
        stage.mkdir(parents=True, exist_ok=True)
        return {
            "stage_root": str(stage),
            "summary": {"state": "training"},
            "routing": {"route": "resume"},
        }

    monkeypatch.setattr(standard_training, "execute_standard_training", execute)
    base = tmp_path / "out"
    arguments = {
        "config_path": CONFIG,
        "run_id": "s6-standard-single-r1-20260717T020300Z",
        "stage5_gate_path": tmp_path / "gate.json",
        "base_output_root": base,
        "review_authorization_path": tmp_path / "launch-authorization.json",
    }

    first = module._run_stage6_workflow_for_test(**arguments)
    second = module._run_stage6_workflow_for_test(**arguments)

    assert first.stage_root == second.stage_root
    assert [event for event in events if event.startswith("preflight:")] == [
        "preflight:new",
        "preflight:resume",
    ]
    assert events.count("execute") == 2
    assert len(issued_capabilities) == 2
    assert issued_capabilities[0] is not issued_capabilities[1]
    for capability in issued_capabilities:
        with pytest.raises(module.Stage6WorkflowError, match="revoked|inactive"):
            module._require_stage6_execution_capability(
                capability,
                label="completed full-route capability scope",
            )
    assert len(authorization_calls) == 2
    assert all(
        call["expected_run_id"] == arguments["run_id"]
        for call in authorization_calls
    )
    assert events.index("preflight:new") < events.index("execute")
    assert events == [
        "review:Stage 6 review authorization before input pin acquisition",
        "review:Stage 6 review authorization before run output side effects",
        "review:Stage 6 execution capability issuance",
        "review:Stage 6 review authorization after run lease",
        "preflight:new",
        "review:Stage 6 review authorization after machine preflight before backend import",
        "review:Stage 6 review authorization before training call",
        "execute",
        "review:Stage 6 review authorization after training return",
        "review:Stage 6 review authorization before input pin acquisition",
        "review:Stage 6 review authorization before run output side effects",
        "review:Stage 6 execution capability issuance",
        "review:Stage 6 review authorization after run lease",
        "preflight:resume",
        "review:Stage 6 review authorization after machine preflight before backend import",
        "review:Stage 6 review authorization before training call",
        "execute",
        "review:Stage 6 review authorization after training return",
    ]
    assert (base / arguments["run_id"] / ".stage6.lease").is_file()
    assert not (base / ".stage6-locks").exists()


def test_review_drift_after_initial_verification_fails_before_run_output_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    base = tmp_path / "out"
    run_id = "s6-standard-single-r1-20260717T020301Z"
    events: list[str] = []
    handle = _FakeReviewAuthorizationHandle(
        events=events,
        fail_label="Stage 6 review authorization before run output side effects",
    )
    _allow_review_authorization(monkeypatch, tmp_path=tmp_path, handle=handle)
    monkeypatch.setattr(
        module,
        "_run_stage6_workflow_locked",
        lambda **kwargs: pytest.fail("drift must fail before locked workflow"),
    )

    with pytest.raises(module.Stage6WorkflowError, match="review.*authorization"):
        module._run_stage6_workflow_for_test(
            config_path=CONFIG,
            run_id=run_id,
            stage5_gate_path=tmp_path / "gate.json",
            base_output_root=base,
            review_authorization_path=tmp_path / "launch-authorization.json",
        )

    assert events == [
        "review:Stage 6 review authorization before input pin acquisition",
        "review:Stage 6 review authorization before run output side effects"
    ]
    assert not (base / run_id).exists()


def test_review_drift_after_run_lease_fails_before_preflight_or_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    base = tmp_path / "out"
    run_id = "s6-standard-single-r1-20260717T020302Z"
    events: list[str] = []
    handle = _FakeReviewAuthorizationHandle(
        events=events,
        fail_label="Stage 6 review authorization after run lease",
    )
    _allow_review_authorization(monkeypatch, tmp_path=tmp_path, handle=handle)
    monkeypatch.setattr(
        stage6_terminal_recovery,
        "detect_stage6_terminal_recovery",
        lambda **kwargs: pytest.fail("drift must fail before recovery detection"),
    )
    monkeypatch.setattr(
        module,
        "_run_stage6_machine_preflight",
        lambda **kwargs: pytest.fail("drift must fail before preflight"),
    )

    with pytest.raises(module.Stage6WorkflowError, match="review.*authorization"):
        module._run_stage6_workflow_for_test(
            config_path=CONFIG,
            run_id=run_id,
            stage5_gate_path=tmp_path / "gate.json",
            base_output_root=base,
            review_authorization_path=tmp_path / "launch-authorization.json",
        )

    assert events == [
        "review:Stage 6 review authorization before input pin acquisition",
        "review:Stage 6 review authorization before run output side effects",
        "review:Stage 6 execution capability issuance",
        "review:Stage 6 review authorization after run lease",
    ]
    assert (base / run_id / ".stage6.lease").is_file()
    assert not (base / run_id / "s6").exists()


def test_review_drift_after_machine_preflight_fails_before_backend_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    module = _module()
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    base = tmp_path / "out"
    run_id = "s6-standard-single-r1-20260717T020303Z"
    events: list[str] = []
    handle = _FakeReviewAuthorizationHandle(
        events=events,
        fail_label=(
            "Stage 6 review authorization after machine preflight before backend import"
        ),
    )
    _allow_review_authorization(monkeypatch, tmp_path=tmp_path, handle=handle)

    monkeypatch.setattr(
        stage6_terminal_recovery,
        "detect_stage6_terminal_recovery",
        lambda **kwargs: {"status": "no_terminal", "reason": ""},
    )

    def preflight(**kwargs):
        events.append("preflight")
        stage = base / run_id / "s6/preflight"
        stage.mkdir(parents=True)
        (stage / "audit.json").write_bytes(b"{}\n")
        return {"passed": True}

    monkeypatch.setattr(module, "_run_stage6_machine_preflight", preflight)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "lunar_exploration_ppo.ppo.standard_training":
            events.append("backend_import")
            raise AssertionError("authorization drift imported backend")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(module.Stage6WorkflowError, match="review.*authorization"):
        module._run_stage6_workflow_for_test(
            config_path=CONFIG,
            run_id=run_id,
            stage5_gate_path=tmp_path / "gate.json",
            base_output_root=base,
            review_authorization_path=tmp_path / "launch-authorization.json",
        )

    assert "backend_import" not in events
    assert events[-1] == (
        "review:Stage 6 review authorization after machine preflight before backend import"
    )


def test_review_drift_before_training_call_fails_before_backend_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    base = tmp_path / "out"
    events: list[str] = []
    handle = _FakeReviewAuthorizationHandle(
        events=events,
        fail_label="Stage 6 review authorization before training call",
    )
    _allow_review_authorization(monkeypatch, tmp_path=tmp_path, handle=handle)
    _install_stage6_new_run_fakes(module, monkeypatch, events=events)

    with pytest.raises(module.Stage6WorkflowError, match="review.*authorization"):
        module._run_stage6_workflow_for_test(
            config_path=CONFIG,
            run_id="s6-standard-single-r1-20260717T020304Z",
            stage5_gate_path=tmp_path / "gate.json",
            base_output_root=base,
            review_authorization_path=tmp_path / "launch-authorization.json",
        )

    assert "execute" not in events
    assert events[-1] == "review:Stage 6 review authorization before training call"


def test_review_drift_after_training_return_fails_before_workflow_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    events: list[str] = []
    handle = _FakeReviewAuthorizationHandle(
        events=events,
        fail_label="Stage 6 review authorization after training return",
    )
    _allow_review_authorization(monkeypatch, tmp_path=tmp_path, handle=handle)
    _install_stage6_new_run_fakes(module, monkeypatch, events=events)

    with pytest.raises(module.Stage6WorkflowError, match="review.*authorization"):
        module._run_stage6_workflow_for_test(
            config_path=CONFIG,
            run_id="s6-standard-single-r1-20260717T020305Z",
            stage5_gate_path=tmp_path / "gate.json",
            base_output_root=tmp_path / "out",
            review_authorization_path=tmp_path / "launch-authorization.json",
        )

    assert "execute" in events
    assert events[-1] == "review:Stage 6 review authorization after training return"


def test_terminal_existing_formal_entry_imports_no_training_workload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    module = _module()
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    authorization_calls: list[dict[str, object]] = []
    events: list[str] = []
    review_handle = _FakeReviewAuthorizationHandle(events=events)
    _allow_review_authorization(
        monkeypatch,
        authorization_calls,
        tmp_path=tmp_path,
        handle=review_handle,
    )
    base = tmp_path / "out"
    stage = base / FORMAL_STAGE6_RUN_ID / "s6"
    _write_minimal_handle_capture_fixture(stage)
    expected = module.Stage6WorkflowResult(
        run_id=FORMAL_STAGE6_RUN_ID,
        stage_root=stage,
        summary={"state": "awaiting_independent_review"},
        routing={"route": "awaiting_independent_review"},
    )
    monkeypatch.setattr(
        stage6_terminal_recovery,
        "detect_stage6_terminal_recovery",
        lambda **kwargs: {
            "status": "valid_terminal_recovery",
            "reason": "",
            "receipt_identity": {"sha256": "a" * 64, "size_bytes": 1},
            "phase_states": stage6_terminal_recovery.COMPLETE_STATES,
        },
    )
    monkeypatch.setattr(
        stage6_terminal_recovery,
        "recover_stage6_terminal_commit",
        lambda **kwargs: events.append("pure_commit") or {
            "receipt_identity": {"sha256": "a" * 64, "size_bytes": 1},
            "phase_states": stage6_terminal_recovery.COMPLETE_STATES,
            "manifest_identity": {"sha256": "b" * 64, "size_bytes": 1},
        },
    )
    monkeypatch.setattr(
        module,
        "_verify_stage6_recovery_authorization_bindings",
        lambda **kwargs: events.append("authorization_lineage"),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_stage6_recovery_workflow_result",
        lambda **kwargs: expected,
        raising=False,
    )

    monkeypatch.setattr(
        module,
        "_run_stage6_machine_preflight",
        lambda **kwargs: pytest.fail("terminal recovery must not run preflight"),
    )
    monkeypatch.setattr(
        module,
        "_verify_stage6_machine_preflight_for_resume_at_root",
        lambda **kwargs: pytest.fail("terminal recovery must not verify preflight"),
    )

    forbidden = {
        "lunar_exploration_ppo.ppo.standard_training": "backend/execute",
        "lunar_exploration_ppo.ppo.checkpoint": "checkpoint",
        "lunar_exploration_ppo.ppo.collector": "worker",
        "lunar_exploration_ppo.eval.standard": "evaluator",
        "lunar_exploration_ppo.env.standard_training": "environment",
        "torch": "model/optimizer",
    }
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        label = next(
            (value for prefix, value in forbidden.items() if name == prefix),
            None,
        )
        if label is not None:
            events.append(label)
            raise AssertionError(f"terminal recovery imported {label}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = module._run_stage6_workflow_for_test(
        config_path=CONFIG,
        run_id=FORMAL_STAGE6_RUN_ID,
        stage5_gate_path=tmp_path / "gate.json",
        base_output_root=base,
        review_authorization_path=tmp_path / "launch-authorization.json",
    )

    assert result == expected
    assert events == [
        "review:Stage 6 review authorization before input pin acquisition",
        "review:Stage 6 review authorization before run output side effects",
        "review:Stage 6 execution capability issuance",
        "review:Stage 6 review authorization after run lease",
        "authorization_lineage",
        "review:Stage 6 review authorization before terminal recovery commit",
        "pure_commit",
        "review:Stage 6 review authorization after terminal recovery commit",
        "authorization_lineage",
        "review:Stage 6 review authorization before recovery result return",
    ]
    assert len(authorization_calls) == 1


def test_recovery_result_rejects_minimal_self_consistent_receipt(
    tmp_path: Path,
) -> None:
    module = _module()
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    stage = tmp_path / "s6"
    stage.mkdir()
    summary = module.ArtifactStore.canonical_json_bytes(
        {"machine_passed": False, "state": "forged-unverified-state"}
    )
    routing = module.ArtifactStore.canonical_json_bytes(
        {
            "route": "forged-unverified-route",
            "state": "forged-unverified-state",
        }
    )
    rows = []
    for relative, payload in (
        ("summary.json", summary),
        ("routing.json", routing),
    ):
        (stage / relative).write_bytes(payload)
        rows.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "utf8": payload.decode("utf-8"),
            }
        )
    (stage / "preterminal_acceptance.json").write_bytes(
        module.ArtifactStore.canonical_json_bytes({"terminal_artifacts": rows})
    )
    (stage / "resource_audit.jsonl").write_bytes(b"{}\n")
    (stage / "phase-state.jsonl").write_bytes(b"{}\n")

    with stage6_terminal_recovery._capture_terminal_evidence_handle(
        stage
    ) as evidence_handle:
        kwargs = {
            "stage_root": stage,
            "run_id": FORMAL_STAGE6_RUN_ID,
        }
        if "evidence_handle" in inspect.signature(
            module._stage6_recovery_workflow_result
        ).parameters:
            kwargs["evidence_handle"] = evidence_handle
        with pytest.raises(
            module.Stage6WorkflowError,
            match="receipt|schema|incomplete|terminal artifact",
        ):
            module._stage6_recovery_workflow_result(**kwargs)


def test_formal_terminal_recovery_rejects_post_validation_replacement_without_raw_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    run_root = tmp_path / FORMAL_STAGE6_RUN_ID
    stage = run_root / "s6"
    stage.mkdir(parents=True)
    for relative, payload in {
        "preterminal_acceptance.json": b"{}\n",
        "resource_audit.jsonl": b"{}\n",
        "phase-state.jsonl": b"{}\n",
        "summary.json": b'{"state":"validated-state"}\n',
        "routing.json": b'{"route":"validated-route"}\n',
    }.items():
        (stage / relative).write_bytes(payload)

    terminal_detection_handle = None
    authorization_handles: list[object | None] = []
    result_handles: list[object | None] = []
    replacement_outcomes: list[str] = []

    def detect(**kwargs: object) -> dict[str, object]:
        nonlocal terminal_detection_handle
        candidate = kwargs.get("evidence_handle")
        if candidate is not None:
            terminal_detection_handle = candidate
        return {
            "status": "valid_terminal_recovery",
            "reason": "",
            "receipt_identity": {"sha256": "a" * 64, "size_bytes": 1},
            "phase_states": stage6_terminal_recovery.COMPLETE_STATES,
        }

    def verify_authorization(**kwargs: object) -> None:
        authorization_handles.append(kwargs.get("evidence_handle"))

    expected = module.Stage6WorkflowResult(
        run_id=FORMAL_STAGE6_RUN_ID,
        stage_root=stage,
        summary={"state": "validated-state"},
        routing={"route": "validated-route"},
    )

    def replace_after_validation(**kwargs: object) -> object:
        result_handles.append(kwargs.get("evidence_handle"))
        replacement = stage / "summary.replace.tmp"
        replacement.write_bytes(b'{"state":"replaced-after-validation"}\n')
        try:
            os.replace(replacement, stage / "summary.json")
        except PermissionError:
            replacement_outcomes.append("blocked_by_strong_pin")
            replacement.unlink()
        else:
            replacement_outcomes.append("replaced")
        return expected

    monkeypatch.setattr(
        module,
        "_require_stage6_execution_capability",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "_require_review_authorization_current",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "_require_input_pin_current",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "_verify_stage6_recovery_authorization_bindings",
        verify_authorization,
    )
    monkeypatch.setattr(
        module,
        "_stage6_recovery_workflow_result",
        replace_after_validation,
    )
    monkeypatch.setattr(
        stage6_terminal_recovery,
        "detect_stage6_terminal_recovery",
        detect,
    )
    monkeypatch.setattr(
        stage6_terminal_recovery,
        "recover_stage6_terminal_commit",
        lambda **kwargs: {"phase_states": stage6_terminal_recovery.COMPLETE_STATES},
    )

    original_read_bytes = Path.read_bytes
    raw_reopens: list[str] = []
    terminal_names = {
        "preterminal_acceptance.json",
        "lineage_audit.json",
        "manifest.json",
        "summary.json",
        "routing.json",
    }

    def reject_raw_terminal_reopen(self: Path) -> bytes:
        if self.parent == stage and self.name in terminal_names:
            raw_reopens.append(self.name)
            raise AssertionError(f"formal recovery raw-reopened {self.name}")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", reject_raw_terminal_reopen)

    try:
        result = module._run_stage6_workflow_locked(
            config=object(),
            config_path=CONFIG,
            run_id=FORMAL_STAGE6_RUN_ID,
            stage5_gate_path=tmp_path / "stage5-gate.json",
            base=tmp_path,
            run_root=run_root,
            repo=ROOT,
            run_preexisted=True,
            review_authorization_handle=_FakeReviewAuthorizationHandle(),
            stage5_authority=SimpleNamespace(identity={}),
            input_pin=object(),
            execution_identity={},
            execution_capability=object(),
            run_lease=object(),
        )
    except module.Stage6WorkflowError as exc:
        assert replacement_outcomes == ["replaced"]
        assert any(
            token in str(exc).lower()
            for token in ("evidence", "changed", "binding", "recovery")
        )
    else:
        assert replacement_outcomes == ["blocked_by_strong_pin"]
        assert result == expected
        assert original_read_bytes(stage / "summary.json") == (
            b'{"state":"validated-state"}\n'
        )

    assert raw_reopens == []
    assert terminal_detection_handle is not None
    assert authorization_handles[-1] is terminal_detection_handle
    assert result_handles == [terminal_detection_handle]


def test_formal_terminal_recovery_revalidates_handle_after_result_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    run_root = tmp_path / FORMAL_STAGE6_RUN_ID
    stage = run_root / "s6"
    stage.mkdir(parents=True)
    for relative, payload in {
        "preterminal_acceptance.json": b"{}\n",
        "resource_audit.jsonl": b"{}\n",
        "phase-state.jsonl": b"{}\n",
        "summary.json": b'{"state":"validated-state"}\n',
        "routing.json": b'{"route":"validated-route"}\n',
    }.items():
        (stage / relative).write_bytes(payload)

    events: list[tuple[str, object | None]] = []
    terminal_detection_handle = None

    def detect(**kwargs: object) -> dict[str, object]:
        nonlocal terminal_detection_handle
        candidate = kwargs.get("evidence_handle")
        if candidate is not None:
            terminal_detection_handle = candidate
        return {
            "status": "valid_terminal_recovery",
            "reason": "",
            "receipt_identity": {"sha256": "a" * 64, "size_bytes": 1},
            "phase_states": stage6_terminal_recovery.COMPLETE_STATES,
        }

    def require_review(_handle: object, label: str) -> None:
        if label == "Stage 6 review authorization before recovery result return":
            events.append(("review_current", None))

    def require_input(_pin: object, label: str, *, rehash: bool = True) -> None:
        del rehash
        if label == "Stage 6 input pin before recovery result return":
            events.append(("input_pin_current", None))

    def construct_result(**kwargs: object) -> object:
        handle = kwargs.get("evidence_handle")
        result = module.Stage6WorkflowResult(
            run_id=FORMAL_STAGE6_RUN_ID,
            stage_root=stage,
            summary={"state": "validated-state"},
            routing={"route": "validated-route"},
        )
        events.append(("result_constructed", handle))
        return result

    original_require_current = (
        stage6_terminal_recovery._Stage6EvidenceHandle.require_current
    )

    def track_require_current(self: object, label: str) -> None:
        if label == "Stage 6 terminal recovery evidence immediately before return":
            events.append(("terminal_evidence_current", self))
        original_require_current(self, label)

    monkeypatch.setattr(
        module,
        "_require_stage6_execution_capability",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(module, "_require_review_authorization_current", require_review)
    monkeypatch.setattr(module, "_require_input_pin_current", require_input)
    monkeypatch.setattr(
        module,
        "_verify_stage6_recovery_authorization_bindings",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(module, "_stage6_recovery_workflow_result", construct_result)
    monkeypatch.setattr(
        stage6_terminal_recovery,
        "detect_stage6_terminal_recovery",
        detect,
    )
    monkeypatch.setattr(
        stage6_terminal_recovery,
        "recover_stage6_terminal_commit",
        lambda **kwargs: {"phase_states": stage6_terminal_recovery.COMPLETE_STATES},
    )
    monkeypatch.setattr(
        stage6_terminal_recovery._Stage6EvidenceHandle,
        "require_current",
        track_require_current,
    )

    result = module._run_stage6_workflow_locked(
        config=object(),
        config_path=CONFIG,
        run_id=FORMAL_STAGE6_RUN_ID,
        stage5_gate_path=tmp_path / "stage5-gate.json",
        base=tmp_path,
        run_root=run_root,
        repo=ROOT,
        run_preexisted=True,
        review_authorization_handle=_FakeReviewAuthorizationHandle(),
        stage5_authority=SimpleNamespace(identity={}),
        input_pin=object(),
        execution_identity={},
        execution_capability=object(),
        run_lease=object(),
    )
    events.append(("caller_received", None))

    assert result.summary == {"state": "validated-state"}
    assert [name for name, _ in events] == [
        "result_constructed",
        "review_current",
        "input_pin_current",
        "terminal_evidence_current",
        "caller_received",
    ]
    assert terminal_detection_handle is not None
    assert events[0][1] is terminal_detection_handle
    assert events[3][1] is terminal_detection_handle


@pytest.mark.parametrize(
    ("fail_label", "commit_expected", "result_read_expected"),
    (
        (
            "Stage 6 review authorization before terminal recovery commit",
            False,
            False,
        ),
        (
            "Stage 6 review authorization after terminal recovery commit",
            True,
            False,
        ),
        (
            "Stage 6 review authorization before recovery result return",
            True,
            True,
        ),
    ),
)
def test_terminal_recovery_revalidates_review_around_commit_and_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_label: str,
    commit_expected: bool,
    result_read_expected: bool,
) -> None:
    module = _module()
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    base = tmp_path / "out"
    run_id = "s6-standard-single-r1-20260717T020304Z"
    stage = base / run_id / "s6"
    _write_minimal_handle_capture_fixture(stage)
    events: list[str] = []
    handle = _FakeReviewAuthorizationHandle(
        events=events,
        fail_label=fail_label,
    )
    _allow_review_authorization(monkeypatch, tmp_path=tmp_path, handle=handle)
    monkeypatch.setattr(
        stage6_terminal_recovery,
        "detect_stage6_terminal_recovery",
        lambda **kwargs: {
            "status": "valid_terminal_recovery",
            "reason": "",
            "receipt_identity": {"sha256": "a" * 64, "size_bytes": 1},
            "phase_states": stage6_terminal_recovery.COMPLETE_STATES,
        },
    )
    monkeypatch.setattr(
        module,
        "_verify_stage6_recovery_authorization_bindings",
        lambda **kwargs: events.append("authorization_lineage"),
    )
    monkeypatch.setattr(
        stage6_terminal_recovery,
        "recover_stage6_terminal_commit",
        lambda **kwargs: events.append("commit") or {"phase_states": ()},
    )
    expected = module.Stage6WorkflowResult(
        run_id=run_id,
        stage_root=stage,
        summary={"state": "awaiting_independent_review"},
        routing={"route": "awaiting_independent_review"},
    )
    monkeypatch.setattr(
        module,
        "_stage6_recovery_workflow_result",
        lambda **kwargs: events.append("result_read") or expected,
    )

    with pytest.raises(module.Stage6WorkflowError, match="review.*authorization"):
        module._run_stage6_workflow_for_test(
            config_path=CONFIG,
            run_id=run_id,
            stage5_gate_path=tmp_path / "gate.json",
            base_output_root=base,
            review_authorization_path=tmp_path / "launch-authorization.json",
        )

    assert ("commit" in events) is commit_expected
    assert ("result_read" in events) is result_read_expected


@pytest.mark.parametrize(
    ("fail_label", "commit_expected"),
    (
        ("Stage 6 input pin before terminal recovery commit", False),
        ("Stage 6 input pin after terminal recovery commit", True),
    ),
)
def test_terminal_recovery_revalidates_input_pin_around_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_label: str,
    commit_expected: bool,
) -> None:
    module = _module()
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    base = tmp_path / "out"
    run_id = "s6-standard-single-r1-20260717T020305Z"
    stage = base / run_id / "s6"
    _write_minimal_handle_capture_fixture(stage)
    events: list[str] = []
    pin = _FakeInputPin(events=events, fail_label=fail_label)
    _allow_review_authorization(monkeypatch, tmp_path=tmp_path, input_pin=pin)
    monkeypatch.setattr(
        stage6_terminal_recovery,
        "detect_stage6_terminal_recovery",
        lambda **kwargs: {
            "status": "valid_terminal_recovery",
            "reason": "",
            "receipt_identity": {"sha256": "a" * 64, "size_bytes": 1},
            "phase_states": stage6_terminal_recovery.COMPLETE_STATES,
        },
    )
    monkeypatch.setattr(
        module,
        "_verify_stage6_recovery_authorization_bindings",
        lambda **kwargs: events.append("authorization_lineage"),
    )
    monkeypatch.setattr(
        stage6_terminal_recovery,
        "recover_stage6_terminal_commit",
        lambda **kwargs: events.append("commit") or {"phase_states": ()},
    )
    monkeypatch.setattr(
        module,
        "_stage6_recovery_workflow_result",
        lambda **kwargs: pytest.fail("input drift must fail before result read"),
    )

    with pytest.raises(module.Stage6WorkflowError, match="input pin"):
        module._run_stage6_workflow_for_test(
            config_path=CONFIG,
            run_id=run_id,
            stage5_gate_path=tmp_path / "gate.json",
            base_output_root=base,
            review_authorization_path=tmp_path / "launch-authorization.json",
        )

    assert ("commit" in events) is commit_expected


@pytest.mark.parametrize(
    ("fail_label", "expected_persistent_calls"),
    (
        (
            "Stage 6 review authorization at preterminal recovery entry",
            (),
        ),
        (
            "Stage 6 review authorization before recovery resource segment append",
            (),
        ),
        (
            "Stage 6 review authorization before recovery resource terminal append",
            ("segment",),
        ),
        (
            "Stage 6 review authorization before preterminal recovery commit",
            ("segment", "terminal"),
        ),
        (
            "Stage 6 review authorization after preterminal recovery commit",
            ("segment", "terminal", "commit"),
        ),
    ),
)
def test_preterminal_recovery_revalidates_review_before_and_after_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_execution_authority_factory,
    fail_label: str,
    expected_persistent_calls: tuple[str, ...],
) -> None:
    module = _module()
    from lunar_exploration_ppo.utils import resources
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    active = active_execution_authority_factory()
    stage = Path(active["run_root"]) / "s6"
    stage.mkdir(parents=True)
    events: list[str] = []

    class FakeMonitor:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.running = False
            self.root_pid = os.getpid()
            self.sample_count = 0
            self.latest_process_count = 1
            self.peak_process_count = 1
            self.peak_aggregate_rss_bytes = 1024

        def start(self):
            self.running = True
            self.sample_count = 1
            return self

        def sample_now(self):
            self.sample_count = 2
            return object()

        def stop(self) -> None:
            self.running = False

    monkeypatch.setattr(resources, "ProcessTreeRSSMonitor", FakeMonitor)
    monkeypatch.setattr(
        module,
        "_recovery_resource_gate_from_monitor",
        lambda monitor: {
            "rss_sample_count": monitor.sample_count,
            "passed": True,
        },
    )
    monkeypatch.setattr(
        stage6_terminal_recovery,
        "append_stage6_recovery_resource_segment",
        lambda **kwargs: (
            kwargs["execution_capability"] is active["capability"]
            and events.append("segment")
        ),
    )
    monkeypatch.setattr(
        stage6_terminal_recovery,
        "append_stage6_recovery_resource_terminal",
        lambda **kwargs: (
            kwargs["execution_capability"] is active["capability"]
            and events.append("terminal")
        ),
    )
    monkeypatch.setattr(
        stage6_terminal_recovery,
        "recover_stage6_terminal_commit",
        lambda **kwargs: (
            kwargs["execution_capability"] is active["capability"]
            and events.append("commit")
        ) or {"phase_states": ()},
    )
    handle = active["review_handle"]
    handle.events = events
    handle.fail_label = fail_label
    pin = active["pin"]
    pin.events = events

    with pytest.raises(module.Stage6WorkflowError, match="review.*authorization"):
        module._recover_stage6_preterminal_receipt(
            stage_root=stage,
            repo_root=ROOT,
            review_authorization_handle=handle,
            input_pin=pin,
            execution_capability=active["capability"],
        )

    assert tuple(
        event for event in events if event in {"segment", "terminal", "commit"}
    ) == expected_persistent_calls


@pytest.mark.parametrize(
    ("fail_label", "commit_expected"),
    (
        ("Stage 6 input pin before preterminal recovery commit", False),
        ("Stage 6 input pin after preterminal recovery commit", True),
    ),
)
def test_preterminal_recovery_revalidates_input_pin_around_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_execution_authority_factory,
    fail_label: str,
    commit_expected: bool,
) -> None:
    module = _module()
    from lunar_exploration_ppo.utils import resources
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    active = active_execution_authority_factory()
    stage = Path(active["run_root"]) / "s6"
    stage.mkdir(parents=True)
    events: list[str] = []

    class FakeMonitor:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.running = False
            self.root_pid = os.getpid()
            self.sample_count = 0
            self.latest_process_count = 1
            self.peak_process_count = 1
            self.peak_aggregate_rss_bytes = 1024

        def start(self):
            self.running = True
            self.sample_count = 1
            return self

        def sample_now(self):
            self.sample_count = 2
            return object()

        def stop(self) -> None:
            self.running = False

    monkeypatch.setattr(resources, "ProcessTreeRSSMonitor", FakeMonitor)
    monkeypatch.setattr(
        module,
        "_recovery_resource_gate_from_monitor",
        lambda monitor: {
            "rss_sample_count": monitor.sample_count,
            "passed": True,
        },
    )
    monkeypatch.setattr(
        stage6_terminal_recovery,
        "append_stage6_recovery_resource_segment",
        lambda **kwargs: (
            kwargs["execution_capability"] is active["capability"]
            and events.append("segment")
        ),
    )
    monkeypatch.setattr(
        stage6_terminal_recovery,
        "append_stage6_recovery_resource_terminal",
        lambda **kwargs: (
            kwargs["execution_capability"] is active["capability"]
            and events.append("terminal")
        ),
    )
    monkeypatch.setattr(
        stage6_terminal_recovery,
        "recover_stage6_terminal_commit",
        lambda **kwargs: (
            kwargs["execution_capability"] is active["capability"]
            and events.append("commit")
        ) or {"phase_states": ()},
    )
    pin = active["pin"]
    pin.events = events
    pin.fail_label = fail_label
    handle = active["review_handle"]
    handle.events = events

    with pytest.raises(module.Stage6WorkflowError, match="input pin"):
        module._recover_stage6_preterminal_receipt(
            stage_root=stage,
            repo_root=ROOT,
            review_authorization_handle=handle,
            input_pin=pin,
            execution_capability=active["capability"],
        )

    assert ("commit" in events) is commit_expected


def test_input_pin_lifetime_covers_preflight_training_and_final_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    events: list[str] = []
    pin = _FakeInputPin(events=events)
    _allow_review_authorization(monkeypatch, tmp_path=tmp_path, input_pin=pin)
    _install_stage6_new_run_fakes(module, monkeypatch, events=events)

    module._run_stage6_workflow_for_test(
        config_path=CONFIG,
        run_id="s6-standard-single-r1-20260717T020306Z",
        stage5_gate_path=tmp_path / "gate.json",
        base_output_root=tmp_path / "out",
        review_authorization_path=tmp_path / "launch-authorization.json",
    )

    pin_events = [event for event in events if event.startswith("pin:")]
    assert pin_events == [
        "pin:enter",
        "pin:Stage 6 input pin acquisition complete",
        "pin:Stage 6 input pin before run output side effects",
        "pin:Stage 6 input pin acquisition complete",
        "pin:Stage 6 execution capability issuance",
        "pin:Stage 6 execution capability issued",
        "pin:Stage 6 protected workflow entry",
        "pin:Stage 6 input pin after run lease",
        "pin:Stage 6 input pin after machine preflight before backend import",
        "pin:Stage 6 input pin before training call",
        "pin:Stage 6 input pin after training return",
        "pin:Stage 6 input pin before final workflow return:rehash",
        "pin:exit",
    ]


def test_receipt_only_recovery_first_sample_is_exactly_one_without_sampler_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_execution_authority_factory,
) -> None:
    import shutil

    module = _module()
    from lunar_exploration_ppo.utils import resources
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    active = active_execution_authority_factory()
    stage = Path(active["run_root"]) / "s6"
    stage.mkdir(parents=True)
    events: list[str] = []

    class FakeMonitor:
        def __init__(self, *, interval_seconds: float, **kwargs: object) -> None:
            del kwargs
            assert interval_seconds >= 3600.0
            self.running = False
            self.root_pid = os.getpid()
            self.sample_count = 0
            self.latest_process_count = 1
            self.peak_process_count = 1
            self.peak_aggregate_rss_bytes = 1024

        def start(self):
            self.running = True
            self.sample_count = 1
            events.append("monitor_start:1")
            return self

        def sample_now(self):
            assert self.running is True
            self.sample_count += 1
            self.peak_aggregate_rss_bytes = 2048
            events.append(f"sample_now:{self.sample_count}")
            return object()

        def stop(self) -> None:
            self.running = False
            events.append("monitor_stop")

    monkeypatch.setattr(resources, "ProcessTreeRSSMonitor", FakeMonitor)
    monkeypatch.setattr(
        resources,
        "capture_resource_snapshot",
        lambda **kwargs: pytest.fail(
            "receipt-only first_sample must not call capture_resource_snapshot"
        ),
    )
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda path: type("Usage", (), {"free": 200 * 1024**3})(),
    )

    def append_segment(
        *,
        stage_root,
        first_sample,
        execution_capability,
        planning_child_recovery_capability=None,
    ):
        assert Path(stage_root) == stage
        assert first_sample["rss_sample_count"] == 1
        assert execution_capability is active["capability"]
        assert planning_child_recovery_capability is None
        events.append("segment:1")
        return {"segment_id": "a" * 64, "segment_index": 2}

    def append_terminal(
        *,
        stage_root,
        terminal_resource,
        execution_capability,
        planning_child_recovery_capability=None,
    ):
        assert Path(stage_root) == stage
        assert terminal_resource["rss_sample_count"] == 2
        assert execution_capability is active["capability"]
        assert planning_child_recovery_capability is None
        events.append("terminal:2")
        return {"schema_version": "stage6_terminal_resource_evidence/v4"}

    monkeypatch.setattr(
        stage6_terminal_recovery,
        "append_stage6_recovery_resource_segment",
        append_segment,
    )
    monkeypatch.setattr(
        stage6_terminal_recovery,
        "append_stage6_recovery_resource_terminal",
        append_terminal,
    )
    monkeypatch.setattr(
        stage6_terminal_recovery,
        "recover_stage6_terminal_commit",
        lambda **kwargs: (
            kwargs["execution_capability"] is active["capability"]
            and events.append("pure_commit")
        ) or {"phase_states": ()},
    )

    review_handle = active["review_handle"]
    review_handle.events = events

    module._recover_stage6_preterminal_receipt(
        stage_root=stage,
        repo_root=ROOT,
        review_authorization_handle=review_handle,
        input_pin=active["pin"],
        execution_capability=active["capability"],
    )

    assert events == [
        "review:Stage 6 review authorization at preterminal recovery entry",
        "monitor_start:1",
        "review:Stage 6 review authorization before recovery resource segment append",
        "segment:1",
        "sample_now:2",
        "monitor_stop",
        "review:Stage 6 review authorization before recovery resource terminal append",
        "terminal:2",
        "review:Stage 6 review authorization before preterminal recovery commit",
        "pure_commit",
        "review:Stage 6 review authorization after preterminal recovery commit",
    ]


def test_receipt_only_formal_recovery_survives_segment_crash_and_restarts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins
    import shutil

    module = _module()
    from test_stage6_terminal_recovery import (
        _identity,
        _manifest_committer,
        _write_preterminal_stage,
        _write_receipt,
    )
    from lunar_exploration_ppo.utils import resources

    run_id = FORMAL_STAGE6_RUN_ID
    base = tmp_path / "out"
    stage = base / run_id / "s6"
    preterminal = _write_preterminal_stage(stage, formal_run_id=run_id)
    semantic = preterminal["semantic"]
    global_checkpoint = preterminal["global_checkpoint"]
    artifacts = dict(preterminal["artifacts"])
    assert isinstance(semantic, dict)
    assert isinstance(global_checkpoint, dict)
    artifacts["summary.json"] = module.ArtifactStore.canonical_json_bytes(
        {
            "schema_version": "stage6_summary/v1",
            "machine_passed": True,
            "state": semantic["state"],
            "final_evaluation_count": semantic["final_evaluation_count"],
            "final_episode_count": semantic["final_episode_count"],
            "checkpoint_receipt_count": semantic["checkpoint_receipt_count"],
            "global_best": global_checkpoint["record"],
        }
    )
    artifacts["routing.json"] = module.ArtifactStore.canonical_json_bytes(
        {
            "schema_version": "stage6_routing/v1",
            "machine_passed": True,
            "state": semantic["state"],
            "route": "awaiting_independent_review",
        }
    )
    preterminal["artifacts"] = artifacts
    with _active_execution_capability(
        tmp_path / "receipt-setup-authority",
        monkeypatch,
        run_root=stage.parent,
    ) as setup_authority:
        preterminal["execution_capability"] = setup_authority["capability"]
        _write_receipt(preterminal)
    authorization_calls: list[dict[str, object]] = []
    _allow_review_authorization(
        monkeypatch,
        authorization_calls,
        tmp_path=tmp_path,
    )
    monkeypatch.setattr(
        module,
        "_verify_stage6_recovery_authorization_bindings",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "write_or_verify_stage6_manifest",
        lambda candidate,
        *,
        repo_root,
        planning_child_recovery_capability=None: _manifest_committer(
            Path(candidate)
        ),
    )
    monkeypatch.setattr(
        module,
        "_verify_existing_stage6_manifest_identity",
        lambda candidate,
        *,
        repo_root,
        planning_child_recovery_capability=None: _identity(
            (Path(candidate) / "manifest.json").read_bytes()
        ),
    )

    def verify_test_manifest(
        candidate: Path,
        *,
        source_identity: object,
        repo_root: Path,
        evidence_handle: object,
        planning_child_recovery_capability: object | None = None,
    ) -> dict[str, object]:
        del source_identity
        assert planning_child_recovery_capability is None
        assert Path(candidate) == stage
        assert Path(repo_root) == ROOT.resolve()
        payload = evidence_handle.read_bytes(
            "manifest.json",
            label="Stage 6 test terminal manifest",
        )
        return _identity(payload)

    monkeypatch.setattr(
        module,
        "_verify_stage6_terminal_manifest_evidence",
        verify_test_manifest,
    )
    monkeypatch.setattr(
        module,
        "_run_stage6_machine_preflight",
        lambda **kwargs: pytest.fail("receipt recovery must not run preflight"),
    )
    monkeypatch.setattr(
        module,
        "_verify_stage6_machine_preflight_for_resume_at_root",
        lambda **kwargs: pytest.fail("receipt recovery must not verify preflight"),
    )
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda path: type("Usage", (), {"free": 200 * 1024**3})(),
    )

    fake_pid = {"value": os.getpid() + 100_000}
    monkeypatch.setattr(os, "getpid", lambda: fake_pid["value"])
    monitor_pids: list[int] = []
    crash_next_terminal_sample = {"value": True}

    class FakeMonitor:
        def __init__(self, *, interval_seconds: float, **kwargs: object) -> None:
            del kwargs
            assert interval_seconds >= 3600.0
            self.running = False
            self.root_pid = os.getpid()
            self.sample_count = 0
            self.latest_process_count = 1
            self.peak_process_count = 1
            self.peak_aggregate_rss_bytes = 1024
            monitor_pids.append(self.root_pid)

        def start(self):
            self.running = True
            self.sample_count = 1
            return self

        def sample_now(self):
            if crash_next_terminal_sample["value"]:
                crash_next_terminal_sample["value"] = False
                raise OSError("injected crash after recovery segment")
            self.sample_count = 2
            self.peak_aggregate_rss_bytes = 2048
            return object()

        def stop(self) -> None:
            self.running = False

    monkeypatch.setattr(resources, "ProcessTreeRSSMonitor", FakeMonitor)

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "lunar_exploration_ppo.ppo.standard_training":
            raise AssertionError("receipt recovery imported training backend")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    arguments = {
        "config_path": CONFIG,
        "run_id": run_id,
        "stage5_gate_path": tmp_path / "gate.json",
        "base_output_root": base,
        "review_authorization_path": tmp_path / "launch-authorization.json",
    }
    with pytest.raises(module.Stage6WorkflowError, match="lease|recovery|active"):
        module._run_stage6_workflow_for_test(**arguments)

    rows_after_crash = [
        json.loads(line)
        for line in (stage / "resource_audit.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [row["phase"] for row in rows_after_crash] == [
        "segment_start",
        "segment_start",
    ]
    assert rows_after_crash[-1]["first_sample"]["rss_sample_count"] == 1

    fake_pid["value"] += 1
    second = module._run_stage6_workflow_for_test(**arguments)
    fake_pid["value"] += 1
    third = module._run_stage6_workflow_for_test(**arguments)

    assert second == third
    assert second.summary["state"] == "awaiting_independent_review"
    assert second.routing["route"] == "awaiting_independent_review"
    assert len(authorization_calls) == 3
    assert len(monitor_pids) == 2
    assert len(set(monitor_pids)) == 2
    final_rows = [
        json.loads(line)
        for line in (stage / "resource_audit.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [
        row["first_sample"]["rss_sample_count"]
        for row in final_rows
        if row["phase"] == "segment_start"
    ] == [1, 1, 1]
    assert final_rows[-1]["schema_version"] == (
        "stage6_terminal_resource_evidence/v4"
    )
    assert final_rows[-1]["resource"]["rss_sample_count"] == 2


@pytest.mark.parametrize("run_id", (".", "..", "nested/run"))
def test_workflow_rejects_non_single_component_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_id: str,
) -> None:
    module = _module()
    from lunar_exploration_ppo.ppo import standard_training
    _allow_review_authorization(monkeypatch, tmp_path=tmp_path)

    class Authority:
        identity = {"gate_sha256": "a" * 64}

        def require_current(self, label="authority"):
            return None

    monkeypatch.setattr(
        module,
        "verify_frozen_stage5_authority",
        lambda **kwargs: Authority(),
    )
    monkeypatch.setattr(
        module,
        "_run_stage6_machine_preflight",
        lambda **kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        module,
        "_verify_stage6_machine_preflight_for_resume_at_root",
        lambda **kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        standard_training,
        "execute_standard_training",
        lambda **kwargs: {
            "stage_root": str(Path(kwargs["run_root"]) / "s6"),
            "summary": {},
            "routing": {},
        },
    )

    with pytest.raises(module.Stage6WorkflowError, match="run_id|run root|path"):
        module._run_stage6_workflow_for_test(
            config_path=CONFIG,
            run_id=run_id,
            stage5_gate_path=tmp_path / "gate.json",
            base_output_root=tmp_path / "out",
            review_authorization_path=tmp_path / "launch-authorization.json",
        )


def test_workflow_rejects_linked_run_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    from lunar_exploration_ppo.ppo import standard_training
    _allow_review_authorization(monkeypatch, tmp_path=tmp_path)

    base = tmp_path / "out"
    outside = tmp_path / "outside-run"
    base.mkdir()
    outside.mkdir()
    linked = base / "linked-run"
    try:
        os.symlink(outside, linked, target_is_directory=True)
    except OSError:
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(linked), str(outside)],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            pytest.skip("directory link unavailable")

    class Authority:
        identity = {"gate_sha256": "a" * 64}

        def require_current(self, label="authority"):
            return None

    monkeypatch.setattr(
        module,
        "verify_frozen_stage5_authority",
        lambda **kwargs: Authority(),
    )
    monkeypatch.setattr(
        module,
        "_verify_stage6_machine_preflight_for_resume_at_root",
        lambda **kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        standard_training,
        "execute_standard_training",
        lambda **kwargs: {
            "stage_root": str(Path(kwargs["run_root"]) / "s6"),
            "summary": {},
            "routing": {},
        },
    )

    with pytest.raises(module.Stage6WorkflowError, match="link|reparse|run root"):
        module._run_stage6_workflow_for_test(
            config_path=CONFIG,
            run_id="linked-run",
            stage5_gate_path=tmp_path / "gate.json",
            base_output_root=base,
            review_authorization_path=tmp_path / "launch-authorization.json",
        )


@pytest.mark.parametrize("reparse_component", ("s6", "preflight"))
def test_resume_rejects_reparse_in_stage_chain_before_audit_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reparse_component: str,
) -> None:
    module = _module()
    base = tmp_path / "out"
    run_root = base / "stage6-reparse-resume"
    run_root.mkdir(parents=True)
    outside = tmp_path / f"outside-{reparse_component}"

    if reparse_component == "s6":
        (outside / "preflight").mkdir(parents=True)
        (outside / "preflight/audit.json").write_bytes(b"{}\n")
        _make_directory_reparse(run_root / "s6", outside)
    else:
        (run_root / "s6").mkdir()
        outside.mkdir()
        (outside / "audit.json").write_bytes(b"{}\n")
        _make_directory_reparse(run_root / "s6/preflight", outside)

    class Authority:
        identity = {"gate_sha256": "a" * 64}

        def require_current(self, label="authority"):
            return None

    monkeypatch.setattr(
        module,
        "verify_frozen_stage5_authority",
        lambda **kwargs: Authority(),
    )

    with pytest.raises(module.Stage6WorkflowError, match="link|reparse"):
        module._verify_stage6_machine_preflight_for_resume_for_test(
            config_path=CONFIG,
            run_id=run_root.name,
            stage5_gate_path=tmp_path / "gate.json",
            base_output_root=base,
        )


@pytest.mark.parametrize(
    ("field", "mutated_value"),
    (
        ("schema_version", "stage6_environment_identity/v2"),
        ("python_version", "3.12.14"),
        ("python_implementation", "PyPy"),
        ("os_name", "posix"),
        ("platform_system", "Linux"),
        ("platform_machine", "x86_64"),
        ("numpy_version", "2.3.0"),
        ("torch_version", "2.12.2+cu130"),
        ("torch_cuda_version", "13.1"),
        ("cudnn_version", 91003),
        ("cuda_available", False),
        ("cuda_device_count", 2),
        ("cuda_current_device", -1),
        ("cuda_device_name", "NVIDIA drifted GPU"),
        ("cuda_compute_capability", [12, 1]),
        ("cuda_total_vram_bytes", 16 * 1024**3 + 1),
        ("compute_dtype", "float16"),
        ("amp_enabled", True),
    ),
)
def test_resume_recomputes_environment_identity_and_rejects_runtime_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    mutated_value: object,
) -> None:
    from test_stage6_machine_preflight import _environment_sha256, _valid_audit

    module = _module()
    run_id = "stage6-environment-resume-drift"
    base = tmp_path / "out"
    preflight_root = base / run_id / "s6/preflight"
    preflight_root.mkdir(parents=True)
    audit = _valid_audit()
    audit["config_sha256"] = hashlib.sha256(CONFIG.read_bytes()).hexdigest()

    class Authority:
        identity = {"gate_sha256": audit["stage5_gate_sha256"]}

        @staticmethod
        def require_current(label: str) -> None:
            assert label == "Stage 6 resume preflight authority"

    monkeypatch.setattr(
        module,
        "verify_frozen_stage5_authority",
        lambda **kwargs: Authority(),
    )
    (preflight_root / "audit.json").write_bytes(
        module.ArtifactStore.canonical_json_bytes(audit)
    )
    current_environment = dict(audit["environment_identity"])
    current_environment[field] = mutated_value
    monkeypatch.setattr(
        module,
        "stage6_environment_identity",
        lambda: dict(current_environment),
    )
    assert _environment_sha256(current_environment) != audit["environment_sha256"]

    with pytest.raises(module.Stage6WorkflowError, match="environment"):
        module._verify_stage6_machine_preflight_for_resume_for_test(
            config_path=CONFIG,
            run_id=run_id,
            stage5_gate_path=tmp_path / "stage5-gate.json",
            base_output_root=base,
        )


def test_workflow_rejects_concurrent_owner_via_run_level_os_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    from lunar_exploration_ppo.utils.durable_jsonl import RunLease
    _allow_review_authorization(monkeypatch, tmp_path=tmp_path)

    base = tmp_path / "out"
    run_id = "single-writer-run"
    run_root = base / run_id
    run_root.mkdir(parents=True)
    lease_path = run_root / ".stage6.lease"
    with RunLease(lease_path):
        with pytest.raises(module.Stage6WorkflowError, match="single-writer|active"):
            module._run_stage6_workflow_for_test(
                config_path=CONFIG,
                run_id=run_id,
                stage5_gate_path=tmp_path / "gate.json",
                base_output_root=base,
                review_authorization_path=tmp_path / "launch-authorization.json",
            )

    assert lease_path.is_file()


def test_planning_child_launch_acquires_run_lease_before_capability_issuance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    from lunar_exploration_ppo.utils.durable_jsonl import RunLease

    _allow_review_authorization(monkeypatch, tmp_path=tmp_path)
    base = tmp_path / "out"
    run_id = "planning-child-single-writer-run"
    run_root = base / run_id
    run_root.mkdir(parents=True)
    lease_path = run_root / ".stage6.lease"
    capability_calls: list[dict[str, object]] = []

    def issue_capability(**kwargs: object) -> object:
        capability_calls.append(dict(kwargs))
        raise AssertionError(
            "planning-child capability was issued before the run lease"
        )

    monkeypatch.setattr(
        module,
        "_load_planning_child_recovery_capability_for_workflow",
        issue_capability,
    )

    with RunLease(lease_path):
        with pytest.raises(module.Stage6WorkflowError, match="single-writer|active"):
            module._run_stage6_workflow_for_test(
                config_path=CONFIG,
                run_id=run_id,
                stage5_gate_path=tmp_path / "gate.json",
                base_output_root=base,
                review_authorization_path=(
                    tmp_path / "launch-authorization.json"
                ),
                planning_warm_start_path=tmp_path / "planning-warm-start.json",
                planning_child_source_repair_path=(
                    run_root / "s6" / "planning-child-source-repair.json"
                ),
                planning_child_source_repair_continuation_path=(
                    run_root
                    / "s6"
                    / "planning-child-source-repair-continuation.json"
                ),
            )

    assert capability_calls == []
    assert not (run_root / "s6" / "resource_audit.jsonl").exists()


def test_public_workflow_rejects_output_root_override_argument(tmp_path: Path) -> None:
    module = _module()

    with pytest.raises(TypeError, match="base_output_root"):
        module.run_stage6_workflow(
            config_path=CONFIG,
            run_id=FORMAL_STAGE6_RUN_ID,
            stage5_gate_path=tmp_path / "gate.json",
            review_authorization_path=tmp_path / "launch-authorization.json",
            base_output_root=tmp_path / "alternate-output",
        )
