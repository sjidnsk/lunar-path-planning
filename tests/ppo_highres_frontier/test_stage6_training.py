"""Stage 6 正式训练状态机、best、checkpoint retention 与 hard gate。"""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import os
import subprocess
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from pathlib import Path

import pytest


SEEDS = (20260716,)


def _training_module():
    return importlib.import_module("lunar_exploration_ppo.ppo.standard_training")


def _retention_module():
    return importlib.import_module("lunar_exploration_ppo.ppo.checkpoint_retention")


def _resources_module():
    return importlib.import_module("lunar_exploration_ppo.utils.resources")


def _phase_immutable_bindings(marker: str) -> dict[str, object]:
    from hashlib import sha256

    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from test_stage6_source_repair import _environment_identity

    environment = _environment_identity()
    reviewed_tree = marker * 40
    return {
        "config_sha256": marker * 64,
        "source_set_sha256": marker * 64,
        "prospective_tree_sha256": sha256(
            reviewed_tree.encode("ascii")
        ).hexdigest(),
        "data_sha256": marker * 64,
        "environment_identity": environment,
        "environment_sha256": sha256(
            ArtifactStore.canonical_json_bytes(environment)
        ).hexdigest(),
        "stage5_gate_sha256": marker * 64,
        "formal_run_id": "s6-standard-single-r1-20260724T000124Z",
        "changed_path_set_sha256": marker * 64,
        "review_authorization_record_sha256": marker * 64,
        "authorization_file_sha256": marker * 64,
        "review_identity_sha256": marker * 64,
        "reviewed_prospective_git_tree": reviewed_tree,
        "frozen_diff_sha256": marker * 64,
        "spec_review_sha256": marker * 64,
        "quality_review_sha256": marker * 64,
    }


def _full_phase_immutable_bindings(marker: str) -> dict[str, object]:
    return {
        **_phase_immutable_bindings(marker),
        "coverage_cache_manifest_path": "cache/coverage-cache-manifest.json",
        "coverage_cache_manifest_sha256": marker * 64,
        "coverage_cache_manifest_size_bytes": 1,
        "coverage_cache_root": "cache/coverage-v1",
        "coverage_cache_entry_set_sha256": marker * 64,
        "coverage_cache_runtime_mode": "persistent_exact_manifest_read_only/v1",
        "coverage_cache_formal_audit_sha256": marker * 64,
        "coverage_cache_formal_audit_size_bytes": 1,
    }


def _journal_cutover_fixture(
    tmp_path: Path,
    historical_immutable_bindings: dict[str, object],
):
    module = _training_module()
    historical_transaction = module.StandardTrainingTransaction(
        sequence=83,
        seed=SEEDS[0],
        seed_index=0,
        update=84,
        validation_episodes=0,
        commit_states=(f"seed_{SEEDS[0]}_training_update_84",),
    )
    current_transaction = module.StandardTrainingTransaction(
        sequence=84,
        seed=SEEDS[0],
        seed_index=0,
        update=85,
        validation_episodes=0,
        commit_states=(f"seed_{SEEDS[0]}_training_update_85",),
    )
    checkpoint = module.StandardTransactionCheckpoint(
        transaction_key=historical_transaction.key,
        seed=historical_transaction.seed,
        update=historical_transaction.update,
        checkpoint_sha256="a" * 64,
        complete_marker_sha256="b" * 64,
        policy_state_sha256="c" * 64,
    )
    receipt_index = module.CheckpointReceiptIndex(tmp_path / "receipts.jsonl")
    assert receipt_index.append_once(checkpoint)
    records = tuple(
        {
            "state": state,
            "bindings": {
                **historical_immutable_bindings,
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
            },
        }
        for state in historical_transaction.commit_states
    )
    return (
        module,
        (historical_transaction, current_transaction),
        records,
        receipt_index.verify(),
    )


def _final_aggregate_proxy_safety_fixture():
    from dataclasses import replace
    from hashlib import sha256

    import numpy as np

    from lunar_exploration_ppo.configs.stage6 import (
        SafetyContract,
        load_stage6_config,
    )
    from lunar_exploration_ppo.eval.metrics import summarize_episodes
    from lunar_exploration_ppo.eval.standard import STANDARD_EVALUATION_METHODS
    from test_stage6_execution import CONFIG, _deterministic_final_summary

    config = load_stage6_config(CONFIG)
    config_sha256 = sha256(CONFIG.read_bytes()).hexdigest()
    verified: dict[str, dict[str, object]] = {}
    isolation: dict[str, dict[str, object]] = {}
    for split in ("test", "unseen"):
        for method in STANDARD_EVALUATION_METHODS:
            key = f"{split}:{method}"
            summary = _deterministic_final_summary(method, split, "e" * 64)
            episodes = list(summary.episodes)
            if key == "test:random_valid_frontier":
                episodes[:3] = [
                    replace(
                        episode,
                        safety_violation_count=1,
                        termination_reason="safety_done",
                    )
                    for episode in episodes[:3]
                ]
            metrics, bootstrap = summarize_episodes(
                episodes,
                bootstrap_resamples=config.evaluation.bootstrap_resamples,
                bootstrap_seed=config.evaluation.bootstrap_seed,
            )
            termination_classification: dict[str, int] = {}
            for episode in episodes:
                termination_classification[episode.termination_reason] = (
                    termination_classification.get(episode.termination_reason, 0) + 1
                )
            verified[key] = {
                "result": {
                    "episode_count": 64,
                    "metrics": metrics,
                    "bootstrap_audit": bootstrap,
                    "fairness_audit": dict(summary.fairness_audit),
                },
                "schedule": tuple(
                    (
                        episode.scenario_key,
                        episode.scenario_seed,
                        episode.terrain_seed,
                        episode.start_pose_seed,
                        episode.evaluation_seed,
                    )
                    for episode in episodes
                ),
                "coverage_curve": tuple(
                    float(
                        np.mean(
                            [episode.coverage_curve[step] for episode in episodes],
                            dtype=np.float64,
                        )
                    )
                    for step in range(129)
                ),
                "termination_classification": termination_classification,
                "safety_violation_count": sum(
                    episode.safety_violation_count for episode in episodes
                ),
                "planner_failure_counts": {
                    "endpoint_physical_unsafe": 0,
                    "endpoint_unknown_buffer_unsafe": 0,
                    "path_physical_unsafe": 0,
                    "path_unknown_buffer_unsafe": 0,
                    "planner_no_path": 0,
                },
                "reset_scan_audit": {
                    "schema_version": "stage6_reset_scan_replay/v1",
                    "episode_count": 64,
                    "scan_order": [
                        "reset_local_safety",
                        "reset_exploration",
                    ],
                    "dual_scan_episode_count": 64,
                },
                "episode_count": 64,
            }
            isolation[key] = {"passed": True}
    return (
        verified,
        isolation,
        SafetyContract.from_stage6_config(config),
        config_sha256,
    )


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


def test_final_aggregate_keeps_proxy_safety_as_adverse_nonblocking_evidence() -> None:
    from copy import deepcopy

    module = _training_module()
    verified, isolation, safety_contract, config_sha256 = (
        _final_aggregate_proxy_safety_fixture()
    )

    aggregate = module.build_standard_final_aggregate_artifacts(
        verified,
        isolation,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )

    failure = aggregate["failure_audit"]
    assert failure["schema_version"] == "stage6_failure_audit/v3"
    assert failure["passed"] is True
    assert failure["episode_count"] == 640
    assert failure["safety_violation_count"] == 3
    assert failure["safety_violation_count_by_split"] == {
        "test": 3,
        "unseen": 0,
    }
    assert failure["planner_failure_counts"] == {
        "endpoint_physical_unsafe": 0,
        "endpoint_unknown_buffer_unsafe": 0,
        "path_physical_unsafe": 0,
        "path_unknown_buffer_unsafe": 0,
        "planner_no_path": 0,
    }
    assert failure["reset_scan_audit"] == {
        "schema_version": "stage6_final_reset_scan_audit/v1",
        "episode_count": 640,
        "scan_order": [
            "reset_local_safety",
            "reset_exploration",
        ],
        "dual_scan_episode_count": 640,
        "passed": True,
    }
    safety_row = next(
        row
        for row in failure["evaluations"]
        if row["split"] == "test" and row["method"] == "random_valid_frontier"
    )
    assert safety_row["safety_violation_count"] == 3
    assert safety_row["termination_classification"]["safety_done"] == 3
    assert failure["machine_blocking_findings"] == []
    assert failure["adverse_findings"] == [
        {
            "code": "synthetic_proxy_safety_termination_observed",
            "count": 3,
            "machine_blocking": False,
        }
    ]
    assert failure["safety_interpretation"] == {
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "physical_obstacle_cells_written": False,
        "physical_safety_claim": False,
        "claim_scope": "no_physical_safety_claim/v1",
    }

    count_drift = deepcopy(verified)
    count_drift["test:ppo_policy"]["episode_count"] = 63
    with pytest.raises(module.StandardTrainingError):
        module.build_standard_final_aggregate_artifacts(
            count_drift,
            isolation,
            safety_contract=safety_contract,
            config_sha256=config_sha256,
        )

    isolation_drift = deepcopy(isolation)
    isolation_drift["test:ppo_policy"]["passed"] = False
    with pytest.raises(module.StandardTrainingError):
        module.build_standard_final_aggregate_artifacts(
            verified,
            isolation_drift,
            safety_contract=safety_contract,
            config_sha256=config_sha256,
        )

    fairness_drift = deepcopy(verified)
    fairness_drift["test:ppo_policy"]["result"]["fairness_audit"]["passed"] = False
    with pytest.raises(module.StandardTrainingError):
        module.build_standard_final_aggregate_artifacts(
            fairness_drift,
            isolation,
            safety_contract=safety_contract,
            config_sha256=config_sha256,
        )

    leakage_drift = deepcopy(verified)
    leakage_drift["test:ppo_policy"]["result"]["fairness_audit"][
        "decision_audit"
    ]["runtime_truth_like_selector_field_count"] = 1
    with pytest.raises(module.StandardTrainingError):
        module.build_standard_final_aggregate_artifacts(
            leakage_drift,
            isolation,
            safety_contract=safety_contract,
            config_sha256=config_sha256,
        )

    reset_drift = deepcopy(verified)
    reset_drift["test:ppo_policy"]["reset_scan_audit"][
        "dual_scan_episode_count"
    ] = 63
    with pytest.raises(module.StandardTrainingError, match="reset"):
        module.build_standard_final_aggregate_artifacts(
            reset_drift,
            isolation,
            safety_contract=safety_contract,
            config_sha256=config_sha256,
        )

    unknown_reason = deepcopy(verified)
    unknown_reason["test:ppo_policy"]["planner_failure_counts"] = {
        "unexpected_reason": 1
    }
    with pytest.raises(module.StandardTrainingError, match="failure"):
        module.build_standard_final_aggregate_artifacts(
            unknown_reason,
            isolation,
            safety_contract=safety_contract,
            config_sha256=config_sha256,
        )


def test_single_seed_state_machine_is_complete_and_resumable() -> None:
    module = _training_module()
    assert module.FROZEN_SEEDS == SEEDS
    machine = module.StandardTrainingStateMachine(
        seeds=SEEDS,
        updates_per_seed=100,
        validation_every_updates=10,
        validation_episodes=16,
    )

    units = machine.units
    training = [unit for unit in units if unit.kind == "train"]
    validations = [unit for unit in units if unit.kind == "validate"]
    assert len(training) == 100
    assert len(validations) == 10
    assert sum(unit.episode_count for unit in validations) == 160
    assert [unit.seed for unit in training] == [SEEDS[0]] * 100
    assert [unit.update for unit in training] == list(range(1, 101))
    assert [unit.update for unit in validations] == list(range(10, 101, 10))
    assert units[-1].state == "seed_20260716_complete"

    completed = [unit.key for unit in units[:37]]
    remaining = machine.remaining_after(completed)
    assert remaining[0] == units[37]
    assert not set(completed).intersection(unit.key for unit in remaining)
    with pytest.raises(module.StandardTrainingError, match="prefix"):
        machine.remaining_after([units[1].key])


def test_seed_and_global_best_use_frozen_deterministic_tie_breaks() -> None:
    module = _training_module()
    records = [
        module.ValidationRecord(SEEDS[0], 20, 0.50, 0.80, "s0-u20"),
        module.ValidationRecord(SEEDS[0], 10, 0.50, 0.80, "s0-u10"),
        module.ValidationRecord(SEEDS[0], 30, 0.50, 0.81, "s0-u30"),
    ]
    assert module.select_seed_best(records).checkpoint_ref == "s0-u30"
    records[-1] = module.ValidationRecord(SEEDS[0], 30, 0.50, 0.80, "s0-u30")
    assert module.select_seed_best(records).checkpoint_ref == "s0-u10"

    seed_best = module.ValidationRecord(SEEDS[0], 20, 0.75, 0.90, "seed-best")
    assert module.select_global_best([seed_best], configured_seeds=SEEDS) is seed_best

    invalid_inputs = (
        ((), SEEDS),
        ((seed_best,), ()),
        ((seed_best, SimpleNamespace(seed=20260717)), SEEDS),
        (
            (SimpleNamespace(seed=20260717), seed_best),
            (20260717, 20260716),
        ),
    )
    for records, configured_seeds in invalid_inputs:
        with pytest.raises(module.StandardTrainingError, match="completed seeds in order"):
            module.select_global_best(records, configured_seeds=configured_seeds)


def test_audit_schedule_math_and_checkpoint_payload_fail_closed() -> None:
    module = _training_module()
    assert [update for update in range(1, 101) if module.audit_required(update)] == [
        1,
        10,
        50,
        100,
    ]
    valid = {
        "observation_finite": True,
        "action_finite": True,
        "logprob_finite": True,
        "value_finite": True,
        "advantage_finite": True,
        "return_finite": True,
        "ratio_finite": True,
        "loss_finite": True,
        "kl_finite": True,
        "grad_finite": True,
        "mask_violation_count": 0,
        "snapshot_mismatch_count": 0,
        "stale_policy_transition_count": 0,
        "initial_ratio_max_abs_error": 1.9073e-6,
        "device": "cuda",
        "allowed_initial_ratio_tolerance": 1e-5,
        "compute_dtype": "float32",
        "joint_logprob": True,
        "joint_logprob_factorization_max_abs_error": 0.0,
        "grad_post_clip_norm_max": 0.500001,
        "evidence_source": "ppo_update_math_evidence/v1",
        "evidence_sha256": "a" * 64,
        "sample_count": 1024,
        "snapshot_list_sha256": "b" * 64,
        "initial_forward_sample_count": 1024,
        "forward_sample_count": 2048,
        "loss_sample_count": 1024,
        "gradient_step_count": 4,
    }
    assert module.validate_math_audit(valid)["passed"] is True
    with pytest.raises(module.StandardTrainingError, match="nonfinite"):
        module.validate_math_audit({**valid, "loss_finite": False})
    with pytest.raises(module.StandardTrainingError, match="stale"):
        module.validate_math_audit({**valid, "stale_policy_transition_count": 1})
    with pytest.raises(module.StandardTrainingError, match="grad"):
        module.validate_math_audit({**valid, "grad_post_clip_norm_max": 0.500002})
    assert module.validate_math_audit(
        {
            **valid,
            "device": "cpu",
            "allowed_initial_ratio_tolerance": 1e-6,
            "initial_ratio_max_abs_error": 1e-6,
        }
    )["passed"] is True
    with pytest.raises(module.StandardTrainingError, match="ratio"):
        module.validate_math_audit(
            {
                **valid,
                "device": "cpu",
                "allowed_initial_ratio_tolerance": 1e-6,
            }
        )
    with pytest.raises(module.StandardTrainingError, match="tolerance"):
        module.validate_math_audit(
            {**valid, "allowed_initial_ratio_tolerance": 1e-6}
        )

    payload = {
        "normalization_stats": {},
        "scenario_sampler_state": {"cursor": 1},
        "vector_env_states": tuple({"worker": index} for index in range(8)),
        "best_record": {},
        "config_sha256": "a" * 64,
        "lineage": {"stage5_gate_sha256": "b" * 64},
        "eval_metrics": {},
    }
    assert module.validate_checkpoint_runtime_payload(payload)["vector_env_state_count"] == 8
    with pytest.raises(module.StandardTrainingError, match="eight vector-env"):
        module.validate_checkpoint_runtime_payload(
            {**payload, "vector_env_states": payload["vector_env_states"][:-1]}
        )


@pytest.mark.parametrize(
    ("preflight", "disk_bytes", "rss_bytes", "vram_bytes", "warning", "hard_stop"),
    [
        (True, 100 * 1024**3, 0, 0, False, False),
        (True, 100 * 1024**3 - 1, 0, 0, False, True),
        (False, 50 * 1024**3, 0, 0, False, False),
        (False, 50 * 1024**3 - 1, 0, 0, False, True),
        (False, 100 * 1024**3, 16 * 1024**3, 0, True, False),
        (False, 100 * 1024**3, 20 * 1024**3, 0, True, True),
        (False, 100 * 1024**3, 20 * 1024**3 + 1, 0, True, True),
        (False, 100 * 1024**3, 0, 9 * 1024**3, False, False),
        (False, 100 * 1024**3, 0, 9 * 1024**3 + 1, True, False),
        (False, 100 * 1024**3, 0, int(10.1 * 1024**3), True, True),
        (False, 100 * 1024**3, 0, int(10.1 * 1024**3) + 1, True, True),
    ],
)
def test_resource_gate_exact_boundaries(
    preflight: bool,
    disk_bytes: int,
    rss_bytes: int,
    vram_bytes: int,
    warning: bool,
    hard_stop: bool,
) -> None:
    module = _resources_module()
    decision = module.evaluate_resource_gates(
        module.ResourceSnapshot(
            d_free_bytes=disk_bytes,
            rss_bytes=rss_bytes,
            peak_vram_bytes=vram_bytes,
        ),
        preflight=preflight,
    )
    assert bool(decision.warnings) is warning
    assert bool(decision.hard_stops) is hard_stop


def _checkpoint_dir(root: Path, update: int, *, extra: bool = False) -> Path:
    directory = root / f"update-{update:08d}"
    directory.mkdir(parents=True)
    for name in ("checkpoint.pt", "manifest.json", "complete.json"):
        (directory / name).write_bytes(name.encode("ascii"))
    if extra:
        (directory / "unknown.bin").write_bytes(b"unknown")
    return directory


_RETENTION_MEMBER_ORDER = ("checkpoint.pt", "manifest.json", "complete.json")


def _write_latest_index(module, root: Path, update: int) -> None:
    manifest_sha256 = "a" * 64
    checkpoint_sha256 = "b" * 64
    complete = {
        "schema_version": "test_complete/v1",
        "update_step": update,
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    latest = {
        "schema_version": module.CHECKPOINT_LATEST_SCHEMA_VERSION,
        "update_step": update,
        "directory": f"update-{update:08d}",
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    (root / f"update-{update:08d}" / "complete.json").write_bytes(
        module.ArtifactStore.canonical_json_bytes(complete)
    )
    (root / "latest.json").write_bytes(
        module.ArtifactStore.canonical_json_bytes(latest)
    )


def _transaction_dir(root: Path, name: str, member_count: int) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    for member in _RETENTION_MEMBER_ORDER[:member_count]:
        (directory / member).write_bytes(member.encode("ascii"))
    return directory


@pytest.mark.parametrize(
    ("crash_phase", "remaining_members"),
    [
        ("after_update_tombstone_rename", _RETENTION_MEMBER_ORDER),
        ("before_transaction_member_unlink:checkpoint.pt", _RETENTION_MEMBER_ORDER),
        (
            "after_transaction_member_unlink:checkpoint.pt",
            ("manifest.json", "complete.json"),
        ),
        (
            "before_transaction_member_unlink:manifest.json",
            ("manifest.json", "complete.json"),
        ),
        ("after_transaction_member_unlink:manifest.json", ("complete.json",)),
        ("before_transaction_member_unlink:complete.json", ("complete.json",)),
        ("after_transaction_member_unlink:complete.json", ()),
    ],
)
def test_checkpoint_retention_recovers_every_atomic_delete_crash_window(
    tmp_path: Path,
    crash_phase: str,
    remaining_members: tuple[str, ...],
) -> None:
    module = _retention_module()
    root = tmp_path / "retention-crash-recovery"
    for update in range(1, 7):
        _checkpoint_dir(root, update)
    _write_latest_index(module, root, 6)
    crashed = False

    def hard_crash(phase: str) -> None:
        nonlocal crashed
        if not crashed and phase == crash_phase:
            crashed = True
            raise RuntimeError("injected hard crash")

    manager = module.CheckpointRetentionManager(root, fault_injector=hard_crash)
    with pytest.raises(RuntimeError, match="injected hard crash"):
        manager.apply(
            latest_update=6,
            periodic_updates=(2, 4),
            best_update=3,
            periodic_keep_count=5,
        )
    assert crashed is True
    assert not (root / "update-00000001").exists()
    tombstones = tuple(
        path
        for path in root.iterdir()
        if path.name.startswith(".retention-tombstone-update-00000001-")
    )
    assert len(tombstones) == 1
    assert {path.name for path in tombstones[0].iterdir()} == set(remaining_members)

    recovered = module.CheckpointRetentionManager(root)
    assert not tombstones[0].exists()
    receipt = recovered.apply(
        latest_update=6,
        periodic_updates=(2, 4),
        best_update=3,
        periodic_keep_count=5,
    )
    verified = recovered.verify_retained(
        latest_update=6,
        expected_updates=(2, 3, 4, 6),
    )

    assert receipt.kept_updates == (2, 3, 4, 6)
    assert receipt.removed_updates == (5,)
    assert verified.kept_updates == (2, 3, 4, 6)
    assert {path.name for path in root.iterdir()} == {
        "latest.json",
        "update-00000002",
        "update-00000003",
        "update-00000004",
        "update-00000006",
    }


@pytest.mark.parametrize("member_count", (0, 1, 2, 3))
def test_checkpoint_retention_startup_recovers_legal_partial_pending_update(
    tmp_path: Path,
    member_count: int,
) -> None:
    module = _retention_module()
    root = tmp_path / "retention-partial-pending"
    _checkpoint_dir(root, 1)
    _checkpoint_dir(root, 2)
    _write_latest_index(module, root, 2)
    pending = _transaction_dir(
        root,
        f".pending-update-00000003-{member_count:032x}",
        member_count,
    )

    manager = module.CheckpointRetentionManager(root)

    assert not pending.exists()
    assert manager.verify_retained(
        latest_update=2,
        expected_updates=(1, 2),
    ).kept_updates == (1, 2)


@pytest.mark.parametrize("entrypoint", ("apply", "verify"))
def test_checkpoint_retention_entrypoints_recover_pending_created_after_startup(
    tmp_path: Path,
    entrypoint: str,
) -> None:
    module = _retention_module()
    root = tmp_path / f"retention-{entrypoint}-pending"
    _checkpoint_dir(root, 1)
    _checkpoint_dir(root, 2)
    _write_latest_index(module, root, 2)
    manager = module.CheckpointRetentionManager(root)
    pending = _transaction_dir(
        root,
        ".pending-update-00000003-11111111111111111111111111111111",
        2,
    )

    if entrypoint == "apply":
        receipt = manager.apply(
            latest_update=2,
            periodic_updates=(1,),
            best_update=1,
            periodic_keep_count=5,
        )
    else:
        receipt = manager.verify_retained(
            latest_update=2,
            expected_updates=(1, 2),
        )

    assert not pending.exists()
    assert receipt.kept_updates == (1, 2)


def test_checkpoint_retention_recovery_rejects_unknown_transaction_name(
    tmp_path: Path,
) -> None:
    module = _retention_module()
    root = tmp_path / "retention-unknown-transaction"
    unknown = _transaction_dir(
        root,
        ".pending-update-00000001-not-a-uuid",
        1,
    )

    with pytest.raises(module.CheckpointRetentionError, match="unknown"):
        module.CheckpointRetentionManager(root)

    assert (unknown / "checkpoint.pt").is_file()


def test_checkpoint_retention_recovery_preflights_all_members_before_deleting(
    tmp_path: Path,
) -> None:
    module = _retention_module()
    root = tmp_path / "retention-injected-member"
    valid = _transaction_dir(
        root,
        ".pending-update-00000001-00000000000000000000000000000000",
        1,
    )
    injected = _transaction_dir(
        root,
        ".pending-update-00000002-ffffffffffffffffffffffffffffffff",
        3,
    )
    (injected / "unknown.bin").write_bytes(b"injected")

    with pytest.raises(module.CheckpointRetentionError, match="unknown"):
        module.CheckpointRetentionManager(root)

    assert (valid / "checkpoint.pt").is_file()
    assert {path.name for path in injected.iterdir()} == {
        *_RETENTION_MEMBER_ORDER,
        "unknown.bin",
    }


@pytest.mark.parametrize("reparse_kind", ("directory", "member"))
def test_checkpoint_retention_recovery_rejects_transaction_reparse_before_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reparse_kind: str,
) -> None:
    module = _retention_module()
    root = tmp_path / f"retention-transaction-{reparse_kind}"
    pending = _transaction_dir(
        root,
        ".pending-update-00000001-22222222222222222222222222222222",
        3,
    )
    flagged = pending if reparse_kind == "directory" else pending / "checkpoint.pt"
    original = module._is_link_or_reparse
    monkeypatch.setattr(
        module,
        "_is_link_or_reparse",
        lambda path: path == flagged or original(path),
    )

    with pytest.raises(module.CheckpointRetentionError, match="link|reparse"):
        module.CheckpointRetentionManager(root)

    assert {path.name for path in pending.iterdir()} == set(_RETENTION_MEMBER_ORDER)


def test_checkpoint_retention_recovery_rechecks_member_identity_before_unlink(
    tmp_path: Path,
) -> None:
    module = _retention_module()
    root = tmp_path / "retention-member-identity"
    _checkpoint_dir(root, 1)
    _write_latest_index(module, root, 1)
    pending_name = ".pending-update-00000002-33333333333333333333333333333333"
    pending = root / pending_name
    displaced = tmp_path / "displaced-pending-checkpoint.pt"

    def replace_member(phase: str) -> None:
        if phase == "before_transaction_member_unlink:checkpoint.pt":
            member = pending / "checkpoint.pt"
            member.replace(displaced)
            member.write_bytes(b"replacement")

    manager = module.CheckpointRetentionManager(root, fault_injector=replace_member)
    _transaction_dir(root, pending_name, 1)

    with pytest.raises(module.CheckpointRetentionError, match="identity changed"):
        manager.verify_retained(latest_update=1, expected_updates=(1,))

    assert displaced.is_file()
    assert (pending / "checkpoint.pt").read_bytes() == b"replacement"


def test_checkpoint_retention_recovery_rechecks_directory_identity_before_rmdir(
    tmp_path: Path,
) -> None:
    module = _retention_module()
    root = tmp_path / "retention-directory-identity"
    _checkpoint_dir(root, 1)
    _write_latest_index(module, root, 1)
    pending_name = ".pending-update-00000002-44444444444444444444444444444444"
    pending = root / pending_name
    displaced = tmp_path / "displaced-pending-directory"

    def replace_directory(phase: str) -> None:
        if phase == "before_transaction_directory_rmdir":
            pending.replace(displaced)
            pending.mkdir()

    manager = module.CheckpointRetentionManager(root, fault_injector=replace_directory)
    _transaction_dir(root, pending_name, 0)

    with pytest.raises(module.CheckpointRetentionError, match="identity changed"):
        manager.verify_retained(latest_update=1, expected_updates=(1,))

    assert displaced.is_dir()
    assert pending.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-bound namespace contract")
def test_checkpoint_retention_member_unlink_binds_validated_file_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.utils.path_security as path_security

    module = _retention_module()
    root = tmp_path / "retention-member-handle-bound"
    _checkpoint_dir(root, 1)
    _write_latest_index(module, root, 1)
    manager = module.CheckpointRetentionManager(root)
    pending = _transaction_dir(
        root,
        ".pending-update-00000002-55555555555555555555555555555555",
        1,
    )
    member = pending / "checkpoint.pt"
    displaced = tmp_path / "retention-handle-bound-checkpoint.pt"
    boundary_observed = False
    replacement_blocked = False

    def attempt_replacement(
        event: str,
        event_source: Path,
        event_destination: Path | None,
    ) -> None:
        nonlocal boundary_observed, replacement_blocked
        if (
            event != "after_source_handle_bound"
            or event_source != member
            or event_destination is not None
        ):
            return
        boundary_observed = True
        try:
            os.replace(event_source, displaced)
        except OSError:
            replacement_blocked = True

    monkeypatch.setattr(
        path_security,
        "_namespace_event",
        attempt_replacement,
        raising=False,
    )

    receipt = manager.verify_retained(latest_update=1, expected_updates=(1,))

    assert receipt.kept_updates == (1,)
    assert boundary_observed is True
    assert replacement_blocked is True
    assert not pending.exists()
    assert not displaced.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-bound namespace contract")
def test_checkpoint_retention_final_rmdir_binds_validated_directory_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.utils.path_security as path_security

    module = _retention_module()
    root = tmp_path / "retention-rmdir-handle-bound"
    _checkpoint_dir(root, 1)
    _write_latest_index(module, root, 1)
    manager = module.CheckpointRetentionManager(root)
    pending = _transaction_dir(
        root,
        ".pending-update-00000002-66666666666666666666666666666666",
        0,
    )
    displaced = tmp_path / "retention-handle-bound-directory"
    boundary_observed = False
    replacement_blocked = False

    def attempt_replacement(
        event: str,
        event_source: Path,
        event_destination: Path | None,
    ) -> None:
        nonlocal boundary_observed, replacement_blocked
        if (
            event != "after_source_handle_bound"
            or event_source != pending
            or event_destination is not None
        ):
            return
        boundary_observed = True
        try:
            os.replace(event_source, displaced)
        except OSError:
            replacement_blocked = True

    monkeypatch.setattr(
        path_security,
        "_namespace_event",
        attempt_replacement,
        raising=False,
    )

    receipt = manager.verify_retained(latest_update=1, expected_updates=(1,))

    assert receipt.kept_updates == (1,)
    assert boundary_observed is True
    assert replacement_blocked is True
    assert not pending.exists()
    assert not displaced.exists()


def test_checkpoint_retention_keeps_latest_periodic_best_and_rejects_unknown(
    tmp_path: Path,
) -> None:
    module = _retention_module()
    root = tmp_path / "retention"
    for update in range(1, 7):
        _checkpoint_dir(root, update)
    manager = module.CheckpointRetentionManager(root)

    receipt = manager.apply(
        latest_update=6,
        periodic_updates=(2, 4),
        best_update=3,
        periodic_keep_count=5,
    )

    assert receipt.kept_updates == (2, 3, 4, 6)
    assert receipt.removed_updates == (1, 5)
    assert {path.name for path in root.iterdir()} == {
        "update-00000002",
        "update-00000003",
        "update-00000004",
        "update-00000006",
    }

    unknown = _checkpoint_dir(root, 7, extra=True)
    with pytest.raises(module.CheckpointRetentionError, match="unknown"):
        manager.apply(
            latest_update=6,
            periodic_updates=(2, 4),
            best_update=3,
            periodic_keep_count=5,
        )
    assert (unknown / "checkpoint.pt").is_file()


def _planning_child_run_update_guard_fixture(
    tmp_path: Path,
) -> tuple[object, object, object, dict[str, object]]:
    from lunar_exploration_ppo.configs.stage6 import (
        SafetyContract,
        load_stage6_config,
    )
    from test_stage6_execution import CONFIG

    module = _training_module()
    config = load_stage6_config(CONFIG)
    transaction = module.StandardTrainingTransaction(
        sequence=11,
        seed=SEEDS[0],
        seed_index=0,
        update=86,
        validation_episodes=0,
        commit_states=(f"seed_{SEEDS[0]}_training_update_86",),
    )
    backend = object.__new__(module.StandardProductionBackend)
    backend._execution_operation = lambda _label: nullcontext()
    backend._immutable_bindings = {"formal_run_id": "guard-fixture"}
    backend._planning_warm_start = object()
    backend._planning_child_recovery_capability = (
        _planning_child_recovery_capability(tmp_path)
    )
    backend.config = config
    backend.config_sha256 = "a" * 64
    backend.stage_root = tmp_path / "s6"
    backend.run_root = tmp_path
    backend.transactions = (transaction,)
    backend.receipt_index = object()
    backend.journal = object()
    backend._next_resource_attempt = lambda _key: 1
    backend._new_runtime_resource_latch = lambda **_kwargs: object()
    backend._poll_runtime_resource_latch = lambda _latch, _boundary: {
        "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
        "rss_root_pid": os.getpid(),
        "rss_sample_count": 1,
        "rss_bytes": 1,
        "passed": True,
    }
    backend._append_resource_phase = lambda **_kwargs: None
    backend._safety_contract = lambda: SafetyContract.from_stage6_config(
        config
    )
    backend._checkpoint_lineage = lambda _update: {}
    backend._journal_binding_split = lambda: (None, None)
    runtime = {
        "seed": transaction.seed,
        "collector": object(),
        "trainer": object(),
        "checkpoint_manager": object(),
        "policy": object(),
        "optimizer": object(),
        "normalization_stats": {},
        "checkpoint_root": tmp_path / "checkpoints",
        "best_record": {},
        "continue_from_current_state": True,
    }
    return module, backend, transaction, runtime


def _planning_child_recovery_capability(
    tmp_path: Path,
) -> object:
    from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
        PlanningChildLineageEpoch,
        PlanningChildRecoveryCapability,
        PlanningChildResumeCursor,
    )

    return PlanningChildRecoveryCapability(
        formal_run_id="s6-standard-single-r1-20260724T000124Z",
        seed=SEEDS[0],
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
        input_pin_requests=(("capability", tmp_path / "bound.json"),),
        acceptance_binding={
            "schema_version": "test-capability",
            "journal_binding_epochs": (
                ("0001:test:update:075", {"epoch": "origin"}),
                ("0010:test:update:085", {"epoch": "parent"}),
                ("0011:test:update:086", {"epoch": "continuation"}),
            ),
            "checkpoint_lineage_templates": {
                "origin": {
                    "lineage_generation": "origin",
                },
                "parent": {
                    "lineage_generation": "parent",
                    "planning_child_source_repair": {
                        "epoch": "parent",
                    },
                },
                "continuation": {
                    "lineage_generation": "continuation",
                    "planning_child_source_repair": {
                        "epoch": "continuation",
                    },
                },
            },
        },
    )


@contextmanager
def _same_size_mutable_stage6_input_pin(tmp_path: Path):
    from lunar_exploration_ppo.utils.stage6_input_pinning import (
        acquire_stage6_input_pin,
    )

    path = tmp_path / "bound-input.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    original = b"planning-child-input"
    mutated = b"Planning-child-input"
    assert len(mutated) == len(original)
    path.write_bytes(original)
    with acquire_stage6_input_pin((("planning-child-input", path),)) as pin:
        resource = pin._resources[0]
        os.close(resource.descriptor)
        resource.descriptor = resource.parent_guard.open_file(
            resource.path,
            os.O_RDWR,
        )

        def mutate() -> None:
            os.lseek(resource.descriptor, 0, os.SEEK_SET)
            assert os.write(resource.descriptor, mutated) == len(mutated)
            os.fsync(resource.descriptor)

        try:
            yield pin, mutate
        finally:
            os.lseek(resource.descriptor, 0, os.SEEK_SET)
            assert os.write(resource.descriptor, original) == len(original)
            os.fsync(resource.descriptor)


def _f3_collection_audit(policy_sha256: str) -> dict[str, object]:
    return {
        "schema_version": "stage4_spawn_collector/v1",
        "worker_pids": list(range(400001, 400009)),
        "worker_start_methods": ["spawn"] * 8,
        "per_env_trainable_counts": [128] * 8,
        "diagnostic_reset_counts": [0] * 8,
        "trainable_transition_count": 1024,
        "inference_pids": [os.getpid()],
        "inference_batch_count": 128,
        "policy_device": "cuda:0",
        "policy_state_sha256": policy_sha256,
        "snapshot_sha256": [
            hashlib.sha256(f"f3-snapshot-{index}".encode("ascii")).hexdigest()
            for index in range(1024)
        ],
        "terminal_transition_count": 0,
    }


@pytest.mark.parametrize(
    ("mutation_boundary", "validation_episodes", "relative_path"),
    (
        ("training", 0, "training_metrics.jsonl"),
        ("validation", 16, "validation_metrics.jsonl"),
        ("retention", 0, None),
    ),
)
def test_planning_child_formal_mutations_rehash_real_input_pin(
    tmp_path: Path,
    mutation_boundary: str,
    validation_episodes: int,
    relative_path: str | None,
) -> None:
    from lunar_exploration_ppo.utils.stage6_input_pinning import (
        Stage6InputPinError,
    )

    module, backend, original_transaction, runtime = (
        _planning_child_run_update_guard_fixture(tmp_path / "backend")
    )
    transaction = module.StandardTrainingTransaction(
        sequence=original_transaction.sequence,
        seed=original_transaction.seed,
        seed_index=original_transaction.seed_index,
        update=original_transaction.update,
        validation_episodes=validation_episodes,
        commit_states=original_transaction.commit_states,
    )
    backend.transactions = (transaction,)
    backend._planning_warm_start = None
    backend.stage_root.mkdir(parents=True, exist_ok=True)
    backend._checkpoint_writes = 0
    runtime["seed"] = transaction.seed
    runtime["best_record"] = {"update": transaction.update}
    checkpoint_root = Path(runtime["checkpoint_root"])
    for update in (83, 84, 85, 86):
        _checkpoint_dir(checkpoint_root, update)
    checkpoint = module.StandardTransactionCheckpoint(
        transaction_key=transaction.key,
        seed=transaction.seed,
        update=transaction.update,
        checkpoint_sha256="1" * 64,
        complete_marker_sha256="2" * 64,
        policy_state_sha256="3" * 64,
    )
    validation_result: dict[str, object] = {}
    if validation_episodes:
        validation_result["validation_trace_binding"] = {
            "schema_version": (
                module._VALIDATION_TRACE_BINDING_SCHEMA_VERSION
            ),
            "sha256": "4" * 64,
            "size_bytes": 0,
            "line_count": validation_episodes,
        }
        backend._publish_validation_trace = lambda **_kwargs: None
    result = module.StandardUpdateResult(
        transaction=transaction,
        update_metrics={
            "policy_state_sha256_before": checkpoint.policy_state_sha256,
            "policy_state_sha256_after": checkpoint.policy_state_sha256,
        },
        validation_result=validation_result,
        eval_isolation_audit={"passed": True},
        best_record={"update": transaction.update},
        checkpoint=checkpoint,
        journal_records=(),
        collection_audit=_f3_collection_audit(
            checkpoint.policy_state_sha256
        ),
    )

    target_path = (
        backend.stage_root / relative_path
        if relative_path is not None
        else None
    )
    target_before = (
        target_path.read_bytes()
        if target_path is not None and target_path.is_file()
        else None
    )
    removable_checkpoint = checkpoint_root / "retention-delete-me.pt"
    removable_checkpoint.write_bytes(b"checkpoint-to-delete")
    assert removable_checkpoint.is_file()

    with _same_size_mutable_stage6_input_pin(
        tmp_path / f"pin-{mutation_boundary}"
    ) as (input_pin, mutate):
        mutated = False
        target_label = {
            "training": "run_update:before training metrics publication",
            "validation": "run_update:before validation metrics publication",
            "retention": "run_update:before checkpoint retention",
        }[mutation_boundary]

        def require_current(
            label: str,
            *,
            rehash_inputs: bool = False,
        ) -> None:
            nonlocal mutated
            if label == target_label:
                mutate()
                mutated = True
            try:
                input_pin.require_current(
                    label,
                    rehash=rehash_inputs,
                )
            except Stage6InputPinError as exc:
                raise module.StandardTrainingError(
                    "execution capability rejected"
                ) from exc

        def append_metric(path: Path, value: object) -> None:
            module.append_transaction_metric_once(path, value)
            if target_path is not None and path == target_path:
                raise AssertionError(
                    f"input drift reached {mutation_boundary} journal append"
                )

        class DeletingRetentionManager:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def apply(self, **_kwargs: object) -> object:
                removable_checkpoint.unlink()
                return object()

        def component(name: str):
            if name == "run_standard_update_transaction":
                return lambda **_kwargs: result
            if name == "append_transaction_metric_once":
                return append_metric
            if name == "CheckpointRetentionManager":
                return DeletingRetentionManager
            raise AssertionError(name)

        backend._require_execution_capability_current = require_current
        backend._require_source_repair_current = lambda _label: None
        backend._call = component
        try:
            with pytest.raises(
                module.StandardTrainingError,
                match="execution capability rejected",
            ):
                backend.run_update(runtime, transaction)
        finally:
            assert mutated is True
            if target_path is not None:
                current = (
                    target_path.read_bytes()
                    if target_path.is_file()
                    else None
                )
                assert current == target_before
            else:
                assert removable_checkpoint.is_file()


def test_planning_child_stale_resource_segment_is_rejected_before_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils import resource_lifecycle

    module = _training_module()
    capability = _planning_child_recovery_capability(tmp_path)
    stage_root = tmp_path / "s6"
    append_calls: list[dict[str, object]] = []
    active_segment_index = 5

    monkeypatch.setattr(
        resource_lifecycle,
        "validate_resource_lifecycle_ledger",
        lambda *_args, **_kwargs: {
            "passed": True,
            "active_segment_index": active_segment_index,
        },
    )

    def append_segment(
        *args: object,
        **kwargs: object,
    ) -> dict[str, object]:
        append_calls.append({"args": args, "kwargs": kwargs})
        return {"segment_index": 5}

    monkeypatch.setattr(
        resource_lifecycle,
        "append_resource_segment_start",
        append_segment,
    )

    with pytest.raises(
        module.StandardTrainingError,
        match="resource segment.*drifted",
    ):
        module._append_initial_resource_segment_start(
            stage_root=stage_root,
            first_sample={"passed": True},
            planning_child_recovery_capability=capability,
        )

    assert append_calls == []

    active_segment_index = 4
    assert module._append_initial_resource_segment_start(
        stage_root=stage_root,
        first_sample={"passed": True},
        planning_child_recovery_capability=capability,
    ) == {"segment_index": 5}
    assert len(append_calls) == 1


def test_planning_child_runtime_poll_does_not_reopen_recovery_inputs(
    tmp_path: Path,
) -> None:
    module, backend, transaction, runtime = (
        _planning_child_run_update_guard_fixture(tmp_path)
    )
    heavy_scan_bytes = 84_000 + 5_000
    counters = {
        "heavy_calls": 0,
        "heavy_bytes": 0,
        "execution_calls": 0,
    }

    def require_heavy(_label: str) -> None:
        counters["heavy_calls"] += 1
        counters["heavy_bytes"] += heavy_scan_bytes

    rehash_modes: list[bool] = []

    def require_execution(
        _label: str,
        *,
        rehash_inputs: bool = False,
    ) -> None:
        counters["execution_calls"] += 1
        rehash_modes.append(rehash_inputs)

    class StopAfterGuards(RuntimeError):
        pass

    def run_transaction(**kwargs: object) -> object:
        resource_guard = kwargs["resource_guard"]
        capability_guard = kwargs["capability_guard"]
        assert callable(resource_guard)
        assert callable(capability_guard)
        for poll in range(5):
            resource_guard(f"step-poll-{poll}")
        capability_guard("checkpoint:before-publication")
        raise StopAfterGuards

    backend._require_source_repair_current = require_heavy
    backend._require_execution_capability_current = require_execution
    backend._call = lambda name: (
        run_transaction
        if name == "run_standard_update_transaction"
        else (_ for _ in ()).throw(AssertionError(name))
    )

    with pytest.raises(StopAfterGuards):
        backend.run_update(runtime, transaction)

    assert counters == {
        "heavy_calls": 0,
        "heavy_bytes": 0,
        "execution_calls": 6,
    }
    assert rehash_modes == [False, False, False, False, False, True]


@pytest.mark.parametrize("mutation_boundary", ("post", "accepted"))
def test_planning_child_mutation_boundary_rehashes_pinned_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_boundary: str,
) -> None:
    from lunar_exploration_ppo.utils import stage6_input_pinning
    from lunar_exploration_ppo.utils.durable_jsonl import RunLease
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        append_resource_segment_start,
    )
    from lunar_exploration_ppo.workflows import stage6 as stage6_workflow
    from test_stage6_workflow import (
        CONFIG,
        FORMAL_STAGE6_RUN_ID,
        _build_execution_authority,
    )

    repo_root = tmp_path / "repo"
    config_path = repo_root / "configs/ppo_highres_frontier_stage6_v1.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(CONFIG.read_bytes())
    pinned_relative = next(
        relative
        for relative in stage6_workflow.STAGE6_SOURCE_PATHS
        if relative != "configs/ppo_highres_frontier_stage6_v1.json"
    )
    pinned_path = (repo_root / pinned_relative).resolve()
    real_pin_type = stage6_input_pinning.Stage6InputPin
    active = _build_execution_authority(
        tmp_path / "authority",
        monkeypatch,
        repo_root=repo_root,
        config_path=config_path,
        run_root=tmp_path / FORMAL_STAGE6_RUN_ID,
    )
    monkeypatch.setattr(
        stage6_input_pinning,
        "Stage6InputPin",
        real_pin_type,
    )
    run_root = Path(active["run_root"])
    stage_root = run_root / "s6"
    stage_root.mkdir(parents=True)
    fixture_root = tmp_path.resolve()
    for label, raw_path in active["requests"]:
        path = Path(raw_path)
        if path.is_file():
            continue
        try:
            path.relative_to(fixture_root)
        except ValueError as exc:
            raise AssertionError(
                f"required external pinned input is missing: {label}={path}"
            ) from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(label.encode("utf-8"))
    original_pinned_bytes = pinned_path.read_bytes()
    mutated_pinned_bytes = bytes(
        (original_pinned_bytes[0] ^ 1, *original_pinned_bytes[1:])
    )
    rehash_calls: list[tuple[str, bool]] = []
    original_require_current = real_pin_type.require_current
    original_descriptor_sha256 = stage6_input_pinning._descriptor_sha256

    def acquisition_descriptor_sha256(
        descriptor: int,
    ) -> tuple[str, int]:
        return "0" * 64, os.fstat(descriptor).st_size

    monkeypatch.setattr(
        stage6_input_pinning,
        "_descriptor_sha256",
        acquisition_descriptor_sha256,
    )

    def track_require_current(
        pin: object,
        label: str,
        *,
        rehash: bool = False,
    ) -> None:
        rehash_calls.append((label, rehash))
        original_require_current(pin, label, rehash=rehash)

    monkeypatch.setattr(
        real_pin_type,
        "require_current",
        track_require_current,
    )

    with stage6_input_pinning.acquire_stage6_input_pin(
        active["requests"]
    ) as input_pin:
        expected_records = {
            os.path.normcase(os.fspath(record.path)): record
            for record in active["pin"].records
        }
        for resource in input_pin._resources:
            expected = expected_records[
                os.path.normcase(os.fspath(resource.path))
            ]
            resource.sha256 = expected.sha256
            resource.size_bytes = expected.size_bytes
        pinned_resource = next(
            resource
            for resource in input_pin._resources
            if resource.path == pinned_path.resolve()
        )
        os.close(pinned_resource.descriptor)
        pinned_resource.descriptor = pinned_resource.parent_guard.open_file(
            pinned_resource.path,
            os.O_RDWR,
        )
        resources_by_descriptor = {
            resource.descriptor: resource
            for resource in input_pin._resources
        }

        def bound_descriptor_sha256(
            descriptor: int,
        ) -> tuple[str, int]:
            resource = resources_by_descriptor[descriptor]
            if resource is pinned_resource:
                return original_descriptor_sha256(descriptor)
            return resource.sha256, resource.size_bytes

        monkeypatch.setattr(
            stage6_input_pinning,
            "_descriptor_sha256",
            bound_descriptor_sha256,
        )
        try:
            lease = RunLease(run_root / ".stage6.lease")
            with lease:
                with stage6_workflow._stage6_execution_capability_scope(
                    config=active["config"],
                    config_path=active["config_path"],
                    formal_run_id=FORMAL_STAGE6_RUN_ID,
                    run_root=run_root,
                    stage_root=stage_root,
                    repo_root=active["repo_root"],
                    execution_identity=active["identity"],
                    expected_input_requests=active["requests"],
                    review_authorization_handle=active["review_handle"],
                    input_pin=input_pin,
                    stage5_authority=active["stage5_handle"],
                    run_lease=lease,
                ) as execution_capability:
                    module, backend, transaction, runtime = (
                        _planning_child_run_update_guard_fixture(
                            tmp_path / "backend"
                        )
                    )
                    backend._execution_capability = execution_capability
                    backend.config = active["config"]
                    backend.run_root = run_root
                    backend.stage_root = stage_root
                    backend.repo_root = Path(active["repo_root"])
                    backend.stage5_authority = dict(
                        active["stage5_handle"].identity
                    )
                    backend._require_source_repair_current = lambda _label: None
                    backend._append_resource_phase = (
                        module.StandardProductionBackend._append_resource_phase.__get__(
                            backend
                        )
                    )
                    resource = {
                        "rss_source": (
                            "process_tree_lifecycle_peak_current_sum/v1"
                        ),
                        "rss_root_pid": os.getpid(),
                        "rss_sample_count": 1,
                        "rss_bytes": 1,
                        "passed": True,
                    }
                    backend._resource_segment_start = (
                        append_resource_segment_start(
                            stage_root / "resource_audit.jsonl",
                            first_sample=resource,
                        )
                    )
                    backend._poll_runtime_resource_latch = (
                        lambda _latch, _boundary: dict(resource)
                    )
                    checkpoint = module.StandardTransactionCheckpoint(
                        transaction_key=transaction.key,
                        seed=transaction.seed,
                        update=transaction.update,
                        checkpoint_sha256="1" * 64,
                        complete_marker_sha256="2" * 64,
                        policy_state_sha256="3" * 64,
                    )

                    def mutate_pinned_bytes() -> None:
                        os.lseek(pinned_resource.descriptor, 0, os.SEEK_SET)
                        assert os.write(
                            pinned_resource.descriptor,
                            mutated_pinned_bytes,
                        ) == len(mutated_pinned_bytes)
                        os.fsync(pinned_resource.descriptor)

                    def run_transaction(**kwargs: object) -> object:
                        resource_guard = kwargs["resource_guard"]
                        persist_post_resource = kwargs[
                            "persist_post_resource"
                        ]
                        accept_post_resource = kwargs[
                            "accept_post_resource"
                        ]
                        assert callable(resource_guard)
                        assert callable(persist_post_resource)
                        assert callable(accept_post_resource)
                        resource_guard("step-poll")
                        if mutation_boundary == "post":
                            mutate_pinned_bytes()
                            persist_post_resource(checkpoint)
                        else:
                            persist_post_resource(checkpoint)
                            mutate_pinned_bytes()
                            accept_post_resource(checkpoint)
                        rows = backend._resource_audit_rows()
                        raise AssertionError(
                            "pinned-byte mutation reached durable "
                            f"{mutation_boundary} write; "
                            f"rehash_calls={rehash_calls!r}; "
                            f"phases={[row.get('phase') for row in rows]!r}"
                        )

                    backend._call = lambda name: (
                        run_transaction
                        if name == "run_standard_update_transaction"
                        else (_ for _ in ()).throw(AssertionError(name))
                    )

                    with pytest.raises(
                        module.StandardTrainingError,
                        match="execution capability rejected",
                    ):
                        backend.run_update(runtime, transaction)

                    run_update_calls = [
                        (label, rehash)
                        for label, rehash in rehash_calls
                        if label.startswith("run_update:")
                    ]
                    assert run_update_calls[0][1] is False
                    assert run_update_calls[-1][1] is True
                    rows = backend._resource_audit_rows()
                    phases = [row.get("phase") for row in rows]
                    if mutation_boundary == "post":
                        assert "post" not in phases
                    else:
                        assert "post" in phases
                        assert "accepted" not in phases
        finally:
            os.lseek(pinned_resource.descriptor, 0, os.SEEK_SET)
            assert os.write(
                pinned_resource.descriptor,
                original_pinned_bytes,
            ) == len(original_pinned_bytes)
            os.fsync(pinned_resource.descriptor)


def test_planning_child_execute_terminal_tail_forwards_issued_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows import (
        stage6 as stage6_workflow,
        stage6_planning_warm_start as warm_start,
    )
    from test_stage6_workflow import (
        ROOT,
        _append_task5_accepted_through_u90_restart_tail,
    )
    from test_stage6_planning_child_recovery import (
        _issue,
        _prepare_u100_terminal_evidence,
        _published_chain,
    )

    module = _training_module()
    (
        fixture,
        tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(
        tmp_path,
        monkeypatch,
        retained_checkpoint_updates=(84, 90),
    )
    _append_task5_accepted_through_u90_restart_tail(
        fixture,
        tail,
        final_update=100,
    )
    preterminal_capability = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )
    _prepare_u100_terminal_evidence(
        fixture,
        preterminal_capability,
        identity=identity,
        review=review,
        immutable=immutable,
        monkeypatch=monkeypatch,
    )
    from lunar_exploration_ppo.workflows import (
        stage6_planning_child_recovery as recovery,
    )

    issued_terminal_capability = (
        recovery.issue_planning_child_recovery_capability(
            stage_root=fixture.stage,
            parent_artifact_path=fixture.child_path,
            continuation_artifact_path=continuation_path,
            current_execution_identity=identity,
            current_verified_review_authorization=review,
            current_immutable_bindings=immutable,
        )
    )
    assert issued_terminal_capability.terminal_complete is True
    assert issued_terminal_capability.resume_cursor is None

    warm_payload = fixture.warm_start_path.read_bytes()
    warm_sha256 = hashlib.sha256(warm_payload).hexdigest()
    warm_context = warm_start.PlanningWarmStartContext(
        child_run_id=fixture.stage.parent.name,
        child_effective_config_sha256=hashlib.sha256(
            fixture.effective_config_bytes
        ).hexdigest(),
        child_effective_config_bytes=fixture.effective_config_bytes,
        parent_run_id="s6-standard-single-r1-20260721T000000Z",
        parent_config_sha256="1" * 64,
        child_first_update=75,
        child_final_update=100,
        new_semantics_update_count=26,
        imported_state_names=(),
        discarded_state_names=(),
        child_best_validation_updates=(80, 90, 100),
        artifact={},
        artifact_path=fixture.warm_start_path,
        artifact_sha256=warm_sha256,
    )
    parent = warm_start.VerifiedParentU74(
        parent_update=74,
        checkpoint_sha256="1" * 64,
        manifest_sha256="2" * 64,
        complete_sha256="3" * 64,
        policy_state_sha256="4" * 64,
        config_sha256="5" * 64,
        lineage_sha256="6" * 64,
        resource_accepted=True,
        u75_attempt1_discarded=True,
        checkpoint_payload={},
    )

    monkeypatch.setattr(
        module,
        "_standard_execution_operation",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        module,
        "_require_standard_execution_capability",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "_require_machine_preflight",
        lambda **kwargs: {
            "environment_identity": fixture.current_identity[
                "environment_identity"
            ],
            "environment_sha256": fixture.current_identity[
                "environment_sha256"
            ],
        },
    )
    monkeypatch.setattr(
        module,
        "_persist_execution_identity",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "_resource_gate_record",
        lambda snapshot, decision: {"passed": True},
    )
    monkeypatch.setattr(
        stage6_workflow,
        "stage6_execution_identity",
        lambda **kwargs: copy.deepcopy(fixture.current_identity),
    )

    from lunar_exploration_ppo.utils import resource_lifecycle, resources

    class Monitor:
        def __enter__(self) -> "Monitor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def stop(self) -> None:
            return None

    monkeypatch.setattr(resources, "ProcessTreeRSSMonitor", Monitor)
    monkeypatch.setattr(
        resources,
        "capture_resource_snapshot",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        resources,
        "evaluate_resource_gates",
        lambda *args, **kwargs: object(),
    )
    mutation_calls = {"segment": 0, "schedule": 0}

    def reject_resource_segment(
        *args: object,
        **kwargs: object,
    ) -> object:
        mutation_calls["segment"] += 1
        raise AssertionError(
            "terminal-complete launch appended a resource segment"
        )

    monkeypatch.setattr(
        resource_lifecycle,
        "append_resource_segment_start",
        reject_resource_segment,
    )

    pending = {
        "stage_root": str(fixture.stage),
        "summary": dict(fixture.acceptance["summary"]),
        "routing": dict(fixture.acceptance["routing"]),
    }

    class Backend:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError(
                "terminal-complete launch constructed a training backend"
            )

    monkeypatch.setattr(module, "StandardProductionBackend", Backend)

    def reject_schedule(config: object, backend: object) -> object:
        mutation_calls["schedule"] += 1
        raise AssertionError(
            "terminal-complete launch ran a training schedule"
        )

    monkeypatch.setattr(
        module,
        "run_standard_training_schedule",
        reject_schedule,
    )

    terminal_capabilities: list[object | None] = []
    real_verify = stage6_workflow.verify_stage6_machine_acceptance

    def verify_terminal(**kwargs: object) -> dict[str, object]:
        terminal_capabilities.append(
            kwargs.get("planning_child_recovery_capability")
        )
        return real_verify(**kwargs)

    monkeypatch.setattr(
        stage6_workflow,
        "verify_stage6_machine_acceptance",
        verify_terminal,
    )

    result = module._execute_standard_training(
        config=fixture.config,
        run_root=fixture.stage.parent,
        repo_root=ROOT,
        stage5_authority=fixture.authority.identity,
        verified_review_authorization=fixture.current_review,
        execution_capability=object(),
        planning_warm_start_context=warm_context,
        planning_warm_start_sha256=warm_sha256,
        planning_child_recovery_capability=issued_terminal_capability,
        verified_parent_u74=parent,
    )

    assert result == pending
    assert terminal_capabilities == [issued_terminal_capability]
    assert mutation_calls == {"segment": 0, "schedule": 0}


def test_planning_child_backend_consumes_capability_for_first_resume_boundary(
    tmp_path: Path,
) -> None:
    module = _training_module()
    capability = _planning_child_recovery_capability(tmp_path)
    transaction = module.StandardTrainingTransaction(
        sequence=11,
        seed=SEEDS[0],
        seed_index=0,
        update=86,
        validation_episodes=0,
        commit_states=(f"seed_{SEEDS[0]}_training_update_86",),
    )
    backend = object.__new__(module.StandardProductionBackend)
    backend._planning_child_recovery_capability = capability
    backend._planning_child_resume_boundary_consumed = False
    backend._resource_segment_start = {"segment_index": 5}
    backend._next_resource_attempt = lambda _key: 1

    backend._require_planning_child_resume_boundary((transaction,))

    assert backend._planning_child_resume_boundary_consumed is True


def test_planning_child_backend_releases_frozen_cursor_after_first_transaction(
    tmp_path: Path,
) -> None:
    module = _training_module()
    capability = _planning_child_recovery_capability(tmp_path)
    first = module.StandardTrainingTransaction(
        sequence=11,
        seed=SEEDS[0],
        seed_index=0,
        update=86,
        validation_episodes=0,
        commit_states=(f"seed_{SEEDS[0]}_training_update_86",),
    )
    later = module.StandardTrainingTransaction(
        sequence=12,
        seed=SEEDS[0],
        seed_index=0,
        update=87,
        validation_episodes=0,
        commit_states=(f"seed_{SEEDS[0]}_training_update_87",),
    )
    backend = object.__new__(module.StandardProductionBackend)
    backend._planning_child_recovery_capability = capability
    backend._planning_child_resume_boundary_consumed = False
    backend._resource_segment_start = {"segment_index": 5}
    backend._next_resource_attempt = lambda _key: 1

    backend._require_planning_child_resume_boundary((first,))
    backend._resource_segment_start = {"segment_index": 99}
    backend._next_resource_attempt = lambda _key: 99
    backend._require_planning_child_resume_boundary((later,))

    assert backend._planning_child_resume_boundary_consumed is True


def test_planning_child_backend_retention_protects_authenticated_u84(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.configs.stage6 import load_stage6_config
    from test_stage6_execution import CONFIG

    module = _training_module()
    config = load_stage6_config(CONFIG)
    capability = _planning_child_recovery_capability(tmp_path)
    assert capability.protected_checkpoint_updates == (84, 85)
    calls: list[dict[str, object]] = []

    class Retention:
        def __init__(self, root: Path, *, schema_version: str) -> None:
            del root, schema_version

        def apply(self, **kwargs: object) -> object:
            calls.append(dict(kwargs))
            return SimpleNamespace(kept_updates=(), removed_updates=())

    backend = object.__new__(module.StandardProductionBackend)
    backend.config = config
    backend._planning_warm_start = object()
    backend._planning_child_recovery_capability = capability
    backend._source_repair = None
    backend._components = {"CheckpointRetentionManager": Retention}
    backend._execution_operation = lambda label: nullcontext(label)
    backend._require_execution_capability_current = (
        lambda _label, *, rehash_inputs=False: None
    )

    backend._apply_checkpoint_retention(
        checkpoint_root=tmp_path / "checkpoints",
        update=85,
        best_record={"update": 85},
    )

    assert calls == [
        {
            "latest_update": 85,
            "periodic_updates": (),
            "best_update": 85,
            "periodic_keep_count": (
                config.training.periodic_keep_count
            ),
            "protected_updates": (84, 85),
        }
    ]


def test_classic_source_repair_retention_protected_updates_stay_forwarded(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.configs.stage6 import load_stage6_config
    from test_stage6_execution import CONFIG

    module = _training_module()
    config = load_stage6_config(CONFIG)
    calls: list[dict[str, object]] = []

    class Retention:
        def __init__(self, root: Path, *, schema_version: str) -> None:
            del root, schema_version

        def apply(self, **kwargs: object) -> object:
            calls.append(dict(kwargs))
            return SimpleNamespace(kept_updates=(), removed_updates=())

    backend = object.__new__(module.StandardProductionBackend)
    backend.config = config
    backend._planning_warm_start = None
    backend._source_repair = SimpleNamespace(
        protected_checkpoint_updates=(9, 32)
    )
    backend._components = {"CheckpointRetentionManager": Retention}
    backend._execution_operation = lambda label: nullcontext(label)
    backend._require_execution_capability_current = (
        lambda _label, *, rehash_inputs=False: None
    )

    backend._apply_checkpoint_retention(
        checkpoint_root=tmp_path / "checkpoints",
        update=40,
        best_record={"update": 40},
    )

    assert calls[0]["protected_updates"] == (9, 32)


def test_classic_sensor_acceleration_journal_binding_split_returns_cutover(
) -> None:
    from lunar_exploration_ppo.workflows.stage6_source_repair import (
        SENSOR_ACCELERATION_NEXT_TRANSACTION_KEY,
    )

    module = _training_module()
    origin_immutable_bindings = {
        "binding": "classic-sensor-acceleration-origin"
    }
    backend = object.__new__(module.StandardProductionBackend)
    backend._planning_child_recovery_capability = None
    backend._source_repair = SimpleNamespace(
        sensor_acceleration_sha256="a" * 64,
        origin_immutable_bindings=origin_immutable_bindings,
    )

    assert backend._journal_binding_split() == (
        dict(origin_immutable_bindings),
        SENSOR_ACCELERATION_NEXT_TRANSACTION_KEY,
    )


def test_planning_child_backend_retention_pins_current_u84_u85_chain(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.configs.stage6 import load_stage6_config
    from test_stage6_execution import CONFIG

    module = _training_module()
    retention_module = _retention_module()
    config = load_stage6_config(CONFIG)
    capability = _planning_child_recovery_capability(tmp_path)
    checkpoint_root = tmp_path / "retention" / "checkpoints"
    for update in (84, 85, 86):
        _checkpoint_dir(checkpoint_root, update)

    backend = object.__new__(module.StandardProductionBackend)
    backend.config = config
    backend._planning_warm_start = object()
    backend._planning_child_recovery_capability = capability
    backend._source_repair = None
    backend._components = {
        "CheckpointRetentionManager": (
            retention_module.CheckpointRetentionManager
        )
    }
    backend._execution_operation = lambda label: nullcontext(label)
    backend._require_execution_capability_current = (
        lambda _label, *, rehash_inputs=False: None
    )

    receipt = backend._apply_checkpoint_retention(
        checkpoint_root=checkpoint_root,
        update=86,
        best_record={"update": 86},
    )

    assert receipt.kept_updates == (84, 85, 86)
    assert receipt.removed_updates == ()
    assert (checkpoint_root / "update-00000084").is_dir()
    assert (checkpoint_root / "update-00000085").is_dir()
    assert (checkpoint_root / "update-00000086").is_dir()


def test_planning_child_backend_does_not_alias_capability_to_legacy_context(
) -> None:
    module = _training_module()

    execute_source = inspect.getsource(module._execute_standard_training)
    persist_source = inspect.getsource(module._persist_execution_identity)
    assert (
        "planning_child_source_repair_context = ("
        not in execute_source
    )
    assert "planning_child_source_repair_sha256 = (" not in execute_source
    assert "planning_child_source_repair_context" not in persist_source
    assert "planning_child_recovery_capability" in persist_source


def test_checkpoint_retention_accepts_and_validates_real_manager_latest_index(
    tmp_path: Path,
) -> None:
    from test_stage4_checkpoint import _manager, _policy_and_optimizer, _save

    module = _retention_module()
    policy, optimizer = _policy_and_optimizer()
    checkpoint_manager = _manager(tmp_path)
    _save(
        checkpoint_manager,
        policy=policy,
        optimizer=optimizer,
        update_step=1,
    )
    retention = module.CheckpointRetentionManager(tmp_path / "checkpoints")

    receipt = retention.apply(
        latest_update=1,
        periodic_updates=(),
        best_update=1,
        periodic_keep_count=5,
    )

    assert receipt.kept_updates == (1,)
    assert (tmp_path / "checkpoints/latest.json").is_file()
    (tmp_path / "checkpoints/latest.json").write_bytes(b"{}\n")
    with pytest.raises(module.CheckpointRetentionError, match="latest"):
        retention.apply(
            latest_update=1,
            periodic_updates=(),
            best_update=1,
            periodic_keep_count=5,
        )


@pytest.mark.parametrize("link_kind", ("root", "update", "member"))
def test_checkpoint_retention_rejects_links_before_any_delete(
    tmp_path: Path,
    link_kind: str,
) -> None:
    module = _retention_module()

    def make_link(link: Path, target: Path, *, directory: bool) -> None:
        try:
            os.symlink(target, link, target_is_directory=directory)
        except OSError as exc:
            if not directory:
                pytest.skip(f"file symlink unavailable on this host: {exc}")
            completed = subprocess.run(
                ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True,
                check=False,
                text=True,
            )
            if completed.returncode != 0:
                pytest.skip(
                    "directory link unavailable on this host: "
                    f"{completed.stdout} {completed.stderr}"
                )

    if link_kind == "root":
        target_root = tmp_path / "outside-root"
        _checkpoint_dir(target_root, 1)
        linked_root = tmp_path / "linked-root"
        make_link(linked_root, target_root, directory=True)
        with pytest.raises(module.CheckpointRetentionError, match="link|reparse"):
            module.CheckpointRetentionManager(linked_root)
        assert (target_root / "update-00000001/checkpoint.pt").is_file()
        return

    root = tmp_path / "retention-links"
    root.mkdir()
    if link_kind == "update":
        outside_update = tmp_path / "outside-update"
        outside_update.mkdir()
        for name in ("checkpoint.pt", "manifest.json", "complete.json"):
            (outside_update / name).write_bytes(name.encode("ascii"))
        make_link(root / "update-00000001", outside_update, directory=True)
    else:
        update = root / "update-00000001"
        update.mkdir()
        outside_checkpoint = tmp_path / "outside-checkpoint.pt"
        outside_checkpoint.write_bytes(b"outside")
        make_link(update / "checkpoint.pt", outside_checkpoint, directory=False)
        (update / "manifest.json").write_bytes(b"manifest")
        (update / "complete.json").write_bytes(b"complete")

    manager = module.CheckpointRetentionManager(root)
    with pytest.raises(module.CheckpointRetentionError, match="link|reparse"):
        manager.apply(
            latest_update=1,
            periodic_updates=(),
            best_update=1,
            periodic_keep_count=5,
        )
    if link_kind == "update":
        assert (outside_update / "checkpoint.pt").is_file()
    else:
        assert outside_checkpoint.is_file()


def test_checkpoint_retention_rejects_member_reparse_before_any_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the member-level fail-closed path when file symlinks are unavailable."""

    module = _retention_module()
    root = tmp_path / "retention-member-reparse"
    update = _checkpoint_dir(root, 1)
    original = module._is_link_or_reparse
    monkeypatch.setattr(
        module,
        "_is_link_or_reparse",
        lambda path: path == update / "checkpoint.pt" or original(path),
    )

    manager = module.CheckpointRetentionManager(root)
    with pytest.raises(module.CheckpointRetentionError, match="link|reparse"):
        manager.apply(
            latest_update=1,
            periodic_updates=(),
            best_update=1,
            periodic_keep_count=5,
        )
    assert {path.name for path in update.iterdir()} == {
        "checkpoint.pt",
        "manifest.json",
        "complete.json",
    }


@pytest.mark.parametrize("reparse_component", ("s6", "checkpoints"))
def test_stage6_retention_rejects_ancestor_reparse_before_any_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reparse_component: str,
) -> None:
    module = _retention_module()
    run_root = tmp_path / "run"
    run_root.mkdir()
    outside = tmp_path / f"outside-{reparse_component}"

    if reparse_component == "s6":
        checkpoint_root = outside / "checkpoints/seed-20260716"
        checkpoint_root.mkdir(parents=True)
        _make_directory_reparse(run_root / "s6", outside)
    else:
        (run_root / "s6").mkdir()
        checkpoint_root = outside / "seed-20260716"
        checkpoint_root.mkdir(parents=True)
        _make_directory_reparse(run_root / "s6/checkpoints", outside)

    _checkpoint_dir(checkpoint_root, 1)
    _checkpoint_dir(checkpoint_root, 2)
    guarded_root = run_root / "s6/checkpoints/seed-20260716"
    unlink_calls: list[Path] = []

    def reject_unlink(path: Path, *args, **kwargs) -> None:
        unlink_calls.append(path)
        raise AssertionError("retention reached unlink through ancestor reparse")

    monkeypatch.setattr(Path, "unlink", reject_unlink)
    with pytest.raises(module.CheckpointRetentionError, match="link|reparse"):
        manager = module.CheckpointRetentionManager(
            guarded_root,
            schema_version="stage6_complete_checkpoint/v1",
        )
        manager.apply(
            latest_update=2,
            periodic_updates=(),
            best_update=2,
            periodic_keep_count=5,
        )
    assert unlink_calls == []


def test_planning_child_machine_lineage_splits_origin_u80_and_amended_u90(
    tmp_path: Path,
) -> None:
    stage6 = importlib.import_module(
        "lunar_exploration_ppo.workflows.stage6"
    )
    capability = _planning_child_recovery_capability(tmp_path)
    u80 = stage6._stage6_planning_checkpoint_lineage_for_update(
        update=80,
        current_checkpoint_lineage={},
        historical_checkpoint_lineage={},
        planning_warm_start_context=None,
        planning_warm_start_sha256="",
        planning_child_source_repair=capability,
    )
    u90 = stage6._stage6_planning_checkpoint_lineage_for_update(
        update=90,
        current_checkpoint_lineage={},
        historical_checkpoint_lineage={},
        planning_warm_start_context=None,
        planning_warm_start_sha256="",
        planning_child_source_repair=capability,
    )

    assert u80["lineage_generation"] == "origin"
    assert "planning_child_source_repair" not in u80
    assert u90["lineage_generation"] == "continuation"
    assert u90["planning_child_source_repair"]["epoch"] == "continuation"
    assert u90["checkpoint_update"] == 90

    backend = object.__new__(
        _training_module().StandardProductionBackend
    )
    backend._planning_child_recovery_capability = capability
    backend_lineage = backend._checkpoint_lineage_for_update(90)
    stage6.ArtifactStore.canonical_json_bytes(backend_lineage)
    assert backend_lineage == u90


def test_standard_backend_has_no_legacy_planning_child_context_branches() -> None:
    source = inspect.getsource(
        _training_module().StandardProductionBackend
    )

    assert "_planning_child_source_repair" not in source


def test_planning_child_machine_context_binds_origin_and_current_capability(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from test_stage6_planning_child_source_repair import (
        _fixture,
    )

    stage6 = importlib.import_module(
        "lunar_exploration_ppo.workflows.stage6"
    )
    fixture = _fixture(tmp_path)
    lineage = json.loads(fixture.origin_lineage_bytes.decode("utf-8"))
    capability = _planning_child_recovery_capability(tmp_path)
    current_identity_sha256 = hashlib.sha256(
        stage6.ArtifactStore.canonical_json_bytes(
            fixture.current_execution_identity
        )
    ).hexdigest()
    origin_identity_sha256 = hashlib.sha256(
        stage6.ArtifactStore.canonical_json_bytes(
            lineage["execution_identity"]
        )
    ).hexdigest()
    current_immutable_sha256 = hashlib.sha256(
        stage6.ArtifactStore.canonical_json_bytes(
            fixture.current_immutable_bindings
        )
    ).hexdigest()
    lineage_epochs = (
        replace(
            capability.lineage_epochs[0],
            execution_identity_sha256=origin_identity_sha256,
        ),
        capability.lineage_epochs[1],
        replace(
            capability.lineage_epochs[2],
            execution_identity_sha256=current_identity_sha256,
        ),
    )
    acceptance = {
        **dict(capability.acceptance_binding),
        "schema_version": (
            "stage6_planning_child_recovery_capability_acceptance/v1"
        ),
        "formal_run_id": capability.formal_run_id,
        "seed": capability.seed,
        "parent_artifact_sha256": (
            capability.parent_artifact_sha256
        ),
        "continuation_artifact_sha256": (
            capability.continuation_artifact_sha256
        ),
        "input_snapshot_sha256": (
            capability.input_snapshot_sha256
        ),
        "current_execution_identity_sha256": (
            current_identity_sha256
        ),
        "current_immutable_bindings_sha256": (
            current_immutable_sha256
        ),
        "current_verified_review_authorization": (
            fixture.current_verified_review_authorization
        ),
        "origin_verified_review_authorization": (
            lineage["verified_review_authorization"]
        ),
        "lineage_epochs": [
            {
                "first_update": epoch.first_update,
                "last_update": epoch.last_update,
                "artifact_sha256": epoch.artifact_sha256,
                "execution_identity_sha256": (
                    epoch.execution_identity_sha256
                ),
            }
            for epoch in lineage_epochs
        ],
        "journal_binding_epochs": (
            (
                f"0001:{capability.seed}:update:075",
                lineage["immutable_bindings"],
            ),
            (
                f"0010:{capability.seed}:update:085",
                {"epoch": "parent"},
            ),
            (
                f"0011:{capability.seed}:update:086",
                fixture.current_immutable_bindings,
            ),
        ),
    }
    capability = replace(
        capability,
        lineage_epochs=lineage_epochs,
        acceptance_binding=acceptance,
    )

    validated = stage6._validate_planning_child_machine_context(
        capability=capability,
        lineage_audit=lineage,
        current_execution_identity=fixture.current_execution_identity,
        current_verified_review_authorization=(
            fixture.current_verified_review_authorization
        ),
        current_immutable_bindings=fixture.current_immutable_bindings,
    )

    assert validated["historical_immutable_bindings"] == (
        lineage["immutable_bindings"]
    )
    assert validated["current_immutable_bindings"] == (
        fixture.current_immutable_bindings
    )
    assert len(validated["journal_binding_epochs"]) == 3

    with pytest.raises(
        stage6.Stage6WorkflowError,
        match="recovery.*binding|capability.*drifted",
    ):
        stage6._validate_planning_child_machine_context(
            capability=capability,
            lineage_audit=lineage,
            current_execution_identity={
                **fixture.current_execution_identity,
                "source_set_sha256": "f" * 64,
            },
            current_verified_review_authorization=(
                fixture.current_verified_review_authorization
            ),
            current_immutable_bindings=fixture.current_immutable_bindings,
        )

    with pytest.raises(
        stage6.Stage6WorkflowError,
        match="recovery.*binding|capability.*drifted",
    ):
        stage6._validate_planning_child_machine_context(
            capability=capability,
            lineage_audit=lineage,
            current_execution_identity=fixture.current_execution_identity,
            current_verified_review_authorization={
                **fixture.current_verified_review_authorization,
                "tampered": True,
            },
            current_immutable_bindings=fixture.current_immutable_bindings,
        )

    with pytest.raises(
        stage6.Stage6WorkflowError,
        match="recovery.*binding|capability.*drifted",
    ):
        stage6._validate_planning_child_machine_context(
            capability=capability,
            lineage_audit=lineage,
            current_execution_identity=fixture.current_execution_identity,
            current_verified_review_authorization=(
                fixture.current_verified_review_authorization
            ),
            current_immutable_bindings={
                **fixture.current_immutable_bindings,
                "tampered": True,
            },
        )


def test_journal_cutover_accepts_full_planning_child_historical_bindings(
    tmp_path: Path,
) -> None:
    historical = _full_phase_immutable_bindings("1")
    current = _full_phase_immutable_bindings("2")
    module, transactions, records, receipts = _journal_cutover_fixture(
        tmp_path,
        historical,
    )

    completed = module.verify_journal_checkpoint_bindings(
        records,
        transactions,
        receipts,
        current,
        historical_immutable_bindings=historical,
        current_binding_first_transaction_key=transactions[1].key,
    )

    assert completed == (transactions[0].key,)


def test_journal_cutover_keeps_classic_base_to_full_bindings(
    tmp_path: Path,
) -> None:
    historical = _phase_immutable_bindings("3")
    current = _full_phase_immutable_bindings("4")
    module, transactions, records, receipts = _journal_cutover_fixture(
        tmp_path,
        historical,
    )

    completed = module.verify_journal_checkpoint_bindings(
        records,
        transactions,
        receipts,
        current,
        historical_immutable_bindings=historical,
        current_binding_first_transaction_key=transactions[1].key,
    )

    assert completed == (transactions[0].key,)


def test_journal_cutover_rejects_illegal_historical_binding_keys(
    tmp_path: Path,
) -> None:
    historical = {
        **_full_phase_immutable_bindings("5"),
        "unexpected_binding": "forbidden",
    }
    current = _full_phase_immutable_bindings("6")
    module, transactions, records, receipts = _journal_cutover_fixture(
        tmp_path,
        historical,
    )

    with pytest.raises(
        module.StandardTrainingError,
        match="immutable journal binding drifted",
    ):
        module.verify_journal_checkpoint_bindings(
            records,
            transactions,
            receipts,
            current,
            historical_immutable_bindings=historical,
            current_binding_first_transaction_key=transactions[1].key,
        )


def test_journal_binding_epochs_preserve_u84_parent_u85_and_successor_u86(
    tmp_path: Path,
) -> None:
    module = _training_module()
    origin = _full_phase_immutable_bindings("1")
    parent = _full_phase_immutable_bindings("2")
    successor = _full_phase_immutable_bindings("3")
    transactions = tuple(
        module.StandardTrainingTransaction(
            sequence=index,
            seed=SEEDS[0],
            seed_index=0,
            update=update,
            validation_episodes=0,
            commit_states=(
                f"seed_{SEEDS[0]}_training_update_{update}",
            ),
        )
        for index, update in enumerate((84, 85, 86))
    )
    receipt_index = module.CheckpointReceiptIndex(
        tmp_path / "epoch-receipts.jsonl"
    )
    records = []
    for index, (transaction, immutable) in enumerate(
        zip(
            transactions,
            (origin, parent, successor),
            strict=True,
        )
    ):
        checkpoint = module.StandardTransactionCheckpoint(
            transaction_key=transaction.key,
            seed=transaction.seed,
            update=transaction.update,
            checkpoint_sha256=f"{index + 4}" * 64,
            complete_marker_sha256=f"{index + 7}" * 64,
            policy_state_sha256=f"{index + 1}" * 64,
        )
        assert receipt_index.append_once(checkpoint)
        records.extend(
            {
                "state": state,
                "bindings": {
                    **immutable,
                    "checkpoint_sha256": (
                        checkpoint.checkpoint_sha256
                    ),
                },
            }
            for state in transaction.commit_states
        )

    completed = module.verify_journal_checkpoint_bindings(
        records,
        transactions,
        receipt_index.verify(),
        successor,
        immutable_binding_epochs=(
            (transactions[0].key, origin),
            (transactions[1].key, parent),
            (transactions[2].key, successor),
        ),
    )

    assert completed == tuple(
        transaction.key for transaction in transactions
    )
    drifted = copy.deepcopy(records)
    drifted[1]["bindings"] = {
        **successor,
        "checkpoint_sha256": drifted[1]["bindings"][
            "checkpoint_sha256"
        ],
    }
    with pytest.raises(
        module.StandardTrainingError,
        match="immutable binding",
    ):
        module.verify_journal_checkpoint_bindings(
            drifted,
            transactions,
            receipt_index.verify(),
            successor,
            immutable_binding_epochs=(
                (transactions[0].key, origin),
                (transactions[1].key, parent),
                (transactions[2].key, successor),
            ),
        )


def test_planning_child_phase_preflight_resumes_with_origin_bindings(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from lunar_exploration_ppo.workflows.stage6 import Stage6StateJournal

    module = _training_module()
    origin = _phase_immutable_bindings("1")
    current = _phase_immutable_bindings("2")
    assert origin != current
    capability = _planning_child_recovery_capability(tmp_path)
    capability = replace(
        capability,
        acceptance_binding={
            **dict(capability.acceptance_binding),
            "journal_binding_epochs": (
                ("origin", origin),
                ("parent", current),
                ("continuation", current),
            ),
        },
    )

    journal = Stage6StateJournal(tmp_path / "phase-state.jsonl")
    journal.append(
        "preflight",
        {**origin, "checkpoint_sha256": "7" * 64},
    )
    backend = object.__new__(module.StandardProductionBackend)
    backend._execution_operation = lambda _label: nullcontext()
    backend._immutable_bindings = current
    backend._source_repair = None
    backend._planning_child_recovery_capability = capability
    backend.phase_journal = journal

    backend._append_phase_state(
        "preflight",
        checkpoint_sha256="7" * 64,
    )
    assert len(journal.verify()) == 1

    ordinary_backend = object.__new__(module.StandardProductionBackend)
    ordinary_backend._execution_operation = lambda _label: nullcontext()
    ordinary_backend._immutable_bindings = current
    ordinary_backend._source_repair = None
    ordinary_backend._planning_child_recovery_capability = None
    ordinary_backend.phase_journal = Stage6StateJournal(
        tmp_path / "ordinary-phase-state.jsonl"
    )
    ordinary_backend._append_phase_state(
        "preflight",
        checkpoint_sha256="7" * 64,
    )
    assert ordinary_backend.phase_journal.verify()[0]["bindings"] == {
        **current,
        "checkpoint_sha256": "7" * 64,
    }

    later_journal = Stage6StateJournal(
        tmp_path / "later-phase-state.jsonl"
    )
    later_journal.append(
        "preflight",
        {**origin, "checkpoint_sha256": "7" * 64},
    )
    later_backend = object.__new__(module.StandardProductionBackend)
    later_backend._execution_operation = lambda _label: nullcontext()
    later_backend._immutable_bindings = current
    later_backend._source_repair = None
    later_backend._planning_child_recovery_capability = capability
    later_backend.phase_journal = later_journal
    later_backend._append_phase_state(
        "global_best_frozen",
        checkpoint_sha256="8" * 64,
    )
    assert later_journal.verify()[1]["bindings"] == {
        **current,
        "checkpoint_sha256": "8" * 64,
    }


def test_planning_child_resource_boundary_requires_capability_bound_bytes(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    stage6 = importlib.import_module(
        "lunar_exploration_ppo.workflows.stage6"
    )
    from test_stage6_planning_child_source_repair import _fixture

    fixture = _fixture(tmp_path)
    resource_payload = (
        fixture.stage_root / "resource_audit.jsonl"
    ).read_bytes()
    resource_rows = [
        json.loads(line)
        for line in resource_payload.decode("utf-8").splitlines()
    ]
    capability = _planning_child_recovery_capability(tmp_path)
    acceptance = {
        **dict(capability.acceptance_binding),
        "journal_prefixes": {
            "resource_audit.jsonl": {
                "path": "resource_audit.jsonl",
                "size_bytes": len(resource_payload),
                "sha256": hashlib.sha256(
                    resource_payload
                ).hexdigest(),
                "line_count": resource_payload.count(b"\n"),
            },
        },
    }
    capability = replace(
        capability,
        acceptance_binding=acceptance,
    )

    assert stage6._validate_planning_child_resource_boundary(
        capability=capability,
        resource_rows=resource_rows,
        resource_payload=resource_payload,
    ) == {
        "startup_prefix": {
            "size_bytes": len(resource_payload),
            "sha256": hashlib.sha256(resource_payload).hexdigest(),
            "line_count": resource_payload.count(b"\n"),
        },
        "current_payload": {
            "size_bytes": len(resource_payload),
            "sha256": hashlib.sha256(resource_payload).hexdigest(),
            "line_count": resource_payload.count(b"\n"),
        },
    }

    drifted_payload = bytearray(resource_payload)
    drifted_payload[0] ^= 1
    with pytest.raises(
        stage6.Stage6WorkflowError,
        match="startup bytes drifted",
    ):
        stage6._validate_planning_child_resource_boundary(
            capability=capability,
            resource_rows=resource_rows,
            resource_payload=bytes(drifted_payload),
        )


def test_machine_acceptance_repair_bindings_never_alias_child_to_classic(
    tmp_path: Path,
) -> None:
    stage6 = importlib.import_module(
        "lunar_exploration_ppo.workflows.stage6"
    )
    capability = _planning_child_recovery_capability(tmp_path)

    assert stage6._stage6_acceptance_repair_bindings(
        source_repair=None,
        planning_child_source_repair=capability,
    ) == {
        "source_repair_binding": None,
        "planning_child_source_repair_binding": (
            capability.evidence_binding()
        ),
    }
    with pytest.raises(
        stage6.Stage6WorkflowError,
        match="mutually exclusive",
    ):
        stage6._stage6_acceptance_repair_bindings(
            source_repair=SimpleNamespace(summary_binding={}),
            planning_child_source_repair=capability,
        )
