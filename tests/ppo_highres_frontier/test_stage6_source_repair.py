from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import os
import subprocess
import sys
import textwrap
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from lunar_exploration_ppo.configs.stage6 import load_stage6_config
from lunar_exploration_ppo.ppo.checkpoint_retention import (
    CheckpointRetentionManager,
)
from lunar_exploration_ppo.ppo import standard_training as standard_training_module
from lunar_exploration_ppo.ppo.standard_training import (
    CheckpointReceiptIndex,
    StandardProductionBackend,
    StandardTrainingError,
    StandardTransactionCheckpoint,
    ValidationRecord,
    _review_authorization_immutable_bindings,
    build_stage6_acceptance_artifacts,
    build_standard_training_transactions,
    reconcile_checkpointed_resume,
    verify_journal_checkpoint_bindings,
)
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.durable_jsonl import RunLease
from lunar_exploration_ppo.workflows.stage6 import (
    STAGE5_COMMIT,
    STAGE6_STAGE_ID,
    Stage6StateJournal,
)
from lunar_exploration_ppo.workflows import stage6 as stage6_workflow
from lunar_exploration_ppo.workflows import stage6_source_repair as source_repair


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/ppo_highres_frontier_stage6_v1.json"
RUN_ID = "s6-standard-single-r1-20260718T220434Z"
SEED = 20260716
SENSOR_ACCELERATION_SEGMENT_ID = "73f9e27a3175421d863ce4d9df361315"
SENSOR_ACCELERATION_ROOT_PID = 2_000_000_000


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return ArtifactStore.canonical_json_bytes(value)


def _acceptance_for_source_repair(
    binding: dict[str, object],
    immutable_bindings: dict[str, object],
) -> dict[str, object]:
    return build_stage6_acceptance_artifacts(
        global_best={
            "seed": SEED,
            "update": 100,
            "success_rate_under_fixed_step_budget": 0.5,
            "mean_final_coverage": 0.9,
            "checkpoint_ref": "seed-20260716/update-00000100",
        },
        performance_advantage_established=False,
        performance_claim="performance advantage not established",
        ppo_ci95_low=0.1,
        gain_over_cost_ci95_high=0.2,
        final_evaluation_count=10,
        final_episode_count=640,
        checkpoint_receipt_count=100,
        immutable_bindings=immutable_bindings,
        source_repair_binding=binding,
    )


def _write_json(path: Path, value: object) -> bytes:
    payload = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _identity(
    *,
    tree: str,
    source_sha256: str,
    config_sha256: str,
    data_sha256: str,
    environment: dict[str, object],
) -> dict[str, object]:
    environment_sha256 = _sha(_canonical(environment))
    return {
        "schema_version": "stage6_execution_identity/v1",
        "base_commit": STAGE5_COMMIT,
        "head_commit": STAGE5_COMMIT,
        "prospective_git_tree": tree,
        "prospective_tree_sha256": _sha(tree.encode("ascii")),
        "changed_paths": ["src/lunar_exploration_ppo/env/frontier.py"],
        "changed_path_set_sha256": "3" * 64,
        "real_index_empty": True,
        "config_sha256": config_sha256,
        "source_set_sha256": source_sha256,
        "data_sha256": data_sha256,
        "catalog_sha256": "4" * 64,
        "environment_identity": environment,
        "environment_sha256": environment_sha256,
        "source_identity": {
            "schema_version": "stage6_current_source_set/v1",
            "source_set_sha256": source_sha256,
            "paths": [],
        },
        "data_identity": {
            "schema_version": "stage6_data_catalog_identity/v1",
            "files": [],
            "catalog_sha256": "4" * 64,
        },
    }


def _authorization(identity: dict[str, object], *, marker: str) -> dict[str, object]:
    return {
        "schema_version": "stage6_verified_review_launch_authorization/v1",
        "stage_id": STAGE6_STAGE_ID,
        "formal_run_id": RUN_ID,
        "single_seed_scope": {
            "scope_kind": "single_seed_system_closure/v1",
            "seed": SEED,
            "updates": 100,
        },
        "authorized": True,
        "base_commit": STAGE5_COMMIT,
        "authorization_file_sha256": marker * 64,
        "authorization_file_size_bytes": 100,
        "review_identity_sha256": _sha(_canonical(identity)),
        "reviewed_prospective_git_tree": identity["prospective_git_tree"],
        "prospective_tree_sha256": identity["prospective_tree_sha256"],
        "changed_path_set_sha256": identity["changed_path_set_sha256"],
        "source_set_sha256": identity["source_set_sha256"],
        "config_sha256": identity["config_sha256"],
        "data_sha256": identity["data_sha256"],
        "environment_sha256": identity["environment_sha256"],
        "frozen_diff_sha256": marker * 64,
        "frozen_diff_size_bytes": 200,
        "spec_review_sha256": marker * 64,
        "quality_review_sha256": marker * 64,
    }


def _immutable(
    identity: dict[str, object],
    authorization: dict[str, object],
    *,
    environment: dict[str, object],
) -> dict[str, object]:
    return {
        "config_sha256": identity["config_sha256"],
        "source_set_sha256": identity["source_set_sha256"],
        "prospective_tree_sha256": identity["prospective_tree_sha256"],
        "data_sha256": identity["data_sha256"],
        "environment_identity": environment,
        "environment_sha256": identity["environment_sha256"],
        "stage5_gate_sha256": "5" * 64,
        **_review_authorization_immutable_bindings(authorization),
    }


def _checkpoint_lineage(immutable: dict[str, object]) -> dict[str, object]:
    return {
        "stage5_commit": STAGE5_COMMIT,
        "stage5_gate_sha256": immutable["stage5_gate_sha256"],
        "stage5_manifest_sha256": "6" * 64,
        "stage4_checkpoint_sha256": "7" * 64,
        "stage4_policy_state_sha256": "8" * 64,
        "source_set_sha256": immutable["source_set_sha256"],
        "prospective_tree_sha256": immutable["prospective_tree_sha256"],
        "data_sha256": immutable["data_sha256"],
        "environment_identity": immutable["environment_identity"],
        "environment_sha256": immutable["environment_sha256"],
        "formal_run_id": immutable["formal_run_id"],
        "changed_path_set_sha256": immutable["changed_path_set_sha256"],
        "review_authorization_record_sha256": immutable[
            "review_authorization_record_sha256"
        ],
        "authorization_file_sha256": immutable["authorization_file_sha256"],
        "review_identity_sha256": immutable["review_identity_sha256"],
        "reviewed_prospective_git_tree": immutable[
            "reviewed_prospective_git_tree"
        ],
        "frozen_diff_sha256": immutable["frozen_diff_sha256"],
        "spec_review_sha256": immutable["spec_review_sha256"],
        "quality_review_sha256": immutable["quality_review_sha256"],
        "safety_contract_sha256": "9" * 64,
    }


def _resource_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "schema_version": "stage6_resource_lifecycle_segment/v1",
            "phase": "segment_start",
            "segment_id": "segment-1",
            "segment_index": 1,
        }
    ]
    for update in range(1, 10):
        key = f"{update - 1:04d}:{SEED}:update:{update:03d}"
        pre = {
            "schema_version": "stage6_resource_attempt/v1",
            "kind": "update",
            "transaction_key": key,
            "seed": SEED,
            "update": update,
            "attempt": 1,
            "phase": "pre",
            "accepted": False,
            "resource": {"passed": True},
            "segment_id": "segment-1",
            "segment_index": 1,
        }
        post = {**pre, "phase": "post"}
        accepted = {
            "schema_version": "stage6_resource_acceptance/v2",
            "kind": "update",
            "transaction_key": key,
            "seed": SEED,
            "update": update,
            "attempt": 1,
            "phase": "accepted",
            "accepted": True,
            "pre": {"passed": True},
            "post": {"passed": True},
            "checkpoint": {"checkpoint_sha256": f"{update:x}" * 64},
            "segment_id": "segment-1",
            "segment_index": 1,
        }
        rows.extend((pre, post, accepted))
    rows.append(
        {
            "schema_version": "stage6_resource_attempt/v1",
            "kind": "update",
            "transaction_key": f"0009:{SEED}:update:010",
            "seed": SEED,
            "update": 10,
            "attempt": 1,
            "phase": "pre",
            "accepted": False,
            "resource": {"passed": True},
            "segment_id": "segment-1",
            "segment_index": 1,
        }
    )
    return rows


def _device_failure_resource_rows(
    prior_payload: bytes,
) -> list[dict[str, object]]:
    return [
        {
            "schema_version": "stage6_resource_lifecycle_segment/v1",
            "kind": "resource_lifecycle_segment",
            "phase": "segment_start",
            "segment_id": "segment-2",
            "segment_index": 2,
            "root_pid": 200,
            "prior_resource_log": {
                "sha256": _sha(prior_payload),
                "size_bytes": len(prior_payload),
            },
            "first_sample": {"passed": True},
        },
        {
            "schema_version": "stage6_resource_attempt/v1",
            "kind": "update",
            "transaction_key": f"0009:{SEED}:update:010",
            "seed": SEED,
            "update": 10,
            "attempt": 2,
            "phase": "pre",
            "accepted": False,
            "resource": {"passed": True},
            "segment_id": "segment-2",
            "segment_index": 2,
        },
    ]


def _jsonl_payload(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> bytes:
    payload = _jsonl_payload(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _environment_identity() -> dict[str, object]:
    return {
        "schema_version": "stage6_environment_identity/v1",
        "python_version": "3.12.13",
        "python_implementation": "CPython",
        "os_name": "nt",
        "platform_system": "Windows",
        "platform_machine": "AMD64",
        "numpy_version": "2.2.6",
        "torch_version": "2.12.1+cu130",
        "torch_cuda_version": "13.0",
        "cudnn_version": 91002,
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_current_device": 0,
        "cuda_device_name": "NVIDIA test GPU",
        "cuda_compute_capability": [12, 0],
        "cuda_total_vram_bytes": 16 * 1024**3,
        "compute_dtype": "float32",
        "amp_enabled": False,
    }


@pytest.fixture(autouse=True)
def sensor_acceleration_windows_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model the frozen Windows Stage 6 sensor-acceleration workflow host."""
    monkeypatch.setattr(source_repair.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        source_repair,
        "_sensor_acceleration_process_parent_map",
        lambda: {os.getpid(): 1},
    )


@pytest.fixture
def repair_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_root = tmp_path / RUN_ID
    stage = run_root / "s6"
    stage.mkdir(parents=True)
    config_payload = CONFIG.read_bytes()
    (stage / "config.json").write_bytes(config_payload)
    config_sha256 = _sha(config_payload)
    environment = _environment_identity()
    old_identity = _identity(
        tree="1" * 40,
        source_sha256="a" * 64,
        config_sha256=config_sha256,
        data_sha256="b" * 64,
        environment=environment,
    )
    current_identity = _identity(
        tree="2" * 40,
        source_sha256="c" * 64,
        config_sha256=config_sha256,
        data_sha256="b" * 64,
        environment=environment,
    )
    latest_identity = _identity(
        tree="3" * 40,
        source_sha256="d" * 64,
        config_sha256=config_sha256,
        data_sha256="b" * 64,
        environment=environment,
    )
    supplement_identity = _identity(
        tree="4" * 40,
        source_sha256="e" * 64,
        config_sha256=config_sha256,
        data_sha256="b" * 64,
        environment=environment,
    )
    closure_identity = _identity(
        tree="5" * 40,
        source_sha256="f" * 64,
        config_sha256=config_sha256,
        data_sha256="b" * 64,
        environment=environment,
    )
    frontier_recovery_identity = _identity(
        tree="6" * 40,
        source_sha256="0" * 64,
        config_sha256=config_sha256,
        data_sha256="b" * 64,
        environment=environment,
    )
    sensor_acceleration_identity = _identity(
        tree="7" * 40,
        source_sha256="1" * 64,
        config_sha256=config_sha256,
        data_sha256="b" * 64,
        environment=environment,
    )
    old_authorization = _authorization(old_identity, marker="a")
    current_authorization = _authorization(current_identity, marker="b")
    latest_authorization = _authorization(latest_identity, marker="c")
    supplement_authorization = _authorization(supplement_identity, marker="d")
    closure_authorization = _authorization(closure_identity, marker="e")
    frontier_recovery_authorization = _authorization(
        frontier_recovery_identity,
        marker="f",
    )
    sensor_acceleration_authorization = _authorization(
        sensor_acceleration_identity,
        marker="1",
    )
    old_immutable = _immutable(
        old_identity, old_authorization, environment=environment
    )
    current_immutable = _immutable(
        current_identity, current_authorization, environment=environment
    )
    latest_immutable = _immutable(
        latest_identity, latest_authorization, environment=environment
    )
    supplement_immutable = _immutable(
        supplement_identity,
        supplement_authorization,
        environment=environment,
    )
    closure_immutable = _immutable(
        closure_identity,
        closure_authorization,
        environment=environment,
    )
    frontier_recovery_immutable = _immutable(
        frontier_recovery_identity,
        frontier_recovery_authorization,
        environment=environment,
    )
    sensor_acceleration_immutable = _immutable(
        sensor_acceleration_identity,
        sensor_acceleration_authorization,
        environment=environment,
    )
    old_lineage = _checkpoint_lineage(old_immutable)
    lineage_payload = _write_json(
        stage / "lineage_audit.json",
        {
            "schema_version": "stage6_lineage_audit/v2",
            "execution_identity": old_identity,
            "immutable_bindings": old_immutable,
            "verified_review_authorization": old_authorization,
        },
    )

    config = load_stage6_config(CONFIG)
    transactions = build_standard_training_transactions(config)
    checkpoint_payload = b"test-update-9-checkpoint"
    policy_sha256 = _sha(b"policy-9")
    manifest_value = {
        "schema_version": "stage6_checkpoint_manifest/v1",
        "update_step": 9,
        "checkpoint": {
            "path": "checkpoint.pt",
            "sha256": _sha(checkpoint_payload),
            "size_bytes": len(checkpoint_payload),
        },
        "policy_state_sha256": policy_sha256,
        "config_sha256": config_sha256,
        "lineage_sha256": _sha(_canonical(old_lineage)),
    }
    manifest_payload = _canonical(manifest_value)
    complete_value = {
        "schema_version": "stage6_checkpoint_complete/v1",
        "update_step": 9,
        "manifest_sha256": _sha(manifest_payload),
        "checkpoint_sha256": _sha(checkpoint_payload),
    }
    complete_payload = _canonical(complete_value)
    receipt_index = CheckpointReceiptIndex(stage / "checkpoints/index.jsonl")
    journal = Stage6StateJournal(stage / "job-state.jsonl")
    receipts: list[StandardTransactionCheckpoint] = []
    for transaction in transactions[:9]:
        checkpoint = StandardTransactionCheckpoint(
            transaction_key=transaction.key,
            seed=transaction.seed,
            update=transaction.update,
            checkpoint_sha256=(
                _sha(checkpoint_payload)
                if transaction.update == 9
                else _sha(f"checkpoint-{transaction.update}".encode())
            ),
            complete_marker_sha256=(
                _sha(complete_payload)
                if transaction.update == 9
                else _sha(f"complete-{transaction.update}".encode())
            ),
            policy_state_sha256=(
                policy_sha256
                if transaction.update == 9
                else _sha(f"policy-{transaction.update}".encode())
            ),
        )
        receipt_index.append_once(checkpoint)
        bindings = {**old_immutable, "checkpoint_sha256": checkpoint.checkpoint_sha256}
        for state in transaction.commit_states:
            journal.append(state, bindings)
        receipts.append(checkpoint)
    phase = Stage6StateJournal(stage / "phase-state.jsonl")
    phase.append(
        "preflight",
        {**old_immutable, "checkpoint_sha256": "7" * 64},
    )
    resource_payload = _write_jsonl(stage / "resource_audit.jsonl", _resource_rows())
    _write_jsonl(
        stage / "training_metrics.jsonl",
        [{"update": update} for update in range(1, 10)],
    )
    _write_jsonl(stage / "math_audit.jsonl", [{"accepted_updates": 9}])
    _write_jsonl(stage / "checkpoint_audit.jsonl", [{"accepted_updates": 9}])

    checkpoint_directory = stage / f"checkpoints/seed-{SEED}/update-00000009"
    checkpoint_directory.mkdir(parents=True)
    (checkpoint_directory / "checkpoint.pt").write_bytes(checkpoint_payload)
    (checkpoint_directory / "manifest.json").write_bytes(manifest_payload)
    (checkpoint_directory / "complete.json").write_bytes(complete_payload)
    latest_payload = _write_json(
        checkpoint_directory.parent / "latest.json",
        {
            "schema_version": "stage6_checkpoint_latest/v1",
            "update_step": 9,
            "directory": "update-00000009",
            "checkpoint_sha256": _sha(checkpoint_payload),
            "manifest_sha256": _sha(manifest_payload),
            "policy_state_sha256": policy_sha256,
        },
    )

    evidence_root = tmp_path / "review"
    evidence_root.mkdir()
    evidence_paths: dict[str, Path] = {}
    for kind, payload in {
        "repair_diff": b"reviewed repair diff\n",
        "review_report": b"review pass\n",
        "update10_replay": _canonical({"trainable_transition_count": 1024}),
    }.items():
        path = evidence_root / f"{kind}.txt"
        path.write_bytes(payload)
        evidence_paths[kind] = path

    prefix_paths = (
        "config.json",
        "lineage_audit.json",
        "checkpoints/index.jsonl",
        "job-state.jsonl",
        "phase-state.jsonl",
        "resource_audit.jsonl",
        "training_metrics.jsonl",
        "math_audit.jsonl",
        "checkpoint_audit.jsonl",
        f"checkpoints/seed-{SEED}/update-00000009/checkpoint.pt",
        f"checkpoints/seed-{SEED}/update-00000009/manifest.json",
        f"checkpoints/seed-{SEED}/update-00000009/complete.json",
    )
    fixed_prefix = {
        relative: {
            "path": relative,
            "sha256": _sha((stage / relative).read_bytes()),
            "size_bytes": (stage / relative).stat().st_size,
        }
        for relative in prefix_paths
    }
    fixed_evidence = {
        kind: {
            "kind": kind,
            "path": str(path.resolve()),
            "sha256": _sha(path.read_bytes()),
            "size_bytes": path.stat().st_size,
        }
        for kind, path in evidence_paths.items()
    }
    monkeypatch.setattr(source_repair, "_FIXED_PREFIX_BINDINGS", fixed_prefix)
    monkeypatch.setattr(source_repair, "_FIXED_EVIDENCE_BINDINGS", fixed_evidence)
    monkeypatch.setattr(
        source_repair,
        "_FIXED_LINEAGE_AUDIT_SHA256",
        _sha(lineage_payload),
    )
    monkeypatch.setattr(
        source_repair,
        "_FIXED_ORIGIN_IMMUTABLE_SHA256",
        _sha(_canonical(old_immutable)),
    )
    monkeypatch.setattr(
        source_repair,
        "_FIXED_ORIGIN_CHECKPOINT_LINEAGE_SHA256",
        _sha(_canonical(old_lineage)),
    )
    checkpoint_payload_values: dict[bytes, dict[str, object]] = {
        checkpoint_payload: {
            "update_step": 9,
            "config_sha256": config_sha256,
            "lineage": old_lineage,
            "policy_state_sha256": policy_sha256,
        }
    }
    monkeypatch.setattr(
        source_repair,
        "_read_cutover_checkpoint_payload",
        lambda payload: checkpoint_payload_values[payload],
    )
    fixture_primary_payload = _canonical(
        source_repair._amendment_value(
            stage=stage,
            current_execution_identity=current_identity,
            current_verified_review_authorization=current_authorization,
            current_immutable_bindings=current_immutable,
            allow_growth=False,
        )
    )
    monkeypatch.setattr(
        source_repair,
        "_FIXED_PRIMARY_AMENDMENT_BINDING",
        {
            "path": source_repair.SOURCE_REPAIR_AMENDMENT_NAME,
            "sha256": _sha(fixture_primary_payload),
            "size_bytes": len(fixture_primary_payload),
        },
    )
    return {
        "stage": stage,
        "old_identity": old_identity,
        "old_authorization": old_authorization,
        "old_immutable": old_immutable,
        "old_lineage": old_lineage,
        "current_identity": current_identity,
        "current_authorization": current_authorization,
        "current_immutable": current_immutable,
        "latest_identity": latest_identity,
        "latest_authorization": latest_authorization,
        "latest_immutable": latest_immutable,
        "supplement_identity": supplement_identity,
        "supplement_authorization": supplement_authorization,
        "supplement_immutable": supplement_immutable,
        "closure_identity": closure_identity,
        "closure_authorization": closure_authorization,
        "closure_immutable": closure_immutable,
        "frontier_recovery_identity": frontier_recovery_identity,
        "frontier_recovery_authorization": frontier_recovery_authorization,
        "frontier_recovery_immutable": frontier_recovery_immutable,
        "sensor_acceleration_identity": sensor_acceleration_identity,
        "sensor_acceleration_authorization": sensor_acceleration_authorization,
        "sensor_acceleration_immutable": sensor_acceleration_immutable,
        "resource_payload": resource_payload,
        "evidence_paths": evidence_paths,
        "fixed_prefix": fixed_prefix,
        "checkpoint_payload_values": checkpoint_payload_values,
        "monkeypatch": monkeypatch,
    }


def _create(fixture: dict[str, object]):
    return source_repair.create_stage6_source_repair_amendment(
        stage_root=fixture["stage"],
        current_execution_identity=fixture["current_identity"],
        current_verified_review_authorization=fixture["current_authorization"],
        current_immutable_bindings=fixture["current_immutable"],
        publish=True,
    )


def _record_device_failure(fixture: dict[str, object], primary) -> None:
    stage = fixture["stage"]
    monkeypatch = fixture["monkeypatch"]
    assert isinstance(stage, Path)
    assert isinstance(monkeypatch, pytest.MonkeyPatch)
    resource_rows = _resource_rows()
    resource_rows.extend(
        _device_failure_resource_rows(fixture["resource_payload"])
    )
    resource_payload = _write_jsonl(stage / "resource_audit.jsonl", resource_rows)
    continuation_prefix = copy.deepcopy(fixture["fixed_prefix"])
    continuation_prefix["resource_audit.jsonl"] = {
        "path": "resource_audit.jsonl",
        "sha256": _sha(resource_payload),
        "size_bytes": len(resource_payload),
    }

    stderr_path = stage.parent.parent / "device-failure.stderr.log"
    stderr_payload = b"StandardEvaluationError: PPO evaluation policy device drifted\n"
    stderr_path.write_bytes(stderr_payload)
    device_evidence = {
        "schema_version": "stage6_update10_validation_device_failure/v1",
        "formal_run_id": RUN_ID,
        "accepted_prefix_updates": 9,
        "failed_update": 10,
        "failed_attempt": 2,
        "failure_phase": "validation",
        "requested_policy_device": "cuda",
        "actual_parameter_device": "cuda:0",
        "cuda_current_device": 0,
        "strict_torch_device_equality": False,
        "root_cause": "bare_cuda_alias_compared_by_strict_torch_device_equality/v1",
        "exception": "StandardEvaluationError: PPO evaluation policy device drifted",
        "formal_side_effects": {
            "resource_pre_attempt2_written": True,
            "update10_checkpoint_receipt_written": False,
            "update10_training_metric_written": False,
        },
        "parent_amendment": {
            "path": (stage / source_repair.SOURCE_REPAIR_AMENDMENT_NAME).as_posix(),
            "sha256": primary.amendment_sha256,
            "size_bytes": primary.amendment_size_bytes,
        },
        "resource_prefix": {
            "path": (stage / "resource_audit.jsonl").as_posix(),
            "sha256": _sha(resource_payload),
            "size_bytes": len(resource_payload),
        },
        "stderr": {
            "path": stderr_path.as_posix(),
            "sha256": _sha(stderr_payload),
            "size_bytes": len(stderr_payload),
        },
    }
    evidence_path = stage.parent.parent / "device-failure-evidence.json"
    evidence_payload = _write_json(evidence_path, device_evidence)
    continuation_evidence = {
        "device_failure": {
            "kind": "device_failure",
            "path": evidence_path.as_posix(),
            "sha256": _sha(evidence_payload),
            "size_bytes": len(evidence_payload),
        },
        "stderr": {
            "kind": "stderr",
            "path": stderr_path.as_posix(),
            "sha256": _sha(stderr_payload),
            "size_bytes": len(stderr_payload),
        },
    }
    monkeypatch.setattr(
        source_repair,
        "_FIXED_PRIMARY_AMENDMENT_BINDING",
        {
            "path": source_repair.SOURCE_REPAIR_AMENDMENT_NAME,
            "sha256": primary.amendment_sha256,
            "size_bytes": primary.amendment_size_bytes,
        },
        raising=False,
    )
    monkeypatch.setattr(
        source_repair,
        "_CONTINUATION_PREFIX_BINDINGS",
        continuation_prefix,
        raising=False,
    )
    monkeypatch.setattr(
        source_repair,
        "_CONTINUATION_EVIDENCE_BINDINGS",
        continuation_evidence,
        raising=False,
    )
    fixture_continuation_payload = _canonical(
        source_repair._continuation_value(
            stage=stage,
            primary_context=primary,
            primary_payload=(
                stage / source_repair.SOURCE_REPAIR_AMENDMENT_NAME
            ).read_bytes(),
            current_execution_identity=fixture["latest_identity"],
            current_verified_review_authorization=fixture["latest_authorization"],
            current_immutable_bindings=fixture["latest_immutable"],
            allow_growth=False,
        )
    )
    monkeypatch.setattr(
        source_repair,
        "_FIXED_CONTINUATION_BINDING",
        {
            "path": source_repair.SOURCE_REPAIR_CONTINUATION_NAME,
            "sha256": _sha(fixture_continuation_payload),
            "size_bytes": len(fixture_continuation_payload),
        },
        raising=False,
    )
    monkeypatch.setattr(
        source_repair,
        "_FIXED_CONTINUATION_EXECUTION_IDENTITY_SHA256",
        _sha(_canonical(fixture["latest_identity"])),
        raising=False,
    )
    fixture["continuation_prefix"] = continuation_prefix
    fixture["continuation_evidence_paths"] = {
        "device_failure": evidence_path,
        "stderr": stderr_path,
    }


def _create_continuation(fixture: dict[str, object]):
    primary = _create(fixture)
    _record_device_failure(fixture, primary)
    return source_repair.create_stage6_source_repair_amendment(
        stage_root=fixture["stage"],
        current_execution_identity=fixture["latest_identity"],
        current_verified_review_authorization=fixture["latest_authorization"],
        current_immutable_bindings=fixture["latest_immutable"],
        publish=True,
    )


def _accepted_resource_rows_after_second_cutover(
    prior_payload: bytes,
    *,
    checkpoint_sha256_by_update: dict[int, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "schema_version": "stage6_resource_lifecycle_segment/v1",
            "kind": "resource_lifecycle_segment",
            "phase": "segment_start",
            "segment_id": "segment-3",
            "segment_index": 3,
            "root_pid": 300,
            "prior_resource_log": {
                "sha256": _sha(prior_payload),
                "size_bytes": len(prior_payload),
            },
            "first_sample": {"passed": True},
        }
    ]
    for update in range(10, 33):
        attempt = 3 if update == 10 else 1
        key = f"{update - 1:04d}:{SEED}:update:{update:03d}"
        pre = {
            "schema_version": "stage6_resource_attempt/v1",
            "kind": "update",
            "transaction_key": key,
            "seed": SEED,
            "update": update,
            "attempt": attempt,
            "phase": "pre",
            "accepted": False,
            "resource": {"passed": True},
            "segment_id": "segment-3",
            "segment_index": 3,
        }
        rows.extend(
            (
                pre,
                {**pre, "phase": "post"},
                {
                    "schema_version": "stage6_resource_acceptance/v2",
                    "kind": "update",
                    "transaction_key": key,
                    "seed": SEED,
                    "update": update,
                    "attempt": attempt,
                    "phase": "accepted",
                    "accepted": True,
                    "pre": {"passed": True},
                    "post": {"passed": True},
                    "checkpoint": {
                        "checkpoint_sha256": checkpoint_sha256_by_update[update]
                    },
                    "segment_id": "segment-3",
                    "segment_index": 3,
                },
            )
        )
    rows.append(
        {
            "schema_version": "stage6_resource_attempt/v1",
            "kind": "update",
            "transaction_key": f"0032:{SEED}:update:033",
            "seed": SEED,
            "update": 33,
            "attempt": 1,
            "phase": "pre",
            "accepted": False,
            "resource": {"passed": True},
            "segment_id": "segment-3",
            "segment_index": 3,
        }
    )
    return rows


def _record_initial_ratio_failure(
    fixture: dict[str, object],
    continuation,
) -> None:
    stage = fixture["stage"]
    monkeypatch = fixture["monkeypatch"]
    assert isinstance(stage, Path)
    assert isinstance(monkeypatch, pytest.MonkeyPatch)

    config = load_stage6_config(CONFIG)
    transactions = build_standard_training_transactions(config)
    receipt_index = CheckpointReceiptIndex(stage / "checkpoints/index.jsonl")
    journal = Stage6StateJournal(stage / "job-state.jsonl")

    checkpoint32_payload = b"test-update-32-checkpoint"
    policy32_sha256 = _sha(b"policy-32")
    checkpoint32_lineage = continuation.checkpoint_lineage_for_update(32)
    manifest32_payload = _canonical(
        {
            "schema_version": "stage6_checkpoint_manifest/v1",
            "update_step": 32,
            "checkpoint": {
                "path": "checkpoint.pt",
                "sha256": _sha(checkpoint32_payload),
                "size_bytes": len(checkpoint32_payload),
            },
            "policy_state_sha256": policy32_sha256,
            "config_sha256": fixture["latest_identity"]["config_sha256"],
            "lineage_sha256": _sha(_canonical(checkpoint32_lineage)),
        }
    )
    complete32_payload = _canonical(
        {
            "schema_version": "stage6_checkpoint_complete/v1",
            "update_step": 32,
            "manifest_sha256": _sha(manifest32_payload),
            "checkpoint_sha256": _sha(checkpoint32_payload),
        }
    )
    checkpoint_sha256_by_update: dict[int, str] = {}
    for transaction in transactions[9:32]:
        is_cutover = transaction.update == 32
        receipt = StandardTransactionCheckpoint(
            transaction_key=transaction.key,
            seed=transaction.seed,
            update=transaction.update,
            checkpoint_sha256=(
                _sha(checkpoint32_payload)
                if is_cutover
                else _sha(f"checkpoint-{transaction.update}".encode())
            ),
            complete_marker_sha256=(
                _sha(complete32_payload)
                if is_cutover
                else _sha(f"complete-{transaction.update}".encode())
            ),
            policy_state_sha256=(
                policy32_sha256
                if is_cutover
                else _sha(f"policy-{transaction.update}".encode())
            ),
        )
        receipt_index.append_once(receipt)
        bindings = {
            **fixture["old_immutable"],
            "checkpoint_sha256": receipt.checkpoint_sha256,
        }
        for state in transaction.commit_states:
            journal.append(state, bindings)
        checkpoint_sha256_by_update[transaction.update] = receipt.checkpoint_sha256

    checkpoint32_directory = stage / f"checkpoints/seed-{SEED}/update-00000032"
    checkpoint32_directory.mkdir(parents=True)
    (checkpoint32_directory / "checkpoint.pt").write_bytes(checkpoint32_payload)
    (checkpoint32_directory / "manifest.json").write_bytes(manifest32_payload)
    (checkpoint32_directory / "complete.json").write_bytes(complete32_payload)
    latest32_payload = _write_json(
        checkpoint32_directory.parent / "latest.json",
        {
            "schema_version": "stage6_checkpoint_latest/v1",
            "update_step": 32,
            "directory": "update-00000032",
            "checkpoint_sha256": _sha(checkpoint32_payload),
            "manifest_sha256": _sha(manifest32_payload),
            "policy_state_sha256": policy32_sha256,
        },
    )
    fixture["checkpoint_payload_values"][checkpoint32_payload] = {
        "update_step": 32,
        "config_sha256": fixture["latest_identity"]["config_sha256"],
        "lineage": checkpoint32_lineage,
        "policy_state_sha256": policy32_sha256,
    }

    training_payload = _write_jsonl(
        stage / "training_metrics.jsonl",
        [{"update": update} for update in range(1, 33)],
    )
    validation_payload = _write_jsonl(
        stage / "validation_metrics.jsonl",
        [{"update": update} for update in (10, 20, 30)],
    )
    math_payload = _write_jsonl(
        stage / "math_audit.jsonl",
        [{"accepted_updates": update} for update in (9, 32)],
    )
    checkpoint_audit_payload = _write_jsonl(
        stage / "checkpoint_audit.jsonl",
        [{"accepted_updates": update} for update in (9, 32)],
    )
    resource_rows = _resource_rows()
    resource_rows.extend(_device_failure_resource_rows(fixture["resource_payload"]))
    prior_r14_payload = _jsonl_payload(resource_rows)
    resource_rows.extend(
        _accepted_resource_rows_after_second_cutover(
            prior_r14_payload,
            checkpoint_sha256_by_update=checkpoint_sha256_by_update,
        )
    )
    resource_payload = _write_jsonl(stage / "resource_audit.jsonl", resource_rows)

    supplement_prefix = copy.deepcopy(fixture["continuation_prefix"])
    for relative, payload in {
        "checkpoints/index.jsonl": (stage / "checkpoints/index.jsonl").read_bytes(),
        "job-state.jsonl": (stage / "job-state.jsonl").read_bytes(),
        "phase-state.jsonl": (stage / "phase-state.jsonl").read_bytes(),
        "resource_audit.jsonl": resource_payload,
        "training_metrics.jsonl": training_payload,
        "validation_metrics.jsonl": validation_payload,
        "math_audit.jsonl": math_payload,
        "checkpoint_audit.jsonl": checkpoint_audit_payload,
        f"checkpoints/seed-{SEED}/update-00000032/checkpoint.pt": checkpoint32_payload,
        f"checkpoints/seed-{SEED}/update-00000032/manifest.json": manifest32_payload,
        f"checkpoints/seed-{SEED}/update-00000032/complete.json": complete32_payload,
    }.items():
        supplement_prefix[relative] = {
            "path": relative,
            "sha256": _sha(payload),
            "size_bytes": len(payload),
        }
    monkeypatch.setattr(
        source_repair,
        "_SUPPLEMENT_PREFIX_BINDINGS",
        supplement_prefix,
        raising=False,
    )
    monkeypatch.setattr(
        source_repair,
        "_FIXED_CONTINUATION_BINDING",
        {
            "path": source_repair.SOURCE_REPAIR_CONTINUATION_NAME,
            "sha256": continuation.continuation_sha256,
            "size_bytes": continuation.continuation_size_bytes,
        },
        raising=False,
    )
    monkeypatch.setattr(
        source_repair,
        "_FIXED_CONTINUATION_EXECUTION_IDENTITY_SHA256",
        _sha(_canonical(fixture["latest_identity"])),
        raising=False,
    )

    stderr_path = stage.parent.parent / "initial-ratio-failure.stderr.log"
    stderr_payload = (
        b"lunar_exploration_ppo.ppo.trainer.PPOTrainingError: initial ratio "
        b"mismatch: max_abs_error=1.26361847e-05\n"
    )
    stderr_path.write_bytes(stderr_payload)
    monkeypatch.setattr(
        source_repair,
        "_SUPPLEMENT_EVIDENCE_BINDINGS",
        {
            "stderr": {
                "kind": "stderr",
                "path": stderr_path.as_posix(),
                "sha256": _sha(stderr_payload),
                "size_bytes": len(stderr_payload),
            }
        },
        raising=False,
    )
    fixture["supplement_prefix"] = supplement_prefix
    fixture["supplement_stderr_path"] = stderr_path
    fixture["checkpoint32_lineage"] = checkpoint32_lineage
    fixture["checkpoint32_payload"] = checkpoint32_payload
    fixture["checkpoint32_manifest_payload"] = manifest32_payload
    fixture["checkpoint32_complete_payload"] = complete32_payload
    fixture["latest32_payload"] = latest32_payload


def _prepare_supplement_cutover(fixture: dict[str, object]):
    continuation = _create_continuation(fixture)
    _record_initial_ratio_failure(fixture, continuation)
    return continuation


def _create_supplement(fixture: dict[str, object]):
    _prepare_supplement_cutover(fixture)
    return source_repair.create_stage6_source_repair_amendment(
        stage_root=fixture["stage"],
        current_execution_identity=fixture["supplement_identity"],
        current_verified_review_authorization=fixture["supplement_authorization"],
        current_immutable_bindings=fixture["supplement_immutable"],
        publish=True,
    )


def _closure_resource_rows(prior_payload: bytes) -> list[dict[str, object]]:
    return [
        {
            "schema_version": "stage6_resource_lifecycle_segment/v1",
            "kind": "resource_lifecycle_segment",
            "phase": "segment_start",
            "segment_id": "segment-4",
            "segment_index": 4,
            "root_pid": 400,
            "prior_resource_log": {
                "sha256": _sha(prior_payload),
                "size_bytes": len(prior_payload),
            },
            "first_sample": {"passed": True},
        },
        {
            "schema_version": "stage6_resource_attempt/v1",
            "kind": "update",
            "transaction_key": f"0032:{SEED}:update:033",
            "seed": SEED,
            "update": 33,
            "attempt": 2,
            "phase": "pre",
            "accepted": False,
            "resource": {"passed": True},
            "segment_id": "segment-4",
            "segment_index": 4,
        },
    ]


def _record_proxy_safety_closure_cutover(
    fixture: dict[str, object],
    supplement,
) -> None:
    stage = fixture["stage"]
    monkeypatch = fixture["monkeypatch"]
    assert isinstance(stage, Path)
    assert isinstance(monkeypatch, pytest.MonkeyPatch)

    prior_resource_payload = (stage / "resource_audit.jsonl").read_bytes()
    rows = [
        json.loads(line)
        for line in prior_resource_payload.decode("utf-8").splitlines()
    ]
    rows.extend(_closure_resource_rows(prior_resource_payload))
    resource_payload = _write_jsonl(stage / "resource_audit.jsonl", rows)

    closure_prefix = copy.deepcopy(fixture["supplement_prefix"])
    closure_prefix["resource_audit.jsonl"] = {
        "path": "resource_audit.jsonl",
        "sha256": _sha(resource_payload),
        "size_bytes": len(resource_payload),
    }
    monkeypatch.setattr(
        source_repair,
        "_CLOSURE_PREFIX_BINDINGS",
        closure_prefix,
        raising=False,
    )
    monkeypatch.setattr(
        source_repair,
        "_FIXED_SUPPLEMENT_BINDING",
        {
            "path": source_repair.SOURCE_REPAIR_SUPPLEMENT_NAME,
            "sha256": supplement.supplement_sha256,
            "size_bytes": supplement.supplement_size_bytes,
        },
        raising=False,
    )
    monkeypatch.setattr(
        source_repair,
        "_FIXED_SUPPLEMENT_EXECUTION_IDENTITY_SHA256",
        _sha(_canonical(fixture["supplement_identity"])),
        raising=False,
    )

    finding_value = {
        "schema_version": "stage6_proxy_safety_gate_finding_review/v1",
        "finding_id": "stage6_terminal_proxy_safety_gate_contract_conflict/v1",
        "review_kind": "fresh_independent_contract_review/v1",
        "severity": "Critical",
        "verdict": "REJECT",
        "required_repair_semantics": {
            "classify_as_adverse_synthetic_proxy_finding": True,
            "machine_blocking_only_if_evidence_contract_fails": True,
            "physical_obstacle_cells_written": False,
            "physical_safety_claim_allowed": False,
            "preserve_per_split_method_and_total_counts": True,
            "preserve_safety_done_termination_classification": True,
            "preserve_true_safety_violation_count": True,
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        },
    }
    safe_stop_value = {
        "schema_version": "stage6_r15_safe_stop_audit/v1",
        "formal_run_id": RUN_ID,
        "recovery": "r15",
        "reason": "fresh_independent_review_critical_terminal_contract_conflict/v1",
        "source_repair_supplement_sha256": supplement.supplement_sha256,
        "formal_prefix": {
            "accepted_update_count": 32,
            "last_accepted_update": 32,
            "job_row_count": 36,
            "training_row_count": 32,
            "resource_row_count": 104,
            "resource_audit": closure_prefix["resource_audit.jsonl"],
            "last_resource_row": {
                "accepted": False,
                "attempt": 2,
                "phase": "pre",
                "segment_index": 4,
                "transaction_key": f"0032:{SEED}:update:033",
                "update": 33,
            },
            "stage6_lease_exists": False,
            "update33_checkpoint_exists": False,
        },
        "process_exit": {
            "duplicate_runner_started": False,
            "remaining_child_count_after_cleanup": 0,
            "runner_count_after_stop": 0,
        },
    }
    evidence_root = stage.parent.parent / "proxy-safety-review"
    evidence_root.mkdir(exist_ok=True)
    finding_path = evidence_root / "finding-review.json"
    safe_stop_path = evidence_root / "safe-stop-audit.json"
    finding_payload = _write_json(finding_path, finding_value)
    safe_stop_payload = _write_json(safe_stop_path, safe_stop_value)
    closure_evidence = {
        "finding_review": {
            "kind": "finding_review",
            "path": finding_path.as_posix(),
            "sha256": _sha(finding_payload),
            "size_bytes": len(finding_payload),
        },
        "safe_stop": {
            "kind": "safe_stop",
            "path": safe_stop_path.as_posix(),
            "sha256": _sha(safe_stop_payload),
            "size_bytes": len(safe_stop_payload),
        },
    }
    monkeypatch.setattr(
        source_repair,
        "_CLOSURE_EVIDENCE_BINDINGS",
        closure_evidence,
        raising=False,
    )
    fixture["closure_prefix"] = closure_prefix
    fixture["closure_evidence_paths"] = {
        "finding_review": finding_path,
        "safe_stop": safe_stop_path,
    }
    fixture["closure_resource_payload"] = resource_payload


def _prepare_closure_cutover(fixture: dict[str, object]):
    supplement = _create_supplement(fixture)
    _record_proxy_safety_closure_cutover(fixture, supplement)
    return supplement


def _create_closure(fixture: dict[str, object], *, publish: bool = True):
    _prepare_closure_cutover(fixture)
    return source_repair.create_stage6_source_repair_amendment(
        stage_root=fixture["stage"],
        current_execution_identity=fixture["closure_identity"],
        current_verified_review_authorization=fixture["closure_authorization"],
        current_immutable_bindings=fixture["closure_immutable"],
        publish=publish,
    )


def _accepted_resource_rows_after_closure(
    prior_payload: bytes,
    *,
    checkpoint_sha256_by_update: dict[int, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "schema_version": "stage6_resource_lifecycle_segment/v1",
            "kind": "resource_lifecycle_segment",
            "phase": "segment_start",
            "segment_id": "segment-5",
            "segment_index": 5,
            "root_pid": 500,
            "prior_resource_log": {
                "sha256": _sha(prior_payload),
                "size_bytes": len(prior_payload),
            },
            "first_sample": {"passed": True},
        }
    ]
    for update in range(33, 47):
        attempt = 3 if update == 33 else 1
        key = f"{update - 1:04d}:{SEED}:update:{update:03d}"
        pre = {
            "schema_version": "stage6_resource_attempt/v1",
            "kind": "update",
            "transaction_key": key,
            "seed": SEED,
            "update": update,
            "attempt": attempt,
            "phase": "pre",
            "accepted": False,
            "resource": {"passed": True},
            "segment_id": "segment-5",
            "segment_index": 5,
        }
        rows.extend(
            (
                pre,
                {**pre, "phase": "post"},
                {
                    "schema_version": "stage6_resource_acceptance/v2",
                    "kind": "update",
                    "transaction_key": key,
                    "seed": SEED,
                    "update": update,
                    "attempt": attempt,
                    "phase": "accepted",
                    "accepted": True,
                    "pre": {"passed": True},
                    "post": {"passed": True},
                    "checkpoint": {
                        "checkpoint_sha256": checkpoint_sha256_by_update[update]
                    },
                    "segment_id": "segment-5",
                    "segment_index": 5,
                },
            )
        )
    rows.append(
        {
            "schema_version": "stage6_resource_attempt/v1",
            "kind": "update",
            "transaction_key": f"0046:{SEED}:update:047",
            "seed": SEED,
            "update": 47,
            "attempt": 1,
            "phase": "pre",
            "accepted": False,
            "resource": {"passed": True},
            "segment_id": "segment-5",
            "segment_index": 5,
        }
    )
    return rows


def _record_update47_frontier_recovery_cutover(
    fixture: dict[str, object],
    closure,
) -> None:
    stage = fixture["stage"]
    monkeypatch = fixture["monkeypatch"]
    assert isinstance(stage, Path)
    assert isinstance(monkeypatch, pytest.MonkeyPatch)

    config = load_stage6_config(CONFIG)
    transactions = build_standard_training_transactions(config)
    receipt_index = CheckpointReceiptIndex(stage / "checkpoints/index.jsonl")
    journal = Stage6StateJournal(stage / "job-state.jsonl")

    checkpoint46_payload = b"test-update-46-checkpoint"
    policy46_sha256 = _sha(b"policy-46")
    checkpoint46_lineage = closure.checkpoint_lineage_for_update(46)
    manifest46_payload = _canonical(
        {
            "schema_version": "stage6_checkpoint_manifest/v1",
            "update_step": 46,
            "checkpoint": {
                "path": "checkpoint.pt",
                "sha256": _sha(checkpoint46_payload),
                "size_bytes": len(checkpoint46_payload),
            },
            "policy_state_sha256": policy46_sha256,
            "config_sha256": fixture["closure_identity"]["config_sha256"],
            "lineage_sha256": _sha(_canonical(checkpoint46_lineage)),
        }
    )
    complete46_payload = _canonical(
        {
            "schema_version": "stage6_checkpoint_complete/v1",
            "update_step": 46,
            "manifest_sha256": _sha(manifest46_payload),
            "checkpoint_sha256": _sha(checkpoint46_payload),
        }
    )
    checkpoint_sha256_by_update: dict[int, str] = {}
    for transaction in transactions[32:46]:
        is_cutover = transaction.update == 46
        receipt = StandardTransactionCheckpoint(
            transaction_key=transaction.key,
            seed=transaction.seed,
            update=transaction.update,
            checkpoint_sha256=(
                _sha(checkpoint46_payload)
                if is_cutover
                else _sha(f"checkpoint-{transaction.update}".encode())
            ),
            complete_marker_sha256=(
                _sha(complete46_payload)
                if is_cutover
                else _sha(f"complete-{transaction.update}".encode())
            ),
            policy_state_sha256=(
                policy46_sha256
                if is_cutover
                else _sha(f"policy-{transaction.update}".encode())
            ),
        )
        receipt_index.append_once(receipt)
        bindings = {
            **fixture["old_immutable"],
            "checkpoint_sha256": receipt.checkpoint_sha256,
        }
        for state in transaction.commit_states:
            journal.append(state, bindings)
        checkpoint_sha256_by_update[transaction.update] = receipt.checkpoint_sha256

    checkpoint46_directory = stage / f"checkpoints/seed-{SEED}/update-00000046"
    checkpoint46_directory.mkdir(parents=True)
    (checkpoint46_directory / "checkpoint.pt").write_bytes(checkpoint46_payload)
    (checkpoint46_directory / "manifest.json").write_bytes(manifest46_payload)
    (checkpoint46_directory / "complete.json").write_bytes(complete46_payload)
    latest46_payload = _write_json(
        checkpoint46_directory.parent / "latest.json",
        {
            "schema_version": "stage6_checkpoint_latest/v1",
            "update_step": 46,
            "directory": "update-00000046",
            "checkpoint_sha256": _sha(checkpoint46_payload),
            "manifest_sha256": _sha(manifest46_payload),
            "policy_state_sha256": policy46_sha256,
        },
    )
    fixture["checkpoint_payload_values"][checkpoint46_payload] = {
        "update_step": 46,
        "config_sha256": fixture["closure_identity"]["config_sha256"],
        "lineage": checkpoint46_lineage,
        "policy_state_sha256": policy46_sha256,
    }

    training_payload = _write_jsonl(
        stage / "training_metrics.jsonl",
        [{"update": update} for update in range(1, 47)],
    )
    validation_payload = _write_jsonl(
        stage / "validation_metrics.jsonl",
        [{"update": update} for update in (10, 20, 30, 40)],
    )
    prior_resource_payload = (stage / "resource_audit.jsonl").read_bytes()
    resource_rows = [
        json.loads(line)
        for line in prior_resource_payload.decode("utf-8").splitlines()
    ]
    resource_rows.extend(
        _accepted_resource_rows_after_closure(
            prior_resource_payload,
            checkpoint_sha256_by_update=checkpoint_sha256_by_update,
        )
    )
    resource_payload = _write_jsonl(stage / "resource_audit.jsonl", resource_rows)

    frontier_prefix = copy.deepcopy(fixture["closure_prefix"])
    for relative, payload in {
        "checkpoints/index.jsonl": (stage / "checkpoints/index.jsonl").read_bytes(),
        "job-state.jsonl": (stage / "job-state.jsonl").read_bytes(),
        "resource_audit.jsonl": resource_payload,
        "training_metrics.jsonl": training_payload,
        "validation_metrics.jsonl": validation_payload,
        f"checkpoints/seed-{SEED}/update-00000046/checkpoint.pt": (
            checkpoint46_payload
        ),
        f"checkpoints/seed-{SEED}/update-00000046/manifest.json": (
            manifest46_payload
        ),
        f"checkpoints/seed-{SEED}/update-00000046/complete.json": (
            complete46_payload
        ),
    }.items():
        frontier_prefix[relative] = {
            "path": relative,
            "sha256": _sha(payload),
            "size_bytes": len(payload),
        }
    monkeypatch.setattr(
        source_repair,
        "_FRONTIER_RECOVERY_PREFIX_BINDINGS",
        frontier_prefix,
        raising=False,
    )

    closure_path = stage / source_repair.SOURCE_REPAIR_CLOSURE_NAME
    closure_payload = closure_path.read_bytes()
    monkeypatch.setattr(
        source_repair,
        "_FIXED_CLOSURE_BINDING",
        {
            "path": source_repair.SOURCE_REPAIR_CLOSURE_NAME,
            "sha256": _sha(closure_payload),
            "size_bytes": len(closure_payload),
        },
        raising=False,
    )
    monkeypatch.setattr(
        source_repair,
        "_FIXED_CLOSURE_EXECUTION_IDENTITY_SHA256",
        _sha(_canonical(fixture["closure_identity"])),
        raising=False,
    )

    source_root = stage.parent.parent / "frontier-recovery-source"
    source_payloads = {
        "src/lunar_exploration_ppo/env/frontier.py": b"reviewed frontier source\n",
        "tests/ppo_highres_frontier/test_stage2_frontier.py": (
            b"reviewed frontier tests\n"
        ),
        "src/lunar_exploration_ppo/env/env.py": b"reviewed environment source\n",
    }
    source_bindings: dict[str, dict[str, object]] = {}
    for relative, payload in source_payloads.items():
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        source_bindings[relative] = {
            "path": relative,
            "sha256": _sha(payload),
            "size_bytes": len(payload),
        }
    monkeypatch.setattr(
        source_repair,
        "_SOURCE_REPAIR_REPO_ROOT",
        source_root,
        raising=False,
    )
    monkeypatch.setattr(
        source_repair,
        "_FRONTIER_RECOVERY_SOURCE_BINDINGS",
        source_bindings,
        raising=False,
    )

    evidence_root = stage.parent.parent / "frontier-recovery-review"
    evidence_root.mkdir(exist_ok=True)
    stderr_path = evidence_root / "recovery-r16.stderr.log"
    stderr_payload = (
        b"RuntimeError: production frontier is empty while observed "
        b"opportunities remain\n"
    )
    stderr_path.write_bytes(stderr_payload)
    replay_path = evidence_root / "replay-exception.json"
    replay_payload = _write_json(
        replay_path,
        {
            "schema_version": "stage6_update47_debug_replay/v1",
            "error_type": "WorkerProcessError",
            "message": stderr_payload.decode("utf-8"),
        },
    )
    root_cause_path = evidence_root / "root-cause-analysis.json"
    root_cause_payload = _write_json(
        root_cause_path,
        {
            "schema_version": "stage6_update47_frontier_root_cause/v1",
            "formal_run_id": RUN_ID,
            "failed_update": 47,
            "last_complete_update": 46,
            "formal_resume_authorized": False,
            "replay_result": "exact_failure_reproduced_and_observed_state_captured",
            "root_cause": "irregular nearest landing selection erased positive gain",
        },
    )
    patch_path = evidence_root / "review-package.diff"
    patch_payload = b"reviewed deterministic irregular frontier repair\n"
    patch_path.write_bytes(patch_payload)
    benchmark_path = evidence_root / "final-fix-benchmark.json"
    benchmark_payload = _write_json(
        benchmark_path,
        {
            "schema_version": "stage6_update47_final_fix_benchmark/v1",
            "frontier_source_sha256": source_bindings[
                "src/lunar_exploration_ppo/env/frontier.py"
            ]["sha256"],
            "public_candidate_count": 3,
            "public_fallback_activated": True,
            "positive_scored": 8,
        },
    )
    independent_path = evidence_root / "independent-review.json"
    independent_payload = _write_json(
        independent_path,
        {
            "schema_version": "stage6_update47_independent_review/v2",
            "review_mode": "read_only",
            "spec_compliance": "PASS",
            "code_quality": "PASS",
            "finding_counts": {"critical": 0, "important": 0, "minor": 0},
            "review_package_sha256": _sha(patch_payload),
            "review_package_size_bytes": len(patch_payload),
            "frontier_source_sha256": source_bindings[
                "src/lunar_exploration_ppo/env/frontier.py"
            ]["sha256"],
            "stage2_test_source_sha256": source_bindings[
                "tests/ppo_highres_frontier/test_stage2_frontier.py"
            ]["sha256"],
            "benchmark_sha256": _sha(benchmark_payload),
            "formal_recovery_authorization_may_be_prepared": True,
            "formal_runner_launch_authorized_by_this_record": False,
        },
    )
    manifest_path = evidence_root / "review-manifest.json"
    manifest_payload = _write_json(
        manifest_path,
        {
            "schema_version": "stage6_update47_fix_review_manifest/v2",
            "task_scope": "bounded_irregular_frontier_sampling_update47_repair",
            "files": {
                "review-package.diff": {
                    "sha256": _sha(patch_payload),
                    "size_bytes": len(patch_payload),
                },
                "final-fix-benchmark.json": {
                    "sha256": _sha(benchmark_payload),
                    "size_bytes": len(benchmark_payload),
                },
                "post-fix/frontier.py": {
                    "sha256": source_bindings[
                        "src/lunar_exploration_ppo/env/frontier.py"
                    ]["sha256"]
                },
                "post-fix/env.py": {
                    "sha256": source_bindings[
                        "src/lunar_exploration_ppo/env/env.py"
                    ]["sha256"]
                },
                "post-fix/test_stage2_frontier.py": {
                    "sha256": source_bindings[
                        "tests/ppo_highres_frontier/test_stage2_frontier.py"
                    ]["sha256"]
                },
                "root-cause-analysis.json": {
                    "sha256": _sha(root_cause_payload)
                },
            },
            "formal_replay_or_training_launched": False,
        },
    )
    evidence_payloads = {
        "stderr": (stderr_path, stderr_payload),
        "replay_exception": (replay_path, replay_payload),
        "root_cause": (root_cause_path, root_cause_payload),
        "reviewed_patch": (patch_path, patch_payload),
        "benchmark": (benchmark_path, benchmark_payload),
        "independent_review": (independent_path, independent_payload),
        "review_manifest": (manifest_path, manifest_payload),
    }
    evidence_bindings = {
        kind: {
            "kind": kind,
            "path": path.as_posix(),
            "sha256": _sha(payload),
            "size_bytes": len(payload),
        }
        for kind, (path, payload) in evidence_payloads.items()
    }
    monkeypatch.setattr(
        source_repair,
        "_FRONTIER_RECOVERY_EVIDENCE_BINDINGS",
        evidence_bindings,
        raising=False,
    )
    fixture["frontier_recovery_prefix"] = frontier_prefix
    fixture["frontier_recovery_source_root"] = source_root
    fixture["frontier_recovery_source_bindings"] = source_bindings
    fixture["frontier_recovery_evidence_paths"] = {
        kind: path for kind, (path, _payload) in evidence_payloads.items()
    }
    fixture["checkpoint46_lineage"] = checkpoint46_lineage
    fixture["checkpoint46_payload"] = checkpoint46_payload
    fixture["checkpoint46_manifest_payload"] = manifest46_payload
    fixture["checkpoint46_complete_payload"] = complete46_payload
    fixture["latest46_payload"] = latest46_payload


def _prepare_frontier_recovery_cutover(fixture: dict[str, object]):
    closure = _create_closure(fixture)
    _record_update47_frontier_recovery_cutover(fixture, closure)
    return closure


def _create_frontier_recovery(
    fixture: dict[str, object],
    *,
    publish: bool = True,
):
    _prepare_frontier_recovery_cutover(fixture)
    return source_repair.create_stage6_source_repair_amendment(
        stage_root=fixture["stage"],
        current_execution_identity=fixture["frontier_recovery_identity"],
        current_verified_review_authorization=fixture[
            "frontier_recovery_authorization"
        ],
        current_immutable_bindings=fixture["frontier_recovery_immutable"],
        publish=publish,
    )


def _coverage_cache_manifest_value(cache_root: Path) -> dict[str, object]:
    split_counts = {"test": 150, "train": 700, "unseen": 64, "validation": 150}
    entries: list[dict[str, object]] = []
    for split, count in split_counts.items():
        for index in range(count):
            scenario_id = f"{split}/scenario-{index:04d}/standard-proxy/v1"
            key_sha256 = _sha(f"key:{scenario_id}".encode())
            entries.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_hash": _sha(f"scenario:{scenario_id}".encode()),
                    "split": split,
                    "key_sha256": key_sha256,
                    "path": f"entries/{key_sha256[:2]}/{key_sha256}.npz",
                    "sha256": _sha(f"entry:{scenario_id}".encode()),
                    "size_bytes": 199364,
                }
            )
    entries.sort(key=lambda item: str(item["scenario_id"]))
    return {
        "schema_version": "stage6_exact_coverable_cache_manifest/v1",
        "cache_root": str(cache_root.resolve()),
        "catalog_sha256": "3" * 64,
        "split_counts": {
            "train": 700,
            "validation": 150,
            "test": 150,
            "unseen": 64,
        },
        "generation_environment": {
            "producer": "stage6_coverage_cache_prewarm/v1",
            "python_version": "3.12.13",
            "numpy_version": "2.2.6",
            "platform": "Windows-test",
        },
        "integrity": {
            "complete": True,
            "entry_count": 1064,
            "valid_entry_count": 1064,
            "missing_entry_count": 0,
            "duplicate_entry_count": 0,
            "corrupt_entry_count": 0,
        },
        "entry_set_sha256": _sha(_canonical(entries)),
        "entries": entries,
    }


def _record_update50_sensor_acceleration_cutover(
    fixture: dict[str, object],
    frontier_recovery,
) -> None:
    stage = fixture["stage"]
    monkeypatch = fixture["monkeypatch"]
    assert isinstance(stage, Path)
    assert isinstance(monkeypatch, pytest.MonkeyPatch)

    config = load_stage6_config(CONFIG)
    transactions = build_standard_training_transactions(config)
    receipt_index = CheckpointReceiptIndex(stage / "checkpoints/index.jsonl")
    journal = Stage6StateJournal(stage / "job-state.jsonl")
    checkpoint49_payload = b"test-update-49-checkpoint"
    policy49_sha256 = _sha(b"policy-49")
    checkpoint49_lineage = frontier_recovery.checkpoint_lineage_for_update(49)
    manifest49_payload = _canonical(
        {
            "schema_version": "stage6_checkpoint_manifest/v1",
            "update_step": 49,
            "checkpoint": {
                "path": "checkpoint.pt",
                "sha256": _sha(checkpoint49_payload),
                "size_bytes": len(checkpoint49_payload),
            },
            "policy_state_sha256": policy49_sha256,
            "config_sha256": fixture["frontier_recovery_identity"]["config_sha256"],
            "lineage_sha256": _sha(_canonical(checkpoint49_lineage)),
        }
    )
    complete49_payload = _canonical(
        {
            "schema_version": "stage6_checkpoint_complete/v1",
            "update_step": 49,
            "manifest_sha256": _sha(manifest49_payload),
            "checkpoint_sha256": _sha(checkpoint49_payload),
        }
    )
    checkpoint_sha256_by_update: dict[int, str] = {}
    for transaction in transactions[46:49]:
        is_cutover = transaction.update == 49
        receipt = StandardTransactionCheckpoint(
            transaction_key=transaction.key,
            seed=transaction.seed,
            update=transaction.update,
            checkpoint_sha256=(
                _sha(checkpoint49_payload)
                if is_cutover
                else _sha(f"checkpoint-{transaction.update}".encode())
            ),
            complete_marker_sha256=(
                _sha(complete49_payload)
                if is_cutover
                else _sha(f"complete-{transaction.update}".encode())
            ),
            policy_state_sha256=(
                policy49_sha256
                if is_cutover
                else _sha(f"policy-{transaction.update}".encode())
            ),
        )
        receipt_index.append_once(receipt)
        bindings = {
            **fixture["frontier_recovery_immutable"],
            "checkpoint_sha256": receipt.checkpoint_sha256,
        }
        for state in transaction.commit_states:
            journal.append(state, bindings)
        checkpoint_sha256_by_update[transaction.update] = receipt.checkpoint_sha256

    checkpoint49_directory = stage / f"checkpoints/seed-{SEED}/update-00000049"
    checkpoint49_directory.mkdir(parents=True)
    (checkpoint49_directory / "checkpoint.pt").write_bytes(checkpoint49_payload)
    (checkpoint49_directory / "manifest.json").write_bytes(manifest49_payload)
    (checkpoint49_directory / "complete.json").write_bytes(complete49_payload)
    latest49_payload = _write_json(
        checkpoint49_directory.parent / "latest.json",
        {
            "schema_version": "stage6_checkpoint_latest/v1",
            "update_step": 49,
            "directory": "update-00000049",
            "checkpoint_sha256": _sha(checkpoint49_payload),
            "manifest_sha256": _sha(manifest49_payload),
            "policy_state_sha256": policy49_sha256,
        },
    )
    fixture["checkpoint_payload_values"][checkpoint49_payload] = {
        "update_step": 49,
        "config_sha256": fixture["frontier_recovery_identity"]["config_sha256"],
        "lineage": checkpoint49_lineage,
        "policy_state_sha256": policy49_sha256,
    }

    training_payload = _write_jsonl(
        stage / "training_metrics.jsonl",
        [{"update": update} for update in range(1, 50)],
    )
    prior_resource_payload = (stage / "resource_audit.jsonl").read_bytes()
    resource_rows = [
        json.loads(line)
        for line in prior_resource_payload.decode("utf-8").splitlines()
    ]
    resource_rows.append(
        {
            "schema_version": "stage6_resource_lifecycle_segment/v1",
            "kind": "resource_lifecycle_segment",
            "phase": "segment_start",
            "segment_id": SENSOR_ACCELERATION_SEGMENT_ID,
            "segment_index": 6,
            "root_pid": SENSOR_ACCELERATION_ROOT_PID,
            "prior_resource_log": {
                "sha256": _sha(prior_resource_payload),
                "size_bytes": len(prior_resource_payload),
            },
            "first_sample": {
                "passed": True,
                "rss_root_pid": SENSOR_ACCELERATION_ROOT_PID,
            },
        }
    )
    for update, attempt in ((47, 2), (48, 1), (49, 1)):
        key = f"{update - 1:04d}:{SEED}:update:{update:03d}"
        pre = {
            "schema_version": "stage6_resource_attempt/v1",
            "kind": "update",
            "transaction_key": key,
            "seed": SEED,
            "update": update,
            "attempt": attempt,
            "phase": "pre",
            "accepted": False,
            "resource": {
                "passed": True,
                "rss_root_pid": SENSOR_ACCELERATION_ROOT_PID,
            },
            "segment_id": SENSOR_ACCELERATION_SEGMENT_ID,
            "segment_index": 6,
        }
        resource_rows.extend(
            (
                pre,
                {**pre, "phase": "post"},
                {
                    "schema_version": "stage6_resource_acceptance/v2",
                    "kind": "update",
                    "transaction_key": key,
                    "seed": SEED,
                    "update": update,
                    "attempt": attempt,
                    "phase": "accepted",
                    "accepted": True,
                    "pre": {
                        "passed": True,
                        "rss_root_pid": SENSOR_ACCELERATION_ROOT_PID,
                    },
                    "post": {
                        "passed": True,
                        "rss_root_pid": SENSOR_ACCELERATION_ROOT_PID,
                    },
                    "checkpoint": {
                        "checkpoint_sha256": checkpoint_sha256_by_update[update]
                    },
                    "segment_id": SENSOR_ACCELERATION_SEGMENT_ID,
                    "segment_index": 6,
                },
            )
        )
    resource_rows.append(
        {
            "schema_version": "stage6_resource_attempt/v1",
            "kind": "update",
            "transaction_key": "0049:20260716:update:050",
            "seed": SEED,
            "update": 50,
            "attempt": 1,
            "phase": "pre",
            "accepted": False,
            "resource": {
                "passed": True,
                "rss_root_pid": SENSOR_ACCELERATION_ROOT_PID,
            },
            "segment_id": SENSOR_ACCELERATION_SEGMENT_ID,
            "segment_index": 6,
        }
    )
    assert len(resource_rows) == 159
    resource_payload = _write_jsonl(stage / "resource_audit.jsonl", resource_rows)

    prefix = copy.deepcopy(fixture["frontier_recovery_prefix"])
    for relative, payload in {
        "checkpoints/index.jsonl": (stage / "checkpoints/index.jsonl").read_bytes(),
        "job-state.jsonl": (stage / "job-state.jsonl").read_bytes(),
        "resource_audit.jsonl": resource_payload,
        "training_metrics.jsonl": training_payload,
        f"checkpoints/seed-{SEED}/update-00000049/checkpoint.pt": (
            checkpoint49_payload
        ),
        f"checkpoints/seed-{SEED}/update-00000049/manifest.json": (
            manifest49_payload
        ),
        f"checkpoints/seed-{SEED}/update-00000049/complete.json": (
            complete49_payload
        ),
    }.items():
        prefix[relative] = {
            "path": relative,
            "sha256": _sha(payload),
            "size_bytes": len(payload),
        }
    monkeypatch.setattr(
        source_repair,
        "_SENSOR_ACCELERATION_PREFIX_BINDINGS",
        prefix,
        raising=False,
    )

    frontier_payload = (
        stage / source_repair.SOURCE_REPAIR_FRONTIER_RECOVERY_NAME
    ).read_bytes()
    monkeypatch.setattr(
        source_repair,
        "_FIXED_FRONTIER_RECOVERY_BINDING",
        {
            "path": source_repair.SOURCE_REPAIR_FRONTIER_RECOVERY_NAME,
            "sha256": _sha(frontier_payload),
            "size_bytes": len(frontier_payload),
        },
        raising=False,
    )
    monkeypatch.setattr(
        source_repair,
        "_FIXED_FRONTIER_RECOVERY_EXECUTION_IDENTITY_SHA256",
        _sha(_canonical(fixture["frontier_recovery_identity"])),
        raising=False,
    )

    source_root = fixture["frontier_recovery_source_root"]
    assert isinstance(source_root, Path)
    source_bindings: dict[str, dict[str, object]] = {}
    for relative in (
        "docs/superpowers/plans/2026-07-22-ppo-stage6-sensor-hotpath-acceleration.md",
        "docs/superpowers/plans/2026-07-22-ppo-stage6-coverable-mask-cache.md",
        "docs/superpowers/specs/2026-07-22-ppo-stage6-sensor-hotpath-acceleration-design-addendum.md",
        "docs/superpowers/specs/2026-07-22-ppo-stage6-coverable-mask-cache-design-addendum.md",
        "scripts/benchmark_ppo_stage6_sensor_hotpath.py",
        "scripts/prewarm_ppo_stage6_coverage_cache.py",
        "src/lunar_exploration_ppo/env/action_execution.py",
        "src/lunar_exploration_ppo/env/coverage_cache.py",
        "src/lunar_exploration_ppo/env/env.py",
        "src/lunar_exploration_ppo/env/sensor_model.py",
        "src/lunar_exploration_ppo/env/standard_training.py",
        "src/lunar_exploration_ppo/workflows/stage6_coverage_cache.py",
        "tests/ppo_highres_frontier/test_stage6_coverage_cache.py",
        "tests/ppo_highres_frontier/test_stage6_coverage_cache_workflow.py",
        "tests/ppo_highres_frontier/test_stage6_sensor_acceleration.py",
    ):
        path = source_root / relative
        if path.exists():
            payload = path.read_bytes()
        else:
            payload = f"reviewed source: {relative}\n".encode()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        source_bindings[relative] = {
            "path": relative,
            "sha256": _sha(payload),
            "size_bytes": len(payload),
        }
    monkeypatch.setattr(
        source_repair,
        "_SENSOR_ACCELERATION_SOURCE_BINDINGS",
        source_bindings,
        raising=False,
    )

    cache_root = stage.parent.parent / "coverage-v1"
    cache_root.mkdir()
    manifest_path = stage.parent.parent / "coverage-cache-manifest.json"
    manifest_value = _coverage_cache_manifest_value(cache_root)
    manifest_payload = _write_json(manifest_path, manifest_value)
    audit_path = stage.parent.parent / "formal-audit.json"
    audit_payload = _write_json(
        audit_path,
        {
            "schema_version": "stage6_coverable_cache_formal_audit/v1",
            "status": "pass",
            "formal_run_id": RUN_ID,
            "boundaries": {
                "formal_training_checkpoint_update": 49,
                "formal_training_started": False,
                "checkpoint_published": False,
                "update50_attempt1_discarded": True,
            },
            "prewarm": {
                "completed_count": 1064,
                "entry_set_sha256": manifest_value["entry_set_sha256"],
                "manifest_sha256": _sha(manifest_payload),
                "manifest_size_bytes": len(manifest_payload),
                "split_counts": manifest_value["split_counts"],
            },
            "strict_load": {"loaded_count": 1064, "canonical_manifest": True},
            "performance": {
                "passed": True,
                "cached_median_seconds": 0.8,
                "measured_speedup": 41.0,
            },
            "representative_equivalence": {
                "mask_bitwise_equal": True,
                "reset_observation_equal": True,
                "candidate_equal": True,
                "deterministic_action_equal": True,
                "first_step_equal": True,
            },
            "leakage": {
                "passed": True,
                "policy_frontier_cache_parameter_absent": True,
            },
            "focused_regressions": {"failed": 0, "passed": 86, "skipped": 1},
        },
    )
    evidence_root = stage.parent.parent / "sensor-cache-review"
    evidence_root.mkdir()
    evidence_payloads: dict[str, tuple[Path, bytes]] = {
        "coverage_cache_manifest": (manifest_path, manifest_payload),
        "coverage_cache_formal_audit": (audit_path, audit_payload),
    }
    for kind in (
        "sensor_implementation_report",
        "sensor_fresh_review",
        "cache_b1_fresh_review",
        "cache_b2_fresh_review",
        "cache_formal_spec_review",
        "cache_formal_quality_review",
        "cache_path_safety_verification",
        "cache_path_safety_rereview",
    ):
        path = evidence_root / f"{kind}.md"
        payload = f"{kind}: PASS C0 I0 M0\n".encode()
        path.write_bytes(payload)
        evidence_payloads[kind] = (path, payload)
    evidence_bindings = {
        kind: {
            "kind": kind,
            "path": path.as_posix(),
            "sha256": _sha(payload),
            "size_bytes": len(payload),
        }
        for kind, (path, payload) in evidence_payloads.items()
    }
    monkeypatch.setattr(
        source_repair,
        "_SENSOR_ACCELERATION_EVIDENCE_BINDINGS",
        evidence_bindings,
        raising=False,
    )
    manifest_binding = {
        "path": manifest_path.as_posix(),
        "sha256": _sha(manifest_payload),
        "size_bytes": len(manifest_payload),
        "cache_root": str(cache_root.resolve()).replace("\\", "/"),
        "entry_set_sha256": manifest_value["entry_set_sha256"],
        "entry_count": 1064,
        "split_counts": manifest_value["split_counts"],
    }
    monkeypatch.setattr(
        source_repair,
        "_FIXED_COVERAGE_CACHE_MANIFEST_BINDING",
        manifest_binding,
        raising=False,
    )
    sensor_immutable = fixture["sensor_acceleration_immutable"]
    assert isinstance(sensor_immutable, dict)
    sensor_immutable.update(
        {
            "coverage_cache_manifest_path": manifest_binding["path"],
            "coverage_cache_manifest_sha256": manifest_binding["sha256"],
            "coverage_cache_manifest_size_bytes": manifest_binding["size_bytes"],
            "coverage_cache_root": manifest_binding["cache_root"],
            "coverage_cache_entry_set_sha256": manifest_binding["entry_set_sha256"],
            "coverage_cache_runtime_mode": "persistent_exact_manifest_read_only/v1",
            "coverage_cache_formal_audit_sha256": _sha(audit_payload),
            "coverage_cache_formal_audit_size_bytes": len(audit_payload),
        }
    )
    fixture["sensor_acceleration_prefix"] = prefix
    fixture["sensor_acceleration_source_root"] = source_root
    fixture["sensor_acceleration_source_bindings"] = source_bindings
    fixture["sensor_acceleration_evidence_paths"] = {
        kind: path for kind, (path, _payload) in evidence_payloads.items()
    }
    fixture["coverage_cache_manifest_path"] = manifest_path
    fixture["coverage_cache_manifest_value"] = manifest_value
    fixture["coverage_cache_manifest_payload"] = manifest_payload
    fixture["coverage_cache_formal_audit_path"] = audit_path
    fixture["checkpoint49_lineage"] = checkpoint49_lineage
    fixture["checkpoint49_payload"] = checkpoint49_payload
    fixture["checkpoint49_manifest_payload"] = manifest49_payload
    fixture["checkpoint49_complete_payload"] = complete49_payload
    fixture["latest49_payload"] = latest49_payload


def _prepare_sensor_acceleration_cutover(fixture: dict[str, object]):
    frontier_recovery = _create_frontier_recovery(fixture)
    _record_update50_sensor_acceleration_cutover(fixture, frontier_recovery)
    return frontier_recovery


def _rebind_sensor_acceleration_root_pid(
    fixture: dict[str, object],
    root_pid: int,
) -> None:
    stage = fixture["stage"]
    prefix = fixture["sensor_acceleration_prefix"]
    assert isinstance(stage, Path)
    assert isinstance(prefix, dict)
    resource_path = stage / "resource_audit.jsonl"
    rows = [
        json.loads(line)
        for line in resource_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        if row.get("segment_index") != 6:
            continue
        phase = row.get("phase")
        if phase == "segment_start":
            row["root_pid"] = root_pid
            row["first_sample"]["rss_root_pid"] = root_pid
        elif phase == "accepted":
            row["pre"]["rss_root_pid"] = root_pid
            row["post"]["rss_root_pid"] = root_pid
        else:
            row["resource"]["rss_root_pid"] = root_pid
    payload = _write_jsonl(resource_path, rows)
    prefix["resource_audit.jsonl"] = {
        "path": "resource_audit.jsonl",
        "sha256": _sha(payload),
        "size_bytes": len(payload),
    }


def test_fixed_fifth_cutover_bindings_match_update46_and_reviewed_update47() -> None:
    assert getattr(source_repair, "SOURCE_REPAIR_FRONTIER_RECOVERY_NAME", None) == (
        "source-repair-frontier-recovery.json"
    )
    assert getattr(source_repair, "SOURCE_REPAIR_FRONTIER_RECOVERY_SCHEMA", None) == (
        "stage6_source_repair_frontier_recovery/v1"
    )
    assert getattr(source_repair, "SOURCE_REPAIR_FRONTIER_RECOVERY_ID", None) == (
        "stage6a_irregular_frontier_empty_update47/v1"
    )
    assert getattr(source_repair, "LAST_FRONTIER_RECOVERY_PARENT_UPDATE", None) == 46
    assert getattr(source_repair, "FIRST_FRONTIER_RECOVERY_UPDATE", None) == 47
    assert getattr(source_repair, "FRONTIER_RECOVERY_NEXT_TRANSACTION_KEY", None) == (
        "0046:20260716:update:047"
    )
    assert getattr(source_repair, "_FIXED_CLOSURE_BINDING", None) == {
        "path": "source-repair-closure.json",
        "sha256": "03461ac4e882979c7cc0a6a7f442cfbc4f93d97f0e0cb3891ce59050da47abb7",
        "size_bytes": 28011,
    }

    expected_prefix = {
        "config.json": (
            "7b1c37673c105c34e7bd71d04d4f508d3a167cdf77426b0b35c7aff86ba8adae",
            5298,
        ),
        "lineage_audit.json": (
            "95573b284888e363a3dfe87db012f0de47fe890a1557cd3736989b5b689e66b2",
            19634,
        ),
        "checkpoints/index.jsonl": (
            "1e9facc5c364de8d8857116cdeaa35d95e9a0b16c3ec966a1b845f7e566ccdb8",
            23635,
        ),
        "job-state.jsonl": (
            "347456c9b6826c071fb8779d364f97d6015f2867615890c856e9a53a0a813820",
            110765,
        ),
        "phase-state.jsonl": (
            "b3f54b83468c9a5aa88ceb37b5bf67bea4c47528a29160bd3708a4ab8012ad5b",
            2149,
        ),
        "resource_audit.jsonl": (
            "e8520c6bb6d7dbf20dfeebbd0fcbbcd14e45241656dc02d55cb939e45af6ae1c",
            111162,
        ),
        "training_metrics.jsonl": (
            "3f42d64567ff0aa86c69f75466e9c673f25feafc5bb7a59114366cabd76bd1c5",
            3489401,
        ),
        "validation_metrics.jsonl": (
            "e79bc23e771aea6e0f488bf129f37817a63b2f1fd851a33ee13b194bb813063b",
            6350,
        ),
        "math_audit.jsonl": (
            "aa8267d7039245fe4b7eff7d48235b66bc263a1735836e1fb345a7bf49d7b4f6",
            142772,
        ),
        "checkpoint_audit.jsonl": (
            "6c0fb6a132d86cb146bcf47c8fdcfbbd58bad6aea4a27e2b5904e447fe38da56",
            1705,
        ),
        f"checkpoints/seed-{SEED}/update-00000046/checkpoint.pt": (
            "4c55325c3342d477354bd7e5a909bfd157a9fa6daee39b4f96fe926a7f647a96",
            30808555,
        ),
        f"checkpoints/seed-{SEED}/update-00000046/manifest.json": (
            "ed607a1168f5377c87e90a62c6651c09c0e9a580f0e0fbc7e91bb3add98434ff",
            965,
        ),
        f"checkpoints/seed-{SEED}/update-00000046/complete.json": (
            "3fec78dc799b770d57beda63c2b6139c5277ba1bf0426fad60c9a4b4995978b0",
            257,
        ),
    }
    prefix = getattr(source_repair, "_FRONTIER_RECOVERY_PREFIX_BINDINGS", {})
    for path, (sha256, size_bytes) in expected_prefix.items():
        assert prefix.get(path) == {
            "path": path,
            "sha256": sha256,
            "size_bytes": size_bytes,
        }

    assert getattr(source_repair, "_FRONTIER_RECOVERY_EVIDENCE_BINDINGS", None) == {
        "stderr": {
            "kind": "stderr",
            "path": (
                "D:/xunce/out/ppo_frontier-logs/"
                "s6-standard-single-r1-20260718T220434Z.recovery-r16.stderr.log"
            ),
            "sha256": "06b78828149ba47cc8f1fd5f8ee5f5e753e3fc7d30e9d78b9c8ed14ba736689c",
            "size_bytes": 5221,
        },
        "replay_exception": {
            "kind": "replay_exception",
            "path": (
                "D:/xunce/review/s6-update47-debug-r1-20260721T1635-CST/"
                "replay-exception.json"
            ),
            "sha256": "aeb302256ef29b7ce01ae66adf4101c1c69c9de63522b01ea0389ee79dca62ba",
            "size_bytes": 1486,
        },
        "root_cause": {
            "kind": "root_cause",
            "path": (
                "D:/xunce/review/s6-update47-debug-r1-20260721T1635-CST/"
                "root-cause-analysis.json"
            ),
            "sha256": "7cfbb1f7c906677a1a0bbff96a36288d9a4e4487a78d5845b1a13863db426f48",
            "size_bytes": 2412,
        },
        "reviewed_patch": {
            "kind": "reviewed_patch",
            "path": "D:/xunce/review/s6-update47-fix-r2/review-package.diff",
            "sha256": "131120db74b0e4a6977dd545bbeb3f717ffa10fba4d565160e5d256e26e429e1",
            "size_bytes": 9994,
        },
        "benchmark": {
            "kind": "benchmark",
            "path": "D:/xunce/review/s6-update47-fix-r2/final-fix-benchmark.json",
            "sha256": "325a3e9e71ca147fa88d4b2ba078899d30106a34782ba1271471b01c018beed6",
            "size_bytes": 799,
        },
        "independent_review": {
            "kind": "independent_review",
            "path": "D:/xunce/review/s6-update47-fix-r2/independent-review.json",
            "sha256": "57cebaf1572d33bab51bc7d06cf74435bfb5f13c5d72fc4526f8e6e1559d04e3",
            "size_bytes": 1139,
        },
        "review_manifest": {
            "kind": "review_manifest",
            "path": "D:/xunce/review/s6-update47-fix-r2/review-manifest.json",
            "sha256": "9515183d748a2a993d679687f3dba9d97e8ebf67c0718730a5034b37f43f5add",
            "size_bytes": 1464,
        },
    }
    assert getattr(source_repair, "_FRONTIER_RECOVERY_SOURCE_BINDINGS", None) == {
        "src/lunar_exploration_ppo/env/frontier.py": {
            "path": "src/lunar_exploration_ppo/env/frontier.py",
            "sha256": "7d94f48440d5421002c2d05aebf13a73f4062452911f0f8442315ace4069a62c",
        },
        "tests/ppo_highres_frontier/test_stage2_frontier.py": {
            "path": "tests/ppo_highres_frontier/test_stage2_frontier.py",
            "sha256": "1062d477bd62b5bb68f60f5781eb08f215134901dafa4f66c9ca1cf27ae7caa8",
        },
        "src/lunar_exploration_ppo/env/env.py": {
            "path": "src/lunar_exploration_ppo/env/env.py",
            "sha256": "feee2a000faa1c3beb3f308db8cdf28a77978c556bba46302ed977430f0f8b07",
        },
    }


def test_fixed_update9_checkpoint_lineage_binding_matches_formal_manifest() -> None:
    assert source_repair._FIXED_ORIGIN_CHECKPOINT_LINEAGE_SHA256 == (
        "9d7362c0c681acd5f36d9819aa507e1de26d8c507959fce6c823d60b781cfce2"
    )


def test_fixed_second_cutover_and_device_failure_bindings_match_production() -> None:
    assert source_repair._FIXED_PRIMARY_AMENDMENT_BINDING == {
        "path": "source-repair-amendment.json",
        "sha256": "803fd655881070b7ff424cebf17a6346300f510f7c5e86082e9fda97a194a54f",
        "size_bytes": 26058,
    }


def test_fixed_third_cutover_and_initial_ratio_failure_bindings_match_production() -> None:
    assert source_repair.SOURCE_REPAIR_SUPPLEMENT_NAME == (
        "source-repair-supplement.json"
    )
    assert source_repair.SOURCE_REPAIR_SUPPLEMENT_SCHEMA == (
        "stage6_source_repair_supplement/v1"
    )
    assert source_repair.SOURCE_REPAIR_SUPPLEMENT_ID == (
        "stage6a_initial_ratio_gate_update33/v1"
    )
    assert source_repair._FIXED_CONTINUATION_BINDING == {
        "path": "source-repair-continuation.json",
        "sha256": "53f9a81690050c68abe16d8b4583ae0587da313440fefe48f73151fe57de9fec",
        "size_bytes": 26190,
    }
    assert source_repair._FIXED_CONTINUATION_EXECUTION_IDENTITY_SHA256 == (
        "9af554d978c339a3d0ad176646dc74ca707748adffef94bc68645beddf6c17b0"
    )
    expected_prefix = {
        "config.json": (
            "7b1c37673c105c34e7bd71d04d4f508d3a167cdf77426b0b35c7aff86ba8adae",
            5298,
        ),
        "lineage_audit.json": (
            "95573b284888e363a3dfe87db012f0de47fe890a1557cd3736989b5b689e66b2",
            19634,
        ),
        "checkpoints/index.jsonl": (
            "b9f9010cfc8474f072dbce1b8ed58651548d536caeb31057290660874d7a9387",
            16439,
        ),
        "job-state.jsonl": (
            "e94e6f5c69417a2a9610eca926a35498f696665106635365cac7b10aa4582494",
            78183,
        ),
        "resource_audit.jsonl": (
            "355fc194cccd8dadce5ac7bbe655eb42ce51da6758f1d32c68f8d876df0acf2f",
            76753,
        ),
        "training_metrics.jsonl": (
            "fcdd582da9934305894879619730c2847eec7ed9fc2d1a60ce85f21c0e82c47b",
            2422515,
        ),
        "validation_metrics.jsonl": (
            "b4fa8c572b5f8db829ee5ae1c73b3f3846fcdd85d4aab7ef364843044fdad7ea",
            4763,
        ),
        "math_audit.jsonl": (
            "aa8267d7039245fe4b7eff7d48235b66bc263a1735836e1fb345a7bf49d7b4f6",
            142772,
        ),
        "checkpoint_audit.jsonl": (
            "6c0fb6a132d86cb146bcf47c8fdcfbbd58bad6aea4a27e2b5904e447fe38da56",
            1705,
        ),
        "phase-state.jsonl": (
            "b3f54b83468c9a5aa88ceb37b5bf67bea4c47528a29160bd3708a4ab8012ad5b",
            2149,
        ),
        f"checkpoints/seed-{SEED}/update-00000032/checkpoint.pt": (
            "c79f1bf9af636c9a14174a367c7ec5fe22996093bb5ee67d7d82fc5f4a4da4d1",
            30808683,
        ),
        f"checkpoints/seed-{SEED}/update-00000032/manifest.json": (
            "62f4ecb7e198ee07196a5713939a1d6eb49c4363921636995756624e63c36f86",
            965,
        ),
        f"checkpoints/seed-{SEED}/update-00000032/complete.json": (
            "56ceeacb31f846b32165da36ec8497b662002c2064498fbff6b96ef3be9dfa0d",
            257,
        ),
    }
    for path, (sha256, size_bytes) in expected_prefix.items():
        assert source_repair._SUPPLEMENT_PREFIX_BINDINGS[path] == {
            "path": path,
            "sha256": sha256,
            "size_bytes": size_bytes,
        }
    assert source_repair._SUPPLEMENT_EVIDENCE_BINDINGS == {
        "stderr": {
            "kind": "stderr",
            "path": (
                "D:/xunce/out/ppo_frontier-logs/"
                "s6-standard-single-r1-20260718T220434Z.recovery-r14.stderr.log"
            ),
            "sha256": "cde50309f20e85be3b7154845d2a7e41189e3f1c78b8c647b9bf61c58d7c62e0",
            "size_bytes": 3266,
        }
    }
    assert source_repair._CONTINUATION_PREFIX_BINDINGS[
        "resource_audit.jsonl"
    ] == {
        "path": "resource_audit.jsonl",
        "sha256": "698d6e7ee5f834b41e968f66d174cbfc3c6ddda04ccfb3d84bf210f85e64cfbd",
        "size_bytes": 22886,
    }
    assert source_repair._CONTINUATION_EVIDENCE_BINDINGS == {
        "device_failure": {
            "kind": "device_failure",
            "path": "D:/xunce/review/s6-update10-device-debug-r1/evidence.json",
            "sha256": "c354a0afeec129825a8c2f00f1003395b63cf35492ad014e1eabd78c4a14ec3d",
            "size_bytes": 1436,
        },
        "stderr": {
            "kind": "stderr",
            "path": "D:/xunce/out/ppo_frontier-logs/s6-standard-single-r1-20260718T220434Z.recovery-r13.stderr.log",
            "sha256": "ee8024c0d0b953cb558ab62bcd3c1eb949f6d54bab5d4e9e1b448ce999d9846f",
            "size_bytes": 4345,
        },
    }


def test_fixed_fourth_cutover_and_proxy_safety_evidence_match_production() -> None:
    assert source_repair.SOURCE_REPAIR_CLOSURE_NAME == "source-repair-closure.json"
    assert source_repair.SOURCE_REPAIR_CLOSURE_SCHEMA == (
        "stage6_source_repair_closure/v1"
    )
    assert source_repair.SOURCE_REPAIR_CLOSURE_ID == (
        "stage6a_proxy_safety_acceptance_contract_update33/v1"
    )
    assert source_repair._FIXED_SUPPLEMENT_BINDING == {
        "path": "source-repair-supplement.json",
        "sha256": "b8cd7c0e91e8d0f595d5b54989ef050ed263748716d28182cf3fa9fc3dc62cc4",
        "size_bytes": 27321,
    }
    assert source_repair._FIXED_SUPPLEMENT_EXECUTION_IDENTITY_SHA256 == (
        "ba5611952e9d6f7c45b3ba7fba8d8eb3802476157398af4771a90a1e6c47b65a"
    )
    expected_prefix = {
        "config.json": (
            "7b1c37673c105c34e7bd71d04d4f508d3a167cdf77426b0b35c7aff86ba8adae",
            5298,
        ),
        "lineage_audit.json": (
            "95573b284888e363a3dfe87db012f0de47fe890a1557cd3736989b5b689e66b2",
            19634,
        ),
        "checkpoints/index.jsonl": (
            "b9f9010cfc8474f072dbce1b8ed58651548d536caeb31057290660874d7a9387",
            16439,
        ),
        "job-state.jsonl": (
            "e94e6f5c69417a2a9610eca926a35498f696665106635365cac7b10aa4582494",
            78183,
        ),
        "phase-state.jsonl": (
            "b3f54b83468c9a5aa88ceb37b5bf67bea4c47528a29160bd3708a4ab8012ad5b",
            2149,
        ),
        "training_metrics.jsonl": (
            "fcdd582da9934305894879619730c2847eec7ed9fc2d1a60ce85f21c0e82c47b",
            2422515,
        ),
        "validation_metrics.jsonl": (
            "b4fa8c572b5f8db829ee5ae1c73b3f3846fcdd85d4aab7ef364843044fdad7ea",
            4763,
        ),
        "math_audit.jsonl": (
            "aa8267d7039245fe4b7eff7d48235b66bc263a1735836e1fb345a7bf49d7b4f6",
            142772,
        ),
        "checkpoint_audit.jsonl": (
            "6c0fb6a132d86cb146bcf47c8fdcfbbd58bad6aea4a27e2b5904e447fe38da56",
            1705,
        ),
        "resource_audit.jsonl": (
            "26034d2d2d14e930d1e73109a3206b1d8d4113ac2fe622d1b9fa187f13bcfa7f",
            77901,
        ),
    }
    for path, (sha256, size_bytes) in expected_prefix.items():
        assert source_repair._CLOSURE_PREFIX_BINDINGS[path] == {
            "path": path,
            "sha256": sha256,
            "size_bytes": size_bytes,
        }
    assert source_repair._CLOSURE_EVIDENCE_BINDINGS == {
        "finding_review": {
            "kind": "finding_review",
            "path": (
                "D:/xunce/review/s6-proxy-safety-r1-20260720T151730Z/"
                "finding-review.json"
            ),
            "sha256": "3ec619c06b01012f541b7facb90af79dcacc4c0c3758c77e390a4e94d6da943a",
            "size_bytes": 2519,
        },
        "safe_stop": {
            "kind": "safe_stop",
            "path": (
                "D:/xunce/review/s6-proxy-safety-r1-20260720T151730Z/"
                "safe-stop-audit.json"
            ),
            "sha256": "46d78e610d61411da3c703321aa612e9ed68c3700f6c1f7254d53c4bf8fb5032",
            "size_bytes": 1613,
        },
    }


def test_normal_run_without_amendment_is_unchanged(tmp_path: Path) -> None:
    stage = tmp_path / "normal" / "s6"
    stage.mkdir(parents=True)
    assert (
        source_repair.load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity={},
            current_verified_review_authorization={},
            current_immutable_bindings={},
        )
        is None
    )


def test_fixed_update9_to_10_amendment_is_exclusive_exact_and_selects_lineage(
    repair_fixture: dict[str, object],
) -> None:
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    protected = {
        relative: (stage / relative).read_bytes()
        for relative in (
            "lineage_audit.json",
            "checkpoints/index.jsonl",
            "job-state.jsonl",
            "phase-state.jsonl",
            "resource_audit.jsonl",
        )
    }
    first = _create(repair_fixture)
    second = _create(repair_fixture)
    assert first.amendment_sha256 == second.amendment_sha256
    assert first.origin_immutable_bindings == repair_fixture["old_immutable"]
    assert first.checkpoint_lineage_for_update(9) == repair_fixture["old_lineage"]
    repaired = first.checkpoint_lineage_for_update(10)
    assert repaired["source_set_sha256"] == repair_fixture["current_identity"][
        "source_set_sha256"
    ]
    assert repaired["source_repair_amendment_sha256"] == first.amendment_sha256
    assert repaired["source_repair_last_origin_update"] == 9
    assert repaired["source_repair_first_repaired_update"] == 10
    assert all((stage / relative).read_bytes() == payload for relative, payload in protected.items())


def test_newer_reviewed_identity_requires_continuation_artifact(
    repair_fixture: dict[str, object],
) -> None:
    _create(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)

    with pytest.raises(source_repair.Stage6SourceRepairError, match="continuation"):
        source_repair.load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=repair_fixture["latest_identity"],
            current_verified_review_authorization=repair_fixture[
                "latest_authorization"
            ],
            current_immutable_bindings=repair_fixture["latest_immutable"],
        )

    assert not (stage / source_repair.SOURCE_REPAIR_CONTINUATION_NAME).exists()


def test_loader_rejects_self_consistent_nonfixed_primary_replacement(
    repair_fixture: dict[str, object],
) -> None:
    primary = _create(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    replacement = _canonical(
        source_repair._amendment_value(
            stage=stage,
            current_execution_identity=repair_fixture["latest_identity"],
            current_verified_review_authorization=repair_fixture[
                "latest_authorization"
            ],
            current_immutable_bindings=repair_fixture["latest_immutable"],
            allow_growth=False,
        )
    )
    assert _sha(replacement) != primary.amendment_sha256
    (stage / source_repair.SOURCE_REPAIR_AMENDMENT_NAME).write_bytes(replacement)

    with pytest.raises(
        source_repair.Stage6SourceRepairError,
        match="primary source-repair amendment drifted",
    ):
        source_repair.load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=repair_fixture["latest_identity"],
            current_verified_review_authorization=repair_fixture[
                "latest_authorization"
            ],
            current_immutable_bindings=repair_fixture["latest_immutable"],
        )


def test_continuation_is_exclusive_idempotent_and_selects_latest_lineage(
    repair_fixture: dict[str, object],
) -> None:
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    context = _create_continuation(repair_fixture)
    replay = source_repair.create_stage6_source_repair_amendment(
        stage_root=stage,
        current_execution_identity=repair_fixture["latest_identity"],
        current_verified_review_authorization=repair_fixture[
            "latest_authorization"
        ],
        current_immutable_bindings=repair_fixture["latest_immutable"],
    )

    assert context.amendment_sha256 == replay.amendment_sha256
    assert context.continuation_sha256 == replay.continuation_sha256
    assert context.continuation_sha256 == _sha(
        (stage / source_repair.SOURCE_REPAIR_CONTINUATION_NAME).read_bytes()
    )
    assert context.origin_execution_identity == repair_fixture["old_identity"]
    assert context.current_execution_identity == repair_fixture["latest_identity"]
    assert context.protected_checkpoint_updates == (9,)
    assert context.checkpoint_lineage_for_update(9) == repair_fixture["old_lineage"]
    for update in (10, 50, 100):
        lineage = context.checkpoint_lineage_for_update(update)
        assert lineage["source_set_sha256"] == repair_fixture["latest_identity"][
            "source_set_sha256"
        ]
        assert lineage["source_repair_amendment_sha256"] == (
            context.amendment_sha256
        )
        assert lineage["source_repair_amendment_ordinal"] == 1
        assert lineage["source_repair_continuation_sha256"] == (
            context.continuation_sha256
        )
        assert lineage["source_repair_continuation_ordinal"] == 2
    assert context.summary_binding["schema_version"] == (
        "stage6_source_repair_summary/v2"
    )
    assert context.summary_binding["continuation_sha256"] == (
        context.continuation_sha256
    )


def test_supplement_is_exclusive_idempotent_and_preserves_three_lineage_segments(
    repair_fixture: dict[str, object],
) -> None:
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    context = _create_supplement(repair_fixture)
    primary_path = stage / source_repair.SOURCE_REPAIR_AMENDMENT_NAME
    continuation_path = stage / source_repair.SOURCE_REPAIR_CONTINUATION_NAME
    supplement_path = stage / source_repair.SOURCE_REPAIR_SUPPLEMENT_NAME
    protected_parent_bytes = {
        primary_path: primary_path.read_bytes(),
        continuation_path: continuation_path.read_bytes(),
    }
    replay = source_repair.create_stage6_source_repair_amendment(
        stage_root=stage,
        current_execution_identity=repair_fixture["supplement_identity"],
        current_verified_review_authorization=repair_fixture[
            "supplement_authorization"
        ],
        current_immutable_bindings=repair_fixture["supplement_immutable"],
    )

    assert context.supplement_sha256 == replay.supplement_sha256
    assert context.supplement_size_bytes == replay.supplement_size_bytes
    assert context.supplement_sha256 == _sha(supplement_path.read_bytes())
    assert all(path.read_bytes() == payload for path, payload in protected_parent_bytes.items())
    assert context.protected_checkpoint_updates == (9, 32)
    assert context.checkpoint_lineage_for_update(9) == repair_fixture["old_lineage"]

    for update in (10, 32):
        lineage = context.checkpoint_lineage_for_update(update)
        assert lineage == repair_fixture["checkpoint32_lineage"]
        assert lineage["source_set_sha256"] == repair_fixture["latest_identity"][
            "source_set_sha256"
        ]
        assert "source_repair_supplement_sha256" not in lineage

    for update in (33, 50, 100):
        lineage = context.checkpoint_lineage_for_update(update)
        assert lineage["source_set_sha256"] == repair_fixture[
            "supplement_identity"
        ]["source_set_sha256"]
        assert lineage["source_repair_amendment_sha256"] == (
            context.amendment_sha256
        )
        assert lineage["source_repair_amendment_ordinal"] == 1
        assert lineage["source_repair_continuation_sha256"] == (
            context.continuation_sha256
        )
        assert lineage["source_repair_continuation_ordinal"] == 2
        assert lineage["source_repair_supplement_sha256"] == (
            context.supplement_sha256
        )
        assert lineage["source_repair_supplement_ordinal"] == 3
        assert lineage["source_repair_supplement_first_repaired_update"] == 33

    binding = context.summary_binding
    assert binding["schema_version"] == "stage6_source_repair_summary/v3"
    assert binding["repair_ordinal"] == 3
    assert binding["supplement_sha256"] == context.supplement_sha256
    assert binding["ordinal_artifacts"] == [
        {
            "repair_ordinal": 1,
            "path": source_repair.SOURCE_REPAIR_AMENDMENT_NAME,
            "sha256": context.amendment_sha256,
        },
        {
            "repair_ordinal": 2,
            "path": source_repair.SOURCE_REPAIR_CONTINUATION_NAME,
            "sha256": context.continuation_sha256,
        },
        {
            "repair_ordinal": 3,
            "path": source_repair.SOURCE_REPAIR_SUPPLEMENT_NAME,
            "sha256": context.supplement_sha256,
        },
    ]


def test_ordinal2_loader_survives_supplement_and_future_identity_fails_closed(
    repair_fixture: dict[str, object],
) -> None:
    continuation = _prepare_supplement_cutover(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)

    original = source_repair.load_stage6_source_repair_context(
        stage_root=stage,
        current_execution_identity=repair_fixture["latest_identity"],
        current_verified_review_authorization=repair_fixture["latest_authorization"],
        current_immutable_bindings=repair_fixture["latest_immutable"],
    )
    assert original is not None
    assert original.current_execution_identity == repair_fixture["latest_identity"]
    assert original.continuation_sha256 == continuation.continuation_sha256
    with pytest.raises(source_repair.Stage6SourceRepairError, match="supplement"):
        source_repair.load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=repair_fixture["supplement_identity"],
            current_verified_review_authorization=repair_fixture[
                "supplement_authorization"
            ],
            current_immutable_bindings=repair_fixture["supplement_immutable"],
        )

    current = source_repair.create_stage6_source_repair_amendment(
        stage_root=stage,
        current_execution_identity=repair_fixture["supplement_identity"],
        current_verified_review_authorization=repair_fixture[
            "supplement_authorization"
        ],
        current_immutable_bindings=repair_fixture["supplement_immutable"],
        publish=True,
    )
    reloaded_original = source_repair.load_stage6_source_repair_context(
        stage_root=stage,
        current_execution_identity=repair_fixture["latest_identity"],
        current_verified_review_authorization=repair_fixture["latest_authorization"],
        current_immutable_bindings=repair_fixture["latest_immutable"],
    )
    assert reloaded_original is not None
    assert reloaded_original.current_execution_identity == repair_fixture[
        "latest_identity"
    ]
    assert reloaded_original.supplement_sha256 is None
    assert current.current_execution_identity == repair_fixture["supplement_identity"]

    environment = _environment_identity()
    future_identity = _identity(
        tree="5" * 40,
        source_sha256="f" * 64,
        config_sha256=repair_fixture["supplement_identity"]["config_sha256"],
        data_sha256="b" * 64,
        environment=environment,
    )
    future_authorization = _authorization(future_identity, marker="e")
    future_immutable = _immutable(
        future_identity,
        future_authorization,
        environment=environment,
    )
    with pytest.raises(source_repair.Stage6SourceRepairError, match="future|fourth|identity"):
        source_repair.load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=future_identity,
            current_verified_review_authorization=future_authorization,
            current_immutable_bindings=future_immutable,
        )


@pytest.mark.parametrize(
    "mutation",
    ("supplement", "primary", "continuation", "prefix", "stderr", "current"),
)
def test_supplement_tamper_and_drift_fail_closed(
    repair_fixture: dict[str, object],
    mutation: str,
) -> None:
    _create_supplement(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    current_identity = copy.deepcopy(repair_fixture["supplement_identity"])
    if mutation == "supplement":
        path = stage / source_repair.SOURCE_REPAIR_SUPPLEMENT_NAME
        value = json.loads(path.read_text(encoding="utf-8"))
        value["cutover"]["next_resource_attempt"] = 3
        path.write_bytes(_canonical(value))
    elif mutation == "primary":
        path = stage / source_repair.SOURCE_REPAIR_AMENDMENT_NAME
        path.write_bytes(path.read_bytes() + b"tamper")
    elif mutation == "continuation":
        path = stage / source_repair.SOURCE_REPAIR_CONTINUATION_NAME
        path.write_bytes(path.read_bytes() + b"tamper")
    elif mutation == "prefix":
        path = stage / "training_metrics.jsonl"
        path.write_bytes(b"{}\n" + path.read_bytes())
    elif mutation == "stderr":
        path = repair_fixture["supplement_stderr_path"]
        path.write_bytes(path.read_bytes() + b"tamper")
    else:
        current_identity["source_set_sha256"] = "0" * 64

    with pytest.raises(source_repair.Stage6SourceRepairError):
        source_repair.load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=current_identity,
            current_verified_review_authorization=repair_fixture[
                "supplement_authorization"
            ],
            current_immutable_bindings=repair_fixture["supplement_immutable"],
        )


def test_different_supplement_is_rejected_without_overwrite(
    repair_fixture: dict[str, object],
) -> None:
    _prepare_supplement_cutover(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    path = stage / source_repair.SOURCE_REPAIR_SUPPLEMENT_NAME
    foreign = _canonical({"foreign": True})
    path.write_bytes(foreign)

    with pytest.raises(source_repair.Stage6SourceRepairError):
        source_repair.create_stage6_source_repair_amendment(
            stage_root=stage,
            current_execution_identity=repair_fixture["supplement_identity"],
            current_verified_review_authorization=repair_fixture[
                "supplement_authorization"
            ],
            current_immutable_bindings=repair_fixture["supplement_immutable"],
        )

    assert path.read_bytes() == foreign


def test_closure_preview_is_non_publishing_then_publication_is_idempotent(
    repair_fixture: dict[str, object],
) -> None:
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    _prepare_closure_cutover(repair_fixture)
    parent_paths = (
        stage / source_repair.SOURCE_REPAIR_AMENDMENT_NAME,
        stage / source_repair.SOURCE_REPAIR_CONTINUATION_NAME,
        stage / source_repair.SOURCE_REPAIR_SUPPLEMENT_NAME,
    )
    parent_payloads = {path: path.read_bytes() for path in parent_paths}
    closure_path = stage / source_repair.SOURCE_REPAIR_CLOSURE_NAME

    preview = source_repair.create_stage6_source_repair_amendment(
        stage_root=stage,
        current_execution_identity=repair_fixture["closure_identity"],
        current_verified_review_authorization=repair_fixture["closure_authorization"],
        current_immutable_bindings=repair_fixture["closure_immutable"],
        publish=False,
    )

    assert preview.closure_sha256 is not None
    assert preview.closure_size_bytes is not None
    assert not closure_path.exists()
    assert all(path.read_bytes() == payload for path, payload in parent_payloads.items())

    published = source_repair.create_stage6_source_repair_amendment(
        stage_root=stage,
        current_execution_identity=repair_fixture["closure_identity"],
        current_verified_review_authorization=repair_fixture["closure_authorization"],
        current_immutable_bindings=repair_fixture["closure_immutable"],
        publish=True,
    )
    replay = source_repair.create_stage6_source_repair_amendment(
        stage_root=stage,
        current_execution_identity=repair_fixture["closure_identity"],
        current_verified_review_authorization=repair_fixture["closure_authorization"],
        current_immutable_bindings=repair_fixture["closure_immutable"],
    )

    assert closure_path.is_file()
    assert preview.closure_sha256 == published.closure_sha256 == replay.closure_sha256
    assert preview.closure_size_bytes == published.closure_size_bytes
    assert published.closure_sha256 == _sha(closure_path.read_bytes())
    assert all(path.read_bytes() == payload for path, payload in parent_payloads.items())


def test_closure_preserves_four_lineage_segments_and_v4_summary(
    repair_fixture: dict[str, object],
) -> None:
    context = _create_closure(repair_fixture)

    assert context.protected_checkpoint_updates == (9, 32)
    assert context.checkpoint_lineage_for_update(9) == repair_fixture["old_lineage"]
    for update in (10, 32):
        lineage = context.checkpoint_lineage_for_update(update)
        assert lineage == repair_fixture["checkpoint32_lineage"]
        assert lineage["source_set_sha256"] == repair_fixture["latest_identity"][
            "source_set_sha256"
        ]
        assert "source_repair_supplement_sha256" not in lineage
        assert "source_repair_closure_sha256" not in lineage

    for update in (33, 50, 100):
        lineage = context.checkpoint_lineage_for_update(update)
        assert lineage["source_set_sha256"] == repair_fixture["closure_identity"][
            "source_set_sha256"
        ]
        assert lineage["source_repair_amendment_ordinal"] == 1
        assert lineage["source_repair_continuation_ordinal"] == 2
        assert lineage["source_repair_supplement_ordinal"] == 3
        assert lineage["source_repair_closure_ordinal"] == 4
        assert lineage["source_repair_amendment_sha256"] == context.amendment_sha256
        assert lineage["source_repair_continuation_sha256"] == (
            context.continuation_sha256
        )
        assert lineage["source_repair_supplement_sha256"] == context.supplement_sha256
        assert lineage["source_repair_closure_sha256"] == context.closure_sha256
        assert lineage["source_repair_closure_first_repaired_update"] == 33

    binding = context.summary_binding
    assert binding["schema_version"] == "stage6_source_repair_summary/v4"
    assert binding["repair_ordinal"] == 4
    assert binding["closure_id"] == source_repair.SOURCE_REPAIR_CLOSURE_ID
    assert binding["closure_sha256"] == context.closure_sha256
    assert binding["supplement_execution_identity_sha256"] == _sha(
        _canonical(repair_fixture["supplement_identity"])
    )
    assert binding["current_execution_identity_sha256"] == _sha(
        _canonical(repair_fixture["closure_identity"])
    )
    assert binding["ordinal_artifacts"] == [
        {
            "repair_ordinal": 1,
            "path": source_repair.SOURCE_REPAIR_AMENDMENT_NAME,
            "sha256": context.amendment_sha256,
        },
        {
            "repair_ordinal": 2,
            "path": source_repair.SOURCE_REPAIR_CONTINUATION_NAME,
            "sha256": context.continuation_sha256,
        },
        {
            "repair_ordinal": 3,
            "path": source_repair.SOURCE_REPAIR_SUPPLEMENT_NAME,
            "sha256": context.supplement_sha256,
        },
        {
            "repair_ordinal": 4,
            "path": source_repair.SOURCE_REPAIR_CLOSURE_NAME,
            "sha256": context.closure_sha256,
        },
    ]


def test_v1_v2_v3_contexts_remain_loadable_after_closure(
    repair_fixture: dict[str, object],
) -> None:
    closure = _create_closure(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    identities = (
        ("current", None, None),
        ("latest", True, None),
        ("supplement", True, True),
        ("closure", True, True),
    )
    for label, has_continuation, has_supplement in identities:
        loaded = source_repair.load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=repair_fixture[f"{label}_identity"],
            current_verified_review_authorization=repair_fixture[
                f"{label}_authorization"
            ],
            current_immutable_bindings=repair_fixture[f"{label}_immutable"],
        )
        assert loaded is not None
        assert (loaded.continuation_sha256 is not None) is bool(has_continuation)
        assert (loaded.supplement_sha256 is not None) is bool(has_supplement)
        assert (loaded.closure_sha256 is not None) is (label == "closure")
    assert closure.closure_sha256 is not None


def test_future_identity_requires_closure_then_fifth_identity_fails_closed(
    repair_fixture: dict[str, object],
) -> None:
    _prepare_closure_cutover(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    with pytest.raises(source_repair.Stage6SourceRepairError, match="closure"):
        source_repair.load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=repair_fixture["closure_identity"],
            current_verified_review_authorization=repair_fixture[
                "closure_authorization"
            ],
            current_immutable_bindings=repair_fixture["closure_immutable"],
        )

    _ = source_repair.create_stage6_source_repair_amendment(
        stage_root=stage,
        current_execution_identity=repair_fixture["closure_identity"],
        current_verified_review_authorization=repair_fixture["closure_authorization"],
        current_immutable_bindings=repair_fixture["closure_immutable"],
        publish=True,
    )
    environment = _environment_identity()
    future_identity = _identity(
        tree="6" * 40,
        source_sha256="0" * 64,
        config_sha256=repair_fixture["closure_identity"]["config_sha256"],
        data_sha256="b" * 64,
        environment=environment,
    )
    future_authorization = _authorization(future_identity, marker="f")
    future_immutable = _immutable(
        future_identity,
        future_authorization,
        environment=environment,
    )
    with pytest.raises(source_repair.Stage6SourceRepairError, match="future|fifth"):
        source_repair.load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=future_identity,
            current_verified_review_authorization=future_authorization,
            current_immutable_bindings=future_immutable,
        )


@pytest.mark.parametrize(
    "mutation",
    ("closure", "supplement", "prefix", "finding_review", "safe_stop", "current"),
)
def test_closure_parent_prefix_evidence_and_identity_drift_fail_closed(
    repair_fixture: dict[str, object],
    mutation: str,
) -> None:
    _create_closure(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    current_identity = copy.deepcopy(repair_fixture["closure_identity"])
    if mutation == "closure":
        path = stage / source_repair.SOURCE_REPAIR_CLOSURE_NAME
        value = json.loads(path.read_text(encoding="utf-8"))
        value["cutover"]["next_resource_attempt"] = 4
        path.write_bytes(_canonical(value))
    elif mutation == "supplement":
        path = stage / source_repair.SOURCE_REPAIR_SUPPLEMENT_NAME
        path.write_bytes(path.read_bytes() + b"tamper")
    elif mutation == "prefix":
        path = stage / "resource_audit.jsonl"
        path.write_bytes(b"{}\n" + path.read_bytes())
    elif mutation in {"finding_review", "safe_stop"}:
        path = repair_fixture["closure_evidence_paths"][mutation]
        path.write_bytes(path.read_bytes() + b"tamper")
    else:
        current_identity["source_set_sha256"] = "1" * 64

    with pytest.raises(source_repair.Stage6SourceRepairError):
        source_repair.load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=current_identity,
            current_verified_review_authorization=repair_fixture[
                "closure_authorization"
            ],
            current_immutable_bindings=repair_fixture["closure_immutable"],
        )


@pytest.mark.parametrize("blocking_kind", ("lease", "update33_checkpoint"))
def test_closure_creation_requires_stopped_checkpoint_free_boundary(
    repair_fixture: dict[str, object],
    blocking_kind: str,
) -> None:
    _prepare_closure_cutover(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    lease = None
    if blocking_kind == "lease":
        lease = RunLease(stage.parent / ".stage6.lease")
        lease.acquire()
    else:
        checkpoint = stage / f"checkpoints/seed-{SEED}/update-00000033"
        checkpoint.mkdir(parents=True)
        (checkpoint / "checkpoint.pt").write_bytes(b"unexpected")

    try:
        with pytest.raises(source_repair.Stage6SourceRepairError):
            source_repair.create_stage6_source_repair_amendment(
                stage_root=stage,
                current_execution_identity=repair_fixture["closure_identity"],
                current_verified_review_authorization=repair_fixture[
                    "closure_authorization"
                ],
                current_immutable_bindings=repair_fixture["closure_immutable"],
            )
    finally:
        if lease is not None:
            lease.release()


def test_closure_creation_accepts_unlocked_persistent_lease_file(
    repair_fixture: dict[str, object],
) -> None:
    _prepare_closure_cutover(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    lease = RunLease(stage.parent / ".stage6.lease")
    lease.acquire()
    lease.release()

    context = source_repair.create_stage6_source_repair_amendment(
        stage_root=stage,
        current_execution_identity=repair_fixture["closure_identity"],
        current_verified_review_authorization=repair_fixture["closure_authorization"],
        current_immutable_bindings=repair_fixture["closure_immutable"],
        publish=True,
    )

    assert context.closure_sha256 is not None


def test_different_closure_is_rejected_without_overwrite(
    repair_fixture: dict[str, object],
) -> None:
    _prepare_closure_cutover(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    path = stage / source_repair.SOURCE_REPAIR_CLOSURE_NAME
    foreign = _canonical({"foreign": True})
    path.write_bytes(foreign)

    with pytest.raises(source_repair.Stage6SourceRepairError):
        source_repair.create_stage6_source_repair_amendment(
            stage_root=stage,
            current_execution_identity=repair_fixture["closure_identity"],
            current_verified_review_authorization=repair_fixture[
                "closure_authorization"
            ],
            current_immutable_bindings=repair_fixture["closure_immutable"],
        )

    assert path.read_bytes() == foreign


def test_continuation_remains_valid_when_append_only_logs_grow(
    repair_fixture: dict[str, object],
) -> None:
    context = _create_continuation(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    for relative in sorted(
        source_repair._APPEND_ONLY_PREFIX_PATHS
        & set(repair_fixture["continuation_prefix"])
    ):
        with (stage / relative).open("ab") as stream:
            stream.write(b"{}\n")

    reloaded = source_repair.load_stage6_source_repair_context(
        stage_root=stage,
        current_execution_identity=repair_fixture["latest_identity"],
        current_verified_review_authorization=repair_fixture[
            "latest_authorization"
        ],
        current_immutable_bindings=repair_fixture["latest_immutable"],
    )

    assert reloaded is not None
    assert reloaded.amendment_sha256 == context.amendment_sha256
    assert reloaded.continuation_sha256 == context.continuation_sha256


@pytest.mark.parametrize(
    "mutation",
    ("amendment_cutover", "checkpoint", "evidence", "current_source"),
)
def test_source_repair_tamper_and_drift_fail_closed(
    repair_fixture: dict[str, object],
    mutation: str,
) -> None:
    context = _create(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    current_identity = copy.deepcopy(repair_fixture["current_identity"])
    current_authorization = copy.deepcopy(repair_fixture["current_authorization"])
    if mutation == "amendment_cutover":
        path = stage / source_repair.SOURCE_REPAIR_AMENDMENT_NAME
        value = json.loads(path.read_text(encoding="utf-8"))
        value["cutover"]["first_repaired_update"] = 11
        path.write_bytes(_canonical(value))
    elif mutation == "checkpoint":
        path = stage / f"checkpoints/seed-{SEED}/update-00000009/checkpoint.pt"
        path.write_bytes(path.read_bytes() + b"tamper")
    elif mutation == "evidence":
        path = repair_fixture["evidence_paths"]["review_report"]
        path.write_bytes(path.read_bytes() + b"tamper")
    else:
        current_identity["source_set_sha256"] = "f" * 64
    with pytest.raises(source_repair.Stage6SourceRepairError):
        source_repair.load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=current_identity,
            current_verified_review_authorization=current_authorization,
            current_immutable_bindings=repair_fixture["current_immutable"],
        )
    del context


@pytest.mark.parametrize(
    "mutation",
    ("continuation", "primary", "second_prefix", "device_evidence", "current"),
)
def test_continuation_tamper_and_drift_fail_closed(
    repair_fixture: dict[str, object],
    mutation: str,
) -> None:
    _create_continuation(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    current_identity = copy.deepcopy(repair_fixture["latest_identity"])
    if mutation == "continuation":
        path = stage / source_repair.SOURCE_REPAIR_CONTINUATION_NAME
        value = json.loads(path.read_text(encoding="utf-8"))
        value["cutover"]["next_resource_attempt"] = 4
        path.write_bytes(_canonical(value))
    elif mutation == "primary":
        path = stage / source_repair.SOURCE_REPAIR_AMENDMENT_NAME
        path.write_bytes(path.read_bytes() + b"tamper")
    elif mutation == "second_prefix":
        path = stage / "resource_audit.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["segment_id"] = "tampered"
        path.write_bytes(_jsonl_payload(rows))
    elif mutation == "device_evidence":
        path = repair_fixture["continuation_evidence_paths"]["device_failure"]
        path.write_bytes(path.read_bytes() + b"tamper")
    else:
        current_identity["source_set_sha256"] = "f" * 64

    with pytest.raises(source_repair.Stage6SourceRepairError):
        source_repair.load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=current_identity,
            current_verified_review_authorization=repair_fixture[
                "latest_authorization"
            ],
            current_immutable_bindings=repair_fixture["latest_immutable"],
        )


def test_second_source_repair_identity_is_rejected_without_overwrite(
    repair_fixture: dict[str, object],
) -> None:
    first = _create(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    path = stage / source_repair.SOURCE_REPAIR_AMENDMENT_NAME
    before = path.read_bytes()
    second_identity = copy.deepcopy(repair_fixture["current_identity"])
    second_identity["source_set_sha256"] = "e" * 64
    with pytest.raises(source_repair.Stage6SourceRepairError):
        source_repair.create_stage6_source_repair_amendment(
            stage_root=stage,
            current_execution_identity=second_identity,
            current_verified_review_authorization=repair_fixture[
                "current_authorization"
            ],
            current_immutable_bindings=repair_fixture["current_immutable"],
        )
    assert path.read_bytes() == before
    assert first.amendment_sha256 == _sha(before)


def test_different_continuation_is_rejected_without_overwrite(
    repair_fixture: dict[str, object],
) -> None:
    context = _create_continuation(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    path = stage / source_repair.SOURCE_REPAIR_CONTINUATION_NAME
    before = path.read_bytes()
    environment = _environment_identity()
    other_identity = _identity(
        tree="4" * 40,
        source_sha256="e" * 64,
        config_sha256=repair_fixture["latest_identity"]["config_sha256"],
        data_sha256="b" * 64,
        environment=environment,
    )
    other_authorization = _authorization(other_identity, marker="d")
    other_immutable = _immutable(
        other_identity,
        other_authorization,
        environment=environment,
    )

    with pytest.raises(
        source_repair.Stage6SourceRepairError,
        match="source-repair supplement prefix drifted",
    ):
        source_repair.create_stage6_source_repair_amendment(
            stage_root=stage,
            current_execution_identity=other_identity,
            current_verified_review_authorization=other_authorization,
            current_immutable_bindings=other_immutable,
        )

    assert path.read_bytes() == before
    assert context.continuation_sha256 == _sha(before)


def test_existing_update10_pre_attempt_resumes_as_attempt_two(
    repair_fixture: dict[str, object],
) -> None:
    _create(repair_fixture)
    rows = _resource_rows()
    backend = object.__new__(StandardProductionBackend)
    backend._resource_audit_rows = lambda: tuple(rows)  # type: ignore[method-assign]
    assert backend._next_resource_attempt(f"0009:{SEED}:update:010") == 2
    assert len(rows) == 29
    assert sum(row.get("phase") == "accepted" for row in rows) == 9


def test_second_cutover_resumes_update10_as_exact_attempt_three(
    repair_fixture: dict[str, object],
) -> None:
    _create_continuation(repair_fixture)
    rows = _resource_rows()
    rows.extend(_device_failure_resource_rows(repair_fixture["resource_payload"]))
    backend = object.__new__(StandardProductionBackend)
    backend._resource_audit_rows = lambda: tuple(rows)  # type: ignore[method-assign]

    assert len(rows) == 31
    assert backend._next_resource_attempt(f"0009:{SEED}:update:010") == 3


def test_production_backend_selects_origin_then_repaired_checkpoint_lineage(
    repair_fixture: dict[str, object],
) -> None:
    context = _create(repair_fixture)
    backend = object.__new__(StandardProductionBackend)
    backend._immutable_bindings = dict(repair_fixture["old_immutable"])
    backend._source_repair = context

    assert backend._checkpoint_lineage_for_update(9) == repair_fixture["old_lineage"]
    assert backend._checkpoint_lineage_for_update(10) == (
        context.checkpoint_lineage_for_update(10)
    )
    assert backend._checkpoint_lineage(10) == context.checkpoint_lineage_for_update(10)


def test_production_backend_uses_continuation_lineage_at_update10_50_and_100(
    repair_fixture: dict[str, object],
) -> None:
    context = _create_continuation(repair_fixture)
    backend = object.__new__(StandardProductionBackend)
    backend._immutable_bindings = dict(repair_fixture["old_immutable"])
    backend._source_repair = context

    assert backend._checkpoint_lineage_for_update(9) == repair_fixture["old_lineage"]
    for update in (10, 50, 100):
        assert backend._checkpoint_lineage_for_update(update) == (
            context.checkpoint_lineage_for_update(update)
        )
    assert backend._checkpoint_lineage(10) == context.checkpoint_lineage_for_update(10)


def test_production_lineage_calls_bind_actual_transaction_updates() -> None:
    expected = {
        "prepare_seed": ["collector_update"],
        "run_update": ["transaction.update"],
        "_checkpoint_replay_audit": ["transaction.update"] * 4,
        "_load_final_runtime": ["record.update"],
    }

    tree = ast.parse(textwrap.dedent(inspect.getsource(StandardProductionBackend)))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    methods = {
        node.name: node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for method_name, expected_arguments in expected.items():
        calls = [
            node
            for node in ast.walk(methods[method_name])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_checkpoint_lineage"
        ]
        assert [ast.unparse(call.args[0]) if call.args else None for call in calls] == (
            expected_arguments
        )


def test_production_backend_switches_to_supplement_lineage_at_update33(
    repair_fixture: dict[str, object],
) -> None:
    context = _create_supplement(repair_fixture)
    backend = object.__new__(StandardProductionBackend)
    backend._immutable_bindings = dict(repair_fixture["old_immutable"])
    backend._source_repair = context

    assert backend._checkpoint_lineage(32) == context.checkpoint_lineage_for_update(32)
    assert backend._checkpoint_lineage(33) == context.checkpoint_lineage_for_update(33)
    with pytest.raises(TypeError):
        backend._checkpoint_lineage()


def test_retention_keeps_cutover_checkpoint_as_explicit_protected_update() -> None:
    manager = object.__new__(CheckpointRetentionManager)
    directories = {9: object(), 10: object(), 11: object()}
    removed: list[object] = []
    manager._validated_directories = lambda: (  # type: ignore[method-assign]
        directories,
        {"update_step": 11},
    )
    manager._remove_update = removed.append  # type: ignore[method-assign]

    receipt = manager.apply(
        latest_update=11,
        periodic_updates=(),
        best_update=11,
        periodic_keep_count=5,
        protected_updates=(9,),
    )

    assert receipt.kept_updates == (9, 11)
    assert receipt.removed_updates == (10,)
    assert removed == [directories[10]]


def test_repaired_summary_and_manifest_review_distinguish_origin_from_current(
    repair_fixture: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = _create(repair_fixture)
    binding = context.summary_binding
    assert binding["schema_version"] == "stage6_source_repair_summary/v1"
    assert binding["origin_source_set_sha256"] == repair_fixture["old_identity"][
        "source_set_sha256"
    ]
    assert binding["current_source_set_sha256"] == repair_fixture[
        "current_identity"
    ]["source_set_sha256"]
    assert binding["amendment_sha256"] == context.amendment_sha256

    monkeypatch.setattr(
        stage6_workflow,
        "stage6_execution_identity",
        lambda **_kwargs: dict(repair_fixture["current_identity"]),
    )
    assert stage6_workflow._stage6_lineage_review_authorization(
        repair_fixture["stage"],
        repo_root=tmp_path,
    ) == repair_fixture["current_authorization"]

    acceptance = build_stage6_acceptance_artifacts(
        global_best={
            "seed": SEED,
            "update": 10,
            "success_rate_under_fixed_step_budget": 0.5,
            "mean_final_coverage": 0.9,
            "checkpoint_ref": "seed-20260716/update-00000010",
        },
        performance_advantage_established=False,
        performance_claim="未建立性能优势",
        ppo_ci95_low=0.1,
        gain_over_cost_ci95_high=0.2,
        final_evaluation_count=10,
        final_episode_count=640,
        checkpoint_receipt_count=100,
        immutable_bindings=repair_fixture["old_immutable"],
        source_repair_binding=binding,
    )
    assert acceptance["summary"]["source_repair"] == binding


def test_final_identity_loader_and_acceptance_bind_optional_continuation(
    repair_fixture: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = _create_continuation(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    loaded = stage6_workflow._load_stage6_source_repair_for_current_identity(
        stage=stage,
        current_execution_identity=repair_fixture["latest_identity"],
        amendment_payload=(
            stage / source_repair.SOURCE_REPAIR_AMENDMENT_NAME
        ).read_bytes(),
        continuation_payload=(
            stage / source_repair.SOURCE_REPAIR_CONTINUATION_NAME
        ).read_bytes(),
    )
    assert loaded is not None
    assert loaded.current_execution_identity == repair_fixture["latest_identity"]
    assert loaded.continuation_sha256 == context.continuation_sha256

    monkeypatch.setattr(
        stage6_workflow,
        "stage6_execution_identity",
        lambda **_kwargs: dict(repair_fixture["latest_identity"]),
    )
    assert stage6_workflow._stage6_lineage_review_authorization(
        stage,
        repo_root=tmp_path,
    ) == repair_fixture["latest_authorization"]

    binding = context.summary_binding
    acceptance = build_stage6_acceptance_artifacts(
        global_best={
            "seed": SEED,
            "update": 100,
            "success_rate_under_fixed_step_budget": 0.5,
            "mean_final_coverage": 0.9,
            "checkpoint_ref": "seed-20260716/update-00000100",
        },
        performance_advantage_established=False,
        performance_claim="未建立性能优势",
        ppo_ci95_low=0.1,
        gain_over_cost_ci95_high=0.2,
        final_evaluation_count=10,
        final_episode_count=640,
        checkpoint_receipt_count=100,
        immutable_bindings=repair_fixture["old_immutable"],
        source_repair_binding=binding,
    )
    assert acceptance["summary"]["source_repair"] == binding
    assert source_repair.SOURCE_REPAIR_CONTINUATION_NAME in (
        stage6_workflow.STAGE6_OPTIONAL_MANIFEST_BOUND_ARTIFACTS
    )


def test_v3_source_repair_summary_is_accepted(
    repair_fixture: dict[str, object],
) -> None:
    binding = _create_supplement(repair_fixture).summary_binding

    acceptance = _acceptance_for_source_repair(
        binding,
        dict(repair_fixture["old_immutable"]),
    )

    assert acceptance["summary"]["source_repair"] == binding


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_supplement_hash",
        "supplement_hash",
        "continuation_identity",
        "ordinal_artifacts",
        "unexpected_field",
        "repair_ordinal",
        "last_continuation_update",
        "first_supplement_update",
        "identity_not_unique",
        "source_not_unique",
        "ordinal_artifact_order",
        "ordinal_artifact_hash",
    ),
)
def test_v3_source_repair_summary_tamper_fails_closed(
    repair_fixture: dict[str, object],
    mutation: str,
) -> None:
    binding = copy.deepcopy(_create_supplement(repair_fixture).summary_binding)
    if mutation == "missing_supplement_hash":
        binding.pop("supplement_sha256")
    elif mutation == "supplement_hash":
        binding["supplement_sha256"] = "f" * 64
    elif mutation == "continuation_identity":
        binding["continuation_execution_identity_sha256"] = binding[
            "parent_execution_identity_sha256"
        ]
    elif mutation == "ordinal_artifacts":
        binding["ordinal_artifacts"][2]["path"] = "wrong-supplement.json"
    elif mutation == "unexpected_field":
        binding["unexpected_field"] = True
    elif mutation == "repair_ordinal":
        binding["repair_ordinal"] = 2
    elif mutation == "last_continuation_update":
        binding["last_continuation_update"] = 31
    elif mutation == "first_supplement_update":
        binding["first_supplement_update"] = 34
    elif mutation == "identity_not_unique":
        binding["current_execution_identity_sha256"] = binding[
            "continuation_execution_identity_sha256"
        ]
    elif mutation == "source_not_unique":
        binding["current_source_set_sha256"] = binding[
            "continuation_source_set_sha256"
        ]
    elif mutation == "ordinal_artifact_order":
        binding["ordinal_artifacts"][1], binding["ordinal_artifacts"][2] = (
            binding["ordinal_artifacts"][2],
            binding["ordinal_artifacts"][1],
        )
    else:
        binding["ordinal_artifacts"][2]["sha256"] = "e" * 64

    with pytest.raises(StandardTrainingError, match="source-repair summary drifted"):
        _acceptance_for_source_repair(
            binding,
            dict(repair_fixture["old_immutable"]),
        )


def test_v4_source_repair_summary_is_accepted(
    repair_fixture: dict[str, object],
) -> None:
    binding = _create_closure(repair_fixture).summary_binding

    acceptance = _acceptance_for_source_repair(
        binding,
        dict(repair_fixture["old_immutable"]),
    )

    assert acceptance["summary"]["source_repair"] == binding


@pytest.mark.parametrize(
    "mutation",
    (
        "closure_id",
        "closure_sha256",
        "closure_sha256_format",
        "first_closure_update",
        "fixed_supplement_parent",
        "fixed_supplement_execution_parent",
        "ordinal_artifact",
        "repair_ordinal",
        "current_identity_not_unique",
        "current_source_not_unique",
        "unexpected_field",
    ),
)
def test_v4_source_repair_summary_tamper_fails_closed(
    repair_fixture: dict[str, object],
    mutation: str,
) -> None:
    binding = copy.deepcopy(_create_closure(repair_fixture).summary_binding)
    if mutation == "closure_id":
        binding["closure_id"] = "wrong-closure/v1"
    elif mutation == "closure_sha256":
        binding["closure_sha256"] = "7" * 64
    elif mutation == "closure_sha256_format":
        binding["closure_sha256"] = "not-a-sha256"
    elif mutation == "first_closure_update":
        binding["first_closure_update"] = 34
    elif mutation == "fixed_supplement_parent":
        binding["supplement_sha256"] = "8" * 64
        binding["ordinal_artifacts"][2]["sha256"] = "8" * 64
    elif mutation == "fixed_supplement_execution_parent":
        binding["supplement_execution_identity_sha256"] = "9" * 64
    elif mutation == "ordinal_artifact":
        binding["ordinal_artifacts"][3]["path"] = "wrong-closure.json"
    elif mutation == "repair_ordinal":
        binding["repair_ordinal"] = 3
    elif mutation == "current_identity_not_unique":
        binding["current_execution_identity_sha256"] = binding[
            "supplement_execution_identity_sha256"
        ]
    elif mutation == "current_source_not_unique":
        binding["current_source_set_sha256"] = binding[
            "supplement_source_set_sha256"
        ]
    else:
        binding["unexpected_field"] = True

    with pytest.raises(StandardTrainingError, match="source-repair summary drifted"):
        _acceptance_for_source_repair(
            binding,
            dict(repair_fixture["old_immutable"]),
        )


@pytest.mark.parametrize("summary_version", (1, 2, 3))
def test_v1_v2_v3_source_repair_summaries_remain_accepted(
    repair_fixture: dict[str, object],
    summary_version: int,
) -> None:
    factory = {
        1: _create,
        2: _create_continuation,
        3: _create_supplement,
    }[summary_version]
    binding = factory(repair_fixture).summary_binding

    acceptance = _acceptance_for_source_repair(
        binding,
        dict(repair_fixture["old_immutable"]),
    )

    assert binding["schema_version"] == (
        f"stage6_source_repair_summary/v{summary_version}"
    )
    assert acceptance["summary"]["source_repair"] == binding


def test_final_loader_manifest_and_review_authorization_bind_optional_supplement(
    repair_fixture: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = _create_supplement(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    loaded = stage6_workflow._load_stage6_source_repair_for_current_identity(
        stage=stage,
        current_execution_identity=repair_fixture["supplement_identity"],
        amendment_payload=(
            stage / source_repair.SOURCE_REPAIR_AMENDMENT_NAME
        ).read_bytes(),
        continuation_payload=(
            stage / source_repair.SOURCE_REPAIR_CONTINUATION_NAME
        ).read_bytes(),
        supplement_payload=(
            stage / source_repair.SOURCE_REPAIR_SUPPLEMENT_NAME
        ).read_bytes(),
    )
    assert loaded is not None
    assert loaded.current_execution_identity == repair_fixture["supplement_identity"]
    assert loaded.supplement_sha256 == context.supplement_sha256

    monkeypatch.setattr(
        stage6_workflow,
        "stage6_execution_identity",
        lambda **_kwargs: dict(repair_fixture["supplement_identity"]),
    )
    assert stage6_workflow._stage6_lineage_review_authorization(
        stage,
        repo_root=tmp_path,
        source_repair_amendment_payload=(
            stage / source_repair.SOURCE_REPAIR_AMENDMENT_NAME
        ).read_bytes(),
        source_repair_continuation_payload=(
            stage / source_repair.SOURCE_REPAIR_CONTINUATION_NAME
        ).read_bytes(),
        source_repair_supplement_payload=(
            stage / source_repair.SOURCE_REPAIR_SUPPLEMENT_NAME
        ).read_bytes(),
    ) == repair_fixture["supplement_authorization"]
    assert source_repair.SOURCE_REPAIR_SUPPLEMENT_NAME in (
        stage6_workflow.STAGE6_OPTIONAL_MANIFEST_BOUND_ARTIFACTS
    )


def test_final_loader_manifest_and_review_authorization_bind_optional_closure(
    repair_fixture: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = _create_closure(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    payloads = {
        name: (stage / name).read_bytes()
        for name in (
            source_repair.SOURCE_REPAIR_AMENDMENT_NAME,
            source_repair.SOURCE_REPAIR_CONTINUATION_NAME,
            source_repair.SOURCE_REPAIR_SUPPLEMENT_NAME,
            source_repair.SOURCE_REPAIR_CLOSURE_NAME,
        )
    }
    loaded = stage6_workflow._load_stage6_source_repair_for_current_identity(
        stage=stage,
        current_execution_identity=repair_fixture["closure_identity"],
        amendment_payload=payloads[source_repair.SOURCE_REPAIR_AMENDMENT_NAME],
        continuation_payload=payloads[source_repair.SOURCE_REPAIR_CONTINUATION_NAME],
        supplement_payload=payloads[source_repair.SOURCE_REPAIR_SUPPLEMENT_NAME],
        closure_payload=payloads[source_repair.SOURCE_REPAIR_CLOSURE_NAME],
    )
    assert loaded is not None
    assert loaded.current_execution_identity == repair_fixture["closure_identity"]
    assert loaded.closure_sha256 == context.closure_sha256

    monkeypatch.setattr(
        stage6_workflow,
        "stage6_execution_identity",
        lambda **_kwargs: dict(repair_fixture["closure_identity"]),
    )
    assert stage6_workflow._stage6_lineage_review_authorization(
        stage,
        repo_root=tmp_path,
        source_repair_amendment_payload=payloads[
            source_repair.SOURCE_REPAIR_AMENDMENT_NAME
        ],
        source_repair_continuation_payload=payloads[
            source_repair.SOURCE_REPAIR_CONTINUATION_NAME
        ],
        source_repair_supplement_payload=payloads[
            source_repair.SOURCE_REPAIR_SUPPLEMENT_NAME
        ],
        source_repair_closure_payload=payloads[
            source_repair.SOURCE_REPAIR_CLOSURE_NAME
        ],
    ) == repair_fixture["closure_authorization"]
    assert source_repair.SOURCE_REPAIR_CLOSURE_NAME in (
        stage6_workflow.STAGE6_OPTIONAL_MANIFEST_BOUND_ARTIFACTS
    )


def test_stage6_loader_rejects_closure_without_complete_predecessor_chain(
    repair_fixture: dict[str, object],
) -> None:
    _create_closure(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    closure_payload = (stage / source_repair.SOURCE_REPAIR_CLOSURE_NAME).read_bytes()
    (stage / source_repair.SOURCE_REPAIR_SUPPLEMENT_NAME).unlink()

    with pytest.raises(stage6_workflow.Stage6WorkflowError, match="closure"):
        stage6_workflow._load_stage6_source_repair_for_current_identity(
            stage=stage,
            current_execution_identity=repair_fixture["closure_identity"],
            amendment_payload=(
                stage / source_repair.SOURCE_REPAIR_AMENDMENT_NAME
            ).read_bytes(),
            continuation_payload=(
                stage / source_repair.SOURCE_REPAIR_CONTINUATION_NAME
            ).read_bytes(),
            supplement_payload=None,
            closure_payload=closure_payload,
        )


def test_terminal_and_recovery_evidence_paths_bind_closure_payload() -> None:
    assert "closure_payload" in inspect.signature(
        stage6_workflow._load_stage6_source_repair_for_current_identity
    ).parameters
    assert "source_repair_closure_payload" in inspect.signature(
        stage6_workflow._stage6_lineage_review_authorization
    ).parameters
    for function in (
        stage6_workflow._verify_stage6_terminal_manifest_evidence,
        stage6_workflow._verify_stage6_machine_acceptance_from_bound_graph,
        stage6_workflow.verify_stage6_machine_acceptance,
        stage6_workflow._verify_stage6_recovery_authorization_bindings,
    ):
        assert "source-repair-closure.json" in inspect.getsource(function)


def test_creation_script_reports_continuation_without_changing_primary_output(
    tmp_path: Path,
) -> None:
    from scripts import create_ppo_stage6_source_repair_amendment as script

    primary = script._publication_result(
        stage_root=tmp_path / "s6",
        context=SimpleNamespace(
            amendment_sha256="a" * 64,
            amendment_size_bytes=100,
            continuation_sha256=None,
            continuation_size_bytes=None,
        ),
    )
    continued = script._publication_result(
        stage_root=tmp_path / "s6",
        context=SimpleNamespace(
            amendment_sha256="a" * 64,
            amendment_size_bytes=100,
            continuation_sha256="b" * 64,
            continuation_size_bytes=200,
        ),
    )

    assert set(primary) == {
        "formal_run_id",
        "stage_root",
        "amendment_sha256",
        "amendment_size_bytes",
        "cutover",
    }
    assert continued["continuation_sha256"] == "b" * 64
    assert continued["continuation_size_bytes"] == 200
    assert continued["repair_ordinal"] == 2
    assert continued["next_resource_attempt"] == 3


def test_creation_script_reports_ordinal3_supplement(tmp_path: Path) -> None:
    from scripts import create_ppo_stage6_source_repair_amendment as script

    result = script._publication_result(
        stage_root=tmp_path / "s6",
        context=SimpleNamespace(
            amendment_sha256="a" * 64,
            amendment_size_bytes=100,
            continuation_sha256="b" * 64,
            continuation_size_bytes=200,
            supplement_sha256="c" * 64,
            supplement_size_bytes=300,
        ),
    )

    assert result["amendment_sha256"] == "a" * 64
    assert result["continuation_sha256"] == "b" * 64
    assert result["supplement_sha256"] == "c" * 64
    assert result["supplement_size_bytes"] == 300
    assert result["repair_ordinal"] == 3
    assert result["accepted_prefix_updates"] == 32
    assert result["failed_update"] == 33
    assert result["next_resource_attempt"] == 2


def test_creation_script_reports_ordinal4_closure_and_supports_dry_run(
    tmp_path: Path,
) -> None:
    from scripts import create_ppo_stage6_source_repair_amendment as script

    result = script._publication_result(
        stage_root=tmp_path / "s6",
        context=SimpleNamespace(
            amendment_sha256="a" * 64,
            amendment_size_bytes=100,
            continuation_sha256="b" * 64,
            continuation_size_bytes=200,
            supplement_sha256="c" * 64,
            supplement_size_bytes=300,
            closure_sha256="d" * 64,
            closure_size_bytes=400,
        ),
    )
    parser = script.build_parser()
    args = parser.parse_args(
        ["--review-authorization", "review.json", "--dry-run"]
    )
    default_args = parser.parse_args(
        ["--review-authorization", "review.json"]
    )
    publish_args = parser.parse_args(
        ["--review-authorization", "review.json", "--publish"]
    )

    assert result["closure_sha256"] == "d" * 64
    assert result["closure_size_bytes"] == 400
    assert result["repair_ordinal"] == 4
    assert result["accepted_prefix_updates"] == 32
    assert result["failed_update"] == 33
    assert result["next_resource_attempt"] == 3
    assert result["next_segment_index"] == 5
    assert args.dry_run is True
    assert args.publish is False
    assert default_args.publish is False
    assert publish_args.publish is True
    main_source = inspect.getsource(script.main)
    assert "publish=args.publish" in main_source


def test_frontier_recovery_preview_is_nonpublishing_atomic_and_deterministic(
    repair_fixture: dict[str, object],
) -> None:
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    _prepare_frontier_recovery_cutover(repair_fixture)
    parent_paths = tuple(
        stage / relative
        for relative in (
            "source-repair-amendment.json",
            "source-repair-continuation.json",
            "source-repair-supplement.json",
            "source-repair-closure.json",
        )
    )
    parent_payloads = {path: path.read_bytes() for path in parent_paths}
    frontier_path = stage / "source-repair-frontier-recovery.json"
    lease_path = stage.parent / ".stage6.lease"
    lease_path.unlink(missing_ok=True)
    assert not lease_path.exists()

    preview = source_repair.create_stage6_source_repair_amendment(
        stage_root=stage,
        current_execution_identity=repair_fixture["frontier_recovery_identity"],
        current_verified_review_authorization=repair_fixture[
            "frontier_recovery_authorization"
        ],
        current_immutable_bindings=repair_fixture["frontier_recovery_immutable"],
        publish=False,
    )

    assert preview.frontier_recovery_sha256 is not None
    assert preview.frontier_recovery_size_bytes is not None
    assert not frontier_path.exists()
    assert not lease_path.exists()
    assert all(path.read_bytes() == payload for path, payload in parent_payloads.items())

    foreign = _canonical({"foreign": True})
    frontier_path.write_bytes(foreign)
    with pytest.raises(source_repair.Stage6SourceRepairError):
        source_repair.create_stage6_source_repair_amendment(
            stage_root=stage,
            current_execution_identity=repair_fixture[
                "frontier_recovery_identity"
            ],
            current_verified_review_authorization=repair_fixture[
                "frontier_recovery_authorization"
            ],
            current_immutable_bindings=repair_fixture[
                "frontier_recovery_immutable"
            ],
        )
    assert frontier_path.read_bytes() == foreign
    frontier_path.unlink()

    published = source_repair.create_stage6_source_repair_amendment(
        stage_root=stage,
        current_execution_identity=repair_fixture["frontier_recovery_identity"],
        current_verified_review_authorization=repair_fixture[
            "frontier_recovery_authorization"
        ],
        current_immutable_bindings=repair_fixture["frontier_recovery_immutable"],
        publish=True,
    )
    first_payload = frontier_path.read_bytes()
    replay = source_repair.create_stage6_source_repair_amendment(
        stage_root=stage,
        current_execution_identity=repair_fixture["frontier_recovery_identity"],
        current_verified_review_authorization=repair_fixture[
            "frontier_recovery_authorization"
        ],
        current_immutable_bindings=repair_fixture["frontier_recovery_immutable"],
    )

    assert first_payload == frontier_path.read_bytes()
    assert preview.frontier_recovery_sha256 == _sha(first_payload)
    assert preview.frontier_recovery_sha256 == published.frontier_recovery_sha256
    assert published.frontier_recovery_sha256 == replay.frontier_recovery_sha256
    assert published.frontier_recovery_size_bytes == len(first_payload)
    assert all(path.read_bytes() == payload for path, payload in parent_payloads.items())


def test_current_frontier_identity_requires_missing_ordinal5_fail_closed(
    repair_fixture: dict[str, object],
) -> None:
    _prepare_frontier_recovery_cutover(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)

    with pytest.raises(
        source_repair.Stage6SourceRepairError,
        match="frontier recovery is required",
    ):
        source_repair.load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=repair_fixture[
                "frontier_recovery_identity"
            ],
            current_verified_review_authorization=repair_fixture[
                "frontier_recovery_authorization"
            ],
            current_immutable_bindings=repair_fixture[
                "frontier_recovery_immutable"
            ],
        )
    assert not (stage / "source-repair-frontier-recovery.json").exists()


def test_frontier_recovery_preserves_update46_and_emits_full_v5_lineage(
    repair_fixture: dict[str, object],
) -> None:
    closure = _prepare_frontier_recovery_cutover(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    closure_lineage = closure.checkpoint_lineage_for_update(46)
    context = source_repair.create_stage6_source_repair_amendment(
        stage_root=stage,
        current_execution_identity=repair_fixture["frontier_recovery_identity"],
        current_verified_review_authorization=repair_fixture[
            "frontier_recovery_authorization"
        ],
        current_immutable_bindings=repair_fixture["frontier_recovery_immutable"],
        publish=True,
    )

    assert context.protected_checkpoint_updates == (9, 32, 46)
    assert context.checkpoint_lineage_for_update(46) == closure_lineage
    assert "source_repair_frontier_recovery_sha256" not in closure_lineage
    for update in (47, 50, 100):
        lineage = context.checkpoint_lineage_for_update(update)
        assert lineage["source_set_sha256"] == repair_fixture[
            "frontier_recovery_identity"
        ]["source_set_sha256"]
        assert lineage["source_repair_amendment_ordinal"] == 1
        assert lineage["source_repair_continuation_ordinal"] == 2
        assert lineage["source_repair_supplement_ordinal"] == 3
        assert lineage["source_repair_closure_ordinal"] == 4
        assert lineage["source_repair_frontier_recovery_ordinal"] == 5
        assert lineage["source_repair_frontier_recovery_sha256"] == (
            context.frontier_recovery_sha256
        )
        assert lineage["source_repair_frontier_recovery_last_parent_update"] == 46
        assert lineage["source_repair_frontier_recovery_first_repaired_update"] == 47

    binding = context.summary_binding
    assert binding["schema_version"] == "stage6_source_repair_summary/v5"
    assert binding["repair_ordinal"] == 5
    assert binding["frontier_recovery_id"] == (
        "stage6a_irregular_frontier_empty_update47/v1"
    )
    assert binding["frontier_recovery_sha256"] == context.frontier_recovery_sha256
    assert binding["last_closure_update"] == 46
    assert binding["first_frontier_recovery_update"] == 47
    assert binding["closure_execution_identity_sha256"] == _sha(
        _canonical(repair_fixture["closure_identity"])
    )
    assert binding["closure_source_set_sha256"] == repair_fixture[
        "closure_identity"
    ]["source_set_sha256"]
    assert binding["ordinal_artifacts"][-1] == {
        "repair_ordinal": 5,
        "path": "source-repair-frontier-recovery.json",
        "sha256": context.frontier_recovery_sha256,
    }


def test_frontier_recovery_artifact_parent_prefix_checkpoint_evidence_and_source_drift(
    repair_fixture: dict[str, object],
) -> None:
    context = _create_frontier_recovery(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    source_root = repair_fixture["frontier_recovery_source_root"]
    assert isinstance(source_root, Path)
    paths = (
        stage / "source-repair-frontier-recovery.json",
        stage / "source-repair-closure.json",
        stage / "training_metrics.jsonl",
        stage / f"checkpoints/seed-{SEED}/update-00000046/checkpoint.pt",
        repair_fixture["frontier_recovery_evidence_paths"]["root_cause"],
        source_root / "src/lunar_exploration_ppo/env/frontier.py",
    )
    for path in paths:
        assert isinstance(path, Path)
        original = path.read_bytes()
        path.write_bytes(b"tamper\n" + original)
        try:
            with pytest.raises(source_repair.Stage6SourceRepairError):
                source_repair.load_stage6_source_repair_context(
                    stage_root=stage,
                    current_execution_identity=repair_fixture[
                        "frontier_recovery_identity"
                    ],
                    current_verified_review_authorization=repair_fixture[
                        "frontier_recovery_authorization"
                    ],
                    current_immutable_bindings=repair_fixture[
                        "frontier_recovery_immutable"
                    ],
                )
        finally:
            path.write_bytes(original)

    changed_identity = copy.deepcopy(repair_fixture["frontier_recovery_identity"])
    changed_identity["source_set_sha256"] = "2" * 64
    with pytest.raises(source_repair.Stage6SourceRepairError):
        source_repair.load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=changed_identity,
            current_verified_review_authorization=repair_fixture[
                "frontier_recovery_authorization"
            ],
            current_immutable_bindings=repair_fixture["frontier_recovery_immutable"],
        )

    closure_path = stage / "source-repair-closure.json"
    closure_payload = closure_path.read_bytes()
    closure_path.unlink()
    try:
        with pytest.raises(
            source_repair.Stage6SourceRepairError,
            match="frontier recovery|closure",
        ):
            source_repair.load_stage6_source_repair_context(
                stage_root=stage,
                current_execution_identity=repair_fixture[
                    "frontier_recovery_identity"
                ],
                current_verified_review_authorization=repair_fixture[
                    "frontier_recovery_authorization"
                ],
                current_immutable_bindings=repair_fixture[
                    "frontier_recovery_immutable"
                ],
            )
    finally:
        closure_path.write_bytes(closure_payload)

    environment = _environment_identity()
    future_identity = _identity(
        tree="7" * 40,
        source_sha256="1" * 64,
        config_sha256=repair_fixture["frontier_recovery_identity"]["config_sha256"],
        data_sha256="b" * 64,
        environment=environment,
    )
    future_authorization = _authorization(future_identity, marker="1")
    future_immutable = _immutable(
        future_identity,
        future_authorization,
        environment=environment,
    )
    with pytest.raises(source_repair.Stage6SourceRepairError, match="future|sixth"):
        source_repair.load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=future_identity,
            current_verified_review_authorization=future_authorization,
            current_immutable_bindings=future_immutable,
        )
    assert context.frontier_recovery_sha256 is not None


def test_v5_source_repair_summary_round_trips_and_tamper_or_future_fails_closed(
    repair_fixture: dict[str, object],
) -> None:
    binding = _create_frontier_recovery(repair_fixture).summary_binding

    acceptance = _acceptance_for_source_repair(
        binding,
        dict(repair_fixture["old_immutable"]),
    )
    assert acceptance["summary"]["source_repair"] == binding

    mutations = []
    missing_hash = copy.deepcopy(binding)
    missing_hash.pop("frontier_recovery_sha256")
    mutations.append(missing_hash)
    bad_parent = copy.deepcopy(binding)
    bad_parent["closure_sha256"] = "7" * 64
    bad_parent["ordinal_artifacts"][3]["sha256"] = "7" * 64
    mutations.append(bad_parent)
    duplicate_identity = copy.deepcopy(binding)
    duplicate_identity["current_execution_identity_sha256"] = duplicate_identity[
        "closure_execution_identity_sha256"
    ]
    mutations.append(duplicate_identity)
    wrong_order = copy.deepcopy(binding)
    wrong_order["ordinal_artifacts"][3], wrong_order["ordinal_artifacts"][4] = (
        wrong_order["ordinal_artifacts"][4],
        wrong_order["ordinal_artifacts"][3],
    )
    mutations.append(wrong_order)
    future = copy.deepcopy(binding)
    future["schema_version"] = "stage6_source_repair_summary/v6"
    future["repair_ordinal"] = 6
    mutations.append(future)
    unexpected = copy.deepcopy(binding)
    unexpected["unexpected_field"] = True
    mutations.append(unexpected)

    for mutated in mutations:
        with pytest.raises(StandardTrainingError, match="source-repair summary drifted"):
            _acceptance_for_source_repair(
                mutated,
                dict(repair_fixture["old_immutable"]),
            )


_ORDINAL5_ARTIFACT_NAMES = (
    source_repair.SOURCE_REPAIR_AMENDMENT_NAME,
    source_repair.SOURCE_REPAIR_CONTINUATION_NAME,
    source_repair.SOURCE_REPAIR_SUPPLEMENT_NAME,
    source_repair.SOURCE_REPAIR_CLOSURE_NAME,
    source_repair.SOURCE_REPAIR_FRONTIER_RECOVERY_NAME,
)
_ORDINAL5_LOADER_ARGUMENTS = (
    "amendment_payload",
    "continuation_payload",
    "supplement_payload",
    "closure_payload",
    "frontier_recovery_payload",
)
_ORDINAL5_LINEAGE_ARGUMENTS = (
    "source_repair_amendment_payload",
    "source_repair_continuation_payload",
    "source_repair_supplement_payload",
    "source_repair_closure_payload",
    "source_repair_frontier_recovery_payload",
)


class _WorkflowConsumerProbeComplete(RuntimeError):
    pass


def _write_workflow_consumer_probe_artifacts(
    stage: Path,
    *,
    source_identity: dict[str, object],
) -> None:
    for relative in (
        "summary.json",
        "routing.json",
        "global-best.json",
        "standard_checkpoint_manifest.json",
    ):
        _write_json(stage / relative, {})
    _write_json(stage / "preterminal_acceptance.json", {"immutable_bindings": {}})
    _write_json(
        stage / "manifest.json",
        {
            "schema_version": "stage6_sha256_manifest/v3",
            "source_identity": source_identity,
            "verified_review_authorization": {},
            "preterminal_evidence_binding": {},
            "artifacts": [],
        },
    )


@contextmanager
def _complete_ordinal5_consumer_graph(stage: Path):
    """Require the runtime consumer fixture to expose the complete chain."""

    frontier_path = stage / source_repair.SOURCE_REPAIR_FRONTIER_RECOVERY_NAME
    assert frontier_path.is_file()
    yield


def _payload_capture(
    captures: dict[str, tuple[bytes | None, ...]],
    *,
    consumer: str,
    argument_names: tuple[str, ...],
):
    def capture(*_args: object, **kwargs: object) -> object:
        captures[consumer] = tuple(
            kwargs.get(argument_name)  # type: ignore[arg-type]
            for argument_name in argument_names
        )
        raise _WorkflowConsumerProbeComplete(consumer)

    return capture


def _assert_ordinal5_payload_capture(
    captured: tuple[bytes | None, ...],
    *,
    payloads: dict[str, bytes],
    context: object,
) -> None:
    expected_payloads = tuple(payloads[name] for name in _ORDINAL5_ARTIFACT_NAMES)
    assert captured == expected_payloads
    identities = tuple(
        {"sha256": _sha(payload), "size_bytes": len(payload)}
        for payload in captured
        if payload is not None
    )
    assert identities == (
        {
            "sha256": context.amendment_sha256,
            "size_bytes": context.amendment_size_bytes,
        },
        {
            "sha256": context.continuation_sha256,
            "size_bytes": context.continuation_size_bytes,
        },
        {
            "sha256": context.supplement_sha256,
            "size_bytes": context.supplement_size_bytes,
        },
        {
            "sha256": context.closure_sha256,
            "size_bytes": context.closure_size_bytes,
        },
        {
            "sha256": context.frontier_recovery_sha256,
            "size_bytes": context.frontier_recovery_size_bytes,
        },
    )
    frontier_value = json.loads(expected_payloads[-1])
    assert [
        row["artifact"]["path"] for row in frontier_value["repair_chain"]
    ] == list(_ORDINAL5_ARTIFACT_NAMES[:-1])
    assert frontier_value["repair_chain"][-1]["artifact"] == {
        "path": source_repair.SOURCE_REPAIR_CLOSURE_NAME,
        **identities[-2],
    }


def _execute_ordinal5_workflow_consumers(
    *,
    repair_fixture: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_identity: dict[str, object],
) -> dict[str, tuple[bytes | None, ...]]:
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    captures: dict[str, tuple[bytes | None, ...]] = {}
    config = load_stage6_config(CONFIG)
    authority = SimpleNamespace(
        identity={
            "gate_sha256": config.stage5_authority.gate_sha256,
            "commit_sha256": config.stage5_authority.commit_sha256,
            "commit_tree": config.stage5_authority.commit_tree,
            "review_sha256": config.stage5_authority.review_sha256,
            "manifest_sha256": config.stage5_authority.manifest_sha256,
            "checkpoint_sha256": config.stage5_authority.checkpoint_sha256,
            "policy_state_sha256": config.stage5_authority.policy_state_sha256,
            "performance_advantage_established": False,
        },
        require_current=lambda _label: None,
    )
    monkeypatch.setattr(
        stage6_workflow,
        "stage6_execution_identity",
        lambda **_kwargs: dict(repair_fixture["frontier_recovery_identity"]),
    )
    monkeypatch.setattr(
        stage6_workflow,
        "verify_frozen_stage5_authority",
        lambda **_kwargs: authority,
    )

    with _complete_ordinal5_consumer_graph(stage):
        with stage6_terminal_recovery._capture_terminal_evidence_handle(
            stage
        ) as evidence_handle:
            monkeypatch.setattr(
                stage6_workflow,
                "_stage6_lineage_review_authorization",
                _payload_capture(
                    captures,
                    consumer="terminal_manifest_evidence",
                    argument_names=_ORDINAL5_LINEAGE_ARGUMENTS,
                ),
            )
            with pytest.raises(
                _WorkflowConsumerProbeComplete,
                match="terminal_manifest_evidence",
            ):
                stage6_workflow._verify_stage6_terminal_manifest_evidence(
                    stage,
                    source_identity=source_identity,
                    repo_root=tmp_path,
                    evidence_handle=evidence_handle,
                )

    with _complete_ordinal5_consumer_graph(stage):
        with stage6_terminal_recovery._capture_terminal_evidence_handle(
            stage
        ) as evidence_handle:
            monkeypatch.setattr(
                stage6_workflow,
                "_load_stage6_source_repair_for_current_identity",
                _payload_capture(
                    captures,
                    consumer="machine_acceptance_from_bound_graph",
                    argument_names=_ORDINAL5_LOADER_ARGUMENTS,
                ),
            )
            with pytest.raises(
                _WorkflowConsumerProbeComplete,
                match="machine_acceptance_from_bound_graph",
            ):
                stage6_workflow._verify_stage6_machine_acceptance_from_bound_graph(
                    stage_root=stage,
                    repo_root=tmp_path,
                    require_terminal=False,
                    evidence_handle=evidence_handle,
                )

    monkeypatch.setattr(
        stage6_terminal_recovery,
        "detect_stage6_terminal_recovery",
        lambda **_kwargs: {
            "status": "valid_terminal_recovery",
            "phase_states": stage6_terminal_recovery.COMPLETE_STATES,
        },
    )
    monkeypatch.setattr(
        stage6_workflow,
        "_stage6_lineage_review_authorization",
        _payload_capture(
            captures,
            consumer="machine_acceptance",
            argument_names=_ORDINAL5_LINEAGE_ARGUMENTS,
        ),
    )
    with _complete_ordinal5_consumer_graph(stage):
        with pytest.raises(
            _WorkflowConsumerProbeComplete,
            match="machine_acceptance",
        ):
            stage6_workflow.verify_stage6_machine_acceptance(
                stage_root=stage,
                repo_root=tmp_path,
            )

    monkeypatch.setattr(
        stage6_terminal_recovery,
        "_validate_receipt_payload",
        lambda payload: ({"immutable_bindings": {}}, payload, {}),
    )
    monkeypatch.setattr(
        stage6_workflow,
        "_require_review_authorization_current",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        stage6_workflow,
        "_review_authorization_record",
        lambda _handle: dict(repair_fixture["frontier_recovery_authorization"]),
    )
    monkeypatch.setattr(
        stage6_workflow,
        "_stage6_lineage_review_authorization",
        _payload_capture(
            captures,
            consumer="recovery_authorization_bindings",
            argument_names=_ORDINAL5_LINEAGE_ARGUMENTS,
        ),
    )
    with _complete_ordinal5_consumer_graph(stage):
        with stage6_terminal_recovery._capture_terminal_evidence_handle(
            stage
        ) as evidence_handle:
            with pytest.raises(
                _WorkflowConsumerProbeComplete,
                match="recovery_authorization_bindings",
            ):
                stage6_workflow._verify_stage6_recovery_authorization_bindings(
                    stage_root=stage,
                    repo_root=tmp_path,
                    config_path=CONFIG,
                    run_id=RUN_ID,
                    stage5_gate_path=tmp_path / "stage5-gate.json",
                    review_authorization_handle=object(),
                    evidence_handle=evidence_handle,
                )
    return captures


def test_workflow_round_trips_frontier_recovery_and_binds_all_consumers(
    repair_fixture: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = _create_frontier_recovery(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    payloads = {
        name: (stage / name).read_bytes()
        for name in (
            "source-repair-amendment.json",
            "source-repair-continuation.json",
            "source-repair-supplement.json",
            "source-repair-closure.json",
            "source-repair-frontier-recovery.json",
        )
    }
    loaded = stage6_workflow._load_stage6_source_repair_for_current_identity(
        stage=stage,
        current_execution_identity=repair_fixture["frontier_recovery_identity"],
        amendment_payload=payloads["source-repair-amendment.json"],
        continuation_payload=payloads["source-repair-continuation.json"],
        supplement_payload=payloads["source-repair-supplement.json"],
        closure_payload=payloads["source-repair-closure.json"],
        frontier_recovery_payload=payloads["source-repair-frontier-recovery.json"],
    )
    assert loaded is not None
    assert loaded.frontier_recovery_sha256 == context.frontier_recovery_sha256

    monkeypatch.setattr(
        stage6_workflow,
        "stage6_execution_identity",
        lambda **_kwargs: dict(repair_fixture["frontier_recovery_identity"]),
    )
    assert stage6_workflow._stage6_lineage_review_authorization(
        stage,
        repo_root=tmp_path,
        source_repair_amendment_payload=payloads["source-repair-amendment.json"],
        source_repair_continuation_payload=payloads[
            "source-repair-continuation.json"
        ],
        source_repair_supplement_payload=payloads["source-repair-supplement.json"],
        source_repair_closure_payload=payloads["source-repair-closure.json"],
        source_repair_frontier_recovery_payload=payloads[
            "source-repair-frontier-recovery.json"
        ],
    ) == repair_fixture["frontier_recovery_authorization"]
    assert "source-repair-frontier-recovery.json" in (
        stage6_workflow.STAGE6_OPTIONAL_MANIFEST_BOUND_ARTIFACTS
    )
    source_identity = {"schema_version": "ordinal5_consumer_probe/v1"}
    _write_workflow_consumer_probe_artifacts(
        stage,
        source_identity=source_identity,
    )
    captures = _execute_ordinal5_workflow_consumers(
        repair_fixture=repair_fixture,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        source_identity=source_identity,
    )
    assert tuple(captures) == (
        "terminal_manifest_evidence",
        "machine_acceptance_from_bound_graph",
        "machine_acceptance",
        "recovery_authorization_bindings",
    )
    for captured in captures.values():
        _assert_ordinal5_payload_capture(
            captured,
            payloads=payloads,
            context=context,
        )


def test_creation_script_reports_ordinal5_frontier_recovery(tmp_path: Path) -> None:
    from scripts import create_ppo_stage6_source_repair_amendment as script

    result = script._publication_result(
        stage_root=tmp_path / "s6",
        context=SimpleNamespace(
            amendment_sha256="a" * 64,
            amendment_size_bytes=100,
            continuation_sha256="b" * 64,
            continuation_size_bytes=200,
            supplement_sha256="c" * 64,
            supplement_size_bytes=300,
            closure_sha256="d" * 64,
            closure_size_bytes=400,
            frontier_recovery_sha256="e" * 64,
            frontier_recovery_size_bytes=500,
        ),
    )

    assert result["frontier_recovery_sha256"] == "e" * 64
    assert result["frontier_recovery_size_bytes"] == 500
    assert result["repair_ordinal"] == 5
    assert result["accepted_prefix_updates"] == 46
    assert result["failed_update"] == 47
    assert result["next_resource_attempt"] == 2
    assert result["next_segment_index"] == 6
    normalized_help = " ".join(script.build_parser().format_help().split()).lower()
    assert "ordinal5 frontier recovery" in normalized_help
    assert "default mode is preview-only" in normalized_help
    assert "--publish" in normalized_help
    assert "explicitly publish" in normalized_help


def test_fixed_sixth_cutover_contract_binds_formal_cache_and_update50() -> None:
    assert source_repair.SOURCE_REPAIR_SENSOR_ACCELERATION_NAME == (
        "source-repair-sensor-acceleration.json"
    )
    assert source_repair.SOURCE_REPAIR_SENSOR_ACCELERATION_SCHEMA == (
        "stage6_source_repair_sensor_acceleration/v1"
    )
    assert source_repair.SOURCE_REPAIR_SENSOR_ACCELERATION_SUMMARY_SCHEMA == (
        "stage6_source_repair_summary/v6"
    )
    assert source_repair.SOURCE_REPAIR_SENSOR_ACCELERATION_ID == (
        "stage6a_sensor_hotpath_exact_coverable_cache_update50/v1"
    )
    assert source_repair.LAST_SENSOR_ACCELERATION_PARENT_UPDATE == 49
    assert source_repair.FIRST_SENSOR_ACCELERATION_UPDATE == 50
    assert source_repair.SENSOR_ACCELERATION_NEXT_TRANSACTION_KEY == (
        "0049:20260716:update:050"
    )
    assert source_repair.SENSOR_ACCELERATION_FAILED_RESOURCE_ATTEMPT == 1
    assert source_repair.SENSOR_ACCELERATION_NEXT_RESOURCE_ATTEMPT == 2
    assert source_repair.SENSOR_ACCELERATION_NEXT_SEGMENT_INDEX == 7
    assert source_repair._FIXED_FRONTIER_RECOVERY_BINDING == {
        "path": "source-repair-frontier-recovery.json",
        "sha256": "705367a220455b3577b103e3277e0692aa99eaa47ada24e5fcd1e0dc155d732f",
        "size_bytes": 30690,
    }
    assert source_repair._FIXED_COVERAGE_CACHE_MANIFEST_BINDING == {
        "path": (
            "D:/xunce/review/s6-coverable-cache-r1-20260722T144932/"
            "coverage-cache-manifest.json"
        ),
        "sha256": "675b923b64cee4cde90d5728550a7f939c41cfbc56e9c237268078b59641081c",
        "size_bytes": 519284,
        "cache_root": (
            "D:/xunce/cache/ppo_frontier/"
            "s6-standard-single-r1-20260718T220434Z/coverage-v1"
        ),
        "entry_set_sha256": (
            "c543a277154b41d14bd1a194d6c6e4f2b43075d5b93096c50d1952ee44c43558"
        ),
        "entry_count": 1064,
        "split_counts": {
            "train": 700,
            "validation": 150,
            "test": 150,
            "unseen": 64,
        },
    }


def test_fixed_frontier_recovery_parent_matches_published_ordinal5_contract() -> None:
    assert source_repair._FIXED_FRONTIER_RECOVERY_BINDING == {
        "path": "source-repair-frontier-recovery.json",
        "sha256": "705367a220455b3577b103e3277e0692aa99eaa47ada24e5fcd1e0dc155d732f",
        "size_bytes": 30690,
    }
    assert source_repair._FIXED_FRONTIER_RECOVERY_EXECUTION_IDENTITY_SHA256 == (
        "77cd6e0423b4f3942877fffe20176fbed842e404e0cf5798333b2bfe68e083e3"
    )


def test_ordinal6_context_exposes_sensor_cache_identity_fields() -> None:
    fields = set(source_repair.Stage6SourceRepairContext.__dataclass_fields__)
    assert {
        "sensor_acceleration_sha256",
        "sensor_acceleration_size_bytes",
        "frontier_recovery_execution_identity",
        "frontier_recovery_immutable_bindings",
        "frontier_recovery_verified_review_authorization",
        "coverage_cache_manifest_path",
        "coverage_cache_manifest_sha256",
        "coverage_cache_manifest_size_bytes",
        "coverage_cache_entry_set_sha256",
    } <= fields


def test_creation_script_reports_ordinal6_preview_and_update50_cutover(
    tmp_path: Path,
) -> None:
    from scripts import create_ppo_stage6_source_repair_amendment as script

    result = script._publication_result(
        stage_root=tmp_path / "s6",
        context=SimpleNamespace(
            amendment_sha256="a" * 64,
            amendment_size_bytes=100,
            continuation_sha256="b" * 64,
            continuation_size_bytes=200,
            supplement_sha256="c" * 64,
            supplement_size_bytes=300,
            closure_sha256="d" * 64,
            closure_size_bytes=400,
            frontier_recovery_sha256="e" * 64,
            frontier_recovery_size_bytes=500,
            sensor_acceleration_sha256="f" * 64,
            sensor_acceleration_size_bytes=600,
            coverage_cache_manifest_path=(
                "D:/xunce/review/s6-coverable-cache-r1-20260722T144932/"
                "coverage-cache-manifest.json"
            ),
            coverage_cache_manifest_sha256="1" * 64,
            coverage_cache_manifest_size_bytes=519284,
            coverage_cache_entry_set_sha256="2" * 64,
        ),
    )

    assert result["sensor_acceleration_sha256"] == "f" * 64
    assert result["sensor_acceleration_size_bytes"] == 600
    assert result["coverage_cache_manifest_sha256"] == "1" * 64
    assert result["coverage_cache_manifest_size_bytes"] == 519284
    assert result["coverage_cache_entry_set_sha256"] == "2" * 64
    assert result["repair_ordinal"] == 6
    assert result["accepted_prefix_updates"] == 49
    assert result["failed_update"] == 50
    assert result["next_resource_attempt"] == 2
    assert result["next_segment_index"] == 7
    normalized_help = " ".join(script.build_parser().format_help().split()).lower()
    assert "ordinal6 sensor acceleration and exact coverable cache" in normalized_help
    assert "default mode is preview-only" in normalized_help
    assert script.build_parser().parse_args(
        ["--review-authorization", "authorization.json"]
    ).publish is False


def test_creation_cli_builds_full_ordinal6_current_immutable_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import create_ppo_stage6_source_repair_amendment as script

    identity = {
        "config_sha256": "1" * 64,
        "source_set_sha256": "2" * 64,
        "prospective_tree_sha256": "3" * 64,
        "data_sha256": "4" * 64,
        "environment_identity": {"schema_version": "test/v1"},
        "environment_sha256": "5" * 64,
    }
    monkeypatch.setattr(
        script,
        "_review_authorization_immutable_bindings",
        lambda value: {"review_binding": value["review_binding"]},
    )

    result = script._current_immutable_bindings(
        identity=identity,
        review={"review_binding": "bound"},
        stage5_gate_sha256="6" * 64,
    )

    assert result == {
        **identity,
        "stage5_gate_sha256": "6" * 64,
        "review_binding": "bound",
        "coverage_cache_manifest_path": (
            "D:/xunce/review/s6-coverable-cache-r1-20260722T144932/"
            "coverage-cache-manifest.json"
        ),
        "coverage_cache_manifest_sha256": (
            "675b923b64cee4cde90d5728550a7f939c41cfbc56e9c237268078b59641081c"
        ),
        "coverage_cache_manifest_size_bytes": 519284,
        "coverage_cache_root": (
            "D:/xunce/cache/ppo_frontier/"
            "s6-standard-single-r1-20260718T220434Z/coverage-v1"
        ),
        "coverage_cache_entry_set_sha256": (
            "c543a277154b41d14bd1a194d6c6e4f2b43075d5b93096c50d1952ee44c43558"
        ),
        "coverage_cache_runtime_mode": "persistent_exact_manifest_read_only/v1",
        "coverage_cache_formal_audit_sha256": (
            "d7c12737da4c9a43353e02347b9c9d6abd527a78f7a3c6675e9164b6ae5b0068"
        ),
        "coverage_cache_formal_audit_size_bytes": 4416,
    }


def test_sensor_acceleration_preview_emits_full_v6_summary_and_lineage(
    repair_fixture: dict[str, object],
) -> None:
    frontier_recovery = _prepare_sensor_acceleration_cutover(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    resource_rows = [
        json.loads(line)
        for line in (stage / "resource_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert resource_rows[-1]["segment_id"] == SENSOR_ACCELERATION_SEGMENT_ID
    ordinal6_path = stage / source_repair.SOURCE_REPAIR_SENSOR_ACCELERATION_NAME
    parent_path = stage / source_repair.SOURCE_REPAIR_FRONTIER_RECOVERY_NAME
    parent_payload = parent_path.read_bytes()

    preview = source_repair.create_stage6_source_repair_amendment(
        stage_root=stage,
        current_execution_identity=repair_fixture["sensor_acceleration_identity"],
        current_verified_review_authorization=repair_fixture[
            "sensor_acceleration_authorization"
        ],
        current_immutable_bindings=repair_fixture["sensor_acceleration_immutable"],
        publish=False,
    )

    assert not ordinal6_path.exists()
    assert parent_path.read_bytes() == parent_payload
    assert preview.sensor_acceleration_sha256 is not None
    assert preview.sensor_acceleration_size_bytes is not None
    assert preview.coverage_cache_manifest_path == (
        repair_fixture["coverage_cache_manifest_path"].as_posix()
    )
    assert preview.coverage_cache_manifest_sha256 == _sha(
        repair_fixture["coverage_cache_manifest_payload"]
    )
    assert preview.coverage_cache_manifest_size_bytes == len(
        repair_fixture["coverage_cache_manifest_payload"]
    )
    assert preview.coverage_cache_entry_set_sha256 == repair_fixture[
        "coverage_cache_manifest_value"
    ]["entry_set_sha256"]
    assert preview.protected_checkpoint_updates == (9, 32, 46, 49)

    assert preview.checkpoint_lineage_for_update(49) == (
        frontier_recovery.checkpoint_lineage_for_update(49)
    )
    for update in (50, 100):
        lineage = preview.checkpoint_lineage_for_update(update)
        assert lineage["source_repair_sensor_acceleration_ordinal"] == 6
        assert lineage["source_repair_sensor_acceleration_sha256"] == (
            preview.sensor_acceleration_sha256
        )
        assert lineage["source_repair_sensor_acceleration_last_parent_update"] == 49
        assert lineage["source_repair_sensor_acceleration_first_repaired_update"] == 50
        assert lineage["coverage_cache_manifest_sha256"] == (
            preview.coverage_cache_manifest_sha256
        )
        assert lineage["coverage_cache_entry_set_sha256"] == (
            preview.coverage_cache_entry_set_sha256
        )
        assert lineage["coverage_cache_runtime_mode"] == (
            "persistent_exact_manifest_read_only/v1"
        )

    summary = preview.summary_binding
    assert summary["schema_version"] == "stage6_source_repair_summary/v6"
    assert summary["repair_ordinal"] == 6
    assert summary["sensor_acceleration_id"] == (
        "stage6a_sensor_hotpath_exact_coverable_cache_update50/v1"
    )
    assert summary["sensor_acceleration_sha256"] == (
        preview.sensor_acceleration_sha256
    )
    assert summary["last_frontier_recovery_update"] == 49
    assert summary["first_sensor_acceleration_update"] == 50
    assert summary["frontier_recovery_execution_identity_sha256"] == _sha(
        _canonical(repair_fixture["frontier_recovery_identity"])
    )
    assert summary["frontier_recovery_source_set_sha256"] == repair_fixture[
        "frontier_recovery_identity"
    ]["source_set_sha256"]
    assert summary["coverage_cache_manifest"] == {
        "path": repair_fixture["coverage_cache_manifest_path"].as_posix(),
        "sha256": preview.coverage_cache_manifest_sha256,
        "size_bytes": preview.coverage_cache_manifest_size_bytes,
        "cache_root": repair_fixture["sensor_acceleration_immutable"][
            "coverage_cache_root"
        ],
        "entry_set_sha256": preview.coverage_cache_entry_set_sha256,
        "entry_count": 1064,
        "split_counts": {
            "train": 700,
            "validation": 150,
            "test": 150,
            "unseen": 64,
        },
        "runtime_mode": "persistent_exact_manifest_read_only/v1",
    }
    assert summary["ordinal_artifacts"][-1] == {
        "repair_ordinal": 6,
        "path": "source-repair-sensor-acceleration.json",
        "sha256": preview.sensor_acceleration_sha256,
    }


def test_sensor_acceleration_preview_accepts_fixed_historical_frontier_sources(
    repair_fixture: dict[str, object],
) -> None:
    _prepare_sensor_acceleration_cutover(repair_fixture)
    stage = repair_fixture["stage"]
    source_root = repair_fixture["frontier_recovery_source_root"]
    assert isinstance(stage, Path)
    assert isinstance(source_root, Path)

    historical_source = source_root / "src/lunar_exploration_ppo/env/env.py"
    current_payload = b"legitimate ordinal6 source evolution\n"
    historical_source.write_bytes(current_payload)
    current_source_bindings = copy.deepcopy(
        repair_fixture["sensor_acceleration_source_bindings"]
    )
    current_source_bindings["src/lunar_exploration_ppo/env/env.py"] = {
        "path": "src/lunar_exploration_ppo/env/env.py",
        "sha256": _sha(current_payload),
        "size_bytes": len(current_payload),
    }
    monkeypatch = repair_fixture["monkeypatch"]
    assert isinstance(monkeypatch, pytest.MonkeyPatch)
    monkeypatch.setattr(
        source_repair,
        "_SENSOR_ACCELERATION_SOURCE_BINDINGS",
        current_source_bindings,
    )

    preview = source_repair.create_stage6_source_repair_amendment(
        stage_root=stage,
        current_execution_identity=repair_fixture["sensor_acceleration_identity"],
        current_verified_review_authorization=repair_fixture[
            "sensor_acceleration_authorization"
        ],
        current_immutable_bindings=repair_fixture["sensor_acceleration_immutable"],
        publish=False,
    )

    assert preview.sensor_acceleration_sha256 is not None
    assert not (
        stage / source_repair.SOURCE_REPAIR_SENSOR_ACCELERATION_NAME
    ).exists()


@pytest.mark.parametrize(
    "tamper_kind",
    (
        "attempt_id_mismatch",
        "existing_segment_record_id_mismatch",
        "segment_start_id_mismatch",
        "segment_start_missing",
        "segment_start_duplicate",
        "empty_segment_id",
        "forged_index_id_pair",
    ),
)
def test_sensor_acceleration_prefix_requires_unique_opaque_segment_binding(
    repair_fixture: dict[str, object],
    tamper_kind: str,
) -> None:
    _prepare_sensor_acceleration_cutover(repair_fixture)
    stage = repair_fixture["stage"]
    prefix = repair_fixture["sensor_acceleration_prefix"]
    assert isinstance(stage, Path)
    assert isinstance(prefix, dict)
    resource_path = stage / "resource_audit.jsonl"
    rows = [
        json.loads(line)
        for line in resource_path.read_text(encoding="utf-8").splitlines()
    ]
    segment_start_indexes = [
        index
        for index, row in enumerate(rows)
        if row.get("phase") == "segment_start" and row.get("segment_index") == 6
    ]
    assert len(segment_start_indexes) == 1
    segment_start_index = segment_start_indexes[0]

    if tamper_kind == "attempt_id_mismatch":
        rows[-1]["segment_id"] = "attempt-drift"
    elif tamper_kind == "existing_segment_record_id_mismatch":
        existing = next(
            row
            for row in rows
            if row.get("segment_index") == 6
            and row.get("phase") == "post"
            and row.get("update") == 48
        )
        existing["segment_id"] = "existing-record-drift"
    elif tamper_kind == "segment_start_id_mismatch":
        rows[segment_start_index]["segment_id"] = "segment-start-drift"
    elif tamper_kind == "segment_start_missing":
        rows.pop(segment_start_index)
    elif tamper_kind == "segment_start_duplicate":
        rows.insert(segment_start_index + 1, copy.deepcopy(rows[segment_start_index]))
    elif tamper_kind == "empty_segment_id":
        for row in rows:
            if row.get("segment_index") == 6:
                row["segment_id"] = ""
    else:
        rows[-1]["segment_index"] = 5

    payload = _write_jsonl(resource_path, rows)
    prefix["resource_audit.jsonl"] = {
        "path": "resource_audit.jsonl",
        "sha256": _sha(payload),
        "size_bytes": len(payload),
    }

    with pytest.raises(
        source_repair.Stage6SourceRepairError,
        match="accepted Update 1-49 sensor acceleration prefix drifted",
    ):
        source_repair.create_stage6_source_repair_amendment(
            stage_root=stage,
            current_execution_identity=repair_fixture[
                "sensor_acceleration_identity"
            ],
            current_verified_review_authorization=repair_fixture[
                "sensor_acceleration_authorization"
            ],
            current_immutable_bindings=repair_fixture[
                "sensor_acceleration_immutable"
            ],
            publish=False,
        )


def test_sensor_acceleration_creation_defaults_to_preview(
    repair_fixture: dict[str, object],
) -> None:
    _prepare_sensor_acceleration_cutover(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    ordinal6_path = stage / source_repair.SOURCE_REPAIR_SENSOR_ACCELERATION_NAME

    preview = source_repair.create_stage6_source_repair_amendment(
        stage_root=stage,
        current_execution_identity=repair_fixture["sensor_acceleration_identity"],
        current_verified_review_authorization=repair_fixture[
            "sensor_acceleration_authorization"
        ],
        current_immutable_bindings=repair_fixture["sensor_acceleration_immutable"],
    )

    assert preview.sensor_acceleration_sha256 is not None
    assert not ordinal6_path.exists()


def test_sensor_acceleration_repeat_publish_fails_closed(
    repair_fixture: dict[str, object],
) -> None:
    _prepare_sensor_acceleration_cutover(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    arguments = {
        "stage_root": stage,
        "current_execution_identity": repair_fixture[
            "sensor_acceleration_identity"
        ],
        "current_verified_review_authorization": repair_fixture[
            "sensor_acceleration_authorization"
        ],
        "current_immutable_bindings": repair_fixture[
            "sensor_acceleration_immutable"
        ],
        "publish": True,
    }

    published = source_repair.create_stage6_source_repair_amendment(**arguments)
    payload = (
        stage / source_repair.SOURCE_REPAIR_SENSOR_ACCELERATION_NAME
    ).read_bytes()
    with pytest.raises(
        source_repair.Stage6SourceRepairError,
        match="already published",
    ):
        source_repair.create_stage6_source_repair_amendment(**arguments)

    assert published.sensor_acceleration_sha256 == _sha(payload)
    assert (
        stage / source_repair.SOURCE_REPAIR_SENSOR_ACCELERATION_NAME
    ).read_bytes() == payload


@pytest.mark.parametrize(
    "tamper_kind",
    (
        "manifest",
        "source",
        "review",
        "parent",
        "update49_checkpoint",
        "update49_manifest",
        "update49_complete",
        "update49_policy",
        "resource_prefix",
    ),
)
def test_sensor_acceleration_preview_rejects_bound_input_tamper(
    repair_fixture: dict[str, object],
    tamper_kind: str,
) -> None:
    _prepare_sensor_acceleration_cutover(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    if tamper_kind == "manifest":
        path = repair_fixture["coverage_cache_manifest_path"]
    elif tamper_kind == "source":
        root = repair_fixture["sensor_acceleration_source_root"]
        assert isinstance(root, Path)
        path = root / "src/lunar_exploration_ppo/env/coverage_cache.py"
    elif tamper_kind == "review":
        paths = repair_fixture["sensor_acceleration_evidence_paths"]
        assert isinstance(paths, dict)
        path = paths["cache_b2_fresh_review"]
    elif tamper_kind == "parent":
        path = stage / source_repair.SOURCE_REPAIR_FRONTIER_RECOVERY_NAME
    elif tamper_kind == "update49_checkpoint":
        path = (
            stage
            / f"checkpoints/seed-{SEED}/update-00000049/checkpoint.pt"
        )
    elif tamper_kind == "update49_manifest":
        path = stage / f"checkpoints/seed-{SEED}/update-00000049/manifest.json"
    elif tamper_kind == "update49_complete":
        path = stage / f"checkpoints/seed-{SEED}/update-00000049/complete.json"
    elif tamper_kind == "resource_prefix":
        path = stage / "resource_audit.jsonl"
    else:
        values = repair_fixture["checkpoint_payload_values"]
        payload = repair_fixture["checkpoint49_payload"]
        assert isinstance(values, dict)
        assert isinstance(payload, bytes)
        values[payload]["policy_state_sha256"] = "f" * 64
        path = None
    if path is not None:
        assert isinstance(path, Path)
        path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(source_repair.Stage6SourceRepairError):
        source_repair.create_stage6_source_repair_amendment(
            stage_root=stage,
            current_execution_identity=repair_fixture[
                "sensor_acceleration_identity"
            ],
            current_verified_review_authorization=repair_fixture[
                "sensor_acceleration_authorization"
            ],
            current_immutable_bindings=repair_fixture[
                "sensor_acceleration_immutable"
            ],
            publish=False,
        )


def test_sensor_acceleration_manifest_matrix_fails_closed(
    repair_fixture: dict[str, object],
) -> None:
    _prepare_sensor_acceleration_cutover(repair_fixture)
    manifest = repair_fixture["coverage_cache_manifest_value"]
    assert isinstance(manifest, dict)

    malformed: list[dict[str, object]] = []
    missing = copy.deepcopy(manifest)
    missing["entries"].pop()  # type: ignore[union-attr]
    malformed.append(missing)
    extra = copy.deepcopy(manifest)
    extra["entries"].append(copy.deepcopy(extra["entries"][-1]))  # type: ignore[index,union-attr]
    malformed.append(extra)
    wrong_split = copy.deepcopy(manifest)
    wrong_split["split_counts"]["train"] = 699  # type: ignore[index]
    malformed.append(wrong_split)
    wrong_entry_set = copy.deepcopy(manifest)
    wrong_entry_set["entry_set_sha256"] = "f" * 64
    malformed.append(wrong_entry_set)
    wrong_root = copy.deepcopy(manifest)
    wrong_root["cache_root"] = "D:/wrong/cache/root"
    malformed.append(wrong_root)

    for value in malformed:
        with pytest.raises(source_repair.Stage6SourceRepairError):
            source_repair._validate_coverage_cache_manifest_payload(
                _canonical(value)
            )

    duplicate = copy.deepcopy(manifest)
    duplicate["entries"][1]["scenario_id"] = duplicate["entries"][0][  # type: ignore[index]
        "scenario_id"
    ]
    duplicate_entry_set = _sha(_canonical(duplicate["entries"]))
    duplicate["entry_set_sha256"] = duplicate_entry_set
    rebound = {
        **source_repair._FIXED_COVERAGE_CACHE_MANIFEST_BINDING,
        "entry_set_sha256": duplicate_entry_set,
    }
    monkeypatch = repair_fixture["monkeypatch"]
    assert isinstance(monkeypatch, pytest.MonkeyPatch)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            source_repair,
            "_FIXED_COVERAGE_CACHE_MANIFEST_BINDING",
            rebound,
        )
        with pytest.raises(
            source_repair.Stage6SourceRepairError,
            match="scenario set",
        ):
            source_repair._validate_coverage_cache_manifest_payload(
                _canonical(duplicate)
            )

    for field, value in (("sha256", "f" * 64), ("size_bytes", 1)):
        bindings = copy.deepcopy(
            source_repair._SENSOR_ACCELERATION_EVIDENCE_BINDINGS
        )
        bindings["coverage_cache_manifest"][field] = value
        with monkeypatch.context() as scoped:
            scoped.setattr(
                source_repair,
                "_SENSOR_ACCELERATION_EVIDENCE_BINDINGS",
                bindings,
            )
            with pytest.raises(source_repair.Stage6SourceRepairError):
                source_repair._sensor_acceleration_evidence_payloads()


@pytest.mark.parametrize("blocking_kind", ("active_lease", "update50_checkpoint"))
def test_sensor_acceleration_preview_requires_stopped_checkpoint_free_boundary(
    repair_fixture: dict[str, object],
    blocking_kind: str,
) -> None:
    _prepare_sensor_acceleration_cutover(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    lease = None
    if blocking_kind == "active_lease":
        lease = RunLease(stage.parent / ".stage6.lease")
        lease.acquire()
    else:
        checkpoint = stage / f"checkpoints/seed-{SEED}/update-00000050"
        checkpoint.mkdir(parents=True)
        (checkpoint / "checkpoint.pt").write_bytes(b"partial")
    try:
        with pytest.raises(source_repair.Stage6SourceRepairError):
            source_repair.create_stage6_source_repair_amendment(
                stage_root=stage,
                current_execution_identity=repair_fixture[
                    "sensor_acceleration_identity"
                ],
                current_verified_review_authorization=repair_fixture[
                    "sensor_acceleration_authorization"
                ],
                current_immutable_bindings=repair_fixture[
                    "sensor_acceleration_immutable"
                ],
                publish=False,
            )
    finally:
        if lease is not None:
            lease.release()


def test_sensor_acceleration_process_gate_tracks_root_and_descendant_parent_chain() -> None:
    snapshot_identity_pid = 91_000
    prior_root_pid = 91_100
    descendant_pid = 91_101

    with pytest.raises(
        source_repair.Stage6SourceRepairError,
        match="previous process tree is still alive",
    ):
        source_repair._require_sensor_acceleration_process_tree_stopped(
            prior_root_pid,
            process_parent_map_provider=lambda: {
                snapshot_identity_pid: 1,
                prior_root_pid: 1,
            },
            snapshot_identity_pid=snapshot_identity_pid,
            platform_system_provider=lambda: "Windows",
        )

    with pytest.raises(
        source_repair.Stage6SourceRepairError,
        match="previous process tree is still alive",
    ):
        source_repair._require_sensor_acceleration_process_tree_stopped(
            prior_root_pid,
            process_parent_map_provider=lambda: {
                snapshot_identity_pid: 1,
                descendant_pid: prior_root_pid,
            },
            snapshot_identity_pid=snapshot_identity_pid,
            platform_system_provider=lambda: "Windows",
        )

    source_repair._require_sensor_acceleration_process_tree_stopped(
        prior_root_pid,
        process_parent_map_provider=lambda: {
            snapshot_identity_pid: 1,
            91_200: snapshot_identity_pid,
        },
        snapshot_identity_pid=snapshot_identity_pid,
        platform_system_provider=lambda: "Windows",
    )


def test_sensor_acceleration_process_gate_rejects_linux_reparented_child() -> None:
    snapshot_identity_pid = 91_000
    prior_root_pid = 91_100
    reparented_child_pid = 91_101

    with pytest.raises(
        source_repair.Stage6SourceRepairError,
        match="identity/reparent unverifiable",
    ):
        source_repair._require_sensor_acceleration_process_tree_stopped(
            prior_root_pid,
            process_parent_map_provider=lambda: {
                snapshot_identity_pid: 1,
                reparented_child_pid: 1,
            },
            snapshot_identity_pid=snapshot_identity_pid,
            platform_system_provider=lambda: "Linux",
        )


@pytest.mark.parametrize("uncertainty_kind", ("provider_error", "identity_missing", "cycle"))
def test_sensor_acceleration_process_gate_fails_closed_on_snapshot_uncertainty(
    uncertainty_kind: str,
) -> None:
    snapshot_identity_pid = 92_000
    if uncertainty_kind == "provider_error":
        def provider() -> dict[int, int]:
            raise PermissionError("process snapshot denied")
    elif uncertainty_kind == "identity_missing":
        provider = lambda: {92_100: 1}
    else:
        provider = lambda: {
            snapshot_identity_pid: 1,
            92_100: 92_101,
            92_101: 92_100,
        }

    with pytest.raises(
        source_repair.Stage6SourceRepairError,
        match="process snapshot is uncertain",
    ):
        source_repair._require_sensor_acceleration_process_tree_stopped(
            92_200,
            process_parent_map_provider=provider,
            snapshot_identity_pid=snapshot_identity_pid,
        )


@pytest.mark.parametrize(
    "tamper_kind",
    ("segment_start_root_missing", "attempt_root_mismatch"),
)
def test_sensor_acceleration_prefix_cross_validates_recorded_root_pid(
    repair_fixture: dict[str, object],
    tamper_kind: str,
) -> None:
    _prepare_sensor_acceleration_cutover(repair_fixture)
    stage = repair_fixture["stage"]
    prefix = repair_fixture["sensor_acceleration_prefix"]
    assert isinstance(stage, Path)
    assert isinstance(prefix, dict)
    resource_path = stage / "resource_audit.jsonl"
    rows = [
        json.loads(line)
        for line in resource_path.read_text(encoding="utf-8").splitlines()
    ]
    if tamper_kind == "segment_start_root_missing":
        segment_start = next(
            row
            for row in rows
            if row.get("phase") == "segment_start" and row.get("segment_index") == 6
        )
        segment_start.pop("root_pid")
    else:
        rows[-1]["resource"]["rss_root_pid"] = SENSOR_ACCELERATION_ROOT_PID + 1
    payload = _write_jsonl(resource_path, rows)
    prefix["resource_audit.jsonl"] = {
        "path": "resource_audit.jsonl",
        "sha256": _sha(payload),
        "size_bytes": len(payload),
    }

    with pytest.raises(
        source_repair.Stage6SourceRepairError,
        match="accepted Update 1-49 sensor acceleration prefix drifted",
    ):
        source_repair.create_stage6_source_repair_amendment(
            stage_root=stage,
            current_execution_identity=repair_fixture[
                "sensor_acceleration_identity"
            ],
            current_verified_review_authorization=repair_fixture[
                "sensor_acceleration_authorization"
            ],
            current_immutable_bindings=repair_fixture[
                "sensor_acceleration_immutable"
            ],
            publish=False,
        )


def test_sensor_acceleration_preview_rejects_live_root_from_frozen_prefix(
    repair_fixture: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_sensor_acceleration_cutover(repair_fixture)
    monkeypatch.setattr(source_repair.platform, "system", lambda: "Windows")
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,time; print(os.getpid(), flush=True); time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert int(child.stdout.readline().strip()) == child.pid
        monkeypatch.setattr(
            source_repair,
            "_sensor_acceleration_process_parent_map",
            lambda: {
                **{os.getpid(): 1},
                **({child.pid: 1} if child.poll() is None else {}),
            },
        )
        _rebind_sensor_acceleration_root_pid(repair_fixture, child.pid)
        with pytest.raises(
            source_repair.Stage6SourceRepairError,
            match="previous process tree is still alive",
        ):
            source_repair.create_stage6_source_repair_amendment(
                stage_root=stage,
                current_execution_identity=repair_fixture[
                    "sensor_acceleration_identity"
                ],
                current_verified_review_authorization=repair_fixture[
                    "sensor_acceleration_authorization"
                ],
                current_immutable_bindings=repair_fixture[
                    "sensor_acceleration_immutable"
                ],
                publish=False,
            )
    finally:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)

    preview = source_repair.create_stage6_source_repair_amendment(
        stage_root=stage,
        current_execution_identity=repair_fixture["sensor_acceleration_identity"],
        current_verified_review_authorization=repair_fixture[
            "sensor_acceleration_authorization"
        ],
        current_immutable_bindings=repair_fixture["sensor_acceleration_immutable"],
        publish=False,
    )
    assert preview.sensor_acceleration_sha256 is not None


def test_sensor_acceleration_publish_rechecks_process_tree_inside_lease(
    repair_fixture: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_sensor_acceleration_cutover(repair_fixture)
    monkeypatch.setattr(source_repair.platform, "system", lambda: "Windows")
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    snapshots = iter(
        (
            {os.getpid(): 1},
            {
                os.getpid(): 1,
                SENSOR_ACCELERATION_ROOT_PID + 1: SENSOR_ACCELERATION_ROOT_PID,
            },
        )
    )
    snapshot_count = 0

    def snapshot() -> dict[int, int]:
        nonlocal snapshot_count
        snapshot_count += 1
        return next(snapshots)

    monkeypatch.setattr(
        source_repair,
        "_sensor_acceleration_process_parent_map",
        snapshot,
        raising=False,
    )

    with pytest.raises(
        source_repair.Stage6SourceRepairError,
        match="previous process tree is still alive",
    ):
        source_repair.create_stage6_source_repair_amendment(
            stage_root=stage,
            current_execution_identity=repair_fixture[
                "sensor_acceleration_identity"
            ],
            current_verified_review_authorization=repair_fixture[
                "sensor_acceleration_authorization"
            ],
            current_immutable_bindings=repair_fixture[
                "sensor_acceleration_immutable"
            ],
            publish=True,
        )

    assert snapshot_count == 2
    assert not (
        stage / source_repair.SOURCE_REPAIR_SENSOR_ACCELERATION_NAME
    ).exists()


def test_sensor_acceleration_preview_fails_closed_for_linux_reparented_root(
    repair_fixture: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_sensor_acceleration_cutover(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    monkeypatch.setattr(source_repair.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        source_repair,
        "_sensor_acceleration_process_parent_map",
        lambda: {
            os.getpid(): 1,
            SENSOR_ACCELERATION_ROOT_PID + 1: 1,
        },
    )
    target = stage / source_repair.SOURCE_REPAIR_SENSOR_ACCELERATION_NAME

    with pytest.raises(
        source_repair.Stage6SourceRepairError,
        match="identity/reparent unverifiable",
    ):
        source_repair.create_stage6_source_repair_amendment(
            stage_root=stage,
            current_execution_identity=repair_fixture[
                "sensor_acceleration_identity"
            ],
            current_verified_review_authorization=repair_fixture[
                "sensor_acceleration_authorization"
            ],
            current_immutable_bindings=repair_fixture[
                "sensor_acceleration_immutable"
            ],
            publish=False,
        )

    assert not target.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("transaction_key", "0049:20260716:update:051"),
        ("attempt", 2),
        ("segment_index", 7),
        ("segment_id", "segment-7"),
    ),
)
def test_sensor_acceleration_rejects_coherently_rehashed_update50_attempt1_boundary(
    repair_fixture: dict[str, object],
    field: str,
    value: object,
) -> None:
    _prepare_sensor_acceleration_cutover(repair_fixture)
    stage = repair_fixture["stage"]
    prefix = repair_fixture["sensor_acceleration_prefix"]
    assert isinstance(stage, Path)
    assert isinstance(prefix, dict)
    resource_path = stage / "resource_audit.jsonl"
    rows = [
        json.loads(line)
        for line in resource_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[-1]["transaction_key"] == "0049:20260716:update:050"
    rows[-1][field] = value
    payload = _write_jsonl(resource_path, rows)
    prefix["resource_audit.jsonl"] = {
        "path": "resource_audit.jsonl",
        "sha256": _sha(payload),
        "size_bytes": len(payload),
    }

    with pytest.raises(
        source_repair.Stage6SourceRepairError,
        match="accepted Update 1-49 sensor acceleration prefix drifted",
    ):
        source_repair.create_stage6_source_repair_amendment(
            stage_root=stage,
            current_execution_identity=repair_fixture[
                "sensor_acceleration_identity"
            ],
            current_verified_review_authorization=repair_fixture[
                "sensor_acceleration_authorization"
            ],
            current_immutable_bindings=repair_fixture[
                "sensor_acceleration_immutable"
            ],
            publish=False,
        )


def test_sensor_acceleration_require_current_rejects_post_publish_tamper(
    repair_fixture: dict[str, object],
) -> None:
    _prepare_sensor_acceleration_cutover(repair_fixture)
    stage = repair_fixture["stage"]
    assert isinstance(stage, Path)
    published = source_repair.create_stage6_source_repair_amendment(
        stage_root=stage,
        current_execution_identity=repair_fixture["sensor_acceleration_identity"],
        current_verified_review_authorization=repair_fixture[
            "sensor_acceleration_authorization"
        ],
        current_immutable_bindings=repair_fixture["sensor_acceleration_immutable"],
        publish=True,
    )
    manifest = repair_fixture["coverage_cache_manifest_path"]
    assert isinstance(manifest, Path)
    manifest.write_bytes(manifest.read_bytes() + b"tamper")

    with pytest.raises(source_repair.Stage6SourceRepairError):
        published.require_current("post-publish manifest tamper")


def test_v6_source_repair_summary_is_accepted_with_exact_cache_binding(
    repair_fixture: dict[str, object],
) -> None:
    _prepare_sensor_acceleration_cutover(repair_fixture)
    context = source_repair.create_stage6_source_repair_amendment(
        stage_root=repair_fixture["stage"],
        current_execution_identity=repair_fixture["sensor_acceleration_identity"],
        current_verified_review_authorization=repair_fixture[
            "sensor_acceleration_authorization"
        ],
        current_immutable_bindings=repair_fixture["sensor_acceleration_immutable"],
        publish=False,
    )

    acceptance = _acceptance_for_source_repair(
        context.summary_binding,
        dict(context.current_immutable_bindings),
    )

    assert acceptance["summary"]["source_repair"] == context.summary_binding
    assert acceptance["summary"]["coverage_cache_manifest_sha256"] == (
        context.coverage_cache_manifest_sha256
    )
    assert acceptance["summary"]["coverage_cache_runtime_mode"] == (
        "persistent_exact_manifest_read_only/v1"
    )
    assert acceptance["summary"]["source_repair"]["coverage_cache_manifest"][
        "runtime_mode"
    ] == "persistent_exact_manifest_read_only/v1"


def test_v6_source_repair_summary_rejects_origin_only_immutable_binding(
    repair_fixture: dict[str, object],
) -> None:
    _prepare_sensor_acceleration_cutover(repair_fixture)
    context = source_repair.create_stage6_source_repair_amendment(
        stage_root=repair_fixture["stage"],
        current_execution_identity=repair_fixture["sensor_acceleration_identity"],
        current_verified_review_authorization=repair_fixture[
            "sensor_acceleration_authorization"
        ],
        current_immutable_bindings=repair_fixture["sensor_acceleration_immutable"],
        publish=False,
    )

    with pytest.raises(StandardTrainingError, match="acceptance payload drifted"):
        _acceptance_for_source_repair(
            context.summary_binding,
            dict(context.origin_immutable_bindings),
        )


def test_v6_journal_recovery_splits_historical_and_current_immutable_bindings(
    repair_fixture: dict[str, object],
    tmp_path: Path,
) -> None:
    _prepare_sensor_acceleration_cutover(repair_fixture)
    context = source_repair.create_stage6_source_repair_amendment(
        stage_root=repair_fixture["stage"],
        current_execution_identity=repair_fixture["sensor_acceleration_identity"],
        current_verified_review_authorization=repair_fixture[
            "sensor_acceleration_authorization"
        ],
        current_immutable_bindings=repair_fixture["sensor_acceleration_immutable"],
        publish=False,
    )
    transactions = build_standard_training_transactions(load_stage6_config(CONFIG))
    historical = dict(context.origin_immutable_bindings)
    current = dict(context.current_immutable_bindings)
    first_current = transactions[49]
    assert first_current.update == 50

    receipt_index = CheckpointReceiptIndex(tmp_path / "mixed/receipts.jsonl")
    checkpoints: list[StandardTransactionCheckpoint] = []
    for transaction in transactions[:50]:
        checkpoint = StandardTransactionCheckpoint(
            transaction_key=transaction.key,
            seed=transaction.seed,
            update=transaction.update,
            checkpoint_sha256=_sha(f"checkpoint-{transaction.key}".encode()),
            complete_marker_sha256=_sha(f"complete-{transaction.key}".encode()),
            policy_state_sha256=_sha(f"policy-{transaction.key}".encode()),
        )
        receipt_index.append_once(checkpoint)
        checkpoints.append(checkpoint)
    receipt_rows = receipt_index.verify()

    def journal_records(
        name: str,
        *,
        wrong_historical: bool = False,
        wrong_current: bool = False,
    ) -> tuple[dict[str, object], ...]:
        journal = Stage6StateJournal(tmp_path / name / "job-state.jsonl")
        for transaction, checkpoint in zip(
            transactions[:50], checkpoints, strict=True
        ):
            immutable = historical if transaction.sequence < 49 else current
            if wrong_historical and transaction.sequence == 48:
                immutable = {**historical, "source_set_sha256": "8" * 64}
            if wrong_current and transaction.sequence == 49:
                immutable = {
                    **current,
                    "coverage_cache_manifest_sha256": "9" * 64,
                }
            bindings = {
                **immutable,
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
            }
            for state in transaction.commit_states:
                journal.append(state, bindings)
        return journal.verify()

    valid_records = journal_records("valid")
    completed = verify_journal_checkpoint_bindings(
        valid_records,
        transactions,
        receipt_rows,
        current,
        historical_immutable_bindings=historical,
        current_binding_first_transaction_key=first_current.key,
    )
    assert completed == tuple(transaction.key for transaction in transactions[:50])

    journal = Stage6StateJournal(tmp_path / "resume/job-state.jsonl")
    _write_jsonl(journal.path, list(valid_records))
    remaining, appended = reconcile_checkpointed_resume(
        transactions=transactions,
        journal=journal,
        checkpoint=checkpoints[-1],
        bindings={
            **current,
            "checkpoint_sha256": checkpoints[-1].checkpoint_sha256,
        },
        receipt_rows=receipt_rows,
        historical_immutable_bindings=historical,
        current_binding_first_transaction_key=first_current.key,
    )
    assert appended == ()
    assert remaining[0] == transactions[50]

    for records in (
        journal_records("wrong-historical", wrong_historical=True),
        journal_records("wrong-current", wrong_current=True),
    ):
        with pytest.raises(
            StandardTrainingError,
            match="transaction journal immutable binding drifted",
        ):
            verify_journal_checkpoint_bindings(
                records,
                transactions,
                receipt_rows,
                current,
                historical_immutable_bindings=historical,
                current_binding_first_transaction_key=first_current.key,
            )


def test_v6_backend_phase_journal_keeps_only_original_preflight_historical(
    repair_fixture: dict[str, object],
    tmp_path: Path,
) -> None:
    _prepare_sensor_acceleration_cutover(repair_fixture)
    context = source_repair.create_stage6_source_repair_amendment(
        stage_root=repair_fixture["stage"],
        current_execution_identity=repair_fixture["sensor_acceleration_identity"],
        current_verified_review_authorization=repair_fixture[
            "sensor_acceleration_authorization"
        ],
        current_immutable_bindings=repair_fixture["sensor_acceleration_immutable"],
        publish=False,
    )
    backend = object.__new__(StandardProductionBackend)
    backend._execution_operation = lambda _label: nullcontext()
    backend._immutable_bindings = dict(context.current_immutable_bindings)
    backend._source_repair = context
    backend.phase_journal = Stage6StateJournal(tmp_path / "phase-state.jsonl")

    backend._append_phase_state("preflight", checkpoint_sha256="7" * 64)
    backend._append_phase_state("global_best_frozen", checkpoint_sha256="8" * 64)

    rows = backend.phase_journal.verify()
    assert rows[0]["bindings"] == {
        **dict(context.origin_immutable_bindings),
        "checkpoint_sha256": "7" * 64,
    }
    assert rows[1]["bindings"] == {
        **dict(context.current_immutable_bindings),
        "checkpoint_sha256": "8" * 64,
    }


def test_v6_machine_verifier_rejects_full_current_binding_on_original_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.configs import stage6 as stage6_config
    from test_stage6_terminal_recovery import (
        ABSENT_HASH,
        PRETERMINAL_STATES,
        _canonical_row,
        _phase_row,
    )

    environment = _environment_identity()
    reviewed_tree = "0123456789abcdef0123456789abcdef01234567"
    origin_immutable = {
        "config_sha256": "1" * 64,
        "source_set_sha256": "2" * 64,
        "prospective_tree_sha256": _sha(reviewed_tree.encode("ascii")),
        "data_sha256": "4" * 64,
        "environment_identity": environment,
        "environment_sha256": _sha(_canonical(environment)),
        "stage5_gate_sha256": "5" * 64,
        "formal_run_id": RUN_ID,
        "changed_path_set_sha256": "a" * 64,
        "review_authorization_record_sha256": "b" * 64,
        "authorization_file_sha256": "c" * 64,
        "review_identity_sha256": "d" * 64,
        "reviewed_prospective_git_tree": reviewed_tree,
        "frozen_diff_sha256": "e" * 64,
        "spec_review_sha256": "f" * 64,
        "quality_review_sha256": "0" * 64,
    }
    origin_identity = {
        key: origin_immutable[key]
        for key in (
            "config_sha256",
            "source_set_sha256",
            "prospective_tree_sha256",
            "data_sha256",
            "environment_identity",
            "environment_sha256",
        )
    }
    origin_authorization = {"authorization": "origin"}
    stage5_authority = SimpleNamespace(
        gate_sha256=origin_immutable["stage5_gate_sha256"],
        commit_sha256="7" * 40,
        commit_tree="8" * 40,
        review_sha256="9" * 64,
        manifest_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        policy_state_sha256="c" * 64,
    )
    current_immutable = {
        **origin_immutable,
        "coverage_cache_manifest_path": str(
            stage6_workflow.STAGE6_COVERAGE_CACHE_MANIFEST_PATH
        ).replace("\\", "/"),
        "coverage_cache_manifest_sha256": (
            stage6_workflow.STAGE6_COVERAGE_CACHE_MANIFEST_SHA256
        ),
        "coverage_cache_manifest_size_bytes": (
            stage6_workflow.STAGE6_COVERAGE_CACHE_MANIFEST_SIZE_BYTES
        ),
        "coverage_cache_root": str(stage6_workflow.STAGE6_COVERAGE_CACHE_ROOT).replace(
            "\\", "/"
        ),
        "coverage_cache_entry_set_sha256": (
            stage6_workflow.STAGE6_COVERAGE_CACHE_ENTRY_SET_SHA256
        ),
        "coverage_cache_runtime_mode": stage6_workflow.STAGE6_COVERAGE_CACHE_RUNTIME_MODE,
        "coverage_cache_formal_audit_sha256": (
            stage6_workflow.STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SHA256
        ),
        "coverage_cache_formal_audit_size_bytes": (
            stage6_workflow.STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SIZE_BYTES
        ),
    }
    global_best = {
        "transaction_key": "update-50",
        "checkpoint_sha256": "d" * 64,
        "policy_state_sha256": "e" * 64,
    }
    receipt = dict(global_best)
    evaluation_result = {
        "episode_count": 64,
        "bootstrap_audit": {
            "metrics": {
                "success_rate_under_fixed_step_budget": {
                    "ci95_low": 0.1,
                    "ci95_high": 0.2,
                }
            }
        },
    }
    metrics = tuple(
        {
            "transaction_key": f"final:{split}:{method}",
            "kind": "final_evaluation",
            "split": split,
            "method": method,
            "result": evaluation_result,
        }
        for split in ("test", "unseen")
        for method in stage6_workflow.FINAL_EVALUATION_METHODS
    )
    aggregate = {
        "fairness_audit": {"passed": True},
        "leakage_audit": {"passed": True},
        "failure_audit": {"passed": True},
        "comparison_csv": b"comparison\n",
        "coverage_curves_csv": b"coverage\n",
    }
    phase_rows: list[dict[str, object]] = []
    previous = ABSENT_HASH
    for index, state in enumerate(PRETERMINAL_STATES):
        row = _phase_row(
            state,
            previous,
            {
                **current_immutable,
                "checkpoint_sha256": (
                    stage5_authority.checkpoint_sha256
                    if index == 0
                    else global_best["checkpoint_sha256"]
                ),
            },
        )
        phase_rows.append(row)
        previous = str(row["record_hash"])
    phase_payload = b"".join(_canonical_row(row) for row in phase_rows)

    class BoundEvidence:
        file_paths = (
            "source-repair-amendment.json",
            "source-repair-continuation.json",
            "source-repair-supplement.json",
            "source-repair-closure.json",
            "source-repair-frontier-recovery.json",
            "source-repair-sensor-acceleration.json",
        )

        def read_bytes(self, relative: str, *, label: str) -> bytes:
            del label
            if relative in {"phase-state.jsonl", "job-state.jsonl"}:
                return phase_payload
            if relative == "standard_baseline_comparison.csv":
                return aggregate["comparison_csv"]
            if relative == "standard_coverage_curves.csv":
                return aggregate["coverage_curves_csv"]
            return b"{}"

        def directory_snapshot(self) -> list[object]:
            return []

    def strict_json(_payload: bytes, label: str) -> dict[str, object]:
        if label == "Stage 6 global best":
            return global_best
        if label == "Stage 6 checkpoint manifest":
            return {
                "schema_version": "stage6_checkpoint_manifest/v1",
                "receipt_count": 1,
                "receipt_index": {
                    "path": "checkpoints/index.jsonl",
                    "sha256": _sha(b"{}"),
                    "size_bytes": 2,
                },
                "global_best": global_best,
                "receipts": [receipt],
            }
        if label == "Stage 6 lineage audit":
            return {
                "schema_version": "stage6_lineage_audit/v2",
                "execution_identity": origin_identity,
                "immutable_bindings": origin_immutable,
                "verified_review_authorization": origin_authorization,
            }
        if label == "Stage 6 scenario split audit":
            return {
                "schema_version": "stage6_scenario_split_audit/v1",
                "catalog_sha256": "f" * 64,
                "spatial_audit": {"parent_cross_split_count": 0},
            }
        if label == "Stage 6 fairness audit":
            return {"eval_only_isolation": {}}
        if label in {
            "Stage 6 fairness_audit.json",
            "Stage 6 leakage_audit.json",
            "Stage 6 failure_audit.json",
        }:
            return aggregate[label.removeprefix("Stage 6 ").removesuffix(".json")]
        raise AssertionError(f"unexpected strict JSON read: {label}")

    monkeypatch.setattr(stage6_workflow, "_strict_canonical_json", strict_json)
    monkeypatch.setattr(
        stage6_workflow,
        "verify_frozen_stage5_authority",
        lambda **_kwargs: SimpleNamespace(
            identity={
                "gate_sha256": stage5_authority.gate_sha256,
                "commit_sha256": stage5_authority.commit_sha256,
                "commit_tree": stage5_authority.commit_tree,
                "review_sha256": stage5_authority.review_sha256,
                "manifest_sha256": stage5_authority.manifest_sha256,
                "checkpoint_sha256": stage5_authority.checkpoint_sha256,
                "policy_state_sha256": stage5_authority.policy_state_sha256,
                "performance_advantage_established": False,
            }
        ),
    )
    monkeypatch.setattr(
        stage6_workflow,
        "stage6_execution_identity",
        lambda **_kwargs: {**origin_identity, "catalog_sha256": "f" * 64},
    )
    monkeypatch.setattr(
        stage6_workflow,
        "_load_stage6_source_repair_for_current_identity",
        lambda **_kwargs: SimpleNamespace(
            origin_execution_identity=origin_identity,
            origin_immutable_bindings=origin_immutable,
            origin_verified_review_authorization=origin_authorization,
            current_verified_review_authorization={"authorization": "current"},
            current_immutable_bindings=current_immutable,
            sensor_acceleration_sha256="f" * 64,
        ),
    )
    monkeypatch.setattr(
        stage6_config,
        "parse_stage6_config_bytes",
        lambda _payload: SimpleNamespace(
            stage5_authority=stage5_authority,
            evaluation=SimpleNamespace(bootstrap_resamples=1, bootstrap_seed=1),
        ),
    )
    monkeypatch.setattr(
        stage6_config,
        "SafetyContract",
        SimpleNamespace(from_stage6_config=lambda _config: object()),
    )
    monkeypatch.setattr(
        standard_training_module,
        "_review_authorization_immutable_bindings",
        lambda _authorization: {
            key: origin_immutable[key]
            for key in (
                "formal_run_id",
                "changed_path_set_sha256",
                "review_authorization_record_sha256",
                "authorization_file_sha256",
                "review_identity_sha256",
                "reviewed_prospective_git_tree",
                "frozen_diff_sha256",
                "spec_review_sha256",
                "quality_review_sha256",
            )
        },
    )
    monkeypatch.setattr(
        standard_training_module,
        "CheckpointReceiptIndex",
        SimpleNamespace(verify_snapshot_bytes=lambda _payload: (receipt,)),
    )
    monkeypatch.setattr(
        standard_training_module,
        "build_standard_training_transactions",
        lambda _config: (),
    )
    monkeypatch.setattr(
        stage6_workflow,
        "_strict_canonical_jsonl",
        lambda *_args, **_kwargs: metrics,
    )
    monkeypatch.setattr(
        standard_training_module,
        "verify_standard_final_evaluation_artifacts_from_bytes",
        lambda **_kwargs: {"result": evaluation_result},
    )
    monkeypatch.setattr(
        standard_training_module,
        "build_standard_final_aggregate_artifacts",
        lambda *_args, **_kwargs: aggregate,
    )
    monkeypatch.setattr(
        stage6_workflow,
        "decide_performance_advantage",
        lambda **_kwargs: SimpleNamespace(
            established=False,
            claim="performance advantage not established",
            ppo_ci95_low=0.1,
            gain_over_cost_ci95_high=0.2,
        ),
    )
    monkeypatch.setattr(
        standard_training_module,
        "verify_journal_checkpoint_bindings",
        lambda *_args, **_kwargs: pytest.fail(
            "machine verifier accepted the malformed original preflight binding"
        ),
    )

    summary = {
        **current_immutable,
        "state": "awaiting_independent_review",
        "machine_passed": True,
        "acceptance_scope": "single_seed_system_closure/v1",
        "optional_seed_extension_blocks_next_stage": False,
        "optional_seed_extension_trigger": "explicit_user_request_only/v1",
        "final_evaluation_count": 10,
        "final_episode_count": 640,
        "performance_advantage_established": False,
        "performance_claim": "performance advantage not established",
        "ppo_ci95_low": 0.1,
        "gain_over_cost_ci95_high": 0.2,
        "cross_seed_performance_conclusion": False,
    }
    routing = {
        "state": "awaiting_independent_review",
        "route": "awaiting_independent_review",
        "machine_passed": True,
        "next_stage_after_human_approval": (
            "ppo_highres_frontier_stage7_kilometer_stress/v1"
        ),
        "performance_advantage_established": False,
    }
    stage = tmp_path / "s6"
    stage.mkdir()

    with pytest.raises(
        stage6_workflow.Stage6WorkflowError,
        match="phase binding drifted",
    ):
        stage6_workflow._verify_stage6_machine_acceptance_from_bound_graph(
            stage_root=stage,
            repo_root=ROOT,
            require_terminal=False,
            summary_override=summary,
            routing_override=routing,
            reports_override={},
            evidence_handle=BoundEvidence(),
        )


def test_v6_finalize_pending_preterminal_recovery_uses_current_cache_binding(
    repair_fixture: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval.standard import STANDARD_EVALUATION_METHODS
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery
    from test_stage6_terminal_recovery import (
        ABSENT_HASH,
        PRETERMINAL_STATES,
        _append_terminal,
        _canonical_row,
        _manifest_committer,
        _phase_row,
        _semantic_result,
        _write_preterminal_stage,
    )
    from test_stage6_workflow import _active_execution_capability

    _prepare_sensor_acceleration_cutover(repair_fixture)
    context = source_repair.create_stage6_source_repair_amendment(
        stage_root=repair_fixture["stage"],
        current_execution_identity=repair_fixture["sensor_acceleration_identity"],
        current_verified_review_authorization=repair_fixture[
            "sensor_acceleration_authorization"
        ],
        current_immutable_bindings=repair_fixture["sensor_acceleration_immutable"],
        publish=True,
    )
    historical = dict(context.origin_immutable_bindings)
    current = dict(context.current_immutable_bindings)
    run_root = tmp_path / "targeted-finalize" / "s6-standard-single-r1-20260717T010203Z"

    with _active_execution_capability(
        tmp_path / "targeted-authority",
        monkeypatch,
        run_root=run_root,
    ) as active:
        stage = run_root / "s6"
        preterminal = _write_preterminal_stage(stage)
        global_checkpoint = dict(preterminal["global_checkpoint"])
        phase_rows: list[dict[str, object]] = []
        previous = ABSENT_HASH
        for index, state in enumerate(PRETERMINAL_STATES):
            bindings = {
                **(historical if index == 0 else current),
                "checkpoint_sha256": (
                    "9" * 64
                    if index == 0
                    else global_checkpoint["checkpoint_sha256"]
                ),
            }
            row = _phase_row(state, previous, bindings)
            phase_rows.append(row)
            previous = str(row["record_hash"])
        (stage / "phase-state.jsonl").write_bytes(
            b"".join(_canonical_row(row) for row in phase_rows)
        )
        config_bytes = CONFIG.read_bytes()
        (stage / "config.json").write_bytes(config_bytes)
        _write_json(
            stage / "lineage_audit.json",
            {
                "schema_version": "stage6_lineage_audit/v2",
                "execution_identity": dict(context.origin_execution_identity),
                "immutable_bindings": historical,
                "verified_review_authorization": dict(
                    context.origin_verified_review_authorization
                ),
            },
        )
        receipt_path = stage / "checkpoints/index.jsonl"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_bytes(b"targeted receipt index\n")

        record = ValidationRecord(**global_checkpoint["record"])
        evaluations = {
            f"{split}:{method}": {
                "episode_count": 64,
                "bootstrap_audit": {
                    "metrics": {
                        "success_rate_under_fixed_step_budget": {
                            "ci95_low": 0.4,
                            "ci95_high": 0.6,
                        }
                    }
                },
            }
            for split in ("test", "unseen")
            for method in STANDARD_EVALUATION_METHODS
        }
        receipts = [
            {
                "transaction_key": (
                    global_checkpoint["transaction_key"]
                    if index == 99
                    else f"targeted:{index:03d}"
                ),
                "seed": SEED,
                "update": index + 1,
                "checkpoint_sha256": (
                    global_checkpoint["checkpoint_sha256"]
                    if index == 99
                    else _sha(f"checkpoint-{index}".encode())
                ),
                "complete_marker_sha256": _sha(f"complete-{index}".encode()),
                "policy_state_sha256": (
                    global_checkpoint["policy_state_sha256"]
                    if index == 99
                    else _sha(f"policy-{index}".encode())
                ),
            }
            for index in range(100)
        ]
        backend = object.__new__(StandardProductionBackend)
        backend._execution_capability = active["capability"]
        backend.config = active["config"]
        backend.run_root = run_root
        backend.repo_root = active["repo_root"]
        backend.stage_root = stage
        backend.stage5_authority = dict(active["stage5_handle"].identity)
        backend.config_bytes = config_bytes
        backend.config_sha256 = _sha(config_bytes)
        backend.catalog = SimpleNamespace(
            sha256="a" * 64,
            spatial_audit=lambda: {"parent_cross_split_count": 0},
        )
        backend.receipt_index = SimpleNamespace(
            path=receipt_path,
            verify=lambda: tuple(receipts),
        )
        backend.phase_journal = Stage6StateJournal(stage / "phase-state.jsonl")
        backend._global_best_identity = global_checkpoint
        backend._immutable_bindings = current
        backend._source_repair = context
        backend._planning_warm_start = None
        backend._planning_child_source_repair = None
        backend._final_eval_audits = {
            key: {"passed": True} for key in evaluations
        }
        backend._evaluation_result_record = lambda result: dict(result)
        backend._safety_contract = lambda: SimpleNamespace()

        monkeypatch.setattr(
            standard_training_module,
            "verify_standard_final_evaluation_artifacts",
            lambda **kwargs: {
                "result": evaluations[f"{kwargs['split']}:{kwargs['method']}"]
            },
        )
        monkeypatch.setattr(
            standard_training_module,
            "build_standard_final_aggregate_artifacts",
            lambda *_args, **_kwargs: {
                "fairness_audit": {"passed": True},
                "leakage_audit": {"passed": True},
                "failure_audit": {"passed": True},
                "comparison_csv": b"method,value\nppo,1\n",
                "coverage_curves_csv": b"step,value\n1,1\n",
            },
        )
        monkeypatch.setattr(
            standard_training_module,
            "append_transaction_metric_once",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            stage6_workflow,
            "decide_performance_advantage",
            lambda **_kwargs: SimpleNamespace(
                established=False,
                claim="performance advantage not established",
                ppo_ci95_low=0.4,
                gain_over_cost_ci95_high=0.6,
            ),
        )
        pending_calls: list[dict[str, object]] = []

        def targeted_pending_verifier(**kwargs: object) -> dict[str, object]:
            summary = kwargs["summary"]
            assert isinstance(summary, dict)
            for key, value in current.items():
                assert summary[key] == value
            with stage6_terminal_recovery._capture_preterminal_evidence_handle(
                stage
            ) as evidence:
                result = _semantic_result(evidence.binding)
            pending_calls.append(result)
            return result

        monkeypatch.setattr(
            stage6_workflow,
            "verify_stage6_pending_acceptance",
            targeted_pending_verifier,
        )

        pending = backend.finalize(record, evaluations)
        assert len(pending_calls) == 1
        receipt = json.loads(
            (stage / "preterminal_acceptance.json").read_text(encoding="utf-8")
        )
        assert receipt["immutable_bindings"] == current
        assert pending["summary"]["coverage_cache_manifest_sha256"] == current[
            "coverage_cache_manifest_sha256"
        ]
        _append_terminal(stage, backend._preterminal_acceptance_identity)
        recovery = stage6_terminal_recovery.recover_stage6_terminal_commit(
            stage_root=stage,
            manifest_committer=_manifest_committer,
            execution_capability=active["capability"],
        )

        assert recovery["phase_states"][-2:] == (
            "machine_passed",
            "awaiting_independent_review",
        )
        terminal_summary = json.loads(
            (stage / "summary.json").read_text(encoding="utf-8")
        )
        assert terminal_summary["coverage_cache_manifest_sha256"] == current[
            "coverage_cache_manifest_sha256"
        ]
        recovered_phase = Stage6StateJournal(stage / "phase-state.jsonl").verify()
        assert {
            key: recovered_phase[0]["bindings"][key] for key in historical
        } == historical
        assert all(
            all(row["bindings"][key] == value for key, value in current.items())
            for row in recovered_phase[1:]
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("entry_count", 1063),
        ("entry_set_sha256", "0" * 64),
        ("runtime_mode", "process_local_compute_fallback/v1"),
    ),
)
def test_v6_source_repair_acceptance_rejects_cache_summary_tamper(
    repair_fixture: dict[str, object],
    field: str,
    value: object,
) -> None:
    _prepare_sensor_acceleration_cutover(repair_fixture)
    context = source_repair.create_stage6_source_repair_amendment(
        stage_root=repair_fixture["stage"],
        current_execution_identity=repair_fixture["sensor_acceleration_identity"],
        current_verified_review_authorization=repair_fixture[
            "sensor_acceleration_authorization"
        ],
        current_immutable_bindings=repair_fixture["sensor_acceleration_immutable"],
        publish=False,
    )
    binding = copy.deepcopy(context.summary_binding)
    binding["coverage_cache_manifest"][field] = value

    with pytest.raises(StandardTrainingError):
        _acceptance_for_source_repair(
            binding,
            dict(context.current_immutable_bindings),
        )


def test_standard_backend_binds_v6_cache_into_every_training_spawn_spec(
    repair_fixture: dict[str, object],
) -> None:
    _prepare_sensor_acceleration_cutover(repair_fixture)
    context = source_repair.create_stage6_source_repair_amendment(
        stage_root=repair_fixture["stage"],
        current_execution_identity=repair_fixture["sensor_acceleration_identity"],
        current_verified_review_authorization=repair_fixture[
            "sensor_acceleration_authorization"
        ],
        current_immutable_bindings=repair_fixture["sensor_acceleration_immutable"],
        publish=False,
    )
    backend = object.__new__(StandardProductionBackend)
    backend.catalog = object()
    backend.config_sha256 = "a" * 64
    backend._source_repair = context

    def legacy_specs(*args, **kwargs):
        del args, kwargs
        return tuple(
            SimpleNamespace(factory=lambda: None, kwargs={"worker": index})
            for index in range(8)
        )

    backend._components = {"standard_env_specs": legacy_specs}
    specs = backend._production_training_env_specs(
        split="train",
        sampler_seeds=tuple(range(8)),
        safety_contract=object(),
    )

    assert len(specs) == 8
    assert all(
        spec.factory is standard_training_module._build_cached_standard_training_env
        for spec in specs
    )
    assert all(
        spec.kwargs["coverage_cache_manifest_path"]
        == context.coverage_cache_manifest_path
        and spec.kwargs["coverage_cache_manifest_sha256"]
        == context.coverage_cache_manifest_sha256
        for spec in specs
    )
    assert backend._checkpoint_lineage_for_update(49) == (
        context.checkpoint_lineage_for_update(49)
    )
    assert backend._checkpoint_lineage_for_update(50) == (
        context.checkpoint_lineage_for_update(50)
    )


def test_production_evaluation_component_binds_cache_into_every_spawn_spec() -> None:
    components = standard_training_module._default_production_components()
    assert components["run_standard_evaluation"] is (
        standard_training_module._run_cached_standard_evaluation
    )
    legacy_specs = tuple(
        SimpleNamespace(factory=lambda: None, kwargs={"lane": index})
        for index in range(8)
    )

    specs = standard_training_module._cache_bound_standard_evaluation_specs(
        legacy_specs,
        coverage_cache_manifest_path="D:/fixed/coverage-cache-manifest.json",
        coverage_cache_manifest_sha256="a" * 64,
    )

    assert len(specs) == 8
    assert all(
        spec.factory
        is standard_training_module._build_cached_standard_evaluation_env
        for spec in specs
    )
    assert all(
        spec.kwargs["coverage_cache_manifest_path"]
        == "D:/fixed/coverage-cache-manifest.json"
        and spec.kwargs["coverage_cache_manifest_sha256"] == "a" * 64
        for spec in specs
    )


def test_standard_training_current_immutable_contract_includes_cache_identity() -> None:
    assert {
        "coverage_cache_manifest_path",
        "coverage_cache_manifest_sha256",
        "coverage_cache_manifest_size_bytes",
        "coverage_cache_root",
        "coverage_cache_entry_set_sha256",
        "coverage_cache_runtime_mode",
        "coverage_cache_formal_audit_sha256",
        "coverage_cache_formal_audit_size_bytes",
    } <= standard_training_module.STAGE6_IMMUTABLE_BINDING_KEYS
