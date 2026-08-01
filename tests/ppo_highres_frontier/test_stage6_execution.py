"""Stage 6 正式执行入口、update 事务与精确 resume 合同。"""

from __future__ import annotations

import csv
import importlib
import inspect
import json
import os
import random
import gc
import subprocess
import weakref
from contextlib import ExitStack, nullcontext
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lunar_exploration_ppo.configs.stage6 import SafetyContract, load_stage6_config
from lunar_exploration_ppo.ppo import standard_training
from lunar_exploration_ppo.ppo.checkpoint import CheckpointManager
from lunar_exploration_ppo.ppo.checkpoint_retention import CheckpointRetentionManager
from lunar_exploration_ppo.ppo.collector import CollectorAudit
from lunar_exploration_ppo.ppo.trainer import (
    policy_state_sha256,
    snapshot_hash_list_sha256,
)
from lunar_exploration_ppo.utils.resources import ResourceSnapshot
from lunar_exploration_ppo.workflows.stage6 import (
    Stage6StateJournal,
    Stage6WorkflowError,
    stage6_environment_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/ppo_highres_frontier_stage6_v1.json"
FORMAL_RUN_ID = "s6-standard-single-r1-20260717T010203Z"


def _forge_task5_resource_prefix_line_count(
    fixture: SimpleNamespace,
    *,
    delta: int,
) -> None:
    from test_stage6_planning_child_source_repair import _rehash_artifact

    artifact = json.loads(
        fixture.child_path.read_text(encoding="utf-8")
    )
    binding = artifact["append_only_prefixes"]["resource_audit.jsonl"]
    binding["line_count"] = int(binding["line_count"]) + delta
    fixture.child_path.write_bytes(
        fixture.module.ArtifactStore.canonical_json_bytes(
            _rehash_artifact(artifact)
        )
    )


def test_planning_child_public_terminal_verifier_executes_production_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_stage6_workflow import (
        _build_task5_planning_child_machine_fixture,
        _finalize_task5_planning_child_terminal_fixture,
        _run_task5_planning_child_semantic_verifier,
    )

    fixture = _build_task5_planning_child_machine_fixture(
        tmp_path,
        monkeypatch,
    )
    _run_task5_planning_child_semantic_verifier(fixture)
    finalized = _finalize_task5_planning_child_terminal_fixture(fixture)

    result = fixture.module.verify_stage6_machine_acceptance(
        stage_root=fixture.stage,
        repo_root=ROOT,
        planning_child_recovery_capability=(
            fixture.recovery_capability
        ),
    )

    assert result["passed"] is True
    assert result["checkpoint_receipt_count"] == 26
    assert result["final_evaluation_count"] == 10
    assert result["final_episode_count"] == 640
    assert result == fixture.receipt["semantic_verification"]
    assert finalized["receipt_identity"] == fixture.receipt_identity
    assert fixture.critical_calls["context"] == 2
    assert fixture.acceptance["summary"][
        "parent_semantics_update_count"
    ] == 74
    assert fixture.acceptance["summary"][
        "child_new_semantics_update_count"
    ] == 26
    assert "source_repair" not in fixture.acceptance["summary"]
    assert "planning_child_recovery" in (
        fixture.acceptance["summary"]
    )


def test_planning_child_public_terminal_verifier_rejects_rehashed_inflated_prefix_line_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_stage6_workflow import (
        _build_task5_planning_child_machine_fixture,
        _finalize_task5_planning_child_terminal_fixture,
        _run_task5_planning_child_semantic_verifier,
    )

    fixture = _build_task5_planning_child_machine_fixture(
        tmp_path,
        monkeypatch,
    )
    _run_task5_planning_child_semantic_verifier(fixture)
    _finalize_task5_planning_child_terminal_fixture(fixture)
    _forge_task5_resource_prefix_line_count(fixture, delta=3)

    with pytest.raises(
        fixture.module.Stage6WorkflowError,
        match="planning child|resource|prefix|line count|binding",
    ):
        fixture.module.verify_stage6_machine_acceptance(
            stage_root=fixture.stage,
            repo_root=ROOT,
            planning_child_recovery_capability=(
                fixture.recovery_capability
            ),
        )


def test_planning_child_public_terminal_verifier_rejects_rehashed_deflated_prefix_line_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_stage6_workflow import (
        _build_task5_planning_child_machine_fixture,
        _finalize_task5_planning_child_terminal_fixture,
        _run_task5_planning_child_semantic_verifier,
    )

    fixture = _build_task5_planning_child_machine_fixture(
        tmp_path,
        monkeypatch,
    )
    _run_task5_planning_child_semantic_verifier(fixture)
    _finalize_task5_planning_child_terminal_fixture(fixture)
    _forge_task5_resource_prefix_line_count(fixture, delta=-1)

    with pytest.raises(
        fixture.module.Stage6WorkflowError,
        match="planning child|resource|prefix|line count|binding",
    ):
        fixture.module.verify_stage6_machine_acceptance(
            stage_root=fixture.stage,
            repo_root=ROOT,
            planning_child_recovery_capability=(
                fixture.recovery_capability
            ),
        )


@pytest.mark.parametrize(
    "drift",
    (
        "malformed_child",
        "noncanonical_child",
        "current_identity",
        "current_review",
        "current_immutable",
        "append_only_prefix",
        "missing_child",
        "mixed_classic_profile",
    ),
)
def test_planning_child_public_terminal_verifier_rejects_currentness_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    from test_stage6_workflow import (
        _build_task5_planning_child_machine_fixture,
        _finalize_task5_planning_child_terminal_fixture,
        _run_task5_planning_child_semantic_verifier,
    )

    fixture = _build_task5_planning_child_machine_fixture(
        tmp_path,
        monkeypatch,
    )
    _run_task5_planning_child_semantic_verifier(fixture)
    _finalize_task5_planning_child_terminal_fixture(fixture)
    if drift == "malformed_child":
        fixture.child_path.write_bytes(b"{")
    elif drift == "noncanonical_child":
        fixture.child_path.write_text(
            json.dumps(
                json.loads(fixture.child_payload.decode("utf-8")),
                ensure_ascii=False,
                indent=1,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    elif drift in {
        "current_identity",
        "current_review",
        "current_immutable",
    }:
        from test_stage6_planning_child_source_repair import (
            _rehash_artifact,
        )

        artifact = json.loads(
            fixture.child_path.read_text(encoding="utf-8")
        )
        current = artifact["current"]
        if drift == "current_identity":
            current["execution_identity"]["source_set_sha256"] = "d" * 64
            current["execution_identity_sha256"] = sha256(
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
    elif drift == "append_only_prefix":
        resource_path = fixture.stage / "resource_audit.jsonl"
        rows = [
            json.loads(line)
            for line in resource_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        rows[0]["root_pid"] = int(rows[0]["root_pid"]) + 1
        resource_path.write_bytes(
            b"".join(
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
        )
    elif drift == "missing_child":
        fixture.child_path.unlink()
    elif drift == "mixed_classic_profile":
        (fixture.stage / "source-repair-amendment.json").write_bytes(
            fixture.module.ArtifactStore.canonical_json_bytes(
                {"schema_version": "forged_classic_profile/v1"}
            )
        )
    else:
        raise AssertionError(f"unknown drift: {drift}")

    with pytest.raises(
        fixture.module.Stage6WorkflowError,
        match=(
            "evidence|manifest|planning child|changed|binding|"
            "canonical|source-repair|receipt|resource"
        ),
    ):
        fixture.module.verify_stage6_machine_acceptance(
            stage_root=fixture.stage,
            repo_root=ROOT,
            planning_child_recovery_capability=(
                fixture.recovery_capability
            ),
        )


def _test_spawn_env_factory(**kwargs: object) -> dict[str, object]:
    return dict(kwargs)


@pytest.fixture
def active_execution_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from test_stage6_workflow import _active_execution_capability

    with _active_execution_capability(
        tmp_path,
        monkeypatch,
        environment_identity=_stage6_environment_identity(),
    ) as active:
        yield active


@pytest.fixture
def execution_authority_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from test_stage6_workflow import _active_execution_capability

    with ExitStack() as stack:
        def activate(**kwargs: object) -> dict[str, object]:
            return stack.enter_context(
                _active_execution_capability(tmp_path, monkeypatch, **kwargs)
            )

        yield activate


def _bind_object_backend_capability(
    backend: standard_training.StandardProductionBackend,
    active: dict[str, object],
) -> None:
    backend._execution_capability = active["capability"]
    backend.config = active["config"]
    backend.run_root = active["run_root"]
    backend.repo_root = active["repo_root"]
    backend.stage_root = Path(active["run_root"]) / "s6"
    backend.stage5_authority = dict(active["stage5_handle"].identity)


def _attach_backend_resource_segment(
    backend: standard_training.StandardProductionBackend,
) -> dict[str, object]:
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        append_resource_segment_start,
    )

    segment = append_resource_segment_start(
        backend.stage_root / "resource_audit.jsonl",
        first_sample={
            "d_free_bytes": 200 * 1024**3,
            "rss_bytes": 1024,
            "peak_vram_bytes": 2048,
            "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
            "rss_root_pid": os.getpid(),
            "rss_sample_count": 1,
            "rss_latest_process_count": 1,
            "rss_peak_process_count": 1,
            "warnings": [],
            "hard_stops": [],
            "passed": True,
        },
    )
    backend._resource_segment_start = segment
    return segment


def _validation_trace_backend(
    active_execution_authority: dict[str, object],
) -> standard_training.StandardProductionBackend:
    backend = object.__new__(standard_training.StandardProductionBackend)
    _bind_object_backend_capability(backend, active_execution_authority)
    backend.stage_root.mkdir(parents=True, exist_ok=True)
    backend.run_root.joinpath("stage6-attempts").mkdir(parents=True, exist_ok=True)
    return backend


def _validation_trace_payload(*, attempt: int, episode_count: int = 16) -> bytes:
    return b"".join(
        (
            json.dumps(
                {"attempt": attempt, "episode_index": index},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for index in range(episode_count)
    )


def _validation_trace_binding(payload: bytes, *, line_count: int = 16) -> dict[str, object]:
    return {
        "schema_version": "stage6_validation_trace_binding/v1",
        "sha256": sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "line_count": line_count,
    }


def _append_validation_trace_pre_attempt(
    backend: standard_training.StandardProductionBackend,
    transaction: standard_training.StandardTrainingTransaction,
    *,
    attempt: int,
) -> None:
    backend._append_resource_phase(
        identity={
            "kind": "update",
            "transaction_key": transaction.key,
            "seed": transaction.seed,
            "update": transaction.update,
        },
        attempt=attempt,
        phase="pre",
        resource={
            "d_free_bytes": 200 * 1024**3,
            "rss_bytes": 1024,
            "peak_vram_bytes": 2048,
            "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
            "rss_root_pid": os.getpid(),
            "rss_sample_count": attempt,
            "rss_latest_process_count": 1,
            "rss_peak_process_count": 1,
            "warnings": [],
            "hard_stops": [],
            "passed": True,
        },
    )


def _stage6_environment_identity() -> dict[str, object]:
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


def _verified_review_authorization(
    identity: dict[str, object],
    *,
    formal_run_id: str = FORMAL_RUN_ID,
) -> dict[str, object]:
    from lunar_exploration_ppo.workflows import stage6 as stage6_workflow

    return {
        "schema_version": "stage6_verified_review_launch_authorization/v1",
        "stage_id": stage6_workflow.STAGE6_STAGE_ID,
        "formal_run_id": formal_run_id,
        "single_seed_scope": {
            "scope_kind": "single_seed_system_closure/v1",
            "seed": 20260716,
            "updates": 100,
        },
        "authorized": True,
        "base_commit": stage6_workflow.STAGE5_COMMIT,
        "authorization_file_sha256": "a" * 64,
        "authorization_file_size_bytes": 4096,
        "review_identity_sha256": sha256(
            standard_training.ArtifactStore.canonical_json_bytes(identity)
        ).hexdigest(),
        "reviewed_prospective_git_tree": identity["prospective_git_tree"],
        "prospective_tree_sha256": identity["prospective_tree_sha256"],
        "changed_path_set_sha256": identity["changed_path_set_sha256"],
        "source_set_sha256": identity["source_set_sha256"],
        "config_sha256": identity["config_sha256"],
        "data_sha256": identity["data_sha256"],
        "environment_sha256": identity["environment_sha256"],
        "frozen_diff_sha256": "c" * 64,
        "frozen_diff_size_bytes": 8192,
        "spec_review_sha256": "d" * 64,
        "quality_review_sha256": "e" * 64,
    }


def _review_immutable_bindings(
    verified: dict[str, object],
) -> dict[str, object]:
    return {
        "formal_run_id": verified["formal_run_id"],
        "changed_path_set_sha256": verified["changed_path_set_sha256"],
        "review_authorization_record_sha256": sha256(
            standard_training.ArtifactStore.canonical_json_bytes(verified)
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


def _collector_audit_dict(
    policy_sha256: str,
    *,
    prefix: str,
) -> dict[str, object]:
    return {
        "schema_version": "stage4_spawn_collector/v1",
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
            sha256(f"{prefix}-snapshot-{index}".encode("ascii")).hexdigest()
            for index in range(1024)
        ],
        "terminal_transition_count": 0,
    }


def _legacy_single_scan_collector_audit(
    policy_sha256: str,
) -> dict[str, object]:
    audit = _collector_audit_dict(policy_sha256, prefix="legacy-single-scan")
    sensor = {
        "sample_count": 1,
        "ray_count": 91,
        "cell_visit_count": 4,
        "unique_visible_cell_count": 4,
        "duplicate_cell_visits": 0,
        "sample_sources": ["reset"],
        "sample_headings": [0.0],
    }
    audit.update(
        {
            "schema_version": "stage4_spawn_collector/v2",
            "reset_diagnostics": [
                {
                    "worker_index": 0,
                    "reset_index": 0,
                    "schema_version": "stage6_reset_scan_diagnostics/v1",
                    "scan_order": ["reset_exploration"],
                    "local_safety_sensor": dict(sensor),
                    "exploration_sensor": dict(sensor),
                }
            ],
            "planner_failure_counts": {
                "endpoint_physical_unsafe": 0,
                "endpoint_unknown_buffer_unsafe": 0,
                "path_physical_unsafe": 0,
                "path_unknown_buffer_unsafe": 0,
                "planner_no_path": 0,
            },
        }
    )
    return audit


def test_warm_collection_audit_rejects_legacy_single_scan_reset() -> None:
    audit = _legacy_single_scan_collector_audit("a" * 64)
    audit["policy_device"] = "cpu"

    assert standard_training.validate_standard_collection_audit(
        audit,
        expected_device="cpu",
        require_dual_scan=False,
    ) == audit
    with pytest.raises(
        standard_training.StandardTrainingError,
        match="reset diagnostics",
    ):
        standard_training.validate_standard_collection_audit(
            audit,
            expected_device="cpu",
            require_dual_scan=True,
        )

    legacy_v1 = _collector_audit_dict(
        "a" * 64,
        prefix="legacy-v1-without-reset-diagnostics",
    )
    legacy_v1["policy_device"] = "cpu"
    with pytest.raises(
        standard_training.StandardTrainingError,
        match="reset diagnostics",
    ):
        standard_training.validate_standard_collection_audit(
            legacy_v1,
            expected_device="cpu",
            require_dual_scan=True,
        )


@pytest.mark.parametrize(
    ("expected_device", "policy_device", "accepted"),
    (
        pytest.param("cuda", "cuda:0", True, id="bare-expected-current-cuda"),
        pytest.param("cuda:0", "cuda", True, id="bare-audit-current-cuda"),
        pytest.param("cuda", "cuda:1", False, id="wrong-explicit-cuda-index"),
        pytest.param("cuda", "cuda:invalid", False, id="invalid-audit-device"),
        pytest.param("cuda:invalid", "cuda:0", False, id="invalid-expected-device"),
        pytest.param("cpu", "cpu", True, id="cpu-exact"),
        pytest.param("cpu", "cpu:0", False, id="cpu-index-is-not-exact"),
    ),
)
def test_collection_audit_parses_devices_with_exact_current_cuda_alias(
    monkeypatch: pytest.MonkeyPatch,
    expected_device: str,
    policy_device: str,
    accepted: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    policy_sha256 = "a" * 64
    audit = _collector_audit_dict(policy_sha256, prefix="parsed-device")
    audit["policy_device"] = policy_device

    if accepted:
        assert standard_training.validate_standard_collection_audit(
            audit,
            expected_device=expected_device,
        ) == audit
        return

    with pytest.raises(
        standard_training.StandardTrainingError,
        match="device",
    ):
        standard_training.validate_standard_collection_audit(
            audit,
            expected_device=expected_device,
        )


def _math_evidence_dict(
    collection_audit: dict[str, object],
    *,
    policy_sha256: str,
    optimizer_steps: int = 4,
) -> dict[str, object]:
    snapshot_hashes = list(collection_audit["snapshot_sha256"])
    sample_count = int(collection_audit["trainable_transition_count"])
    payload: dict[str, object] = {
        "schema_version": "ppo_update_math_evidence/v1",
        "sample_count": sample_count,
        "snapshot_list_sha256": snapshot_hash_list_sha256(snapshot_hashes),
        "batch_policy_state_sha256": policy_sha256,
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
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {**payload, "evidence_sha256": sha256(encoded).hexdigest()}


def test_math_audit_requires_sealed_runtime_evidence_bound_to_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    policy_sha256 = "a" * 64
    collection = _collector_audit_dict(policy_sha256, prefix="math-evidence")
    evidence = _math_evidence_dict(
        collection,
        policy_sha256=policy_sha256,
        optimizer_steps=4,
    )
    metrics = {
        "initial_ratio_max_abs_error": 2.0e-6,
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "approx_kl": 0.003,
        "grad_pre_clip_norm_max": 0.7,
        "grad_post_clip_norm_max": 0.5,
        "optimizer_steps": 4,
        "policy_state_sha256_before": policy_sha256,
        "policy_state_sha256_after": "b" * 64,
        "math_evidence": evidence,
    }

    audit = standard_training._math_audit_from_completed_update(
        metrics,
        collection_audit=collection,
        device="cuda",
        compute_dtype="float32",
    )
    assert audit["evidence_sha256"] == evidence["evidence_sha256"]
    assert audit["snapshot_list_sha256"] == evidence["snapshot_list_sha256"]
    assert audit["ppo_math_evidence"] == evidence

    with pytest.raises(
        standard_training.StandardTrainingError,
        match="math evidence",
    ):
        standard_training._math_audit_from_completed_update(
            {**metrics, "math_evidence": None},
            collection_audit=collection,
            device="cuda",
            compute_dtype="float32",
        )
    tampered = {**evidence, "loss_finite": False}
    with pytest.raises(
        standard_training.StandardTrainingError,
        match="math evidence",
    ):
        standard_training._math_audit_from_completed_update(
            {**metrics, "math_evidence": tampered},
            collection_audit=collection,
            device="cuda",
            compute_dtype="float32",
        )


def _deterministic_final_summary(
    method: str,
    split: str,
    policy_state_sha256_value: str,
):
    from lunar_exploration_ppo.env.standard_training import (
        build_standard_evaluation_env,
    )
    from lunar_exploration_ppo.eval.evaluator import EvaluationSummary
    from lunar_exploration_ppo.eval.metrics import (
        ZERO_DISTANCE_POLICY,
        build_episode_result,
        summarize_episodes,
    )
    from lunar_exploration_ppo.eval import standard
    from lunar_exploration_ppo.policy.cross_attention import (
        PolicyForwardOutput,
        batch_policy_observations,
    )
    from lunar_exploration_ppo.policy.observation import PolicyObservation
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore

    config = load_stage6_config(CONFIG)
    safety_contract = SafetyContract.from_stage6_config(config)
    config_sha256 = sha256(CONFIG.read_bytes()).hexdigest()
    safety_binding = safety_contract.binding(config_sha256=config_sha256)
    final_coverage = 1.0 if method == "ppo_policy" else 0.5
    coverage = (0.0, final_coverage, *(final_coverage for _ in range(127)))
    path = (0.0, 1.0, *(1.0 for _ in range(127)))
    seed_base = 10_000 if split == "test" else 20_000
    episodes = tuple(
        build_episode_result(
            method=method,
            scale_profile="Standard v1",
            scenario_key=f"{split}/fixture-{index:02d}",
            scenario_seed=seed_base + index,
            terrain_seed=seed_base + 100 + index,
            start_pose_seed=seed_base + 200 + index,
            evaluation_seed=seed_base + 300 + index,
            coverage_curve=coverage,
            cumulative_path_length_curve=path,
            steps_executed=1,
            invalid_action_count=0,
            planner_failure_count=0,
            safety_violation_count=0,
            termination_reason=(
                "success_done" if method == "ppo_policy" else "failure_done"
            ),
            max_steps=128,
            success_threshold=0.99,
            zero_distance_policy=ZERO_DISTANCE_POLICY,
        )
        for index in range(64)
    )
    metrics, bootstrap = summarize_episodes(
        episodes,
        bootstrap_resamples=config.evaluation.bootstrap_resamples,
        bootstrap_seed=config.evaluation.bootstrap_seed,
    )
    schedule = [
        {
            "episode_index": index,
            "scenario_id": episode.scenario_key,
            "scenario_seed": episode.scenario_seed,
            "terrain_seed": episode.terrain_seed,
            "start_pose_seed": episode.start_pose_seed,
            "evaluation_seed": episode.evaluation_seed,
        }
        for index, episode in enumerate(episodes)
    ]
    catalog_payload_sha256 = "a" * 64
    environment_contract = {
        "schema_version": "stage6_standard_shared_environment_contract/v1",
        "catalog": {
            "schema_version": "standard_unseen_scenario_catalog/v1",
            "catalog_sha256": "b" * 64,
            "catalog_payload_sha256": catalog_payload_sha256,
            "split_policy": "fixed_seed_disjoint_split/v1",
            "source_sha256": {"dem": "c" * 64, "slope": "d" * 64},
            "spatial_audit": {
                "schema_version": "catalog_spatial_leakage_audit/v1",
                "parent_cross_split_count": 0,
                "child_overlap_pair_count": 0,
            },
        },
        "split": split,
        "scenario_seed_schedule": schedule,
        "environment_specs": [
            {
                "worker_index": worker_index,
                "factory_source": (
                    f"{build_standard_evaluation_env.__module__}."
                    f"{build_standard_evaluation_env.__qualname__}"
                ),
                "catalog_payload_sha256": catalog_payload_sha256,
                "split": split,
                "scenario_ids": [
                    episode.scenario_key
                    for episode in episodes[worker_index::8]
                ],
                "production": True,
                **safety_binding,
            }
            for worker_index in range(8)
        ],
        "max_steps": 128,
        "success_threshold": 0.99,
        "environment_contract": standard._static_standard_environment_contract(
            safety_contract=safety_contract,
            config_sha256=config_sha256,
        ),
    }
    frontier_features = np.zeros((1, 22), dtype=np.float32)
    frontier_features[0, 5] = 1.0
    frontier_features[0, 15] = 1.0
    frontier_features[0, 18] = 1.0
    observation = PolicyObservation(
        prior_channels=np.zeros((7, 32, 32), dtype=np.float32),
        coverage_summary=np.zeros((8, 32, 32), dtype=np.float32),
        local_crop=np.zeros((8, 96, 96), dtype=np.float32),
        frontier_features=frontier_features,
        pose_features=np.zeros((6,), dtype=np.float32),
        candidate_mask=np.ones((1,), dtype=bool),
    )
    batch = batch_policy_observations((observation,))
    one = torch.zeros((1, 1), dtype=torch.float32)
    token = torch.zeros((1, 1, 4), dtype=torch.float32)
    policy_output = PolicyForwardOutput(
        frontier_logits=torch.ones((1, 1), dtype=torch.float32),
        theta_mu_sin_raw=one,
        theta_mu_cos_raw=one,
        theta_kappa_raw=one,
        theta_mu=one,
        theta_kappa=torch.ones_like(one),
        value=torch.zeros((1,), dtype=torch.float32),
        global_map_tokens=token,
        local_map_tokens=token,
        pose_token=token,
        context_tokens=token,
        refined_frontier_tokens=token,
        action_hidden=token,
    )
    decisions = []
    for index, episode in enumerate(episodes):
        action = standard.select_standard_evaluation_actions(
            method=method,
            observations=(observation,),
            evaluation_seeds=(episode.evaluation_seed,),
            policy_output=(policy_output if method == "ppo_policy" else None),
        )[0]
        decisions.append(
            standard.build_standard_decision_record(
                method=method,
                sequence_index=index,
                episode_index=index,
                step_index=0,
                worker_index=index % 8,
                observation=observation,
                action=action,
                selector_inputs=(
                    {
                        "observation_batch": batch,
                        "policy_output": policy_output,
                    }
                    if method == "ppo_policy"
                    else {"policy_observation": observation}
                ),
                policy_batch_row_index=(0 if method == "ppo_policy" else None),
            )
        )
    decision_schema = standard.build_standard_decision_input_schema()
    action_rule = standard.build_standard_action_rule_provenance(method)
    fairness = standard.validate_standard_fairness_audit(
        {
            "schema_version": "stage6_standard_parallel_fairness_audit/v2",
            "method": method,
            "split": split,
            "worker_count": 8,
            "worker_pids": [os.getpid() + index + 1 for index in range(8)],
            "worker_start_methods": ["spawn"] * 8,
            "evaluation_parent_pid": os.getpid(),
            "inference_pids": [os.getpid()] if method == "ppo_policy" else [],
            "policy_state_unchanged": True,
            "policy_state_sha256_before": (
                policy_state_sha256_value if method == "ppo_policy" else None
            ),
            "policy_state_sha256_after": (
                policy_state_sha256_value if method == "ppo_policy" else None
            ),
            "episode_action_counts": [1] * 64,
            "selected_action_count": 64,
            "scenario_schedule": [episode.scenario_key for episode in episodes],
            "evaluation_seeds": [episode.evaluation_seed for episode in episodes],
            "theta_source": (
                "policy_theta_mu/v1"
                if method == "ppo_policy"
                else "candidate_recommended_theta/v1"
            ),
            "shared_environment_contract": environment_contract,
            "shared_environment_contract_sha256": sha256(
                ArtifactStore.canonical_json_bytes(environment_contract)
            ).hexdigest(),
            "decision_input_schema": decision_schema,
            "decision_input_schema_sha256": sha256(
                ArtifactStore.canonical_json_bytes(decision_schema)
            ).hexdigest(),
            "action_rule": action_rule,
            "action_rule_sha256": sha256(
                ArtifactStore.canonical_json_bytes(action_rule)
            ).hexdigest(),
            "decision_audit": standard._build_standard_decision_audit(
                method,
                decisions,
            ),
        },
        episode_count=64,
        parent_pid=os.getpid(),
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )
    return EvaluationSummary(
        method=method,
        scale_profile="Standard v1",
        episodes=episodes,
        metrics=metrics,
        bootstrap_audit=bootstrap,
        fairness_audit=fairness,
    )


def test_final_evaluation_replay_rejects_rehashed_single_scenario_seed_drift(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.eval.metrics import (
        EPISODE_FIELDS,
        EpisodeResult,
        episode_record,
        summarize_episodes,
    )
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore

    config = load_stage6_config(CONFIG)
    split = "test"
    method = "ppo_policy"
    config_sha256 = sha256(CONFIG.read_bytes()).hexdigest()
    checkpoint_sha256 = "2" * 64
    policy_sha256 = "3" * 64
    evaluation = _deterministic_final_summary(method, split, policy_sha256)
    trace_path = tmp_path / "final-test-ppo_policy.jsonl"
    summary_path = trace_path.with_suffix(".summary.json")
    commit_path = trace_path.with_suffix(".commit.json")
    trace_bytes = b"".join(
        (
            json.dumps(
                {
                    **episode_record(episode),
                    "steps_executed": 1,
                    "reset_diagnostics": {
                        "schema_version": "stage6_reset_scan_diagnostics/v1",
                        "scan_order": [
                            "reset_local_safety",
                            "reset_exploration",
                        ],
                        "local_safety_sensor": {
                            "sample_count": 1,
                            "ray_count": 361,
                            "cell_visit_count": 4,
                            "unique_visible_cell_count": 4,
                            "duplicate_cell_visits": 0,
                            "sample_sources": ["reset_local_safety"],
                            "sample_headings": [0.0],
                        },
                        "exploration_sensor": {
                            "sample_count": 1,
                            "ray_count": 91,
                            "cell_visit_count": 4,
                            "unique_visible_cell_count": 4,
                            "duplicate_cell_visits": 0,
                            "sample_sources": ["reset"],
                            "sample_headings": [0.0],
                        },
                    },
                    "planner_failure_counts": {
                        "endpoint_physical_unsafe": 0,
                        "endpoint_unknown_buffer_unsafe": 0,
                        "path_physical_unsafe": 0,
                        "path_unknown_buffer_unsafe": 0,
                        "planner_no_path": 0,
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for episode in evaluation.episodes
    )
    trace_path.write_bytes(trace_bytes)
    result = {
        "episode_count": 64,
        "metrics": dict(evaluation.metrics),
        "bootstrap_audit": dict(evaluation.bootstrap_audit),
        "fairness_audit": dict(evaluation.fairness_audit),
    }

    def artifact_identity(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "path": path.name,
            "sha256": sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    summary_value = {
        "schema_version": "stage6_final_eval_completion/v1",
        "split": split,
        "method": method,
        "episode_count": 64,
        "trace": artifact_identity(trace_path),
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "policy_state_sha256": policy_sha256,
        "result": result,
    }
    summary_path.write_bytes(ArtifactStore.canonical_json_bytes(summary_value))
    commit_value = {
        "schema_version": "stage6_final_eval_commit/v1",
        "split": split,
        "method": method,
        "attempt": 1,
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "policy_state_sha256": policy_sha256,
        "trace": artifact_identity(trace_path),
        "summary": artifact_identity(summary_path),
    }
    commit_path.write_bytes(ArtifactStore.canonical_json_bytes(commit_value))
    verifier_kwargs = {
        "trace_path": trace_path,
        "summary_path": summary_path,
        "commit_path": commit_path,
        "split": split,
        "method": method,
        "config_sha256": config_sha256,
        "safety_contract": SafetyContract.from_stage6_config(config),
        "checkpoint_sha256": checkpoint_sha256,
        "policy_state_sha256": policy_sha256,
        "bootstrap_resamples": config.evaluation.bootstrap_resamples,
        "bootstrap_seed": config.evaluation.bootstrap_seed,
    }
    path_replay = standard_training.verify_standard_final_evaluation_artifacts(
        **verifier_kwargs
    )
    bound_replay = (
        standard_training.verify_standard_final_evaluation_artifacts_from_bytes(
            trace_bytes=trace_path.read_bytes(),
            trace_name=trace_path.name,
            summary_bytes=summary_path.read_bytes(),
            summary_name=summary_path.name,
            commit_bytes=commit_path.read_bytes(),
            commit_name=commit_path.name,
            **{
                key: value
                for key, value in verifier_kwargs.items()
                if key not in {"trace_path", "summary_path", "commit_path"}
            },
        )
    )
    assert bound_replay == path_replay
    assert path_replay["episode_count"] == 64
    assert path_replay["planner_failure_counts"] == {
        "endpoint_physical_unsafe": 0,
        "endpoint_unknown_buffer_unsafe": 0,
        "path_physical_unsafe": 0,
        "path_unknown_buffer_unsafe": 0,
        "planner_no_path": 0,
    }
    assert path_replay["reset_scan_audit"] == {
        "schema_version": "stage6_reset_scan_replay/v1",
        "episode_count": 64,
        "scan_order": [
            "reset_local_safety",
            "reset_exploration",
        ],
        "dual_scan_episode_count": 64,
    }

    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["scenario_seed"] += 1
    trace_path.write_bytes(
        b"".join(
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
    )
    mutated_episodes = tuple(
        EpisodeResult(
            **{
                **{name: row[name] for name in EPISODE_FIELDS},
                "coverage_curve": tuple(row["coverage_curve"]),
            }
        )
        for row in rows
    )
    mutated_metrics, mutated_bootstrap = summarize_episodes(
        mutated_episodes,
        bootstrap_resamples=config.evaluation.bootstrap_resamples,
        bootstrap_seed=config.evaluation.bootstrap_seed,
    )
    result["metrics"] = mutated_metrics
    result["bootstrap_audit"] = mutated_bootstrap
    summary_value["trace"] = artifact_identity(trace_path)
    summary_path.write_bytes(ArtifactStore.canonical_json_bytes(summary_value))
    commit_value["trace"] = artifact_identity(trace_path)
    commit_value["summary"] = artifact_identity(summary_path)
    commit_path.write_bytes(ArtifactStore.canonical_json_bytes(commit_value))

    with pytest.raises(
        standard_training.StandardTrainingError,
        match="final evaluation trace seed schedule drifted",
    ):
        standard_training.verify_standard_final_evaluation_artifacts(
            **verifier_kwargs
        )


def test_resource_audit_replays_segment_v3_and_rejects_unbound_or_legacy_terminal(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.eval.standard import STANDARD_EVALUATION_METHODS
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        append_bound_resource_lifecycle_terminal,
        append_resource_segment_start,
        bind_resource_row_to_segment,
    )
    from lunar_exploration_ppo.utils.resources import ResourceSnapshot, evaluate_resource_gates

    def resource(sample_count: int, rss_bytes: int) -> dict[str, object]:
        snapshot = ResourceSnapshot(
            d_free_bytes=200 * 1024**3,
            rss_bytes=rss_bytes,
            peak_vram_bytes=2048,
            rss_source="process_tree_lifecycle_peak_current_sum/v1",
            rss_root_pid=os.getpid(),
            rss_sample_count=sample_count,
            rss_latest_process_count=1,
            rss_peak_process_count=1,
        )
        decision = evaluate_resource_gates(snapshot, preflight=False)
        return {
            "d_free_bytes": snapshot.d_free_bytes,
            "rss_bytes": snapshot.rss_bytes,
            "peak_vram_bytes": snapshot.peak_vram_bytes,
            "rss_source": snapshot.rss_source,
            "rss_root_pid": snapshot.rss_root_pid,
            "rss_sample_count": snapshot.rss_sample_count,
            "rss_latest_process_count": snapshot.rss_latest_process_count,
            "rss_peak_process_count": snapshot.rss_peak_process_count,
            "warnings": list(decision.warnings),
            "hard_stops": list(decision.hard_stops),
            "passed": decision.passed,
        }

    resource_path = tmp_path / "resource_audit.jsonl"
    segment = append_resource_segment_start(
        resource_path,
        first_sample=resource(1, 1024),
    )
    sample_count = 1
    rss_bytes = 1024
    for split in ("test", "unseen"):
        for method in STANDARD_EVALUATION_METHODS:
            identity = {
                "kind": "final_evaluation",
                "transaction_key": f"final:{split}:{method}",
                "split": split,
                "method": method,
            }
            sample_count += 1
            pre = resource(sample_count, rss_bytes)
            sample_count += 1
            rss_bytes += 1
            post = resource(sample_count, rss_bytes)
            for row in (
                {
                        "schema_version": "stage6_resource_attempt/v1",
                        **identity,
                        "attempt": 1,
                        "phase": "pre",
                        "accepted": False,
                        "resource": pre,
                },
                {
                        "schema_version": "stage6_resource_attempt/v1",
                        **identity,
                        "attempt": 1,
                        "phase": "post",
                        "accepted": False,
                        "resource": post,
                },
                {
                        "schema_version": "stage6_resource_acceptance/v1",
                        **identity,
                        "attempt": 1,
                        "phase": "accepted",
                        "accepted": True,
                        "pre": pre,
                        "post": post,
                },
            ):
                DurableJsonl(resource_path).append(
                    bind_resource_row_to_segment(row, segment)
                )

    rows = tuple(
        json.loads(line)
        for line in resource_path.read_text(encoding="utf-8").splitlines()
    )

    preterminal = standard_training.validate_stage6_resource_audit(
        rows,
        (),
        require_terminal=False,
    )
    assert preterminal["final_accepted_count"] == 10
    assert preterminal["resource_segment_count"] == 1
    with pytest.raises(standard_training.StandardTrainingError, match="terminal"):
        standard_training.validate_stage6_resource_audit(rows, ())

    terminal_resource = resource(sample_count + 1, rss_bytes + 4096)
    terminal = append_bound_resource_lifecycle_terminal(
        resource_path,
        terminal_resource=terminal_resource,
        preterminal_acceptance={"sha256": "b" * 64, "size_bytes": 2048},
    )
    terminal_rows = tuple(
        json.loads(line)
        for line in resource_path.read_text(encoding="utf-8").splitlines()
    )
    validated = standard_training.validate_stage6_resource_audit(
        terminal_rows,
        (),
    )
    assert validated["terminal_evidence_present"] is True
    assert validated["rss_final_sample_count"] == sample_count + 1
    assert validated["rss_lifecycle_peak_bytes"] == rss_bytes + 4096
    assert terminal["schema_version"] == "stage6_terminal_resource_evidence/v4"
    assert terminal["preterminal_acceptance"] == {
        "sha256": "b" * 64,
        "size_bytes": 2048,
    }

    unbound = [json.loads(json.dumps(row)) for row in terminal_rows]
    del unbound[1]["segment_id"]
    del unbound[1]["segment_index"]
    with pytest.raises(
        standard_training.StandardTrainingError,
        match="lifecycle|segment",
    ):
        standard_training.validate_stage6_resource_audit(unbound, ())

    legacy = [json.loads(json.dumps(row)) for row in terminal_rows]
    legacy[-1]["schema_version"] = "stage6_terminal_resource_evidence/v2"
    with pytest.raises(
        standard_training.StandardTrainingError,
        match="lifecycle|terminal",
    ):
        standard_training.validate_stage6_resource_audit(legacy, ())


def test_backend_binds_delayed_checkpoint_acceptance_to_original_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_execution_authority: dict[str, object],
) -> None:
    from lunar_exploration_ppo.utils import resource_lifecycle
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        append_resource_segment_start,
        validate_resource_lifecycle_ledger,
    )

    def resource(
        *,
        root_pid: int,
        sample_count: int,
        rss_bytes: int,
    ) -> dict[str, object]:
        return {
            "d_free_bytes": 200 * 1024**3,
            "rss_bytes": rss_bytes,
            "peak_vram_bytes": 2048,
            "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
            "rss_root_pid": root_pid,
            "rss_sample_count": sample_count,
            "rss_latest_process_count": 1,
            "rss_peak_process_count": 1,
            "warnings": [],
            "hard_stops": [],
            "passed": True,
        }

    resource_path = Path(active_execution_authority["run_root"]) / "s6/resource_audit.jsonl"
    resource_path.parent.mkdir()
    first_pid = os.getpid()
    first_segment = append_resource_segment_start(
        resource_path,
        first_sample=resource(
            root_pid=first_pid,
            sample_count=1,
            rss_bytes=4096,
        ),
    )
    backend = object.__new__(standard_training.StandardProductionBackend)
    _bind_object_backend_capability(backend, active_execution_authority)
    backend._resource_segment_start = first_segment
    identity = {
        "kind": "update",
        "transaction_key": "seed-20260716-update-001",
        "seed": 20260716,
        "update": 1,
    }
    pre = resource(root_pid=first_pid, sample_count=2, rss_bytes=4096)
    post = resource(root_pid=first_pid, sample_count=3, rss_bytes=8192)
    backend._append_resource_phase(
        identity=identity,
        attempt=1,
        phase="pre",
        resource=pre,
    )
    backend._append_resource_phase(
        identity=identity,
        attempt=1,
        phase="post",
        resource=post,
    )

    second_pid = first_pid + 100_000
    monkeypatch.setattr(resource_lifecycle, "_current_pid", lambda: second_pid)
    second_segment = append_resource_segment_start(
        resource_path,
        first_sample=resource(
            root_pid=second_pid,
            sample_count=1,
            rss_bytes=1024,
        ),
    )
    backend._resource_segment_start = second_segment
    checkpoint = standard_training.StandardTransactionCheckpoint(
        transaction_key="seed-20260716-update-001",
        seed=20260716,
        update=1,
        checkpoint_sha256="a" * 64,
        complete_marker_sha256="b" * 64,
        policy_state_sha256="c" * 64,
    )
    backend._append_resource_acceptance(
        identity=identity,
        attempt=1,
        pre=pre,
        post=post,
        checkpoint=checkpoint,
    )

    rows = backend._resource_audit_rows()
    accepted = next(row for row in rows if row.get("phase") == "accepted")
    assert accepted["segment_id"] == first_segment["segment_id"]
    assert accepted["segment_index"] == first_segment["segment_index"]
    assert backend._accepted_resource_attempt(identity) == accepted
    validation = validate_resource_lifecycle_ledger(
        resource_path,
        require_terminal=False,
    )
    assert validation["segment_count"] == 2
    assert validation["active_root_pid"] == second_pid


def test_backend_terminal_resource_capture_binds_post_finalize_peak_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_execution_authority: dict[str, object],
) -> None:
    stage = Path(active_execution_authority["run_root"]) / "s6"
    stage.mkdir()
    prior_resource = {
        "d_free_bytes": 200 * 1024**3,
        "rss_bytes": 4096,
        "peak_vram_bytes": 2048,
        "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
        "rss_root_pid": os.getpid(),
        "rss_sample_count": 10,
        "rss_latest_process_count": 1,
        "rss_peak_process_count": 1,
        "warnings": [],
        "hard_stops": [],
        "passed": True,
    }
    prior_row = {
        "schema_version": "stage6_resource_attempt/v1",
        "kind": "final_evaluation",
        "transaction_key": "final:unseen:gain_over_cost_frontier",
        "split": "unseen",
        "method": "gain_over_cost_frontier",
        "attempt": 1,
        "phase": "post",
        "accepted": False,
        "resource": prior_resource,
    }
    prior_bytes = (
        json.dumps(prior_row, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    resource_path = stage / "resource_audit.jsonl"
    resource_path.write_bytes(prior_bytes)
    terminal_resource = {
        **prior_resource,
        "rss_bytes": 8192,
        "rss_sample_count": 11,
    }
    captures: list[int] = []

    class Monitor:
        def __init__(self) -> None:
            self.running = True
            self.sample_count = 10

        def sample_now(self) -> object:
            assert self.running is True
            self.sample_count += 1
            return object()

        def stop(self) -> None:
            self.running = False

    monitor = Monitor()
    backend = object.__new__(standard_training.StandardProductionBackend)
    _bind_object_backend_capability(backend, active_execution_authority)
    backend._process_tree_monitor = monitor
    backend._preterminal_acceptance_identity = {
        "sha256": "d" * 64,
        "size_bytes": 1234,
    }
    backend._capture_resource_gate = (  # type: ignore[method-assign]
        lambda: captures.append(1) or dict(terminal_resource)
    )
    terminal_appends: list[dict[str, object]] = []

    def append_terminal(**kwargs):
        assert kwargs == {
            "stage_root": stage,
            "terminal_resource": terminal_resource,
            "execution_capability": active_execution_authority["capability"],
        }
        terminal_appends.append(dict(kwargs))
        return {
            "schema_version": "stage6_terminal_resource_evidence/v4",
            "kind": "terminal_lifecycle",
            "transaction_key": "terminal:formal_backend",
            "phase": "terminal",
            "accepted": True,
            "measurement_boundary": (
                "after_finalization_payload_and_monitor_stop_before_"
                "success_commit/v4"
            ),
            "segment_id": "1" * 32,
            "segment_index": 1,
            "prior_resource_audit": {
                "sha256": sha256(prior_bytes).hexdigest(),
                "size_bytes": len(prior_bytes),
            },
            "segment_chain": [],
            "run_rss_lifecycle_peak_bytes": terminal_resource["rss_bytes"],
            "resource": dict(terminal_resource),
            "preterminal_acceptance": dict(
                backend._preterminal_acceptance_identity
            ),
        }

    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    monkeypatch.setattr(
        stage6_terminal_recovery,
        "append_stage6_recovery_resource_terminal",
        append_terminal,
    )

    with pytest.raises(standard_training.StandardTrainingError, match="stopped"):
        backend.record_terminal_resource_evidence()
    sampled = backend.capture_terminal_resource_sample()
    assert sampled["sample_count_before"] == 10
    assert sampled["sample_count_after"] == 11
    monitor.stop()
    terminal = backend.record_terminal_resource_evidence()

    assert captures == [1]
    assert len(terminal_appends) == 1
    assert terminal["resource"] == terminal_resource
    assert terminal["schema_version"] == "stage6_terminal_resource_evidence/v4"
    assert terminal["measurement_boundary"] == (
        "after_finalization_payload_and_monitor_stop_before_success_commit/v4"
    )
    assert terminal["preterminal_acceptance"] == {
        "sha256": "d" * 64,
        "size_bytes": 1234,
    }

    assert backend.record_terminal_resource_evidence() == terminal
    assert captures == [1]
    assert len(terminal_appends) == 1


def test_production_backend_validates_and_hash_binds_five_field_safety_contract() -> None:
    config = load_stage6_config(CONFIG)
    expected = SafetyContract.from_stage6_config(config)
    backend = object.__new__(standard_training.StandardProductionBackend)
    backend.config = config
    backend.config_sha256 = sha256(
        standard_training.ArtifactStore.canonical_json_bytes(
            config.model_dump(mode="json")
        )
    ).hexdigest()
    backend.safety_contract = expected
    review_bindings = _test_review_immutable_bindings()
    backend._immutable_bindings = {
        "config_sha256": "1" * 64,
        "source_set_sha256": "2" * 64,
        "prospective_tree_sha256": sha256(
            str(review_bindings["reviewed_prospective_git_tree"]).encode("ascii")
        ).hexdigest(),
        "data_sha256": "4" * 64,
        "environment_identity": _stage6_environment_identity(),
        "environment_sha256": stage6_environment_sha256(
            _stage6_environment_identity()
        ),
        "stage5_gate_sha256": "5" * 64,
        **review_bindings,
    }
    evaluation = {
        "fairness_audit": {
            "shared_environment_contract": {
                "environment_contract": expected.binding(
                    config_sha256=backend.config_sha256
                )
            }
        }
    }

    assert backend._validated_evaluation_safety_contract(evaluation) == expected
    final_evaluation = {
        "episode_count": 64,
        "metrics": {"episode_count": 64},
        **evaluation,
    }
    assert backend._evaluation_result_record(final_evaluation) == final_evaluation
    assert backend._checkpoint_lineage(1)["safety_contract_sha256"] == expected.sha256

    drifted = json.loads(json.dumps(evaluation))
    drifted["fairness_audit"]["shared_environment_contract"][  # type: ignore[index]
        "environment_contract"
    ]["safety_contract"]["vehicle_radius_m"] = 0.4215874762
    with pytest.raises(standard_training.StandardTrainingError, match="safety"):
        backend._validated_evaluation_safety_contract(drifted)
    drifted_final = {
        "episode_count": 64,
        "metrics": {"episode_count": 64},
        **drifted,
    }
    with pytest.raises(standard_training.StandardTrainingError, match="safety"):
        backend._evaluation_result_record(drifted_final)


def _test_review_immutable_bindings() -> dict[str, object]:
    return {
        "formal_run_id": FORMAL_RUN_ID,
        "changed_path_set_sha256": "6" * 64,
        "review_authorization_record_sha256": "7" * 64,
        "authorization_file_sha256": "8" * 64,
        "review_identity_sha256": "9" * 64,
        "reviewed_prospective_git_tree": "1" * 40,
        "frozen_diff_sha256": "a" * 64,
        "spec_review_sha256": "b" * 64,
        "quality_review_sha256": "c" * 64,
    }


def _install_test_ordinal6_source_repair(
    monkeypatch: pytest.MonkeyPatch,
):
    """Install an explicit test-only ordinal6 context for legacy unit fixtures."""

    from lunar_exploration_ppo.workflows import stage6_source_repair

    class TestOrdinal6SourceRepairContext:
        def __init__(
            self,
            *,
            config,
            current_immutable_bindings: dict[str, object],
        ) -> None:
            self._config = config
            self.current_immutable_bindings = dict(current_immutable_bindings)
            self.origin_immutable_bindings = dict(current_immutable_bindings)
            self.sensor_acceleration_sha256 = "f" * 64
            self.coverage_cache_manifest_path = (
                "D:/xunce/cache/test/coverage-cache-manifest.json"
            )
            self.coverage_cache_manifest_sha256 = "e" * 64
            self.coverage_cache_runtime_mode = (
                "persistent_exact_manifest_read_only/v1"
            )
            self.protected_checkpoint_updates = ()
            self.summary_binding = None

        def require_current(self, label: str) -> None:
            assert isinstance(label, str) and label

        def checkpoint_lineage_for_update(
            self,
            update: int,
        ) -> dict[str, object]:
            assert type(update) is int and update > 0
            immutable = self.current_immutable_bindings
            return {
                "stage5_commit": self._config.stage5_authority.commit_sha256,
                "stage5_gate_sha256": self._config.stage5_authority.gate_sha256,
                "stage5_manifest_sha256": (
                    self._config.stage5_authority.manifest_sha256
                ),
                "stage4_checkpoint_sha256": (
                    self._config.stage5_authority.checkpoint_sha256
                ),
                "stage4_policy_state_sha256": (
                    self._config.stage5_authority.policy_state_sha256
                ),
                "source_set_sha256": immutable["source_set_sha256"],
                "prospective_tree_sha256": immutable[
                    "prospective_tree_sha256"
                ],
                "data_sha256": immutable["data_sha256"],
                "environment_identity": immutable["environment_identity"],
                "environment_sha256": immutable["environment_sha256"],
                "formal_run_id": immutable["formal_run_id"],
                "changed_path_set_sha256": immutable[
                    "changed_path_set_sha256"
                ],
                "review_authorization_record_sha256": immutable[
                    "review_authorization_record_sha256"
                ],
                "authorization_file_sha256": immutable[
                    "authorization_file_sha256"
                ],
                "review_identity_sha256": immutable[
                    "review_identity_sha256"
                ],
                "reviewed_prospective_git_tree": immutable[
                    "reviewed_prospective_git_tree"
                ],
                "frozen_diff_sha256": immutable["frozen_diff_sha256"],
                "spec_review_sha256": immutable["spec_review_sha256"],
                "quality_review_sha256": immutable["quality_review_sha256"],
                "safety_contract_sha256": SafetyContract.from_stage6_config(
                    self._config
                ).sha256,
            }

    monkeypatch.setattr(
        stage6_source_repair,
        "Stage6SourceRepairContext",
        TestOrdinal6SourceRepairContext,
    )
    monkeypatch.setattr(
        standard_training.StandardProductionBackend,
        "_coverage_cache_binding",
        lambda self: {
            "coverage_cache_manifest_path": (
                "D:/xunce/cache/test/coverage-cache-manifest.json"
            ),
            "coverage_cache_manifest_sha256": "e" * 64,
        },
    )
    monkeypatch.setattr(
        standard_training.StandardProductionBackend,
        "_journal_binding_split",
        lambda self: (None, None),
    )

    def make(
        *,
        config,
        current_immutable_bindings: dict[str, object],
    ) -> TestOrdinal6SourceRepairContext:
        return TestOrdinal6SourceRepairContext(
            config=config,
            current_immutable_bindings=current_immutable_bindings,
        )

    def load_context(
        *,
        current_immutable_bindings,
        **kwargs,
    ) -> TestOrdinal6SourceRepairContext:
        del kwargs
        return make(
            config=load_stage6_config(CONFIG),
            current_immutable_bindings=dict(current_immutable_bindings),
        )

    monkeypatch.setattr(
        stage6_source_repair,
        "load_stage6_source_repair_context",
        load_context,
    )
    return make


def _journal_bindings(checkpoint_sha256: str) -> dict[str, object]:
    environment_identity = _stage6_environment_identity()
    review_bindings = _test_review_immutable_bindings()
    return {
        "config_sha256": "1" * 64,
        "source_set_sha256": "2" * 64,
        "prospective_tree_sha256": sha256(
            str(review_bindings["reviewed_prospective_git_tree"]).encode("ascii")
        ).hexdigest(),
        "data_sha256": "4" * 64,
        "environment_identity": environment_identity,
        "environment_sha256": stage6_environment_sha256(environment_identity),
        "stage5_gate_sha256": "5" * 64,
        **review_bindings,
        "checkpoint_sha256": checkpoint_sha256,
    }


def _journal_record_hash(event: dict[str, object]) -> str:
    payload = (
        json.dumps(event, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def test_environment_identity_is_immutable_in_job_phase_and_checkpoint_lineage(
    tmp_path: Path,
    active_execution_authority: dict[str, object],
) -> None:
    checkpoint_sha256 = "a" * 64
    bindings = _journal_bindings(checkpoint_sha256)
    job_journal = Stage6StateJournal(tmp_path / "job-state.jsonl")

    try:
        job_row = job_journal.append("seed_initializing", bindings)
    except Stage6WorkflowError as exc:
        pytest.fail(f"environment-bound job journal append failed: {exc}")
    assert job_row["bindings"]["environment_identity"] == bindings[
        "environment_identity"
    ]
    assert job_row["bindings"]["environment_sha256"] == bindings[
        "environment_sha256"
    ]
    review_binding_keys = (
        "authorization_file_sha256",
        "review_identity_sha256",
        "reviewed_prospective_git_tree",
        "frozen_diff_sha256",
        "spec_review_sha256",
        "quality_review_sha256",
    )
    assert {
        key: job_row["bindings"][key] for key in review_binding_keys
    } == {key: bindings[key] for key in review_binding_keys}

    config = load_stage6_config(CONFIG)
    backend = object.__new__(standard_training.StandardProductionBackend)
    _bind_object_backend_capability(backend, active_execution_authority)
    backend.config = config
    backend.config_sha256 = sha256(
        standard_training.ArtifactStore.canonical_json_bytes(
            config.model_dump(mode="json")
        )
    ).hexdigest()
    backend.safety_contract = SafetyContract.from_stage6_config(config)
    backend._immutable_bindings = {
        key: value for key, value in bindings.items() if key != "checkpoint_sha256"
    }
    backend.stage_root.mkdir()
    backend.phase_journal = Stage6StateJournal(backend.stage_root / "phase-state.jsonl")

    checkpoint_lineage = backend._checkpoint_lineage(1)
    assert checkpoint_lineage["environment_identity"] == bindings[
        "environment_identity"
    ]
    assert checkpoint_lineage["environment_sha256"] == bindings[
        "environment_sha256"
    ]
    assert {
        key: checkpoint_lineage[key] for key in review_binding_keys
    } == {key: bindings[key] for key in review_binding_keys}
    backend._append_phase_state("preflight", checkpoint_sha256=checkpoint_sha256)
    phase_row = backend.phase_journal.verify()[0]
    assert phase_row["bindings"]["environment_identity"] == bindings[
        "environment_identity"
    ]
    assert phase_row["bindings"]["environment_sha256"] == bindings[
        "environment_sha256"
    ]
    assert {
        key: phase_row["bindings"][key] for key in review_binding_keys
    } == {key: bindings[key] for key in review_binding_keys}


def _make_stage6_directory_reparse(link: Path, target: Path) -> None:
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


def _save_stage6_checkpoint_fixture(root: Path):
    from test_stage4_checkpoint import VERSIONS, _policy_and_optimizer

    config = load_stage6_config(CONFIG)
    lineage = {
        "stage5_commit": "b" * 40,
        "stage5_gate_sha256": "1" * 64,
        "stage5_manifest_sha256": "2" * 64,
        "stage4_checkpoint_sha256": "3" * 64,
        "stage4_policy_state_sha256": "4" * 64,
        "source_set_sha256": "5" * 64,
        "prospective_tree_sha256": "6" * 64,
        "data_sha256": "7" * 64,
        "environment_identity": _stage6_environment_identity(),
        "environment_sha256": stage6_environment_sha256(
            _stage6_environment_identity()
        ),
    }
    policy, optimizer = _policy_and_optimizer()
    manager = CheckpointManager(
        root,
        schema_version=config.checkpoint.schema_version,
    )
    receipt = manager.save_complete(
        policy=policy,
        optimizer=optimizer,
        update_step=1,
        normalization_stats={},
        scenario_sampler_state={"workers": list(range(8))},
        vector_env_states=[{"worker": worker} for worker in range(8)],
        best_record={},
        versions=VERSIONS,
        top_m_config={"top_m": 1024},
        scale_profile="Standard v1",
        training_config={"updates_per_seed": 100},
        config_sha256="a" * 64,
        lineage=lineage,
        eval_metrics={},
    )
    return receipt, lineage, config.checkpoint.schema_version


@pytest.mark.parametrize("mutation", ("missing", "extra", "invalid_sha"))
def test_state_journal_verify_rejects_rehashed_binding_schema_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / mutation / "job-state.jsonl"
    journal = Stage6StateJournal(path)
    journal.append("preflight", _journal_bindings("a" * 64))
    row = json.loads(path.read_text(encoding="utf-8"))
    bindings = row["bindings"]
    if mutation == "missing":
        del bindings["data_sha256"]
    elif mutation == "extra":
        bindings["unexpected_sha256"] = "f" * 64
    else:
        bindings["data_sha256"] = "not-a-sha256"
    event = {
        "state": row["state"],
        "previous_record_hash": row["previous_record_hash"],
        "bindings": bindings,
    }
    row["record_hash"] = _journal_record_hash(event)
    path.write_bytes(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    )

    with pytest.raises(Stage6WorkflowError, match="binding"):
        journal.verify()


def test_state_journal_snapshot_bytes_match_runtime_path_verification(
    tmp_path: Path,
) -> None:
    path = tmp_path / "snapshot/job-state.jsonl"
    journal = Stage6StateJournal(path)
    journal.append("preflight", _journal_bindings("a" * 64))
    payload = path.read_bytes()

    assert Stage6StateJournal.verify_snapshot_bytes(payload) == journal.verify()
    with pytest.raises(Stage6WorkflowError, match="snapshot bytes"):
        Stage6StateJournal.verify_snapshot_bytes(bytearray(payload))


def test_receipt_index_binds_completed_prefix_and_allows_only_ahead_one(
    tmp_path: Path,
) -> None:
    config = load_stage6_config(CONFIG)
    transactions = standard_training.build_standard_training_transactions(config)

    def checkpoint(index: int, digest: str) -> standard_training.StandardTransactionCheckpoint:
        transaction = transactions[index]
        return standard_training.StandardTransactionCheckpoint(
            transaction_key=transaction.key,
            seed=transaction.seed,
            update=transaction.update,
            checkpoint_sha256=digest * 64,
            complete_marker_sha256=chr(ord(digest) + 1) * 64,
            policy_state_sha256=chr(ord(digest) + 2) * 64,
        )

    first = checkpoint(0, "a")
    second = checkpoint(1, "d")
    index = standard_training.CheckpointReceiptIndex(
        tmp_path / "checkpoints/index.jsonl"
    )
    assert index.append_once(first) is True
    assert index.append_once(first) is False
    assert index.append_once(second) is True
    rows = index.verify()
    assert standard_training.CheckpointReceiptIndex.verify_snapshot_bytes(
        index.path.read_bytes()
    ) == rows
    with pytest.raises(standard_training.StandardTrainingError, match="snapshot bytes"):
        standard_training.CheckpointReceiptIndex.verify_snapshot_bytes(
            bytearray(index.path.read_bytes())
        )
    assert all(
        set(row)
        == {
            "transaction_key",
            "seed",
            "update",
            "checkpoint_sha256",
            "complete_marker_sha256",
            "policy_state_sha256",
            "previous_record_hash",
            "record_hash",
        }
        for row in rows
    )
    for line in index.path.read_bytes().splitlines(keepends=True):
        assert line == (
            json.dumps(
                json.loads(line.decode("utf-8")),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    immutable = {
        key: value
        for key, value in _journal_bindings(first.checkpoint_sha256).items()
        if key != "checkpoint_sha256"
    }
    journal = Stage6StateJournal(tmp_path / "job-state.jsonl")
    for state in transactions[0].commit_states:
        journal.append(state, _journal_bindings(first.checkpoint_sha256))
    assert standard_training.verify_journal_checkpoint_bindings(
        journal.verify(), transactions, rows, immutable
    ) == (transactions[0].key,)

    mismatched = Stage6StateJournal(tmp_path / "mismatched/job-state.jsonl")
    mismatched.append(
        transactions[0].commit_states[0],
        _journal_bindings(first.checkpoint_sha256),
    )
    mismatched.append(
        transactions[0].commit_states[1],
        _journal_bindings(second.checkpoint_sha256),
    )
    with pytest.raises(standard_training.StandardTrainingError, match="receipt"):
        standard_training.verify_journal_checkpoint_bindings(
            mismatched.verify(), transactions, rows, immutable
        )

    drift = standard_training.StandardTransactionCheckpoint(
        transaction_key=first.transaction_key,
        seed=first.seed,
        update=first.update,
        checkpoint_sha256="9" * 64,
        complete_marker_sha256=first.complete_marker_sha256,
        policy_state_sha256=first.policy_state_sha256,
    )
    with pytest.raises(standard_training.StandardTrainingError, match="receipt"):
        index.append_once(drift)

    assert index.append_once(checkpoint(2, "1")) is True
    with pytest.raises(standard_training.StandardTrainingError, match="ahead"):
        standard_training.verify_journal_checkpoint_bindings(
            journal.verify(), transactions, index.verify(), immutable
        )


def test_training_transactions_bind_validation_checkpoint_and_seed_boundaries() -> None:
    config = load_stage6_config(CONFIG)

    transactions = standard_training.build_standard_training_transactions(config)

    assert standard_training.FROZEN_SEEDS == (20260716,)
    assert len(transactions) == 100
    assert [item.sequence for item in transactions] == list(range(100))
    assert transactions[0].seed == 20260716
    assert transactions[0].update == 1
    assert transactions[0].validation_episodes == 0
    assert transactions[0].commit_states == (
        "seed_20260716_initializing",
        "seed_20260716_training_update_1",
    )
    assert transactions[9].validation_episodes == 16
    assert transactions[9].commit_states == (
        "seed_20260716_training_update_10",
        "seed_20260716_validating_update_10",
    )
    assert transactions[99].commit_states == (
        "seed_20260716_training_update_100",
        "seed_20260716_validating_update_100",
        "seed_20260716_complete",
    )
    assert sum(item.validation_episodes == 16 for item in transactions) == 10
    assert transactions[-1].seed == 20260716
    assert transactions[-1].update == 100
    assert transactions[-1].commit_states[-1] == "seed_20260716_complete"
    assert len({item.key for item in transactions}) == 100


def test_resume_requires_exact_transaction_prefix_and_matching_complete_checkpoint() -> None:
    config = load_stage6_config(CONFIG)
    transactions = standard_training.build_standard_training_transactions(config)
    completed = tuple(item.key for item in transactions[:10])
    checkpoint = standard_training.StandardTransactionCheckpoint(
        transaction_key=transactions[9].key,
        seed=20260716,
        update=10,
        checkpoint_sha256="a" * 64,
        complete_marker_sha256="b" * 64,
        policy_state_sha256="c" * 64,
    )

    remaining = standard_training.resume_standard_training_transactions(
        transactions,
        completed_keys=completed,
        checkpoint=checkpoint,
    )

    assert remaining[0] == transactions[10]
    assert len(remaining) == 90
    assert standard_training.resume_standard_training_transactions(
        transactions,
        completed_keys=(),
        checkpoint=None,
    ) == transactions

    with pytest.raises(standard_training.StandardTrainingError, match="prefix"):
        standard_training.resume_standard_training_transactions(
            transactions,
            completed_keys=(transactions[1].key,),
            checkpoint=checkpoint,
        )
    with pytest.raises(standard_training.StandardTrainingError, match="checkpoint"):
        standard_training.resume_standard_training_transactions(
            transactions,
            completed_keys=completed,
            checkpoint=standard_training.StandardTransactionCheckpoint(
                transaction_key=transactions[8].key,
                seed=20260716,
                update=9,
                checkpoint_sha256="a" * 64,
                complete_marker_sha256="b" * 64,
                policy_state_sha256="c" * 64,
            ),
        )
    with pytest.raises(standard_training.StandardTrainingError, match="checkpoint"):
        standard_training.resume_standard_training_transactions(
            transactions,
            completed_keys=(),
            checkpoint=checkpoint,
        )


def test_execute_standard_training_is_a_production_only_surface() -> None:
    function = standard_training.execute_standard_training
    parameters = inspect.signature(function).parameters

    assert tuple(parameters) == (
        "config",
        "run_root",
        "repo_root",
        "stage5_authority",
        "verified_review_authorization",
        "execution_capability",
        "planning_warm_start_context",
        "planning_warm_start_sha256",
        "planning_child_source_repair_context",
        "planning_child_source_repair_sha256",
        "verified_parent_u74",
    )
    assert all(parameters[name].kind is inspect.Parameter.KEYWORD_ONLY for name in parameters)
    source = inspect.getsource(function).lower()
    for forbidden in (
        "fixture",
        "fake",
        "adapter",
        "dry_run",
        "max_updates",
        "force",
        "skip",
    ):
        assert forbidden not in source
    for name in (
        "_append_resource_phase",
        "record_terminal_resource_evidence",
        "run_update",
        "finalize",
        "commit_terminal_acceptance",
    ):
        assert not hasattr(
            getattr(standard_training.StandardProductionBackend, name),
            "__wrapped__",
        )


def test_planning_child_repair_splits_u84_and_u85_checkpoint_lineage() -> None:
    class WarmStart:
        def checkpoint_lineage_for_update(
            self,
            update: int,
            *,
            warm_start_artifact_sha256: str,
        ) -> dict[str, object]:
            return {
                "warm_start_artifact_sha256": (
                    warm_start_artifact_sha256
                ),
                "checkpoint_update": update,
            }

    class ChildRepair:
        origin_immutable_bindings = {"source_set_sha256": "origin"}
        current_immutable_bindings = {
            "source_set_sha256": "reviewed-u85"
        }
        last_origin_update = 84
        next_transaction_key = f"0010:{20260716}:update:085"

        def expected_historical_checkpoint_lineage(
            self,
            update: int,
        ) -> dict[str, object]:
            assert update == 84
            return {"historical_update": 84}

        def checkpoint_lineage_for_update(
            self,
            update: int,
        ) -> dict[str, object]:
            assert update == 85
            return {
                "schema_version": (
                    "stage6_planning_child_source_repair/v1"
                ),
                "first_repaired_update": 85,
            }

        def checkpoint_immutable_bindings_for_update(
            self,
            update: int,
        ) -> dict[str, object]:
            assert update == 85
            return dict(self.current_immutable_bindings)

    backend = object.__new__(
        standard_training.StandardProductionBackend
    )
    backend._planning_warm_start = WarmStart()
    backend._planning_warm_start_sha256 = "a" * 64
    backend._planning_child_source_repair = ChildRepair()
    backend._source_repair = None
    backend._origin_checkpoint_lineage = lambda bindings=None: {
        "current_source": "reviewed-u85"
    }

    assert backend._checkpoint_lineage_for_update(84) == {
        "historical_update": 84
    }
    assert backend._checkpoint_lineage_for_update(85) == {
        "current_source": "reviewed-u85",
        "warm_start_artifact_sha256": "a" * 64,
        "checkpoint_update": 85,
        "planning_child_source_repair": {
            "schema_version": (
                "stage6_planning_child_source_repair/v1"
            ),
            "first_repaired_update": 85,
        },
    }
    assert backend._journal_binding_split() == (
        {"source_set_sha256": "origin"},
        f"0010:{20260716}:update:085",
    )


def test_planning_child_repair_requires_u85_attempt2_segment2() -> None:
    class ChildRepair:
        first_repaired_update = 85
        next_attempt = 2
        next_resource_segment_index = 2
        next_transaction_key = f"0010:{20260716}:update:085"
        effective_next_update = 85
        effective_next_attempt = 2
        effective_next_resource_segment_index = 2
        effective_next_transaction_key = (
            f"0010:{20260716}:update:085"
        )

    backend = object.__new__(
        standard_training.StandardProductionBackend
    )
    backend._planning_child_source_repair = ChildRepair()
    backend._resource_segment_start = {"segment_index": 2}
    backend._next_resource_attempt = lambda _key: 2
    transaction = standard_training.StandardTrainingTransaction(
        sequence=10,
        seed=20260716,
        seed_index=0,
        update=85,
        validation_episodes=0,
        commit_states=("seed_20260716_training_update_85",),
    )

    backend._require_planning_child_resume_boundary((transaction,))

    backend._resource_segment_start = {"segment_index": 3}
    with pytest.raises(
        standard_training.StandardTrainingError,
        match="resource segment",
    ):
        backend._require_planning_child_resume_boundary((transaction,))


def test_planning_child_repair_profile_combination_is_fail_closed() -> None:
    from lunar_exploration_ppo.workflows import stage6

    stage6._validate_stage6_repair_context_combination(
        planning_warm_start_context=object(),
        planning_child_source_repair_context=object(),
        classic_source_repair_context=None,
    )
    stage6._validate_stage6_repair_context_combination(
        planning_warm_start_context=object(),
        planning_child_source_repair_context=None,
        classic_source_repair_context=None,
    )

    with pytest.raises(
        stage6.Stage6WorkflowError,
        match="requires planning warm-start",
    ):
        stage6._validate_stage6_repair_context_combination(
            planning_warm_start_context=None,
            planning_child_source_repair_context=object(),
            classic_source_repair_context=None,
        )
    with pytest.raises(
        stage6.Stage6WorkflowError,
        match="classic source-repair",
    ):
        stage6._validate_stage6_repair_context_combination(
            planning_warm_start_context=object(),
            planning_child_source_repair_context=object(),
            classic_source_repair_context=object(),
        )
    with pytest.raises(
        stage6.Stage6WorkflowError,
        match="classic source-repair",
    ):
        stage6._validate_stage6_repair_context_combination(
            planning_warm_start_context=object(),
            planning_child_source_repair_context=None,
            classic_source_repair_context=object(),
        )


def test_planning_child_repair_is_forwarded_through_protected_workflow() -> None:
    from lunar_exploration_ppo.workflows import stage6

    for function in (
        stage6.run_stage6_workflow,
        stage6._run_stage6_workflow_for_test,
        stage6._stage6_input_pin_requests,
        stage6._acquire_stage6_workflow_input_pin,
        stage6._stage6_execution_capability_scope,
    ):
        assert "planning_child_source_repair_path" in (
            inspect.signature(function).parameters
        )
    for function in (
        stage6._run_stage6_workflow_at_root,
        stage6._run_stage6_workflow_locked,
    ):
        parameters = inspect.signature(function).parameters
        assert "planning_child_source_repair_context" in parameters
        assert "planning_child_source_repair_sha256" in parameters

    public_source = inspect.getsource(stage6.run_stage6_workflow)
    protected_source = inspect.getsource(stage6._run_stage6_workflow_locked)
    recovery_source = inspect.getsource(
        stage6._verify_stage6_recovery_authorization_bindings
    )
    assert "load_planning_child_source_repair_artifact" in public_source
    assert "planning_child_source_repair_context=" in public_source
    assert "planning_child_source_repair_context=" in protected_source
    assert "planning-child-source-repair.json" in recovery_source


def test_training_mutation_surfaces_require_capability_before_dispatch_or_mkdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_stage6_config(CONFIG)
    run_root = tmp_path / FORMAL_RUN_ID
    authority = {"gate_sha256": config.stage5_authority.gate_sha256}
    forbidden_events: list[str] = []

    def forbidden_private(**kwargs: object) -> dict[str, object]:
        del kwargs
        forbidden_events.append("private_execute")
        raise AssertionError("public capability gate must precede private dispatch")

    monkeypatch.setattr(
        standard_training,
        "_execute_standard_training",
        forbidden_private,
    )
    with pytest.raises(
        standard_training.StandardTrainingError,
        match="execution capability",
    ):
        standard_training.execute_standard_training(
            config=config,
            run_root=run_root,
            repo_root=ROOT,
            stage5_authority=authority,
            verified_review_authorization={},
            execution_capability=None,
        )

    assert forbidden_events == []
    assert not run_root.exists()


@pytest.mark.parametrize("capability", [None, object()])
def test_internal_training_rejects_missing_or_forged_capability_before_import(
    tmp_path: Path,
    capability: object,
) -> None:
    config = load_stage6_config(CONFIG)
    run_root = tmp_path / FORMAL_RUN_ID

    with pytest.raises(
        standard_training.StandardTrainingError,
        match="execution capability",
    ):
        standard_training._execute_standard_training(
            config=config,
            run_root=run_root,
            repo_root=ROOT,
            stage5_authority={
                "gate_sha256": config.stage5_authority.gate_sha256
            },
            verified_review_authorization={},
            execution_capability=capability,
        )

    assert not run_root.exists()


@pytest.mark.parametrize("capability", [None, object()])
def test_backend_constructor_rejects_missing_or_forged_capability_before_mkdir(
    tmp_path: Path,
    capability: object,
) -> None:
    config = load_stage6_config(CONFIG)
    run_root = tmp_path / FORMAL_RUN_ID

    with pytest.raises(
        standard_training.StandardTrainingError,
        match="execution capability",
    ):
        standard_training.StandardProductionBackend(
            config=config,
            run_root=run_root,
            repo_root=ROOT,
            stage5_authority={
                "gate_sha256": config.stage5_authority.gate_sha256
            },
            execution_capability=capability,
            _components={},
        )

    assert not run_root.exists()


def test_retained_backend_cannot_create_final_eval_paths_after_capability_revocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_stage6_workflow import _active_execution_capability

    with _active_execution_capability(tmp_path, monkeypatch) as active:
        backend = object.__new__(standard_training.StandardProductionBackend)
        _bind_object_backend_capability(backend, active)
        attempts_root = Path(active["run_root"]) / "stage6-attempts"
        assert not attempts_root.exists()

    with pytest.raises(
        standard_training.StandardTrainingError,
        match="execution capability",
    ):
        backend._final_eval_paths(
            split="test",
            method="ppo_policy",
            attempt=1,
        )
    assert not attempts_root.exists()


def test_persist_execution_identity_rejects_missing_capability_without_writes(
    tmp_path: Path,
) -> None:
    config = load_stage6_config(CONFIG)
    run_root = tmp_path / FORMAL_RUN_ID
    stage_root = run_root / "s6"

    with pytest.raises(
        standard_training.StandardTrainingError,
        match="execution capability",
    ):
        standard_training._persist_execution_identity(
            stage_root=stage_root,
            config=config,
            run_root=run_root,
            repo_root=ROOT,
            stage5_authority={},
            execution_capability=None,
            config_bytes=b"{}\n",
            identity={},
            immutable_bindings={},
            verified_review_authorization={},
        )

    assert not run_root.exists()


def test_persist_execution_identity_writes_exact_warm_effective_config_and_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows import stage6_planning_warm_start
    from test_stage6_planning_warm_start import _artifact
    from test_stage6_workflow import _active_execution_capability

    effective = stage6_planning_warm_start.planning_effective_config_bytes(
        CONFIG.read_bytes()
    )
    artifact = _artifact()
    artifact["child"]["run_id"] = FORMAL_RUN_ID  # type: ignore[index]
    artifact["child"]["effective_config_sha256"] = sha256(  # type: ignore[index]
        effective
    ).hexdigest()
    warm_path = tmp_path / "planning-warm-start.json"
    warm_path.write_bytes(
        (
            json.dumps(
                artifact,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    )
    context, artifact_sha256 = (
        stage6_planning_warm_start.load_planning_warm_start_artifact(
            warm_path,
            expected_child_run_id=FORMAL_RUN_ID,
            expected_child_config_bytes=effective,
        )
    )
    with _active_execution_capability(
        tmp_path / "authority",
        monkeypatch,
        run_root=tmp_path / "authority" / FORMAL_RUN_ID,
        effective_config_bytes=effective,
        planning_warm_start_path=warm_path,
        environment_identity=_stage6_environment_identity(),
    ) as active:
        stage_root = Path(active["run_root"]) / "s6"
        stage_root.mkdir(parents=True)
        immutable = {
            key: active["identity"][key]
            for key in (
                "config_sha256",
                "source_set_sha256",
                "prospective_tree_sha256",
                "data_sha256",
                "environment_identity",
                "environment_sha256",
            )
        }
        immutable["stage5_gate_sha256"] = active["stage5_handle"].identity[
            "gate_sha256"
        ]
        immutable.update(_review_immutable_bindings(active["verified"]))

        standard_training._persist_execution_identity(
            stage_root=stage_root,
            config=active["config"],
            run_root=active["run_root"],
            repo_root=active["repo_root"],
            stage5_authority=active["stage5_handle"].identity,
            execution_capability=active["capability"],
            config_bytes=effective,
            identity=active["identity"],
            immutable_bindings=immutable,
            verified_review_authorization=active["verified"],
            planning_warm_start_context=context,
        )

    assert (stage_root / "config.json").read_bytes() == effective
    lineage = json.loads(
        (stage_root / "lineage_audit.json").read_text(encoding="utf-8")
    )
    assert lineage["schema_version"] == "stage6_lineage_audit/v3"
    assert lineage["execution_identity"]["config_sha256"] == sha256(
        effective
    ).hexdigest()
    assert lineage["immutable_bindings"]["config_sha256"] == sha256(
        effective
    ).hexdigest()
    assert lineage["verified_review_authorization"]["config_sha256"] == sha256(
        effective
    ).hexdigest()
    assert lineage["planning_warm_start"] == {
        "artifact": artifact,
        "artifact_sha256": artifact_sha256,
        "artifact_size_bytes": len(warm_path.read_bytes()),
    }
    assert lineage["planning_warm_start"]["artifact"]["parent"][
        "config_sha256"
    ] == stage6_planning_warm_start.PARENT_CONFIG_SHA256


def test_planning_child_repair_never_rewrites_origin_lineage_or_u84(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_stage6_planning_child_source_repair import (
        _build,
        _context,
        _fixture,
    )

    fixture = _fixture(tmp_path)
    context = _context(fixture, _build(fixture))
    config = load_stage6_config(CONFIG)
    watched_paths = (
        fixture.stage_root / "config.json",
        fixture.stage_root / "lineage_audit.json",
        fixture.stage_root
        / "checkpoints/seed-20260716/update-00000084/checkpoint.pt",
        fixture.stage_root
        / "checkpoints/seed-20260716/update-00000084/manifest.json",
        fixture.stage_root
        / "checkpoints/seed-20260716/update-00000084/complete.json",
    )
    old_ns = 1_700_000_000_000_000_000
    for path in watched_paths:
        os.utime(path, ns=(old_ns, old_ns))
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in watched_paths
    }
    monkeypatch.setattr(
        standard_training,
        "_standard_execution_operation",
        lambda *args, **kwargs: nullcontext(),
    )

    standard_training._persist_execution_identity(
        stage_root=fixture.stage_root,
        config=config,
        run_root=fixture.stage_root.parent,
        repo_root=ROOT,
        stage5_authority={},
        execution_capability=object(),
        config_bytes=(fixture.stage_root / "config.json").read_bytes(),
        identity=fixture.current_execution_identity,
        immutable_bindings=fixture.current_immutable_bindings,
        verified_review_authorization=(
            fixture.current_verified_review_authorization
        ),
        planning_warm_start_context=object(),
        planning_child_source_repair_context=context,
    )

    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in watched_paths
    } == before


def test_child_repair_currentness_is_rechecked_at_mutation_boundaries() -> None:
    for function in (
        standard_training.StandardProductionBackend.run_update,
        standard_training.StandardProductionBackend.finalize,
        standard_training.StandardProductionBackend.commit_terminal_acceptance,
    ):
        protected = next(
            cell.cell_contents
            for cell in (function.__closure__ or ())
            if inspect.isfunction(cell.cell_contents)
        )
        assert "_require_source_repair_current" in inspect.getsource(protected)


def test_execute_engine_wires_real_training_resume_and_final_eval_collaborators() -> None:
    wrapped = standard_training._execute_standard_training
    source = next(
        candidate
        for candidate in (
            inspect.getsource(cell.cell_contents)
            for cell in (wrapped.__closure__ or ())
            if inspect.isfunction(cell.cell_contents)
        )
        if "build_standard_catalog" in candidate
    )
    lowered = source.lower()

    assert "incomplete" not in lowered
    for required in (
        "build_standard_catalog",
        "load_stage4_policy_for_standard",
        "standard_env_specs",
        "SpawnVectorEnv",
        "RolloutCollector",
        "PPOTrainer",
        "CheckpointManager",
        "load_last_complete",
        "restore_states",
        "run_standard_update_transaction",
        "select_global_best",
        "run_standard_evaluation",
        "STANDARD_EVALUATION_METHODS",
        "write_stage6_manifest",
        "ProcessTreeRSSMonitor",
    ):
        assert required in source


def test_execute_constructs_production_backend_and_calls_fixed_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_authority_factory,
) -> None:
    config = load_stage6_config(CONFIG)
    _install_test_ordinal6_source_repair(monkeypatch)
    repo_root = tmp_path / "repo"
    canonical_config = standard_training.ArtifactStore.canonical_json_bytes(
        config.model_dump(mode="json")
    )
    repo_root.joinpath("configs").mkdir(parents=True)
    repo_root.joinpath("configs/ppo_highres_frontier_stage6_v1.json").write_bytes(
        canonical_config
    )
    from lunar_exploration_ppo.workflows import stage6 as stage6_workflow

    captured: dict[str, object] = {}
    environment_identity = _stage6_environment_identity()
    identity = {
        "schema_version": "stage6_execution_identity/v1",
        "base_commit": stage6_workflow.STAGE5_COMMIT,
        "head_commit": stage6_workflow.STAGE5_COMMIT,
        "prospective_git_tree": "1" * 40,
        "prospective_tree_sha256": sha256(("1" * 40).encode("ascii")).hexdigest(),
        "changed_paths": list(stage6_workflow.STAGE6_SOURCE_PATHS),
        "changed_path_set_sha256": "1" * 64,
        "real_index_empty": True,
        "config_sha256": sha256(canonical_config).hexdigest(),
        "source_set_sha256": "2" * 64,
        "data_sha256": "4" * 64,
        "catalog_sha256": "5" * 64,
        "environment_identity": environment_identity,
        "environment_sha256": stage6_environment_sha256(environment_identity),
        "source_identity": {"schema_version": "test-source/v1"},
        "data_identity": {"schema_version": "test-data/v1"},
    }
    active = execution_authority_factory(
        repo_root=repo_root,
        config_path=(repo_root / "configs/ppo_highres_frontier_stage6_v1.json"),
        run_root=tmp_path / FORMAL_RUN_ID,
        environment_identity=environment_identity,
    )
    identity = dict(active["identity"])
    verified_review = dict(active["verified"])
    from lunar_exploration_ppo.utils import resource_lifecycle, resources as resource_utils
    from lunar_exploration_ppo.utils.resources import ResourceSnapshot

    monitor_events: list[str] = []

    class Monitor:
        def __init__(self):
            self.peak_rss_bytes = 4096
            self.stop_calls = 0
            self.stopped = False
            self.running = False
            self.root_pid = os.getpid()
            self.sample_count = 1
            self.latest_process_count = 1
            self.peak_process_count = 1
            self.peak_aggregate_rss_bytes = 4096

        def __enter__(self):
            self.running = True
            monitor_events.append("monitor_start")
            return self

        def stop(self):
            self.stop_calls += 1
            if self.stopped:
                return
            assert monitor_events[-1] == "terminal_sample"
            self.peak_rss_bytes = 16 * 1024**3
            self.stopped = True
            self.running = False
            monitor_events.append("stop")

        def sample_now(self):
            assert self.stopped is False
            assert monitor_events[-1] == "preterminal_receipt"
            self.sample_count += 1
            monitor_events.append("terminal_sample")
            return object()

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback
            self.stop()
            return False

    monkeypatch.setattr(resource_utils, "ProcessTreeRSSMonitor", Monitor)

    def capture_segment_snapshot(*, peak_vram_bytes, process_tree_monitor):
        assert peak_vram_bytes == 0
        assert process_tree_monitor.running is True
        assert monitor_events == ["monitor_start"]
        monitor_events.append("segment_gate")
        return ResourceSnapshot(
            d_free_bytes=200 * 1024**3,
            rss_bytes=process_tree_monitor.peak_aggregate_rss_bytes,
            peak_vram_bytes=0,
            rss_source="process_tree_lifecycle_peak_current_sum/v1",
            rss_root_pid=process_tree_monitor.root_pid,
            rss_sample_count=process_tree_monitor.sample_count,
            rss_latest_process_count=process_tree_monitor.latest_process_count,
            rss_peak_process_count=process_tree_monitor.peak_process_count,
        )

    monkeypatch.setattr(
        resource_utils,
        "capture_resource_snapshot",
        capture_segment_snapshot,
    )

    segment_start = {
        "schema_version": "stage6_resource_lifecycle_segment/v1",
        "kind": "resource_lifecycle_segment",
        "phase": "segment_start",
        "segment_id": "1" * 32,
        "segment_index": 1,
        "root_pid": os.getpid(),
        "first_sample": {},
        "prior_resource_log": {"sha256": "0" * 64, "size_bytes": 0},
    }

    def append_segment(path, *, first_sample):
        assert path == tmp_path / FORMAL_RUN_ID / "s6/resource_audit.jsonl"
        assert first_sample["passed"] is True
        assert first_sample["rss_root_pid"] == os.getpid()
        assert monitor_events == ["monitor_start", "segment_gate"]
        monitor_events.append("segment_start")
        return {**segment_start, "first_sample": dict(first_sample)}

    monkeypatch.setattr(
        resource_lifecycle,
        "append_resource_segment_start",
        append_segment,
    )

    class Backend:
        def __init__(self, **kwargs):
            assert monitor_events == [
                "monitor_start",
                "segment_gate",
                "segment_start",
            ]
            monitor_events.append("backend")
            captured["backend_kwargs"] = kwargs
            self.monitor = kwargs["_process_tree_monitor"]
            self.terminal_rss_bytes = None

        def record_preflight_phase(self):
            captured["phase"] = "preflight"
            monitor_events.append("preflight_phase")

        def capture_terminal_resource_sample(self):
            self.monitor.sample_now()
            return {"passed": True}

        def record_terminal_resource_evidence(self):
            assert monitor_events == [
                "monitor_start",
                "segment_gate",
                "segment_start",
                "backend",
                "preflight_phase",
                "schedule",
                "semantic_verifier",
                "preterminal_receipt",
                "terminal_sample",
                "stop",
            ]
            assert self.monitor.stopped is True
            self.terminal_rss_bytes = self.monitor.peak_rss_bytes
            monitor_events.append("terminal_persist")
            return {"passed": True}

        def commit_terminal_acceptance(self, pending):
            assert monitor_events == [
                "monitor_start",
                "segment_gate",
                "segment_start",
                "backend",
                "preflight_phase",
                "schedule",
                "semantic_verifier",
                "preterminal_receipt",
                "terminal_sample",
                "stop",
                "terminal_persist",
            ]
            monitor_events.append("success_commit")
            return pending

    def schedule(bound_config, backend):
        assert monitor_events == [
            "monitor_start",
            "segment_gate",
            "segment_start",
            "backend",
            "preflight_phase",
        ]
        monitor_events.append("schedule")
        monitor_events.append("semantic_verifier")
        monitor_events.append("preterminal_receipt")
        captured["schedule_config"] = bound_config
        captured["backend"] = backend
        return {
            "stage_root": str(tmp_path / FORMAL_RUN_ID / "s6"),
            "summary": {"state": "training"},
            "routing": {"route": "resume"},
        }

    monkeypatch.setattr(
        standard_training,
        "StandardProductionBackend",
        Backend,
        raising=False,
    )
    monkeypatch.setattr(standard_training, "run_standard_training_schedule", schedule)
    monkeypatch.setattr(
        stage6_workflow,
        "stage6_execution_identity",
        lambda **kwargs: dict(identity),
    )
    def public_verify(**kwargs):
        del kwargs
        backend = captured["backend"]
        assert backend.monitor.peak_rss_bytes == 16 * 1024**3
        assert backend.terminal_rss_bytes == 16 * 1024**3
        monitor_events.append("public_verified")
        return {"passed": True}

    monkeypatch.setattr(
        stage6_workflow,
        "verify_stage6_machine_acceptance",
        public_verify,
    )
    def require_preflight(**kwargs):
        captured["preflight"] = kwargs
        return {
            "environment_identity": identity["environment_identity"],
            "environment_sha256": identity["environment_sha256"],
        }

    monkeypatch.setattr(
        standard_training,
        "_require_machine_preflight",
        require_preflight,
        raising=False,
    )

    result = standard_training._execute_standard_training(
        config=config,
        run_root=tmp_path / FORMAL_RUN_ID,
        repo_root=repo_root,
        stage5_authority=dict(active["stage5_handle"].identity),
        verified_review_authorization=verified_review,
        execution_capability=active["capability"],
    )

    assert captured["schedule_config"] is config
    assert captured["backend"].__class__ is Backend
    source_repair = captured["backend_kwargs"]["_source_repair"]
    assert source_repair.sensor_acceleration_sha256 == "f" * 64
    expected_immutable = dict(source_repair.current_immutable_bindings)
    assert set(expected_immutable) == set(
        standard_training.STAGE6_IMMUTABLE_BINDING_KEYS
    )
    assert captured["backend_kwargs"] == {
        "config": config,
        "run_root": (tmp_path / FORMAL_RUN_ID).resolve(),
        "repo_root": repo_root.resolve(),
        "stage5_authority": dict(active["stage5_handle"].identity),
        "execution_capability": active["capability"],
        "_immutable_bindings": expected_immutable,
        "_source_repair": source_repair,
        "_process_tree_monitor": captured["backend_kwargs"][
            "_process_tree_monitor"
        ],
        "_resource_segment_start": captured["backend_kwargs"][
            "_resource_segment_start"
        ],
    }
    assert isinstance(captured["backend_kwargs"]["_process_tree_monitor"], Monitor)
    monitor = captured["backend_kwargs"]["_process_tree_monitor"]
    assert monitor.stop_calls == 2
    assert captured["backend"].terminal_rss_bytes == 16 * 1024**3
    assert monitor.peak_rss_bytes == 16 * 1024**3
    assert monitor_events == [
        "monitor_start",
        "segment_gate",
        "segment_start",
        "backend",
        "preflight_phase",
        "schedule",
        "semantic_verifier",
        "preterminal_receipt",
        "terminal_sample",
        "stop",
        "terminal_persist",
        "success_commit",
        "public_verified",
    ]
    assert captured["preflight"]["stage_root"] == (
        tmp_path / FORMAL_RUN_ID / "s6"
    ).resolve()
    assert captured["phase"] == "preflight"
    assert result["summary"]["state"] == "training"
    lineage = json.loads(
        (tmp_path / FORMAL_RUN_ID / "s6/lineage_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert lineage["verified_review_authorization"] == verified_review


def test_verified_review_identity_drift_is_rejected_before_training_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_execution_authority: dict[str, object],
) -> None:
    config = load_stage6_config(CONFIG)
    from lunar_exploration_ppo.utils import resources as resource_utils
    from lunar_exploration_ppo.workflows import stage6 as stage6_workflow

    repo_root = tmp_path / "repo"
    config_bytes = CONFIG.read_bytes()
    repo_root.joinpath("configs").mkdir(parents=True)
    repo_root.joinpath("configs/ppo_highres_frontier_stage6_v1.json").write_bytes(
        config_bytes
    )
    environment_identity = _stage6_environment_identity()
    reviewed_tree = "1" * 40
    identity = {
        "schema_version": "stage6_execution_identity/v1",
        "base_commit": stage6_workflow.STAGE5_COMMIT,
        "head_commit": stage6_workflow.STAGE5_COMMIT,
        "prospective_git_tree": reviewed_tree,
        "prospective_tree_sha256": sha256(reviewed_tree.encode("ascii")).hexdigest(),
        "changed_paths": sorted(stage6_workflow.STAGE6_SOURCE_PATHS),
        "changed_path_set_sha256": "1" * 64,
        "real_index_empty": True,
        "config_sha256": sha256(config_bytes).hexdigest(),
        "source_set_sha256": "2" * 64,
        "data_sha256": "3" * 64,
        "catalog_sha256": "4" * 64,
        "environment_identity": environment_identity,
        "environment_sha256": stage6_environment_sha256(environment_identity),
        "source_identity": {"schema_version": "test-source/v1"},
        "data_identity": {"schema_version": "test-data/v1"},
    }
    identity = dict(active_execution_authority["identity"])
    drifted_review = dict(active_execution_authority["verified"])
    drifted_review["source_set_sha256"] = "f" * 64
    repo_root = Path(active_execution_authority["repo_root"])
    initialization_events: list[str] = []

    monkeypatch.setattr(
        stage6_workflow,
        "stage6_execution_identity",
        lambda **kwargs: dict(identity),
    )

    class ForbiddenMonitor:
        def __init__(self):
            initialization_events.append("monitor")

    class ForbiddenBackend:
        def __init__(self, **kwargs):
            del kwargs
            initialization_events.append("backend")

    def forbidden_preflight(**kwargs):
        del kwargs
        initialization_events.append("preflight")
        raise AssertionError("review drift must precede preflight consumption")

    monkeypatch.setattr(resource_utils, "ProcessTreeRSSMonitor", ForbiddenMonitor)
    monkeypatch.setattr(standard_training, "StandardProductionBackend", ForbiddenBackend)
    monkeypatch.setattr(
        standard_training,
        "_require_machine_preflight",
        forbidden_preflight,
    )
    run_root = Path(active_execution_authority["run_root"])

    with pytest.raises(
        standard_training.StandardTrainingError,
        match="execution capability",
    ):
        standard_training._execute_standard_training(
            config=config,
            run_root=run_root,
            repo_root=repo_root,
            stage5_authority=dict(
                active_execution_authority["stage5_handle"].identity
            ),
            verified_review_authorization=drifted_review,
            execution_capability=active_execution_authority["capability"],
        )

    assert initialization_events == []
    assert not (run_root / "s6").exists()


@pytest.mark.parametrize(
    "drift_key",
    ("config_sha256", "source_set_sha256", "prospective_tree_sha256"),
)
def test_execute_resume_rejects_persisted_execution_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift_key: str,
    execution_authority_factory,
) -> None:
    config = load_stage6_config(CONFIG)
    _install_test_ordinal6_source_repair(monkeypatch)
    from lunar_exploration_ppo.workflows import stage6 as stage6_workflow

    environment_identity = _stage6_environment_identity()
    tree = "1" * 40
    identity = {
        "schema_version": "stage6_execution_identity/v1",
        "base_commit": stage6_workflow.STAGE5_COMMIT,
        "head_commit": stage6_workflow.STAGE5_COMMIT,
        "prospective_git_tree": tree,
        "prospective_tree_sha256": sha256(tree.encode("ascii")).hexdigest(),
        "changed_paths": list(stage6_workflow.STAGE6_SOURCE_PATHS),
        "changed_path_set_sha256": "1" * 64,
        "real_index_empty": True,
        "config_sha256": sha256(CONFIG.read_bytes()).hexdigest(),
        "source_set_sha256": "2" * 64,
        "data_sha256": "4" * 64,
        "catalog_sha256": "5" * 64,
        "environment_identity": environment_identity,
        "environment_sha256": stage6_environment_sha256(environment_identity),
        "source_identity": {"schema_version": "test-source/v1"},
        "data_identity": {"schema_version": "test-data/v1"},
    }
    active = execution_authority_factory(
        run_root=tmp_path / FORMAL_RUN_ID,
        environment_identity=environment_identity,
    )
    identity = dict(active["identity"])
    verified_review = dict(active["verified"])
    monkeypatch.setattr(
        stage6_workflow,
        "stage6_execution_identity",
        lambda **kwargs: dict(identity),
    )
    monkeypatch.setattr(
        standard_training,
        "_require_machine_preflight",
        lambda **kwargs: {
            "environment_identity": identity["environment_identity"],
            "environment_sha256": identity["environment_sha256"],
        },
        raising=False,
    )

    class Backend:
        def __init__(self, **kwargs):
            pass

        def record_preflight_phase(self):
            pass

        def capture_terminal_resource_sample(self):
            return {"passed": True}

        def record_terminal_resource_evidence(self):
            return {"passed": True}

        def commit_terminal_acceptance(self, pending):
            return pending

    monkeypatch.setattr(standard_training, "StandardProductionBackend", Backend)
    monkeypatch.setattr(
        standard_training,
        "run_standard_training_schedule",
            lambda config, backend: {
            "stage_root": str(tmp_path / FORMAL_RUN_ID / "s6"),
            "summary": {"state": "training"},
            "routing": {"route": "resume"},
        },
    )
    monkeypatch.setattr(
        stage6_workflow,
        "rebind_stage6_manifest_for_terminal_resource",
        lambda **kwargs: tmp_path / "run/s6/manifest.json",
    )
    monkeypatch.setattr(
        stage6_workflow,
        "verify_stage6_machine_acceptance",
        lambda **kwargs: {"passed": True},
    )
    arguments = {
        "config": config,
        "run_root": tmp_path / FORMAL_RUN_ID,
        "repo_root": ROOT,
        "stage5_authority": dict(active["stage5_handle"].identity),
        "verified_review_authorization": verified_review,
        "execution_capability": active["capability"],
    }

    standard_training._execute_standard_training(**arguments)
    if drift_key == "config_sha256":
        persisted_config_path = tmp_path / FORMAL_RUN_ID / "s6/config.json"
        persisted_config = json.loads(
            persisted_config_path.read_bytes().decode("utf-8")
        )
        assert persisted_config["training"]["seeds"] == [20260716]
        persisted_config["training"]["seeds"] = [
            20260716,
            20260717,
            20260718,
            20260719,
            20260720,
        ]
        persisted_config_path.write_bytes(
            standard_training.ArtifactStore.canonical_json_bytes(persisted_config)
        )
    else:
        identity[drift_key] = "9" * 64
    with pytest.raises(standard_training.StandardTrainingError) as error:
        standard_training._execute_standard_training(**arguments)
    expected_error = (
        "Stage 6 persisted execution identity drifted"
        if drift_key == "config_sha256"
        else (
            "Stage 6 execution capability rejected at "
            "_execute_standard_training after execution identity recomputation"
        )
    )
    assert str(error.value) == expected_error


def test_single_seed_training_schedule_executes_resume_and_complete_final_eval() -> None:
    config = load_stage6_config(CONFIG)

    class Backend:
        def __init__(self) -> None:
            self.events: list[tuple[object, ...]] = []
            self.runtime_tokens: list[str] = []
            self.validation_count = 0
            self.checkpoint_writes = 0
            self.final_count = 0
            self.previous_policy_ref = None
            self.previous_optimizer_ref = None

        class RuntimeObject:
            pass

        def prepare_seed(self, seed, transactions):
            gc.collect()
            if self.previous_policy_ref is not None:
                assert self.previous_policy_ref() is None
                assert self.previous_optimizer_ref() is None
            policy = self.RuntimeObject()
            optimizer = self.RuntimeObject()
            runtime_token = f"seed-{seed}"
            self.runtime_tokens.append(runtime_token)
            self.events.append(("prepare", seed))
            remaining = transactions[2:] if seed == 20260716 else transactions
            resumed = seed == 20260716
            return {
                "seed": seed,
                "policy": policy,
                "optimizer": optimizer,
                "initial_policy_sha256": config.stage5_authority.policy_state_sha256,
                "optimizer_fresh": not resumed,
                "independent_seed_origin": True,
                "continue_from_current_state": resumed,
                "sampler_seed_derivation_version": "stage6_seed_times_16_plus_lane/v1",
                "sampler_seeds": standard_training.derive_standard_sampler_seeds(seed),
                "runtime_token": runtime_token,
            }, remaining

        def run_update(self, runtime, transaction):
            assert runtime["seed"] == transaction.seed
            self.events.append(("update", transaction.seed, transaction.update))
            self.checkpoint_writes += 1
            if transaction.validation_episodes:
                self.validation_count += 1

        def finish_seed(self, runtime):
            self.events.append(("finish", runtime["seed"]))
            self.previous_policy_ref = weakref.ref(runtime["policy"])
            self.previous_optimizer_ref = weakref.ref(runtime["optimizer"])
            return standard_training.ValidationRecord(
                runtime["seed"], 100, 0.5, 0.6, "update-00000100"
            )

        def freeze_global_best(self, record):
            self.events.append(("global", record.seed))
            return record

        def checkpoint_write_count(self):
            return self.checkpoint_writes

        def run_final_evaluation(self, record, split, method):
            self.final_count += 1
            self.events.append(("final", split, method))
            return {"episode_count": 64}

        def finalize(self, record, evaluations):
            return {"global_best": record.to_dict(), "evaluation_count": len(evaluations)}

    backend = Backend()

    result = standard_training.run_standard_training_schedule(config, backend)

    update_events = [event for event in backend.events if event[0] == "update"]
    assert len(update_events) == 98
    assert update_events[0] == ("update", 20260716, 3)
    assert backend.validation_count == 10
    assert len(backend.runtime_tokens) == 1
    assert len(set(backend.runtime_tokens)) == 1
    assert backend.final_count == 10
    assert backend.checkpoint_writes == 98
    assert result["evaluation_count"] == 10
    assert [event for event in backend.events if event[0] == "final"] == [
        ("final", "test", "ppo_policy"),
        ("final", "unseen", "ppo_policy"),
        ("final", "test", "random_valid_frontier"),
        ("final", "unseen", "random_valid_frontier"),
        ("final", "test", "nearest_frontier"),
        ("final", "unseen", "nearest_frontier"),
        ("final", "test", "max_potential_gain_frontier"),
        ("final", "unseen", "max_potential_gain_frontier"),
        ("final", "test", "gain_over_cost_frontier"),
        ("final", "unseen", "gain_over_cost_frontier"),
    ]


def test_schedule_closes_vector_env_when_update_raises() -> None:
    config = load_stage6_config(CONFIG)
    closed: list[bool] = []

    class Backend:
        def prepare_seed(self, seed, transactions):
            return {
                "seed": seed,
                "policy": object(),
                "optimizer": object(),
                "initial_policy_sha256": config.stage5_authority.policy_state_sha256,
                "optimizer_fresh": True,
                "independent_seed_origin": True,
                "continue_from_current_state": False,
                "sampler_seed_derivation_version": "stage6_seed_times_16_plus_lane/v1",
                "sampler_seeds": standard_training.derive_standard_sampler_seeds(seed),
                "runtime_token": f"seed-{seed}",
                "vector_env": SimpleNamespace(
                    closed=False,
                    close=lambda: closed.append(True),
                ),
            }, transactions

        def run_update(self, runtime, transaction):
            raise RuntimeError("injected update failure")

        def finish_seed(self, runtime):
            raise AssertionError("unreachable")

        freeze_global_best = checkpoint_write_count = run_final_evaluation = finalize = (
            lambda *args, **kwargs: None
        )

    with pytest.raises(RuntimeError, match="injected"):
        standard_training.run_standard_training_schedule(config, Backend())
    assert closed == [True]


def test_single_seed_sampler_lane_derivation_is_stable_and_unique() -> None:
    config = load_stage6_config(CONFIG)
    derived = [
        standard_training.derive_standard_sampler_seeds(seed)
        for seed in config.training.seeds
    ]
    assert all(len(value) == 8 for value in derived)
    assert len({seed for values in derived for seed in values}) == 8
    assert derived[0] == tuple(config.training.seeds[0] * 16 + lane for lane in range(8))


def test_validation_trace_publication_fails_closed_when_pending_is_replaced(
    active_execution_authority: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_security = importlib.import_module(
        "lunar_exploration_ppo.utils.path_security"
    )
    backend = _validation_trace_backend(active_execution_authority)
    pending = backend.run_root / (
        "stage6-attempts/validation-seed-20260716-update-010-"
        "attempt-001.pending.jsonl"
    )
    canonical = backend.stage_root / (
        "episode-traces/validation-seed-20260716-update-010.jsonl"
    )
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending_payload = _validation_trace_payload(attempt=1)
    attacker_payload = _validation_trace_payload(attempt=9)
    pending.write_bytes(pending_payload)
    expected_binding = _validation_trace_binding(pending_payload)
    calls: list[str] = []

    def replacing_secure_read(event: str, path: Path, descriptor: int | None = None):
        del descriptor
        if Path(path) == pending:
            calls.append(event)
            if event == "after_path_identity":
                replacement = pending.with_suffix(".attacker")
                replacement.write_bytes(attacker_payload)
                os.replace(replacement, pending)

    monkeypatch.setattr(
        path_security,
        "_secure_read_event",
        replacing_secure_read,
        raising=False,
    )

    with pytest.raises(
        standard_training.StandardTrainingError,
        match="validation trace publication",
    ):
        backend._publish_validation_trace(
            pending_path=pending,
            canonical_path=canonical,
            episode_count=16,
            expected_binding=expected_binding,
        )

    assert calls
    assert not canonical.exists()
    assert pending.read_bytes() == attacker_payload


def test_validation_trace_publication_fails_closed_for_substituted_canonical_parent(
    active_execution_authority: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _validation_trace_backend(active_execution_authority)
    pending = backend.run_root / (
        "stage6-attempts/validation-seed-20260716-update-010-"
        "attempt-001.pending.jsonl"
    )
    canonical = backend.stage_root / (
        "episode-traces/validation-seed-20260716-update-010.jsonl"
    )
    pending.parent.mkdir(parents=True, exist_ok=True)
    payload = _validation_trace_payload(attempt=1)
    pending.write_bytes(payload)
    expected_binding = _validation_trace_binding(payload)
    original_require_plain_path = standard_training.require_plain_path

    def substituted_parent(path, **kwargs):
        if kwargs.get("label") == "validation trace canonical parent":
            raise standard_training.PathSecurityError(
                "injected canonical parent substitution"
            )
        return original_require_plain_path(path, **kwargs)

    monkeypatch.setattr(
        standard_training,
        "require_plain_path",
        substituted_parent,
    )

    with pytest.raises(
        standard_training.StandardTrainingError,
        match="validation trace publication",
    ):
        backend._publish_validation_trace(
            pending_path=pending,
            canonical_path=canonical,
            episode_count=16,
            expected_binding=expected_binding,
        )

    assert pending.is_file()
    assert not canonical.exists()


@pytest.mark.skipif(
    os.name != "nt",
    reason="canonical reparse substitution race coverage is Windows-specific",
)
def test_validation_trace_publication_never_writes_through_a_substituted_parent(
    active_execution_authority: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _validation_trace_backend(active_execution_authority)
    pending = backend.run_root / (
        "stage6-attempts/validation-seed-20260716-update-010-"
        "attempt-001.pending.jsonl"
    )
    canonical = backend.stage_root / (
        "episode-traces/validation-seed-20260716-update-010.jsonl"
    )
    pending.parent.mkdir(parents=True, exist_ok=True)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    pending_payload = _validation_trace_payload(attempt=1)
    pending.write_bytes(pending_payload)
    expected_binding = _validation_trace_binding(pending_payload)
    external_parent = backend.run_root / "attacker-traces"
    external_parent.mkdir(parents=True, exist_ok=True)
    displaced_parent = canonical.parent.with_name("episode-traces-displaced")
    real_store = standard_training.ArtifactStore
    substituted = False

    class ParentSubstitutingArtifactStore:
        def __init__(self, root: Path) -> None:
            nonlocal substituted
            if not substituted:
                os.replace(canonical.parent, displaced_parent)
                _make_stage6_directory_reparse(canonical.parent, external_parent)
                substituted = True
            self._store = real_store(root)

        def write_bytes_exclusive(self, relative_path: str | Path, payload: bytes) -> Path:
            return self._store.write_bytes_exclusive(relative_path, payload)

    monkeypatch.setattr(
        standard_training,
        "ArtifactStore",
        ParentSubstitutingArtifactStore,
    )

    with pytest.raises(
        standard_training.StandardTrainingError,
        match="validation trace publication",
    ):
        backend._publish_validation_trace(
            pending_path=pending,
            canonical_path=canonical,
            episode_count=16,
            expected_binding=expected_binding,
        )

    assert substituted
    assert not (external_parent / canonical.name).exists()
    assert pending.read_bytes() == pending_payload


def test_validation_trace_recovery_rejects_canonical_hardlink_binding_drift(
    active_execution_authority: dict[str, object],
) -> None:
    config = load_stage6_config(CONFIG)
    transaction = standard_training.build_standard_training_transactions(config)[9]
    backend = _validation_trace_backend(active_execution_authority)
    canonical = backend.stage_root / (
        "episode-traces/validation-seed-20260716-update-010.jsonl"
    )
    canonical.parent.mkdir(parents=True, exist_ok=True)
    source = backend.run_root / "attacker-traces/validation-trace.jsonl"
    source.parent.mkdir(parents=True, exist_ok=True)
    payload = _validation_trace_payload(attempt=2)
    source.write_bytes(payload)
    os.link(source, canonical)
    loaded = SimpleNamespace(
        eval_metrics={
            "validation": {
                "episode_count": 16,
                "success_rate_under_fixed_step_budget": 0.7,
                "mean_final_coverage": 0.4,
                "validation_trace_binding": _validation_trace_binding(payload),
            }
        }
    )

    with pytest.raises(
        standard_training.StandardTrainingError,
        match="validation trace.*binding",
    ):
        backend._recover_validation_trace(transaction, loaded=loaded)


@pytest.mark.skipif(
    os.name != "nt",
    reason="identity-bound durable unlink is currently Windows-only",
)
def test_validation_trace_publication_is_bit_exact_and_unlinks_bound_pending(
    active_execution_authority: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_security = importlib.import_module(
        "lunar_exploration_ppo.utils.path_security"
    )
    backend = _validation_trace_backend(active_execution_authority)
    pending = backend.run_root / (
        "stage6-attempts/validation-seed-20260716-update-010-"
        "attempt-001.pending.jsonl"
    )
    canonical = backend.stage_root / (
        "episode-traces/validation-seed-20260716-update-010.jsonl"
    )
    pending.parent.mkdir(parents=True, exist_ok=True)
    payload = _validation_trace_payload(attempt=1)
    pending.write_bytes(payload)
    expected_identity = path_security.durable_file_identity(pending)
    real_durable_unlink = standard_training.durable_unlink
    observed_identities: list[tuple[Path, object]] = []

    def observe_identity_bound_unlink(path, *, expected_identity):
        observed_identities.append((Path(path), expected_identity))
        assert expected_identity == path_security.durable_file_identity(path)
        real_durable_unlink(path, expected_identity=expected_identity)

    monkeypatch.setattr(
        standard_training,
        "durable_unlink",
        observe_identity_bound_unlink,
    )

    backend._publish_validation_trace(
        pending_path=pending,
        canonical_path=canonical,
        episode_count=16,
        expected_binding=_validation_trace_binding(payload),
    )

    assert canonical.read_bytes() == payload
    assert observed_identities == [(pending, expected_identity)]
    assert not pending.exists()


def test_validation_trace_recovery_rejects_canonical_binding_mismatch(
    active_execution_authority: dict[str, object],
) -> None:
    config = load_stage6_config(CONFIG)
    transaction = standard_training.build_standard_training_transactions(config)[9]
    backend = _validation_trace_backend(active_execution_authority)
    canonical = backend.stage_root / (
        "episode-traces/validation-seed-20260716-update-010.jsonl"
    )
    canonical.parent.mkdir(parents=True, exist_ok=True)
    good_payload = _validation_trace_payload(attempt=2)
    canonical.write_bytes(_validation_trace_payload(attempt=7))
    loaded = SimpleNamespace(
        eval_metrics={
            "validation": {
                "episode_count": 16,
                "success_rate_under_fixed_step_budget": 0.7,
                "mean_final_coverage": 0.4,
                "validation_trace_binding": _validation_trace_binding(good_payload),
            }
        }
    )

    with pytest.raises(
        standard_training.StandardTrainingError,
        match="validation trace.*binding",
    ):
        backend._recover_validation_trace(transaction, loaded=loaded)


@pytest.mark.skipif(
    os.name != "nt",
    reason="identity-bound durable unlink is currently Windows-only",
)
def test_validation_trace_recovery_cleans_max_attempt_pending_after_canonical_publish(
    active_execution_authority: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_security = importlib.import_module(
        "lunar_exploration_ppo.utils.path_security"
    )
    config = load_stage6_config(CONFIG)
    transaction = standard_training.build_standard_training_transactions(config)[9]
    backend = _validation_trace_backend(active_execution_authority)
    _attach_backend_resource_segment(backend)
    _append_validation_trace_pre_attempt(backend, transaction, attempt=1)
    _append_validation_trace_pre_attempt(backend, transaction, attempt=2)
    canonical = backend.stage_root / (
        "episode-traces/validation-seed-20260716-update-010.jsonl"
    )
    lower_pending = backend.run_root / (
        "stage6-attempts/validation-seed-20260716-update-010-"
        "attempt-001.pending.jsonl"
    )
    max_pending = backend.run_root / (
        "stage6-attempts/validation-seed-20260716-update-010-"
        "attempt-002.pending.jsonl"
    )
    payload = _validation_trace_payload(attempt=2)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(payload)
    lower_pending.write_bytes(_validation_trace_payload(attempt=1))
    max_pending.write_bytes(payload)
    expected_identity = path_security.durable_file_identity(max_pending)
    observed_reads: list[Path] = []
    observed_unlinks: list[tuple[Path, object]] = []
    real_secure_read = standard_training.secure_read_bytes
    real_durable_unlink = standard_training.durable_unlink

    def observe_secure_read(path, **kwargs):
        observed_reads.append(Path(path))
        return real_secure_read(path, **kwargs)

    def observe_identity_bound_unlink(path, *, expected_identity):
        observed_unlinks.append((Path(path), expected_identity))
        assert expected_identity == path_security.durable_file_identity(path)
        real_durable_unlink(path, expected_identity=expected_identity)

    monkeypatch.setattr(standard_training, "secure_read_bytes", observe_secure_read)
    monkeypatch.setattr(
        standard_training,
        "durable_unlink",
        observe_identity_bound_unlink,
    )
    loaded = SimpleNamespace(
        eval_metrics={
            "validation": {
                "episode_count": 16,
                "success_rate_under_fixed_step_budget": 0.7,
                "mean_final_coverage": 0.4,
                "validation_trace_binding": _validation_trace_binding(payload),
            }
        }
    )

    backend._recover_validation_trace(transaction, loaded=loaded)

    assert canonical.read_bytes() == payload
    assert max_pending in observed_reads
    assert observed_unlinks == [(max_pending, expected_identity)]
    assert not max_pending.exists()
    assert lower_pending.read_bytes() == _validation_trace_payload(attempt=1)


def test_validation_trace_recovery_rejects_conflicting_max_attempt_pending_after_canonical_publish(
    active_execution_authority: dict[str, object],
) -> None:
    config = load_stage6_config(CONFIG)
    transaction = standard_training.build_standard_training_transactions(config)[9]
    backend = _validation_trace_backend(active_execution_authority)
    _attach_backend_resource_segment(backend)
    _append_validation_trace_pre_attempt(backend, transaction, attempt=1)
    _append_validation_trace_pre_attempt(backend, transaction, attempt=2)
    canonical = backend.stage_root / (
        "episode-traces/validation-seed-20260716-update-010.jsonl"
    )
    max_pending = backend.run_root / (
        "stage6-attempts/validation-seed-20260716-update-010-"
        "attempt-002.pending.jsonl"
    )
    canonical_payload = _validation_trace_payload(attempt=2)
    conflicting_payload = _validation_trace_payload(attempt=7)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(canonical_payload)
    max_pending.write_bytes(conflicting_payload)
    loaded = SimpleNamespace(
        eval_metrics={
            "validation": {
                "episode_count": 16,
                "success_rate_under_fixed_step_budget": 0.7,
                "mean_final_coverage": 0.4,
                "validation_trace_binding": _validation_trace_binding(
                    canonical_payload
                ),
            }
        }
    )

    with pytest.raises(
        standard_training.StandardTrainingError,
        match="validation trace.*binding",
    ):
        backend._recover_validation_trace(transaction, loaded=loaded)

    assert canonical.read_bytes() == canonical_payload
    assert max_pending.read_bytes() == conflicting_payload


def test_validation_trace_recovery_rejects_hardlinked_max_attempt_pending_after_canonical_publish(
    active_execution_authority: dict[str, object],
) -> None:
    config = load_stage6_config(CONFIG)
    transaction = standard_training.build_standard_training_transactions(config)[9]
    backend = _validation_trace_backend(active_execution_authority)
    _attach_backend_resource_segment(backend)
    _append_validation_trace_pre_attempt(backend, transaction, attempt=2)
    canonical = backend.stage_root / (
        "episode-traces/validation-seed-20260716-update-010.jsonl"
    )
    source = backend.run_root / "attacker-traces/validation-trace.jsonl"
    max_pending = backend.run_root / (
        "stage6-attempts/validation-seed-20260716-update-010-"
        "attempt-002.pending.jsonl"
    )
    payload = _validation_trace_payload(attempt=2)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    source.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(payload)
    source.write_bytes(payload)
    os.link(source, max_pending)
    loaded = SimpleNamespace(
        eval_metrics={
            "validation": {
                "episode_count": 16,
                "success_rate_under_fixed_step_budget": 0.7,
                "mean_final_coverage": 0.4,
                "validation_trace_binding": _validation_trace_binding(payload),
            }
        }
    )

    with pytest.raises(
        standard_training.StandardTrainingError,
        match="validation trace recovery binding drifted",
    ):
        backend._recover_validation_trace(transaction, loaded=loaded)

    assert canonical.read_bytes() == payload
    assert max_pending.read_bytes() == payload


@pytest.mark.skipif(
    os.name != "nt",
    reason="identity-bound durable publication/unlink is currently Windows-only",
)
def test_validation_trace_publication_and_checkpoint_ahead_recovery_are_binding_exact(
    tmp_path: Path,
    execution_authority_factory,
) -> None:
    config = load_stage6_config(CONFIG)
    transaction = standard_training.build_standard_training_transactions(config)[9]
    publish_active = execution_authority_factory(
        run_root=tmp_path / "publish" / FORMAL_RUN_ID
    )
    backend = _validation_trace_backend(publish_active)
    _attach_backend_resource_segment(backend)
    pending = backend.run_root / (
        "stage6-attempts/validation-seed-20260716-update-010-"
        "attempt-001.pending.jsonl"
    )
    canonical = backend.stage_root / (
        "episode-traces/validation-seed-20260716-update-010.jsonl"
    )
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending_payload = _validation_trace_payload(attempt=1)
    pending.write_bytes(pending_payload)
    binding = _validation_trace_binding(pending_payload)

    backend._publish_validation_trace(
        pending_path=pending,
        canonical_path=canonical,
        episode_count=16,
        expected_binding=binding,
    )

    assert canonical.read_bytes() == pending_payload
    assert not pending.exists()

    recovery_active = execution_authority_factory(
        run_root=tmp_path / "recover" / FORMAL_RUN_ID
    )
    recovered = _validation_trace_backend(recovery_active)
    _attach_backend_resource_segment(recovered)
    recovery_pending = recovered.run_root / (
        "stage6-attempts/validation-seed-20260716-update-010-"
        "attempt-002.pending.jsonl"
    )
    recovery_canonical = recovered.stage_root / (
        "episode-traces/validation-seed-20260716-update-010.jsonl"
    )
    recovery_payload = _validation_trace_payload(attempt=2)
    recovery_pending.parent.mkdir(parents=True, exist_ok=True)
    recovery_pending.write_bytes(recovery_payload)
    recovery_identity = {
        "kind": "update",
        "transaction_key": transaction.key,
        "seed": transaction.seed,
        "update": transaction.update,
    }
    recovery_resource = {
        "d_free_bytes": 200 * 1024**3,
        "rss_bytes": 1024,
        "peak_vram_bytes": 2048,
        "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
        "rss_root_pid": os.getpid(),
        "rss_sample_count": 2,
        "rss_latest_process_count": 1,
        "rss_peak_process_count": 1,
        "warnings": [],
        "hard_stops": [],
        "passed": True,
    }
    recovered._append_resource_phase(
        identity=recovery_identity,
        attempt=2,
        phase="pre",
        resource=recovery_resource,
    )
    loaded = SimpleNamespace(
        eval_metrics={
            "validation": {
                "episode_count": 16,
                "success_rate_under_fixed_step_budget": 0.7,
                "mean_final_coverage": 0.4,
                "validation_trace_binding": _validation_trace_binding(
                    recovery_payload
                ),
            }
        }
    )

    recovered._recover_validation_trace(transaction, loaded=loaded)

    assert recovery_canonical.read_bytes() == recovery_payload
    assert not recovery_pending.exists()


def test_production_backend_prepare_seed_uses_fresh_runtime_and_restores_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_authority_factory,
) -> None:
    config = load_stage6_config(CONFIG)
    transactions = standard_training.build_standard_training_transactions(config)
    seed_transactions = transactions[:100]
    events: list[tuple[object, ...]] = []
    vectors: list[object] = []
    recovery_prefix_root: Path | None = None
    recovery_prefix_checkpoint: standard_training.StandardTransactionCheckpoint | None = None

    class Vector:
        def __init__(self, specs, *, timeout_seconds):
            vectors.append(self)
            events.append(("vector", tuple(specs), timeout_seconds))
            self.restored = None
            self.closed = False

        def restore_states(self, states):
            self.restored = tuple(states)
            events.append(("restore_states", len(self.restored)))

        def capture_states(self):
            return self.restored

        def current_observations(self):
            return {
                index: SimpleNamespace(observation=f"resume-observation-{index}")
                for index in range(8)
            }

        def close(self):
            self.closed = True

    class Trainer:
        def __init__(self, policy, *, device, optimizer, shuffle_seed):
            self.policy = policy
            self.device = device
            self.optimizer = optimizer
            self.shuffle_seed = shuffle_seed
            self.update_step = 0
            events.append(("trainer", shuffle_seed, id(optimizer)))

        def restore_update_step(self, update_step):
            self.update_step = update_step
            events.append(("restore_update", update_step))

    class Collector:
        def __init__(self, **kwargs):
            self.vector_env = kwargs["vector_env"]
            events.append(("collector", kwargs["device"]))

    class Contract:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    loaded_by_root: dict[Path, object] = {}

    class Manager:
        def __init__(self, root, *, schema_version):
            self.root = Path(root).resolve()
            self.root.mkdir(parents=True, exist_ok=True)
            events.append(("manager", self.root, schema_version))

        def load_last_complete(self, **kwargs):
            assert kwargs["expected_safety_contract"] == (
                SafetyContract.from_stage6_config(config)
            )
            events.append(("load_last_complete", self.root))
            return loaded_by_root[self.root]

        def recover_accepted_checkpoint(self, **kwargs):
            loaded = loaded_by_root[self.root]
            canonical = self.root / f"update-{loaded.update_step:08d}"
            if canonical.is_dir():
                complete_path = canonical / "complete.json"
            else:
                pending = tuple(
                    path
                    for path in self.root.iterdir()
                    if path.name.startswith(
                        f".pending-update-{loaded.update_step:08d}-"
                    )
                )
                assert len(pending) == 1
                complete_path = pending[0] / "complete.json"
            events.append(("recover_accepted_checkpoint", self.root, dict(kwargs)))
            assert kwargs == {
                "update_step": loaded.update_step,
                "checkpoint_sha256": loaded.checkpoint_sha256,
                "complete_marker_sha256": sha256(complete_path.read_bytes()).hexdigest(),
                "policy_state_sha256": loaded.policy_state_sha256,
            }
            if not canonical.is_dir():
                os.replace(complete_path.parent, canonical)
            return SimpleNamespace(
                update_step=loaded.update_step,
                directory=canonical,
                checkpoint_sha256=loaded.checkpoint_sha256,
                policy_state_sha256=loaded.policy_state_sha256,
            )

    class Retention:
        def __init__(self, root, *, schema_version):
            self.root = Path(root).resolve()

        def apply(self, **kwargs):
            events.append(("retention", self.root, dict(kwargs)))
            return SimpleNamespace(kept_updates=(), removed_updates=())

    stage4_checkpoint = tmp_path / "stage4/checkpoint.pt"
    stage4_checkpoint.parent.mkdir(parents=True)
    stage4_checkpoint.write_bytes(b"stage4")

    def load_policy(**kwargs):
        if recovery_prefix_root is not None:
            assert recovery_prefix_checkpoint is not None
            recovered_receipts = standard_training.CheckpointReceiptIndex(
                recovery_prefix_root / "s6/checkpoints/index.jsonl"
            ).verify()
            assert len(recovered_receipts) == 1
            assert recovered_receipts[0]["transaction_key"] == (
                recovery_prefix_checkpoint.transaction_key
            )
            assert recovered_receipts[0]["checkpoint_sha256"] == (
                recovery_prefix_checkpoint.checkpoint_sha256
            )
            assert [
                row["state"]
                for row in Stage6StateJournal(
                    recovery_prefix_root / "s6/job-state.jsonl"
                ).verify()
            ] == list(transactions[0].commit_states)
        events.append(("load_policy", kwargs["device"], kwargs["checkpoint_path"]))
        return nn.Linear(3, 2)

    def env_specs(
        catalog,
        *,
        split,
        sampler_seeds,
        safety_contract,
        config_sha256,
    ):
        from lunar_exploration_ppo.ppo.collector import SpawnEnvSpec

        assert safety_contract == SafetyContract.from_stage6_config(config)
        assert config_sha256 == immutable["config_sha256"]
        events.append(("env_specs", catalog, split, sampler_seeds))
        return tuple(
            SpawnEnvSpec(
                factory=_test_spawn_env_factory,
                kwargs={"worker_index": index},
            )
            for index in range(8)
        )

    components = {
        "build_standard_catalog": lambda **kwargs: "catalog",
        "load_stage4_policy_for_standard": load_policy,
        "stage4_checkpoint_path": stage4_checkpoint,
        "standard_env_specs": env_specs,
        "SpawnVectorEnv": Vector,
        "RolloutCollector": Collector,
        "CollectorContract": Contract,
        "PPOTrainer": Trainer,
        "CheckpointManager": Manager,
        "CheckpointRetentionManager": Retention,
        "CheckpointReceiptIndex": standard_training.CheckpointReceiptIndex,
        "Stage6StateJournal": Stage6StateJournal,
        "capture_resource_snapshot": lambda **kwargs: SimpleNamespace(
            d_free_bytes=200 * 1024**3,
            rss_bytes=1024,
            peak_vram_bytes=2048,
            rss_source="process_tree_lifecycle_peak_current_sum/v1",
            rss_root_pid=os.getpid(),
            rss_sample_count=2,
            rss_latest_process_count=1,
            rss_peak_process_count=1,
        ),
        "evaluate_resource_gates": lambda snapshot, **kwargs: SimpleNamespace(
            passed=True,
            warnings=(),
            hard_stops=(),
        ),
        "deterministic_action_record": (
            lambda policy, observation, *, device: {
                "observation": observation,
                "device": device,
            }
        ),
    }
    immutable = {
        key: value
        for key, value in _journal_bindings("0" * 64).items()
        if key != "checkpoint_sha256"
    }
    immutable["config_sha256"] = sha256(
        standard_training.ArtifactStore.canonical_json_bytes(
            config.model_dump(mode="json")
        )
    ).hexdigest()
    source_repair = _install_test_ordinal6_source_repair(monkeypatch)(
        config=config,
        current_immutable_bindings=immutable,
    )

    fresh_root = tmp_path / "fresh" / FORMAL_RUN_ID
    fresh_active = execution_authority_factory(run_root=fresh_root)
    fresh = standard_training.StandardProductionBackend(
        config=config,
        run_root=fresh_root,
        repo_root=ROOT,
        stage5_authority=dict(fresh_active["stage5_handle"].identity),
        execution_capability=fresh_active["capability"],
        _components=components,
        _immutable_bindings=immutable,
        _source_repair=source_repair,
    )
    fresh_runtime, fresh_remaining = fresh.prepare_seed(
        config.training.seeds[0], seed_transactions
    )
    assert fresh_remaining == seed_transactions
    assert isinstance(fresh_runtime["optimizer"], torch.optim.AdamW)
    assert fresh_runtime["optimizer_fresh"] is True
    assert fresh_runtime["independent_seed_origin"] is True
    assert fresh_runtime["rng_initialization"] == "fresh_seed_all_rng/v1"
    assert fresh_runtime["sampler_seed_derivation_version"] == (
        "stage6_seed_times_16_plus_lane/v1"
    )
    assert fresh_runtime["sampler_seeds"] == (
        standard_training.derive_standard_sampler_seeds(config.training.seeds[0])
    )
    assert fresh_runtime["trainer"].update_step == 0
    assert fresh_runtime["checkpoint_root"] == (
        fresh_root / "s6/checkpoints/seed-20260716"
    ).resolve()
    assert fresh_runtime["runtime_token"] == "stage6-seed-20260716"
    fresh_runtime["vector_env"].close()

    resume_root = tmp_path / "resume" / FORMAL_RUN_ID
    checkpoint_root = (
        resume_root / "s6/checkpoints/seed-20260716"
    ).resolve()
    checkpoint_root.mkdir(parents=True)
    complete_directory = checkpoint_root / "update-00000001"
    complete_directory.mkdir()
    (complete_directory / "checkpoint.pt").write_bytes(b"checkpoint")
    (complete_directory / "manifest.json").write_bytes(b"manifest")
    (complete_directory / "complete.json").write_bytes(b"complete")
    checkpoint = standard_training.StandardTransactionCheckpoint(
        transaction_key=transactions[0].key,
        seed=transactions[0].seed,
        update=transactions[0].update,
        checkpoint_sha256="a" * 64,
        complete_marker_sha256=sha256(b"complete").hexdigest(),
        policy_state_sha256="c" * 64,
    )
    receipt_index = standard_training.CheckpointReceiptIndex(
        resume_root / "s6/checkpoints/index.jsonl"
    )
    receipt_index.append_once(checkpoint)
    journal = Stage6StateJournal(resume_root / "s6/job-state.jsonl")
    bindings = {**immutable, "checkpoint_sha256": checkpoint.checkpoint_sha256}
    for state in transactions[0].commit_states[:1]:
        journal.append(state, bindings)
    vector_states = tuple(
        {"episode_state": {"worker": index}, "sampler_state": {"cursor": index}}
        for index in range(8)
    )
    resume_collection_audit = _collector_audit_dict(
        config.stage5_authority.policy_state_sha256,
        prefix="resume-update-1",
    )
    loaded_by_root[checkpoint_root] = SimpleNamespace(
        update_step=1,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        policy_state_sha256=checkpoint.policy_state_sha256,
        vector_env_states=vector_states,
        normalization_stats={},
        best_record={},
        eval_metrics={
            "update": {
                "update_step": 1,
                "initial_ratio_max_abs_error": 1.0e-6,
                "policy_loss": 0.1,
                "value_loss": 0.2,
                "approx_kl": 0.003,
                "grad_pre_clip_norm_max": 0.7,
                "grad_post_clip_norm_max": 0.5,
                "optimizer_steps": 4,
                "policy_state_sha256_before": (
                    config.stage5_authority.policy_state_sha256
                ),
                "policy_state_sha256_after": checkpoint.policy_state_sha256,
                "math_evidence": _math_evidence_dict(
                    resume_collection_audit,
                    policy_sha256=config.stage5_authority.policy_state_sha256,
                ),
            },
            "validation": {},
            "eval_isolation": {},
            "collection_audit": resume_collection_audit,
        },
    )
    resumed_active = execution_authority_factory(run_root=resume_root)
    resumed = standard_training.StandardProductionBackend(
        config=config,
        run_root=resume_root,
        repo_root=ROOT,
        stage5_authority=dict(resumed_active["stage5_handle"].identity),
        execution_capability=resumed_active["capability"],
        _components=components,
        _immutable_bindings=immutable,
        _source_repair=source_repair,
    )
    _attach_backend_resource_segment(resumed)
    loaded_by_root[checkpoint_root].config_sha256 = resumed.config_sha256
    loaded_by_root[checkpoint_root].lineage = resumed._checkpoint_lineage(1)
    loaded_by_root[checkpoint_root].scenario_sampler_state = {
        "workers": [state["sampler_state"] for state in vector_states]
    }
    resume_resource_identity = {
        "kind": "update",
        "transaction_key": transactions[0].key,
        "seed": transactions[0].seed,
        "update": transactions[0].update,
    }
    resume_resource = {
        "d_free_bytes": 200 * 1024**3,
        "rss_bytes": 1024,
        "peak_vram_bytes": 2048,
        "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
        "rss_root_pid": os.getpid(),
        "rss_sample_count": 2,
        "rss_latest_process_count": 1,
        "rss_peak_process_count": 1,
        "warnings": [],
        "hard_stops": [],
        "passed": True,
    }
    for phase in ("pre", "post"):
        resumed._append_resource_phase(
            identity=resume_resource_identity,
            attempt=1,
            phase=phase,
            resource=resume_resource,
        )
    resumed._append_resource_acceptance(
        identity=resume_resource_identity,
        attempt=1,
        pre=resume_resource,
        post=resume_resource,
        checkpoint=checkpoint,
    )
    resumed_runtime, resumed_remaining = resumed.prepare_seed(
        config.training.seeds[0], seed_transactions
    )
    assert resumed_remaining == seed_transactions[1:]
    assert resumed_runtime["trainer"].update_step == 1
    assert resumed_runtime["optimizer_fresh"] is False
    assert resumed_runtime["independent_seed_origin"] is True
    assert resumed_runtime["rng_initialization"] == "checkpoint_restore_only/v1"
    assert resumed_runtime["sampler_seeds"] == fresh_runtime["sampler_seeds"]
    assert resumed_runtime["vector_env"].restored == vector_states
    assert ("load_last_complete", checkpoint_root) in events
    assert any(
        event[0] == "recover_accepted_checkpoint" and event[1] == checkpoint_root
        for event in events
    )
    assert any(
        event[0] == "retention"
        and event[1] == checkpoint_root
        and event[2]["latest_update"] == 1
        for event in events
    )
    assert [row["state"] for row in resumed.journal.verify()] == list(
        transactions[0].commit_states
    )
    metric_path = resume_root / "s6/training_metrics.jsonl"
    assert len(metric_path.read_text(encoding="utf-8").splitlines()) == 1
    resumed_metric = json.loads(metric_path.read_text(encoding="utf-8"))
    assert resumed_metric["collection_audit"] == (
        loaded_by_root[checkpoint_root].eval_metrics["collection_audit"]
    )
    checkpoint_audit_path = resume_root / "s6/checkpoint_audit.jsonl"
    assert len(
        checkpoint_audit_path.read_text(encoding="utf-8").splitlines()
    ) == 1
    resumed_runtime["vector_env"].close()
    repeated_runtime, repeated_remaining = resumed.prepare_seed(
        config.training.seeds[0], seed_transactions
    )
    assert repeated_remaining == seed_transactions[1:]
    assert len(metric_path.read_text(encoding="utf-8").splitlines()) == 1
    assert len(
        checkpoint_audit_path.read_text(encoding="utf-8").splitlines()
    ) == 1
    repeated_runtime["vector_env"].close()

    accepted_pending_run = tmp_path / "accepted-pending" / FORMAL_RUN_ID
    accepted_pending_root = (
        accepted_pending_run / "s6/checkpoints/seed-20260716"
    ).resolve()
    accepted_pending_dir = accepted_pending_root / (
        ".pending-update-00000001-" + "a" * 32
    )
    accepted_pending_dir.mkdir(parents=True)
    (accepted_pending_dir / "checkpoint.pt").write_bytes(b"checkpoint-pending")
    (accepted_pending_dir / "manifest.json").write_bytes(b"manifest-pending")
    (accepted_pending_dir / "complete.json").write_bytes(b"complete-pending")
    accepted_pending_checkpoint = standard_training.StandardTransactionCheckpoint(
        transaction_key=transactions[0].key,
        seed=transactions[0].seed,
        update=transactions[0].update,
        checkpoint_sha256="6" * 64,
        complete_marker_sha256=sha256(b"complete-pending").hexdigest(),
        policy_state_sha256="7" * 64,
    )
    loaded_by_root[accepted_pending_root] = SimpleNamespace(
        directory=accepted_pending_root / "update-00000001",
        update_step=1,
        checkpoint_sha256=accepted_pending_checkpoint.checkpoint_sha256,
        policy_state_sha256=accepted_pending_checkpoint.policy_state_sha256,
        vector_env_states=vector_states,
        normalization_stats={},
        best_record={},
        config_sha256="pending-config-placeholder",
        lineage={},
        scenario_sampler_state={
            "workers": [state["sampler_state"] for state in vector_states]
        },
        eval_metrics=loaded_by_root[checkpoint_root].eval_metrics,
    )
    accepted_pending_active = execution_authority_factory(
        run_root=accepted_pending_run
    )
    accepted_pending = standard_training.StandardProductionBackend(
        config=config,
        run_root=accepted_pending_run,
        repo_root=ROOT,
        stage5_authority=dict(
            accepted_pending_active["stage5_handle"].identity
        ),
        execution_capability=accepted_pending_active["capability"],
        _components=components,
        _immutable_bindings=immutable,
        _source_repair=source_repair,
    )
    _attach_backend_resource_segment(accepted_pending)
    loaded_by_root[accepted_pending_root].config_sha256 = (
        accepted_pending.config_sha256
    )
    loaded_by_root[accepted_pending_root].lineage = (
        accepted_pending._checkpoint_lineage(1)
    )
    accepted_pending_identity = {
        "kind": "update",
        "transaction_key": transactions[0].key,
        "seed": transactions[0].seed,
        "update": transactions[0].update,
    }
    for phase in ("pre", "post"):
        accepted_pending._append_resource_phase(
            identity=accepted_pending_identity,
            attempt=1,
            phase=phase,
            resource=resume_resource,
        )
    accepted_pending._append_resource_acceptance(
        identity=accepted_pending_identity,
        attempt=1,
        pre=resume_resource,
        post=resume_resource,
        checkpoint=accepted_pending_checkpoint,
    )
    recovery_prefix_root = accepted_pending_run
    recovery_prefix_checkpoint = accepted_pending_checkpoint
    accepted_pending_runtime, accepted_pending_remaining = accepted_pending.prepare_seed(
        config.training.seeds[0], seed_transactions
    )
    recovery_prefix_root = None
    recovery_prefix_checkpoint = None
    assert accepted_pending_remaining == seed_transactions[1:]
    assert accepted_pending.receipt_index.verify()[0]["checkpoint_sha256"] == (
        accepted_pending_checkpoint.checkpoint_sha256
    )
    assert [row["state"] for row in accepted_pending.journal.verify()] == list(
        transactions[0].commit_states
    )
    assert (accepted_pending_root / "update-00000001").is_dir()
    assert not accepted_pending_dir.exists()
    accepted_pending_runtime["vector_env"].close()

    class FailingCollector:
        def __init__(self, **kwargs):
            raise RuntimeError("injected collector construction failure")

    failing_root = tmp_path / "failing-prepare" / FORMAL_RUN_ID
    failing_active = execution_authority_factory(run_root=failing_root)
    failing = standard_training.StandardProductionBackend(
        config=config,
        run_root=failing_root,
        repo_root=ROOT,
        stage5_authority=dict(failing_active["stage5_handle"].identity),
        execution_capability=failing_active["capability"],
        _components={**components, "RolloutCollector": FailingCollector},
        _immutable_bindings=immutable,
        _source_repair=source_repair,
    )
    with pytest.raises(RuntimeError, match="collector construction"):
        failing.prepare_seed(config.training.seeds[0], seed_transactions)
    assert vectors[-1].closed is True

    incomplete_run = tmp_path / "incomplete" / FORMAL_RUN_ID
    incomplete_root = incomplete_run / "s6/checkpoints/seed-20260716"
    (incomplete_root / "update-00000001").mkdir(parents=True)
    (incomplete_root / "update-00000001/checkpoint.pt").write_bytes(b"partial")
    incomplete_active = execution_authority_factory(run_root=incomplete_run)
    incomplete = standard_training.StandardProductionBackend(
        config=config,
        run_root=incomplete_run,
        repo_root=ROOT,
        stage5_authority=dict(incomplete_active["stage5_handle"].identity),
        execution_capability=incomplete_active["capability"],
        _components=components,
        _immutable_bindings=immutable,
        _source_repair=source_repair,
    )
    incomplete_runtime, incomplete_remaining = incomplete.prepare_seed(
        config.training.seeds[0], seed_transactions
    )
    assert incomplete_remaining == seed_transactions
    assert incomplete_runtime["optimizer_fresh"] is True
    assert (incomplete_root / "update-00000001/checkpoint.pt").is_file()
    incomplete_runtime["vector_env"].close()

    ahead_run = tmp_path / "ahead" / FORMAL_RUN_ID
    ahead_root = ahead_run / "s6/checkpoints/seed-20260716"
    ahead_directory = ahead_root / "update-00000001"
    ahead_directory.mkdir(parents=True)
    (ahead_directory / "checkpoint.pt").write_bytes(b"checkpoint-ahead")
    (ahead_directory / "manifest.json").write_bytes(b"manifest-ahead")
    (ahead_directory / "complete.json").write_bytes(b"complete-ahead")
    ahead_active = execution_authority_factory(run_root=ahead_run)
    ahead = standard_training.StandardProductionBackend(
        config=config,
        run_root=ahead_run,
        repo_root=ROOT,
        stage5_authority=dict(ahead_active["stage5_handle"].identity),
        execution_capability=ahead_active["capability"],
        _components=components,
        _immutable_bindings=immutable,
        _source_repair=source_repair,
    )
    _attach_backend_resource_segment(ahead)
    loaded_by_root[ahead_root.resolve()] = SimpleNamespace(
        directory=ahead_directory.resolve(),
        update_step=1,
        checkpoint_sha256="d" * 64,
        policy_state_sha256="e" * 64,
        vector_env_states=vector_states,
        normalization_stats={},
        best_record={},
        config_sha256=ahead.config_sha256,
        lineage=ahead._checkpoint_lineage(1),
        scenario_sampler_state={
            "workers": [state["sampler_state"] for state in vector_states]
        },
        eval_metrics=loaded_by_root[checkpoint_root].eval_metrics,
    )
    ahead_resource = {
        "d_free_bytes": 200 * 1024**3,
        "rss_bytes": 1024,
        "peak_vram_bytes": 2048,
        "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
        "rss_root_pid": os.getpid(),
        "rss_sample_count": 2,
        "rss_latest_process_count": 1,
        "rss_peak_process_count": 1,
        "warnings": [],
        "hard_stops": [],
        "passed": True,
    }
    ahead_identity = {
        "kind": "update",
        "transaction_key": transactions[0].key,
        "seed": transactions[0].seed,
        "update": transactions[0].update,
    }
    ahead._append_resource_phase(
        identity=ahead_identity,
        attempt=1,
        phase="pre",
        resource=ahead_resource,
    )
    with pytest.raises(
        standard_training.StandardTrainingError,
        match="durable acceptance",
    ):
        ahead.prepare_seed(config.training.seeds[0], seed_transactions)
    assert ahead.receipt_index.verify() == ()
    assert ahead.journal.verify() == ()
    assert [
        row["phase"]
        for row in ahead._resource_audit_rows()
        if row.get("transaction_key") == transactions[0].key
    ] == ["pre"]

    ahead._append_resource_phase(
        identity=ahead_identity,
        attempt=1,
        phase="post",
        resource=ahead_resource,
    )
    with pytest.raises(
        standard_training.StandardTrainingError,
        match="acceptance|accepted",
    ):
        ahead.prepare_seed(config.training.seeds[0], seed_transactions)
    assert ahead.receipt_index.verify() == ()
    assert ahead.journal.verify() == ()
    assert [
        row["phase"]
        for row in ahead._resource_audit_rows()
        if row.get("transaction_key") == transactions[0].key
    ] == ["pre", "post"]

    ahead_checkpoint = standard_training.StandardTransactionCheckpoint(
        transaction_key=transactions[0].key,
        seed=transactions[0].seed,
        update=transactions[0].update,
        checkpoint_sha256="d" * 64,
        complete_marker_sha256=sha256(b"complete-ahead").hexdigest(),
        policy_state_sha256="e" * 64,
    )
    ahead._append_resource_acceptance(
        identity=ahead_identity,
        attempt=1,
        pre=ahead_resource,
        post=ahead_resource,
        checkpoint=ahead_checkpoint,
    )
    ahead_runtime, ahead_remaining = ahead.prepare_seed(
        config.training.seeds[0], seed_transactions
    )
    assert ahead_remaining == seed_transactions[1:]
    assert len(ahead.receipt_index.verify()) == 1
    assert [row["state"] for row in ahead.journal.verify()] == list(
        transactions[0].commit_states
    )
    assert [
        row["phase"]
        for row in ahead._resource_audit_rows()
        if row.get("transaction_key") == transactions[0].key
    ] == ["pre", "post", "accepted"]
    ahead_runtime["vector_env"].close()

    validation_run = tmp_path / "validation-ahead" / FORMAL_RUN_ID
    validation_root = validation_run / "s6/checkpoints/seed-20260716"
    validation_directory = validation_root / "update-00000010"
    validation_directory.mkdir(parents=True)
    (validation_directory / "checkpoint.pt").write_bytes(b"checkpoint-validation")
    (validation_directory / "manifest.json").write_bytes(b"manifest-validation")
    (validation_directory / "complete.json").write_bytes(b"complete-validation")
    validation_active = execution_authority_factory(run_root=validation_run)
    validation_ahead = standard_training.StandardProductionBackend(
        config=config,
        run_root=validation_run,
        repo_root=ROOT,
        stage5_authority=dict(validation_active["stage5_handle"].identity),
        execution_capability=validation_active["capability"],
        _components=components,
        _immutable_bindings=immutable,
        _source_repair=source_repair,
    )
    _attach_backend_resource_segment(validation_ahead)
    for item in transactions[:9]:
        prior_checkpoint = standard_training.StandardTransactionCheckpoint(
            transaction_key=item.key,
            seed=item.seed,
            update=item.update,
            checkpoint_sha256=sha256(f"checkpoint:{item.key}".encode()).hexdigest(),
            complete_marker_sha256=sha256(f"complete:{item.key}".encode()).hexdigest(),
            policy_state_sha256=sha256(f"policy:{item.key}".encode()).hexdigest(),
        )
        validation_ahead.receipt_index.append_once(prior_checkpoint)
        prior_bindings = {
            **immutable,
            "checkpoint_sha256": prior_checkpoint.checkpoint_sha256,
        }
        for state in item.commit_states:
            validation_ahead.journal.append(state, prior_bindings)
    validation_best = standard_training.ValidationRecord(
        transactions[9].seed,
        transactions[9].update,
        0.7,
        0.4,
        "update-00000010",
    ).to_dict()
    validation_policy_before = str(
        validation_ahead.receipt_index.verify()[-1]["policy_state_sha256"]
    )
    validation_collection_audit = _collector_audit_dict(
        validation_policy_before,
        prefix="resume-update-10",
    )
    validation_recovery_payload = _validation_trace_payload(attempt=2)
    loaded_by_root[validation_root.resolve()] = SimpleNamespace(
        directory=validation_directory.resolve(),
        update_step=10,
        checkpoint_sha256="8" * 64,
        policy_state_sha256="9" * 64,
        vector_env_states=vector_states,
        normalization_stats={},
        best_record=validation_best,
        config_sha256=validation_ahead.config_sha256,
        lineage=validation_ahead._checkpoint_lineage(10),
        scenario_sampler_state={
            "workers": [state["sampler_state"] for state in vector_states]
        },
        eval_metrics={
            "update": {
                "update_step": 10,
                "initial_ratio_max_abs_error": 1.0e-6,
                "policy_loss": 0.1,
                "value_loss": 0.2,
                "approx_kl": 0.003,
                "grad_pre_clip_norm_max": 0.7,
                "grad_post_clip_norm_max": 0.5,
                "optimizer_steps": 4,
                "policy_state_sha256_before": validation_policy_before,
                "policy_state_sha256_after": "9" * 64,
                "math_evidence": _math_evidence_dict(
                    validation_collection_audit,
                    policy_sha256=validation_policy_before,
                ),
            },
            "validation": {
                "episode_count": 16,
                "success_rate_under_fixed_step_budget": 0.7,
                "mean_final_coverage": 0.4,
                "validation_trace_binding": _validation_trace_binding(
                    validation_recovery_payload
                ),
            },
            "eval_isolation": {"passed": True},
            "collection_audit": validation_collection_audit,
        },
    )
    validation_identity = {
        "kind": "update",
        "transaction_key": transactions[9].key,
        "seed": transactions[9].seed,
        "update": transactions[9].update,
    }
    for attempt in (1, 2):
        validation_ahead._append_resource_phase(
            identity=validation_identity,
            attempt=attempt,
            phase="pre",
            resource=resume_resource,
        )
        pending = validation_ahead.run_root / (
            "stage6-attempts/validation-seed-20260716-update-010-"
            f"attempt-{attempt:03d}.pending.jsonl"
        )
        pending.parent.mkdir(parents=True, exist_ok=True)
        pending.write_bytes(
            _validation_trace_payload(attempt=attempt)
        )
    validation_ahead._append_resource_phase(
        identity=validation_identity,
        attempt=2,
        phase="post",
        resource=resume_resource,
    )
    validation_ahead._append_resource_acceptance(
        identity=validation_identity,
        attempt=2,
        pre=resume_resource,
        post=resume_resource,
        checkpoint=standard_training.StandardTransactionCheckpoint(
            transaction_key=transactions[9].key,
            seed=transactions[9].seed,
            update=transactions[9].update,
            checkpoint_sha256="8" * 64,
            complete_marker_sha256=sha256(b"complete-validation").hexdigest(),
            policy_state_sha256="9" * 64,
        ),
    )
    validation_runtime, validation_remaining = validation_ahead.prepare_seed(
        config.training.seeds[0], seed_transactions
    )
    assert validation_remaining == seed_transactions[10:]
    canonical_validation = validation_ahead.stage_root / (
        "episode-traces/validation-seed-20260716-update-010.jsonl"
    )
    assert canonical_validation.is_file()
    assert b'"attempt":2' in canonical_validation.read_bytes()
    assert (
        validation_ahead.run_root
        / "stage6-attempts/validation-seed-20260716-update-010-"
        "attempt-001.pending.jsonl"
    ).is_file()
    assert not (
        validation_ahead.run_root
        / "stage6-attempts/validation-seed-20260716-update-010-"
        "attempt-002.pending.jsonl"
    ).exists()
    validation_runtime["vector_env"].close()


def test_production_backend_run_update_orders_resource_transaction_metrics_retention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_execution_authority: dict[str, object],
) -> None:
    config = load_stage6_config(CONFIG)
    expected_contract = SafetyContract.from_stage6_config(config)
    expected_safety = config.safety.model_dump(mode="json")
    expected_safety_sha256 = expected_contract.sha256
    transaction = standard_training.build_standard_training_transactions(config)[9]
    events: list[str] = []
    policy = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.ppo.learning_rate,
        eps=config.ppo.adam_eps,
        weight_decay=config.ppo.weight_decay,
    )
    checkpoint = standard_training.StandardTransactionCheckpoint(
        transaction_key=transaction.key,
        seed=transaction.seed,
        update=transaction.update,
        checkpoint_sha256="c" * 64,
        complete_marker_sha256="d" * 64,
        policy_state_sha256=policy_state_sha256(policy),
    )
    selected_best = standard_training.ValidationRecord(
        transaction.seed,
        transaction.update,
        0.7,
        0.4,
        "update-00000010",
    ).to_dict()
    production_collection_audit = {
        "schema_version": "stage4_spawn_collector/v1",
        "worker_pids": list(range(20_001, 20_009)),
        "worker_start_methods": ["spawn"] * 8,
        "per_env_trainable_counts": [128] * 8,
        "diagnostic_reset_counts": [0] * 8,
        "trainable_transition_count": 1024,
        "inference_pids": [os.getpid()],
        "inference_batch_count": 128,
        "policy_device": "cuda:0",
        "policy_state_sha256": checkpoint.policy_state_sha256,
        "snapshot_sha256": [
            sha256(f"production-snapshot-{index}".encode("ascii")).hexdigest()
            for index in range(1024)
        ],
        "terminal_transition_count": 0,
    }

    def run_update_transaction(**kwargs):
        events.append("transaction:start")
        assert kwargs["transaction"] == transaction
        assert kwargs["checkpoint_receipt_index"] is backend.receipt_index
        assert kwargs["journal"] is backend.journal
        assert kwargs["checkpoint_metadata"]["safety_contract"] == expected_contract
        assert (
            kwargs["checkpoint_metadata"]["training_config"]["safety"]
            == expected_safety
        )
        assert (
            kwargs["checkpoint_metadata"]["lineage"][
                "safety_contract_sha256"
            ]
            == expected_safety_sha256
        )
        assert callable(kwargs["resource_guard"])
        assert callable(kwargs["capability_guard"])
        assert callable(kwargs["persist_post_resource"])
        assert callable(kwargs["accept_post_resource"])
        kwargs["capability_guard"]("transaction:fake-capability-boundary")
        kwargs["resource_guard"]("transaction:fake-boundary")
        binding = kwargs["eval_binding_state_provider"]()
        assert binding["config_bytes"] == backend.config_bytes
        assert b"checkpoint-v9" in binding["checkpoint_bytes"]
        validation = kwargs["validation_evaluator"]()
        kwargs["persist_post_resource"](checkpoint)
        kwargs["accept_post_resource"](checkpoint)
        accepted = backend._accepted_resource_attempt(
            {
                "kind": "update",
                "transaction_key": transaction.key,
                "seed": transaction.seed,
                "update": transaction.update,
            }
        )
        assert accepted is not None
        assert accepted["schema_version"] == "stage6_resource_acceptance/v2"
        assert accepted["checkpoint"] == {
            "transaction_key": checkpoint.transaction_key,
            "seed": checkpoint.seed,
            "update": checkpoint.update,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "complete_marker_sha256": checkpoint.complete_marker_sha256,
            "policy_state_sha256": checkpoint.policy_state_sha256,
        }
        events.append("transaction:done")
        return standard_training.StandardUpdateResult(
            transaction=transaction,
            update_metrics={
                "update_step": 10,
                "initial_ratio_max_abs_error": 2.0e-6,
                "policy_loss": 0.1,
                "value_loss": 0.2,
                "approx_kl": 0.003,
                "grad_pre_clip_norm_max": 0.7,
                "grad_post_clip_norm_max": 0.5,
                "optimizer_steps": 4,
                "policy_state_sha256_before": checkpoint.policy_state_sha256,
                "policy_state_sha256_after": checkpoint.policy_state_sha256,
                "math_evidence": _math_evidence_dict(
                    production_collection_audit,
                    policy_sha256=checkpoint.policy_state_sha256,
                ),
            },
            validation_result=validation,
            eval_isolation_audit={"passed": True},
            best_record=selected_best,
            checkpoint=checkpoint,
            journal_records=({"state": transaction.commit_states[-1]},),
            collection_audit=production_collection_audit,
        )

    def run_evaluation(**kwargs):
        events.append("validation")
        assert kwargs["split"] == "validation"
        assert kwargs["method"] == "ppo_policy"
        assert kwargs["episode_count"] == 16
        assert kwargs["safety_contract"] == expected_contract
        assert kwargs["config_sha256"] == backend.config_sha256
        assert callable(kwargs["resource_guard"])
        kwargs["resource_guard"]("evaluation:fake-boundary")
        pending_trace = Path(kwargs["trace_path"])
        assert pending_trace.parent == backend.run_root / "stage6-attempts"
        assert pending_trace.name.endswith("attempt-002.pending.jsonl")
        pending_trace.parent.mkdir(parents=True, exist_ok=True)
        pending_trace.write_bytes(
            b"".join(
                (
                    json.dumps(
                        {"episode_index": index},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                for index in range(16)
            )
        )
        return {
            "episode_count": 16,
            "success_rate_under_fixed_step_budget": 0.7,
            "mean_final_coverage": 0.4,
            "fairness_audit": {
                "shared_environment_contract": {
                    "environment_contract": {
                        **expected_contract.binding(
                            config_sha256=backend.config_sha256
                        )
                    }
                }
            },
        }

    class Retention:
        def __init__(self, root, *, schema_version):
            assert Path(root).resolve() == checkpoint_root

        def apply(self, **kwargs):
            events.append("retention")
            assert kwargs == {
                "latest_update": 10,
                "periodic_updates": (),
                "best_update": 10,
                "periodic_keep_count": 5,
                "protected_updates": (),
            }
            return SimpleNamespace(kept_updates=(10,), removed_updates=())

    written_metrics: dict[str, list[dict[str, object]]] = {}

    def append_metric(path, record):
        events.append(f"metrics:{Path(path).name}")
        assert record["transaction_key"] == transaction.key
        written_metrics.setdefault(Path(path).name, []).append(dict(record))
        return True

    vector_states = tuple(
        {"episode_state": {"worker": index}, "sampler_state": {"cursor": index}}
        for index in range(8)
    )

    class AuditVector:
        def __init__(self) -> None:
            self.states = vector_states
            self.restore_count = 0
            self.drift_on_restore = False

        def capture_states(self):
            return self.states

        def restore_states(self, states):
            events.append("checkpoint:restore")
            self.restore_count += 1
            self.states = tuple(states)
            if self.drift_on_restore:
                self.states = (
                    {
                        **self.states[0],
                        "episode_state": {"worker": "drifted"},
                    },
                    *self.states[1:],
                )

        def current_observations(self):
            return {
                index: SimpleNamespace(observation=f"observation-{index}")
                for index in range(8)
            }

    audit_vector = AuditVector()

    class AuditManager:
        def load_last_complete(self, **kwargs):
            events.append("checkpoint:load")
            assert kwargs["expected_config_sha256"] == backend.config_sha256
            assert kwargs["expected_lineage"] == backend._checkpoint_lineage(
                transaction.update
            )
            return SimpleNamespace(
                update_step=transaction.update,
                checkpoint_sha256=checkpoint.checkpoint_sha256,
                policy_state_sha256=checkpoint.policy_state_sha256,
                normalization_stats={},
                scenario_sampler_state={
                    "workers": [state["sampler_state"] for state in vector_states]
                },
                vector_env_states=vector_states,
                best_record=selected_best,
                config_sha256=backend.config_sha256,
                lineage=backend._checkpoint_lineage(transaction.update),
            )

    def deterministic_action(policy, observation, *, device):
        events.append("checkpoint:action")
        return {
            "observation": observation,
            "policy_sha256": policy_state_sha256(policy),
            "device": device,
        }

    components = {
        "build_standard_catalog": lambda **kwargs: "catalog",
        "load_stage4_policy_for_standard": lambda **kwargs: policy,
        "stage4_checkpoint_path": tmp_path / "stage4.pt",
        "standard_env_specs": lambda *args, **kwargs: (),
        "SpawnVectorEnv": lambda *args, **kwargs: None,
        "RolloutCollector": lambda **kwargs: None,
        "CollectorContract": lambda **kwargs: None,
        "PPOTrainer": lambda *args, **kwargs: None,
        "CheckpointManager": lambda *args, **kwargs: None,
        "CheckpointReceiptIndex": standard_training.CheckpointReceiptIndex,
        "Stage6StateJournal": Stage6StateJournal,
        "run_standard_update_transaction": run_update_transaction,
        "run_standard_evaluation": run_evaluation,
        "capture_resource_snapshot": lambda **kwargs: (
            events.append("resource") or ResourceSnapshot(
                d_free_bytes=200 * 1024**3,
                rss_bytes=1024,
                peak_vram_bytes=2048,
                rss_source="process_tree_lifecycle_peak_current_sum/v1",
                rss_root_pid=os.getpid(),
                rss_sample_count=2,
                rss_latest_process_count=1,
                rss_peak_process_count=1,
            )
        ),
        "evaluate_resource_gates": lambda snapshot, **kwargs: SimpleNamespace(
            passed=True,
            warnings=(),
            hard_stops=(),
        ),
        "CheckpointRetentionManager": Retention,
        "append_transaction_metric_once": append_metric,
        "deterministic_action_record": deterministic_action,
    }
    immutable = {
        key: value
        for key, value in _journal_bindings("0" * 64).items()
        if key != "checkpoint_sha256"
    }
    immutable["config_sha256"] = sha256(CONFIG.read_bytes()).hexdigest()
    source_repair = _install_test_ordinal6_source_repair(monkeypatch)(
        config=config,
        current_immutable_bindings=immutable,
    )
    backend = standard_training.StandardProductionBackend(
        config=config,
        run_root=active_execution_authority["run_root"],
        repo_root=ROOT,
        stage5_authority=dict(
            active_execution_authority["stage5_handle"].identity
        ),
        execution_capability=active_execution_authority["capability"],
        _components=components,
        _immutable_bindings=immutable,
        _source_repair=source_repair,
    )
    _attach_backend_resource_segment(backend)
    (backend.stage_root / "config.json").write_bytes(backend.config_bytes)
    backend._append_resource_phase(
        identity={
            "kind": "update",
            "transaction_key": transaction.key,
            "seed": transaction.seed,
            "update": transaction.update,
        },
        attempt=1,
        phase="pre",
        resource={
            "d_free_bytes": 200 * 1024**3,
            "rss_bytes": 1024,
            "peak_vram_bytes": 2048,
            "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
            "rss_root_pid": os.getpid(),
            "rss_sample_count": 2,
            "rss_latest_process_count": 1,
            "rss_peak_process_count": 1,
            "warnings": [],
            "hard_stops": [],
            "passed": True,
        },
    )
    checkpoint_root = (
        backend.stage_root / "checkpoints/seed-20260716"
    ).resolve()
    previous = checkpoint_root / "update-00000009"
    previous.mkdir(parents=True)
    (previous / "checkpoint.pt").write_bytes(b"checkpoint-v9")
    (previous / "manifest.json").write_bytes(b"manifest-v9")
    (previous / "complete.json").write_bytes(b"complete-v9")
    runtime = {
        "seed": transaction.seed,
        "policy": policy,
        "optimizer": optimizer,
        "trainer": SimpleNamespace(update_step=9),
        "collector": object(),
        "vector_env": audit_vector,
        "checkpoint_manager": AuditManager(),
        "checkpoint_root": checkpoint_root,
        "normalization_stats": {},
        "best_record": {},
        "continue_from_current_state": True,
    }

    result = backend.run_update(runtime, transaction)

    assert result.checkpoint == checkpoint
    assert runtime["best_record"] == selected_best
    assert runtime["continue_from_current_state"] is True
    assert backend.checkpoint_write_count() == 1
    assert events == [
        "resource",
        "transaction:start",
        "resource",
        "validation",
        "resource",
        "resource",
        "transaction:done",
        "metrics:training_metrics.jsonl",
        "metrics:validation_metrics.jsonl",
        "checkpoint:action",
        "checkpoint:load",
        "checkpoint:restore",
        "checkpoint:action",
        "retention",
    ]
    assert audit_vector.restore_count == 1
    resource_rows = [
        json.loads(line)
        for line in (backend.stage_root / "resource_audit.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [row["phase"] for row in resource_rows] == [
        "segment_start",
        "pre",
        "pre",
        "post",
        "accepted",
    ]
    assert [row["attempt"] for row in resource_rows[1:]] == [1, 2, 2, 2]
    assert [row["accepted"] for row in resource_rows[1:]] == [
        False,
        False,
        False,
        True,
    ]
    assert resource_rows[-1]["pre"]["passed"] is True
    assert resource_rows[-1]["post"]["passed"] is True
    assert (
        backend.stage_root
        / "episode-traces/validation-seed-20260716-update-010.jsonl"
    ).is_file()
    assert not (
        backend.run_root
        / "stage6-attempts/validation-seed-20260716-update-010-attempt-002.pending.jsonl"
    ).exists()
    math_rows = [
        json.loads(line)
        for line in (backend.stage_root / "math_audit.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(math_rows) == 1
    assert math_rows[0]["transaction_key"] == transaction.key
    assert math_rows[0]["passed"] is True
    assert math_rows[0]["collection_audit"] == production_collection_audit
    assert math_rows[0]["policy_state_sha256_before"] == checkpoint.policy_state_sha256
    assert math_rows[0]["policy_state_sha256_after"] == checkpoint.policy_state_sha256
    training_row = written_metrics["training_metrics.jsonl"][0]
    assert training_row["collection_audit"] == production_collection_audit
    checkpoint_rows = [
        json.loads(line)
        for line in (backend.stage_root / "checkpoint_audit.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(checkpoint_rows) == 1
    assert checkpoint_rows[0]["transaction_key"] == transaction.key
    assert checkpoint_rows[0]["deterministic_action_bit_exact"] is True
    assert checkpoint_rows[0]["vector_env_state_count"] == 8
    assert checkpoint_rows[0]["vector_env_state_round_trip_bit_exact"] is True
    assert checkpoint_rows[0]["observation_round_trip_bit_exact"] is True
    audit_vector.drift_on_restore = True
    with pytest.raises(
        standard_training.StandardTrainingError,
        match="deterministic replay",
    ):
        backend._checkpoint_replay_audit(
            runtime=runtime,
            transaction=transaction,
            result=result,
        )


def test_production_backend_finishes_seed_and_freezes_receipt_bound_global_best(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_execution_authority: dict[str, object],
) -> None:
    config = load_stage6_config(CONFIG)
    components = {
        "build_standard_catalog": lambda **kwargs: "catalog",
        "load_stage4_policy_for_standard": lambda **kwargs: nn.Linear(3, 2),
        "stage4_checkpoint_path": tmp_path / "stage4.pt",
        "standard_env_specs": lambda *args, **kwargs: (),
        "SpawnVectorEnv": lambda *args, **kwargs: None,
        "RolloutCollector": lambda **kwargs: None,
        "CollectorContract": lambda **kwargs: None,
        "PPOTrainer": lambda *args, **kwargs: None,
        "CheckpointManager": lambda *args, **kwargs: None,
        "CheckpointReceiptIndex": standard_training.CheckpointReceiptIndex,
        "Stage6StateJournal": Stage6StateJournal,
    }
    immutable = {
        key: value
        for key, value in _journal_bindings("0" * 64).items()
        if key != "checkpoint_sha256"
    }
    source_repair = _install_test_ordinal6_source_repair(monkeypatch)(
        config=config,
        current_immutable_bindings=immutable,
    )
    backend = standard_training.StandardProductionBackend(
        config=config,
        run_root=active_execution_authority["run_root"],
        repo_root=ROOT,
        stage5_authority=dict(
            active_execution_authority["stage5_handle"].identity
        ),
        execution_capability=active_execution_authority["capability"],
        _components=components,
        _immutable_bindings=immutable,
        _source_repair=source_repair,
    )
    closed: list[bool] = []
    best = standard_training.ValidationRecord(
        config.training.seeds[0],
        10,
        0.7,
        0.4,
        "update-00000010",
    )
    runtime = {
        "seed": best.seed,
        "best_record": best.to_dict(),
        "vector_env": SimpleNamespace(close=lambda: closed.append(True)),
    }

    assert backend.finish_seed(runtime) == best
    assert closed == [True]
    checkpoint = standard_training.StandardTransactionCheckpoint(
        transaction_key=backend.transactions[9].key,
        seed=best.seed,
        update=best.update,
        checkpoint_sha256="a" * 64,
        complete_marker_sha256="b" * 64,
        policy_state_sha256="c" * 64,
    )
    backend.receipt_index.append_once(checkpoint)

    backend.record_preflight_phase()
    assert backend.freeze_global_best(best) == best
    frozen_path = backend.stage_root / "global-best.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert frozen == {
        "schema_version": "stage6_global_best/v1",
        "record": best.to_dict(),
        "transaction_key": checkpoint.transaction_key,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "complete_marker_sha256": checkpoint.complete_marker_sha256,
        "policy_state_sha256": checkpoint.policy_state_sha256,
    }
    assert backend.freeze_global_best(best) == best
    frozen_path.write_bytes(
        frozen_path.read_bytes().replace(b"a" * 64, b"f" * 64)
    )
    with pytest.raises(standard_training.StandardTrainingError, match="global best"):
        backend.freeze_global_best(best)


def test_final_eval_recovers_committed_pair_and_attempt_scoped_crash_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_execution_authority: dict[str, object],
) -> None:
    config = load_stage6_config(CONFIG)
    expected_safety = SafetyContract.from_stage6_config(config)
    expected_safety_binding = expected_safety.binding(
        config_sha256=sha256(CONFIG.read_bytes()).hexdigest()
    )
    policy = nn.Linear(3, 2)
    policy_hash = policy_state_sha256(policy)
    evaluations: list[tuple[str, str]] = []
    resource_captures: list[int] = []
    mutate_checkpoint = False
    checkpoint_root = (
        Path(active_execution_authority["run_root"])
        / "s6/checkpoints/seed-20260716"
    ).resolve()

    loaded = SimpleNamespace(
        update_step=10,
        checkpoint_sha256="a" * 64,
        policy_state_sha256=policy_hash,
        normalization_stats={},
        vector_env_states=tuple(
            {"episode_state": {"worker": index}, "sampler_state": {"cursor": index}}
            for index in range(8)
        ),
        best_record={},
    )

    class Manager:
        def __init__(self, root, *, schema_version):
            assert Path(root).resolve() == checkpoint_root

        def load_complete(self, **kwargs):
            assert kwargs["update_step"] == 10
            return loaded

    def run_evaluation(**kwargs):
        nonlocal mutate_checkpoint
        evaluations.append((kwargs["split"], kwargs["method"]))
        assert callable(kwargs["resource_guard"])
        kwargs["resource_guard"]("evaluation:fake-boundary")
        trace = Path(kwargs["trace_path"])
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.write_bytes(
            b"".join(
                (
                    json.dumps(
                        {"episode_index": index},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                for index in range(64)
            )
        )
        if mutate_checkpoint:
            (checkpoint_root / "update-00000010/checkpoint.pt").write_bytes(
                b"mutated-during-eval"
            )
        return {
            "episode_count": 64,
            "metrics": {"episode_count": 64, "mean_final_coverage": 0.4},
            "bootstrap_audit": {"resamples": 2000},
            "fairness_audit": {
                "passed": True,
                "shared_environment_contract": {
                    "environment_contract": {
                        **expected_safety_binding,
                    }
                },
            },
        }

    def verify_evaluation(**kwargs):
        trace = Path(kwargs["trace_path"])
        summary = Path(kwargs["summary_path"])
        commit = Path(kwargs["commit_path"])
        assert trace.is_file() and summary.is_file() and commit.is_file()
        value = json.loads(summary.read_text(encoding="utf-8"))
        return {"result": value["result"]}

    components = {
        "build_standard_catalog": lambda **kwargs: "catalog",
        "load_stage4_policy_for_standard": lambda **kwargs: policy,
        "stage4_checkpoint_path": tmp_path / "stage4.pt",
        "standard_env_specs": lambda *args, **kwargs: (),
        "SpawnVectorEnv": lambda *args, **kwargs: None,
        "RolloutCollector": lambda **kwargs: None,
        "CollectorContract": lambda **kwargs: None,
        "PPOTrainer": lambda *args, **kwargs: None,
        "CheckpointManager": Manager,
        "CheckpointReceiptIndex": standard_training.CheckpointReceiptIndex,
        "Stage6StateJournal": Stage6StateJournal,
        "run_standard_evaluation": run_evaluation,
        "verify_standard_final_evaluation_artifacts": verify_evaluation,
        "run_eval_only_transaction": standard_training.run_eval_only_transaction,
        "capture_resource_snapshot": lambda **kwargs: (
            resource_captures.append(len(resource_captures) + 1)
            or SimpleNamespace(
                d_free_bytes=200 * 1024**3,
                rss_bytes=1024,
                peak_vram_bytes=2048,
                rss_source="process_tree_lifecycle_peak_current_sum/v1",
                rss_root_pid=os.getpid(),
                rss_sample_count=2,
                rss_latest_process_count=1,
                rss_peak_process_count=1,
            )
        ),
        "evaluate_resource_gates": lambda snapshot, **kwargs: SimpleNamespace(
            passed=True,
            warnings=(),
            hard_stops=(),
        ),
    }
    immutable = {
        key: value
        for key, value in _journal_bindings("0" * 64).items()
        if key != "checkpoint_sha256"
    }
    source_repair = _install_test_ordinal6_source_repair(monkeypatch)(
        config=config,
        current_immutable_bindings=immutable,
    )
    backend = standard_training.StandardProductionBackend(
        config=config,
        run_root=active_execution_authority["run_root"],
        repo_root=ROOT,
        stage5_authority=dict(
            active_execution_authority["stage5_handle"].identity
        ),
        execution_capability=active_execution_authority["capability"],
        _components=components,
        _immutable_bindings=immutable,
        _source_repair=source_repair,
    )
    _attach_backend_resource_segment(backend)
    backend.stage_root.joinpath("config.json").write_bytes(backend.config_bytes)
    complete = checkpoint_root / "update-00000010"
    complete.mkdir(parents=True)
    (complete / "checkpoint.pt").write_bytes(b"checkpoint-v10")
    (complete / "manifest.json").write_bytes(b"manifest-v10")
    (complete / "complete.json").write_bytes(b"complete-v10")
    best = standard_training.ValidationRecord(
        config.training.seeds[0], 10, 0.7, 0.4, "update-00000010"
    )
    receipt = standard_training.StandardTransactionCheckpoint(
        transaction_key=backend.transactions[9].key,
        seed=best.seed,
        update=best.update,
        checkpoint_sha256=loaded.checkpoint_sha256,
        complete_marker_sha256="b" * 64,
        policy_state_sha256=policy_hash,
    )
    backend.receipt_index.append_once(receipt)
    backend.record_preflight_phase()
    backend.freeze_global_best(best)

    first = backend.run_final_evaluation(best, "test", "ppo_policy")
    second = backend.run_final_evaluation(best, "test", "ppo_policy")

    assert first == second
    assert first["episode_count"] == 64
    assert evaluations == [("test", "ppo_policy")]
    assert resource_captures == [1, 2, 3]
    resource_rows = backend._resource_audit_rows()
    assert [row["phase"] for row in resource_rows] == [
        "segment_start",
        "pre",
        "post",
        "accepted",
    ]
    assert all(
        row["transaction_key"] == "final:test:ppo_policy"
        for row in resource_rows[1:]
    )
    summary_path = backend.stage_root / "episode-traces/final-test-ppo_policy.summary.json"
    trace_path = backend.stage_root / "episode-traces/final-test-ppo_policy.jsonl"
    commit_path = backend.stage_root / "episode-traces/final-test-ppo_policy.commit.json"
    assert summary_path.is_file() and trace_path.is_file() and commit_path.is_file()
    trace_path.write_bytes(trace_path.read_bytes() + b"tamper\n")
    with pytest.raises(standard_training.StandardTrainingError, match="final eval"):
        backend.run_final_evaluation(best, "test", "ppo_policy")

    mutate_checkpoint = True
    with pytest.raises(
        standard_training.StandardTrainingError,
        match="config/checkpoint binding",
    ):
        backend.run_final_evaluation(best, "unseen", "ppo_policy")
    unseen_trace = backend.stage_root / "episode-traces/final-unseen-ppo_policy.jsonl"
    unseen_summary = unseen_trace.with_suffix(".summary.json")
    unseen_commit = unseen_trace.with_suffix(".commit.json")
    assert not unseen_trace.exists()
    assert not unseen_summary.exists()
    assert not unseen_commit.exists()
    assert tuple(
        (backend.run_root / "stage6-attempts").rglob(
            "final-unseen-ppo_policy.jsonl"
        )
    )

    mutate_checkpoint = False
    (checkpoint_root / "update-00000010/checkpoint.pt").write_bytes(
        b"checkpoint-v10"
    )
    recovered_unseen = backend.run_final_evaluation(best, "unseen", "ppo_policy")
    assert recovered_unseen["episode_count"] == 64
    assert unseen_trace.is_file() and unseen_summary.is_file() and unseen_commit.is_file()

    original_publish = standard_training.StandardProductionBackend._publish_final_eval_attempt

    def crash_after_trace_publish(self, **kwargs):
        canonical_trace = Path(kwargs["canonical_trace"])
        attempt_trace = Path(kwargs["attempt_trace"])
        canonical_trace.parent.mkdir(parents=True, exist_ok=True)
        canonical_trace.write_bytes(attempt_trace.read_bytes())
        raise OSError("injected crash after canonical trace publication")

    monkeypatch.setattr(
        standard_training.StandardProductionBackend,
        "_publish_final_eval_attempt",
        crash_after_trace_publish,
    )
    with pytest.raises(OSError, match="injected crash"):
        backend.run_final_evaluation(best, "test", "random_valid_frontier")
    baseline_trace = (
        backend.stage_root / "episode-traces/final-test-random_valid_frontier.jsonl"
    )
    baseline_summary = baseline_trace.with_suffix(".summary.json")
    baseline_commit = baseline_trace.with_suffix(".commit.json")
    assert baseline_trace.is_file()
    assert not baseline_summary.exists()
    assert not baseline_commit.exists()

    monkeypatch.setattr(
        standard_training.StandardProductionBackend,
        "_publish_final_eval_attempt",
        original_publish,
    )
    evaluation_count_before_recovery = len(evaluations)
    recovered_baseline = backend.run_final_evaluation(
        best,
        "test",
        "random_valid_frontier",
    )
    assert recovered_baseline["episode_count"] == 64
    assert len(evaluations) == evaluation_count_before_recovery
    assert baseline_trace.is_file() and baseline_summary.is_file()
    assert baseline_commit.is_file()


def test_production_backend_finalize_writes_and_verifies_canonical_machine_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_execution_authority: dict[str, object],
) -> None:
    from lunar_exploration_ppo.eval.standard import STANDARD_EVALUATION_METHODS
    from lunar_exploration_ppo.workflows import stage6 as stage6_workflow
    from test_stage4_checkpoint import _CheckpointPolicy, VERSIONS

    original_write_manifest = stage6_workflow.write_stage6_manifest

    def write_manifest(
        *,
        stage_root: Path,
        repo_root: Path,
        execution_capability: object | None = None,
        run_lease: object | None = None,
    ) -> Path:
        return original_write_manifest(
            stage_root=stage_root,
            repo_root=repo_root,
            execution_capability=(
                active_execution_authority["capability"]
                if execution_capability is None
                else execution_capability
            ),
            run_lease=(
                active_execution_authority["lease"]
                if run_lease is None
                else run_lease
            ),  # type: ignore[arg-type]
        )

    monkeypatch.setattr(stage6_workflow, "write_stage6_manifest", write_manifest)

    config = load_stage6_config(CONFIG)
    authority_root = tmp_path / "authority"
    authority_root.mkdir()
    authority_gate = authority_root / "gate.json"
    authority_checkpoint = authority_root / "checkpoint.pt"
    authority_gate.write_bytes(b"canonical-stage5-gate")
    authority_checkpoint.write_bytes(b"canonical-stage4-checkpoint")
    authority_gate_bytes = authority_gate.read_bytes()
    authority_checkpoint_bytes = authority_checkpoint.read_bytes()
    authority_calls: list[str] = []

    class AuthorityHandle:
        identity = {
            "gate_sha256": config.stage5_authority.gate_sha256,
            "commit_sha256": config.stage5_authority.commit_sha256,
            "commit_tree": config.stage5_authority.commit_tree,
            "review_sha256": config.stage5_authority.review_sha256,
            "manifest_sha256": config.stage5_authority.manifest_sha256,
            "checkpoint_sha256": config.stage5_authority.checkpoint_sha256,
            "policy_state_sha256": config.stage5_authority.policy_state_sha256,
            "performance_advantage_established": False,
        }

        def require_current(self, label="authority"):
            authority_calls.append(label)
            if (
                authority_gate.read_bytes() != authority_gate_bytes
                or authority_checkpoint.read_bytes() != authority_checkpoint_bytes
            ):
                raise Stage6WorkflowError("Stage 5 authority changed")

    def verify_authority(*, gate_path, repo_root):
        assert Path(gate_path).resolve() == authority_gate.resolve()
        assert Path(repo_root).resolve() == ROOT.resolve()
        handle = AuthorityHandle()
        handle.require_current("verify")
        return handle

    monkeypatch.setattr(
        stage6_workflow,
        "CANONICAL_STAGE5_GATE",
        authority_gate,
    )
    monkeypatch.setattr(
        stage6_workflow,
        "verify_frozen_stage5_authority",
        verify_authority,
    )
    catalog = SimpleNamespace(
        sha256=str(active_execution_authority["identity"]["catalog_sha256"]),
        spatial_audit=lambda: {"parent_cross_split_count": 0},
    )
    components = {
        "build_standard_catalog": lambda **kwargs: catalog,
        "load_stage4_policy_for_standard": lambda **kwargs: nn.Linear(3, 2),
        "stage4_checkpoint_path": tmp_path / "stage4.pt",
        "standard_env_specs": lambda *args, **kwargs: (),
        "SpawnVectorEnv": lambda *args, **kwargs: None,
        "RolloutCollector": lambda **kwargs: None,
        "CollectorContract": lambda **kwargs: None,
        "PPOTrainer": lambda *args, **kwargs: None,
        "CheckpointManager": lambda *args, **kwargs: None,
        "CheckpointReceiptIndex": standard_training.CheckpointReceiptIndex,
        "Stage6StateJournal": Stage6StateJournal,
    }
    immutable = {
        key: value
        for key, value in _journal_bindings("0" * 64).items()
        if key != "checkpoint_sha256"
    }
    immutable["config_sha256"] = sha256(
        standard_training.ArtifactStore.canonical_json_bytes(
            config.model_dump(mode="json")
        )
    ).hexdigest()
    immutable["stage5_gate_sha256"] = config.stage5_authority.gate_sha256
    tree = str(immutable["reviewed_prospective_git_tree"])
    fake_execution_identity = {
        "schema_version": "stage6_execution_identity/v1",
        "base_commit": stage6_workflow.STAGE5_COMMIT,
        "head_commit": stage6_workflow.STAGE5_COMMIT,
        "prospective_git_tree": tree,
        "prospective_tree_sha256": sha256(tree.encode("ascii")).hexdigest(),
        "changed_paths": sorted(stage6_workflow.STAGE6_SOURCE_PATHS),
        "changed_path_set_sha256": immutable["changed_path_set_sha256"],
        "real_index_empty": True,
        "config_sha256": immutable["config_sha256"],
        "source_set_sha256": immutable["source_set_sha256"],
        "data_sha256": immutable["data_sha256"],
        "catalog_sha256": catalog.sha256,
        "environment_identity": immutable["environment_identity"],
        "environment_sha256": immutable["environment_sha256"],
        "source_identity": {"schema_version": "test-source/v1"},
        "data_identity": {"schema_version": "test-data/v1"},
    }
    fake_execution_identity = dict(active_execution_authority["identity"])
    for key in (
        "config_sha256",
        "source_set_sha256",
        "prospective_tree_sha256",
        "data_sha256",
        "environment_identity",
        "environment_sha256",
    ):
        immutable[key] = fake_execution_identity[key]
    immutable["reviewed_prospective_git_tree"] = fake_execution_identity[
        "prospective_git_tree"
    ]
    verified_review = dict(active_execution_authority["verified"])
    immutable.update(_review_immutable_bindings(verified_review))
    source_repair = _install_test_ordinal6_source_repair(monkeypatch)(
        config=config,
        current_immutable_bindings=immutable,
    )
    monkeypatch.setattr(
        stage6_workflow,
        "stage6_execution_identity",
        lambda **kwargs: dict(fake_execution_identity),
    )
    backend = standard_training.StandardProductionBackend(
        config=config,
        run_root=active_execution_authority["run_root"],
        repo_root=ROOT,
        stage5_authority=dict(
            active_execution_authority["stage5_handle"].identity
        ),
        execution_capability=active_execution_authority["capability"],
        _components=components,
        _immutable_bindings=immutable,
        _source_repair=source_repair,
    )
    monkeypatch.setattr(
        stage6_workflow,
        "load_stage4_policy_for_standard",
        lambda **kwargs: _CheckpointPolicy(),
    )
    retained_checkpoint_receipts: dict[tuple[int, int], object] = {}
    checkpoint_lineage = backend._checkpoint_lineage(10)
    for seed_index, seed in enumerate(config.training.seeds):
        torch.manual_seed(100 + seed_index)
        checkpoint_policy = _CheckpointPolicy()
        checkpoint_optimizer = torch.optim.AdamW(
            checkpoint_policy.parameters(),
            lr=config.ppo.learning_rate,
            eps=config.ppo.adam_eps,
            weight_decay=config.ppo.weight_decay,
        )
        checkpoint_loss = checkpoint_policy.linear(
            torch.ones((2, 3), dtype=torch.float32)
        ).square().mean()
        checkpoint_loss.backward()
        checkpoint_optimizer.step()
        checkpoint_optimizer.zero_grad(set_to_none=True)
        seed_best = standard_training.ValidationRecord(
            seed,
            10,
            0.7 - 0.01 * seed_index,
            0.4 - 0.01 * seed_index,
            "update-00000010",
        ).to_dict()
        checkpoint_manager = CheckpointManager(
            backend.stage_root / f"checkpoints/seed-{seed}",
            schema_version=config.checkpoint.schema_version,
        )
        for retained_update in (10, 50, 100):
            if retained_update != 10:
                with torch.no_grad():
                    for parameter in checkpoint_policy.parameters():
                        parameter.add_(retained_update / 100_000.0)
            retained_checkpoint_receipts[(seed, retained_update)] = (
                checkpoint_manager.save_complete(
                    policy=checkpoint_policy,
                    optimizer=checkpoint_optimizer,
                    update_step=retained_update,
                    normalization_stats={},
                    scenario_sampler_state={"workers": list(range(8))},
                    vector_env_states=[
                        {"worker": worker, "seed": seed}
                        for worker in range(8)
                    ],
                    best_record=seed_best,
                    versions=VERSIONS,
                    top_m_config={"frontier_top_m": 1024},
                    scale_profile="Standard v1",
                    training_config={"updates_per_seed": 100},
                        config_sha256=backend.config_sha256,
                        lineage=checkpoint_lineage,
                        eval_metrics={},
                        safety_contract=backend._safety_contract(),
                    )
            )
    (backend.stage_root / "preflight").mkdir(parents=True)
    (backend.stage_root / "preflight/audit.json").write_bytes(b'{"passed":true}\n')
    (backend.stage_root / "episode-traces").mkdir()
    (backend.stage_root / "lineage_audit.json").write_bytes(
        standard_training.ArtifactStore.canonical_json_bytes(
            {
                "schema_version": "stage6_lineage_audit/v2",
                "execution_identity": fake_execution_identity,
                "immutable_bindings": immutable,
                "verified_review_authorization": verified_review,
            }
        )
    )
    backend.record_preflight_phase()
    backend.record_preflight_phase()
    best = standard_training.ValidationRecord(
        config.training.seeds[0], 10, 0.7, 0.4, "update-00000010"
    )
    def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            b"".join(
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
        )

    receipt_rows: list[dict[str, object]] = []
    previous_receipt_hash = standard_training._INITIAL_RECORD_HASH
    for transaction in backend.transactions:
        retained_receipt = retained_checkpoint_receipts.get(
            (transaction.seed, transaction.update)
        )
        event = {
            "transaction_key": transaction.key,
            "seed": transaction.seed,
            "update": transaction.update,
            "checkpoint_sha256": (
                retained_receipt.checkpoint_sha256
                if retained_receipt is not None
                else sha256(
                    f"checkpoint:{transaction.key}".encode("ascii")
                ).hexdigest()
            ),
            "complete_marker_sha256": (
                sha256(retained_receipt.complete_marker_path.read_bytes()).hexdigest()
                if retained_receipt is not None
                else sha256(
                    f"complete:{transaction.key}".encode("ascii")
                ).hexdigest()
            ),
            "policy_state_sha256": (
                retained_receipt.policy_state_sha256
                if retained_receipt is not None
                else sha256(
                    f"policy:{transaction.key}".encode("ascii")
                ).hexdigest()
            ),
            "previous_record_hash": previous_receipt_hash,
        }
        row = {
            **event,
            "record_hash": standard_training._receipt_record_hash(event),
        }
        receipt_rows.append(row)
        previous_receipt_hash = str(row["record_hash"])
    write_jsonl(backend.receipt_index.path, receipt_rows)
    best_receipt = receipt_rows[9]
    receipt = standard_training.StandardTransactionCheckpoint(
        transaction_key=str(best_receipt["transaction_key"]),
        seed=int(best_receipt["seed"]),
        update=int(best_receipt["update"]),
        checkpoint_sha256=str(best_receipt["checkpoint_sha256"]),
        complete_marker_sha256=str(best_receipt["complete_marker_sha256"]),
        policy_state_sha256=str(best_receipt["policy_state_sha256"]),
    )
    backend.freeze_global_best(best)
    backend.record_final_evaluation_phase("test", "ppo_policy")
    backend.record_final_evaluation_phase("test", "ppo_policy")
    backend.record_final_evaluation_phase("unseen", "ppo_policy")
    backend.record_final_evaluation_phase("test", "random_valid_frontier")
    journal_rows: list[dict[str, object]] = []
    previous_journal_hash = stage6_workflow.ABSENT_AUTHORITY_SHA256
    for transaction, receipt_row in zip(backend.transactions, receipt_rows):
        bindings = {
            **immutable,
            "checkpoint_sha256": str(receipt_row["checkpoint_sha256"]),
        }
        for state in transaction.commit_states:
            event = {
                "state": state,
                "previous_record_hash": previous_journal_hash,
                "bindings": bindings,
            }
            row = {**event, "record_hash": _journal_record_hash(event)}
            journal_rows.append(row)
            previous_journal_hash = str(row["record_hash"])
    write_jsonl(backend.journal.path, journal_rows)
    validation_rows_data: list[dict[str, object]] = []
    validation_result_by_key: dict[str, dict[str, object]] = {}
    validation_best_by_seed: dict[int, dict[str, object]] = {}
    for item in backend.transactions:
        if item.validation_episodes != 16:
            continue
        seed_index = config.training.seeds.index(item.seed)
        result_record = {
            "episode_count": 16,
            "success_rate_under_fixed_step_budget": (
                0.7 - 0.01 * seed_index
                if item.update == 10
                else 0.6 - 0.01 * seed_index
            ),
            "mean_final_coverage": (
                0.4 - 0.01 * seed_index
                if item.update == 10
                else 0.3 - 0.01 * seed_index
            ),
        }
        selected = standard_training.select_validation_checkpoint_best(
            seed=item.seed,
            update=item.update,
            validation_metrics=result_record,
            previous_best=validation_best_by_seed.get(item.seed, {}),
        )
        validation_best_by_seed[item.seed] = selected
        validation_result_by_key[item.key] = result_record
        validation_rows_data.append(
            {
                "transaction_key": item.key,
                "seed": item.seed,
                "update": item.update,
                "episode_count": 16,
                "result": result_record,
                "eval_isolation": {"passed": True},
                "best_record": selected,
            }
        )
    shared_collection_audit = _collector_audit_dict(
        config.stage5_authority.policy_state_sha256,
        prefix="machine-verifier",
    )
    training_rows: list[dict[str, object]] = []
    math_rows: list[dict[str, object]] = []
    previous_policy_by_seed: dict[int, str] = {}
    for item, receipt_row in zip(backend.transactions, receipt_rows, strict=True):
        policy_before = previous_policy_by_seed.get(
            item.seed,
            config.stage5_authority.policy_state_sha256,
        )
        policy_after = str(receipt_row["policy_state_sha256"])
        collection_audit = {
            **shared_collection_audit,
            "policy_state_sha256": policy_before,
        }
        update_metrics = {
            "update_step": item.update,
            "initial_ratio_max_abs_error": 2.0e-6,
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "approx_kl": 0.003,
            "grad_pre_clip_norm_max": 0.7,
            "grad_post_clip_norm_max": 0.5,
            "optimizer_steps": 4,
            "policy_state_sha256_before": policy_before,
            "policy_state_sha256_after": policy_after,
            "math_evidence": _math_evidence_dict(
                collection_audit,
                policy_sha256=policy_before,
            ),
        }
        training_rows.append(
            {
                "transaction_key": item.key,
                "seed": item.seed,
                "update": item.update,
                "checkpoint_sha256": receipt_row["checkpoint_sha256"],
                "policy_state_sha256": policy_after,
                "update_metrics": update_metrics,
                "validation": validation_result_by_key.get(item.key, {}),
                "collection_audit": collection_audit,
            }
        )
        if item.update in {1, 10, 50, 100}:
            math_rows.append(
                {
                    "transaction_key": item.key,
                    "seed": item.seed,
                    "update": item.update,
                    **standard_training._math_audit_from_completed_update(
                        update_metrics,
                        collection_audit=collection_audit,
                        device=config.device,
                        compute_dtype=config.ppo.compute_dtype,
                    ),
                }
            )
        previous_policy_by_seed[item.seed] = policy_after
    write_jsonl(backend.stage_root / "training_metrics.jsonl", training_rows)
    write_jsonl(
        backend.stage_root / "validation_metrics.jsonl",
        validation_rows_data,
    )
    resource_payload = {
        "d_free_bytes": 200 * 1024**3,
        "rss_bytes": 1024,
        "peak_vram_bytes": 2048,
        "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
        "rss_root_pid": os.getpid(),
        "rss_sample_count": 2,
        "rss_latest_process_count": 1,
        "rss_peak_process_count": 1,
        "warnings": [],
        "hard_stops": [],
        "passed": True,
    }
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        append_resource_segment_start,
        bind_resource_row_to_segment,
    )

    resource_path = backend.stage_root / "resource_audit.jsonl"
    resource_segment = append_resource_segment_start(
        resource_path,
        first_sample={
            **resource_payload,
            "rss_sample_count": 1,
        },
    )
    backend._resource_segment_start = resource_segment
    resource_rows: list[dict[str, object]] = [resource_segment]
    receipt_by_transaction = {
        str(row["transaction_key"]): row for row in receipt_rows
    }

    def append_resource_unit(identity: dict[str, object], *, attempt: int = 1) -> None:
        for phase in ("pre", "post"):
            resource_rows.append(
                bind_resource_row_to_segment(
                    {
                    "schema_version": "stage6_resource_attempt/v1",
                    **identity,
                    "attempt": attempt,
                    "phase": phase,
                    "accepted": False,
                    "resource": resource_payload,
                    },
                    resource_segment,
                )
            )
        acceptance = {
            "schema_version": (
                "stage6_resource_acceptance/v2"
                if identity["kind"] == "update"
                else "stage6_resource_acceptance/v1"
            ),
            **identity,
            "attempt": attempt,
            "phase": "accepted",
            "accepted": True,
            "pre": resource_payload,
            "post": resource_payload,
        }
        if identity["kind"] == "update":
            receipt_row = receipt_by_transaction[str(identity["transaction_key"])]
            acceptance["checkpoint"] = {
                name: receipt_row[name]
                for name in (
                    "transaction_key",
                    "seed",
                    "update",
                    "checkpoint_sha256",
                    "complete_marker_sha256",
                    "policy_state_sha256",
                )
            }
        resource_rows.append(
            bind_resource_row_to_segment(acceptance, resource_segment)
        )

    for item in backend.transactions:
        append_resource_unit(
            {
                "kind": "update",
                "transaction_key": item.key,
                "seed": item.seed,
                "update": item.update,
            },
            attempt=1,
        )
    for split in ("test", "unseen"):
        for method in STANDARD_EVALUATION_METHODS:
            append_resource_unit(
                {
                    "kind": "final_evaluation",
                    "transaction_key": f"final:{split}:{method}",
                    "split": split,
                    "method": method,
                }
            )
    write_jsonl(resource_path, resource_rows)
    audited_transactions = [
        item for item in backend.transactions if item.update in {1, 10, 50, 100}
    ]
    write_jsonl(backend.stage_root / "math_audit.jsonl", math_rows)
    checkpoint_lineage_sha256 = standard_training._runtime_value_sha256(
        checkpoint_lineage
    )
    checkpoint_audit_rows = []
    for item in audited_transactions:
        bound_receipt = receipt_by_transaction[item.key]
        checkpoint_audit_rows.append(
            {
                "transaction_key": item.key,
                "seed": item.seed,
                "update": item.update,
                "schema_version": "stage6_checkpoint_replay_audit/v1",
                "passed": True,
                "last_complete_loaded": True,
                "policy_state_unchanged": True,
                "optimizer_state_unchanged": True,
                "rng_state_unchanged": True,
                "normalizer_restored": True,
                "scenario_sampler_state_restored": True,
                "vector_env_state_count": 8,
                "vector_env_state_round_trip_bit_exact": True,
                "observation_round_trip_bit_exact": True,
                "deterministic_action_bit_exact": True,
                "checkpoint_sha256": bound_receipt["checkpoint_sha256"],
                "complete_marker_sha256": bound_receipt[
                    "complete_marker_sha256"
                ],
                "policy_state_sha256": bound_receipt["policy_state_sha256"],
                "lineage_sha256": checkpoint_lineage_sha256,
                "lineage_unchanged": True,
            }
        )
    write_jsonl(
        backend.stage_root / "checkpoint_audit.jsonl",
        checkpoint_audit_rows,
    )
    evaluations = {
        f"{split}:{method}": _deterministic_final_summary(
            method,
            split,
            str(backend._global_best_identity["policy_state_sha256"]),
        )
        for split in ("test", "unseen")
        for method in STANDARD_EVALUATION_METHODS
    }
    from lunar_exploration_ppo.eval.metrics import episode_record
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore

    trace_store = ArtifactStore(backend.stage_root / "episode-traces")
    for key, evaluation in evaluations.items():
        split, method = key.split(":", 1)
        trace_name = f"final-{split}-{method}.jsonl"
        for episode in evaluation.episodes:
            trace_store.append_jsonl(
                trace_name,
                {
                    **episode_record(episode),
                    "steps_executed": 1,
                },
            )
        trace_bytes = (trace_store.root / trace_name).read_bytes()
        summary_name = f"final-{split}-{method}.summary.json"
        trace_store.write_json_exclusive(
            summary_name,
            {
                "schema_version": "stage6_final_eval_completion/v1",
                "split": split,
                "method": method,
                "episode_count": 64,
                "trace": {
                    "path": trace_name,
                    "sha256": sha256(trace_bytes).hexdigest(),
                    "size_bytes": len(trace_bytes),
                },
                "config_sha256": backend.config_sha256,
                "checkpoint_sha256": backend._global_best_identity[
                    "checkpoint_sha256"
                ],
                "policy_state_sha256": backend._global_best_identity[
                    "policy_state_sha256"
                ],
                "result": backend._evaluation_result_record(evaluation),
            },
        )
        trace_store.write_json_exclusive(
            f"final-{split}-{method}.commit.json",
            backend._final_eval_commit_value(
                trace_path=trace_store.root / trace_name,
                summary_path=trace_store.root / summary_name,
                split=split,
                method=method,
                attempt=1,
            ),
        )
    backend._final_eval_audits = {
        key: {"passed": True} for key in evaluations
    }

    checkpoint_audit_path = backend.stage_root / "checkpoint_audit.jsonl"
    valid_checkpoint_audit = checkpoint_audit_path.read_bytes()
    preterminal_bad_audits = [dict(row) for row in checkpoint_audit_rows]
    preterminal_bad_audits[0]["vector_env_state_count"] = 7
    write_jsonl(checkpoint_audit_path, preterminal_bad_audits)
    with pytest.raises(
        standard_training.StandardTrainingError,
        match="machine|verification|checkpoint",
    ):
        backend.finalize(best, evaluations)
    assert [
        row["state"] for row in backend.phase_journal.verify()
    ] == [
        "preflight",
        "global_best_frozen",
        "final_test_running",
        "final_unseen_running",
        "baselines_running",
    ]
    assert not (backend.stage_root / "manifest.json").exists()
    assert not (backend.stage_root / "preterminal_acceptance.json").exists()
    checkpoint_audit_path.write_bytes(valid_checkpoint_audit)

    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    original_pending_verifier = stage6_workflow.verify_stage6_pending_acceptance
    original_receipt_writer = (
        stage6_terminal_recovery.write_stage6_preterminal_acceptance
    )

    def capture_verified_pending_semantic() -> dict[str, object]:
        verified_results: list[dict[str, object]] = []
        writer_arguments: dict[str, object] = {}

        def recording_pending_verifier(**kwargs: object) -> dict[str, object]:
            result = original_pending_verifier(**kwargs)
            assert result["passed"] is True
            verified_results.append(result)
            return result

        def capture_before_writer(**kwargs: object) -> dict[str, object]:
            writer_arguments.update(kwargs)
            semantic_result = kwargs.get("semantic_result")
            assert semantic_result is verified_results[-1]
            raise stage6_terminal_recovery.TerminalRecoveryError(
                "deterministic verify-return writer-capture seam"
            )

        monkeypatch.setattr(
            stage6_workflow,
            "verify_stage6_pending_acceptance",
            recording_pending_verifier,
        )
        monkeypatch.setattr(
            stage6_terminal_recovery,
            "write_stage6_preterminal_acceptance",
            capture_before_writer,
        )
        try:
            with pytest.raises(
                standard_training.StandardTrainingError,
                match="pending Stage 6 machine verification failed",
            ) as raised:
                backend.finalize(best, evaluations)
        finally:
            monkeypatch.setattr(
                stage6_workflow,
                "verify_stage6_pending_acceptance",
                original_pending_verifier,
            )
            monkeypatch.setattr(
                stage6_terminal_recovery,
                "write_stage6_preterminal_acceptance",
                original_receipt_writer,
            )
        causes: list[str] = []
        cause: BaseException | None = raised.value
        while cause is not None:
            causes.append(repr(cause))
            cause = cause.__cause__
        assert len(verified_results) == 1, " <- ".join(causes)
        assert writer_arguments["semantic_result"] is verified_results[0]
        assert not (backend.stage_root / "preterminal_acceptance.json").exists()
        return writer_arguments

    fairness_path = backend.stage_root / "fairness_audit.json"
    valid_fairness_bytes = fairness_path.read_bytes()
    failed_fairness = json.loads(valid_fairness_bytes.decode("utf-8"))
    failed_fairness["passed"] = False
    failed_fairness_bytes = standard_training.ArtifactStore.canonical_json_bytes(
        failed_fairness
    )

    atomic_replace_arguments = capture_verified_pending_semantic()
    replacement_path = fairness_path.with_name("fairness_audit.replace.tmp")
    replacement_path.write_bytes(failed_fairness_bytes)
    os.replace(replacement_path, fairness_path)
    try:
        with pytest.raises(stage6_terminal_recovery.TerminalRecoveryError):
            original_receipt_writer(**atomic_replace_arguments)
    finally:
        fairness_path.write_bytes(valid_fairness_bytes)
    assert not (backend.stage_root / "preterminal_acceptance.json").exists()

    same_inode_arguments = capture_verified_pending_semantic()
    fairness_inode = fairness_path.stat().st_ino
    with fairness_path.open("r+b") as stream:
        stream.seek(0)
        stream.write(failed_fairness_bytes)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
    assert fairness_path.stat().st_ino == fairness_inode
    try:
        with pytest.raises(stage6_terminal_recovery.TerminalRecoveryError):
            original_receipt_writer(**same_inode_arguments)
    finally:
        with fairness_path.open("r+b") as stream:
            stream.seek(0)
            stream.write(valid_fairness_bytes)
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
    assert fairness_path.stat().st_ino == fairness_inode
    assert not (backend.stage_root / "preterminal_acceptance.json").exists()

    # S6-QS-002A: the pending verifier must make every formal Stage 6
    # decision from the one captured evidence handle.  Keep the legacy path
    # APIs functional here so the real verifier can traverse the complete
    # graph, but record every forbidden reopen.  The pre-fix implementation
    # reaches all eight reader categories and therefore fails the final
    # zero-call assertion deterministically.
    from lunar_exploration_ppo.configs import stage6 as stage6_config_module

    pending_terminal_artifacts = atomic_replace_arguments["terminal_artifacts"]
    assert isinstance(pending_terminal_artifacts, dict)
    pending_summary = json.loads(
        pending_terminal_artifacts["summary.json"].decode("utf-8")
    )
    pending_routing = json.loads(
        pending_terminal_artifacts["routing.json"].decode("utf-8")
    )
    pending_reports = {
        name: payload
        for name, payload in pending_terminal_artifacts.items()
        if name not in {"summary.json", "routing.json"}
    }
    forbidden_pending_path_calls: list[str] = []
    original_config_loader = stage6_config_module.load_stage6_config
    original_journal_verify = Stage6StateJournal.verify
    original_final_eval_path_verify = (
        standard_training.verify_standard_final_evaluation_artifacts
    )
    original_receipt_index_verify = standard_training.CheckpointReceiptIndex.verify
    original_retention_init = CheckpointRetentionManager.__init__
    original_retention_verify = CheckpointRetentionManager.verify_retained
    original_checkpoint_init = CheckpointManager.__init__
    original_checkpoint_inspect = CheckpointManager.inspect_complete
    original_checkpoint_load = CheckpointManager.load_complete

    def record_config_path_read(*args, **kwargs):
        forbidden_pending_path_calls.append("config")
        return original_config_loader(*args, **kwargs)

    def record_journal_path_read(self, *args, **kwargs):
        forbidden_pending_path_calls.append(f"journal:{self.path.name}")
        return original_journal_verify(self, *args, **kwargs)

    def record_final_eval_path_read(*args, **kwargs):
        trace_path = Path(kwargs["trace_path"])
        forbidden_pending_path_calls.append(f"final-eval:{trace_path.name}")
        return original_final_eval_path_verify(*args, **kwargs)

    def record_receipt_index_path_read(self, *args, **kwargs):
        forbidden_pending_path_calls.append("checkpoint-receipt-index")
        return original_receipt_index_verify(self, *args, **kwargs)

    def record_retention_init(self, *args, **kwargs):
        forbidden_pending_path_calls.append("checkpoint-retention:init")
        original_retention_init(self, *args, **kwargs)

    def record_retention_path_read(self, *args, **kwargs):
        forbidden_pending_path_calls.append("checkpoint-retention:verify")
        return original_retention_verify(self, *args, **kwargs)

    def record_checkpoint_init(self, *args, **kwargs):
        forbidden_pending_path_calls.append("checkpoint:init")
        original_checkpoint_init(self, *args, **kwargs)

    def record_checkpoint_inspect_path_read(self, *args, **kwargs):
        forbidden_pending_path_calls.append("checkpoint:inspect")
        return original_checkpoint_inspect(self, *args, **kwargs)

    def record_checkpoint_load_path_read(self, *args, **kwargs):
        forbidden_pending_path_calls.append("checkpoint:load")
        return original_checkpoint_load(self, *args, **kwargs)

    with monkeypatch.context() as pending_reader_patch:
        pending_reader_patch.setattr(
            stage6_config_module,
            "load_stage6_config",
            record_config_path_read,
        )
        pending_reader_patch.setattr(
            Stage6StateJournal,
            "verify",
            record_journal_path_read,
        )
        pending_reader_patch.setattr(
            standard_training,
            "verify_standard_final_evaluation_artifacts",
            record_final_eval_path_read,
        )
        pending_reader_patch.setattr(
            standard_training.CheckpointReceiptIndex,
            "verify",
            record_receipt_index_path_read,
        )
        pending_reader_patch.setattr(
            CheckpointRetentionManager,
            "__init__",
            record_retention_init,
        )
        pending_reader_patch.setattr(
            CheckpointRetentionManager,
            "verify_retained",
            record_retention_path_read,
        )
        pending_reader_patch.setattr(
            CheckpointManager,
            "__init__",
            record_checkpoint_init,
        )
        pending_reader_patch.setattr(
            CheckpointManager,
            "inspect_complete",
            record_checkpoint_inspect_path_read,
        )
        pending_reader_patch.setattr(
            CheckpointManager,
            "load_complete",
            record_checkpoint_load_path_read,
        )
        path_bound_pending = original_pending_verifier(
            stage_root=backend.stage_root,
            repo_root=ROOT,
            summary=pending_summary,
            routing=pending_routing,
            reports=pending_reports,
        )
    assert path_bound_pending["passed"] is True
    assert forbidden_pending_path_calls == []

    pending = backend.finalize(best, evaluations)
    receipt_path = backend.stage_root / "preterminal_acceptance.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "stage6_preterminal_acceptance/v2"
    assert receipt["semantic_verification"]["passed"] is True
    assert receipt["immutable_bindings"] == immutable
    assert receipt["global_checkpoint_identity"] == backend._global_best_identity
    assert [row["path"] for row in receipt["terminal_artifacts"]] == [
        "summary.json",
        "routing.json",
        "standard_training_report.md",
        "standard_eval_report.md",
        "report.md",
    ]
    assert backend._preterminal_acceptance_identity == {
        "sha256": sha256(receipt_path.read_bytes()).hexdigest(),
        "size_bytes": len(receipt_path.read_bytes()),
    }

    assert set(pending) == {"stage_root", "summary", "routing"}
    assert pending["summary"]["state"] == "awaiting_independent_review"
    assert pending["summary"]["performance_advantage_established"] is True
    assert pending["summary"]["acceptance_scope"] == (
        "single_seed_system_closure/v1"
    )
    assert pending["summary"]["cross_seed_performance_conclusion"] is False
    assert pending["summary"]["optional_seed_extension_blocks_next_stage"] is False
    assert pending["summary"]["optional_seed_extension_trigger"] == (
        "explicit_user_request_only/v1"
    )
    assert pending["summary"]["environment_identity"] == immutable[
        "environment_identity"
    ]
    assert pending["summary"]["environment_sha256"] == immutable[
        "environment_sha256"
    ]
    assert pending["routing"]["route"] == "awaiting_independent_review"
    assert pending["routing"]["next_stage_after_human_approval"] == (
        "ppo_highres_frontier_stage7_kilometer_stress/v1"
    )
    success_paths = (
        "summary.json",
        "routing.json",
        "standard_training_report.md",
        "standard_eval_report.md",
        "report.md",
        "manifest.json",
    )
    assert all(not (backend.stage_root / name).exists() for name in success_paths)
    assert [row["state"] for row in backend.phase_journal.verify()] == [
        "preflight",
        "global_best_frozen",
        "final_test_running",
        "final_unseen_running",
        "baselines_running",
    ]

    backend._capture_resource_gate = lambda: {  # type: ignore[method-assign]
        **resource_payload,
        "rss_bytes": 8192,
        "rss_sample_count": 3,
        "passed": False,
        "hard_stops": ["rss_hard_stop"],
    }
    backend._process_tree_monitor = SimpleNamespace(running=False)
    with pytest.raises(standard_training.StandardTrainingError, match="hard gate"):
        backend.record_terminal_resource_evidence()
    assert all(not (backend.stage_root / name).exists() for name in success_paths)

    backend._capture_resource_gate = lambda: {  # type: ignore[method-assign]
        **resource_payload,
        "rss_bytes": 8192,
        "rss_sample_count": 3,
    }
    terminal_resource = backend.record_terminal_resource_evidence()
    assert terminal_resource["resource"]["rss_bytes"] == 8192  # type: ignore[index]
    assert terminal_resource["schema_version"] == (
        "stage6_terminal_resource_evidence/v4"
    )
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery

    original_preterminal_verifier = (
        stage6_workflow.verify_stage6_preterminal_acceptance
    )
    original_final_verifier = stage6_workflow.verify_stage6_machine_acceptance

    def forbidden_semantic_verifier(**kwargs):
        del kwargs
        raise AssertionError("terminal commit must not replay semantic verification")

    monkeypatch.setattr(
        stage6_workflow,
        "verify_stage6_preterminal_acceptance",
        forbidden_semantic_verifier,
    )
    monkeypatch.setattr(
        stage6_workflow,
        "verify_stage6_machine_acceptance",
        forbidden_semantic_verifier,
    )

    def crash_after_machine_passed(event: str) -> None:
        if event == "after_phase:machine_passed":
            raise OSError("injected crash after machine_passed phase")

    monkeypatch.setattr(
        stage6_terminal_recovery,
        "_terminal_recovery_event",
        crash_after_machine_passed,
    )
    with pytest.raises(OSError, match="injected crash"):
        backend.commit_terminal_acceptance(pending)
    assert [row["state"] for row in backend.phase_journal.verify()] == [
        "preflight",
        "global_best_frozen",
        "final_test_running",
        "final_unseen_running",
        "baselines_running",
        "machine_passed",
    ]
    assert not (backend.stage_root / "manifest.json").exists()
    assert all(
        (backend.stage_root / name).is_file()
        for name in success_paths
        if name != "manifest.json"
    )

    monkeypatch.setattr(
        stage6_terminal_recovery,
        "_terminal_recovery_event",
        lambda event: None,
    )
    result = backend.commit_terminal_acceptance(pending)

    assert result == pending
    assert (backend.stage_root / "report.md").read_text(encoding="utf-8") == (
        "# Stage 6 Standard v1\n\n"
        "机器验收已完成，状态为 awaiting_independent_review。\n\n"
        "单 seed 只证明系统闭环，不形成跨 seed 性能结论；"
        "可选追加不阻塞 Stage 7/8。\n\n"
        "未写入 review、approval 或 gate authority artifact。\n"
    )
    assert {path.name for path in backend.stage_root.iterdir()} == {
        *stage6_workflow.STAGE6_MANIFEST_BOUND_ARTIFACTS,
        "manifest.json",
        "checkpoints",
        "episode-traces",
        "preflight",
    }
    handle = stage6_workflow.verify_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    handle.require_current()
    monkeypatch.setattr(
        stage6_workflow,
        "verify_stage6_preterminal_acceptance",
        original_preterminal_verifier,
    )
    monkeypatch.setattr(
        stage6_workflow,
        "verify_stage6_machine_acceptance",
        original_final_verifier,
    )

    def forbidden_terminal_workload(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "final receipt verifier must not load model/optimizer/checkpoint/evaluator"
        )

    monkeypatch.setattr(
        stage6_workflow,
        "load_stage4_policy_for_standard",
        forbidden_terminal_workload,
    )
    monkeypatch.setattr(torch.optim, "AdamW", forbidden_terminal_workload)
    monkeypatch.setattr(
        CheckpointManager,
        "load_complete",
        forbidden_terminal_workload,
    )
    monkeypatch.setattr(
        standard_training,
        "verify_standard_final_evaluation_artifacts",
        forbidden_terminal_workload,
    )
    acceptance = stage6_workflow.verify_stage6_machine_acceptance(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    assert acceptance["passed"] is True
    assert acceptance["checkpoint_receipt_count"] == 100

    original_terminal_detector = (
        stage6_terminal_recovery.detect_stage6_terminal_recovery
    )
    summary_path = backend.stage_root / "summary.json"
    terminal_summary_bytes = summary_path.read_bytes()

    def assert_terminal_verifier_race_rejected(mode: str) -> None:
        drifted_summary = json.loads(terminal_summary_bytes.decode("utf-8"))
        drifted_summary["s6_qs_002_terminal_race"] = mode
        drifted_bytes = standard_training.ArtifactStore.canonical_json_bytes(
            drifted_summary
        )
        replacement = summary_path.with_name("summary.replace.tmp")

        def detector_then_mutate(**kwargs: object) -> dict[str, object]:
            detection = original_terminal_detector(**kwargs)
            assert detection["status"] == "valid_terminal_recovery"
            if mode == "atomic_replace":
                replacement.write_bytes(drifted_bytes)
                os.replace(replacement, summary_path)
            elif mode == "same_inode":
                original_inode = summary_path.stat().st_ino
                with summary_path.open("r+b") as stream:
                    stream.seek(0)
                    stream.write(drifted_bytes)
                    stream.truncate()
                    stream.flush()
                    os.fsync(stream.fileno())
                assert summary_path.stat().st_ino == original_inode
            else:
                raise AssertionError(f"unknown terminal race mode: {mode}")
            return detection

        monkeypatch.setattr(
            stage6_terminal_recovery,
            "detect_stage6_terminal_recovery",
            detector_then_mutate,
        )
        try:
            with pytest.raises(
                stage6_workflow.Stage6WorkflowError,
                match="evidence|bound|recovery|changed",
            ):
                original_final_verifier(
                    stage_root=backend.stage_root,
                    repo_root=ROOT,
                )
        finally:
            monkeypatch.setattr(
                stage6_terminal_recovery,
                "detect_stage6_terminal_recovery",
                original_terminal_detector,
            )
            if summary_path.read_bytes() != terminal_summary_bytes:
                summary_path.write_bytes(terminal_summary_bytes)
            if replacement.exists():
                replacement.unlink()

    assert_terminal_verifier_race_rejected("atomic_replace")
    assert_terminal_verifier_race_rejected("same_inode")
    stage6_workflow.verify_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    ).require_current()

    original_recovery_read_plain = stage6_terminal_recovery._read_plain
    raw_terminal_reads: list[str] = []
    bound_terminal_names = {
        "preterminal_acceptance.json",
        "resource_audit.jsonl",
        "phase-state.jsonl",
        "manifest.json",
        *stage6_terminal_recovery.TERMINAL_ARTIFACT_NAMES,
    }

    def reject_raw_terminal_reopen(
        stage: Path,
        relative: str,
        *,
        label: str,
    ) -> bytes:
        if relative in bound_terminal_names:
            raw_terminal_reads.append(relative)
            raise AssertionError(
                f"terminal public verifier raw-reopened {relative}: {label}"
            )
        return original_recovery_read_plain(stage, relative, label=label)

    monkeypatch.setattr(
        stage6_terminal_recovery,
        "_read_plain",
        reject_raw_terminal_reopen,
    )
    try:
        bound_acceptance = original_final_verifier(
            stage_root=backend.stage_root,
            repo_root=ROOT,
        )
    finally:
        monkeypatch.setattr(
            stage6_terminal_recovery,
            "_read_plain",
            original_recovery_read_plain,
        )
    assert bound_acceptance == acceptance
    assert raw_terminal_reads == []

    from test_stage6_workflow import _FakeInputPin

    original_pinned_read = _FakeInputPin.read_bytes
    ephemeral_reads: list[str] = []

    def inject_ephemeral_summary_bytes(
        self: object,
        path: str | Path,
        *,
        label: str,
    ) -> bytes:
        payload = original_pinned_read(self, path, label=label)
        if Path(path).name == "summary.json" and not ephemeral_reads:
            alternate = json.loads(payload.decode("utf-8"))
            alternate["s6_qs_002_ephemeral_replace_restore"] = True
            ephemeral_reads.append(label)
            assert summary_path.read_bytes() == terminal_summary_bytes
            return standard_training.ArtifactStore.canonical_json_bytes(alternate)
        return payload

    monkeypatch.setattr(
        _FakeInputPin,
        "read_bytes",
        inject_ephemeral_summary_bytes,
    )
    try:
        with pytest.raises(
            stage6_workflow.Stage6WorkflowError,
            match="continuously bound|pinned|changed",
        ):
            original_final_verifier(
                stage_root=backend.stage_root,
                repo_root=ROOT,
            )
    finally:
        monkeypatch.setattr(
            _FakeInputPin,
            "read_bytes",
            original_pinned_read,
        )
    assert len(ephemeral_reads) == 1
    assert summary_path.read_bytes() == terminal_summary_bytes

    assert backend.commit_terminal_acceptance(pending) == result
    metric_names = (
        "success_rate_under_fixed_step_budget",
        "mean_final_coverage",
        "steps_to_99_success_only",
        "path_length_to_99_success_only",
        "coverage_auc_over_steps",
        "coverage_per_meter",
        "invalid_action_count_mean",
        "planner_failure_count_mean",
        "safety_violation_count",
    )
    with (backend.stage_root / "standard_baseline_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        comparison = list(csv.DictReader(stream))
    assert len(comparison) == 10
    assert set(comparison[0]) == {
        "split",
        "method",
        "episode_count",
        *metric_names,
        *(f"{name}_ci95_low" for name in metric_names),
        *(f"{name}_ci95_high" for name in metric_names),
    }
    with (backend.stage_root / "standard_coverage_curves.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        curves = list(csv.DictReader(stream))
    assert len(curves) == 10 * 129
    assert set(curves[0]) == {"split", "method", "step", "mean_coverage"}
    for key in evaluations:
        split, method = key.split(":", 1)
        rows = [
            row
            for row in curves
            if row["split"] == split and row["method"] == method
        ]
        values = [float(row["mean_coverage"]) for row in rows]
        assert [int(row["step"]) for row in rows] == list(range(129))
        assert all(left <= right for left, right in zip(values, values[1:]))
        assert values[-1] == pytest.approx(
            float(evaluations[key].metrics["mean_final_coverage"])
        )
    failure = json.loads(
        (backend.stage_root / "failure_audit.json").read_text(encoding="utf-8")
    )
    assert failure["passed"] is True
    assert failure["episode_count"] == 640
    assert failure["safety_violation_count"] == 0
    assert failure["termination_classification"] == {
        "failure_done": 512,
        "success_done": 128,
    }
    fairness = json.loads(
        (backend.stage_root / "fairness_audit.json").read_text(encoding="utf-8")
    )
    assert fairness["schema_version"] == "stage6_final_fairness_audit/v2"
    assert fairness["passed"] is True
    assert set(fairness["cohorts"]) == {"test", "unseen"}
    assert all(row["passed"] is True for row in fairness["cohorts"].values())
    leakage = json.loads(
        (backend.stage_root / "leakage_audit.json").read_text(encoding="utf-8")
    )
    assert leakage == {
        "schema_version": "stage6_final_leakage_audit/v2",
        "passed": True,
        "runtime_truth_like_selector_field_count": 0,
        "frozen_truth_like_observation_field_count": 0,
    }
    phase_states = [
        json.loads(line)["state"]
        for line in (backend.stage_root / "phase-state.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert phase_states == [
        "preflight",
        "global_best_frozen",
        "final_test_running",
        "final_unseen_running",
        "baselines_running",
        "machine_passed",
        "awaiting_independent_review",
    ]
    assert not any(
        (backend.stage_root / name).exists()
        for name in ("review.json", "approval.json", "gate.json")
    )
    receipt_path = backend.stage_root / "preterminal_acceptance.json"
    original_receipt = receipt_path.read_bytes()
    resource_path = backend.stage_root / "resource_audit.jsonl"
    original_resource = resource_path.read_bytes()
    synchronized_receipt = json.loads(original_receipt.decode("utf-8"))
    synchronized_receipt["semantic_verification"]["final_evaluation_count"] = 9
    synchronized_receipt_bytes = standard_training.ArtifactStore.canonical_json_bytes(
        synchronized_receipt
    )
    receipt_path.write_bytes(synchronized_receipt_bytes)
    synchronized_receipt_identity = {
        "sha256": sha256(synchronized_receipt_bytes).hexdigest(),
        "size_bytes": len(synchronized_receipt_bytes),
    }
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        build_bound_terminal_resource_evidence,
    )

    synchronized_rows = [
        json.loads(line) for line in original_resource.decode("utf-8").splitlines()
    ]
    synchronized_rows[-1] = build_bound_terminal_resource_evidence(
        synchronized_rows[:-1],
        terminal_resource=synchronized_rows[-1]["resource"],
        preterminal_acceptance=synchronized_receipt_identity,
    )
    write_jsonl(resource_path, synchronized_rows)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    with pytest.raises(
        Stage6WorkflowError,
        match="receipt-fixed|semantic|summary|receipt",
    ):
        stage6_workflow.verify_stage6_machine_acceptance(
            stage_root=backend.stage_root,
            repo_root=ROOT,
        )
    receipt_path.write_bytes(original_receipt)
    resource_path.write_bytes(original_resource)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    rolled_back_terminal_rows = [
        json.loads(line) for line in original_resource.decode("utf-8").splitlines()
    ]
    rolled_back_terminal_rows[-1]["resource"]["rss_sample_count"] = 2
    write_jsonl(resource_path, rolled_back_terminal_rows)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    with pytest.raises(Stage6WorkflowError, match="resource|terminal"):
        stage6_workflow.verify_stage6_machine_acceptance(
            stage_root=backend.stage_root,
            repo_root=ROOT,
        )
    resource_path.write_bytes(original_resource)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    forged_prior_rows = [
        json.loads(line) for line in original_resource.decode("utf-8").splitlines()
    ]
    forged_prior_rows[-1]["prior_resource_audit"]["sha256"] = "f" * 64
    write_jsonl(resource_path, forged_prior_rows)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    with pytest.raises(Stage6WorkflowError, match="manifest|terminal|resource"):
        stage6_workflow.verify_stage6_machine_acceptance(
            stage_root=backend.stage_root,
            repo_root=ROOT,
        )
    resource_path.write_bytes(original_resource)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    mutated_resource = original_resource.replace(
        b'"peak_vram_bytes":2048', b'"peak_vram_bytes":10844792423', 1
    )
    assert mutated_resource != original_resource
    resource_path.write_bytes(mutated_resource)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    with pytest.raises(Stage6WorkflowError, match="resource"):
        stage6_workflow.verify_stage6_machine_acceptance(
            stage_root=backend.stage_root,
            repo_root=ROOT,
        )
    resource_path.write_bytes(original_resource)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    tampered_resource_rows = [
        json.loads(line) for line in original_resource.decode("utf-8").splitlines()
    ]
    tampered_resource_rows[0]["first_sample"]["rss_sample_count"] = 3
    write_jsonl(resource_path, tampered_resource_rows)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    with pytest.raises(Stage6WorkflowError, match="resource"):
        stage6_workflow.verify_stage6_machine_acceptance(
            stage_root=backend.stage_root,
            repo_root=ROOT,
        )
    resource_path.write_bytes(original_resource)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    trace_path = backend.stage_root / "episode-traces/final-test-ppo_policy.jsonl"
    original_trace = trace_path.read_bytes()
    mutated_trace = original_trace.replace(
        b'"final_coverage":1.0', b'"final_coverage":0.9', 1
    )
    assert mutated_trace != original_trace
    trace_path.write_bytes(mutated_trace)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    with pytest.raises(Stage6WorkflowError, match="evaluation|trace"):
        stage6_workflow.verify_stage6_machine_acceptance(
            stage_root=backend.stage_root,
            repo_root=ROOT,
        )
    trace_path.write_bytes(original_trace)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    assert stage6_workflow.verify_stage6_machine_acceptance(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )["passed"] is True
    training_path = backend.stage_root / "training_metrics.jsonl"
    original_training = training_path.read_bytes()
    tampered_training_rows = [dict(row) for row in training_rows]
    tampered_update = dict(tampered_training_rows[0]["update_metrics"])
    tampered_update["policy_state_sha256_before"] = "f" * 64
    tampered_training_rows[0]["update_metrics"] = tampered_update
    write_jsonl(training_path, tampered_training_rows)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    with pytest.raises(Stage6WorkflowError, match="policy|on-policy|training"):
        stage6_workflow.verify_stage6_machine_acceptance(
            stage_root=backend.stage_root,
            repo_root=ROOT,
        )
    training_path.write_bytes(original_training)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    assert authority_calls
    authority_checkpoint.write_bytes(b"tampered-stage4-checkpoint")
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    with pytest.raises(Stage6WorkflowError, match="authority"):
        stage6_workflow.verify_stage6_machine_acceptance(
            stage_root=backend.stage_root,
            repo_root=ROOT,
        )
    authority_checkpoint.write_bytes(authority_checkpoint_bytes)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    lineage_path = backend.stage_root / "lineage_audit.json"
    original_lineage = lineage_path.read_bytes()
    drifted_lineage = json.loads(original_lineage.decode("utf-8"))
    drifted_lineage["execution_identity"]["source_set_sha256"] = "f" * 64
    lineage_path.write_bytes(
        standard_training.ArtifactStore.canonical_json_bytes(drifted_lineage)
    )
    (backend.stage_root / "manifest.json").unlink()
    with pytest.raises(Stage6WorkflowError, match="review|lineage|identity"):
        stage6_workflow.write_stage6_manifest(
            stage_root=backend.stage_root,
            repo_root=ROOT,
        )
    lineage_path.write_bytes(original_lineage)
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    validation_path = backend.stage_root / "validation_metrics.jsonl"
    original_validation = validation_path.read_bytes()
    tampered_validation_rows = [dict(row) for row in validation_rows_data]
    tampered_result = dict(tampered_validation_rows[0]["result"])
    tampered_result["success_rate_under_fixed_step_budget"] = 0.1
    tampered_validation_rows[0]["result"] = tampered_result
    write_jsonl(validation_path, tampered_validation_rows)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    with pytest.raises(Stage6WorkflowError, match="validation|global best"):
        stage6_workflow.verify_stage6_machine_acceptance(
            stage_root=backend.stage_root,
            repo_root=ROOT,
        )
    validation_path.write_bytes(original_validation)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    global_checkpoint_path = (
        backend.stage_root
        / "checkpoints/seed-20260716/update-00000010/checkpoint.pt"
    )
    original_global_checkpoint = global_checkpoint_path.read_bytes()
    global_checkpoint_path.write_bytes(
        original_global_checkpoint[:-1]
        + bytes((original_global_checkpoint[-1] ^ 0x01,))
    )
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    with pytest.raises(Stage6WorkflowError, match="checkpoint|global best"):
        stage6_workflow.verify_stage6_machine_acceptance(
            stage_root=backend.stage_root,
            repo_root=ROOT,
        )
    global_checkpoint_path.write_bytes(original_global_checkpoint)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    checkpoint_audit_path = backend.stage_root / "checkpoint_audit.jsonl"
    original_checkpoint_audit = checkpoint_audit_path.read_bytes()
    tampered_checkpoint_audits = [dict(row) for row in checkpoint_audit_rows]
    tampered_checkpoint_audits[0]["vector_env_state_count"] = 7
    write_jsonl(checkpoint_audit_path, tampered_checkpoint_audits)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    with pytest.raises(Stage6WorkflowError, match="checkpoint"):
        stage6_workflow.verify_stage6_machine_acceptance(
            stage_root=backend.stage_root,
            repo_root=ROOT,
        )
    checkpoint_audit_path.write_bytes(original_checkpoint_audit)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    summary_path = backend.stage_root / "summary.json"
    routing_path = backend.stage_root / "routing.json"
    original_summary = summary_path.read_bytes()
    original_routing = routing_path.read_bytes()
    for field, tampered_value in (
        ("acceptance_scope", "multi_seed_performance/v1"),
        ("cross_seed_performance_conclusion", True),
        ("optional_seed_extension_blocks_next_stage", True),
        ("optional_seed_extension_trigger", "automatic_metric_trigger/v1"),
    ):
        tampered_summary = json.loads(original_summary.decode("utf-8"))
        tampered_summary[field] = tampered_value
        summary_path.write_bytes(
            standard_training.ArtifactStore.canonical_json_bytes(tampered_summary)
        )
        routing_path.write_bytes(original_routing)
        (backend.stage_root / "manifest.json").unlink()
        stage6_workflow.write_stage6_manifest(
            stage_root=backend.stage_root,
            repo_root=ROOT,
        )
        with pytest.raises(Stage6WorkflowError, match="terminal artifact|summary"):
            stage6_workflow.verify_stage6_machine_acceptance(
                stage_root=backend.stage_root,
                repo_root=ROOT,
            )
    summary_path.write_bytes(original_summary)
    tampered_routing = json.loads(original_routing.decode("utf-8"))
    tampered_routing["next_stage_after_human_approval"] = (
        "ppo_highres_frontier_stage6b_optional_multiseed/v1"
    )
    routing_path.write_bytes(
        standard_training.ArtifactStore.canonical_json_bytes(tampered_routing)
    )
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    with pytest.raises(Stage6WorkflowError, match="terminal artifact|routing"):
        stage6_workflow.verify_stage6_machine_acceptance(
            stage_root=backend.stage_root,
            repo_root=ROOT,
        )
    routing_path.write_bytes(original_routing)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    performance_summary = json.loads(original_summary.decode("utf-8"))
    performance_routing = json.loads(original_routing.decode("utf-8"))
    performance_summary["performance_advantage_established"] = False
    performance_routing["performance_advantage_established"] = False
    summary_path.write_bytes(
        standard_training.ArtifactStore.canonical_json_bytes(performance_summary)
    )
    routing_path.write_bytes(
        standard_training.ArtifactStore.canonical_json_bytes(performance_routing)
    )
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    with pytest.raises(Stage6WorkflowError, match="terminal artifact|summary"):
        stage6_workflow.verify_stage6_machine_acceptance(
            stage_root=backend.stage_root,
            repo_root=ROOT,
        )
    summary_path.write_bytes(original_summary)
    routing_path.write_bytes(original_routing)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    unknown_claim = json.loads(original_summary.decode("utf-8"))
    unknown_claim["cross_seed_robustness_established"] = True
    summary_path.write_bytes(
        standard_training.ArtifactStore.canonical_json_bytes(unknown_claim)
    )
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    with pytest.raises(Stage6WorkflowError, match="terminal artifact|summary"):
        stage6_workflow.verify_stage6_machine_acceptance(
            stage_root=backend.stage_root,
            repo_root=ROOT,
        )
    summary_path.write_bytes(original_summary)
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    for report_name in (
        "report.md",
        "standard_training_report.md",
        "standard_eval_report.md",
    ):
        report_path = backend.stage_root / report_name
        original_report = report_path.read_bytes()
        report_path.write_bytes(original_report + b"\nforged semantic claim\n")
        (backend.stage_root / "manifest.json").unlink()
        stage6_workflow.write_stage6_manifest(
            stage_root=backend.stage_root,
            repo_root=ROOT,
        )
        with pytest.raises(Stage6WorkflowError, match="report|semantic|acceptance"):
            stage6_workflow.verify_stage6_machine_acceptance(
                stage_root=backend.stage_root,
                repo_root=ROOT,
            )
        report_path.write_bytes(original_report)
        (backend.stage_root / "manifest.json").unlink()
        stage6_workflow.write_stage6_manifest(
            stage_root=backend.stage_root,
            repo_root=ROOT,
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["final_evaluation_count"] = 9
    summary_path.write_bytes(
        standard_training.ArtifactStore.canonical_json_bytes(summary)
    )
    (backend.stage_root / "manifest.json").unlink()
    stage6_workflow.write_stage6_manifest(
        stage_root=backend.stage_root,
        repo_root=ROOT,
    )
    with pytest.raises(Stage6WorkflowError, match="terminal artifact|summary"):
        stage6_workflow.verify_stage6_machine_acceptance(
            stage_root=backend.stage_root,
            repo_root=ROOT,
        )


def test_checkpointed_transaction_repairs_partial_journal_once_and_resume_advances(
    tmp_path: Path,
) -> None:
    config = load_stage6_config(CONFIG)
    transactions = standard_training.build_standard_training_transactions(config)
    transaction = transactions[0]
    checkpoint = standard_training.StandardTransactionCheckpoint(
        transaction_key=transaction.key,
        seed=transaction.seed,
        update=transaction.update,
        checkpoint_sha256="a" * 64,
        complete_marker_sha256="b" * 64,
        policy_state_sha256="c" * 64,
    )
    journal = Stage6StateJournal(tmp_path / "run/s6/job-state.jsonl")
    bindings = _journal_bindings(checkpoint.checkpoint_sha256)
    receipt_index = standard_training.CheckpointReceiptIndex(
        tmp_path / "run/s6/checkpoints/index.jsonl"
    )
    receipt_index.append_once(checkpoint)
    journal.append(transaction.commit_states[0], bindings)

    appended = standard_training.commit_checkpointed_transaction(
        journal=journal,
        transactions=transactions,
        transaction=transaction,
        checkpoint=checkpoint,
        bindings=bindings,
        receipt_rows=receipt_index.verify(),
    )

    assert [record["state"] for record in appended] == [
        transaction.commit_states[1]
    ]
    assert [record["state"] for record in journal.verify()] == list(
        transaction.commit_states
    )
    assert standard_training.completed_transaction_keys_from_journal(
        journal.verify(), transactions
    ) == (transaction.key,)
    assert standard_training.commit_checkpointed_transaction(
        journal=journal,
        transactions=transactions,
        transaction=transaction,
        checkpoint=checkpoint,
        bindings=bindings,
        receipt_rows=receipt_index.verify(),
    ) == ()
    remaining = standard_training.resume_standard_training_transactions(
        transactions,
        completed_keys=(transaction.key,),
        checkpoint=checkpoint,
    )
    assert remaining[0] == transactions[1]


def test_checkpointed_transaction_rejects_mismatched_receipt_before_journal_append(
    tmp_path: Path,
) -> None:
    config = load_stage6_config(CONFIG)
    transactions = standard_training.build_standard_training_transactions(config)
    transaction = transactions[0]
    wrong = standard_training.StandardTransactionCheckpoint(
        transaction_key=transactions[1].key,
        seed=transactions[1].seed,
        update=transactions[1].update,
        checkpoint_sha256="a" * 64,
        complete_marker_sha256="b" * 64,
        policy_state_sha256="c" * 64,
    )
    journal = Stage6StateJournal(tmp_path / "run/s6/job-state.jsonl")
    receipt_index = standard_training.CheckpointReceiptIndex(
        tmp_path / "run/s6/checkpoints/index.jsonl"
    )
    receipt_index.append_once(wrong)

    with pytest.raises(standard_training.StandardTrainingError, match="checkpoint"):
        standard_training.commit_checkpointed_transaction(
            journal=journal,
            transactions=transactions,
            transaction=transaction,
            checkpoint=wrong,
            bindings=_journal_bindings(wrong.checkpoint_sha256),
            receipt_rows=receipt_index.verify(),
        )
    assert journal.verify() == ()


@pytest.mark.parametrize("partial_state_count", (0, 1))
def test_resume_repairs_checkpoint_ahead_by_exactly_one_transaction(
    tmp_path: Path,
    partial_state_count: int,
) -> None:
    config = load_stage6_config(CONFIG)
    transactions = standard_training.build_standard_training_transactions(config)
    transaction = transactions[0]
    checkpoint = standard_training.StandardTransactionCheckpoint(
        transaction_key=transaction.key,
        seed=transaction.seed,
        update=transaction.update,
        checkpoint_sha256="a" * 64,
        complete_marker_sha256="b" * 64,
        policy_state_sha256="c" * 64,
    )
    journal = Stage6StateJournal(tmp_path / f"resume-{partial_state_count}/job-state.jsonl")
    bindings = _journal_bindings(checkpoint.checkpoint_sha256)
    receipt_index = standard_training.CheckpointReceiptIndex(
        tmp_path / f"resume-{partial_state_count}/checkpoints/index.jsonl"
    )
    receipt_index.append_once(checkpoint)
    for state in transaction.commit_states[:partial_state_count]:
        journal.append(state, bindings)

    remaining, appended = standard_training.reconcile_checkpointed_resume(
        transactions=transactions,
        journal=journal,
        checkpoint=checkpoint,
        bindings=bindings,
        receipt_rows=receipt_index.verify(),
    )

    assert remaining[0] == transactions[1]
    assert len(appended) == len(transaction.commit_states) - partial_state_count
    assert standard_training.completed_transaction_keys_from_journal(
        journal.verify(), transactions
    ) == (transaction.key,)

    ahead_two = standard_training.StandardTransactionCheckpoint(
        transaction_key=transactions[1].key,
        seed=transactions[1].seed,
        update=transactions[1].update,
        checkpoint_sha256="d" * 64,
        complete_marker_sha256="e" * 64,
        policy_state_sha256="f" * 64,
    )
    empty = Stage6StateJournal(tmp_path / f"ahead-two-{partial_state_count}/job-state.jsonl")
    ahead_index = standard_training.CheckpointReceiptIndex(
        tmp_path / f"ahead-two-{partial_state_count}/checkpoints/index.jsonl"
    )
    ahead_index.append_once(checkpoint)
    ahead_index.append_once(ahead_two)
    with pytest.raises(standard_training.StandardTrainingError, match="ahead"):
        standard_training.reconcile_checkpointed_resume(
            transactions=transactions,
            journal=empty,
            checkpoint=ahead_two,
            bindings=_journal_bindings(ahead_two.checkpoint_sha256),
            receipt_rows=ahead_index.verify(),
        )


def test_transaction_metrics_jsonl_is_idempotent_after_durable_recovery(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training_metrics.jsonl"
    record = {
        "transaction_key": "0000:20260716:update:001",
        "seed": 20260716,
        "update": 1,
        "batch_size": 1024,
    }

    assert standard_training.append_transaction_metric_once(path, record) is True
    assert standard_training.append_transaction_metric_once(path, record) is False
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(standard_training.StandardTrainingError, match="metric"):
        standard_training.append_transaction_metric_once(
            path,
            {**record, "batch_size": 2048},
        )


def test_authoritative_execution_ledgers_do_not_use_target_path_read_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl

    receipt_path = tmp_path / "checkpoints/index.jsonl"
    checkpoint = standard_training.StandardTransactionCheckpoint(
        transaction_key="0000:20260716:update:001",
        seed=20260716,
        update=1,
        checkpoint_sha256="a" * 64,
        complete_marker_sha256="b" * 64,
        policy_state_sha256="c" * 64,
    )
    receipt_index = standard_training.CheckpointReceiptIndex(receipt_path)
    assert receipt_index.append_once(checkpoint) is True

    metric_path = tmp_path / "training_metrics.jsonl"
    metric = {
        "transaction_key": "0000:20260716:update:001",
        "seed": 20260716,
        "update": 1,
        "batch_size": 1024,
    }
    DurableJsonl(metric_path).append(metric)

    stage_root = tmp_path / "s6"
    resource_path = stage_root / "resource_audit.jsonl"
    DurableJsonl(resource_path).append({"phase": "segment_start"})
    backend = object.__new__(standard_training.StandardProductionBackend)
    backend.stage_root = stage_root
    backend._execution_operation = lambda _label: nullcontext()

    guarded_paths = {receipt_path, metric_path, resource_path}
    original_read_bytes = Path.read_bytes

    def reject_target_read(self: Path) -> bytes:
        if self in guarded_paths:
            raise AssertionError("authoritative ledger used naked Path.read_bytes")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", reject_target_read)

    assert receipt_index.verify()[0]["transaction_key"] == checkpoint.transaction_key
    assert standard_training.append_transaction_metric_once(metric_path, metric) is False
    assert backend._resource_audit_rows() == ({"phase": "segment_start"},)


def test_transaction_metric_recovers_full_pending_row_exactly_once(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl

    path = tmp_path / "training_metrics-pending.jsonl"
    record = {
        "transaction_key": "0000:20260716:update:001",
        "seed": 20260716,
        "update": 1,
        "batch_size": 1024,
    }

    def fail_after_fsync(phase: str) -> None:
        if phase == "after_append_fsync":
            raise OSError("injected post-fsync crash")

    with pytest.raises(OSError, match="post-fsync"):
        DurableJsonl(path, fault_injector=fail_after_fsync).append(record)

    assert standard_training.append_transaction_metric_once(path, record) is False
    assert not Path(f"{path}.pending").exists()
    assert len(path.read_bytes().splitlines()) == 1


def test_transaction_metric_fails_closed_when_committed_append_is_not_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl

    path = tmp_path / "training_metrics-exact.jsonl"
    record = {
        "transaction_key": "0000:20260716:update:001",
        "seed": 20260716,
        "update": 1,
        "batch_size": 1024,
    }
    original_snapshot = DurableJsonl.recover_and_snapshot
    calls = 0

    def drifted_snapshot(self: DurableJsonl) -> bytes:
        nonlocal calls
        payload = original_snapshot(self)
        if self.path == path:
            calls += 1
            if calls == 2:
                return payload + b'{"forged":true}\n'
        return payload

    monkeypatch.setattr(DurableJsonl, "recover_and_snapshot", drifted_snapshot)

    with pytest.raises(standard_training.StandardTrainingError, match="prefix|exact|drift"):
        standard_training.append_transaction_metric_once(path, record)
    assert calls == 2


def test_checkpoint_receipt_index_recovers_torn_append_before_verify(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl

    path = tmp_path / "checkpoints/index-torn.jsonl"
    checkpoint = standard_training.StandardTransactionCheckpoint(
        transaction_key="0000:20260716:update:001",
        seed=20260716,
        update=1,
        checkpoint_sha256="a" * 64,
        complete_marker_sha256="b" * 64,
        policy_state_sha256="c" * 64,
    )

    def fail_mid_append(phase: str) -> None:
        if phase == "mid_append":
            raise OSError("injected torn receipt")

    failing = standard_training.CheckpointReceiptIndex(path)
    failing._durable = DurableJsonl(path, fault_injector=fail_mid_append)
    with pytest.raises(standard_training.StandardTrainingError, match="durable"):
        failing.append_once(checkpoint)

    recovered = standard_training.CheckpointReceiptIndex(path)
    rows = recovered.verify()
    assert len(rows) == 1
    assert rows[0]["transaction_key"] == checkpoint.transaction_key
    assert recovered.append_once(checkpoint) is False
    assert not Path(f"{path}.pending").exists()


def test_state_journal_recovers_torn_hash_chained_row_before_verify(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl
    from lunar_exploration_ppo.workflows import stage6 as stage6_workflow

    path = tmp_path / "job-state-torn.jsonl"
    journal = Stage6StateJournal(path)
    bindings = _journal_bindings("a" * 64)
    first = journal.append("seed_20260716_initializing", bindings)
    event = {
        "state": "seed_20260716_training_update_1",
        "previous_record_hash": first["record_hash"],
        "bindings": bindings,
    }
    second = {
        **event,
        "record_hash": stage6_workflow._canonical_sha256(event),
    }

    def fail_mid_append(phase: str) -> None:
        if phase == "mid_append":
            raise OSError("injected torn journal")

    with pytest.raises(OSError, match="torn journal"):
        DurableJsonl(path, fault_injector=fail_mid_append).append(second)

    rows = Stage6StateJournal(path).verify()
    assert tuple(row["state"] for row in rows) == (
        "seed_20260716_initializing",
        "seed_20260716_training_update_1",
    )
    assert not Path(f"{path}.pending").exists()


class _EvalIsolationState:
    def __init__(self) -> None:
        self.value = 7

    def capture_states(self):
        return tuple(
            {
                "episode_state": {"worker": worker, "value": self.value},
                "sampler_state": {"cursor": self.value + worker},
            }
            for worker in range(8)
        )


def test_eval_only_transaction_enforces_all_runtime_state_hashes(tmp_path: Path) -> None:
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    policy = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=3e-4,
        eps=1e-5,
        weight_decay=1e-4,
    )
    loss = policy(torch.ones((1, 3))).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    runtime = _EvalIsolationState()
    normalizer = {"count": 3, "mean": [0.1, 0.2]}
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    config_path.write_bytes(b'{"frozen":true}\n')
    checkpoint_path.write_bytes(b"checkpoint-v1")

    def binding_state():
        return {
            "config_bytes": config_path.read_bytes(),
            "checkpoint_bytes": checkpoint_path.read_bytes(),
        }

    result, audit = standard_training.run_eval_only_transaction(
        evaluator=lambda: {"episode_count": 16},
        policy=policy,
        optimizer=optimizer,
        normalization_stats=normalizer,
        runtime_state_provider=runtime.capture_states,
        binding_state_provider=binding_state,
    )

    assert result == {"episode_count": 16}
    assert audit["passed"] is True
    assert audit["policy_unchanged"] is True
    assert audit["optimizer_unchanged"] is True
    assert audit["rng_unchanged"] is True
    assert audit["normalizer_unchanged"] is True
    assert audit["scenario_sampler_vector_state_unchanged"] is True
    assert audit["config_checkpoint_binding_unchanged"] is True

    def mutate_checkpoint():
        checkpoint_path.write_bytes(b"checkpoint-tampered")
        return None

    with pytest.raises(
        standard_training.StandardTrainingError,
        match="config/checkpoint binding",
    ):
        standard_training.run_eval_only_transaction(
            evaluator=mutate_checkpoint,
            policy=policy,
            optimizer=optimizer,
            normalization_stats=normalizer,
            runtime_state_provider=runtime.capture_states,
            binding_state_provider=binding_state,
        )

    def mutate_optimizer():
        optimizer.param_groups[0]["lr"] = 1e-4
        return None

    with pytest.raises(standard_training.StandardTrainingError, match="optimizer"):
        standard_training.run_eval_only_transaction(
            evaluator=mutate_optimizer,
            policy=policy,
            optimizer=optimizer,
            normalization_stats=normalizer,
            runtime_state_provider=runtime.capture_states,
            binding_state_provider=binding_state,
        )


def test_update_transaction_calls_real_collaborator_contract_checkpoint_first(
    tmp_path: Path,
) -> None:
    config = load_stage6_config(CONFIG)
    expected_contract = SafetyContract.from_stage6_config(config)
    expected_safety = config.safety.model_dump(mode="json")
    expected_safety_sha256 = expected_contract.sha256
    transaction = standard_training.StandardTrainingTransaction(
        sequence=0,
        seed=20260716,
        seed_index=0,
        update=10,
        validation_episodes=16,
        commit_states=(
            "seed_20260716_training_update_10",
            "seed_20260716_validating_update_10",
        ),
    )
    events: list[str] = []
    vector_states = tuple(
        {
            "episode_state": {"worker": worker},
            "sampler_state": {"cursor": worker + 10},
        }
        for worker in range(8)
    )
    policy = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=3e-4, eps=1e-5, weight_decay=1e-4
    )
    policy_hash = policy_state_sha256(policy)
    collection_audit = CollectorAudit(
        schema_version="stage4_spawn_collector/v1",
        worker_pids=tuple(range(10_001, 10_009)),
        worker_start_methods=("spawn",) * 8,
        per_env_trainable_counts=(128,) * 8,
        diagnostic_reset_counts=(0,) * 8,
        trainable_transition_count=1024,
        inference_pids=(os.getpid(),),
        inference_batch_count=128,
        policy_device="cuda:0",
        policy_state_sha256=policy_hash,
        snapshot_sha256=tuple(
            sha256(f"snapshot-{index}".encode("ascii")).hexdigest()
            for index in range(1024)
        ),
        terminal_transition_count=0,
    )

    class Collector:
        class Vector:
            def capture_states(self):
                return vector_states

        vector_env = Vector()

        def collect(self, *, continue_from_current_state, resource_guard=None):
            events.append("collect")
            assert continue_from_current_state is True
            if resource_guard is not None:
                resource_guard("collector:inside")
            return SimpleNamespace(
                batch=SimpleNamespace(size=1024),
                vector_env_states=vector_states,
                audit=collection_audit,
            )

    class Trainer:
        def update(self, batch, *, resource_guard=None):
            events.append("update")
            assert batch.size == 1024
            if resource_guard is not None:
                resource_guard("trainer:inside")
            return {
                "update_step": 10,
                "initial_ratio_max_abs_error": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "approx_kl": 0.0,
                "grad_pre_clip_norm_max": 0.0,
                "grad_post_clip_norm_max": 0.0,
                "policy_state_sha256_before": policy_hash,
                "policy_state_sha256_after": policy_hash,
            }

    complete_path = tmp_path / "complete.json"
    saved_best_records: list[dict[str, object]] = []
    saved_eval_metrics: list[dict[str, object]] = []

    class Checkpoints:
        def prepare_complete(self, **kwargs):
            events.append("checkpoint:prepare")
            assert len(kwargs["vector_env_states"]) == 8
            assert kwargs["update_step"] == 10
            assert kwargs["training_config"]["safety"] == expected_safety
            assert kwargs["safety_contract"] == expected_contract
            assert (
                kwargs["lineage"]["safety_contract_sha256"]
                == expected_safety_sha256
            )
            saved_best_records.append(dict(kwargs["best_record"]))
            saved_eval_metrics.append(dict(kwargs["eval_metrics"]))
            return SimpleNamespace(
                checkpoint_sha256="c" * 64,
                complete_marker_sha256=sha256(b"complete-marker").hexdigest(),
                policy_state_sha256=policy_state_sha256(kwargs["policy"]),
            )

        def publish_prepared(self, prepared):
            events.append("checkpoint:publish")
            complete_path.write_bytes(b"complete-marker")
            return SimpleNamespace(
                checkpoint_sha256=prepared.checkpoint_sha256,
                complete_marker_path=complete_path,
                policy_state_sha256=prepared.policy_state_sha256,
            )

        def abort_prepared(self, prepared):
            events.append("checkpoint:abort")

    class Journal:
        def __init__(self):
            self.records: list[dict[str, object]] = []

        def verify(self):
            return tuple(self.records)

        def append(self, state, bindings):
            events.append(f"journal:{state}")
            record = {"state": state, "bindings": dict(bindings)}
            self.records.append(record)
            return record

    class ReceiptIndex:
        def __init__(self):
            self.delegate = standard_training.CheckpointReceiptIndex(
                tmp_path / "checkpoints/index.jsonl"
            )

        def append_once(self, checkpoint):
            events.append("receipt")
            return self.delegate.append_once(checkpoint)

        def verify(self):
            return self.delegate.verify()

    def validate():
        events.append("validation")
        return {
            "episode_count": 16,
            "success_rate_under_fixed_step_budget": 0.7,
            "mean_final_coverage": 0.4,
        }

    config_backing = tmp_path / "config.json"
    previous_checkpoint = tmp_path / "previous-checkpoint.pt"
    config_backing.write_bytes(CONFIG.read_bytes())
    previous_checkpoint.write_bytes(b"previous-complete-checkpoint")

    transaction_kwargs = dict(
        transaction=transaction,
        transactions=(transaction,),
        collector=Collector(),
        trainer=Trainer(),
        checkpoint_manager=Checkpoints(),
        checkpoint_receipt_index=ReceiptIndex(),
        journal=Journal(),
        policy=policy,
        optimizer=optimizer,
        config=config,
        normalization_stats={},
        checkpoint_metadata={
            "best_record": {},
            "versions": {},
            "top_m_config": {},
            "scale_profile": "Standard v1",
            "training_config": {"safety": dict(expected_safety)},
            "config_sha256": "a" * 64,
            "lineage": {
                "stage5_gate_sha256": "b" * 64,
                "safety_contract_sha256": expected_safety_sha256,
            },
            "safety_contract": expected_contract,
        },
        journal_bindings=_journal_bindings("0" * 64),
        validation_evaluator=validate,
        eval_binding_state_provider=lambda: {
            "config_bytes": config_backing.read_bytes(),
            "checkpoint_bytes": previous_checkpoint.read_bytes(),
        },
        continue_from_current_state=True,
    )

    def refuse_checkpoint(boundary: str) -> None:
        events.append(f"guard:{boundary}")
        if boundary == "checkpoint:before-publication":
            raise RuntimeError("latched before checkpoint publication")

    with pytest.raises(RuntimeError, match="latched before checkpoint publication"):
        standard_training.run_standard_update_transaction(
            **transaction_kwargs,
            resource_guard=refuse_checkpoint,
            capability_guard=lambda boundary: None,
            persist_post_resource=lambda checkpoint: events.append(
                "resource:post-persisted"
            ),
            accept_post_resource=lambda checkpoint: events.append(
                "resource:post-accepted"
            ),
        )
    assert "checkpoint:prepare" not in events
    assert "checkpoint:publish" not in events
    assert "receipt" not in events
    assert not any(event.startswith("journal:") for event in events)

    events.clear()

    def fail_post_resource(checkpoint) -> None:
        events.append("resource:post-started")
        raise RuntimeError("injected post-resource persistence failure")

    with pytest.raises(RuntimeError, match="post-resource persistence"):
        standard_training.run_standard_update_transaction(
            **transaction_kwargs,
            resource_guard=lambda boundary: events.append(f"guard:{boundary}"),
            capability_guard=lambda boundary: None,
            persist_post_resource=fail_post_resource,
            accept_post_resource=lambda checkpoint: events.append(
                "resource:post-accepted"
            ),
        )
    assert "checkpoint:prepare" in events
    assert "resource:post-started" in events
    assert "checkpoint:abort" in events
    assert "checkpoint:publish" not in events
    assert "receipt" not in events
    assert not any(event.startswith("journal:") for event in events)

    events.clear()
    saved_best_records.clear()
    saved_eval_metrics.clear()

    def observe_resource_boundary(boundary: str) -> None:
        events.append(f"guard:{boundary}")

    def persist_post_resource(checkpoint) -> None:
        assert checkpoint.transaction_key == transaction.key
        events.append("resource:post-persisted")

    def accept_post_resource(checkpoint) -> None:
        assert checkpoint.transaction_key == transaction.key
        assert "checkpoint:publish" not in events
        assert not complete_path.exists()
        events.append("resource:post-accepted")

    result = standard_training.run_standard_update_transaction(
        **transaction_kwargs,
        resource_guard=observe_resource_boundary,
        capability_guard=lambda boundary: None,
        persist_post_resource=persist_post_resource,
        accept_post_resource=accept_post_resource,
    )

    assert events == [
        "collect",
        "guard:collector:inside",
        "update",
        "guard:trainer:inside",
        "validation",
        "guard:checkpoint:before-publication",
        "checkpoint:prepare",
        "resource:post-persisted",
        "guard:checkpoint:before-resource-acceptance",
        "resource:post-accepted",
        "checkpoint:publish",
        "receipt",
        "journal:seed_20260716_training_update_10",
        "journal:seed_20260716_validating_update_10",
    ]
    assert result.checkpoint.checkpoint_sha256 == "c" * 64
    assert result.checkpoint.complete_marker_sha256 == sha256(
        b"complete-marker"
    ).hexdigest()
    assert result.validation_result["episode_count"] == 16
    assert result.eval_isolation_audit["passed"] is True
    assert result.collection_audit["trainable_transition_count"] == 1024
    assert result.collection_audit["per_env_trainable_counts"] == [128] * 8
    assert result.collection_audit["policy_state_sha256"] == policy_hash
    assert saved_eval_metrics[0]["collection_audit"] == result.collection_audit
    assert saved_eval_metrics[0]["update"]["policy_state_sha256_before"] == policy_hash
    assert saved_eval_metrics[0]["update"]["policy_state_sha256_after"] == policy_hash
    assert saved_best_records == [
        {
            "seed": 20260716,
            "update": 10,
            "success_rate_under_fixed_step_budget": 0.7,
            "mean_final_coverage": 0.4,
            "checkpoint_ref": "update-00000010",
        }
    ]
    assert result.best_record == saved_best_records[0]


def test_later_worse_validation_keeps_prior_best_for_checkpoint() -> None:
    prior = standard_training.ValidationRecord(
        20260716,
        10,
        0.8,
        0.6,
        "update-00000010",
    ).to_dict()

    selected = standard_training.select_validation_checkpoint_best(
        seed=20260716,
        update=20,
        validation_metrics={
            "episode_count": 16,
            "success_rate_under_fixed_step_budget": 0.7,
            "mean_final_coverage": 0.9,
        },
        previous_best=prior,
    )

    assert selected == prior


def test_stage6_checkpoint_schema_and_lineage_round_trip_without_stage3_aliases(
    tmp_path: Path,
) -> None:
    from test_stage4_checkpoint import VERSIONS, _policy_and_optimizer

    config = load_stage6_config(CONFIG)
    safety_contract = SafetyContract.from_stage6_config(config)
    lineage = {
        "stage5_commit": "b" * 40,
        "stage5_gate_sha256": "1" * 64,
        "stage5_manifest_sha256": "2" * 64,
        "stage4_checkpoint_sha256": "3" * 64,
        "stage4_policy_state_sha256": "4" * 64,
        "source_set_sha256": "5" * 64,
        "prospective_tree_sha256": "6" * 64,
        "data_sha256": "7" * 64,
        "safety_contract_sha256": safety_contract.sha256,
    }
    policy, optimizer = _policy_and_optimizer()
    manager = CheckpointManager(
        tmp_path / "stage6-checkpoints",
        schema_version=config.checkpoint.schema_version,
    )

    receipt = manager.save_complete(
        policy=policy,
        optimizer=optimizer,
        update_step=1,
        normalization_stats={},
        scenario_sampler_state={"workers": list(range(8))},
        vector_env_states=[{"worker": worker} for worker in range(8)],
        best_record={},
        versions=VERSIONS,
        top_m_config={"top_m": 1024},
        scale_profile="Standard v1",
        training_config={"updates_per_seed": 100},
        config_sha256="a" * 64,
        lineage=lineage,
        eval_metrics={},
        safety_contract=safety_contract,
    )
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(0.25)
    second = manager.save_complete(
        policy=policy,
        optimizer=optimizer,
        update_step=2,
        normalization_stats={},
        scenario_sampler_state={"workers": list(range(8))},
        vector_env_states=[{"worker": worker} for worker in range(8)],
        best_record={"update": 1},
        versions=VERSIONS,
        top_m_config={"top_m": 1024},
        scale_profile="Standard v1",
        training_config={"updates_per_seed": 100},
        config_sha256="a" * 64,
        lineage=lineage,
        eval_metrics={},
        safety_contract=safety_contract,
    )
    retention = CheckpointRetentionManager(
        tmp_path / "stage6-checkpoints",
        schema_version=config.checkpoint.schema_version,
    )
    retention_receipt = retention.apply(
        latest_update=2,
        periodic_updates=(),
        best_update=1,
        periodic_keep_count=5,
    )
    retained = retention.verify_retained(
        latest_update=2,
        expected_updates=(1, 2),
    )
    inspected = manager.inspect_complete(
        update_step=2,
        expected_config_sha256="a" * 64,
        expected_lineage=lineage,
        expected_safety_contract=safety_contract,
    )
    restored_policy, restored_optimizer = _policy_and_optimizer(seed=19)
    loaded = manager.load_complete(
        update_step=1,
        policy=restored_policy,
        optimizer=restored_optimizer,
        expected_config_sha256="a" * 64,
        expected_lineage=lineage,
        expected_safety_contract=safety_contract,
    )

    payload = torch.load(receipt.checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == "stage6_complete_checkpoint/v1"
    assert not any(key.startswith("stage3_") for key in payload["lineage"])
    assert loaded.update_step == 1
    assert retention_receipt.kept_updates == (1, 2)
    assert retained.kept_updates == (1, 2)
    assert retained.removed_updates == ()
    assert inspected.update_step == 2
    assert inspected.checkpoint_sha256 == second.checkpoint_sha256
    assert inspected.policy_state_sha256 == second.policy_state_sha256
    assert loaded.lineage == lineage
    assert loaded.policy_state_sha256 == receipt.policy_state_sha256
    assert second.policy_state_sha256 != receipt.policy_state_sha256

    with pytest.raises(Exception, match="schema"):
        CheckpointManager(
            tmp_path / "unknown-checkpoints",
            schema_version="unknown_checkpoint/v1",
        )
