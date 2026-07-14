"""Stage 4 machine acceptance, artifact, verifier, and runner contracts."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from lunar_exploration_ppo.configs.stage4 import load_stage4_config
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.workflows.stage1_artifacts import FrozenFileSnapshot
from lunar_exploration_ppo.workflows.stage4 import (
    STAGE3_APPROVAL_SHA256,
    STAGE3_COMMIT,
    STAGE3_GATE_SHA256,
    STAGE4_CHECKPOINT_ARTIFACTS,
    STAGE4_EVIDENCE_ARTIFACTS,
    STAGE4_MANIFEST_BOUND_ARTIFACTS,
    STAGE4_ROOT_ARTIFACTS,
    FrozenStage3AuthorityHandle,
    Stage4MachineAuditBundle,
    Stage4MachineVerificationHandle,
    Stage4WorkflowError,
    _build_stage4_machine_payload,
    _build_stage4_runtime_binding,
    run_stage4_smoke_acceptance,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/ppo_highres_frontier_stage4_v1.json"


def _authority() -> dict[str, object]:
    return {
        "schema_version": "ppo_highres_frontier_stage4_stage3_authority_audit/v1",
        "verified": True,
        "stage3_commit": STAGE3_COMMIT,
        "stage3_commit_parent": "898911559ccc9ae8ef701b69a58a93b1d8a4d8d8",
        "stage3_commit_tree": "689876d5031eac53290397c534be3a155d017b10",
        "stage3_run_id": "s3-task4-cifix-final-20260713T213739Z",
        "gate_path": "D:/frozen/s3/gate.json",
        "gate_sha256": STAGE3_GATE_SHA256,
        "approval_sha256": STAGE3_APPROVAL_SHA256,
        "review_sha256": (
            "62bf0869ba66f22bfec07464b7d17478d27fa618c3ae645039c98333262b2afc"
        ),
        "manifest_sha256": (
            "2715e790d843ac9eb873b9123a569f5254e603a734f86aa9629ac8abce142a90"
        ),
        "source_set_sha256": (
            "26ef6026b40107347ea35b6d2545f47fee1297f7a181a5d4f517183609d9759c"
        ),
        "environment_sha256": (
            "dac42c3bc7d28e4acb78264ab7b5955c89f5716b7a841eae93a4658704bdd7c0"
        ),
        "stage2_gate_sha256": (
            "997203efa6bedd0d0e7d124f8b2362568983c1eac48a4b9e588c80d41b33dcb2"
        ),
        "reviewed_prospective_git_tree": "689876d5031eac53290397c534be3a155d017b10",
        "reviewed_path_set_sha256": (
            "dc75183d3be7993893fdbc4b169b00e3d24d27986a2c8e85442c2a8ba6a0754d"
        ),
        "reviewed_path_count": 14,
        "authorized_stage": "ppo_highres_frontier_stage4_rollout_ppo_update/v1",
        "authority_scope": (
            "external_controller_claim_integrity_not_local_identity_authentication/v1"
        ),
    }


def _source() -> dict[str, object]:
    return {
        "schema_version": "stage4_reviewed_source_set/v1",
        "source_set_sha256": "a" * 64,
        "paths": [
            {"path": "src/example.py", "sha256": "b" * 64, "size_bytes": 1}
        ],
    }


def _environment() -> dict[str, object]:
    return {
        "schema_version": "ppo_highres_frontier_stage4_environment/v1",
        "cuda_available": True,
        "compute_dtype": "float32",
        "amp_enabled": False,
        "multiprocessing_start_method": "spawn",
        "cuda_device_name": "fixture CUDA",
    }


def _git() -> dict[str, object]:
    return {
        "schema_version": "ppo_highres_frontier_stage4_prospective_git_tree/v1",
        "head_commit": STAGE3_COMMIT,
        "base_commit": STAGE3_COMMIT,
        "prospective_git_tree": "c" * 40,
        "changed_paths": ["src/example.py"],
        "changed_path_set_sha256": "d" * 64,
        "real_index_empty": True,
    }


def _stage1_binding() -> dict[str, object]:
    return {
        "path": str(ROOT / "configs/ppo_highres_frontier_smoke_v1.json"),
        "sha256": "7220d8fc0b039872fa278a0227f1dc58c7a42709c3deab592e9ec31ce9d1e3b2",
    }


def _passing_audit(config_sha256: str) -> Stage4MachineAuditBundle:
    updates = []
    for step in range(1, 4):
        updates.append(
            {
                "schema_version": "stage4_smoke_update_audit/v1",
                "update_step": step,
                "config_sha256": config_sha256,
                "collection": {
                    "trainable_transition_count": 1024,
                    "per_env_trainable_counts": [128] * 8,
                    "diagnostic_reset_counts": [0] * 8,
                    "worker_pids": list(range(1000, 1008)),
                    "worker_start_methods": ["spawn"] * 8,
                    "inference_pids": [999],
                    "inference_main_process_only": True,
                    "inference_batch_count": 129,
                    "policy_device": "cuda:0",
                    "policy_state_sha256": f"{step}" * 64,
                    "snapshot_count": 1024,
                    "unique_snapshot_count": 1000,
                    "terminal_transition_count": 8,
                    "old_logprob_factorization_exact": True,
                },
                "gae": {
                    "finite": True,
                    "normalized_advantage_mean": 0.0,
                    "normalized_advantage_std": 1.0,
                },
                "ppo": {
                    "initial_ratio_max_abs_error": 0.0,
                    "policy_loss": 0.1,
                    "value_loss": 0.2,
                    "frontier_entropy_mean": 0.3,
                    "approx_kl": 0.01,
                    "early_stopped": False,
                    "grad_pre_clip_norm_max": 1.0,
                    "grad_post_clip_norm_max": 0.5,
                    "parameter_change_l2": 0.01,
                    "optimizer_steps": 16,
                    "planned_optimizer_steps": 16,
                    "policy_state_sha256_before": f"{step}" * 64,
                    "policy_state_sha256_after": f"{step + 1}" * 64,
                },
                "buffer": {"consumed": True, "cleared": True},
                "checkpoint": {
                    "update_step": step,
                    "checkpoint_sha256": f"{step + 3}" * 64,
                    "manifest_sha256": f"{step + 4}" * 64,
                    "loaded_last_complete_step": step,
                    "policy_state_sha256": f"{step + 1}" * 64,
                    "deterministic_action_bit_exact": True,
                },
            }
        )
    tiny = {
        "schema_version": "stage4_tiny_overfit_acceptance/v1",
        "context_count": 4,
        "passed": True,
        "seeds": [
            {
                "seed": seed,
                "passed": True,
                "stop_update": 23,
                "correct_frontier_probability_min": 0.951,
                "theta_error_rad_max": 0.01,
                "initial_value_mse": 1.0,
                "final_value_mse": 0.01,
                "value_mse_reduction": 0.99,
                "ppo_update_count": 23,
                "rollout_batch_count": 23,
                "curve": [],
            }
            for seed in (17, 29, 43)
        ],
    }
    return Stage4MachineAuditBundle(
        updates=tuple(updates),
        tiny_overfit=tiny,
        checkpoint_action_fixture_npz=b"fixture-npz",
        training_resume={
            "schema_version": "stage4_training_resume_audit/v1",
            "fresh_trainer_initial_update_step": 0,
            "resume_receipt": {
                "schema_version": "stage4_training_resume/v1",
                "update_step": 1,
                "next_update_step": 2,
                "checkpoint_sha256": "4" * 64,
                "manifest_sha256": "5" * 64,
                "policy_state_sha256": "2" * 64,
                "optimizer_state_sha256": "a" * 64,
                "normalization_state_sha256": "b" * 64,
                "scenario_sampler_state_sha256": "c" * 64,
                "vector_env_states_sha256": "d" * 64,
            },
            "normalization_state_applied": True,
            "scenario_sampler_state_applied": True,
            "vector_env_states_applied": True,
            "vector_env_state_count": 8,
            "continued_from_current_state": True,
            "next_update_executed": True,
            "next_update_step": 2,
            "collection_policy_state_sha256": "2" * 64,
            "policy_state_sha256_before_next_update": "2" * 64,
            "policy_state_sha256_after_next_update": "3" * 64,
            "optimizer_state_sha256_after_next_update": "e" * 64,
        },
    )


def _checkpoint_artifacts() -> dict[str, bytes]:
    return {
        path: (path + "\n").encode("utf-8")
        for path in STAGE4_CHECKPOINT_ARTIFACTS
    }


def test_stage4_payload_is_canonical_manifest_bound_and_stops_at_review() -> None:
    config = load_stage4_config(CONFIG)
    runtime = _build_stage4_runtime_binding(
        config=config,
        run_id="s4-machine-fixture",
        source_identity=_source(),
        environment_identity=_environment(),
        git_identity=_git(),
        stage3_authority=_authority(),
        stage1_config_binding=_stage1_binding(),
    )
    assert runtime.payload["stage3_authority"] == config.stage3_authority.model_dump(
        mode="json"
    )
    assert runtime.payload["execution_stage3_authority"] == _authority()
    payload = _build_stage4_machine_payload(
        config=config,
        runtime=runtime,
        audit=_passing_audit(runtime.config_sha256),
        checkpoint_artifacts=_checkpoint_artifacts(),
    )

    assert set(payload.artifacts) == set(STAGE4_MANIFEST_BOUND_ARTIFACTS)
    assert payload.summary["state"] == "machine_passed"
    assert payload.summary["acceptance"]["items"]["checkpoint_training_resume"]
    assert "evidence/training_resume_audit.json" in payload.artifacts
    routing = json.loads(payload.artifacts["routing.json"])
    assert routing["route"] == "awaiting_independent_review"
    assert routing["next_stage_entered"] is False
    phases = [json.loads(line) for line in payload.artifacts["phase-state.jsonl"].splitlines()]
    assert [item["state"] for item in phases] == [
        "machine_passed",
        "awaiting_independent_review",
    ]
    assert len(payload.artifacts["training_progress.jsonl"].splitlines()) == 3
    assert payload.artifacts["config.json"] == ArtifactStore.canonical_json_bytes(
        runtime.payload
    )
    manifest_paths = [entry["path"] for entry in payload.manifest["artifacts"]]
    assert manifest_paths == sorted(STAGE4_MANIFEST_BOUND_ARTIFACTS)
    for relative, body in payload.artifacts.items():
        entry = next(item for item in payload.manifest["artifacts"] if item["path"] == relative)
        assert entry["sha256"] == hashlib.sha256(body).hexdigest()
        assert entry["size_bytes"] == len(body)
    assert not {"review.json", "approval.json", "gate.json"} & set(
        STAGE4_ROOT_ARTIFACTS
    )


def test_stage4_payload_rejects_nonpassing_smoke_or_tiny_gate() -> None:
    config = load_stage4_config(CONFIG)
    runtime = _build_stage4_runtime_binding(
        config=config,
        run_id="s4-machine-reject",
        source_identity=_source(),
        environment_identity=_environment(),
        git_identity=_git(),
        stage3_authority=_authority(),
        stage1_config_binding=_stage1_binding(),
    )
    audit = _passing_audit(runtime.config_sha256)
    audit.updates[1]["collection"]["trainable_transition_count"] = 1023
    with pytest.raises(Stage4WorkflowError, match="machine acceptance"):
        _build_stage4_machine_payload(
            config=config,
            runtime=runtime,
            audit=audit,
            checkpoint_artifacts=_checkpoint_artifacts(),
        )


@pytest.mark.parametrize(
    "field",
    ("old_logprob_factorization_exact", "policy_state_sha256"),
)
def test_stage4_payload_rejects_missing_collection_provenance(
    field: str,
) -> None:
    config = load_stage4_config(CONFIG)
    runtime = _build_stage4_runtime_binding(
        config=config,
        run_id="s4-machine-missing-provenance",
        source_identity=_source(),
        environment_identity=_environment(),
        git_identity=_git(),
        stage3_authority=_authority(),
        stage1_config_binding=_stage1_binding(),
    )
    audit = _passing_audit(runtime.config_sha256)
    del audit.updates[0]["collection"][field]

    with pytest.raises(Stage4WorkflowError, match="machine acceptance"):
        _build_stage4_machine_payload(
            config=config,
            runtime=runtime,
            audit=audit,
            checkpoint_artifacts=_checkpoint_artifacts(),
        )


def test_stage4_payload_rejects_extra_collection_field() -> None:
    config = load_stage4_config(CONFIG)
    runtime = _build_stage4_runtime_binding(
        config=config,
        run_id="s4-machine-extra-collection-field",
        source_identity=_source(),
        environment_identity=_environment(),
        git_identity=_git(),
        stage3_authority=_authority(),
        stage1_config_binding=_stage1_binding(),
    )
    audit = _passing_audit(runtime.config_sha256)
    audit.updates[0]["collection"]["unbound_evidence"] = True

    with pytest.raises(Stage4WorkflowError, match="collection schema"):
        _build_stage4_machine_payload(
            config=config,
            runtime=runtime,
            audit=audit,
            checkpoint_artifacts=_checkpoint_artifacts(),
        )


def test_machine_verification_handle_detects_content_and_aba(tmp_path: Path) -> None:
    stage = tmp_path / "run" / "s4"
    stage.mkdir(parents=True)
    (stage / "evidence").mkdir()
    (stage / "checkpoints").mkdir()
    for name in STAGE4_ROOT_ARTIFACTS:
        path = stage / name
        if name not in {"evidence", "checkpoints"}:
            path.write_bytes((name + "\n").encode("utf-8"))
    for relative in STAGE4_EVIDENCE_ARTIFACTS:
        path = stage / relative
        path.write_bytes((relative + "\n").encode("utf-8"))
    for relative in STAGE4_CHECKPOINT_ARTIFACTS:
        path = stage / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((relative + "\n").encode("utf-8"))

    target = stage / "summary.json"
    snapshot = FrozenFileSnapshot.capture(target, authority_root=stage)
    authority = FrozenStage3AuthorityHandle(
        identity={"verified": True},
        snapshots=(),
        stage2_authority=None,
    )
    handle = Stage4MachineVerificationHandle(
        stage_root=stage,
        manifest_snapshot=FrozenFileSnapshot.capture(
            stage / "manifest.json", authority_root=stage
        ),
        artifacts=(("summary.json", snapshot),),
        stage3_authority=authority,
    )
    target.write_bytes(b"changed\n")
    with pytest.raises(Stage4WorkflowError, match="changed"):
        handle.require_current()
    target.write_bytes(b"summary.json\n")
    with pytest.raises(Stage4WorkflowError, match="changed"):
        handle.require_current()


def test_stage4_smoke_executor_source_uses_real_env_collector_ppo_and_checkpoint() -> None:
    source = inspect.getsource(run_stage4_smoke_acceptance)
    required = (
        "lunar_env_specs(",
        "SpawnVectorEnv(",
        ".collect()",
        ".update(",
        "save_complete(",
        "load_last_complete(",
        "resume_training_from_last_checkpoint(",
        "run_tiny_overfit_acceptance(",
    )
    assert all(token in source for token in required)
    assert "Mock" not in source
    assert "monkeypatch" not in source


def test_stage4_runner_is_thin_has_main_guard_and_no_force() -> None:
    runner = ROOT / "scripts/run_ppo_highres_frontier_stage4.py"
    source = runner.read_text(encoding="utf-8")
    assert "run_stage4_workflow" in source
    assert 'if __name__ == "__main__"' in source
    assert "--config" in source
    assert "--run-id" in source
    assert "--stage3-gate" in source
    assert "--output-root" in source
    assert "--force" not in source
