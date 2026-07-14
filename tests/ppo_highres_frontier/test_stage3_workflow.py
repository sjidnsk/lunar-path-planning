"""Stage 3 frozen config and machine-gate contracts."""

from __future__ import annotations

import importlib
import importlib.util
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.workflows.stage1_artifacts import (
    FrozenFileSnapshot,
    Stage1WorkflowError,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/ppo_highres_frontier_stage3_v1.json"


class _AuthorityProbe(dict):
    def __init__(
        self,
        identity: dict[str, object],
        snapshots: tuple[tuple[str, FrozenFileSnapshot], ...],
    ) -> None:
        super().__init__(identity)
        self.identity = dict(identity)
        self.snapshots = snapshots
        self.require_calls = 0

    def require_current(self, label: str = "Stage 2 authority") -> None:
        self.require_calls += 1
        for snapshot_label, snapshot in self.snapshots:
            snapshot.require_current(f"{label} {snapshot_label}")


class _MachineVerificationProbe:
    def __init__(
        self,
        *,
        artifacts: dict[str, FrozenFileSnapshot],
        stage2_authority: _AuthorityProbe,
    ) -> None:
        self.artifacts = artifacts
        self.manifest_snapshot = artifacts["manifest.json"]
        self.stage2_authority = stage2_authority

    def artifact_snapshot(self, relative: str) -> FrozenFileSnapshot:
        return self.artifacts[relative]

    def require_current(self, label: str = "Stage 3 machine verification") -> None:
        self.stage2_authority.require_current(label)
        for relative, snapshot in self.artifacts.items():
            snapshot.require_current(f"{label} {relative}")


def _install_minimal_stage3_verifier_fixture(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority_path: Path,
) -> tuple[object, Path, _AuthorityProbe, dict[str, object]]:
    workflow = importlib.import_module("lunar_exploration_ppo.workflows.stage3")
    stage = tmp_path / "probe-run" / "s3"
    (stage / "evidence").mkdir(parents=True)
    source_identity = {
        "schema_version": "stage3_reviewed_source_set/v1",
        "source_set_sha256": "1" * 64,
    }
    environment_identity = {
        "schema_version": "ppo_highres_frontier_stage3_environment/v1",
        "environment_sha256": "2" * 64,
    }
    git_identity = {
        "schema_version": "ppo_highres_frontier_stage3_prospective_git_tree/v1",
        "prospective_git_tree": "3" * 40,
        "changed_paths": [],
    }
    authority_identity = {
        "schema_version": "ppo_highres_frontier_stage3_stage2_authority_audit/v1",
        "verified": True,
        "gate_sha256": "4" * 64,
    }
    authority = _AuthorityProbe(
        authority_identity,
        ((authority_path.name, FrozenFileSnapshot.capture(authority_path)),),
    )
    execution_identity = {
        "schema_version": "ppo_highres_frontier_stage3_execution_identity_audit/v1",
        "source": source_identity,
        "environment": environment_identity,
        "git": git_identity,
    }
    config_payload = {
        "execution_source_identity": source_identity,
        "execution_environment_identity": environment_identity,
        "execution_git_identity": git_identity,
        "stage2_authority": authority_identity,
    }
    artifacts = {
        relative: b"probe\n" for relative in workflow.STAGE3_MANIFEST_BOUND_ARTIFACTS
    }
    artifacts["config.json"] = ArtifactStore.canonical_json_bytes(config_payload)
    for relative in (
        "evidence/network_shape_audit.json",
        "evidence/logprob_recompute_audit.json",
        "evidence/mask_handling_audit.json",
        "evidence/cuda_latency_audit.json",
        "evidence/cuda_microbatch_audit.json",
    ):
        artifacts[relative] = ArtifactStore.canonical_json_bytes({})
    artifacts["evidence/stage2_authority_audit.json"] = (
        ArtifactStore.canonical_json_bytes(authority_identity)
    )
    artifacts["evidence/execution_identity_audit.json"] = (
        ArtifactStore.canonical_json_bytes(execution_identity)
    )
    for relative, payload in artifacts.items():
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    manifest = workflow._manifest_for_artifacts(artifacts)
    (stage / "manifest.json").write_bytes(ArtifactStore.canonical_json_bytes(manifest))

    machine_config = SimpleNamespace(
        run_id="probe-run",
        stage2_approval_commit="898911559ccc9ae8ef701b69a58a93b1d8a4d8d8",
        model_dump=lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        workflow.Stage3Config,
        "model_validate",
        classmethod(lambda _cls, _payload: machine_config),
    )
    monkeypatch.setattr(workflow, "load_stage3_config", lambda _path: machine_config)
    monkeypatch.setattr(workflow, "stage3_source_identity", lambda _repo: source_identity)
    monkeypatch.setattr(workflow, "stage3_environment_identity", lambda: environment_identity)
    monkeypatch.setattr(
        workflow,
        "stage3_git_identity",
        lambda _repo, *, base_commit: git_identity,
    )
    monkeypatch.setattr(
        workflow,
        "verify_frozen_stage2_authority",
        lambda **_kwargs: authority,
    )
    mutation: dict[str, object] = {"callback": lambda: None}

    def build_payload(**_kwargs):
        mutation["callback"]()
        return workflow._Stage3MachinePayload(
            artifacts=artifacts,
            manifest=manifest,
            summary={},
        )

    monkeypatch.setattr(workflow, "_build_stage3_machine_payload", build_payload)
    return workflow, stage, authority, mutation


def _load_stage3_review_controller():
    path = ROOT / ".superpowers/sdd/task-4-review-controller.py"
    spec = importlib.util.spec_from_file_location("task_4_review_controller", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _copy_stage3_machine_for_latency_verifier(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workflow = importlib.import_module("lunar_exploration_ppo.workflows.stage3")
    config_module = importlib.import_module("lunar_exploration_ppo.configs.stage3")
    run_id = "programmatic-latency-verifier-fixture"
    config = config_module.load_stage3_config(CONFIG_PATH).model_copy(
        update={"run_id": run_id}
    )
    source_identity = {
        "schema_version": "stage3_reviewed_source_set/v1",
        "source_set_sha256": "1" * 64,
    }
    environment_identity = {
        "schema_version": "ppo_highres_frontier_stage3_environment/v1",
        "compute_dtype": "float32",
        "amp_enabled": False,
        "cuda_device_name": "programmatic-ci-fixture",
    }
    git_identity = {
        "schema_version": "ppo_highres_frontier_stage3_prospective_git_tree/v1",
        "prospective_git_tree": "2" * 40,
        "changed_paths": [],
    }
    authority_identity = {
        "schema_version": "ppo_highres_frontier_stage3_stage2_authority_audit/v1",
        "verified": True,
        "gate_sha256": "3" * 64,
        "stage2_commit": config.stage2_approval_commit,
        "authorized_stage": config.stage_id,
    }
    authority = _AuthorityProbe(authority_identity, ())
    audit = _programmatic_stage3_audit(config)
    payload = workflow._build_stage3_machine_payload(
        config=config,
        run_id=run_id,
        source_identity=source_identity,
        environment_identity=environment_identity,
        git_identity=git_identity,
        stage2_authority=authority_identity,
        audit=audit,
    )
    stage = tmp_path / run_id / "s3"
    for relative, artifact in payload.artifacts.items():
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(artifact)
    (stage / "manifest.json").write_bytes(
        ArtifactStore.canonical_json_bytes(payload.manifest)
    )
    monkeypatch.setattr(workflow, "stage3_source_identity", lambda _repo: source_identity)
    monkeypatch.setattr(workflow, "stage3_environment_identity", lambda: environment_identity)
    monkeypatch.setattr(
        workflow,
        "stage3_git_identity",
        lambda _repo, *, base_commit: git_identity,
    )
    monkeypatch.setattr(
        workflow,
        "verify_frozen_stage2_authority",
        lambda **_kwargs: authority,
    )
    return workflow, stage


def _programmatic_stage3_audit(config):
    gpu = importlib.import_module("lunar_exploration_ppo.workflows.stage3_gpu")
    metadata = {
        "network_architecture_version": config.architecture.network_architecture_version,
        "network_memory_mode": config.architecture.network_memory_mode,
        "context_token_count": config.architecture.context_token_count,
        "token_dim": config.architecture.token_dim,
        "cross_attention_layers": config.architecture.cross_attention_layers,
        "attention_heads": config.architecture.attention_heads,
        "ffn_hidden_dim": config.architecture.ffn_hidden_dim,
        "action_hidden_dim": config.architecture.action_hidden_dim,
        "dropout": config.architecture.dropout,
    }
    network = {
        "schema_version": "ppo_highres_frontier_stage3_network_shape_audit/v1",
        "passed": True,
        "all_parameters_fp32": True,
        "forbidden_modules_present": {},
        "metadata": metadata,
        "parameter_count": 1,
    }
    zero_errors = {
        "log_prob_frontier": 0.0,
        "log_prob_theta": 0.0,
        "log_prob_total": 0.0,
    }
    logprob = {
        "passed": True,
        "factorization_exact": True,
        "cpu_tolerance": config.distribution.cpu_logprob_tolerance,
        "cuda_tolerance": config.distribution.cuda_logprob_tolerance,
        "cpu_max_abs_errors": dict(zero_errors),
        "cuda_max_abs_errors": dict(zero_errors),
    }
    mask = {
        key: True
        for key in (
            "passed",
            "invalid_probability_exact_zero",
            "invalid_frontier_feature_gradient_exact_zero",
            "deterministic_action_bit_exact",
            "selected_candidate_valid",
            "selected_theta_in_half_open_interval",
            "theta_kappa_in_bounds",
            "all_outputs_and_gradients_finite",
        )
    }
    sample_count = config.gpu.timing_measure_iterations
    smoke_samples = [10.0 + index / 100.0 for index in range(sample_count)]
    standard_samples = [20.0 + index / 100.0 for index in range(sample_count)]
    latency_gate = gpu.evaluate_latency_gate(
        smoke_samples_ms=smoke_samples,
        standard_samples_ms=standard_samples,
        smoke_limit_ms=config.gpu.smoke_batch1_forward_p95_ms,
        standard_limit_ms=config.gpu.standard_batch1_forward_p95_ms,
    )
    latency = {
        "schema_version": "ppo_highres_frontier_stage3_cuda_latency_audit/v1",
        "device": "programmatic-ci-fixture",
        "synchronized_cuda_events": True,
        "batch_size": 1,
        "warmup_iterations": config.gpu.timing_warmup_iterations,
        "measure_iterations": sample_count,
        "smoke_samples_ms": smoke_samples,
        "standard_samples_ms": standard_samples,
        "smoke_peak_memory_gib": 0.125,
        "standard_peak_memory_gib": 0.25,
        **latency_gate.to_dict(),
    }
    trials = tuple(
        gpu.MicrobatchTrial(
            batch_size=batch_size,
            status="success",
            peak_memory_gib=float(index + 1) / 10.0,
            error_kind=None,
        )
        for index, batch_size in enumerate(config.gpu.microbatch_candidates)
    )
    microbatch_gate = gpu.evaluate_microbatch_gate(
        trials,
        candidates=config.gpu.microbatch_candidates,
        warning_gib=config.gpu.memory_warning_gib,
        hard_stop_gib=config.gpu.memory_hard_stop_gib,
    )
    microbatch = {
        "schema_version": "ppo_highres_frontier_stage3_cuda_microbatch_audit/v1",
        "device": "programmatic-ci-fixture",
        "compute_dtype": "float32",
        "amp_enabled": False,
        "shape_profile": "programmatic-ci-fixture/v1",
        "forward": True,
        "representative_backward": True,
        "memory_warning_gib": config.gpu.memory_warning_gib,
        "memory_hard_stop_gib": config.gpu.memory_hard_stop_gib,
        "frozen_microbatch_size": config.gpu.frozen_microbatch_size,
        **microbatch_gate.to_dict(),
    }
    sample_outputs = io.BytesIO()
    np.savez(
        sample_outputs,
        schema_version=np.asarray(
            "ppo_highres_frontier_stage3_sample_policy_outputs/v1"
        ),
        candidate_mask=np.asarray([[True, False]], dtype=np.bool_),
        frontier_logits=np.asarray([[0.0, -1.0e9]], dtype=np.float32),
        theta_mu=np.asarray([[0.0, 0.0]], dtype=np.float32),
        theta_kappa=np.asarray([[0.1, 0.1]], dtype=np.float32),
        value=np.asarray([0.0], dtype=np.float32),
        selected_frontier_index=np.asarray([0], dtype=np.int64),
        selected_theta=np.asarray([0.0], dtype=np.float32),
        log_prob_frontier=np.asarray([0.0], dtype=np.float32),
        log_prob_theta=np.asarray([0.0], dtype=np.float32),
        log_prob_total=np.asarray([0.0], dtype=np.float32),
    )
    return gpu.Stage3CudaAuditBundle(
        network_shape_audit=network,
        logprob_recompute_audit=logprob,
        mask_handling_audit=mask,
        cuda_latency_audit=latency,
        cuda_microbatch_audit=microbatch,
        sample_policy_outputs_npz=sample_outputs.getvalue(),
    )


def _rewrite_latency_evidence_and_manifest(
    stage: Path,
    mutate,
) -> None:
    latency_path = stage / "evidence/cuda_latency_audit.json"
    latency = json.loads(latency_path.read_text(encoding="utf-8"))
    mutate(latency)
    payload = ArtifactStore.canonical_json_bytes(latency)
    latency_path.write_bytes(payload)
    manifest_path = stage / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in manifest["artifacts"]
        if item["path"] == "evidence/cuda_latency_audit.json"
    )
    entry["sha256"] = hashlib.sha256(payload).hexdigest()
    entry["size_bytes"] = len(payload)
    manifest_path.write_bytes(ArtifactStore.canonical_json_bytes(manifest))


def _config_module():
    return importlib.import_module("lunar_exploration_ppo.configs.stage3")


def _gpu_module():
    return importlib.import_module("lunar_exploration_ppo.workflows.stage3_gpu")


def test_stage3_config_freezes_observation_architecture_distribution_and_gpu_contracts() -> None:
    config_module = _config_module()
    config = config_module.load_stage3_config(CONFIG_PATH)

    assert config.schema_version == "ppo_highres_frontier_stage3_config/v1"
    assert config.goal_id == "ppo-highres-frontier-map-exploration"
    assert config.stage_id == "ppo_highres_frontier_stage3_cross_attention_policy/v1"
    assert config.authorized_next_stage == "ppo_highres_frontier_stage4_rollout_ppo_update/v1"
    assert config.stage2_approval_commit == "898911559ccc9ae8ef701b69a58a93b1d8a4d8d8"
    assert config.output_root == "D:/xunce/out/ppo_frontier"
    assert config.run_id is None

    assert config.observation.model_dump(mode="json") == {
        "schema_version": "policy_observation/v1",
        "prior_channels": 7,
        "coverage_summary_channels": 8,
        "local_crop_channels": 8,
        "frontier_feature_dim": 22,
        "pose_feature_dim": 6,
        "candidate_mask_dtype": "bool",
        "forbidden_policy_inputs": ["truth", "coverable_mask", "history", "previous_action"],
    }
    assert config.architecture.model_dump(mode="json") == {
        "network_architecture_version": "cross_attention_frontier_policy/v1",
        "network_memory_mode": "stateless_observation_only/v1",
        "global_encoder_input_channels": 15,
        "map_pool_shape": [16, 16],
        "global_token_count": 256,
        "local_token_count": 256,
        "pose_token_count": 1,
        "context_token_count": 513,
        "token_dim": 128,
        "cross_attention_layers": 2,
        "attention_heads": 4,
        "ffn_hidden_dim": 512,
        "action_hidden_dim": 128,
        "dropout": 0.0,
        "normalization": "groupnorm_cnn_layernorm_transformer_mlp/v1",
        "cross_attention_direction": "frontier_queries_to_context_keys_values/v1",
        "candidate_self_attention": False,
    }
    assert config.distribution.model_dump(mode="json") == {
        "action_space_type": "frontier_index_continuous_theta/v1",
        "factorization": "masked_categorical_times_selected_frontier_von_mises/v1",
        "frontier_distribution": "masked_categorical/v1",
        "theta_distribution": "von_mises/v1",
        "theta_parameterization": "clamped_softplus_kappa/v1",
        "invalid_logit_value": -1.0e9,
        "theta_min_inclusive": -3.141592653589793,
        "theta_max_exclusive": 3.141592653589793,
        "kappa_epsilon": 0.001,
        "kappa_min": 0.001,
        "kappa_max": 20.0,
        "initial_kappa": 0.1,
        "theta_sin_bias": 0.0,
        "theta_cos_bias": 1.0,
        "compute_dtype": "float32",
        "amp_enabled": False,
        "cpu_logprob_tolerance": 1e-6,
        "cuda_logprob_tolerance": 1e-5,
    }
    assert config.gpu.model_dump(mode="json") == {
        "device": "cuda",
        "memory_warning_gib": 9.0,
        "memory_hard_stop_gib": 10.1,
        "smoke_batch1_forward_p95_ms": 50.0,
        "standard_batch1_forward_p95_ms": 100.0,
        "microbatch_candidates": [4, 8, 16, 32],
        "frozen_microbatch_size": 32,
        "timing_warmup_iterations": 10,
        "timing_measure_iterations": 30,
        "standard_prior_shape": [7, 32, 32],
        "standard_coverage_shape": [8, 32, 32],
        "standard_local_shape": [8, 96, 96],
        "standard_frontier_top_m": 1024,
        "representative_backward": True,
    }


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("architecture", "token_dim"), 256),
        (("architecture", "candidate_self_attention"), True),
        (("distribution", "amp_enabled"), True),
        (("distribution", "compute_dtype"), "float16"),
        (("gpu", "memory_hard_stop_gib"), 10.2),
        (("gpu", "microbatch_candidates"), [4, 8, 16]),
        (("gpu", "frozen_microbatch_size"), 16),
    ],
)
def test_stage3_config_rejects_frozen_contract_drift(
    path: tuple[str, str], replacement: object
) -> None:
    config_module = _config_module()
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload[path[0]][path[1]] = replacement

    with pytest.raises(ValidationError, match="drift"):
        config_module.Stage3Config.model_validate(payload)


def test_microbatch_gate_selects_largest_safe_success_and_records_warning() -> None:
    gpu = _gpu_module()
    trials = (
        gpu.MicrobatchTrial(4, "success", 2.0, None),
        gpu.MicrobatchTrial(8, "success", 4.0, None),
        gpu.MicrobatchTrial(16, "success", 9.2, None),
        gpu.MicrobatchTrial(32, "oom", None, "cuda_out_of_memory"),
    )

    result = gpu.evaluate_microbatch_gate(
        trials,
        candidates=(4, 8, 16, 32),
        warning_gib=9.0,
        hard_stop_gib=10.1,
    )

    assert result.selected_microbatch_size == 16
    assert result.warning is True
    assert result.hard_gate_passed is True
    assert result.attempted_sizes == (4, 8, 16, 32)


@pytest.mark.parametrize(
    "trials",
    [
        (
            (4, "success", 2.0, None),
            (8, "success", 10.2, None),
            (16, "oom", None, "cuda_out_of_memory"),
            (32, "oom", None, "cuda_out_of_memory"),
        ),
        (
            (4, "success", 2.0, None),
            (8, "error", None, "kernel_launch_failure"),
            (16, "oom", None, "cuda_out_of_memory"),
            (32, "oom", None, "cuda_out_of_memory"),
        ),
    ],
    ids=("hard-memory-breach", "non-oom-error"),
)
def test_microbatch_gate_fails_closed_on_hard_memory_or_unknown_failure(trials) -> None:
    gpu = _gpu_module()
    records = tuple(gpu.MicrobatchTrial(*trial) for trial in trials)

    with pytest.raises(gpu.Stage3GpuGateError):
        gpu.evaluate_microbatch_gate(
            records,
            candidates=(4, 8, 16, 32),
            warning_gib=9.0,
            hard_stop_gib=10.1,
        )


def test_microbatch_gate_rejects_missing_reordered_or_nonfinite_trials() -> None:
    gpu = _gpu_module()
    complete = (
        gpu.MicrobatchTrial(4, "success", 2.0, None),
        gpu.MicrobatchTrial(8, "success", 4.0, None),
        gpu.MicrobatchTrial(16, "success", 6.0, None),
        gpu.MicrobatchTrial(32, "success", 8.0, None),
    )
    for malformed in (
        complete[:-1],
        (complete[1], complete[0], complete[2], complete[3]),
        complete[:-1] + (gpu.MicrobatchTrial(32, "success", float("nan"), None),),
    ):
        with pytest.raises(gpu.Stage3GpuGateError):
            gpu.evaluate_microbatch_gate(
                malformed,
                candidates=(4, 8, 16, 32),
                warning_gib=9.0,
                hard_stop_gib=10.1,
            )


def test_latency_gate_uses_p95_thresholds_and_fails_closed() -> None:
    gpu = _gpu_module()
    passed = gpu.evaluate_latency_gate(
        smoke_samples_ms=(10.0, 20.0, 30.0, 40.0, 50.0),
        standard_samples_ms=(20.0, 40.0, 60.0, 80.0, 100.0),
        smoke_limit_ms=50.0,
        standard_limit_ms=100.0,
    )
    assert passed.gate_passed is True
    assert passed.smoke_p95_ms == pytest.approx(48.0)
    assert passed.standard_p95_ms == pytest.approx(96.0)

    with pytest.raises(gpu.Stage3GpuGateError, match="latency"):
        gpu.evaluate_latency_gate(
            smoke_samples_ms=(51.0,) * 5,
            standard_samples_ms=(10.0,) * 5,
            smoke_limit_ms=50.0,
            standard_limit_ms=100.0,
        )


def test_latency_verifier_fixture_has_no_external_machine_skip() -> None:
    source = inspect.getsource(_copy_stage3_machine_for_latency_verifier)

    assert "R2_BASE_MACHINE_ROOT" not in source
    assert "pytest.skip" not in source


@pytest.mark.parametrize(
    "sample_field",
    ("smoke_samples_ms", "standard_samples_ms"),
)
def test_stage3_verifier_rejects_truncated_latency_samples_with_synced_manifest(
    sample_field: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, stage = _copy_stage3_machine_for_latency_verifier(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    _rewrite_latency_evidence_and_manifest(
        stage,
        lambda latency: latency.__setitem__(
            sample_field,
            latency[sample_field][:1],
        ),
    )

    with pytest.raises(workflow.Stage3WorkflowError, match="latency"):
        workflow.verify_stage3_machine_run(
            stage_root=stage,
            repo_root=ROOT,
            stage2_gate_path=Path(
                "D:/xunce/out/ppo_frontier/"
                "s2-task3-r4fix-r5-20260713T134447251Z/s2/gate.json"
            ),
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("smoke_p95_ms", lambda value: value + 0.25),
        ("standard_p95_ms", lambda value: value + 0.25),
        ("smoke_limit_ms", lambda value: value - 1.0),
        ("standard_limit_ms", lambda value: value - 1.0),
        ("smoke_sample_count", float),
        ("standard_sample_count", float),
        ("gate_passed", lambda _value: 1),
        ("smoke_p95_ms", lambda _value: float("nan")),
        ("standard_limit_ms", lambda _value: float("inf")),
    ),
    ids=(
        "smoke-p95",
        "standard-p95",
        "smoke-limit",
        "standard-limit",
        "smoke-count-numeric-type",
        "standard-count-numeric-type",
        "gate-bool-type",
        "p95-nan",
        "limit-inf",
    ),
)
def test_stage3_verifier_recomputes_every_recorded_latency_gate_field(
    field: str,
    replacement,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, stage = _copy_stage3_machine_for_latency_verifier(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    _rewrite_latency_evidence_and_manifest(
        stage,
        lambda latency: latency.__setitem__(field, replacement(latency[field])),
    )

    with pytest.raises(workflow.Stage3WorkflowError, match="latency|JSON"):
        workflow.verify_stage3_machine_run(
            stage_root=stage,
            repo_root=ROOT,
            stage2_gate_path=Path(
                "D:/xunce/out/ppo_frontier/"
                "s2-task3-r4fix-r5-20260713T134447251Z/s2/gate.json"
            ),
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-recorded-field",
        "extra-semantic-field",
        "bool-sample",
        "bool-peak-memory",
        "bool-batch-size",
    ),
)
def test_stage3_verifier_rejects_latency_schema_and_numeric_type_drift(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, stage = _copy_stage3_machine_for_latency_verifier(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    def mutate(latency: dict[str, object]) -> None:
        if mutation == "missing-recorded-field":
            latency.pop("smoke_p95_ms")
        elif mutation == "extra-semantic-field":
            latency["recorded_percentile_method"] = "nearest"
        elif mutation == "bool-sample":
            latency["smoke_samples_ms"][0] = True
        elif mutation == "bool-peak-memory":
            latency["smoke_peak_memory_gib"] = True
        elif mutation == "bool-batch-size":
            latency["batch_size"] = True
        else:  # pragma: no cover - parametrization is closed above
            raise AssertionError(f"unknown mutation: {mutation}")

    _rewrite_latency_evidence_and_manifest(stage, mutate)

    with pytest.raises(workflow.Stage3WorkflowError, match="latency|JSON"):
        workflow.verify_stage3_machine_run(
            stage_root=stage,
            repo_root=ROOT,
            stage2_gate_path=Path(
                "D:/xunce/out/ppo_frontier/"
                "s2-task3-r4fix-r5-20260713T134447251Z/s2/gate.json"
            ),
        )


def test_stage3_production_surface_has_no_review_approval_or_gate_issuer() -> None:
    workflow = importlib.import_module("lunar_exploration_ppo.workflows.stage3")
    public_names = set(getattr(workflow, "__all__", ()))
    forbidden_verbs = ("approve", "issue", "record_review", "create_gate", "write_gate")
    assert not any(any(verb in name.lower() for verb in forbidden_verbs) for name in public_names)
    assert workflow.STAGE3_ROOT_ARTIFACTS == (
        "config.json",
        "summary.json",
        "routing.json",
        "manifest.json",
        "report.md",
        "metrics.jsonl",
        "phase-state.jsonl",
        "evidence",
    )


def test_frozen_stage2_source_identity_is_verified_against_approved_commit() -> None:
    workflow = importlib.import_module("lunar_exploration_ppo.workflows.stage3")
    commit = "898911559ccc9ae8ef701b69a58a93b1d8a4d8d8"
    relative = "pyproject.toml"
    payload = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:{relative}"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    digest = hashlib.sha256()
    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    digest.update(payload)
    digest.update(b"\0")
    identity = {
        "schema_version": "stage2_reviewed_source_set/v1",
        "source_set_sha256": digest.hexdigest(),
        "paths": [
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        ],
    }

    workflow.verify_frozen_git_source_identity(
        source_identity=identity,
        repo_root=ROOT,
        commit=commit,
    )

    tampered = json.loads(json.dumps(identity))
    tampered["paths"][0]["sha256"] = "0" * 64
    with pytest.raises(workflow.Stage3WorkflowError, match="source"):
        workflow.verify_frozen_git_source_identity(
            source_identity=tampered,
            repo_root=ROOT,
            commit=commit,
        )


def test_frozen_source_identity_uses_strict_current_bytes_for_recorded_ignored_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = importlib.import_module("lunar_exploration_ppo.workflows.stage3")
    commit = "898911559ccc9ae8ef701b69a58a93b1d8a4d8d8"
    relative = "ignored-brief.md"
    path = tmp_path / relative
    path.write_bytes(b"frozen ignored source\n")
    payload = path.read_bytes()
    monkeypatch.setattr(workflow, "_git_bytes", lambda *_args, **_kwargs: b"")
    monkeypatch.setattr(
        workflow.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )
    digest = hashlib.sha256(
        relative.encode("utf-8") + b"\0" + payload + b"\0"
    ).hexdigest()
    identity = {
        "schema_version": "stage2_reviewed_source_set/v1",
        "source_set_sha256": digest,
        "paths": [
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        ],
    }

    workflow.verify_frozen_git_source_identity(
        source_identity=identity,
        repo_root=tmp_path,
        commit=commit,
    )


def test_stage3_workflow_exports_machine_runner_verifier_and_bound_source_set() -> None:
    workflow = importlib.import_module("lunar_exploration_ppo.workflows.stage3")
    assert callable(workflow.run_stage3_workflow)
    assert callable(workflow.verify_stage3_machine_run)
    identity = workflow.stage3_source_identity(ROOT)
    paths = {entry["path"] for entry in identity["paths"]}
    assert {
        "configs/ppo_highres_frontier_stage3_v1.json",
        "scripts/run_ppo_highres_frontier_stage3.py",
        "src/lunar_exploration_ppo/policy/cross_attention.py",
        "src/lunar_exploration_ppo/workflows/stage3.py",
        "src/lunar_exploration_ppo/workflows/stage3_gpu.py",
        "tests/ppo_highres_frontier/test_stage3_policy_cpu.py",
        "tests/ppo_highres_frontier/test_stage3_policy_cuda.py",
        "tests/ppo_highres_frontier/test_stage3_workflow.py",
        ".github/workflows/platform-compatibility.yml",
    } <= paths
    assert identity["schema_version"] == "stage3_reviewed_source_set/v1"
    assert len(identity["source_set_sha256"]) == 64

    forbidden = ("model_explorer", "scripts.xunce_", "sys.path")
    for entry in identity["paths"]:
        if entry["path"].startswith("src/lunar_exploration_ppo/"):
            source = (ROOT / entry["path"]).read_text(encoding="utf-8")
            assert not any(token in source for token in forbidden)


def test_stage3_runner_is_cli_only_and_requires_external_gate(tmp_path: Path) -> None:
    runner = ROOT / "scripts/run_ppo_highres_frontier_stage3.py"
    source = runner.read_text(encoding="utf-8")
    assert "--stage2-gate" in source
    assert "--run-id" in source
    assert "run_stage3_workflow" in source
    assert "approval.json" not in source
    assert "gate.json" not in source
    assert "sys.path" not in source


def test_stage3_artifact_snapshot_set_detects_post_capture_mutation(
    tmp_path: Path,
) -> None:
    workflow = importlib.import_module("lunar_exploration_ppo.workflows.stage3")
    stage = tmp_path / "s3"
    evidence = stage / "evidence"
    evidence.mkdir(parents=True)
    payloads = {
        "config.json": b"{}\n",
        "evidence/sample.bin": b"sample-v1",
    }
    for relative, payload in payloads.items():
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    manifest = {
        "schema_version": "sha256_manifest/v1",
        "artifacts": [
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for relative, payload in sorted(payloads.items())
        ],
    }

    snapshots = workflow._capture_stage3_artifact_snapshots(
        stage=stage,
        manifest=manifest,
        expected_paths=set(payloads),
    )
    assert {path: snapshot.payload for path, snapshot in snapshots.items()} == payloads
    (stage / "evidence/sample.bin").write_bytes(b"sample-v2")
    with pytest.raises(Stage1WorkflowError, match="drift"):
        snapshots["evidence/sample.bin"].require_current("sample")


def test_platform_workflow_runs_stage3_cpu_contract_without_cuda_fallback() -> None:
    workflow_text = (ROOT / ".github/workflows/platform-compatibility.yml").read_text(
        encoding="utf-8"
    )
    assert "Stage 3 CPU policy contract" in workflow_text
    assert "test_stage3_policy_cpu.py" in workflow_text
    assert "test_stage3_workflow.py" in workflow_text
    assert "test_stage3_policy_cuda.py" not in workflow_text


def test_stage3_verifier_fails_closed_when_stage2_manifest_artifact_drifts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subordinate = tmp_path / "stage2-summary.json"
    subordinate.write_bytes(b"summary-a\n")
    workflow, stage, _authority, mutation = _install_minimal_stage3_verifier_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        authority_path=subordinate,
    )
    mutation["callback"] = lambda: subordinate.write_bytes(b"summary-b\n")

    with pytest.raises(workflow.Stage3WorkflowError, match="authority.*changed"):
        workflow.verify_stage3_machine_run(
            stage_root=stage,
            repo_root=tmp_path,
            stage2_gate_path=tmp_path / "s2" / "gate.json",
        )


def test_stage3_verifier_fails_closed_when_stage2_authority_drifts_after_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = tmp_path / "stage2-gate.json"
    gate.write_bytes(b"gate-a\n")
    workflow, stage, _authority, mutation = _install_minimal_stage3_verifier_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        authority_path=gate,
    )
    mutation["callback"] = lambda: gate.write_bytes(b"gate-b\n")

    with pytest.raises(workflow.Stage3WorkflowError, match="authority.*changed"):
        workflow.verify_stage3_machine_run(
            stage_root=stage,
            repo_root=tmp_path,
            stage2_gate_path=tmp_path / "s2" / "gate.json",
        )


def test_stage3_verifier_fails_closed_on_stage2_authority_aba_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = tmp_path / "stage2-gate.json"
    gate.write_bytes(b"gate-a\n")
    workflow, stage, _authority, mutation = _install_minimal_stage3_verifier_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        authority_path=gate,
    )

    def replace_with_same_bytes() -> None:
        replacement = tmp_path / "stage2-gate-replacement.json"
        replacement.write_bytes(b"gate-a\n")
        os.replace(replacement, gate)

    mutation["callback"] = replace_with_same_bytes

    with pytest.raises(workflow.Stage3WorkflowError, match="authority.*changed"):
        workflow.verify_stage3_machine_run(
            stage_root=stage,
            repo_root=tmp_path,
            stage2_gate_path=tmp_path / "s2" / "gate.json",
        )


def test_stage3_review_controller_fails_closed_on_post_verifier_authority_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _load_stage3_review_controller()
    stage = tmp_path / "machine-run" / "s3"
    stage.mkdir(parents=True)
    review_root = tmp_path / "review"
    review_root.mkdir()
    stage2_gate = tmp_path / "stage2-gate.json"
    stage2_gate.write_bytes(b"gate-a\n")
    subordinate = tmp_path / "stage2-summary.json"
    subordinate.write_bytes(b"summary-a\n")
    source_identity = {"source_set_sha256": "1" * 64}
    environment_identity = {"python": "probe"}
    git_identity = {
        "prospective_git_tree": "2" * 40,
        "changed_paths": ["probe.txt"],
    }
    documents = {
        "config.json": {
            "execution_source_identity": source_identity,
            "execution_environment_identity": environment_identity,
            "execution_git_identity": git_identity,
        },
        "summary.json": {
            "goal_id": "ppo-highres-frontier-map-exploration",
            "stage_id": "ppo_highres_frontier_stage3_cross_attention_policy/v1",
            "run_id": "machine-run",
            "prospective_git_tree": git_identity["prospective_git_tree"],
            "execution_source_set_sha256": source_identity["source_set_sha256"],
            "checkpoint_state": "stage3_no_checkpoint/v1",
        },
        "routing.json": {
            "route": "awaiting_independent_review",
            "authorized_next_stage": "ppo_highres_frontier_stage4_rollout_ppo_update/v1",
        },
        "manifest.json": {"schema_version": "probe/v1"},
    }
    for name, value in documents.items():
        (stage / name).write_bytes(ArtifactStore.canonical_json_bytes(value))
    authority = _AuthorityProbe(
        {
            "verified": True,
            "gate_sha256": hashlib.sha256(stage2_gate.read_bytes()).hexdigest(),
        },
        (("summary.json", FrozenFileSnapshot.capture(subordinate)),),
    )
    machine = _MachineVerificationProbe(
        artifacts={
            name: FrozenFileSnapshot.capture(stage / name)
            for name in documents
        },
        stage2_authority=authority,
    )
    monkeypatch.setattr(controller, "verify_stage3_machine_run", lambda **_kwargs: machine)

    def prepare_package(*, package_path: Path, **_kwargs):
        package_path.write_bytes(b"review-package\n")
        return SimpleNamespace(
            base_commit=controller.BASE_COMMIT,
            prospective_git_tree=git_identity["prospective_git_tree"],
            paths=tuple(git_identity["changed_paths"]),
        )

    monkeypatch.setattr(controller, "prepare_stage2_review_package", prepare_package)
    monkeypatch.setattr(controller, "replay_stage2_review_package", lambda **_kwargs: None)
    monkeypatch.setattr(controller, "stage3_source_identity", lambda _repo: source_identity)
    monkeypatch.setattr(controller, "stage3_environment_identity", lambda: environment_identity)

    def mutate_authority_then_return_git(*_args, **_kwargs):
        subordinate.write_bytes(b"summary-b\n")
        return git_identity

    monkeypatch.setattr(controller, "stage3_git_identity", mutate_authority_then_return_git)

    with pytest.raises(RuntimeError, match="authority.*drift"):
        controller._build_bindings(
            repo=tmp_path,
            stage=stage,
            stage2_gate=stage2_gate,
            review_root=review_root,
            create_package=True,
        )


def test_stage3_machine_controller_rechecks_original_authority_and_config_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = importlib.import_module("lunar_exploration_ppo.workflows.stage3")
    config_source = tmp_path / "stage3-config.json"
    config_source.write_bytes(b"config-a\n")
    config_snapshot = FrozenFileSnapshot.capture(config_source)
    authority_file = tmp_path / "stage2-authority.json"
    authority_file.write_bytes(b"authority-a\n")
    authority_snapshot = FrozenFileSnapshot.capture(authority_file)

    class CountingConfigSnapshot:
        def __init__(self, snapshot: FrozenFileSnapshot) -> None:
            self.payload = snapshot.payload
            self.require_calls = 0
            self._snapshot = snapshot

        def require_current(self, label: str) -> None:
            self.require_calls += 1
            self._snapshot.require_current(label)

    counting_config = CountingConfigSnapshot(config_snapshot)

    class ConfigCaptureFactory:
        @classmethod
        def capture(cls, *_args, **_kwargs):
            return counting_config

    monkeypatch.setattr(workflow, "FrozenFileSnapshot", ConfigCaptureFactory)
    config = SimpleNamespace(
        run_id=None,
        output_root=str(tmp_path / "out"),
        stage2_approval_commit="898911559ccc9ae8ef701b69a58a93b1d8a4d8d8",
    )
    monkeypatch.setattr(
        workflow.Stage3Config,
        "model_validate_json",
        classmethod(lambda _cls, _payload: config),
    )
    source_identity = {"source_set_sha256": "1" * 64}
    environment_identity = {"environment_sha256": "2" * 64}
    git_identity = {"prospective_git_tree": "3" * 40}
    monkeypatch.setattr(workflow, "stage3_source_identity", lambda _repo: source_identity)
    monkeypatch.setattr(workflow, "stage3_environment_identity", lambda: environment_identity)
    monkeypatch.setattr(
        workflow,
        "stage3_git_identity",
        lambda _repo, *, base_commit: git_identity,
    )
    authority = _AuthorityProbe(
        {"verified": True},
        (("authority", authority_snapshot),),
    )
    monkeypatch.setattr(
        workflow,
        "verify_frozen_stage2_authority",
        lambda **_kwargs: authority,
    )
    monkeypatch.setattr(workflow, "run_stage3_cuda_audit", lambda _config: object())
    artifacts = {
        relative: b"controller-probe\n"
        for relative in workflow.STAGE3_MANIFEST_BOUND_ARTIFACTS
    }
    manifest = workflow._manifest_for_artifacts(artifacts)
    monkeypatch.setattr(
        workflow,
        "_build_stage3_machine_payload",
        lambda **_kwargs: workflow._Stage3MachinePayload(
            artifacts=artifacts,
            manifest=manifest,
            summary={"state": "machine_passed"},
        ),
    )

    class MachineProbe:
        require_calls = 0

        def require_current(self, _label: str) -> None:
            self.require_calls += 1

    machine = MachineProbe()

    def replace_config_then_verify(**_kwargs):
        replacement = tmp_path / "stage3-config-replacement.json"
        replacement.write_bytes(config_snapshot.payload)
        os.replace(replacement, config_source)
        return machine

    monkeypatch.setattr(workflow, "verify_stage3_machine_run", replace_config_then_verify)

    with pytest.raises(workflow.Stage3WorkflowError, match="config.*changed"):
        workflow.run_stage3_workflow(
            config_path=config_source,
            run_id="controller-aba-probe",
            stage2_gate_path=tmp_path / "s2" / "gate.json",
            base_output_root=tmp_path / "out",
        )
    assert authority.require_calls == 2
    assert counting_config.require_calls == 2
    assert machine.require_calls == 0


@pytest.mark.parametrize("mode", ("prepare", "verify"))
def test_stage3_review_controller_main_keeps_handle_through_bindings_finalization(
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _load_stage3_review_controller()
    review_root = tmp_path / f"review-{mode}"
    bindings = {"schema_version": "probe-bindings/v1", "run_id": "probe"}
    recorded_snapshot = None
    if mode == "verify":
        review_root.mkdir()
        bindings_path = review_root / "bindings.json"
        bindings_path.write_bytes(ArtifactStore.canonical_json_bytes(bindings))
        recorded_snapshot = FrozenFileSnapshot.capture(bindings_path)
    authority_path = tmp_path / f"stage2-authority-{mode}.json"
    authority_path.write_bytes(b"authority-a\n")
    authority_snapshot = FrozenFileSnapshot.capture(authority_path)
    mutated = False

    def mutate_authority() -> None:
        nonlocal mutated
        if not mutated:
            authority_path.write_bytes(b"authority-b\n")
            mutated = True

    class ControllerHandle(dict):
        recorded_bindings_snapshot = recorded_snapshot

        @property
        def bindings(self) -> dict[str, object]:
            if mode == "verify":
                mutate_authority()
            return dict(self)

        def require_current(self, _label: str) -> None:
            authority_snapshot.require_current("controller authority")

    handle = ControllerHandle(bindings)
    monkeypatch.setattr(
        controller.argparse.ArgumentParser,
        "parse_args",
        lambda _self: SimpleNamespace(
            mode=mode,
            repo=str(tmp_path),
            stage=str(tmp_path / "machine" / "s3"),
            stage2_gate=str(tmp_path / "stage2" / "s2" / "gate.json"),
            review_root=str(review_root),
        ),
    )
    monkeypatch.setattr(controller, "_build_bindings", lambda **_kwargs: handle)
    original_write_json = controller.ArtifactStore.write_json

    def write_then_mutate(store, relative: str, value: object) -> None:
        original_write_json(store, relative, value)
        mutate_authority()

    monkeypatch.setattr(controller.ArtifactStore, "write_json", write_then_mutate)

    with pytest.raises(RuntimeError, match="authority.*drift"):
        controller.main()
