"""Stage 6 真实 Standard/CUDA/spawn machine preflight 合同。"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/ppo_highres_frontier_stage6_v1.json"
STAGE5_GATE = Path(
    "D:/xunce/out/ppo_frontier/"
    "s5-task6-fix-r1-20260714T223702Z/s5/gate.json"
)
FORMAL_CACHE_MANIFEST = Path(
    "D:/xunce/review/s6-coverable-cache-r1-20260722T144932/"
    "coverage-cache-manifest.json"
)


def _valid_environment_identity() -> dict[str, object]:
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


def _environment_sha256(identity: dict[str, object]) -> str:
    return hashlib.sha256(
        (
            json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
    ).hexdigest()


def _valid_audit() -> dict[str, object]:
    timing_names = (
        "catalog_build",
        "catalog_cache_reuse",
        "scenario_build",
        "coverable_env_init",
        "reset",
        "step",
        "cuda_policy_load",
        "cuda_forward",
        "cuda_backward",
        "spawn_startup",
        "spawn_reset",
        "spawn_inference",
        "spawn_step",
        "spawn_close",
    )
    environment_identity = _valid_environment_identity()
    return {
        "schema_version": "stage6_machine_preflight/v1",
        "passed": True,
        "config_sha256": "1" * 64,
        "stage5_gate_sha256": "2" * 64,
        "environment_identity": environment_identity,
        "environment_sha256": _environment_sha256(environment_identity),
        "resources": {
            "d_free_bytes": 101 * 1024**3,
            "rss_bytes": 2 * 1024**3,
            "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
            "rss_root_pid": 99,
            "rss_sample_count": 3,
            "rss_latest_process_count": 9,
            "rss_peak_process_count": 9,
            "peak_vram_bytes": 1 * 1024**3,
            "warnings": [],
            "hard_stops": [],
        },
        "standard": {
            "catalog_sha256": "3" * 64,
            "scenario_id": "train/medium/0000",
            "prior_shape": [7, 32, 32],
            "truth_shape": [256, 256],
            "local_crop_shape": [8, 96, 96],
            "candidate_slots": 1024,
            "coverable_exact": True,
            "step_trainable": True,
            "cache_scope": {
                "catalog_process_cache_hit": True,
                "runtime_mode": "persistent_exact_manifest_read_only/v1",
                "manifest": {
                    "path": (
                        "D:/xunce/review/s6-coverable-cache-r1-20260722T144932/"
                        "coverage-cache-manifest.json"
                    ),
                    "sha256": (
                        "675b923b64cee4cde90d5728550a7f939c41cfbc56e9c237268078b59641081c"
                    ),
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
                },
                "formal_audit": {
                    "path": (
                        "D:/xunce/review/s6-coverable-cache-r1-20260722T144932/"
                        "formal-audit.json"
                    ),
                    "sha256": (
                        "d7c12737da4c9a43353e02347b9c9d6abd527a78f7a3c6675e9164b6ae5b0068"
                    ),
                    "size_bytes": 4416,
                },
                "cold_spawn": {
                    "verification": "spawn_worker_reset/v1",
                    "scenario_id": "train/medium/0000",
                    "worker_pid": 101,
                    "parent_pid": 99,
                    "start_method": "spawn",
                    "cache_hit": True,
                    "compute_fallback_called": False,
                },
            },
        },
        "cuda": {
            "available": True,
            "device": "cuda:0",
            "compute_dtype": "float32",
            "amp_enabled": False,
            "autocast_enabled": False,
            "checkpoint_sha256": "4" * 64,
            "policy_state_sha256": "5" * 64,
            "forward_warmup_count": 5,
            "forward_measurement_count": 20,
            "forward_p95_ms": 99.999,
            "outputs_finite": True,
            "joint_logprob_finite": True,
            "backward_completed": True,
            "gradient_tensor_count": 90,
            "gradients_finite": True,
            "peak_vram_bytes": 1 * 1024**3,
        },
        "collector": {
            "worker_count": 8,
            "worker_pids": list(range(101, 109)),
            "worker_start_methods": ["spawn"] * 8,
            "inference_pids": [99],
            "parent_pid": 99,
            "trainable_transition_count": 8,
            "snapshot_count": 8,
            "policy_state_sha256": "5" * 64,
            "all_workers_closed": True,
            "residual_child_pids": [],
        },
        "timings_seconds": {name: 0.01 for name in timing_names},
    }


def test_stage6_environment_identity_is_complete_canonical_and_cuda_bound() -> None:
    from lunar_exploration_ppo.workflows import stage6

    capture = getattr(stage6, "stage6_environment_identity", None)
    identity_hash = getattr(stage6, "stage6_environment_sha256", None)

    assert callable(capture)
    assert callable(identity_hash)
    identity = capture()
    assert set(identity) == {
        "schema_version",
        "python_version",
        "python_implementation",
        "os_name",
        "platform_system",
        "platform_machine",
        "numpy_version",
        "torch_version",
        "torch_cuda_version",
        "cudnn_version",
        "cuda_available",
        "cuda_device_count",
        "cuda_current_device",
        "cuda_device_name",
        "cuda_compute_capability",
        "cuda_total_vram_bytes",
        "compute_dtype",
        "amp_enabled",
    }
    assert identity["schema_version"] == "stage6_environment_identity/v1"
    assert identity["cuda_available"] is True
    assert identity["cuda_device_count"] >= 1
    assert 0 <= identity["cuda_current_device"] < identity["cuda_device_count"]
    assert len(identity["cuda_compute_capability"]) == 2
    assert identity["cuda_total_vram_bytes"] > 0
    assert identity["compute_dtype"] == "float32"
    assert identity["amp_enabled"] is False
    assert len(identity_hash(identity)) == 64


def test_stage6_environment_identity_rejects_cuda_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from lunar_exploration_ppo.workflows import stage6

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(stage6.Stage6WorkflowError, match="requires CUDA"):
        stage6.stage6_environment_identity()


def test_machine_preflight_binds_environment_identity_and_hash() -> None:
    from lunar_exploration_ppo.workflows import stage6

    audit = _valid_audit()
    assert stage6.validate_stage6_machine_preflight_audit(audit) == audit

    field_drift = _valid_audit()
    field_drift["environment_identity"]["python_version"] = "3.12.14"  # type: ignore[index]
    with pytest.raises(stage6.Stage6WorkflowError, match="environment"):
        stage6.validate_stage6_machine_preflight_audit(field_drift)

    hash_drift = _valid_audit()
    hash_drift["environment_sha256"] = "f" * 64
    with pytest.raises(stage6.Stage6WorkflowError, match="environment"):
        stage6.validate_stage6_machine_preflight_audit(hash_drift)


def test_machine_preflight_requires_exact_manifest_cold_spawn_hit_and_no_fallback() -> None:
    from lunar_exploration_ppo.workflows import stage6

    audit = _valid_audit()
    assert stage6.validate_stage6_machine_preflight_audit(audit) == audit

    bad_mode = _valid_audit()
    bad_mode["standard"]["cache_scope"]["runtime_mode"] = (  # type: ignore[index]
        "process_local_scenario_hash/v1"
    )
    with pytest.raises(stage6.Stage6WorkflowError, match="Standard"):
        stage6.validate_stage6_machine_preflight_audit(bad_mode)

    wrong_entry_set = _valid_audit()
    wrong_entry_set["standard"]["cache_scope"]["manifest"][  # type: ignore[index]
        "entry_set_sha256"
    ] = "f" * 64
    with pytest.raises(stage6.Stage6WorkflowError, match="Standard"):
        stage6.validate_stage6_machine_preflight_audit(wrong_entry_set)

    fallback = _valid_audit()
    fallback["standard"]["cache_scope"]["cold_spawn"][  # type: ignore[index]
        "compute_fallback_called"
    ] = True
    with pytest.raises(stage6.Stage6WorkflowError, match="Standard"):
        stage6.validate_stage6_machine_preflight_audit(fallback)


def test_machine_preflight_accepts_latest_parent_sample_when_peak_covers_workers() -> None:
    from lunar_exploration_ppo.workflows import stage6

    audit = _valid_audit()
    audit["resources"]["rss_latest_process_count"] = 1  # type: ignore[index]

    assert stage6.validate_stage6_machine_preflight_audit(audit) == audit


def test_machine_preflight_validator_requires_real_cuda_spawn_and_all_segments() -> None:
    from lunar_exploration_ppo.workflows import stage6

    audit = _valid_audit()
    assert stage6.validate_stage6_machine_preflight_audit(audit) == audit

    bad_latency = _valid_audit()
    bad_latency["cuda"]["forward_p95_ms"] = 100.000001  # type: ignore[index]
    with pytest.raises(stage6.Stage6WorkflowError, match="forward"):
        stage6.validate_stage6_machine_preflight_audit(bad_latency)

    bad_workers = _valid_audit()
    bad_workers["collector"]["worker_count"] = 7  # type: ignore[index]
    with pytest.raises(stage6.Stage6WorkflowError, match="collector"):
        stage6.validate_stage6_machine_preflight_audit(bad_workers)

    residual = _valid_audit()
    residual["collector"]["residual_child_pids"] = [108]  # type: ignore[index]
    with pytest.raises(stage6.Stage6WorkflowError, match="collector"):
        stage6.validate_stage6_machine_preflight_audit(residual)

    missing_timing = _valid_audit()
    missing_timing["timings_seconds"].pop("coverable_env_init")  # type: ignore[union-attr]
    with pytest.raises(stage6.Stage6WorkflowError, match="timing"):
        stage6.validate_stage6_machine_preflight_audit(missing_timing)


def test_machine_preflight_binds_cuda_vram_to_resource_gate_and_rejects_exact_limit() -> None:
    from lunar_exploration_ppo.workflows import stage6

    hard_limit = int(10.1 * 1024**3)

    exact_cuda_limit = _valid_audit()
    exact_cuda_limit["cuda"]["peak_vram_bytes"] = hard_limit  # type: ignore[index]
    with pytest.raises(stage6.Stage6WorkflowError, match="CUDA"):
        stage6.validate_stage6_machine_preflight_audit(exact_cuda_limit)

    mismatched_sections = _valid_audit()
    mismatched_sections["cuda"]["peak_vram_bytes"] += 1  # type: ignore[index,operator]
    with pytest.raises(stage6.Stage6WorkflowError, match="VRAM binding"):
        stage6.validate_stage6_machine_preflight_audit(mismatched_sections)


def test_machine_preflight_and_stage4_loader_surfaces_are_production_only() -> None:
    from lunar_exploration_ppo.workflows import stage6

    assert tuple(inspect.signature(stage6.run_stage6_machine_preflight).parameters) == (
        "config_path",
        "run_id",
        "stage5_gate_path",
    )
    assert tuple(
        inspect.signature(stage6.load_stage4_policy_for_standard).parameters
    ) == (
        "checkpoint_path",
        "checkpoint_sha256",
        "policy_state_sha256",
        "device",
    )
    source = inspect.getsource(stage6.run_stage6_machine_preflight).lower()
    for forbidden in ("fixture", "fake", "adapter", "force", "skip", "cpu"):
        assert forbidden not in source


def test_machine_preflight_sources_spawn_and_direct_env_safety_from_frozen_config() -> None:
    from lunar_exploration_ppo.workflows import stage6

    source_path = inspect.getsourcefile(stage6)
    assert source_path is not None
    source = Path(source_path).read_text(encoding="utf-8")
    assert "SafetyContract.from_stage6_config(config)" in source
    assert source.count("safety_contract=safety_contract") >= 2
    assert source.count("config_sha256=config_sha256") >= 2


@pytest.mark.skipif(
    not FORMAL_CACHE_MANIFEST.is_file(),
    reason="fixed Stage 6 formal coverage-cache manifest is unavailable",
)
def test_preflight_cache_specs_trace_no_process_local_compute_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.configs.stage6 import SafetyContract, load_stage6_config
    from lunar_exploration_ppo.env.standard_training import build_standard_catalog
    from lunar_exploration_ppo.eval.standard import (
        build_standard_evaluation_schedule,
        standard_evaluation_env_specs,
    )
    from lunar_exploration_ppo.ppo import standard_training as training_module
    from lunar_exploration_ppo.workflows import stage6

    env_module = importlib.import_module("lunar_exploration_ppo.env.env")
    config = load_stage6_config(CONFIG)
    config_sha256 = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    catalog = build_standard_catalog(verify_hashes=True)
    specs = stage6._stage6_cached_preflight_env_specs(
        catalog=catalog,
        sampler_seeds=tuple(
            config.training.seeds[0] + index
            for index in range(config.rollout.num_envs)
        ),
        safety_contract=SafetyContract.from_stage6_config(config),
        config_sha256=config_sha256,
    )

    def forbidden_compute(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("production cache hit must not compute coverage masks")

    env_module._COVERAGE_CACHE.clear()
    monkeypatch.setattr(env_module, "compute_coverage_masks", forbidden_compute)
    environment = specs[0].factory(**dict(specs[0].kwargs))
    try:
        observation = environment.reset()
        assert observation.frontier_features.shape[0] == 1024
        assert environment.coverage_metadata["exact"] is True
        assert environment._coverage_cache_manifest is not None
        assert env_module._COVERAGE_CACHE == {}
    finally:
        environment.close()

    schedule = build_standard_evaluation_schedule(
        catalog,
        split="validation",
        episode_count=16,
        evaluation_seed_start=config.evaluation.bootstrap_seed,
        theta_source="policy_theta_mu/v1",
    )
    legacy_evaluation_specs = standard_evaluation_env_specs(
        catalog,
        schedule,
        safety_contract=SafetyContract.from_stage6_config(config),
        config_sha256=config_sha256,
    )
    evaluation_specs = training_module._cache_bound_standard_evaluation_specs(
        legacy_evaluation_specs,
        coverage_cache_manifest_path=(
            stage6.STAGE6_COVERAGE_CACHE_MANIFEST_PATH.as_posix()
        ),
        coverage_cache_manifest_sha256=(
            stage6.STAGE6_COVERAGE_CACHE_MANIFEST_SHA256
        ),
    )
    evaluation_environment = evaluation_specs[0].factory(
        **dict(evaluation_specs[0].kwargs)
    )
    try:
        evaluation_observation = evaluation_environment.reset()
        assert evaluation_observation.frontier_features.shape[0] == 1024
        assert evaluation_environment._coverage_cache_manifest is not None
        assert env_module._COVERAGE_CACHE == {}
    finally:
        evaluation_environment.close()


@pytest.mark.skipif(
    not FORMAL_CACHE_MANIFEST.is_file(),
    reason="fixed Stage 6 formal coverage-cache manifest is unavailable",
)
def test_preflight_cache_specs_hit_manifest_in_a_fresh_spawn_process() -> None:
    from lunar_exploration_ppo.configs.stage6 import SafetyContract, load_stage6_config
    from lunar_exploration_ppo.env.standard_training import build_standard_catalog
    from lunar_exploration_ppo.ppo.collector import SpawnVectorEnv
    from lunar_exploration_ppo.workflows import stage6

    config = load_stage6_config(CONFIG)
    config_sha256 = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    catalog = build_standard_catalog(verify_hashes=True)
    specs = stage6._stage6_cached_preflight_env_specs(
        catalog=catalog,
        sampler_seeds=tuple(
            config.training.seeds[0] + index
            for index in range(config.rollout.num_envs)
        ),
        safety_contract=SafetyContract.from_stage6_config(config),
        config_sha256=config_sha256,
    )
    vector = SpawnVectorEnv(specs, timeout_seconds=60.0)
    worker_pids = vector.worker_pids
    worker_start_methods = vector.worker_start_methods
    try:
        observations = vector.reset(worker_indices=(0,))
        assert tuple(observations) == (0,)
        assert observations[0].observation.frontier_features.shape[0] == 1024
    finally:
        vector.close()

    assert len(worker_pids) == 8
    assert all(pid != os.getpid() for pid in worker_pids)
    assert worker_start_methods == ("spawn",) * 8
    assert vector.closed is True
    assert not any(vector.worker_alive)


def test_preflight_rss_monitor_brackets_spawn_worker_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows import stage6

    collector_module = importlib.import_module(
        "lunar_exploration_ppo.ppo.collector"
    )
    resources_module = importlib.import_module(
        "lunar_exploration_ppo.utils.resources"
    )
    events: list[str] = []
    monitors: list[object] = []
    vectors: list[object] = []
    result = object()

    class FakeMonitor:
        def __init__(self) -> None:
            self.running = False
            self.peak_aggregate_rss_bytes = 0
            monitors.append(self)

        def start(self):
            events.append("monitor:start")
            self.running = True
            return self

        def sample_now(self) -> object:
            assert self.running is True
            assert vectors and any(vectors[0].worker_alive)
            events.append("monitor:sample-workers-alive")
            self.peak_aggregate_rss_bytes = 18 * 1024**3
            return object()

        def stop(self) -> None:
            assert vectors and vectors[0].closed is True
            events.append("monitor:stop")
            self.running = False

    class FakeVectorEnv:
        def __init__(self, specs: object, *, timeout_seconds: float) -> None:
            assert specs == ("spec",) * 8
            assert timeout_seconds == 900.0
            assert monitors and monitors[0].running is True
            events.append("spawn:init")
            self.worker_pids = tuple(range(7101, 7109))
            self.worker_alive = [True] * 8
            self.closed = False
            vectors.append(self)

        def close(self) -> None:
            assert monitors[0].running is True
            events.append("spawn:close")
            self.worker_alive = [False] * 8
            self.closed = True

    class FakeCollector:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["vector_env"] is vectors[0]

        def preflight_one_step(self) -> object:
            assert monitors[0].running is True
            assert any(vectors[0].worker_alive)
            events.append("collector:preflight")
            return result

    monkeypatch.setattr(resources_module, "ProcessTreeRSSMonitor", FakeMonitor)
    monkeypatch.setattr(collector_module, "SpawnVectorEnv", FakeVectorEnv)
    monkeypatch.setattr(collector_module, "RolloutCollector", FakeCollector)

    observed = stage6._run_stage6_spawn_preflight(
        specs=("spec",) * 8,
        policy=object(),
        device="cuda",
        collector_contract=object(),
        timeout_seconds=900.0,
    )

    assert events == [
        "monitor:start",
        "spawn:init",
        "collector:preflight",
        "monitor:sample-workers-alive",
        "spawn:close",
        "monitor:stop",
    ]
    assert observed["preflight_result"] is result
    assert observed["worker_pids"] == tuple(range(7101, 7109))
    assert observed["all_workers_closed"] is True
    assert observed["residual_child_pids"] == []
    assert observed["process_tree_monitor"] is monitors[0]
    assert monitors[0].peak_aggregate_rss_bytes == 18 * 1024**3


def test_preflight_resource_audit_uses_spawn_peak_and_hard_stops_at_20_gib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows import stage6

    resources_module = importlib.import_module(
        "lunar_exploration_ppo.utils.resources"
    )
    monkeypatch.setattr(
        resources_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=101 * 1024**3),
    )
    monitor = SimpleNamespace(
        running=False,
        peak_aggregate_rss_bytes=20 * 1024**3,
        root_pid=os.getpid(),
        sample_count=3,
        latest_process_count=1,
        peak_process_count=9,
    )

    audit, decision = stage6._stage6_preflight_resource_audit(
        process_tree_monitor=monitor,
        peak_vram_bytes=1024,
    )

    assert audit["rss_bytes"] == 20 * 1024**3
    assert audit["rss_source"] == "process_tree_lifecycle_peak_current_sum/v1"
    assert audit["rss_root_pid"] == os.getpid()
    assert audit["rss_sample_count"] == 3
    assert audit["rss_latest_process_count"] == 1
    assert audit["rss_peak_process_count"] == 9
    assert audit["hard_stops"] == ["process RSS reached 20 GiB hard stop"]
    assert decision.passed is False


def test_stage4_loader_rejects_non_cuda_and_hash_drift_before_decode(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.workflows import stage6

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not-the-frozen-checkpoint")
    with pytest.raises(stage6.Stage6WorkflowError, match="CUDA"):
        stage6.load_stage4_policy_for_standard(
            checkpoint_path=checkpoint,
            checkpoint_sha256="a" * 64,
            policy_state_sha256="b" * 64,
            device="cpu",
        )
    with pytest.raises(stage6.Stage6WorkflowError, match="SHA-256"):
        stage6.load_stage4_policy_for_standard(
            checkpoint_path=checkpoint,
            checkpoint_sha256="a" * 64,
            policy_state_sha256="b" * 64,
            device="cuda",
        )


@pytest.mark.skipif(
    os.environ.get("RUN_STAGE6_REAL_PREFLIGHT") != "1",
    reason="explicit real Standard/CUDA/spawn preflight only",
)
def test_real_stage6_machine_preflight_writes_canonical_audit(tmp_path: Path) -> None:
    from lunar_exploration_ppo.workflows import stage6

    run_id = "s6-standard-single-preflight-r1-20260717T010203Z"
    audit = stage6._run_stage6_machine_preflight_for_test(
        config_path=CONFIG,
        run_id=run_id,
        stage5_gate_path=STAGE5_GATE,
        base_output_root=tmp_path,
    )
    path = tmp_path / run_id / "s6/preflight/audit.json"

    assert stage6.validate_stage6_machine_preflight_audit(audit) == audit
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == audit
    assert audit["cuda"]["checkpoint_sha256"] == stage6.STAGE4_CHECKPOINT_SHA256
    assert audit["cuda"]["policy_state_sha256"] == stage6.STAGE4_POLICY_STATE_SHA256
    assert audit["collector"]["worker_count"] == 8
    assert audit["collector"]["all_workers_closed"] is True
