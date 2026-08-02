"""Stage 6 Standard 8-worker evaluation 调度与动作公平性。"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

from lunar_exploration_ppo.env.standard_training import build_standard_catalog
from lunar_exploration_ppo.eval.evaluator import EvaluationSummary
from lunar_exploration_ppo.policy.cross_attention import (
    PolicyForwardOutput,
    batch_policy_observations,
)
from lunar_exploration_ppo.policy.observation import PolicyObservation
from lunar_exploration_ppo.ppo.collector import SpawnEnvSpec, SpawnVectorEnv
from lunar_exploration_ppo.ppo.trainer import policy_state_sha256
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore

from test_stage4_collector import _FastMainProcessPolicy, _SpawnFixtureEnv


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/ppo_highres_frontier_stage6_v1.json"


class _EvalSpawnFixtureEnv(_SpawnFixtureEnv):
    def reset(self):
        if self.reset_count == 0:
            super().reset()
        return super().reset()


def _build_eval_spawn_env(
    *,
    worker_index: int,
    fail_mode: str = "none",
) -> _EvalSpawnFixtureEnv:
    return _EvalSpawnFixtureEnv(
        worker_index=worker_index,
        fail_mode=fail_mode,
    )


def _eval_specs(
    *,
    failure_worker: int | None = None,
    fail_mode: str = "none",
) -> tuple[SpawnEnvSpec, ...]:
    return tuple(
        SpawnEnvSpec(
            factory=_build_eval_spawn_env,
            kwargs={
                "worker_index": worker_index,
                "fail_mode": (
                    fail_mode
                    if worker_index == failure_worker
                    else "none"
                ),
            },
        )
        for worker_index in range(8)
    )


class _CountingMainProcessPolicy(_FastMainProcessPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(self, batch):
        self.forward_calls += 1
        return super().forward(batch)


class _RolloverAuditVector(SpawnVectorEnv):
    def __init__(self, *, inject_cleanup_failure: bool) -> None:
        self.reset_requests: list[tuple[int, ...] | None] = []
        self.step_calls = 0
        self.close_calls = 0
        self.inject_cleanup_failure = inject_cleanup_failure
        super().__init__(_eval_specs(), timeout_seconds=30.0)

    def reset(self, worker_indices=None):
        self.reset_requests.append(
            None if worker_indices is None else tuple(worker_indices)
        )
        return super().reset(worker_indices)

    def step(self, actions):
        self.step_calls += 1
        return super().step(actions)

    def close(self) -> None:
        self.close_calls += 1
        super().close()
        if self.inject_cleanup_failure:
            self.inject_cleanup_failure = False
            raise RuntimeError("secondary vector cleanup failure")


@lru_cache(maxsize=1)
def _catalog():
    return build_standard_catalog(verify_hashes=False)


@lru_cache(maxsize=1)
def _safety_lineage():
    from lunar_exploration_ppo.configs.stage6 import (
        SafetyContract,
        load_stage6_config,
    )

    return (
        SafetyContract.from_stage6_config(load_stage6_config(CONFIG)),
        hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    )


def _safety_validation_kwargs() -> dict[str, object]:
    safety_contract, config_sha256 = _safety_lineage()
    return {
        "safety_contract": safety_contract,
        "config_sha256": config_sha256,
    }


def _observation(recommended_theta: tuple[float, ...]) -> PolicyObservation:
    count = len(recommended_theta)
    features = np.zeros((count, 22), dtype=np.float32)
    features[:, 2] = np.arange(1, count + 1, dtype=np.float32)
    features[:, 5] = np.arange(count, 0, -1, dtype=np.float32)
    features[:, 18] = 1.0
    for index, theta in enumerate(recommended_theta):
        features[index, 14] = np.sin(theta)
        features[index, 15] = np.cos(theta)
    return PolicyObservation(
        prior_channels=np.zeros((7, 32, 32), dtype=np.float32),
        coverage_summary=np.zeros((8, 32, 32), dtype=np.float32),
        local_crop=np.zeros((8, 96, 96), dtype=np.float32),
        frontier_features=features,
        pose_features=np.zeros((6,), dtype=np.float32),
        candidate_mask=np.ones((count,), dtype=bool),
    )


def _policy_output() -> PolicyForwardOutput:
    logits = torch.tensor([[0.0, 2.0], [3.0, 1.0]], dtype=torch.float32)
    theta = torch.tensor([[0.1, 0.7], [-0.4, 0.2]], dtype=torch.float32)
    batch, candidates = logits.shape
    zeros = torch.zeros((batch, candidates), dtype=torch.float32)
    token = torch.zeros((batch, 1, 4), dtype=torch.float32)
    return PolicyForwardOutput(
        frontier_logits=logits,
        theta_mu_sin_raw=zeros,
        theta_mu_cos_raw=zeros,
        theta_kappa_raw=zeros,
        theta_mu=theta,
        theta_kappa=torch.ones_like(theta),
        value=torch.zeros((batch,), dtype=torch.float32),
        global_map_tokens=token,
        local_map_tokens=token,
        pose_token=token,
        context_tokens=token,
        refined_frontier_tokens=torch.zeros((batch, candidates, 4)),
        action_hidden=torch.zeros((batch, candidates, 4)),
    )


def _shared_environment_contract(standard, jobs):
    safety_contract, config_sha256 = _safety_lineage()
    specs = standard.standard_evaluation_env_specs(
        _catalog(),
        jobs,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )
    return standard.build_standard_environment_contract(
        catalog=_catalog(),
        env_specs=specs,
        jobs=jobs,
        max_steps=128,
        success_threshold=0.99,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )


def test_shared_environment_contract_binds_explicit_safety_decomposition() -> None:
    from lunar_exploration_ppo.eval import standard

    jobs = standard.build_standard_evaluation_schedule(
        _catalog(),
        split="validation",
        episode_count=16,
        evaluation_seed_start=20260716,
        theta_source="policy_theta_mu/v1",
    )
    contract = _shared_environment_contract(standard, jobs)

    assert contract["environment_contract"]["safety_contract"] == {
        "vehicle_radius_m": 0.4215874761,
        "safety_margin_m": 0.10,
        "min_clearance_m": 0.5215874761,
        "traversability_threshold": 0.50,
        "max_traversable_slope_deg": 30.0,
    }
    assert contract["environment_contract"]["safety_contract_source"] == (
        "Stage6Config.safety/v1"
    )
    assert contract["environment_contract"]["safety_contract_config_sha256"] == (
        _safety_lineage()[1]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vehicle_radius_m", 0.4215874762),
        ("safety_margin_m", 0.11),
        ("min_clearance_m", 0.5215874762),
        ("traversability_threshold", 0.51),
        ("max_traversable_slope_deg", 29.0),
    ],
)
def test_shared_environment_contract_rejects_each_safety_component_drift(
    field: str,
    value: float,
) -> None:
    from lunar_exploration_ppo.eval import standard

    jobs = standard.build_standard_evaluation_schedule(
        _catalog(),
        split="validation",
        episode_count=16,
        evaluation_seed_start=20260716,
        theta_source="policy_theta_mu/v1",
    )
    contract = _shared_environment_contract(standard, jobs)
    contract["environment_contract"]["safety_contract"][field] = value

    with pytest.raises(standard.StandardEvaluationError, match="safety"):
        safety_contract, config_sha256 = _safety_lineage()
        standard._validate_standard_environment_contract(
            contract,
            episode_count=16,
            safety_contract=safety_contract,
            config_sha256=config_sha256,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("safety_contract_source", "hardcoded_defaults/v1"),
        ("safety_contract_config_sha256", "0" * 64),
    ],
)
def test_shared_environment_contract_rejects_config_source_drift(
    field: str,
    value: str,
) -> None:
    from lunar_exploration_ppo.eval import standard

    jobs = standard.build_standard_evaluation_schedule(
        _catalog(),
        split="validation",
        episode_count=16,
        evaluation_seed_start=20260716,
        theta_source="policy_theta_mu/v1",
    )
    contract = _shared_environment_contract(standard, jobs)
    contract["environment_contract"][field] = value
    safety_contract, config_sha256 = _safety_lineage()

    with pytest.raises(standard.StandardEvaluationError, match="safety|config"):
        standard._validate_standard_environment_contract(
            contract,
            episode_count=16,
            safety_contract=safety_contract,
            config_sha256=config_sha256,
        )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(ArtifactStore.canonical_json_bytes(value)).hexdigest()


def _single_policy_output() -> PolicyForwardOutput:
    output = _policy_output()
    return PolicyForwardOutput(
        **{
            field: getattr(output, field)[:1]
            for field in output.__dataclass_fields__
        }
    )


def _valid_fairness_audit(
    standard,
    *,
    method: str,
    zero_step_first_episode: bool = False,
) -> dict[str, object]:
    theta_source = (
        "policy_theta_mu/v1"
        if method == "ppo_policy"
        else "candidate_recommended_theta/v1"
    )
    jobs = standard.build_standard_evaluation_schedule(
        _catalog(),
        split="validation",
        episode_count=16,
        evaluation_seed_start=20260716,
        theta_source=theta_source,
    )
    contract = _shared_environment_contract(standard, jobs)
    decision_schema = standard.build_standard_decision_input_schema()
    action_rule = standard.build_standard_action_rule_provenance(method)
    decisions = []
    action_counts = []
    for episode_index, job in enumerate(jobs):
        if zero_step_first_episode and episode_index == 0:
            action_counts.append(0)
            continue
        observation = _observation((0.2, 1.1))
        if method == "ppo_policy":
            batch = batch_policy_observations((observation,))
            output = _single_policy_output()
            action = standard.select_standard_evaluation_actions(
                method=method,
                observations=(observation,),
                evaluation_seeds=(job.evaluation_seed,),
                policy_output=output,
            )[0]
            selector_inputs = {
                "observation_batch": batch,
                "policy_output": output,
            }
            batch_row = 0
        else:
            action = standard.select_standard_evaluation_actions(
                method=method,
                observations=(observation,),
                evaluation_seeds=(job.evaluation_seed,),
                policy_output=None,
            )[0]
            selector_inputs = {"policy_observation": observation}
            batch_row = None
        decisions.append(
            standard.build_standard_decision_record(
                method=method,
                sequence_index=len(decisions),
                episode_index=episode_index,
                step_index=0,
                worker_index=episode_index % 8,
                observation=observation,
                action=action,
                selector_inputs=selector_inputs,
                policy_batch_row_index=batch_row,
            )
        )
        action_counts.append(1)
    parent_pid = 12_345
    raw = {
        "schema_version": "stage6_standard_parallel_fairness_audit/v2",
        "method": method,
        "split": "validation",
        "worker_count": 8,
        "worker_pids": list(range(20_000, 20_008)),
        "worker_start_methods": ["spawn"] * 8,
        "evaluation_parent_pid": parent_pid,
        "inference_pids": [parent_pid] if method == "ppo_policy" else [],
        "policy_state_unchanged": True,
        "policy_state_sha256_before": ("e" * 64 if method == "ppo_policy" else None),
        "policy_state_sha256_after": ("e" * 64 if method == "ppo_policy" else None),
        "episode_action_counts": action_counts,
        "selected_action_count": sum(action_counts),
        "scenario_schedule": [job.scenario_id for job in jobs],
        "evaluation_seeds": [job.evaluation_seed for job in jobs],
        "theta_source": theta_source,
        "shared_environment_contract": contract,
        "shared_environment_contract_sha256": _canonical_sha256(contract),
        "decision_input_schema": decision_schema,
        "decision_input_schema_sha256": _canonical_sha256(decision_schema),
        "action_rule": action_rule,
        "action_rule_sha256": _canonical_sha256(action_rule),
        "decision_audit": standard._build_standard_decision_audit(
            method,
            decisions,
        ),
    }
    return standard.validate_standard_fairness_audit(
        raw,
        episode_count=16,
        parent_pid=parent_pid,
        **_safety_validation_kwargs(),
    )


def test_standard_eval_schedule_is_fixed_split_isolated_and_eight_lanes() -> None:
    from lunar_exploration_ppo.eval import standard

    schedule = standard.build_standard_evaluation_schedule(
        _catalog(),
        split="validation",
        episode_count=16,
        evaluation_seed_start=20260716,
    )
    lanes = standard.partition_standard_evaluation_jobs(schedule)

    expected_ids = tuple(
        record.scenario_id
        for record in _catalog().records
        if record.split == "validation"
    )[:16]
    assert len(schedule) == 16
    assert tuple(job.scenario_id for job in schedule) == expected_ids
    assert tuple(job.evaluation_seed for job in schedule) == tuple(
        range(20260716, 20260732)
    )
    assert len(lanes) == 8
    assert all(len(lane) == 2 for lane in lanes)
    assert tuple(
        sorted((job for lane in lanes for job in lane), key=lambda job: job.episode_index)
    ) == schedule
    assert all(job.split == "validation" for job in schedule)


@pytest.mark.parametrize(
    ("split", "episode_count"),
    (("validation", 64), ("test", 16), ("unseen", 16), ("train", 16)),
)
def test_standard_eval_schedule_rejects_noncanonical_split_counts(
    split: str,
    episode_count: int,
) -> None:
    from lunar_exploration_ppo.eval import standard

    with pytest.raises(standard.StandardEvaluationError, match="schedule"):
        standard.build_standard_evaluation_schedule(
            _catalog(),
            split=split,
            episode_count=episode_count,
            evaluation_seed_start=20260716,
        )


def test_canonical_builder_still_rejects_24_episodes() -> None:
    from lunar_exploration_ppo.eval import standard

    for split in ("validation", "test", "unseen"):
        with pytest.raises(
            standard.StandardEvaluationError,
            match="schedule is not canonical",
        ):
            standard.build_standard_evaluation_schedule(
                _catalog(),
                split=split,
                episode_count=24,
                evaluation_seed_start=20260726,
            )


def _explicit_test_24_jobs(standard):
    records = tuple(
        record for record in _catalog().records if record.split == "test"
    )[:24]
    return tuple(
        standard.StandardEvaluationJob(
            split="test",
            episode_index=index,
            scenario_id=record.scenario_id,
            scenario_seed=int(record.scenario_seed_hex, 16),
            terrain_seed=int(record.terrain_seed_hex, 16),
            start_pose_seed=int(record.start_pose_seed_hex, 16),
            evaluation_seed=2026072600 + index,
            theta_source="policy_theta_mu/v1",
        )
        for index, record in enumerate(records)
    )


def test_explicit_job_runner_reuses_standard_env_action_and_safety_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import standard

    jobs = _explicit_test_24_jobs(standard)
    safety_contract, config_sha256 = _safety_lineage()
    captured: dict[str, object] = {}

    class VectorToken:
        def __init__(self, specs, *, timeout_seconds: float) -> None:
            captured["specs"] = tuple(specs)
            captured["timeout_seconds"] = timeout_seconds
            self.closed = False

        def close(self) -> None:
            self.closed = True

    summary = EvaluationSummary(
        method="ppo_policy",
        scale_profile="Standard v1",
        episodes=(),
        metrics={},
        bootstrap_audit={},
        fairness_audit={},
    )
    execution_token = object()

    def run_driver(**kwargs):
        captured["driver"] = kwargs
        return execution_token

    monkeypatch.setattr(standard, "SpawnVectorEnv", VectorToken)
    monkeypatch.setattr(standard, "_run_parallel_episode_batch", run_driver)

    result = standard.run_standard_evaluation_jobs(
        catalog=_catalog(),
        jobs=jobs,
        method="ppo_policy",
        policy=_FastMainProcessPolicy(),
        policy_device="cpu",
        bootstrap_resamples=32,
        bootstrap_seed=17,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )

    assert result is execution_token
    specs = captured["specs"]
    assert len(specs) == 8
    assert all(spec.factory.__name__ == "build_standard_evaluation_env" for spec in specs)
    assert all(dict(spec.kwargs)["production"] is True for spec in specs)
    assert all(len(dict(spec.kwargs)["scenario_ids"]) == 3 for spec in specs)
    driver = captured["driver"]
    assert driver["trace_path"] is None
    assert driver["jobs"] == jobs
    assert driver["max_steps"] == 128
    assert driver["success_threshold"] == 0.99
    contract = driver["shared_environment_contract"]
    assert contract["environment_contract"]["settings"]["max_steps"] == 128
    assert contract["environment_contract"]["safety_contract"] == (
        safety_contract.to_dict()
    )
    assert summary.scale_profile == "Standard v1"


def test_existing_validation16_test64_unseen64_contracts_are_unchanged() -> None:
    from lunar_exploration_ppo.eval import standard

    expected = (("validation", 16, 2), ("test", 64, 8), ("unseen", 64, 8))
    for split, count, jobs_per_lane in expected:
        jobs = standard.build_standard_evaluation_schedule(
            _catalog(),
            split=split,
            episode_count=count,
            evaluation_seed_start=2026072600,
        )
        lanes = standard.partition_standard_evaluation_jobs(jobs)
        assert len(jobs) == count
        assert len(lanes) == 8
        assert tuple(len(lane) for lane in lanes) == (jobs_per_lane,) * 8
        assert tuple(
            sorted(
                (job for lane in lanes for job in lane),
                key=lambda job: job.episode_index,
            )
        ) == jobs


def test_all_final_methods_share_scenarios_seeds_and_bind_theta_source() -> None:
    from lunar_exploration_ppo.eval import standard

    schedules = standard.build_standard_final_method_schedules(
        _catalog(),
        split="test",
        evaluation_seed_start=20260716,
    )

    assert tuple(schedules) == standard.STANDARD_EVALUATION_METHODS
    reference = tuple(
        (job.scenario_id, job.evaluation_seed)
        for job in schedules["ppo_policy"]
    )
    assert len(reference) == 64
    assert all(
        tuple((job.scenario_id, job.evaluation_seed) for job in jobs) == reference
        for jobs in schedules.values()
    )
    assert all(
        job.theta_source == (
            "policy_theta_mu/v1"
            if method == "ppo_policy"
            else "candidate_recommended_theta/v1"
        )
        for method, jobs in schedules.items()
        for job in jobs
    )


def test_batched_action_selection_keeps_ppo_inference_and_baseline_theta_contracts() -> None:
    from lunar_exploration_ppo.eval import standard

    observations = (_observation((0.2, 1.1)), _observation((-1.2, 0.4)))
    ppo_actions = standard.select_standard_evaluation_actions(
        method="ppo_policy",
        observations=observations,
        evaluation_seeds=(11, 12),
        policy_output=_policy_output(),
    )
    baseline_actions = standard.select_standard_evaluation_actions(
        method="nearest_frontier",
        observations=observations,
        evaluation_seeds=(11, 12),
        policy_output=None,
    )

    assert [(action.candidate_index, action.target_theta) for action in ppo_actions] == [
        (1, pytest.approx(0.7)),
        (0, pytest.approx(-0.4)),
    ]
    assert [action.candidate_index for action in baseline_actions] == [0, 0]
    assert [action.target_theta for action in baseline_actions] == [
        pytest.approx(0.2),
        pytest.approx(-1.2),
    ]


def test_standard_evaluation_production_surface_has_no_worker_adapter() -> None:
    from lunar_exploration_ppo.eval import standard

    parameters = inspect.signature(standard.run_standard_evaluation).parameters
    assert tuple(parameters) == (
        "catalog",
        "split",
        "method",
        "episode_count",
        "evaluation_seed_start",
        "policy",
        "policy_device",
        "bootstrap_resamples",
        "bootstrap_seed",
        "trace_path",
        "safety_contract",
        "config_sha256",
        "resource_guard",
    )
    source = inspect.getsource(standard.run_standard_evaluation).lower()
    for forbidden in ("fixture", "fake", "adapter", "specs", "worker_factory"):
        assert forbidden not in source


def test_standard_evaluation_env_specs_bind_fixed_round_robin_lanes() -> None:
    from lunar_exploration_ppo.eval import standard

    schedule = standard.build_standard_evaluation_schedule(
        _catalog(),
        split="validation",
        episode_count=16,
        evaluation_seed_start=20260716,
    )
    safety_contract, config_sha256 = _safety_lineage()
    specs = standard.standard_evaluation_env_specs(
        _catalog(),
        schedule,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )

    assert len(specs) == 8
    for worker_index, spec in enumerate(specs):
        assert spec.factory.__name__ == "build_standard_evaluation_env"
        assert dict(spec.kwargs)["production"] is True
        assert dict(spec.kwargs)["split"] == "validation"
        assert tuple(dict(spec.kwargs)["scenario_ids"]) == tuple(
            job.scenario_id for job in schedule[worker_index::8]
        )


def test_public_standard_evaluation_wires_only_production_schedule_and_driver(
    monkeypatch,
    tmp_path,
) -> None:
    from lunar_exploration_ppo.eval import standard

    captured: dict[str, object] = {}
    specs_token = object()
    contract_token = object()
    summary_token = object()

    class VectorToken:
        closed = False

        def close(self) -> None:
            self.closed = True

    vector_token = VectorToken()

    safety_contract, config_sha256 = _safety_lineage()

    def resource_guard(boundary: str) -> None:
        assert boundary

    def build_specs(catalog, jobs, *, safety_contract, config_sha256):
        assert safety_contract == _safety_lineage()[0]
        assert config_sha256 == _safety_lineage()[1]
        captured["jobs"] = tuple(jobs)
        return specs_token

    def build_vector(specs, *, timeout_seconds):
        assert specs is specs_token
        captured["timeout_seconds"] = timeout_seconds
        return vector_token

    def build_contract(**kwargs):
        assert kwargs["catalog"] is _catalog()
        assert kwargs["env_specs"] is specs_token
        assert kwargs["jobs"] == captured["jobs"]
        assert kwargs["max_steps"] == 128
        assert kwargs["success_threshold"] == 0.99
        assert kwargs["safety_contract"] == safety_contract
        assert kwargs["config_sha256"] == config_sha256
        captured["contract"] = contract_token
        return contract_token

    def run_driver(**kwargs):
        assert kwargs["vector_env"] is vector_token
        captured["driver"] = kwargs
        return summary_token

    monkeypatch.setattr(standard, "standard_evaluation_env_specs", build_specs)
    monkeypatch.setattr(standard, "build_standard_environment_contract", build_contract)
    monkeypatch.setattr(standard, "SpawnVectorEnv", build_vector)
    monkeypatch.setattr(standard, "_run_parallel_episode_batch", run_driver)

    result = standard.run_standard_evaluation(
        catalog=_catalog(),
        split="validation",
        method="ppo_policy",
        episode_count=16,
        evaluation_seed_start=20260716,
        policy=_FastMainProcessPolicy(),
        policy_device="cpu",
        bootstrap_resamples=32,
        bootstrap_seed=17,
        trace_path=tmp_path / "episodes.jsonl",
        safety_contract=safety_contract,
        config_sha256=config_sha256,
        resource_guard=resource_guard,
    )

    assert result is summary_token
    assert len(captured["jobs"]) == 16
    assert captured["driver"]["jobs"] == captured["jobs"]
    assert captured["driver"]["max_steps"] == 128
    assert captured["driver"]["success_threshold"] == 0.99
    assert captured["driver"]["shared_environment_contract"] is contract_token
    assert captured["driver"]["resource_guard"] is resource_guard
    assert vector_token.closed is True


def test_standard_evaluation_owner_closes_real_vector_when_driver_precheck_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.eval import standard

    vectors: list[SpawnVectorEnv] = []

    class FailingCloseVector(SpawnVectorEnv):
        def __init__(self, _specs, *, timeout_seconds: float) -> None:
            super().__init__(_eval_specs(), timeout_seconds=min(timeout_seconds, 30.0))
            self.close_calls = 0
            vectors.append(self)

        def close(self) -> None:
            self.close_calls += 1
            super().close()
            raise RuntimeError("secondary vector cleanup failure")

    monkeypatch.setattr(standard, "SpawnVectorEnv", FailingCloseVector)
    safety_contract, config_sha256 = _safety_lineage()

    try:
        with pytest.raises(
            standard.StandardEvaluationError,
            match="policy device drifted",
        ):
            standard.run_standard_evaluation(
                catalog=_catalog(),
                split="validation",
                method="ppo_policy",
                episode_count=16,
                evaluation_seed_start=20260716,
                policy=_FastMainProcessPolicy(),
                policy_device="cuda:0",
                bootstrap_resamples=32,
                bootstrap_seed=17,
                trace_path=tmp_path / "must-not-exist.jsonl",
                safety_contract=safety_contract,
                config_sha256=config_sha256,
            )

        assert len(vectors) == 1
        assert vectors[0].close_calls == 1
        assert vectors[0].closed is True
        assert not any(vectors[0].worker_alive)
        assert not (tmp_path / "must-not-exist.jsonl").exists()
    finally:
        for vector in vectors:
            SpawnVectorEnv.close(vector)


def test_policy_device_equivalence_resolves_bare_cuda_to_current_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import standard

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    assert standard._policy_devices_equivalent("cuda", torch.device("cuda:0"))
    assert standard._policy_devices_equivalent("cuda:0", torch.device("cuda:0"))
    assert not standard._policy_devices_equivalent(
        "cuda:1", torch.device("cuda:0")
    )


def test_policy_device_equivalence_keeps_cpu_comparison_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import standard

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    assert standard._policy_devices_equivalent("cpu", torch.device("cpu"))
    assert not standard._policy_devices_equivalent(
        "cpu", torch.device("cuda:0")
    )
    assert not standard._policy_devices_equivalent(
        "cuda", torch.device("cpu")
    )


def test_parallel_episode_driver_uses_eight_spawn_workers_and_parent_inference(
    tmp_path,
) -> None:
    from lunar_exploration_ppo.eval import standard

    jobs = standard.build_standard_evaluation_schedule(
        _catalog(),
        split="validation",
        episode_count=16,
        evaluation_seed_start=20260716,
    )
    policy = _FastMainProcessPolicy()
    policy.train()
    before = policy_state_sha256(policy)
    vector = SpawnVectorEnv(_eval_specs(), timeout_seconds=30.0)
    boundaries: list[str] = []

    try:
        summary = standard._run_parallel_episode_batch(
            vector_env=vector,
            jobs=jobs,
            method="ppo_policy",
            policy=policy,
            policy_device="cpu",
            max_steps=128,
            success_threshold=0.99,
            bootstrap_resamples=32,
            bootstrap_seed=17,
            trace_path=tmp_path / "episodes.jsonl",
            shared_environment_contract=_shared_environment_contract(standard, jobs),
            resource_guard=boundaries.append,
            **_safety_validation_kwargs(),
        )
    finally:
        vector.close()

    assert isinstance(summary, EvaluationSummary)
    assert len(summary.episodes) == 16
    assert summary.metrics["episode_count"] == 16
    assert summary.fairness_audit["worker_count"] == 8
    assert summary.fairness_audit["worker_start_methods"] == ["spawn"] * 8
    assert summary.fairness_audit["inference_pids"] == [os.getpid()]
    assert summary.fairness_audit["policy_state_unchanged"] is True
    assert summary.fairness_audit["policy_state_sha256_before"] == before
    assert summary.fairness_audit["policy_state_sha256_after"] == before
    assert summary.fairness_audit["passed"] is True
    audit = summary.fairness_audit
    assert {
        "shared_environment_contract_identical",
        "method_specific_action_rule_only_difference",
        "hidden_truth_policy_input_count",
    }.isdisjoint(audit)
    assert audit["schema_version"] == "stage6_standard_parallel_fairness_audit/v2"
    assert audit["shared_environment_contract_sha256"] == _canonical_sha256(
        audit["shared_environment_contract"]
    )
    assert audit["decision_input_schema_sha256"] == _canonical_sha256(
        audit["decision_input_schema"]
    )
    assert audit["action_rule_sha256"] == _canonical_sha256(
        audit["action_rule"]
    )
    assert audit["decision_input_schema"]["frozen_field_scan"][
        "truth_like_fields"
    ] == []
    decision_audit = audit["decision_audit"]
    assert decision_audit["decision_count"] == audit["selected_action_count"]
    assert decision_audit["policy_observation_input_count"] == audit[
        "selected_action_count"
    ]
    assert decision_audit["candidate_mask_valid_count"] == audit[
        "selected_action_count"
    ]
    assert decision_audit["selected_index_valid_count"] == audit[
        "selected_action_count"
    ]
    assert decision_audit["ppo_observation_batch_policy_output_count"] == audit[
        "selected_action_count"
    ]
    assert decision_audit["runtime_truth_like_selector_field_count"] == 0
    assert decision_audit["decisions"]
    assert {
        tuple(source["name"] for source in record["selector_inputs"])
        for record in decision_audit["decisions"]
    } == {("observation_batch", "policy_output")}
    assert standard.validate_standard_fairness_audit(
        audit,
        episode_count=16,
        parent_pid=os.getpid(),
        **_safety_validation_kwargs(),
    ) == audit
    assert policy_state_sha256(policy) == before
    assert policy.training is True
    assert vector.closed is True
    assert all(code == 0 for code in vector.worker_exitcodes)
    assert boundaries.count("evaluation:before-episode-reset") == 8
    assert boundaries.count("evaluation:after-episode-reset") == 8
    records = [
        json.loads(line)
        for line in (tmp_path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 16
    assert [record["scenario_key"] for record in records] == [
        job.scenario_id for job in jobs
    ]
    assert all(
        record["reset_diagnostics"]["scan_order"]
        == ["reset_local_safety", "reset_exploration"]
        for record in records
    )
    assert all(
        record["planner_failure_counts"]
        == {
            "endpoint_physical_unsafe": 0,
            "endpoint_unknown_buffer_unsafe": 0,
            "path_physical_unsafe": 0,
            "path_unknown_buffer_unsafe": 0,
            "planner_no_path": 0,
        }
        for record in records
    )


def test_trace_bundle_contains_episode_decision_and_step_rows_without_writing(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.eval import standard

    jobs = _explicit_test_24_jobs(standard)
    policy = _FastMainProcessPolicy()
    vector = SpawnVectorEnv(_eval_specs(), timeout_seconds=30.0)
    before = tuple(tmp_path.iterdir())
    try:
        execution = standard._run_parallel_episode_batch(
            vector_env=vector,
            jobs=jobs,
            method="ppo_policy",
            policy=policy,
            policy_device="cpu",
            max_steps=128,
            success_threshold=0.99,
            bootstrap_resamples=32,
            bootstrap_seed=17,
            trace_path=None,
            shared_environment_contract=_shared_environment_contract(standard, jobs),
            **_safety_validation_kwargs(),
        )
    finally:
        vector.close()

    assert isinstance(execution, standard.StandardEvaluationExecution)
    assert len(execution.episode_rows) == 24
    assert len(execution.decision_rows) == len(execution.step_rows) > 0
    assert tuple(tmp_path.iterdir()) == before
    assert [row["episode_index"] for row in execution.episode_rows] == list(range(24))
    assert {row["lane_id"] for row in execution.episode_rows} == {
        f"lane-{index}" for index in range(8)
    }
    assert all(
        len(
            [
                row
                for row in execution.episode_rows
                if row["lane_id"] == f"lane-{index}"
            ]
        )
        == 3
        for index in range(8)
    )
    decision_by_join_key = {
        row["join_key"]: row for row in execution.decision_rows
    }
    assert len(decision_by_join_key) == len(execution.decision_rows)
    for row in execution.step_rows:
        assert row["join_key"] in decision_by_join_key
        assert row["decision_sha256"] == decision_by_join_key[row["join_key"]][
            "decision_sha256"
        ]
        assert len(row["pre_observation_sha256"]) == 64
        assert len(row["post_observation_sha256"]) == 64
        assert type(row["selected_candidate_index"]) is int
        assert len(row["selected_candidate_cell_xy"]) == 2
        assert isinstance(row["selected_theta"], float)
        assert isinstance(row["planned_path_cells"], list)
        assert row["cumulative_path_length_m"] >= row["path_length_m"] >= 0.0
        assert type(row["coverage_gain_cells"]) is int
        assert 0.0 <= row["coverage_rate"] <= 1.0
        assert type(row["done"]) is bool
        assert isinstance(row["termination_reason"], str)
        assert isinstance(row["planner_diagnostics"], dict)
        assert not {
            "truth",
            "coverable_mask",
            "hidden_highres",
        }.intersection(row)


def test_parallel_episode_driver_rejects_unknown_planner_failure_reason(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.eval import standard

    jobs = standard.build_standard_evaluation_schedule(
        _catalog(),
        split="validation",
        episode_count=16,
        evaluation_seed_start=20260716,
    )
    vector = SpawnVectorEnv(
        _eval_specs(failure_worker=0, fail_mode="unknown_reason"),
        timeout_seconds=30.0,
    )
    try:
        with pytest.raises(
            standard.StandardEvaluationError,
            match="planner failure reason",
        ):
            standard._run_parallel_episode_batch(
                vector_env=vector,
                jobs=jobs,
                method="ppo_policy",
                policy=_FastMainProcessPolicy(),
                policy_device="cpu",
                max_steps=128,
                success_threshold=0.99,
                bootstrap_resamples=32,
                bootstrap_seed=17,
                trace_path=tmp_path / "unknown-reason.jsonl",
                shared_environment_contract=_shared_environment_contract(
                    standard,
                    jobs,
                ),
                **_safety_validation_kwargs(),
            )
    finally:
        vector.close()


def test_final_eval_trace_recovers_durable_pending_without_legacy_append(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.eval import standard
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl

    class InjectedTraceInterruption(RuntimeError):
        pass

    trace_path = tmp_path / "interrupted-episodes.jsonl"
    jobs = standard.build_standard_evaluation_schedule(
        _catalog(),
        split="validation",
        episode_count=16,
        evaluation_seed_start=20260716,
    )
    policy = _FastMainProcessPolicy()
    vector = SpawnVectorEnv(_eval_specs(), timeout_seconds=30.0)
    durable_paths: list[Path] = []
    legacy_append_calls: list[Path] = []

    def interrupt_after_pending(phase: str) -> None:
        if phase == "after_pending":
            raise InjectedTraceInterruption("injected interruption after pending")

    def build_interrupted_writer(path: str | Path) -> DurableJsonl:
        durable_paths.append(Path(path))
        return DurableJsonl(path, fault_injector=interrupt_after_pending)

    def reject_legacy_append(
        _store: ArtifactStore,
        relative_path: str | Path,
        _value: object,
    ) -> Path:
        legacy_append_calls.append(Path(relative_path))
        raise AssertionError("final-eval trace used legacy append_jsonl")

    monkeypatch.setattr(
        standard,
        "DurableJsonl",
        build_interrupted_writer,
        raising=False,
    )
    monkeypatch.setattr(ArtifactStore, "append_jsonl", reject_legacy_append)

    try:
        with pytest.raises(
            InjectedTraceInterruption,
            match="injected interruption after pending",
        ):
            standard._run_parallel_episode_batch(
                vector_env=vector,
                jobs=jobs,
                method="ppo_policy",
                policy=policy,
                policy_device="cpu",
                max_steps=128,
                success_threshold=0.99,
                bootstrap_resamples=32,
                bootstrap_seed=17,
                trace_path=trace_path,
                shared_environment_contract=_shared_environment_contract(
                    standard,
                    jobs,
                ),
                **_safety_validation_kwargs(),
            )
    finally:
        vector.close()

    pending_path = Path(f"{trace_path}.pending")
    assert legacy_append_calls == []
    assert durable_paths == [trace_path]
    assert not trace_path.exists()
    assert pending_path.is_file()

    assert DurableJsonl(trace_path).recover() is True
    assert trace_path.is_file()
    assert not pending_path.exists()
    recovered_bytes = trace_path.read_bytes()
    recovered = json.loads(recovered_bytes)
    assert recovered_bytes == (
        json.dumps(
            recovered,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    assert recovered["scenario_key"] == jobs[0].scenario_id
    assert recovered["evaluation_seed"] == jobs[0].evaluation_seed


def test_parallel_evaluation_resource_guard_aborts_before_worker_step(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.eval import standard

    jobs = standard.build_standard_evaluation_schedule(
        _catalog(),
        split="validation",
        episode_count=16,
        evaluation_seed_start=20260716,
    )
    policy = _FastMainProcessPolicy()
    vector = SpawnVectorEnv(_eval_specs(), timeout_seconds=30.0)
    boundaries: list[str] = []

    def resource_guard(boundary: str) -> None:
        boundaries.append(boundary)
        if boundary == "evaluation:before-step":
            raise RuntimeError("latched evaluation hard stop")

    try:
        with pytest.raises(RuntimeError, match="latched evaluation hard stop"):
            standard._run_parallel_episode_batch(
                vector_env=vector,
                jobs=jobs,
                method="ppo_policy",
                policy=policy,
                policy_device="cpu",
                max_steps=128,
                success_threshold=0.99,
                bootstrap_resamples=32,
                bootstrap_seed=17,
                trace_path=tmp_path / "aborted-episodes.jsonl",
                shared_environment_contract=_shared_environment_contract(standard, jobs),
                resource_guard=resource_guard,
                **_safety_validation_kwargs(),
            )
    finally:
        vector.close()

    assert boundaries[-1] == "evaluation:before-step"
    assert "evaluation:after-inference" in boundaries
    assert vector.closed is True


def _exercise_rollover_hard_stop(
    *,
    standard,
    tmp_path: Path,
    stop_boundary: str,
    primary_message: str,
):
    jobs = standard.build_standard_evaluation_schedule(
        _catalog(),
        split="validation",
        episode_count=16,
        evaluation_seed_start=20260716,
    )
    policy = _CountingMainProcessPolicy()
    vector = _RolloverAuditVector(inject_cleanup_failure=True)
    boundaries: list[str] = []
    caught: BaseException | None = None

    def resource_guard(boundary: str) -> None:
        boundaries.append(boundary)
        if boundary == stop_boundary:
            raise RuntimeError(primary_message)

    try:
        try:
            standard._run_parallel_episode_batch(
                vector_env=vector,
                jobs=jobs,
                method="ppo_policy",
                policy=policy,
                policy_device="cpu",
                max_steps=128,
                success_threshold=0.99,
                bootstrap_resamples=32,
                bootstrap_seed=17,
                trace_path=tmp_path / f"{stop_boundary.replace(':', '-')}.jsonl",
                shared_environment_contract=_shared_environment_contract(standard, jobs),
                resource_guard=resource_guard,
                **_safety_validation_kwargs(),
            )
        except BaseException as exc:
            caught = exc
    finally:
        SpawnVectorEnv.close(vector)

    return caught, vector, policy, boundaries


def test_parallel_evaluation_rollover_before_reset_hard_stop_sends_no_reset_and_preserves_primary(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.eval import standard

    caught, vector, policy, boundaries = _exercise_rollover_hard_stop(
        standard=standard,
        tmp_path=tmp_path,
        stop_boundary="evaluation:before-episode-reset",
        primary_message="rollover before-reset hard stop",
    )

    assert type(caught) is RuntimeError
    assert str(caught) == "rollover before-reset hard stop"
    assert boundaries[-1] == "evaluation:before-episode-reset"
    assert vector.reset_requests == [None]
    assert vector.step_calls == 4
    assert policy.forward_calls == 4
    assert vector.closed is True
    assert not any(vector.worker_alive)
    assert any(
        "secondary vector cleanup failure" in note
        for note in getattr(caught, "__notes__", ())
    )


def test_parallel_evaluation_rollover_after_reset_hard_stop_blocks_inference_and_preserves_primary(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.eval import standard

    caught, vector, policy, boundaries = _exercise_rollover_hard_stop(
        standard=standard,
        tmp_path=tmp_path,
        stop_boundary="evaluation:after-episode-reset",
        primary_message="rollover after-reset hard stop",
    )

    assert type(caught) is RuntimeError
    assert str(caught) == "rollover after-reset hard stop"
    assert boundaries[-1] == "evaluation:after-episode-reset"
    assert vector.reset_requests == [None, (0,)]
    assert vector.step_calls == 4
    assert policy.forward_calls == 4
    assert vector.closed is True
    assert not any(vector.worker_alive)
    assert any(
        "secondary vector cleanup failure" in note
        for note in getattr(caught, "__notes__", ())
    )


@pytest.mark.parametrize(
    "field",
    (
        "shared_environment_contract_sha256",
        "decision_input_schema_sha256",
        "action_rule_sha256",
    ),
)
def test_fairness_validator_rejects_recomputed_provenance_digest_tampering(
    field: str,
) -> None:
    from lunar_exploration_ppo.eval import standard

    audit = _valid_fairness_audit(standard, method="nearest_frontier")
    tampered = copy.deepcopy(audit)
    tampered[field] = "0" * 64

    with pytest.raises(standard.StandardEvaluationError, match="fairness"):
        standard.validate_standard_fairness_audit(
            tampered,
            episode_count=16,
            parent_pid=12_345,
            **_safety_validation_kwargs(),
        )


def test_fairness_validator_rejects_policy_state_hash_self_claim() -> None:
    from lunar_exploration_ppo.eval import standard

    audit = _valid_fairness_audit(standard, method="ppo_policy")
    tampered = copy.deepcopy(audit)
    tampered["policy_state_sha256_after"] = "f" * 64

    with pytest.raises(standard.StandardEvaluationError, match="fairness"):
        standard.validate_standard_fairness_audit(
            tampered,
            episode_count=16,
            parent_pid=12_345,
            **_safety_validation_kwargs(),
        )


def test_fairness_validator_rejects_decision_count_tampering() -> None:
    from lunar_exploration_ppo.eval import standard

    audit = _valid_fairness_audit(standard, method="nearest_frontier")
    tampered = copy.deepcopy(audit)
    tampered["decision_audit"]["decision_count"] += 1

    with pytest.raises(standard.StandardEvaluationError, match="fairness"):
        standard.validate_standard_fairness_audit(
            tampered,
            episode_count=16,
            parent_pid=12_345,
            **_safety_validation_kwargs(),
        )


def test_fairness_validator_rejects_episode_index_outside_requested_cohort() -> None:
    from lunar_exploration_ppo.eval import standard

    audit = _valid_fairness_audit(standard, method="nearest_frontier")
    tampered = copy.deepcopy(audit)
    record = tampered["decision_audit"]["decisions"][0]
    record["episode_index"] = 63
    record["worker_index"] = 7
    record["decision_sha256"] = _canonical_sha256(
        {key: value for key, value in record.items() if key != "decision_sha256"}
    )

    with pytest.raises(standard.StandardEvaluationError, match="fairness"):
        standard.validate_standard_fairness_audit(
            tampered,
            episode_count=16,
            parent_pid=12_345,
            **_safety_validation_kwargs(),
        )


def test_baseline_theta_deviation_is_rejected_after_record_is_redigested() -> None:
    from lunar_exploration_ppo.eval import standard

    audit = _valid_fairness_audit(standard, method="nearest_frontier")
    tampered = copy.deepcopy(audit)
    record = tampered["decision_audit"]["decisions"][0]
    record["target_theta"] += 1.0e-4
    record["theta_circular_error_rad"] = abs(
        math.atan2(
            math.sin(record["target_theta"] - record["recommended_theta"]),
            math.cos(record["target_theta"] - record["recommended_theta"]),
        )
    )
    record["decision_sha256"] = _canonical_sha256(
        {key: value for key, value in record.items() if key != "decision_sha256"}
    )

    with pytest.raises(standard.StandardEvaluationError, match="provenance"):
        standard.validate_standard_fairness_audit(
            tampered,
            episode_count=16,
            parent_pid=12_345,
            **_safety_validation_kwargs(),
        )


def test_baseline_theta_source_is_recomputed_from_selected_candidate_features() -> None:
    from lunar_exploration_ppo.eval import standard

    audit = _valid_fairness_audit(standard, method="nearest_frontier")
    record = audit["decision_audit"]["decisions"][0]
    source = record["recommended_theta_source"]
    assert source["frontier_feature_indices"] == [14, 15]
    assert record["recommended_theta"] == pytest.approx(
        math.atan2(source["sin"], source["cos"]),
        abs=1.0e-12,
    )

    tampered = copy.deepcopy(audit)
    tampered_record = tampered["decision_audit"]["decisions"][0]
    tampered_record["recommended_theta_source"]["sin"] *= -1.0
    tampered_record["decision_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in tampered_record.items()
            if key != "decision_sha256"
        }
    )
    with pytest.raises(standard.StandardEvaluationError, match="provenance"):
        standard.validate_standard_fairness_audit(
            tampered,
            episode_count=16,
            parent_pid=12_345,
            **_safety_validation_kwargs(),
        )


def test_truth_like_selector_field_is_detected_and_rejected() -> None:
    from lunar_exploration_ppo.eval import standard

    observation = _observation((0.2, 1.1))
    action = standard.select_standard_evaluation_actions(
        method="nearest_frontier",
        observations=(observation,),
        evaluation_seeds=(11,),
        policy_output=None,
    )[0]

    with pytest.raises(standard.StandardEvaluationError, match="truth-like"):
        standard.build_standard_decision_record(
            method="nearest_frontier",
            sequence_index=0,
            episode_index=0,
            step_index=0,
            worker_index=0,
            observation=observation,
            action=action,
            selector_inputs={
                "policy_observation": observation,
                "coverable_mask": np.ones((2, 2), dtype=bool),
            },
        )


def test_five_method_cohort_accepts_only_shared_contract_invariants() -> None:
    from lunar_exploration_ppo.eval import standard

    audits = [
        _valid_fairness_audit(standard, method=method)
        for method in standard.STANDARD_EVALUATION_METHODS
    ]

    cohort = standard.validate_standard_fairness_cohort(
        audits,
        split="validation",
        episode_count=16,
        **_safety_validation_kwargs(),
    )

    assert cohort["schema_version"] == "stage6_standard_fairness_cohort/v1"
    assert cohort["methods"] == list(standard.STANDARD_EVALUATION_METHODS)
    assert cohort["shared_environment_contract_sha256"] == audits[0][
        "shared_environment_contract_sha256"
    ]
    assert cohort["decision_input_schema_sha256"] == audits[0][
        "decision_input_schema_sha256"
    ]
    assert len(set(cohort["action_rule_sha256_by_method"].values())) == 5
    assert cohort["passed"] is True


def test_five_method_cohort_rejects_cross_method_schedule_mismatch() -> None:
    from lunar_exploration_ppo.eval import standard

    audits = [
        _valid_fairness_audit(standard, method=method)
        for method in standard.STANDARD_EVALUATION_METHODS
    ]
    tampered = copy.deepcopy(audits)
    changed = tampered[-1]
    contract = changed["shared_environment_contract"]
    replacement = "validation/tampered-scenario"
    contract["scenario_seed_schedule"][0]["scenario_id"] = replacement
    contract["environment_specs"][0]["scenario_ids"][0] = replacement
    changed["scenario_schedule"][0] = replacement
    changed["shared_environment_contract_sha256"] = _canonical_sha256(contract)

    assert standard.validate_standard_fairness_audit(
        changed,
        episode_count=16,
        parent_pid=None,
        **_safety_validation_kwargs(),
    ) == changed
    with pytest.raises(standard.StandardEvaluationError, match="cohort"):
        standard.validate_standard_fairness_cohort(
            tampered,
            split="validation",
            episode_count=16,
            **_safety_validation_kwargs(),
        )


def test_fairness_accepts_zero_step_episode_and_replays_recorded_parent_pid() -> None:
    from lunar_exploration_ppo.eval import standard

    audit = _valid_fairness_audit(
        standard,
        method="ppo_policy",
        zero_step_first_episode=True,
    )

    assert audit["episode_action_counts"][0] == 0
    assert standard.validate_standard_fairness_audit(
        audit,
        episode_count=16,
        parent_pid=None,
        **_safety_validation_kwargs(),
    ) == audit
    with pytest.raises(standard.StandardEvaluationError, match="fairness"):
        tampered = copy.deepcopy(audit)
        tampered["selected_action_count"] += 1
        standard.validate_standard_fairness_audit(
            tampered,
            episode_count=16,
            parent_pid=12_345,
            **_safety_validation_kwargs(),
        )
