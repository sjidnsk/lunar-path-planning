"""缩减规模 G1 显式 24 场景适配层合同。"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import inspect
import json
import os
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lunar_exploration_ppo.configs.stage6 import SafetyContract, load_stage6_config
from lunar_exploration_ppo.env.env import EnvAction
from lunar_exploration_ppo.env.standard_training import build_standard_catalog
from lunar_exploration_ppo.eval.evaluator import EvaluationSummary
from lunar_exploration_ppo.eval.metrics import (
    ZERO_DISTANCE_POLICY,
    build_episode_result,
    episode_record,
    summarize_episodes,
)
from lunar_exploration_ppo.policy.cross_attention import (
    PolicyForwardOutput,
    batch_policy_observations,
)
from lunar_exploration_ppo.policy.observation import PolicyObservation
from lunar_exploration_ppo.ppo.collector import PLANNER_FAILURE_REASONS
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ppo_highres_frontier_stage6_v1.json"
G1_CONFIG = ROOT / "configs/xunce_mid_dual_g1_coverage_v1.json"
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _g1_runner():
    try:
        return importlib.import_module("run_xunce_mid_dual_g1_coverage")
    except ModuleNotFoundError:
        pytest.fail("Task 5 G1 runner is not implemented")


@lru_cache(maxsize=1)
def _catalog():
    return build_standard_catalog(verify_hashes=False)


def _cohorts() -> dict[str, list[str]]:
    by_split = {
        split: [
            f"{record.scenario_id}/standard-proxy/v1"
            for record in _catalog().records
            if record.split == split
        ]
        for split in ("validation", "test", "unseen")
    }
    return {
        "test_q24": by_split["test"][:24],
        "test_c24": by_split["test"][24:48],
        "unseen24": by_split["unseen"][:24],
        "g3_test_q5": by_split["test"][:5],
        "g3_unseen5": by_split["unseen"][:5],
        "validation3": by_split["validation"][:3],
        "replay3": by_split["test"][:3],
    }


def _manifest_payload() -> dict[str, object]:
    cohorts = _cohorts()
    selected_ids = sorted(
        set(cohorts["test_q24"])
        | set(cohorts["test_c24"])
        | set(cohorts["unseen24"])
        | set(cohorts["validation3"])
    )
    hash_value = "a" * 64
    proofs = [
        {
            "scenario_id": scenario_id,
            "scenario_hash": hashlib.sha256(scenario_id.encode("utf-8")).hexdigest(),
            "split": scenario_id.split("/", 1)[0],
            "coverable_mask_sha256": hashlib.sha256(
                f"mask:{scenario_id}".encode("utf-8")
            ).hexdigest(),
            "coverable_cell_count": 4096,
            "algorithm_id": "exact_reachable_safe_pose_range_los/v1",
            "exact": True,
            "coverage_denominator_source": (
                "reachable_observable_free_highres_cells/v1"
            ),
            "coverage_denominator_algorithm": (
                "exact_reachable_safe_pose_range_los/v1"
            ),
            "semantic_alias_proven": True,
        }
        for scenario_id in selected_ids
    ]
    return {
        "schema_version": "mid-dual-scenario-freeze/v1",
        "completion_status": "complete",
        "config_sha256": hash_value,
        "descriptor_catalog_sha256": hash_value,
        "source_manifest_sha256": hash_value,
        "stage6_coverage_manifest_sha256": hash_value,
        "stage6_catalog_sha256": _catalog().sha256,
        "reconstruction_index_sha256": hash_value,
        "denominator_proofs_sha256": hash_value,
        "descriptor_generator": {"id": "fixture", "version": "v1"},
        "policy_blind_attestation": {"attestation_id": "fixture"},
        "source_pools": {
            "validation": hash_value,
            "test": hash_value,
            "unseen": hash_value,
        },
        "cohort_sizes": {
            "test_q24": 24,
            "test_c24": 24,
            "unseen24": 24,
            "g3_test_q5": 5,
            "g3_unseen5": 5,
            "validation3": 3,
            "replay3": 3,
        },
        "cohorts": cohorts,
        "denominator_proofs": proofs,
        "bundle_files": [
            {
                "path": name,
                "sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
                "size_bytes": len(name),
            }
            for name in (
                "config.json",
                "descriptors.jsonl",
                "source-manifest.json",
                "reconstruction-index.jsonl",
                "denominator-proofs.jsonl",
            )
        ],
    }


def _task3_manifest_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _load_fixture_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from lunar_exploration_ppo.eval import midterm_reduced

    root = tmp_path / "frozen"
    root.mkdir()
    payload = _task3_manifest_bytes(_manifest_payload())
    (root / "manifest.json").write_bytes(payload)
    monkeypatch.setattr(midterm_reduced, "_verify_frozen_bundle", lambda _root: True)
    return midterm_reduced.load_frozen_scenario_manifest(
        bundle_root=root,
        expected_manifest_sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_frozen_manifest_accepts_only_task3_compact_canonical_bytes() -> None:
    from lunar_exploration_ppo.eval import midterm_reduced

    payload = _manifest_payload()
    canonical = _task3_manifest_bytes(payload)

    assert midterm_reduced._strict_manifest_payload(canonical) == payload

    pretty = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    with pytest.raises(
        midterm_reduced.MidtermReducedEvaluationError,
        match="canonical bytes",
    ):
        midterm_reduced._strict_manifest_payload(pretty)

    for noncanonical in (
        canonical + b"\n",
        canonical.replace(b'"completion_status":"complete"', b'"completion_status":"complete" '),
        b'{"schema_version":"mid-dual-scenario-freeze/v1","schema_version":"mid-dual-scenario-freeze/v1"}',
        b'{"schema_version":NaN}',
    ):
        with pytest.raises(midterm_reduced.MidtermReducedEvaluationError):
            midterm_reduced._strict_manifest_payload(noncanonical)


def _install_identity_factory(
    monkeypatch: pytest.MonkeyPatch,
    midterm_reduced,
):
    built: list[str] = []

    class IdentityFactory:
        def __init__(self, catalog) -> None:
            assert catalog is _catalog()

        def build(self, record):
            scenario_id = f"{record.scenario_id}/standard-proxy/v1"
            built.append(scenario_id)
            return SimpleNamespace(
                scenario_id=scenario_id,
                scenario_hash=hashlib.sha256(
                    scenario_id.encode("utf-8")
                ).hexdigest(),
            )

    monkeypatch.setattr(midterm_reduced, "StandardScenarioFactory", IdentityFactory)
    return built


def test_reduced_builder_accepts_only_explicit_frozen_24_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced

    frozen = _load_fixture_manifest(tmp_path, monkeypatch)
    _install_identity_factory(monkeypatch, midterm_reduced)
    jobs = midterm_reduced.build_midterm_reduced_jobs(
        catalog=_catalog(),
        frozen_manifest=frozen,
        cohort="test_q24",
        evaluation_seed_start=2026072600,
    )

    assert len(jobs) == 24
    assert tuple(job.scenario_id for job in jobs) == tuple(
        scenario_id.removesuffix("/standard-proxy/v1")
        for scenario_id in _cohorts()["test_q24"]
    )
    assert tuple(job.episode_index for job in jobs) == tuple(range(24))
    assert len({job.scenario_id for job in jobs}) == 24
    assert "scenario_ids" not in inspect.signature(
        midterm_reduced.build_midterm_reduced_jobs
    ).parameters

    tampered = _manifest_payload()
    tampered["cohorts"]["test_q24"][1] = tampered["cohorts"]["test_q24"][0]
    payload = _task3_manifest_bytes(tampered)
    bad_root = tmp_path / "tampered"
    bad_root.mkdir()
    (bad_root / "manifest.json").write_bytes(payload)
    with pytest.raises(
        midterm_reduced.MidtermReducedEvaluationError,
        match="cohort",
    ):
        midterm_reduced.load_frozen_scenario_manifest(
            bundle_root=bad_root,
            expected_manifest_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_reduced_partition_is_eight_lanes_of_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced, standard

    frozen = _load_fixture_manifest(tmp_path, monkeypatch)
    _install_identity_factory(monkeypatch, midterm_reduced)
    jobs = midterm_reduced.build_midterm_reduced_jobs(
        catalog=_catalog(),
        frozen_manifest=frozen,
        cohort="unseen24",
        evaluation_seed_start=2026072700,
    )
    lanes = standard.partition_standard_evaluation_jobs(jobs)

    assert len(lanes) == 8
    assert tuple(len(lane) for lane in lanes) == (3,) * 8
    assert tuple(
        job.episode_index for lane in lanes for job in lane
    ) == (
        0,
        8,
        16,
        1,
        9,
        17,
        2,
        10,
        18,
        3,
        11,
        19,
        4,
        12,
        20,
        5,
        13,
        21,
        6,
        14,
        22,
        7,
        15,
        23,
    )


def test_reduced_builder_cannot_build_64_then_slice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced, standard

    frozen = _load_fixture_manifest(tmp_path, monkeypatch)
    _install_identity_factory(monkeypatch, midterm_reduced)

    def reject_canonical_builder(*_args, **_kwargs):
        raise AssertionError("reduced schedule attempted canonical build-and-slice")

    monkeypatch.setattr(
        standard,
        "build_standard_evaluation_schedule",
        reject_canonical_builder,
    )
    jobs = midterm_reduced.build_midterm_reduced_jobs(
        catalog=_catalog(),
        frozen_manifest=frozen,
        cohort="test_q24",
        evaluation_seed_start=2026072600,
    )

    assert len(jobs) == 24
    assert tuple(job.scenario_id for job in jobs) == tuple(
        scenario_id.removesuffix("/standard-proxy/v1")
        for scenario_id in _cohorts()["test_q24"]
    )


def test_reduced_builder_reconstructs_full_stage6_identity_and_hash_one_to_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced

    frozen = _load_fixture_manifest(tmp_path, monkeypatch)
    built = _install_identity_factory(monkeypatch, midterm_reduced)
    plan = midterm_reduced.build_midterm_reduced_job_plan(
        catalog=_catalog(),
        frozen_manifest=frozen,
        cohort="test_q24",
        evaluation_seed_start=2026072600,
    )

    assert tuple(built) == tuple(_cohorts()["test_q24"])
    assert tuple(plan.frozen_scenario_id_by_record_id.values()) == tuple(
        _cohorts()["test_q24"]
    )
    assert tuple(plan.record_id_by_frozen_scenario_id.values()) == tuple(
        scenario_id.removesuffix("/standard-proxy/v1")
        for scenario_id in _cohorts()["test_q24"]
    )
    assert all(
        plan.scenario_hash_by_record_id[record_id]
        == hashlib.sha256(frozen_id.encode("utf-8")).hexdigest()
        for record_id, frozen_id in plan.frozen_scenario_id_by_record_id.items()
    )

    first_frozen_id = _cohorts()["test_q24"][0]
    bad_proofs = dict(frozen.denominator_proofs)
    bad_proofs[first_frozen_id] = {
        **dict(bad_proofs[first_frozen_id]),
        "scenario_hash": "f" * 64,
    }
    corrupted = midterm_reduced.FrozenScenarioManifest(
        bundle_root=frozen.bundle_root,
        manifest_sha256=frozen.manifest_sha256,
        stage6_catalog_sha256=frozen.stage6_catalog_sha256,
        cohorts=frozen.cohorts,
        denominator_proofs=bad_proofs,
    )
    with pytest.raises(
        midterm_reduced.MidtermReducedEvaluationError,
        match="scenario identity",
    ):
        midterm_reduced.build_midterm_reduced_job_plan(
            catalog=_catalog(),
            frozen_manifest=corrupted,
            cohort="test_q24",
            evaluation_seed_start=2026072600,
        )


def test_frozen_manifest_hash_and_semantic_verification_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced

    root = tmp_path / "frozen"
    root.mkdir()
    payload = _task3_manifest_bytes(_manifest_payload())
    (root / "manifest.json").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(midterm_reduced, "_verify_frozen_bundle", lambda _root: False)
    with pytest.raises(
        midterm_reduced.MidtermReducedEvaluationError,
        match="verification",
    ):
        midterm_reduced.load_frozen_scenario_manifest(
            bundle_root=root,
            expected_manifest_sha256=digest,
        )

    monkeypatch.setattr(midterm_reduced, "_verify_frozen_bundle", lambda _root: True)
    with pytest.raises(
        midterm_reduced.MidtermReducedEvaluationError,
        match="SHA-256",
    ):
        midterm_reduced.load_frozen_scenario_manifest(
            bundle_root=root,
            expected_manifest_sha256="b" * 64,
        )


def test_frozen_manifest_rejects_unsafe_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced
    from lunar_exploration_ppo.utils.path_security import PathSecurityError

    root = tmp_path / "frozen"
    root.mkdir()
    payload = _task3_manifest_bytes(_manifest_payload())
    (root / "manifest.json").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(midterm_reduced, "_verify_frozen_bundle", lambda _root: True)

    def reject_unsafe_read(*_args: object, **_kwargs: object) -> object:
        raise PathSecurityError("test manifest is a reparse point")

    monkeypatch.setattr(
        midterm_reduced,
        "secure_read_bytes",
        reject_unsafe_read,
        raising=False,
    )
    with pytest.raises(
        midterm_reduced.MidtermReducedEvaluationError,
        match="unsafe",
    ):
        midterm_reduced.load_frozen_scenario_manifest(
            bundle_root=root,
            expected_manifest_sha256=digest,
        )


def test_update80_loader_is_fixed_cuda_fp32_and_rechecks_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced

    assert tuple(
        inspect.signature(midterm_reduced.load_midterm_update80_policy).parameters
    ) == ()
    source = Path(midterm_reduced.__file__).read_text(encoding="utf-8")
    imports = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported_modules = {
        alias.name
        for node in imports
        for alias in (
            node.names
            if isinstance(node, ast.Import)
            else [ast.alias(name=node.module or "")]
        )
    }
    assert "lunar_exploration_ppo.ppo.trainer" not in imported_modules
    assert not any("update" in name for name in imported_modules)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(
        midterm_reduced.MidtermReducedEvaluationError,
        match="CUDA",
    ):
        midterm_reduced.load_midterm_update80_policy()

    captured: dict[str, object] = {}
    policy = nn.Linear(2, 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        midterm_reduced,
        "_sha256_file",
        lambda path: midterm_reduced.UPDATE80_CHECKPOINT_SHA256,
    )
    monkeypatch.setattr(
        midterm_reduced,
        "_policy_state_sha256",
        lambda _policy: midterm_reduced.UPDATE80_POLICY_STATE_SHA256,
    )
    monkeypatch.setattr(
        midterm_reduced,
        "_validate_cuda_fp32_policy",
        lambda _policy: None,
    )

    def load_policy(**kwargs):
        captured.update(kwargs)
        return policy

    monkeypatch.setattr(
        midterm_reduced,
        "load_stage4_policy_for_standard",
        load_policy,
    )
    assert midterm_reduced.load_midterm_update80_policy() is policy
    assert captured == {
        "checkpoint_path": midterm_reduced.UPDATE80_CHECKPOINT_PATH,
        "checkpoint_sha256": (
            "35e04c86f9f973af028fb08f0175d42ab45378d09e1f2b96ee6aad6e4c12b5b5"
        ),
        "policy_state_sha256": (
            "3123e6adde99be41e3bd5cc2f3843068e5416892395926d759c2c6a243ffd381"
        ),
        "device": "cuda",
    }


@lru_cache(maxsize=1)
def _safety_lineage() -> tuple[SafetyContract, str]:
    return (
        SafetyContract.from_stage6_config(load_stage6_config(CONFIG)),
        hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    )


def _single_candidate_observation() -> PolicyObservation:
    features = np.zeros((1, 22), dtype=np.float32)
    features[0, 15] = 1.0
    features[0, 18] = 1.0
    return PolicyObservation(
        prior_channels=np.zeros((7, 32, 32), dtype=np.float32),
        coverage_summary=np.zeros((8, 32, 32), dtype=np.float32),
        local_crop=np.zeros((8, 96, 96), dtype=np.float32),
        frontier_features=features,
        pose_features=np.zeros((6,), dtype=np.float32),
        candidate_mask=np.ones((1,), dtype=bool),
    )


def _single_candidate_output() -> PolicyForwardOutput:
    one = torch.ones((1, 1), dtype=torch.float32)
    zero = torch.zeros((1, 1), dtype=torch.float32)
    token = torch.zeros((1, 1, 4), dtype=torch.float32)
    return PolicyForwardOutput(
        frontier_logits=one,
        theta_mu_sin_raw=zero,
        theta_mu_cos_raw=one,
        theta_kappa_raw=zero,
        theta_mu=zero,
        theta_kappa=one,
        value=torch.zeros((1,), dtype=torch.float32),
        global_map_tokens=token,
        local_map_tokens=token,
        pose_token=token,
        context_tokens=token,
        refined_frontier_tokens=token,
        action_hidden=token,
    )


def _reset_diagnostics() -> dict[str, object]:
    return {
        "schema_version": "stage6_reset_scan_diagnostics/v1",
        "scan_order": ["reset_local_safety", "reset_exploration"],
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
    }


def _strict_execution_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, object, object]:
    from lunar_exploration_ppo.eval import midterm_reduced, standard

    frozen = _load_fixture_manifest(tmp_path, monkeypatch)
    _install_identity_factory(monkeypatch, midterm_reduced)
    job_plan = midterm_reduced.build_midterm_reduced_job_plan(
        catalog=_catalog(),
        frozen_manifest=frozen,
        cohort="test_q24",
        evaluation_seed_start=2026072600,
    )
    jobs = job_plan.jobs
    safety_contract, config_sha256 = _safety_lineage()
    specs = standard.standard_evaluation_env_specs(
        _catalog(),
        jobs,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )
    environment_contract = standard.build_standard_environment_contract(
        catalog=_catalog(),
        env_specs=specs,
        jobs=jobs,
        max_steps=128,
        success_threshold=0.99,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )
    observation = _single_candidate_observation()
    batch = batch_policy_observations((observation,))
    output = _single_candidate_output()
    action = EnvAction(candidate_index=0, target_theta=0.0)
    initial_count = 1024
    step_counts = (3277, 4056)
    initial_rate = initial_count / 4096
    step_rates = tuple(value / 4096 for value in step_counts)
    episodes = []
    episode_rows = []
    base_decisions = []
    trace_decisions = []
    step_rows = []
    for index, job in enumerate(jobs):
        coverage_curve = (
            initial_rate,
            *step_rates,
            *((step_rates[-1],) * 126),
        )
        path_curve = (0.0, 1.25, 2.50, *((2.50,) * 126))
        episode = build_episode_result(
            method="ppo_policy",
            scale_profile="Standard v1",
            scenario_key=job.scenario_id,
            scenario_seed=job.scenario_seed,
            terrain_seed=job.terrain_seed,
            start_pose_seed=job.start_pose_seed,
            evaluation_seed=job.evaluation_seed,
            coverage_curve=coverage_curve,
            cumulative_path_length_curve=path_curve,
            steps_executed=2,
            invalid_action_count=0,
            planner_failure_count=0,
            safety_violation_count=0,
            termination_reason="success_done",
            max_steps=128,
            success_threshold=0.99,
            zero_distance_policy=ZERO_DISTANCE_POLICY,
        )
        episodes.append(episode)
        episode_id = standard._standard_episode_id(job)
        lane_id = f"lane-{index % 8}"
        episode_rows.append(
            {
                **episode_record(episode),
                "steps_executed": 2,
                "reset_diagnostics": _reset_diagnostics(),
                "planner_failure_counts": {
                    reason: 0 for reason in PLANNER_FAILURE_REASONS
                },
                "episode_id": episode_id,
                "episode_index": index,
                "scenario_id": job.scenario_id,
                "lane_id": lane_id,
                "initial_coverage_rate": initial_rate,
                "initial_covered_cell_count": initial_count,
            }
        )
        previous_count = initial_count
        for step_index, covered_count in enumerate(step_counts):
            decision = standard.build_standard_decision_record(
                method="ppo_policy",
                sequence_index=len(base_decisions),
                episode_index=index,
                step_index=step_index,
                worker_index=index % 8,
                observation=observation,
                action=action,
                selector_inputs={
                    "observation_batch": batch,
                    "policy_output": output,
                },
                policy_batch_row_index=0,
            )
            base_decisions.append(decision)
            join_key = standard._standard_step_join_key(job, step_index)
            trace_decisions.append(
                {
                    **decision,
                    "join_key": join_key,
                    "episode_id": episode_id,
                    "scenario_id": job.scenario_id,
                    "lane_id": lane_id,
                }
            )
            path_length = 1.25
            cumulative_path_length = path_length * (step_index + 1)
            step_rows.append(
                {
                    "schema_version": "stage6-standard-step-trace/v1",
                    "join_key": join_key,
                    "episode_id": episode_id,
                    "episode_index": index,
                    "scenario_id": job.scenario_id,
                    "lane_id": lane_id,
                    "step_index": step_index,
                    "decision_sha256": decision["decision_sha256"],
                    "pre_observation_sha256": decision[
                        "policy_observation_sha256"
                    ],
                    "selected_candidate_index": 0,
                    "selected_candidate_cell_xy": [step_index, index],
                    "selected_theta": 0.0,
                    "post_observation_sha256": hashlib.sha256(
                        f"{index}:{step_index}:post".encode("utf-8")
                    ).hexdigest(),
                    "planned_path_cells": [[0, 0], [1, 0]],
                    "path_length_m": path_length,
                    "planner_path_length_m": path_length,
                    "cumulative_path_length_m": cumulative_path_length,
                    "coverage_gain_cells": covered_count - previous_count,
                    "coverage_gain_rate": (
                        (covered_count - previous_count) / 4096
                    ),
                    "coverage_rate": covered_count / 4096,
                    "done": step_index == 1,
                    "termination_reason": (
                        "success_done" if step_index == 1 else "none"
                    ),
                    "invalid_action": False,
                    "safety_violation": False,
                    "planner_diagnostics": {"failure_reason": "none"},
                }
            )
            previous_count = covered_count

    metrics, bootstrap = summarize_episodes(
        episodes,
        bootstrap_resamples=32,
        bootstrap_seed=17,
    )
    decision_schema = standard.build_standard_decision_input_schema()
    action_rule = standard.build_standard_action_rule_provenance("ppo_policy")
    parent_pid = os.getpid()
    fairness = standard.validate_standard_fairness_audit(
        {
            "schema_version": "stage6_standard_parallel_fairness_audit/v2",
            "method": "ppo_policy",
            "split": "test",
            "worker_count": 8,
            "worker_pids": list(range(parent_pid + 100, parent_pid + 108)),
            "worker_start_methods": ["spawn"] * 8,
            "evaluation_parent_pid": parent_pid,
            "inference_pids": [parent_pid],
            "policy_state_unchanged": True,
            "policy_state_sha256_before": "e" * 64,
            "policy_state_sha256_after": "e" * 64,
            "episode_action_counts": [2] * 24,
            "selected_action_count": 48,
            "scenario_schedule": [job.scenario_id for job in jobs],
            "evaluation_seeds": [job.evaluation_seed for job in jobs],
            "theta_source": "policy_theta_mu/v1",
            "shared_environment_contract": environment_contract,
            "shared_environment_contract_sha256": hashlib.sha256(
                ArtifactStore.canonical_json_bytes(environment_contract)
            ).hexdigest(),
            "decision_input_schema": decision_schema,
            "decision_input_schema_sha256": hashlib.sha256(
                ArtifactStore.canonical_json_bytes(decision_schema)
            ).hexdigest(),
            "action_rule": action_rule,
            "action_rule_sha256": hashlib.sha256(
                ArtifactStore.canonical_json_bytes(action_rule)
            ).hexdigest(),
            "decision_audit": standard._build_standard_decision_audit(
                "ppo_policy",
                base_decisions,
            ),
        },
        episode_count=24,
        parent_pid=parent_pid,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )
    execution = standard.StandardEvaluationExecution(
        summary=EvaluationSummary(
            method="ppo_policy",
            scale_profile="Standard v1",
            episodes=tuple(episodes),
            metrics=metrics,
            bootstrap_audit=bootstrap,
            fairness_audit=fairness,
        ),
        episode_rows=tuple(episode_rows),
        decision_rows=tuple(trace_decisions),
        step_rows=tuple(step_rows),
    )
    return execution, frozen, job_plan


def _derive_strict_fixture(execution, frozen, job_plan):
    from lunar_exploration_ppo.eval import midterm_reduced

    return midterm_reduced.derive_midterm_reduced_trace_summary(
        execution=execution,
        frozen_manifest=frozen,
        cohort="test_q24",
        job_plan=job_plan,
    )


def test_denominator_audit_and_path_steps_to_80_99_are_derived_from_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution, frozen, job_plan = _strict_execution_fixture(
        tmp_path,
        monkeypatch,
    )
    result = _derive_strict_fixture(execution, frozen, job_plan)
    expected_final = 4056 / 4096

    assert result["schema_version"] == "midterm-reduced-trace-summary/v1"
    assert result["split"] == "test"
    assert result["episode_count"] == 24
    assert result["denominator_audit"] == {
        "source": "reachable_observable_free_highres_cells/v1",
        "algorithm": "exact_reachable_safe_pose_range_los/v1",
        "proof_count": 24,
        "nonempty_count": 24,
        "exact_count": 24,
        "semantic_alias_proven_count": 24,
        "passed": True,
    }
    assert result["episodes"][0]["steps_to_80"] == 1
    assert result["episodes"][0]["path_length_to_80_m"] == pytest.approx(1.25)
    assert result["episodes"][0]["steps_to_99"] == 2
    assert result["episodes"][0]["path_length_to_99_m"] == pytest.approx(2.50)
    assert result["final_coverage"]["mean"] == pytest.approx(expected_final)
    assert result["final_coverage"]["coverage_80_count"] == 24
    assert result["final_coverage"]["coverage_99_count"] == 24


def test_reduced_adapter_binds_reset_rate_to_frozen_integer_denominator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced

    execution, frozen, job_plan = _strict_execution_fixture(
        tmp_path,
        monkeypatch,
    )
    unbound_rows = []
    for row in execution.episode_rows:
        unbound = dict(row)
        unbound.pop("initial_covered_cell_count")
        unbound_rows.append(unbound)
    bound = midterm_reduced._bind_initial_covered_counts(
        execution=replace(execution, episode_rows=tuple(unbound_rows)),
        frozen_manifest=frozen,
        job_plan=job_plan,
        cohort="test_q24",
    )

    assert {
        row["initial_covered_cell_count"] for row in bound.episode_rows
    } == {1024}
    assert _derive_strict_fixture(bound, frozen, job_plan)["episode_count"] == 24


def test_reduced_trace_rejects_empty_fake_summary_and_fairness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced

    execution, frozen, job_plan = _strict_execution_fixture(
        tmp_path,
        monkeypatch,
    )
    fake_summary = EvaluationSummary(
        method="ppo_policy",
        scale_profile="Standard v1",
        episodes=(),
        metrics={"episode_count": 24},
        bootstrap_audit={},
        fairness_audit={},
    )
    with pytest.raises(
        midterm_reduced.MidtermReducedEvaluationError,
        match="summary|fairness",
    ):
        _derive_strict_fixture(
            replace(execution, summary=fake_summary),
            frozen,
            job_plan,
        )


@pytest.mark.parametrize("mutation", ("empty", "reordered", "sha"))
def test_reduced_trace_rejects_missing_reordered_or_tampered_decisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced

    execution, frozen, job_plan = _strict_execution_fixture(
        tmp_path,
        monkeypatch,
    )
    decisions = [copy.deepcopy(row) for row in execution.decision_rows]
    if mutation == "empty":
        decisions = []
    elif mutation == "reordered":
        decisions[0], decisions[1] = decisions[1], decisions[0]
    else:
        decisions[0]["decision_sha256"] = "0" * 64
    with pytest.raises(
        midterm_reduced.MidtermReducedEvaluationError,
        match="decision|join|summary|fairness",
    ):
        _derive_strict_fixture(
            replace(execution, decision_rows=tuple(decisions)),
            frozen,
            job_plan,
        )


@pytest.mark.parametrize(
    ("target", "field", "value"),
    (
        ("episode", "steps_executed", 1),
        ("step", "done", False),
        ("step", "termination_reason", "none"),
        ("step", "join_key", "f" * 64),
    ),
)
def test_reduced_trace_rejects_count_termination_and_join_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    field: str,
    value: object,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced

    execution, frozen, job_plan = _strict_execution_fixture(
        tmp_path,
        monkeypatch,
    )
    episode_rows = [copy.deepcopy(row) for row in execution.episode_rows]
    step_rows = [copy.deepcopy(row) for row in execution.step_rows]
    if target == "episode":
        episode_rows[0][field] = value
    else:
        step_rows[1][field] = value
    with pytest.raises(
        midterm_reduced.MidtermReducedEvaluationError,
        match="step|termination|join|episode",
    ):
        _derive_strict_fixture(
            replace(
                execution,
                episode_rows=tuple(episode_rows),
                step_rows=tuple(step_rows),
            ),
            frozen,
            job_plan,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("coverage_gain_cells", 1),
        ("coverage_gain_rate", 0.5),
        ("coverage_rate", 0.90),
        ("cumulative_path_length_m", 99.0),
    ),
)
def test_reduced_trace_rejects_integer_denominator_and_path_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced

    execution, frozen, job_plan = _strict_execution_fixture(
        tmp_path,
        monkeypatch,
    )
    step_rows = [copy.deepcopy(row) for row in execution.step_rows]
    step_rows[0][field] = value
    with pytest.raises(
        midterm_reduced.MidtermReducedEvaluationError,
        match="coverage|path|denominator|count",
    ):
        _derive_strict_fixture(
            replace(execution, step_rows=tuple(step_rows)),
            frozen,
            job_plan,
        )


def test_reduced_trace_rejects_initial_integer_coverage_count_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced

    execution, frozen, job_plan = _strict_execution_fixture(
        tmp_path,
        monkeypatch,
    )
    episode_rows = [copy.deepcopy(row) for row in execution.episode_rows]
    episode_rows[0]["initial_covered_cell_count"] = 1025

    with pytest.raises(
        midterm_reduced.MidtermReducedEvaluationError,
        match="denominator|initial coverage",
    ):
        _derive_strict_fixture(
            replace(execution, episode_rows=tuple(episode_rows)),
            frozen,
            job_plan,
        )


def test_reduced_trace_rejects_schema_extensions_and_noncanonical_lanes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced

    execution, frozen, job_plan = _strict_execution_fixture(
        tmp_path,
        monkeypatch,
    )
    step_rows = [copy.deepcopy(row) for row in execution.step_rows]
    step_rows[0]["unreviewed_extension"] = True
    with pytest.raises(
        midterm_reduced.MidtermReducedEvaluationError,
        match="schema",
    ):
        _derive_strict_fixture(
            replace(execution, step_rows=tuple(step_rows)),
            frozen,
            job_plan,
        )

    episode_rows = [copy.deepcopy(row) for row in execution.episode_rows]
    for index, row in enumerate(episode_rows):
        row["lane_id"] = f"lane-{(index + 1) % 8}"
    with pytest.raises(
        midterm_reduced.MidtermReducedEvaluationError,
        match="lane",
    ):
        _derive_strict_fixture(
            replace(execution, episode_rows=tuple(episode_rows)),
            frozen,
            job_plan,
        )


_RUNNER_HASH_A = "a" * 64
_RUNNER_HASH_B = "b" * 64
_RUNNER_HASH_C = "c" * 64
_RUNNER_HASH_D = "d" * 64


def _runner_episode_rows(
    split: str,
    covered_counts: list[int],
    *,
    denominator_count: int = 100,
    safety_violation_count: int = 0,
    masked_action_count: int = 0,
) -> list[dict[str, object]]:
    runner = _g1_runner()
    return [
        runner.make_g1_coverage_episode_row(
            run_id="g1-fixture-run",
            split=split,
            episode_index=index,
            scenario_id=f"{split}/scenario-{index:02d}/standard-proxy/v1",
            config_sha256=_RUNNER_HASH_A,
            input_sha256=_RUNNER_HASH_B,
            code_sha256=_RUNNER_HASH_C,
            scenario_manifest_sha256=_RUNNER_HASH_D,
            denominator_sha256=hashlib.sha256(
                f"{split}:mask:{index}".encode("utf-8")
            ).hexdigest(),
            denominator_cell_count=denominator_count,
            initial_covered_cell_count=0,
            final_covered_cell_count=covered_count,
            elapsed_ms=0.0,
            steps_executed=1,
            termination_reason=(
                "success_done"
                if covered_count >= 99
                else "failure_done"
            ),
            safety_violation_count=safety_violation_count,
            masked_action_count=masked_action_count,
        )
        for index, covered_count in enumerate(covered_counts)
    ]


def test_g1_config_pins_update80_and_rejects_any_override(
    tmp_path: Path,
) -> None:
    runner = _g1_runner()
    payload = json.loads(G1_CONFIG.read_text(encoding="utf-8"))
    validated = runner.validate_g1_config_payload(payload)

    assert validated["checkpoint"] == {
        "update": 80,
        "path": (
            "D:/xunce/out/ppo_frontier/"
            "s6-standard-single-r1-20260724T000124Z/s6/"
            "checkpoints/seed-20260716/update-00000080/checkpoint.pt"
        ),
        "sha256": (
            "35e04c86f9f973af028fb08f0175d42ab45378d09e1f2b96ee6aad6e4c12b5b5"
        ),
        "policy_state_sha256": (
            "3123e6adde99be41e3bd5cc2f3843068e5416892395926d759c2c6a243ffd381"
        ),
        "device": "cuda",
        "dtype": "float32",
    }
    assert validated["execution"]["worker_count"] == 8
    assert validated["execution"]["lane_sizes"] == [3] * 8
    assert validated["execution"]["max_steps"] == 128
    assert validated["execution"]["environment_success_threshold"] == 0.99

    for field, value in (
        ("path", "D:/other/checkpoint.pt"),
        ("sha256", "0" * 64),
        ("policy_state_sha256", "1" * 64),
        ("device", "cpu"),
        ("dtype", "float64"),
    ):
        drifted = copy.deepcopy(payload)
        drifted["checkpoint"][field] = value
        with pytest.raises(runner.G1Blocked, match="g1_config_drift"):
            runner.validate_g1_config_payload(drifted)


def test_g1_dry_run_uses_three_validation_scenes_only() -> None:
    runner = _g1_runner()
    scenarios = (
        "validation/scene-a/standard-proxy/v1",
        "validation/scene-b/standard-proxy/v1",
        "validation/scene-c/standard-proxy/v1",
    )
    calls: list[tuple[str, ...]] = []

    audit = runner.run_validation_dry_run(
        scenarios,
        lambda selected: calls.append(tuple(selected))
        or tuple(
            {
                "scenario_id": scenario_id,
                "termination_reason": "success_done",
            }
            for scenario_id in selected
        ),
    )

    assert calls == [scenarios]
    assert audit["scenario_ids"] == list(scenarios)
    assert audit["scenario_count"] == 3
    with pytest.raises(runner.G1Blocked, match="validation3"):
        runner.run_validation_dry_run(scenarios[:2], lambda _selected: ())


def test_g1_q24_must_pass_before_unseen24_runs() -> None:
    runner = _g1_runner()
    q_rows = _runner_episode_rows(
        "test_q24",
        [100] * 22 + [79, 79],
    )
    unseen_rows = _runner_episode_rows("unseen24", [100] * 24)
    calls: list[str] = []

    result = runner.execute_formal_gate(
        lambda split: calls.append(split)
        or (q_rows if split == "test_q24" else unseen_rows)
    )

    assert calls == ["test_q24"]
    assert result["summary"]["status"] == "failed"
    assert result["summary"]["splits"]["unseen24"] is None
    assert result["summary"]["unseen_execution_status"] == (
        "skipped_due_to_test_q24_failure"
    )
    assert all(
        row["split"] == "test_q24"
        for row in result["rows"]
        if row["row_kind"] == "coverage_episode"
    )


def test_g1_test_c24_requires_a_new_repair_lineage_and_explicit_mode(
    tmp_path: Path,
) -> None:
    runner = _g1_runner()
    lineage = tmp_path / "repair-lineage.json"
    lineage.write_text(
        json.dumps(
            {
                "schema_version": "xunce-mid-dual-g1-repair-lineage/v1",
                "repair_id": "repair-after-q24-failure",
                "parent_run_id": "failed-q24-run",
                "code_sha256": _RUNNER_HASH_B,
                "config_sha256": _RUNNER_HASH_C,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(runner.G1Blocked, match="test_c_explicit_mode_required"):
        runner.validate_test_c_request(
            mode="formal",
            repair_lineage=lineage,
            current_code_sha256=_RUNNER_HASH_A,
            current_config_sha256=_RUNNER_HASH_D,
        )
    with pytest.raises(runner.G1Blocked, match="repair_lineage_required"):
        runner.validate_test_c_request(
            mode="test-c-confirmation",
            repair_lineage=None,
            current_code_sha256=_RUNNER_HASH_A,
            current_config_sha256=_RUNNER_HASH_D,
        )

    audit = runner.validate_test_c_request(
        mode="test-c-confirmation",
        repair_lineage=lineage,
        current_code_sha256=_RUNNER_HASH_A,
        current_config_sha256=_RUNNER_HASH_D,
    )
    assert audit["parent_run_id"] == "failed-q24-run"

    reused = json.loads(lineage.read_text(encoding="utf-8"))
    reused["code_sha256"] = _RUNNER_HASH_A
    reused["config_sha256"] = _RUNNER_HASH_D
    lineage.write_text(json.dumps(reused), encoding="utf-8")
    with pytest.raises(runner.G1Blocked, match="repair_lineage_not_new"):
        runner.validate_test_c_request(
            mode="test-c-confirmation",
            repair_lineage=lineage,
            current_code_sha256=_RUNNER_HASH_A,
            current_config_sha256=_RUNNER_HASH_D,
        )


def test_g1_requires_23_of_24_and_macro_mean_for_both_thresholds() -> None:
    runner = _g1_runner()

    count_fails = runner.recompute_g1_split_summary(
        _runner_episode_rows("test_q24", [100] * 22 + [79, 79]),
        expected_split="test_q24",
    )
    assert count_fails["mean"] > 0.80
    assert count_fails["coverage_80_count"] == 22
    assert count_fails["g1_coverage_80_passed"] is False

    mean_fails = runner.recompute_g1_split_summary(
        _runner_episode_rows("test_q24", [99] * 23 + [98]),
        expected_split="test_q24",
    )
    assert mean_fails["coverage_99_count"] == 23
    assert mean_fails["mean"] < 0.99
    assert mean_fails["g1_coverage_99_passed"] is False

    both_pass = runner.recompute_g1_summary(
        test_q24_rows=_runner_episode_rows(
            "test_q24",
            [100] * 23 + [80],
        ),
        unseen24_rows=_runner_episode_rows(
            "unseen24",
            [100] * 23 + [80],
        ),
    )
    assert both_pass["g1_coverage_80_passed"] is True
    assert both_pass["g1_coverage_99_passed"] is True
    assert both_pass["status"] == "passed"


def test_g1_replay_requires_identical_actions_curve_and_termination() -> None:
    runner = _g1_runner()
    reference = tuple(
        {
            "scenario_id": f"test/replay-{index}/standard-proxy/v1",
            "actions": [
                {
                    "selected_candidate_index": index,
                    "selected_theta": 0.125 * index,
                }
            ],
            "coverage_curve": [0.25, 0.99],
            "termination_reason": "success_done",
        }
        for index in range(3)
    )
    replay = copy.deepcopy(reference)

    assert runner.validate_g1_replay(reference, replay)["passed"] is True

    replay[1]["coverage_curve"][-1] = 0.98
    with pytest.raises(runner.G1Blocked, match="replay_mismatch"):
        runner.validate_g1_replay(reference, replay)


def test_g1_results_preserve_all_low_coverage_and_failure_rows() -> None:
    runner = _g1_runner()
    q_rows = _runner_episode_rows("test_q24", [100] * 23 + [79])
    unseen_rows = _runner_episode_rows("unseen24", [100] * 23 + [79])
    result = runner.execute_formal_gate(
        lambda split: q_rows if split == "test_q24" else unseen_rows
    )
    coverage_rows = [
        row
        for row in result["rows"]
        if row["row_kind"] == "coverage_episode"
    ]

    assert len(coverage_rows) == 48
    assert sum(row["coverage"] < 0.80 for row in coverage_rows) == 2
    assert {
        row["termination_reason"]
        for row in coverage_rows
        if row["coverage"] < 0.80
    } == {"failure_done"}


def test_g1_hidden_truth_never_enters_policy_or_candidate_input() -> None:
    runner = _g1_runner()
    tree = ast.parse(inspect.getsource(runner._evaluate_formal_cohort))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    source = inspect.getsource(runner._evaluate_formal_cohort)

    assert "run_midterm_reduced_evaluation" in calls
    assert "freeze_selection" not in source
    assert "scenario_ids=" not in source
    assert ".truth" not in source
    assert "cpu" not in source.lower()


def test_g1_missing_manifest_preflight_does_not_call_cuda_loader() -> None:
    runner = _g1_runner()
    called = False

    def forbidden_loader(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("CUDA loader must not run")

    result = runner.preflight_scenario_manifest(
        "D:/xunce/inputs/mid_dual/missing/manifest.json",
        loader=forbidden_loader,
    )
    assert result == {
        "status": "blocked",
        "blocking_reason": "scenario_manifest_missing",
    }
    assert called is False


@pytest.mark.parametrize(
    "run_id",
    ("..", "g1..child", "g1/child", "g1\\child", "CON", "run."),
)
def test_g1_rejects_unsafe_run_ids_and_non_d_formal_input(
    run_id: str,
) -> None:
    runner = _g1_runner()
    with pytest.raises(runner.G1Blocked, match="run_id_invalid"):
        runner._validated_run_id(run_id)
    with pytest.raises(
        runner.G1Blocked,
        match="scenario_manifest_path_invalid",
    ):
        runner._validate_d_manifest_path(
            "C:/xunce/inputs/mid_dual/frozen/manifest.json"
        )
