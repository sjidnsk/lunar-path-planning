"""Stage 6 Standard v1 固定调度与 8-worker 公平评估。"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn

from lunar_exploration_ppo.configs.stage6 import SafetyContract
from lunar_exploration_ppo.env.env import EnvAction, LunarExplorationEnv
from lunar_exploration_ppo.env.frontier import (
    CANDIDATE_PRIORITY_SOURCE,
    POTENTIAL_GAIN_SOURCE,
    SEGMENT_POLICY,
    TOP_M_SELECTION_POLICY,
    FrontierGenerator,
)
from lunar_exploration_ppo.env.scenario_catalog import StandardScenarioCatalog
from lunar_exploration_ppo.env.sensor_model import SensorUpdater
from lunar_exploration_ppo.env.standard_training import (
    StandardEnvSettings,
    StandardEvaluationEnv,
    build_standard_evaluation_env,
)
from lunar_exploration_ppo.eval.baselines import (
    ALL_METHODS,
    BASELINE_METHODS,
    reconstruct_recommended_theta,
    select_baseline_action,
)
from lunar_exploration_ppo.eval.evaluator import (
    METHOD_ACTION_RULES,
    PPO_EVAL_POLICY_MODE,
    EvaluationSummary,
)
from lunar_exploration_ppo.eval.metrics import (
    ZERO_DISTANCE_POLICY,
    EpisodeResult,
    build_episode_result,
    episode_record,
    summarize_episodes,
)
from lunar_exploration_ppo.integrations.path_planner_adapter import PathPlannerAdapter
from lunar_exploration_ppo.policy.cross_attention import (
    PolicyBatch,
    PolicyForwardOutput,
    batch_policy_observations,
)
from lunar_exploration_ppo.policy.observation import PolicyObservation
from lunar_exploration_ppo.ppo.collector import (
    PLANNER_FAILURE_REASONS,
    SpawnEnvSpec,
    SpawnVectorEnv,
    validate_reset_diagnostics_payload,
)
from lunar_exploration_ppo.ppo.trainer import policy_state_sha256
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl


STANDARD_EVALUATION_METHODS = (
    "ppo_policy",
    "random_valid_frontier",
    "nearest_frontier",
    "max_potential_gain_frontier",
    "gain_over_cost_frontier",
)
STANDARD_EVALUATION_WORKERS = 8
_CANONICAL_EPISODE_COUNTS = {
    "validation": 16,
    "test": 64,
    "unseen": 64,
}
_FAIRNESS_SCHEMA_VERSION = "stage6_standard_parallel_fairness_audit/v2"
_ENVIRONMENT_CONTRACT_SCHEMA_VERSION = (
    "stage6_standard_shared_environment_contract/v1"
)
_DECISION_INPUT_SCHEMA_VERSION = (
    "stage6_policy_observation_decision_input_schema/v1"
)
_ACTION_RULE_SCHEMA_VERSION = "stage6_method_action_rule_provenance/v1"
_DECISION_AUDIT_SCHEMA_VERSION = "stage6_standard_decision_audit/v1"
_DECISION_RECORD_SCHEMA_VERSION = "stage6_standard_decision_record/v1"
_THETA_TOLERANCE_RAD = 1.0e-7
_TRUTH_LIKE_FIELD_TOKENS = (
    "truth",
    "ground_truth",
    "coverable_mask",
    "coverablemask",
    "hidden_highres",
    "hidden_high_resolution",
    "dense_coverable",
)


def _is_supported_explicit_schedule(split: object, episode_count: object) -> bool:
    """Accept canonical schedules plus the explicit reduced Test/Unseen 24 seam."""

    return (
        isinstance(split, str)
        and type(episode_count) is int
        and (
            episode_count == _CANONICAL_EPISODE_COUNTS.get(split)
            or (episode_count == 24 and split in {"test", "unseen"})
        )
    )


class StandardEvaluationError(RuntimeError):
    """Standard evaluation 无法保持固定调度或公平性。"""


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(ArtifactStore.canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _qualified_name(value: object) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        raise StandardEvaluationError("provenance source has no qualified name")
    return f"{module}.{qualname}"


def _source_sha256(value: object) -> str:
    try:
        source = inspect.getsource(value).encode("utf-8")
    except (OSError, TypeError) as exc:
        raise StandardEvaluationError("action rule source is unavailable") from exc
    return hashlib.sha256(source).hexdigest()


def _nested_field_names(value: object) -> tuple[str, ...]:
    names: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                names.append(str(key))
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                if isinstance(child, str):
                    names.append(child)
                else:
                    visit(child)

    visit(value)
    return tuple(names)


def _truth_like_field_names(names: Sequence[str]) -> list[str]:
    matches: set[str] = set()
    for name in names:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(name).casefold()).strip("_")
        if any(token in normalized for token in _TRUTH_LIKE_FIELD_TOKENS):
            matches.add(str(name))
    return sorted(matches)


def validate_standard_fairness_audit(
    value: Mapping[str, object],
    *,
    episode_count: int,
    parent_pid: int | None,
    safety_contract: SafetyContract,
    config_sha256: str,
) -> dict[str, object]:
    """重算 worker、共享合同、决策 schema 与逐动作 provenance。"""

    required = {
        "schema_version",
        "method",
        "split",
        "worker_count",
        "worker_pids",
        "worker_start_methods",
        "evaluation_parent_pid",
        "inference_pids",
        "policy_state_unchanged",
        "policy_state_sha256_before",
        "policy_state_sha256_after",
        "episode_action_counts",
        "selected_action_count",
        "scenario_schedule",
        "evaluation_seeds",
        "theta_source",
        "shared_environment_contract",
        "shared_environment_contract_sha256",
        "decision_input_schema",
        "decision_input_schema_sha256",
        "action_rule",
        "action_rule_sha256",
        "decision_audit",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) not in (required, {*required, "passed"})
        or type(episode_count) is not int
        or episode_count not in {16, 24, 64}
        or (parent_pid is not None and (type(parent_pid) is not int or parent_pid <= 0))
    ):
        raise StandardEvaluationError("Standard fairness audit schema drifted")

    method = value["method"]
    split = value["split"]
    worker_pids = value["worker_pids"]
    start_methods = value["worker_start_methods"]
    recorded_parent_pid = value["evaluation_parent_pid"]
    inference_pids = value["inference_pids"]
    policy_state_before = value["policy_state_sha256_before"]
    policy_state_after = value["policy_state_sha256_after"]
    action_counts = value["episode_action_counts"]
    scenarios = value["scenario_schedule"]
    seeds = value["evaluation_seeds"]
    environment_contract = value["shared_environment_contract"]
    decision_schema = value["decision_input_schema"]
    action_rule = value["action_rule"]
    decision_audit = value["decision_audit"]
    expected_theta = (
        "policy_theta_mu/v1"
        if method == "ppo_policy"
        else "candidate_recommended_theta/v1"
    )

    try:
        _validate_standard_environment_contract(
            environment_contract,
            episode_count=episode_count,
            safety_contract=safety_contract,
            config_sha256=config_sha256,
        )
        expected_decision_schema = build_standard_decision_input_schema()
        if decision_schema != expected_decision_schema:
            raise StandardEvaluationError("decision input schema drifted")
        expected_action_rule = build_standard_action_rule_provenance(str(method))
        if action_rule != expected_action_rule:
            raise StandardEvaluationError("method action rule drifted")
        recomputed_decision_audit = _build_standard_decision_audit(
            str(method),
            decision_audit.get("decisions", ())  # type: ignore[union-attr]
            if isinstance(decision_audit, Mapping)
            else (),
        )
    except (KeyError, TypeError, ValueError, StandardEvaluationError) as exc:
        raise StandardEvaluationError("Standard fairness provenance failed") from exc

    schedule = environment_contract["scenario_seed_schedule"]  # type: ignore[index]
    contract_scenarios = [row["scenario_id"] for row in schedule]
    contract_seeds = [row["evaluation_seed"] for row in schedule]
    selected_action_count = value["selected_action_count"]
    expected_inference = (
        [recorded_parent_pid]
        if method == "ppo_policy" and selected_action_count != 0
        else []
    )
    decision_rows = recomputed_decision_audit["decisions"]
    if any(
        not 0 <= int(row["episode_index"]) < episode_count
        for row in decision_rows
    ):
        raise StandardEvaluationError("Standard fairness decision cohort failed")
    per_episode_steps: list[list[int]] = [[] for _ in range(episode_count)]
    for row in decision_rows:
        per_episode_steps[int(row["episode_index"])].append(int(row["step_index"]))
    recorded_counts = [len(items) for items in per_episode_steps]
    steps_are_contiguous = all(
        sorted(items) == list(range(len(items))) for items in per_episode_steps
    )

    passed = (
        value["schema_version"] == _FAIRNESS_SCHEMA_VERSION
        and method in STANDARD_EVALUATION_METHODS
        and split in _CANONICAL_EPISODE_COUNTS
        and environment_contract["split"] == split  # type: ignore[index]
        and type(recorded_parent_pid) is int
        and recorded_parent_pid > 0
        and (parent_pid is None or parent_pid == recorded_parent_pid)
        and value["worker_count"] == STANDARD_EVALUATION_WORKERS
        and isinstance(worker_pids, list)
        and len(worker_pids) == STANDARD_EVALUATION_WORKERS
        and len(set(worker_pids)) == STANDARD_EVALUATION_WORKERS
        and all(
            type(pid) is int and pid > 0 and pid != recorded_parent_pid
            for pid in worker_pids
        )
        and start_methods == ["spawn"] * STANDARD_EVALUATION_WORKERS
        and inference_pids == expected_inference
        and value["policy_state_unchanged"] is (policy_state_before == policy_state_after)
        and (
            _is_sha256(policy_state_before)
            and policy_state_after == policy_state_before
            if method == "ppo_policy"
            else policy_state_before is None and policy_state_after is None
        )
        and isinstance(action_counts, list)
        and len(action_counts) == episode_count
        and all(type(count) is int and 0 <= count <= 128 for count in action_counts)
        and type(selected_action_count) is int
        and selected_action_count == sum(action_counts)
        and recorded_counts == action_counts
        and steps_are_contiguous
        and isinstance(scenarios, list)
        and scenarios == contract_scenarios
        and len(set(scenarios)) == episode_count
        and isinstance(seeds, list)
        and seeds == contract_seeds
        and len(set(seeds)) == episode_count
        and value["theta_source"] == expected_theta
        and value["shared_environment_contract_sha256"]
        == _canonical_sha256(environment_contract)
        and value["decision_input_schema_sha256"]
        == _canonical_sha256(decision_schema)
        and value["action_rule_sha256"] == _canonical_sha256(action_rule)
        and decision_audit == recomputed_decision_audit
        and recomputed_decision_audit["decision_count"] == selected_action_count
        and recomputed_decision_audit["policy_observation_input_count"]
        == selected_action_count
        and recomputed_decision_audit["candidate_mask_valid_count"]
        == selected_action_count
        and recomputed_decision_audit["selected_index_valid_count"]
        == selected_action_count
        and recomputed_decision_audit["runtime_truth_like_selector_field_count"]
        == 0
        and (
            recomputed_decision_audit[
                "ppo_observation_batch_policy_output_count"
            ]
            == selected_action_count
            if method == "ppo_policy"
            else recomputed_decision_audit["baseline_theta_check_count"]
            == selected_action_count
            and recomputed_decision_audit["baseline_theta_match_count"]
            == selected_action_count
        )
        and ("passed" not in value or value["passed"] is True)
    )
    if not passed:
        raise StandardEvaluationError("Standard fairness audit failed")
    return {**dict(value), "passed": True}


def validate_standard_fairness_cohort(
    audits: Sequence[Mapping[str, object]],
    *,
    split: str,
    episode_count: int,
    safety_contract: SafetyContract,
    config_sha256: str,
) -> dict[str, object]:
    """重放同一 split 的五方法 audit，并校验全部共享合同投影。"""

    if (
        not isinstance(audits, Sequence)
        or len(audits) != len(STANDARD_EVALUATION_METHODS)
        or split not in _CANONICAL_EPISODE_COUNTS
        or episode_count != _CANONICAL_EPISODE_COUNTS[split]
    ):
        raise StandardEvaluationError("Standard fairness cohort schema drifted")
    try:
        validated = [
            validate_standard_fairness_audit(
                audit,
                episode_count=episode_count,
                parent_pid=None,
                safety_contract=safety_contract,
                config_sha256=config_sha256,
            )
            for audit in audits
        ]
    except (TypeError, ValueError, StandardEvaluationError) as exc:
        raise StandardEvaluationError("Standard fairness cohort member failed") from exc

    by_method = {str(audit["method"]): audit for audit in validated}
    if (
        len(by_method) != len(STANDARD_EVALUATION_METHODS)
        or set(by_method) != set(STANDARD_EVALUATION_METHODS)
    ):
        raise StandardEvaluationError("Standard fairness cohort methods drifted")
    ordered = [by_method[method] for method in STANDARD_EVALUATION_METHODS]
    reference = ordered[0]
    shared_keys = (
        "split",
        "worker_count",
        "worker_start_methods",
        "scenario_schedule",
        "evaluation_seeds",
        "shared_environment_contract",
        "shared_environment_contract_sha256",
        "decision_input_schema",
        "decision_input_schema_sha256",
    )
    if any(
        audit["split"] != split
        or any(audit[key] != reference[key] for key in shared_keys)
        for audit in ordered
    ):
        raise StandardEvaluationError("Standard fairness cohort shared contract drifted")
    action_rule_sha256_by_method = {
        method: by_method[method]["action_rule_sha256"]
        for method in STANDARD_EVALUATION_METHODS
    }
    if len(set(action_rule_sha256_by_method.values())) != len(
        STANDARD_EVALUATION_METHODS
    ):
        raise StandardEvaluationError("Standard fairness cohort action rules drifted")
    return {
        "schema_version": "stage6_standard_fairness_cohort/v1",
        "split": split,
        "episode_count": episode_count,
        "methods": list(STANDARD_EVALUATION_METHODS),
        "scenario_schedule_sha256": _canonical_sha256(
            reference["shared_environment_contract"]["scenario_seed_schedule"]
        ),
        "shared_environment_contract_sha256": reference[
            "shared_environment_contract_sha256"
        ],
        "decision_input_schema_sha256": reference[
            "decision_input_schema_sha256"
        ],
        "action_rule_sha256_by_method": action_rule_sha256_by_method,
        "passed": True,
    }


@dataclass(frozen=True, slots=True)
class StandardEvaluationJob:
    split: Literal["validation", "test", "unseen"]
    episode_index: int
    scenario_id: str
    scenario_seed: int
    terrain_seed: int
    start_pose_seed: int
    evaluation_seed: int
    theta_source: str

    def __post_init__(self) -> None:
        if (
            self.split not in _CANONICAL_EPISODE_COUNTS
            or type(self.episode_index) is not int
            or self.episode_index < 0
            or not isinstance(self.scenario_id, str)
            or not self.scenario_id.startswith(f"{self.split}/")
            or any(
                type(value) is not int
                for value in (
                    self.scenario_seed,
                    self.terrain_seed,
                    self.start_pose_seed,
                    self.evaluation_seed,
                )
            )
            or self.theta_source
            not in {"policy_theta_mu/v1", "candidate_recommended_theta/v1"}
        ):
            raise StandardEvaluationError("Standard evaluation job drifted")


@dataclass(frozen=True, slots=True)
class StandardEvaluationExecution:
    """In-memory Standard evidence returned by the explicit job seam."""

    summary: EvaluationSummary
    episode_rows: tuple[dict[str, object], ...]
    decision_rows: tuple[dict[str, object], ...]
    step_rows: tuple[dict[str, object], ...]


def build_standard_evaluation_schedule(
    catalog: StandardScenarioCatalog,
    *,
    split: str,
    episode_count: int,
    evaluation_seed_start: int,
    theta_source: str = "policy_theta_mu/v1",
) -> tuple[StandardEvaluationJob, ...]:
    if (
        not isinstance(catalog, StandardScenarioCatalog)
        or split not in _CANONICAL_EPISODE_COUNTS
        or episode_count != _CANONICAL_EPISODE_COUNTS.get(split)
        or type(evaluation_seed_start) is not int
    ):
        raise StandardEvaluationError("Standard evaluation schedule is not canonical")
    records = tuple(record for record in catalog.records if record.split == split)
    if len(records) < episode_count:
        raise StandardEvaluationError("Standard evaluation schedule lacks scenarios")
    jobs = tuple(
        StandardEvaluationJob(
            split=split,  # type: ignore[arg-type]
            episode_index=index,
            scenario_id=record.scenario_id,
            scenario_seed=int(record.scenario_seed_hex, 16),
            terrain_seed=int(record.terrain_seed_hex, 16),
            start_pose_seed=int(record.start_pose_seed_hex, 16),
            evaluation_seed=evaluation_seed_start + index,
            theta_source=theta_source,
        )
        for index, record in enumerate(records[:episode_count])
    )
    if len({job.scenario_id for job in jobs}) != episode_count:
        raise StandardEvaluationError("Standard evaluation schedule reused a scenario")
    return jobs


def partition_standard_evaluation_jobs(
    jobs: Sequence[StandardEvaluationJob],
) -> tuple[tuple[StandardEvaluationJob, ...], ...]:
    schedule = tuple(jobs)
    if (
        len(schedule) not in {16, 24, 64}
        or any(not isinstance(job, StandardEvaluationJob) for job in schedule)
        or tuple(job.episode_index for job in schedule) != tuple(range(len(schedule)))
    ):
        raise StandardEvaluationError("Standard evaluation worker schedule drifted")
    return tuple(
        tuple(schedule[index::STANDARD_EVALUATION_WORKERS])
        for index in range(STANDARD_EVALUATION_WORKERS)
    )


def build_standard_final_method_schedules(
    catalog: StandardScenarioCatalog,
    *,
    split: str,
    evaluation_seed_start: int,
) -> dict[str, tuple[StandardEvaluationJob, ...]]:
    if split not in {"test", "unseen"}:
        raise StandardEvaluationError("final Standard evaluation schedule split drifted")
    return {
        method: build_standard_evaluation_schedule(
            catalog,
            split=split,
            episode_count=64,
            evaluation_seed_start=evaluation_seed_start,
            theta_source=(
                "policy_theta_mu/v1"
                if method == "ppo_policy"
                else "candidate_recommended_theta/v1"
            ),
        )
        for method in STANDARD_EVALUATION_METHODS
    }


def standard_evaluation_env_specs(
    catalog: StandardScenarioCatalog,
    jobs: Sequence[StandardEvaluationJob],
    *,
    safety_contract: SafetyContract,
    config_sha256: str,
) -> tuple[SpawnEnvSpec, ...]:
    schedule = tuple(jobs)
    if (
        not isinstance(catalog, StandardScenarioCatalog)
        or not schedule
        or any(not isinstance(job, StandardEvaluationJob) for job in schedule)
        or len({job.split for job in schedule}) != 1
        or not _is_supported_explicit_schedule(schedule[0].split, len(schedule))
    ):
        raise StandardEvaluationError("Standard evaluation env schedule drifted")
    lanes = partition_standard_evaluation_jobs(schedule)
    payload = catalog.to_dict()
    split = schedule[0].split
    safety_binding = _validated_safety_binding(
        safety_contract,
        config_sha256=config_sha256,
    )
    return tuple(
        SpawnEnvSpec(
            factory=build_standard_evaluation_env,
            kwargs={
                "catalog_payload": payload,
                "split": split,
                "scenario_ids": [job.scenario_id for job in lane],
                "production": True,
                **safety_binding,
            },
        )
        for lane in lanes
    )


def _validated_safety_binding(
    safety_contract: object,
    *,
    config_sha256: str,
) -> dict[str, object]:
    if not isinstance(safety_contract, SafetyContract):
        raise StandardEvaluationError(
            "shared environment requires validated SafetyContract"
        )
    try:
        binding = safety_contract.binding(config_sha256=config_sha256)
        restored = SafetyContract.from_binding(
            binding,
            expected_config_sha256=config_sha256,
        )
    except (TypeError, ValueError) as exc:
        raise StandardEvaluationError(
            "shared environment safety/config binding drifted"
        ) from exc
    if restored != safety_contract:
        raise StandardEvaluationError("shared environment safety contract drifted")
    return binding


def _validate_standard_safety_contract(
    value: object,
    *,
    safety_contract: SafetyContract,
    config_sha256: str,
) -> None:
    if not isinstance(value, Mapping):
        raise StandardEvaluationError("shared environment safety contract drifted")
    expected_binding = _validated_safety_binding(
        safety_contract,
        config_sha256=config_sha256,
    )
    try:
        restored = SafetyContract.from_binding(
            value,
            expected_config_sha256=config_sha256,
        )
    except (TypeError, ValueError) as exc:
        raise StandardEvaluationError(
            "shared environment safety/config binding drifted"
        ) from exc
    if restored != safety_contract or dict(value) != expected_binding:
        raise StandardEvaluationError("shared environment safety contract drifted")


def _static_standard_environment_contract(
    *,
    safety_contract: SafetyContract,
    config_sha256: str,
) -> dict[str, object]:
    safety_binding = _validated_safety_binding(
        safety_contract,
        config_sha256=config_sha256,
    )
    settings_object = StandardEnvSettings(
        scenario_key="<scheduled-scenario>",
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )
    settings = asdict(settings_object)
    settings.pop("scenario_key")
    settings.pop("safety_contract")
    settings.pop("config_sha256")
    settings.update(
        {
            name: getattr(settings_object, name)
            for name in safety_contract.to_dict()
        }
    )
    return {
        "evaluation_environment_source": _qualified_name(StandardEvaluationEnv),
        "inner_environment_source": _qualified_name(LunarExplorationEnv),
        "settings_source": _qualified_name(StandardEnvSettings),
        "settings": settings,
        **safety_binding,
        "coverage_denominator_contract": {
            "source": "coverable_mask_exact",
            "access": "environment_metric_layer_only/v1",
        },
        "planner_contract": {
            "source": _qualified_name(PathPlannerAdapter),
            "version": "stage1_path_planner_adapter/v1",
            "neighbor_policy": "8-neighbor",
            "prevent_corner_cutting": True,
        },
        "sensor_contract": {
            "source": _qualified_name(SensorUpdater),
            "model_id": (
                "path-tangent-plus-endpoint-theta-fov-90-range-20m-los/v1"
            ),
            "range_m": settings["sensor_range_m"],
            "fov_deg": settings["sensor_fov_deg"],
            "ray_angle_step_deg": settings["sensor_ray_angle_step_deg"],
            "path_observation_step_m": settings["path_observation_step_m"],
        },
        "candidate_contract": {
            "source": _qualified_name(FrontierGenerator),
            "version": "stage2_observed_frontier_top_m/v1",
            "top_m": settings["frontier_top_m"],
            "feature_schema": "frontier_features_22/v1",
            "observation_schema_version": PolicyObservation.schema_version,
            "segment_policy": SEGMENT_POLICY,
            "potential_gain_source": POTENTIAL_GAIN_SOURCE,
            "top_m_selection_policy": TOP_M_SELECTION_POLICY,
            "candidate_priority_source": CANDIDATE_PRIORITY_SOURCE,
            "reachability_prefilter": "observed_safe_connected_component/v1",
        },
    }


def build_standard_environment_contract(
    *,
    catalog: StandardScenarioCatalog,
    env_specs: Sequence[SpawnEnvSpec],
    jobs: Sequence[StandardEvaluationJob],
    max_steps: int,
    success_threshold: float,
    safety_contract: SafetyContract,
    config_sha256: str,
) -> dict[str, object]:
    """从实际 catalog、spawn specs 与固定 schedule 构建共享环境合同。"""

    schedule = tuple(jobs)
    specs = tuple(env_specs)
    if (
        not isinstance(catalog, StandardScenarioCatalog)
        or not schedule
        or any(not isinstance(job, StandardEvaluationJob) for job in schedule)
        or len({job.split for job in schedule}) != 1
        or not _is_supported_explicit_schedule(schedule[0].split, len(schedule))
        or len(specs) != STANDARD_EVALUATION_WORKERS
        or any(not isinstance(spec, SpawnEnvSpec) for spec in specs)
        or max_steps != 128
        or success_threshold != 0.99
    ):
        raise StandardEvaluationError("shared environment contract inputs drifted")
    split = schedule[0].split
    if tuple(job.episode_index for job in schedule) != tuple(range(len(schedule))):
        raise StandardEvaluationError("shared environment schedule drifted")

    catalog_payload = catalog.to_dict()
    catalog_payload_sha256 = _canonical_sha256(catalog_payload)
    safety_binding = _validated_safety_binding(
        safety_contract,
        config_sha256=config_sha256,
    )
    normalized_specs: list[dict[str, object]] = []
    for worker_index, spec in enumerate(specs):
        kwargs = dict(spec.kwargs)
        expected_ids = [
            job.scenario_id
            for job in schedule[worker_index::STANDARD_EVALUATION_WORKERS]
        ]
        if (
            spec.factory is not build_standard_evaluation_env
            or set(kwargs)
            != {
                "catalog_payload",
                "split",
                "scenario_ids",
                "production",
                *safety_binding,
            }
            or _canonical_sha256(kwargs["catalog_payload"])
            != catalog_payload_sha256
            or kwargs["split"] != split
            or kwargs["scenario_ids"] != expected_ids
            or kwargs["production"] is not True
            or any(kwargs[name] != value for name, value in safety_binding.items())
        ):
            raise StandardEvaluationError("shared environment spawn spec drifted")
        normalized_specs.append(
            {
                "worker_index": worker_index,
                "factory_source": _qualified_name(spec.factory),
                "catalog_payload_sha256": catalog_payload_sha256,
                "split": split,
                "scenario_ids": expected_ids,
                "production": True,
                **safety_binding,
            }
        )

    contract = {
        "schema_version": _ENVIRONMENT_CONTRACT_SCHEMA_VERSION,
        "catalog": {
            "schema_version": catalog_payload["schema_version"],
            "catalog_sha256": catalog.sha256,
            "catalog_payload_sha256": catalog_payload_sha256,
            "split_policy": catalog.split_policy,
            "source_sha256": {
                "dem": catalog.sources.dem.sha256,
                "slope": catalog.sources.slope.sha256,
            },
            "spatial_audit": catalog.spatial_audit(),
        },
        "split": split,
        "scenario_seed_schedule": [
            {
                "episode_index": job.episode_index,
                "scenario_id": job.scenario_id,
                "scenario_seed": job.scenario_seed,
                "terrain_seed": job.terrain_seed,
                "start_pose_seed": job.start_pose_seed,
                "evaluation_seed": job.evaluation_seed,
            }
            for job in schedule
        ],
        "environment_specs": normalized_specs,
        "max_steps": max_steps,
        "success_threshold": success_threshold,
        "environment_contract": _static_standard_environment_contract(
            safety_contract=safety_contract,
            config_sha256=config_sha256,
        ),
    }
    _validate_standard_environment_contract(
        contract,
        episode_count=len(schedule),
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )
    return contract


def _validate_standard_environment_contract(
    value: object,
    *,
    episode_count: int,
    safety_contract: SafetyContract,
    config_sha256: str,
) -> None:
    required = {
        "schema_version",
        "catalog",
        "split",
        "scenario_seed_schedule",
        "environment_specs",
        "max_steps",
        "success_threshold",
        "environment_contract",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise StandardEvaluationError("shared environment contract schema drifted")
    runtime_environment_contract = value["environment_contract"]
    if not isinstance(runtime_environment_contract, Mapping):
        raise StandardEvaluationError("shared environment safety contract drifted")
    safety_binding = _validated_safety_binding(
        safety_contract,
        config_sha256=config_sha256,
    )
    _validate_standard_safety_contract(
        {
            name: runtime_environment_contract.get(name)
            for name in safety_binding
        },
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )
    contract_fields = set(_nested_field_names(value))
    if not contract_fields.isdisjoint(
        {
            "method",
            "theta_source",
            "action_rule",
            "candidate_index_rule",
            "theta_rule",
        }
    ):
        raise StandardEvaluationError("shared environment contract contains method data")

    split = value["split"]
    schedule = value["scenario_seed_schedule"]
    specs = value["environment_specs"]
    catalog = value["catalog"]
    if (
        value["schema_version"] != _ENVIRONMENT_CONTRACT_SCHEMA_VERSION
        or split not in _CANONICAL_EPISODE_COUNTS
        or not _is_supported_explicit_schedule(split, episode_count)
        or value["max_steps"] != 128
        or value["success_threshold"] != 0.99
        or value["environment_contract"]
        != _static_standard_environment_contract(
            safety_contract=safety_contract,
            config_sha256=config_sha256,
        )
        or not isinstance(schedule, list)
        or len(schedule) != episode_count
        or not isinstance(specs, list)
        or len(specs) != STANDARD_EVALUATION_WORKERS
        or not isinstance(catalog, Mapping)
        or set(catalog)
        != {
            "schema_version",
            "catalog_sha256",
            "catalog_payload_sha256",
            "split_policy",
            "source_sha256",
            "spatial_audit",
        }
        or catalog["schema_version"] != "standard_unseen_scenario_catalog/v1"
        or not _is_sha256(catalog["catalog_sha256"])
        or not _is_sha256(catalog["catalog_payload_sha256"])
        or catalog["split_policy"] != "fixed_seed_disjoint_split/v1"
        or not isinstance(catalog["source_sha256"], Mapping)
        or set(catalog["source_sha256"]) != {"dem", "slope"}
        or not all(_is_sha256(item) for item in catalog["source_sha256"].values())
        or catalog["spatial_audit"]
        != {
            "schema_version": "catalog_spatial_leakage_audit/v1",
            "parent_cross_split_count": 0,
            "child_overlap_pair_count": 0,
        }
    ):
        raise StandardEvaluationError("shared environment contract failed")

    schedule_required = {
        "episode_index",
        "scenario_id",
        "scenario_seed",
        "terrain_seed",
        "start_pose_seed",
        "evaluation_seed",
    }
    for index, row in enumerate(schedule):
        if (
            not isinstance(row, Mapping)
            or set(row) != schedule_required
            or row["episode_index"] != index
            or not isinstance(row["scenario_id"], str)
            or not row["scenario_id"].startswith(f"{split}/")
            or any(
                type(row[name]) is not int
                for name in (
                    "scenario_seed",
                    "terrain_seed",
                    "start_pose_seed",
                    "evaluation_seed",
                )
            )
        ):
            raise StandardEvaluationError("shared environment seed schedule failed")
    if (
        len({row["scenario_id"] for row in schedule}) != episode_count
        or len({row["evaluation_seed"] for row in schedule}) != episode_count
    ):
        raise StandardEvaluationError("shared environment schedule reused identity")

    expected_factory = _qualified_name(build_standard_evaluation_env)
    for worker_index, spec in enumerate(specs):
        expected_ids = [
            row["scenario_id"]
            for row in schedule[worker_index::STANDARD_EVALUATION_WORKERS]
        ]
        if (
            not isinstance(spec, Mapping)
            or set(spec)
            != {
                "worker_index",
                "factory_source",
                "catalog_payload_sha256",
                "split",
                "scenario_ids",
                "production",
                *safety_binding,
            }
            or spec["worker_index"] != worker_index
            or spec["factory_source"] != expected_factory
            or spec["catalog_payload_sha256"] != catalog["catalog_payload_sha256"]
            or spec["split"] != split
            or spec["scenario_ids"] != expected_ids
            or spec["production"] is not True
            or any(spec[name] != item for name, item in safety_binding.items())
        ):
            raise StandardEvaluationError("shared environment lane provenance failed")


def build_standard_decision_input_schema() -> dict[str, object]:
    """冻结并扫描 `PolicyObservation` 的决策输入字段清单。"""

    metadata = PolicyObservation.schema_metadata()
    observation_fields = [field.name for field in fields(PolicyObservation)]
    scanned_fields = sorted(
        set(observation_fields) | set(_nested_field_names(metadata))
    )
    truth_like = _truth_like_field_names(scanned_fields)
    if truth_like:
        raise StandardEvaluationError("PolicyObservation schema contains truth-like fields")
    return {
        "schema_version": _DECISION_INPUT_SCHEMA_VERSION,
        "policy_observation_type": _qualified_name(PolicyObservation),
        "policy_observation_fields": observation_fields,
        "policy_observation_schema": metadata,
        "frozen_field_scan": {
            "forbidden_tokens": list(_TRUTH_LIKE_FIELD_TOKENS),
            "scanned_fields": scanned_fields,
            "scanned_field_count": len(scanned_fields),
            "truth_like_fields": truth_like,
        },
    }


def select_standard_evaluation_actions(
    *,
    method: str,
    observations: Sequence[PolicyObservation],
    evaluation_seeds: Sequence[int],
    policy_output: PolicyForwardOutput | None,
) -> tuple[EnvAction, ...]:
    rows = tuple(observations)
    seeds = tuple(evaluation_seeds)
    if (
        method not in ALL_METHODS
        or not rows
        or len(rows) != len(seeds)
        or any(not isinstance(row, PolicyObservation) for row in rows)
        or any(type(seed) is not int for seed in seeds)
    ):
        raise StandardEvaluationError("Standard evaluation action batch drifted")
    if method in BASELINE_METHODS:
        if policy_output is not None:
            raise StandardEvaluationError("baseline evaluation received policy output")
        return tuple(
            select_baseline_action(
                method,
                observation,
                np.random.Generator(np.random.PCG64(seed)),
            )
            for observation, seed in zip(rows, seeds, strict=True)
        )
    if not isinstance(policy_output, PolicyForwardOutput):
        raise StandardEvaluationError("PPO evaluation requires batched policy output")
    policy_batch = batch_policy_observations(
        rows,
        device=policy_output.frontier_logits.device,
    )
    return _select_standard_ppo_actions(
        observation_batch=policy_batch,
        policy_output=policy_output,
    )


def _select_standard_ppo_actions(
    *,
    observation_batch: PolicyBatch,
    policy_output: PolicyForwardOutput,
) -> tuple[EnvAction, ...]:
    """仅从 observation batch 与 policy output 选择确定性 PPO 动作。"""

    if not isinstance(observation_batch, PolicyBatch) or not isinstance(
        policy_output, PolicyForwardOutput
    ):
        raise StandardEvaluationError("PPO selector input types drifted")
    logits = policy_output.frontier_logits
    theta_mu = policy_output.theta_mu
    mask = observation_batch.candidate_mask
    if (
        logits.ndim != 2
        or theta_mu.shape != logits.shape
        or mask.shape != logits.shape
        or mask.dtype != torch.bool
        or mask.device != logits.device
        or theta_mu.device != logits.device
        or not bool(mask.any(dim=1).all())
        or not bool(torch.isfinite(logits).all())
        or not bool(torch.isfinite(theta_mu).all())
    ):
        raise StandardEvaluationError("PPO evaluation batch output drifted")
    masked_logits = torch.where(
        mask,
        logits,
        torch.full_like(logits, float("-inf")),
    )
    selected_indices = masked_logits.argmax(dim=1)
    actions: list[EnvAction] = []
    for row_index, raw_selected in enumerate(selected_indices):
        selected = int(raw_selected.item())
        theta = float(theta_mu[row_index, selected].item())
        if not bool(mask[row_index, selected]) or not math.isfinite(theta):
            raise StandardEvaluationError("PPO evaluation selected action drifted")
        actions.append(EnvAction(candidate_index=selected, target_theta=theta))
    return tuple(actions)


def build_standard_action_rule_provenance(method: str) -> dict[str, object]:
    """绑定 Stage 5 method rule ID 与当前实际 selector 源码。"""

    if method not in STANDARD_EVALUATION_METHODS:
        raise StandardEvaluationError("Standard action rule method drifted")
    selector = (
        _select_standard_ppo_actions
        if method == "ppo_policy"
        else select_baseline_action
    )
    provenance: dict[str, object] = {
        "schema_version": _ACTION_RULE_SCHEMA_VERSION,
        "method": method,
        "selector_source": _qualified_name(selector),
        "selector_source_sha256": _source_sha256(selector),
        **METHOD_ACTION_RULES[method],
    }
    if method == "ppo_policy":
        provenance["ppo_eval_policy_mode"] = PPO_EVAL_POLICY_MODE
    return provenance


def _array_payload_sha256(
    schema_version: str,
    arrays: Sequence[tuple[str, np.ndarray]],
) -> str:
    digest = hashlib.sha256(schema_version.encode("ascii"))
    for name, raw_array in arrays:
        array = np.ascontiguousarray(np.asarray(raw_array))
        header = {
            "name": name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }
        digest.update(ArtifactStore.canonical_json_bytes(header))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _policy_observation_sha256(observation: PolicyObservation) -> str:
    if not isinstance(observation, PolicyObservation):
        raise StandardEvaluationError("decision input is not PolicyObservation")
    return _array_payload_sha256(
        "stage6_policy_observation_value/v1",
        tuple(
            (field.name, np.asarray(getattr(observation, field.name)))
            for field in fields(PolicyObservation)
        ),
    )


def _tensor_row_numpy(value: torch.Tensor, row_index: int) -> np.ndarray:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim < 1
        or not 0 <= row_index < value.shape[0]
    ):
        raise StandardEvaluationError("policy tensor row provenance drifted")
    return value[row_index].detach().to(device="cpu").numpy().copy()


def _policy_batch_observation_sha256(
    batch: PolicyBatch,
    *,
    row_index: int,
    candidate_count: int,
) -> str:
    frontier = _tensor_row_numpy(batch.frontier_features, row_index)
    mask = _tensor_row_numpy(batch.candidate_mask, row_index)
    if (
        frontier.ndim != 2
        or mask.ndim != 1
        or frontier.shape[0] != mask.size
        or not 0 < candidate_count <= mask.size
        or bool(mask[candidate_count:].any())
        or bool(np.asarray(frontier[candidate_count:]).any())
    ):
        raise StandardEvaluationError("policy batch padding provenance drifted")
    return _array_payload_sha256(
        "stage6_policy_observation_value/v1",
        (
            ("prior_channels", _tensor_row_numpy(batch.prior_channels, row_index)),
            (
                "coverage_summary",
                _tensor_row_numpy(batch.coverage_summary, row_index),
            ),
            ("local_crop", _tensor_row_numpy(batch.local_crop, row_index)),
            ("frontier_features", frontier[:candidate_count]),
            ("pose_features", _tensor_row_numpy(batch.pose_features, row_index)),
            ("candidate_mask", mask[:candidate_count]),
        ),
    )


def _tensor_row_sha256(value: torch.Tensor, row_index: int) -> str:
    return _array_payload_sha256(
        "stage6_policy_output_row/v1",
        (("value", _tensor_row_numpy(value, row_index)),),
    )


def _candidate_mask_provenance(mask_value: np.ndarray) -> dict[str, object]:
    mask = np.asarray(mask_value)
    if mask.ndim != 1 or mask.dtype != np.bool_ or not bool(mask.any()):
        raise StandardEvaluationError("decision candidate mask is invalid")
    packed = np.packbits(mask.astype(np.uint8), bitorder="little").tobytes()
    digest = hashlib.sha256(len(mask).to_bytes(8, "big") + packed).hexdigest()
    return {
        "encoding": "numpy_packbits_little_hex/v1",
        "size": int(mask.size),
        "valid_count": int(np.count_nonzero(mask)),
        "packed_hex": packed.hex(),
        "sha256": digest,
    }


def _decode_candidate_mask(value: object) -> np.ndarray:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {"encoding", "size", "valid_count", "packed_hex", "sha256"}
        or value["encoding"] != "numpy_packbits_little_hex/v1"
        or type(value["size"]) is not int
        or value["size"] <= 0
        or type(value["valid_count"]) is not int
        or not isinstance(value["packed_hex"], str)
        or not _is_sha256(value["sha256"])
    ):
        raise StandardEvaluationError("candidate mask provenance schema drifted")
    try:
        packed = bytes.fromhex(value["packed_hex"])
    except ValueError as exc:
        raise StandardEvaluationError("candidate mask provenance encoding drifted") from exc
    expected_bytes = (value["size"] + 7) // 8
    if len(packed) != expected_bytes:
        raise StandardEvaluationError("candidate mask provenance length drifted")
    unpacked = np.unpackbits(
        np.frombuffer(packed, dtype=np.uint8),
        bitorder="little",
    )
    mask = np.asarray(unpacked[: value["size"]], dtype=bool)
    if (
        bool(unpacked[value["size"] :].any())
        or int(np.count_nonzero(mask)) != value["valid_count"]
        or not bool(mask.any())
        or hashlib.sha256(value["size"].to_bytes(8, "big") + packed).hexdigest()
        != value["sha256"]
    ):
        raise StandardEvaluationError("candidate mask provenance failed")
    return mask


def _selector_input_rows(
    method: str,
    selector_inputs: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[str]]:
    if method == "ppo_policy":
        expected = (
            ("observation_batch", PolicyBatch),
            ("policy_output", PolicyForwardOutput),
        )
    else:
        expected = (("policy_observation", PolicyObservation),)
    source_name_truth_like = _truth_like_field_names(tuple(selector_inputs))
    if source_name_truth_like:
        raise StandardEvaluationError("selector received truth-like fields")
    if tuple(selector_inputs) != tuple(name for name, _ in expected):
        raise StandardEvaluationError("selector input source set drifted")
    rows: list[dict[str, object]] = []
    scanned_names: list[str] = []
    for name, expected_type in expected:
        item = selector_inputs[name]
        if not isinstance(item, expected_type):
            raise StandardEvaluationError("selector input source type drifted")
        field_names = [field.name for field in fields(expected_type)]
        rows.append(
            {
                "name": name,
                "type": _qualified_name(expected_type),
                "field_names": field_names,
            }
        )
        scanned_names.extend((name, *field_names))
    truth_like = _truth_like_field_names(scanned_names)
    if truth_like:
        raise StandardEvaluationError("selector received truth-like fields")
    return rows, truth_like


def _circular_theta_error(left: float, right: float) -> float:
    return abs(math.atan2(math.sin(left - right), math.cos(left - right)))


def build_standard_decision_record(
    *,
    method: str,
    sequence_index: int,
    episode_index: int,
    step_index: int,
    worker_index: int,
    observation: PolicyObservation,
    action: EnvAction,
    selector_inputs: Mapping[str, object],
    policy_batch_row_index: int | None = None,
) -> dict[str, object]:
    """从实际 selector 输入与输出动作构建一条可重算决策记录。"""

    if (
        method not in STANDARD_EVALUATION_METHODS
        or type(sequence_index) is not int
        or sequence_index < 0
        or type(episode_index) is not int
        or episode_index < 0
        or type(step_index) is not int
        or step_index < 0
        or type(worker_index) is not int
        or not 0 <= worker_index < STANDARD_EVALUATION_WORKERS
        or not isinstance(observation, PolicyObservation)
        or not isinstance(action, EnvAction)
        or type(action.candidate_index) is not int
        or not math.isfinite(float(action.target_theta))
        or not isinstance(selector_inputs, Mapping)
    ):
        raise StandardEvaluationError("Standard decision provenance input drifted")

    selector_rows, truth_like = _selector_input_rows(method, selector_inputs)
    mask = np.asarray(observation.candidate_mask)
    mask_provenance = _candidate_mask_provenance(mask)
    selected = action.candidate_index
    if not 0 <= selected < mask.size or not bool(mask[selected]):
        raise StandardEvaluationError("selected candidate index is invalid")
    observation_sha256 = _policy_observation_sha256(observation)
    schema_sha256 = _canonical_sha256(build_standard_decision_input_schema())

    if method in BASELINE_METHODS:
        if policy_batch_row_index is not None:
            raise StandardEvaluationError("baseline decision received a policy batch row")
        selected_features = np.asarray(observation.frontier_features[selected])
        recommended_theta = reconstruct_recommended_theta(selected_features)
        recommended_theta_source = {
            "source": "PolicyObservation.frontier_features[selected_index]/v1",
            "frontier_feature_indices": [14, 15],
            "sin": float(selected_features[14]),
            "cos": float(selected_features[15]),
        }
        theta_error = _circular_theta_error(
            float(action.target_theta),
            recommended_theta,
        )
        if theta_error > _THETA_TOLERANCE_RAD:
            raise StandardEvaluationError("baseline theta differs from recommended theta")
        ppo_output_provenance = None
    else:
        if type(policy_batch_row_index) is not int:
            raise StandardEvaluationError("PPO decision lacks a policy batch row")
        batch = selector_inputs["observation_batch"]
        output = selector_inputs["policy_output"]
        assert isinstance(batch, PolicyBatch)
        assert isinstance(output, PolicyForwardOutput)
        batch_sha256 = _policy_batch_observation_sha256(
            batch,
            row_index=policy_batch_row_index,
            candidate_count=mask.size,
        )
        if batch_sha256 != observation_sha256:
            raise StandardEvaluationError("PolicyObservation did not bind to policy batch")
        selected_actions = _select_standard_ppo_actions(
            observation_batch=batch,
            policy_output=output,
        )
        if policy_batch_row_index >= len(selected_actions):
            raise StandardEvaluationError("PPO policy batch row is out of range")
        expected_action = selected_actions[policy_batch_row_index]
        if (
            action.candidate_index != expected_action.candidate_index
            or action.target_theta != expected_action.target_theta
        ):
            raise StandardEvaluationError("PPO action does not match policy output")
        recommended_theta = None
        recommended_theta_source = None
        theta_error = None
        ppo_output_provenance = {
            "policy_batch_observation_sha256": batch_sha256,
            "frontier_logits_sha256": _tensor_row_sha256(
                output.frontier_logits,
                policy_batch_row_index,
            ),
            "theta_mu_sha256": _tensor_row_sha256(
                output.theta_mu,
                policy_batch_row_index,
            ),
            "masked_argmax_index": expected_action.candidate_index,
            "selected_theta_mu": expected_action.target_theta,
        }

    record = {
        "schema_version": _DECISION_RECORD_SCHEMA_VERSION,
        "sequence_index": sequence_index,
        "episode_index": episode_index,
        "step_index": step_index,
        "worker_index": worker_index,
        "policy_observation_type": _qualified_name(type(observation)),
        "policy_observation_sha256": observation_sha256,
        "decision_input_schema_sha256": schema_sha256,
        "selector_inputs": selector_rows,
        "selector_truth_like_fields": truth_like,
        "candidate_mask": mask_provenance,
        "selected_index": selected,
        "target_theta": float(action.target_theta),
        "recommended_theta": recommended_theta,
        "recommended_theta_source": recommended_theta_source,
        "theta_circular_error_rad": theta_error,
        "ppo_output": ppo_output_provenance,
    }
    return {**record, "decision_sha256": _canonical_sha256(record)}


def _expected_selector_input_rows(method: str) -> list[dict[str, object]]:
    expected = (
        (
            ("observation_batch", PolicyBatch),
            ("policy_output", PolicyForwardOutput),
        )
        if method == "ppo_policy"
        else (("policy_observation", PolicyObservation),)
    )
    return [
        {
            "name": name,
            "type": _qualified_name(input_type),
            "field_names": [field.name for field in fields(input_type)],
        }
        for name, input_type in expected
    ]


def _validate_standard_decision_record(
    value: object,
    *,
    method: str,
    expected_sequence_index: int,
) -> dict[str, int]:
    required = {
        "schema_version",
        "sequence_index",
        "episode_index",
        "step_index",
        "worker_index",
        "policy_observation_type",
        "policy_observation_sha256",
        "decision_input_schema_sha256",
        "selector_inputs",
        "selector_truth_like_fields",
        "candidate_mask",
        "selected_index",
        "target_theta",
        "recommended_theta",
        "recommended_theta_source",
        "theta_circular_error_rad",
        "ppo_output",
        "decision_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or value["schema_version"] != _DECISION_RECORD_SCHEMA_VERSION
        or value["sequence_index"] != expected_sequence_index
        or type(value["episode_index"]) is not int
        or not 0 <= value["episode_index"] < 64
        or type(value["step_index"]) is not int
        or not 0 <= value["step_index"] < 128
        or type(value["worker_index"]) is not int
        or value["worker_index"] != value["episode_index"] % STANDARD_EVALUATION_WORKERS
        or value["policy_observation_type"] != _qualified_name(PolicyObservation)
        or not _is_sha256(value["policy_observation_sha256"])
        or value["decision_input_schema_sha256"]
        != _canonical_sha256(build_standard_decision_input_schema())
        or value["selector_inputs"] != _expected_selector_input_rows(method)
        or not isinstance(value["selector_truth_like_fields"], list)
        or type(value["selected_index"]) is not int
        or not math.isfinite(float(value["target_theta"]))
        or not _is_sha256(value["decision_sha256"])
    ):
        raise StandardEvaluationError("decision record schema drifted")

    scanned_names = [
        name
        for source in value["selector_inputs"]
        for name in (source["name"], *source["field_names"])
    ]
    truth_like = _truth_like_field_names(scanned_names)
    if value["selector_truth_like_fields"] != truth_like or truth_like:
        raise StandardEvaluationError("decision selector field scan failed")

    mask = _decode_candidate_mask(value["candidate_mask"])
    selected = value["selected_index"]
    if not 0 <= selected < mask.size or not bool(mask[selected]):
        raise StandardEvaluationError("decision selected index provenance failed")

    if method in BASELINE_METHODS:
        recommended = value["recommended_theta"]
        recommended_source = value["recommended_theta_source"]
        recorded_error = value["theta_circular_error_rad"]
        if (
            not isinstance(recommended, (int, float))
            or isinstance(recommended, bool)
            or not math.isfinite(float(recommended))
            or not isinstance(recorded_error, (int, float))
            or isinstance(recorded_error, bool)
            or not math.isfinite(float(recorded_error))
            or not isinstance(recommended_source, Mapping)
            or set(recommended_source)
            != {"source", "frontier_feature_indices", "sin", "cos"}
            or recommended_source["source"]
            != "PolicyObservation.frontier_features[selected_index]/v1"
            or recommended_source["frontier_feature_indices"] != [14, 15]
            or not isinstance(recommended_source["sin"], (int, float))
            or isinstance(recommended_source["sin"], bool)
            or not isinstance(recommended_source["cos"], (int, float))
            or isinstance(recommended_source["cos"], bool)
            or not math.isfinite(float(recommended_source["sin"]))
            or not math.isfinite(float(recommended_source["cos"]))
            or math.hypot(
                float(recommended_source["sin"]),
                float(recommended_source["cos"]),
            )
            <= 1.0e-12
            or value["ppo_output"] is not None
        ):
            raise StandardEvaluationError("baseline theta provenance schema drifted")
        reconstructed_recommended = math.atan2(
            float(recommended_source["sin"]),
            float(recommended_source["cos"]),
        )
        if float(recommended) != reconstructed_recommended:
            raise StandardEvaluationError("baseline recommended theta provenance failed")
        recomputed_error = _circular_theta_error(
            float(value["target_theta"]),
            reconstructed_recommended,
        )
        if (
            not math.isclose(
                float(recorded_error),
                recomputed_error,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            or recomputed_error > _THETA_TOLERANCE_RAD
        ):
            raise StandardEvaluationError("baseline theta provenance failed")
        baseline_theta_check = 1
        baseline_theta_match = 1
        ppo_input_pair = 0
    else:
        ppo = value["ppo_output"]
        if (
            value["recommended_theta"] is not None
            or value["recommended_theta_source"] is not None
            or value["theta_circular_error_rad"] is not None
            or not isinstance(ppo, Mapping)
            or set(ppo)
            != {
                "policy_batch_observation_sha256",
                "frontier_logits_sha256",
                "theta_mu_sha256",
                "masked_argmax_index",
                "selected_theta_mu",
            }
            or ppo["policy_batch_observation_sha256"]
            != value["policy_observation_sha256"]
            or not _is_sha256(ppo["frontier_logits_sha256"])
            or not _is_sha256(ppo["theta_mu_sha256"])
            or ppo["masked_argmax_index"] != selected
            or not isinstance(ppo["selected_theta_mu"], (int, float))
            or isinstance(ppo["selected_theta_mu"], bool)
            or not math.isfinite(float(ppo["selected_theta_mu"]))
            or float(ppo["selected_theta_mu"]) != float(value["target_theta"])
        ):
            raise StandardEvaluationError("PPO output provenance failed")
        baseline_theta_check = 0
        baseline_theta_match = 0
        ppo_input_pair = 1

    unsigned = {key: item for key, item in value.items() if key != "decision_sha256"}
    if value["decision_sha256"] != _canonical_sha256(unsigned):
        raise StandardEvaluationError("decision record digest drifted")
    return {
        "policy_observation_input_count": 1,
        "candidate_mask_valid_count": 1,
        "selected_index_valid_count": 1,
        "baseline_theta_check_count": baseline_theta_check,
        "baseline_theta_match_count": baseline_theta_match,
        "ppo_observation_batch_policy_output_count": ppo_input_pair,
        "selector_field_scan_count": len(scanned_names),
        "runtime_truth_like_selector_field_count": len(truth_like),
    }


def _build_standard_decision_audit(
    method: str,
    decisions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if method not in STANDARD_EVALUATION_METHODS or not isinstance(
        decisions, Sequence
    ):
        raise StandardEvaluationError("decision audit inputs drifted")
    rows = list(decisions)
    counters = {
        "policy_observation_input_count": 0,
        "candidate_mask_valid_count": 0,
        "selected_index_valid_count": 0,
        "baseline_theta_check_count": 0,
        "baseline_theta_match_count": 0,
        "ppo_observation_batch_policy_output_count": 0,
        "selector_field_scan_count": 0,
        "runtime_truth_like_selector_field_count": 0,
    }
    for sequence_index, row in enumerate(rows):
        result = _validate_standard_decision_record(
            row,
            method=method,
            expected_sequence_index=sequence_index,
        )
        for name, count in result.items():
            counters[name] += count
    frozen_truth_like = build_standard_decision_input_schema()["frozen_field_scan"][
        "truth_like_fields"
    ]
    return {
        "schema_version": _DECISION_AUDIT_SCHEMA_VERSION,
        "theta_tolerance_rad": _THETA_TOLERANCE_RAD,
        "decision_count": len(rows),
        **counters,
        "frozen_schema_truth_like_field_count": len(frozen_truth_like),
        "decisions": [dict(row) for row in rows],
    }


def _validate_explicit_standard_jobs(
    *,
    catalog: StandardScenarioCatalog,
    jobs: Sequence[StandardEvaluationJob],
    method: str,
) -> tuple[StandardEvaluationJob, ...]:
    schedule = tuple(jobs)
    if (
        not isinstance(catalog, StandardScenarioCatalog)
        or not schedule
        or any(not isinstance(job, StandardEvaluationJob) for job in schedule)
        or method not in ALL_METHODS
        or len({job.split for job in schedule}) != 1
        or not _is_supported_explicit_schedule(schedule[0].split, len(schedule))
        or tuple(job.episode_index for job in schedule)
        != tuple(range(len(schedule)))
        or len({job.scenario_id for job in schedule}) != len(schedule)
        or len({job.evaluation_seed for job in schedule}) != len(schedule)
    ):
        raise StandardEvaluationError("explicit Standard evaluation jobs drifted")
    expected_theta_source = (
        "policy_theta_mu/v1"
        if method == "ppo_policy"
        else "candidate_recommended_theta/v1"
    )
    records = {record.scenario_id: record for record in catalog.records}
    for job in schedule:
        record = records.get(job.scenario_id)
        if (
            record is None
            or record.split != job.split
            or int(record.scenario_seed_hex, 16) != job.scenario_seed
            or int(record.terrain_seed_hex, 16) != job.terrain_seed
            or int(record.start_pose_seed_hex, 16) != job.start_pose_seed
            or job.theta_source != expected_theta_source
        ):
            raise StandardEvaluationError(
                "explicit Standard evaluation job provenance drifted"
            )
    return schedule


def run_standard_evaluation_jobs(
    *,
    catalog: StandardScenarioCatalog,
    jobs: Sequence[StandardEvaluationJob],
    method: str,
    policy: nn.Module | None,
    policy_device: str | torch.device,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    safety_contract: SafetyContract,
    config_sha256: str,
    resource_guard: Callable[[str], None] | None = None,
) -> StandardEvaluationExecution:
    """Run a caller-selected 16/24/64 Standard schedule without artifact writes."""

    return _run_standard_evaluation_jobs(
        catalog=catalog,
        jobs=jobs,
        method=method,
        policy=policy,
        policy_device=policy_device,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        trace_path=None,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
        resource_guard=resource_guard,
    )


def run_standard_evaluation(
    *,
    catalog: StandardScenarioCatalog,
    split: str,
    method: str,
    episode_count: int,
    evaluation_seed_start: int,
    policy: nn.Module | None,
    policy_device: str | torch.device,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    trace_path: str | Path,
    safety_contract: SafetyContract,
    config_sha256: str,
    resource_guard: Callable[[str], None] | None = None,
) -> EvaluationSummary:
    """运行固定 Standard schedule 的并行只读评估。"""

    return _run_standard_evaluation(
        catalog=catalog,
        split=split,
        method=method,
        episode_count=episode_count,
        evaluation_seed_start=evaluation_seed_start,
        policy=policy,
        policy_device=policy_device,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        trace_path=trace_path,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
        resource_guard=resource_guard,
    )


def _attach_cleanup_secondary_note(
    primary: BaseException,
    *,
    operation: str,
    secondary: BaseException,
) -> None:
    message = str(secondary).replace("\r", "\\r").replace("\n", "\\n")
    note = (
        f"Standard evaluation cleanup secondary during {operation}: "
        f"{type(secondary).__name__}: {message or '<empty>'}"
    )
    if note not in tuple(getattr(primary, "__notes__", ())):
        primary.add_note(note)


def _run_standard_evaluation(
    *,
    catalog: StandardScenarioCatalog,
    split: str,
    method: str,
    episode_count: int,
    evaluation_seed_start: int,
    policy: nn.Module | None,
    policy_device: str | torch.device,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    trace_path: str | Path,
    safety_contract: SafetyContract,
    config_sha256: str,
    resource_guard: Callable[[str], None] | None = None,
) -> EvaluationSummary:
    theta_source = (
        "policy_theta_mu/v1"
        if method == "ppo_policy"
        else "candidate_recommended_theta/v1"
    )
    schedule = build_standard_evaluation_schedule(
        catalog,
        split=split,
        episode_count=episode_count,
        evaluation_seed_start=evaluation_seed_start,
        theta_source=theta_source,
    )
    return _run_standard_evaluation_jobs(
        catalog=catalog,
        jobs=schedule,
        method=method,
        policy=policy,
        policy_device=policy_device,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        trace_path=trace_path,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
        resource_guard=resource_guard,
    )


def _run_standard_evaluation_jobs(
    *,
    catalog: StandardScenarioCatalog,
    jobs: Sequence[StandardEvaluationJob],
    method: str,
    policy: nn.Module | None,
    policy_device: str | torch.device,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    trace_path: str | Path | None,
    safety_contract: SafetyContract,
    config_sha256: str,
    resource_guard: Callable[[str], None] | None = None,
) -> EvaluationSummary | StandardEvaluationExecution:
    schedule = _validate_explicit_standard_jobs(
        catalog=catalog,
        jobs=jobs,
        method=method,
    )
    specs = standard_evaluation_env_specs(
        catalog,
        schedule,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )
    shared_environment_contract = build_standard_environment_contract(
        catalog=catalog,
        env_specs=specs,
        jobs=schedule,
        max_steps=128,
        success_threshold=0.99,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )
    vector = SpawnVectorEnv(specs, timeout_seconds=900.0)
    try:
        result = _run_parallel_episode_batch(
            vector_env=vector,
            jobs=schedule,
            method=method,
            policy=policy,
            policy_device=policy_device,
            max_steps=128,
            success_threshold=0.99,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            trace_path=trace_path,
            shared_environment_contract=shared_environment_contract,
            safety_contract=safety_contract,
            config_sha256=config_sha256,
            resource_guard=resource_guard,
        )
    except BaseException as primary:
        try:
            vector.close()
        except BaseException as secondary:
            _attach_cleanup_secondary_note(
                primary,
                operation="vector.close",
                secondary=secondary,
            )
        raise
    vector.close()
    return result


def _policy_devices_equivalent(
    configured_device: str | torch.device,
    parameter_device: torch.device,
) -> bool:
    """比较配置设备与参数设备，并解析裸 ``cuda`` 的当前设备索引。"""

    expected = torch.device(configured_device)
    if expected.type == "cuda" and expected.index is None:
        expected = torch.device("cuda", torch.cuda.current_device())
    return expected == torch.device(parameter_device)


def _standard_episode_id(job: StandardEvaluationJob) -> str:
    return (
        f"standard:{job.split}:{job.episode_index:03d}:"
        f"{hashlib.sha256(job.scenario_id.encode('utf-8')).hexdigest()[:16]}"
    )


def _standard_step_join_key(job: StandardEvaluationJob, step_index: int) -> str:
    return hashlib.sha256(
        (
            "stage6-standard-step-join/v1:"
            f"{job.split}:{job.episode_index}:{job.scenario_id}:{step_index}"
        ).encode("utf-8")
    ).hexdigest()


def _json_mapping_copy(value: Mapping[str, object]) -> dict[str, object]:
    try:
        decoded = json.loads(ArtifactStore.canonical_json_bytes(value))
    except (TypeError, ValueError) as exc:
        raise StandardEvaluationError(
            "Standard trace diagnostics are not finite JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise StandardEvaluationError("Standard trace diagnostics drifted")
    return decoded


def _run_parallel_episode_batch(
    *,
    vector_env: SpawnVectorEnv,
    jobs: Sequence[StandardEvaluationJob],
    method: str,
    policy: nn.Module | None,
    policy_device: str | torch.device,
    max_steps: int,
    success_threshold: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    trace_path: str | Path | None,
    shared_environment_contract: Mapping[str, object],
    safety_contract: SafetyContract,
    config_sha256: str,
    resource_guard: Callable[[str], None] | None = None,
) -> EvaluationSummary | StandardEvaluationExecution:
    """在 8 个固定 worker lane 中完成一组公平评估 episode。"""

    schedule = tuple(jobs)
    if (
        not isinstance(vector_env, SpawnVectorEnv)
        or len(schedule) not in {8, 16, 24, 64}
        or any(not isinstance(job, StandardEvaluationJob) for job in schedule)
        or method not in ALL_METHODS
        or type(max_steps) is not int
        or max_steps <= 0
        or not isinstance(shared_environment_contract, Mapping)
        or (resource_guard is not None and not callable(resource_guard))
    ):
        raise StandardEvaluationError("parallel Standard episode batch drifted")
    _validate_standard_environment_contract(
        shared_environment_contract,
        episode_count=len(schedule),
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )
    contract_schedule = shared_environment_contract["scenario_seed_schedule"]
    if [
        (
            row["episode_index"],
            row["scenario_id"],
            row["scenario_seed"],
            row["terrain_seed"],
            row["start_pose_seed"],
            row["evaluation_seed"],
        )
        for row in contract_schedule
    ] != [
        (
            job.episode_index,
            job.scenario_id,
            job.scenario_seed,
            job.terrain_seed,
            job.start_pose_seed,
            job.evaluation_seed,
        )
        for job in schedule
    ]:
        raise StandardEvaluationError("parallel schedule differs from shared contract")
    device = torch.device(policy_device)
    if method == "ppo_policy":
        if not isinstance(policy, nn.Module):
            raise StandardEvaluationError("PPO evaluation requires a policy")
        try:
            parameter_device = next(policy.parameters()).device
        except StopIteration as exc:
            raise StandardEvaluationError("PPO evaluation policy has no parameters") from exc
        if not _policy_devices_equivalent(device, parameter_device):
            raise StandardEvaluationError("PPO evaluation policy device drifted")
    elif policy is not None:
        raise StandardEvaluationError("baseline evaluation must not receive a policy")

    original_training = policy.training if policy is not None else None
    initial_policy_hash = policy_state_sha256(policy) if policy is not None else None
    worker_pids = vector_env.worker_pids
    worker_start_methods = vector_env.worker_start_methods
    inference_pids: set[int] = set()
    selected_action_count = 0
    episode_action_counts: dict[int, int] = {}
    states: dict[int, dict[str, object]] = {}
    episodes: dict[int, EpisodeResult] = {}
    episode_diagnostics: dict[int, dict[str, object]] = {}
    decision_records: list[dict[str, object]] = []
    trace_decision_rows: list[dict[str, object]] = []
    step_rows: list[dict[str, object]] = []
    pending_step_rows: dict[int, dict[str, object]] = {}
    lane_positions = tuple(
        tuple(range(worker_index, len(schedule), STANDARD_EVALUATION_WORKERS))
        for worker_index in range(STANDARD_EVALUATION_WORKERS)
    )
    lane_cursors = [0] * STANDARD_EVALUATION_WORKERS
    rngs = {
        index: np.random.Generator(np.random.PCG64(job.evaluation_seed))
        for index, job in enumerate(schedule)
    }

    def finish(worker_index: int, reason: str) -> None:
        state = states[worker_index]
        coverage = list(state["coverage"])
        path = list(state["path"])
        steps_executed = len(coverage) - 1
        if steps_executed > max_steps:
            raise StandardEvaluationError("Standard evaluation exceeded max_steps")
        coverage.extend([coverage[-1]] * (max_steps + 1 - len(coverage)))
        path.extend([path[-1]] * (max_steps + 1 - len(path)))
        job_index = int(state["job_index"])
        episode_action_counts[job_index] = steps_executed
        job = schedule[job_index]
        episodes[job_index] = build_episode_result(
            method=method,
            scale_profile="Standard v1",
            scenario_key=job.scenario_id,
            scenario_seed=job.scenario_seed,
            terrain_seed=job.terrain_seed,
            start_pose_seed=job.start_pose_seed,
            evaluation_seed=job.evaluation_seed,
            coverage_curve=coverage,
            cumulative_path_length_curve=path,
            steps_executed=steps_executed,
            max_steps=max_steps,
            success_threshold=success_threshold,
            zero_distance_policy=ZERO_DISTANCE_POLICY,
            invalid_action_count=int(state["invalid_action_count"]),
            planner_failure_count=int(state["planner_failure_count"]),
            safety_violation_count=int(state["safety_violation_count"]),
            termination_reason=reason,
        )
        failure_counts = dict(state["planner_failure_counts"])
        if (
            set(failure_counts) != set(PLANNER_FAILURE_REASONS)
            or sum(int(value) for value in failure_counts.values())
            != int(state["planner_failure_count"])
        ):
            raise StandardEvaluationError(
                "planner failure reason counts drifted"
            )
        episode_diagnostics[job_index] = {
            "reset_diagnostics": dict(state["reset_diagnostics"]),
            "planner_failure_counts": failure_counts,
        }

    def guarded_reset(
        worker_indices: Sequence[int] | None = None,
        *,
        episode_reset: bool,
    ):
        before_boundary = (
            "evaluation:before-episode-reset"
            if episode_reset
            else "evaluation:before-reset"
        )
        after_boundary = (
            "evaluation:after-episode-reset"
            if episode_reset
            else "evaluation:after-reset"
        )
        if resource_guard is not None:
            resource_guard(before_boundary)
        reset_values = vector_env.reset(worker_indices)
        if resource_guard is not None:
            resource_guard(after_boundary)
        return reset_values

    try:
        if policy is not None:
            policy.eval()
        current = guarded_reset(episode_reset=False)
        active: set[int] = set()

        def start_worker(worker_index: int, envelope) -> None:
            while True:
                lane = lane_positions[worker_index]
                cursor = lane_cursors[worker_index]
                if cursor >= len(lane):
                    return
                job_index = lane[cursor]
                try:
                    reset_diagnostics = validate_reset_diagnostics_payload(
                        envelope.reset_diagnostics,
                        require_dual_scan=True,
                    )
                except ValueError as exc:
                    raise StandardEvaluationError(
                        "evaluation reset diagnostics drifted"
                    ) from exc
                states[worker_index] = {
                    "job_index": job_index,
                    "initial_coverage_rate": float(envelope.reset_coverage_rate),
                    "coverage": [float(envelope.reset_coverage_rate)],
                    "path": [0.0],
                    "invalid_action_count": 0,
                    "planner_failure_count": 0,
                    "planner_failure_counts": {
                        reason: 0 for reason in PLANNER_FAILURE_REASONS
                    },
                    "safety_violation_count": 0,
                    "reset_diagnostics": reset_diagnostics,
                }
                current[worker_index] = envelope
                if envelope.needs_policy:
                    active.add(worker_index)
                    return
                if not envelope.reset_done:
                    raise StandardEvaluationError(
                        "reset has neither action nor terminal"
                    )
                finish(worker_index, envelope.reset_reason)
                lane_cursors[worker_index] += 1
                if lane_cursors[worker_index] >= len(lane):
                    return
                envelope = guarded_reset(
                    (worker_index,),
                    episode_reset=True,
                )[worker_index]

        for worker_index in range(STANDARD_EVALUATION_WORKERS):
            start_worker(worker_index, current[worker_index])

        while active:
            indices = tuple(sorted(active))
            observations = tuple(current[index].observation for index in indices)
            if resource_guard is not None:
                resource_guard("evaluation:before-inference")
            if method == "ppo_policy":
                assert policy is not None
                batch = batch_policy_observations(observations, device=device)
                with torch.inference_mode():
                    output = policy(batch)
                inference_pids.add(os.getpid())
                selected = _select_standard_ppo_actions(
                    observation_batch=batch,
                    policy_output=output,
                )
                selector_inputs: Mapping[str, object] = {
                    "observation_batch": batch,
                    "policy_output": output,
                }
            else:
                selected = tuple(
                    select_baseline_action(
                        method,
                        current[index].observation,
                        rngs[int(states[index]["job_index"])],
                    )
                    for index in indices
                )
                selector_inputs = {}
            if resource_guard is not None:
                resource_guard("evaluation:after-inference")
            for row_index, (worker_index, observation, action) in enumerate(
                zip(indices, observations, selected, strict=True)
            ):
                state = states[worker_index]
                job_index = int(state["job_index"])
                job = schedule[job_index]
                step_index = len(state["coverage"]) - 1
                decision = build_standard_decision_record(
                    method=method,
                    sequence_index=len(decision_records),
                    episode_index=job_index,
                    step_index=step_index,
                    worker_index=worker_index,
                    observation=observation,
                    action=action,
                    selector_inputs=(
                        selector_inputs
                        if method == "ppo_policy"
                        else {"policy_observation": observation}
                    ),
                    policy_batch_row_index=(
                        row_index if method == "ppo_policy" else None
                    ),
                )
                decision_records.append(decision)
                valid_indices = tuple(
                    int(value)
                    for value in np.flatnonzero(
                        np.asarray(observation.candidate_mask, dtype=bool)
                    )
                )
                try:
                    selected_rank = valid_indices.index(action.candidate_index)
                    selected_cell = current[worker_index].frontier_cells[
                        selected_rank
                    ]
                except (IndexError, ValueError) as exc:
                    raise StandardEvaluationError(
                        "selected Standard trace candidate drifted"
                    ) from exc
                join_key = _standard_step_join_key(job, step_index)
                episode_id = _standard_episode_id(job)
                trace_decision = {
                    **decision,
                    "join_key": join_key,
                    "episode_id": episode_id,
                    "scenario_id": job.scenario_id,
                    "lane_id": f"lane-{worker_index}",
                }
                trace_decision_rows.append(trace_decision)
                pending_step_rows[worker_index] = {
                    "schema_version": "stage6-standard-step-trace/v1",
                    "join_key": join_key,
                    "episode_id": episode_id,
                    "episode_index": job_index,
                    "scenario_id": job.scenario_id,
                    "lane_id": f"lane-{worker_index}",
                    "step_index": step_index,
                    "decision_sha256": decision["decision_sha256"],
                    "pre_observation_sha256": decision[
                        "policy_observation_sha256"
                    ],
                    "selected_candidate_index": action.candidate_index,
                    "selected_candidate_cell_xy": [
                        int(selected_cell.x),
                        int(selected_cell.y),
                    ],
                    "selected_theta": float(action.target_theta),
                }
            selected_action_count += len(selected)
            if resource_guard is not None:
                resource_guard("evaluation:before-step")
            steps = vector_env.step(dict(zip(indices, selected, strict=True)))
            if resource_guard is not None:
                resource_guard("evaluation:after-step")
            for worker_index in indices:
                step = steps[worker_index]
                state = states[worker_index]
                previous_coverage = float(state["coverage"][-1])
                state["coverage"].append(float(step.result.coverage_rate))
                path_length = float(state["path"][-1])
                executed_path_length = 0.0
                if step.result.diagnostics.execution is not None:
                    executed_path_length = float(
                        step.result.diagnostics.path_length_m
                    )
                    path_length += executed_path_length
                state["path"].append(path_length)
                state["invalid_action_count"] = int(state["invalid_action_count"]) + int(
                    step.result.diagnostics.invalid_action
                )
                failure_reason = step.result.diagnostics.planner.get("failure_reason")
                if failure_reason != "none":
                    if (
                        not isinstance(failure_reason, str)
                        or failure_reason not in PLANNER_FAILURE_REASONS
                    ):
                        raise StandardEvaluationError(
                            "planner failure reason is unknown"
                        )
                    state["planner_failure_counts"][failure_reason] = (  # type: ignore[index]
                        int(state["planner_failure_counts"][failure_reason])  # type: ignore[index]
                        + 1
                    )
                state["planner_failure_count"] = int(
                    state["planner_failure_count"]
                ) + int(failure_reason not in (None, "none"))
                state["safety_violation_count"] = int(
                    state["safety_violation_count"]
                ) + int(step.result.diagnostics.safety_violation)
                try:
                    trace_prefix = pending_step_rows.pop(worker_index)
                except KeyError as exc:
                    raise StandardEvaluationError(
                        "Standard step trace lacks its decision"
                    ) from exc
                planned_path = [
                    [int(cell.x), int(cell.y)]
                    for cell in step.result.diagnostics.planned_path_cells
                ]
                step_rows.append(
                    {
                        **trace_prefix,
                        "post_observation_sha256": _policy_observation_sha256(
                            step.next_observation.observation
                        ),
                        "planned_path_cells": planned_path,
                        "path_length_m": executed_path_length,
                        "planner_path_length_m": float(
                            step.result.diagnostics.path_length_m
                        ),
                        "cumulative_path_length_m": path_length,
                        "coverage_gain_cells": int(
                            step.result.coverage_gain_cells
                        ),
                        "coverage_gain_rate": (
                            float(step.result.coverage_rate)
                            - previous_coverage
                        ),
                        "coverage_rate": float(step.result.coverage_rate),
                        "done": bool(step.result.done),
                        "termination_reason": (
                            str(step.result.reason)
                            if step.result.done
                            else "none"
                        ),
                        "invalid_action": bool(
                            step.result.diagnostics.invalid_action
                        ),
                        "safety_violation": bool(
                            step.result.diagnostics.safety_violation
                        ),
                        "planner_diagnostics": _json_mapping_copy(
                            step.result.diagnostics.planner
                        ),
                    }
                )
                if step.result.done:
                    finish(worker_index, step.result.reason)
                    active.remove(worker_index)
                    lane_cursors[worker_index] += 1
                    if lane_cursors[worker_index] < len(lane_positions[worker_index]):
                        start_worker(
                            worker_index,
                            guarded_reset(
                                (worker_index,),
                                episode_reset=True,
                            )[worker_index],
                        )
                elif len(state["coverage"]) - 1 >= max_steps:
                    raise StandardEvaluationError(
                        "environment did not terminate at fixed step budget"
                    )
                elif not step.next_observation.needs_policy:
                    raise StandardEvaluationError("nonterminal evaluation lacks action")
                else:
                    current[worker_index] = step.next_observation
    except BaseException as primary:
        if policy is not None and original_training is not None:
            try:
                policy.train(original_training)
            except BaseException as secondary:
                _attach_cleanup_secondary_note(
                    primary,
                    operation="policy mode restoration",
                    secondary=secondary,
                )
        try:
            vector_env.close()
        except BaseException as secondary:
            _attach_cleanup_secondary_note(
                primary,
                operation="vector.close",
                secondary=secondary,
            )
        raise
    else:
        if policy is not None and original_training is not None:
            try:
                policy.train(original_training)
            except BaseException as primary:
                try:
                    vector_env.close()
                except BaseException as secondary:
                    _attach_cleanup_secondary_note(
                        primary,
                        operation="vector.close",
                        secondary=secondary,
                    )
                raise
        vector_env.close()

    ordered = tuple(episodes[index] for index in range(len(schedule)))
    final_policy_hash = policy_state_sha256(policy) if policy is not None else None
    if final_policy_hash != initial_policy_hash:
        raise StandardEvaluationError("evaluation mutated policy state")
    metrics, bootstrap = summarize_episodes(
        ordered,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    ordered_action_counts = [episode_action_counts[index] for index in range(len(schedule))]
    if (
        pending_step_rows
        or len(trace_decision_rows) != selected_action_count
        or len(step_rows) != selected_action_count
        or len({row["join_key"] for row in trace_decision_rows})
        != selected_action_count
        or {row["join_key"] for row in trace_decision_rows}
        != {row["join_key"] for row in step_rows}
    ):
        raise StandardEvaluationError("Standard in-memory trace join drifted")
    legacy_episode_rows = tuple(
        {
            **episode_record(episode),
            "steps_executed": ordered_action_counts[index],
            **episode_diagnostics[index],
        }
        for index, episode in enumerate(ordered)
    )
    if trace_path is not None:
        path = Path(trace_path)
        if path.exists():
            raise StandardEvaluationError("evaluation trace already exists")
        trace = DurableJsonl(path)
        for row in legacy_episode_rows:
            trace.append(row)
    decision_input_schema = build_standard_decision_input_schema()
    action_rule = build_standard_action_rule_provenance(method)
    fairness = validate_standard_fairness_audit({
        "schema_version": _FAIRNESS_SCHEMA_VERSION,
        "method": method,
        "split": schedule[0].split,
        "worker_count": STANDARD_EVALUATION_WORKERS,
        "worker_pids": list(worker_pids),
        "worker_start_methods": list(worker_start_methods),
        "evaluation_parent_pid": os.getpid(),
        "inference_pids": sorted(inference_pids),
        "policy_state_unchanged": final_policy_hash == initial_policy_hash,
        "policy_state_sha256_before": initial_policy_hash,
        "policy_state_sha256_after": final_policy_hash,
        "episode_action_counts": ordered_action_counts,
        "selected_action_count": selected_action_count,
        "scenario_schedule": [job.scenario_id for job in schedule],
        "evaluation_seeds": [job.evaluation_seed for job in schedule],
        "theta_source": schedule[0].theta_source,
        "shared_environment_contract": dict(shared_environment_contract),
        "shared_environment_contract_sha256": _canonical_sha256(
            shared_environment_contract
        ),
        "decision_input_schema": decision_input_schema,
        "decision_input_schema_sha256": _canonical_sha256(decision_input_schema),
        "action_rule": action_rule,
        "action_rule_sha256": _canonical_sha256(action_rule),
        "decision_audit": _build_standard_decision_audit(
            method,
            decision_records,
        ),
    },
        episode_count=len(schedule),
        parent_pid=os.getpid(),
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )
    summary = EvaluationSummary(
        method=method,
        scale_profile="Standard v1",
        episodes=ordered,
        metrics=metrics,
        bootstrap_audit=bootstrap,
        fairness_audit=fairness,
    )
    execution = StandardEvaluationExecution(
        summary=summary,
        episode_rows=tuple(
            {
                **row,
                "episode_id": _standard_episode_id(schedule[index]),
                "episode_index": index,
                "scenario_id": schedule[index].scenario_id,
                "lane_id": f"lane-{index % STANDARD_EVALUATION_WORKERS}",
                "initial_coverage_rate": float(
                    ordered[index].coverage_curve[0]
                ),
            }
            for index, row in enumerate(legacy_episode_rows)
        ),
        decision_rows=tuple(trace_decision_rows),
        step_rows=tuple(step_rows),
    )
    return execution if trace_path is None else summary


__all__ = [
    "STANDARD_EVALUATION_METHODS",
    "STANDARD_EVALUATION_WORKERS",
    "StandardEvaluationError",
    "StandardEvaluationExecution",
    "StandardEvaluationJob",
    "build_standard_action_rule_provenance",
    "build_standard_decision_input_schema",
    "build_standard_decision_record",
    "build_standard_environment_contract",
    "build_standard_evaluation_schedule",
    "build_standard_final_method_schedules",
    "partition_standard_evaluation_jobs",
    "run_standard_evaluation",
    "run_standard_evaluation_jobs",
    "select_standard_evaluation_actions",
    "standard_evaluation_env_specs",
    "validate_standard_fairness_audit",
    "validate_standard_fairness_cohort",
]
