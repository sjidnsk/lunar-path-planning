from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODEL_EXPLORER_SRC = REPO_ROOT / "model-explorer" / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if MODEL_EXPLORER_SRC.is_dir() and str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))

from model_explorer.policy.canonical_reward import compute_canonical_reward_components, load_canonical_reward_profile

import run_xunce_high_fidelity_exploration_coverage_comparison as hf
import run_xunce_high_fidelity_real_map_comparison as real_map
from xunce_obstacle_aware_theta_sensor_coverage import obstacle_aware_theta_coverage_hash
from xunce_hybrid_astar_candidate_path_cost import (
    PATH_COST_SOURCE as HYBRID_ASTAR_PATH_COST_SOURCE,
    Cell as HYBRID_CELL,
    WorldPoint as HYBRID_WORLD_POINT,
    build_cost_grid_from_sidecar,
    evaluate_hybrid_astar_candidate_path_cost,
)
from xunce_continuous_theta_action import (
    CONTINUOUS_THETA_ACTION_SPACE,
    action_sample_hash,
    continuous_theta_enabled,
    sample_continuous_theta_action,
)
from xunce_theta_viewpoint_candidates import candidate_observation_cells, theta_metadata


CONFIG_SCHEMA_VERSION = "xunce-stage21-1-on-policy-ppo-rollout-collector-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage21-1-on-policy-ppo-rollout-collector-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage21-1-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage21-1-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage21_1_on_policy_ppo_rollout_collector_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/"
    "outputs/path_feedback_batch_xunce_stage21_1_on_policy_ppo_rollout_collector_v1"
)

SUMMARY_FILE = "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json"
EPISODES_FILE = "xunce-stage21-1-ppo-rollout-episodes.jsonl"
TRANSITIONS_FILE = "xunce-stage21-1-ppo-rollout-transitions.jsonl"
TRAINABLE_BATCH_FILE = "xunce-stage21-1-ppo-trainable-batch.jsonl"
REJECTION_FILE = "xunce-stage21-1-rejection-report.jsonl"
REWARD_AUDIT_FILE = "xunce-stage21-1-reward-audit.jsonl"
SAMPLING_AUDIT_FILE = "xunce-stage21-1-sampling-audit.jsonl"
ROUTING_FILE = "xunce-stage21-1-next-stage-routing.json"
REPORT_FILE = "xunce-stage21-1-report.md"
MANIFEST_FILE = "xunce-stage21-1-manifest.json"

ROUTE_BOUNDARY = "resolve_stage21_1_on_policy_collector_boundary_rejections"
ROUTE_INPUTS = "repair_stage21_1_on_policy_collector_inputs"
ROUTE_EXPAND_COLLECTION = "expand_stage21_1_on_policy_rollout_collection"
ROUTE_REPAIR_COLLECTOR = "repair_stage21_1_on_policy_collector_contract"
ROUTE_STAGE21_2 = "implement_stage21_2_coverage_first_ppo_reward_contract"

BOUNDARY_FIELDS = (
    "stage21_1_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)

TOLERANCE = 1.0e-12


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CollectionResult:
    episodes: list[dict[str, Any]]
    transitions: list[dict[str, Any]]
    trainable_batch: list[dict[str, Any]]
    rejections: list[dict[str, Any]]
    reward_audit: list[dict[str, Any]]
    sampling_audit: list[dict[str, Any]]
    model_audit: dict[str, Any]
    reason_codes: list[str]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 21.1 Xunce on-policy PPO rollout collector.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    try:
        summary = run_xunce_stage21_1_on_policy_ppo_rollout_collector(
            config_path=Path(args.config),
            output_root=Path(args.output_root),
            repo_root=Path(args.repo_root),
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": summary["status"],
                "next_required_change": summary["next_required_change"],
                "trainable_transition_count": summary["trainable_transition_count"],
                "sampled_transition_count": summary["sampled_transition_count"],
                "stage21_1_authorized": summary["stage21_1_authorized"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage21_1_on_policy_ppo_rollout_collector(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root=repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    profile = load_canonical_reward_profile(Path(config["canonical_reward_profile"]))
    boundary_reasons = _boundary_rejections(config)
    input_reasons = _input_rejections(config, profile)

    if boundary_reasons or input_reasons:
        empty = CollectionResult([], [], [], [], [], [], {}, boundary_reasons + input_reasons)
        return _write_outputs(
            config=config,
            profile=profile,
            output_root=output_root,
            config_path=_resolve_path(config_path, repo_root),
            collection=empty,
            blocking_reason_codes=boundary_reasons + input_reasons,
            route=ROUTE_BOUNDARY if boundary_reasons else ROUTE_INPUTS,
        )

    hf_config = _load_high_fidelity_config(config, repo_root=repo_root)
    collection = _collect_rollouts(
        config=config,
        hf_config=hf_config,
        profile=profile,
        repo_root=repo_root,
        output_root=output_root,
    )
    counts = _contract_counts(collection)
    non_blocking_reason_codes = set()
    if continuous_theta_enabled(config):
        non_blocking_reason_codes.add("selected_continuous_theta_hybrid_astar_unreachable")
    blocking = [reason for reason in collection.reason_codes if reason not in non_blocking_reason_codes]
    if counts["trainable_transition_count"] < int(config["min_trainable_transition_count"]):
        blocking.append("trainable_transition_count_below_minimum")
    if counts["mask_violation_count"] > 0:
        blocking.append("mask_violation_detected")
    if counts["hard_risk_violation_count"] > 0:
        blocking.append("hard_risk_violation_detected")
    if counts["non_finite_old_log_prob_count"] > 0:
        blocking.append("non_finite_old_log_prob_detected")
    if counts["non_finite_old_value_count"] > 0:
        blocking.append("non_finite_old_value_detected")
    if counts["non_finite_reward_count"] > 0:
        blocking.append("non_finite_reward_detected")
    if counts["old_log_prob_recompute_max_abs_error"] > float(config["max_log_prob_recompute_abs_error"]):
        blocking.append("old_log_prob_recompute_error_exceeded")

    route = ROUTE_STAGE21_2
    if blocking:
        route = ROUTE_EXPAND_COLLECTION if "trainable_transition_count_below_minimum" in blocking else ROUTE_REPAIR_COLLECTOR

    return _write_outputs(
        config=config,
        profile=profile,
        output_root=output_root,
        config_path=_resolve_path(config_path, repo_root),
        collection=collection,
        blocking_reason_codes=blocking,
        route=route,
    )


def _collect_rollouts(
    *,
    config: dict[str, Any],
    hf_config: dict[str, Any],
    profile: Any,
    repo_root: Path,
    output_root: Path,
) -> CollectionResult:
    torch.manual_seed(int(config["sampling_seed"]))
    hf_config.setdefault("continuous_theta_head_init_seed", int(config["sampling_seed"]))
    source = hf._load_source(hf_config, repo_root)
    obstacle_source_audit = hf._obstacle_source_audit(source, hf_config, repo_root=repo_root)
    obstacle_source_by_scenario = obstacle_source_audit.get("source_by_scenario", {})
    checkpoint_audit, model, xunce_config = real_map._load_xunce_checkpoint(
        Path(source["xunce_checkpoint"]),
        config=hf_config,
        repo_root=repo_root,
    )
    if model is None or not bool(checkpoint_audit.get("checkpoint_loaded")):
        return CollectionResult(
            [],
            [],
            [],
            [{"reason": "xunce_checkpoint_not_loaded", "checkpoint": str(source["xunce_checkpoint"])}],
            [],
            [],
            {"xunce_checkpoint_audit": checkpoint_audit},
            ["xunce_checkpoint_not_loaded"],
        )
    model.eval()

    scenarios = source["path_feedback"].get("scenarios", [])
    if not isinstance(scenarios, list):
        scenarios = []
    slice_by_id = {str(row.get("scenario_id")): row for row in source["slices"] if isinstance(row, dict)}

    episodes: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    trainable_batch: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    reward_audit: list[dict[str, Any]] = []
    sampling_audit: list[dict[str, Any]] = []
    validation_cache: dict[str, list[dict[str, Any]]] = {}
    reason_codes: list[str] = list(source.get("read_reason_codes") or [])

    for scenario_index, scenario in enumerate(scenarios[: int(hf_config["required_scenario_count"])]):
        if not isinstance(scenario, dict):
            continue
        episode = _collect_episode(
            scenario=scenario,
            scenario_index=scenario_index,
            source=source,
            slice_by_id=slice_by_id,
            obstacle_source_linkage=obstacle_source_by_scenario.get(str(scenario.get("scenario_id", f"scenario-{scenario_index:04d}"))),
            hf_config=hf_config,
            config=config,
            profile=profile,
            model=model,
            xunce_config=xunce_config,
            repo_root=repo_root,
            output_root=output_root,
            validation_cache=validation_cache,
        )
        episodes.append(episode["episode"])
        transitions.extend(episode["transitions"])
        trainable_batch.extend(episode["trainable_batch"])
        rejections.extend(episode["rejections"])
        reward_audit.extend(episode["reward_audit"])
        sampling_audit.extend(episode["sampling_audit"])
        reason_codes.extend(episode["reason_codes"])

    model_audit = {
        "xunce_checkpoint_audit": checkpoint_audit,
        "xunce_config": dict(xunce_config),
        "source_roi_expansion_root": hf_config["source_roi_expansion_root"],
        "xunce_candidate_checkpoint": str(source["xunce_checkpoint"]),
        "hybrid_astar_candidate_eval_workers": int(hf_config.get("hybrid_astar_candidate_eval_workers", 1)),
    }
    return CollectionResult(
        episodes=episodes,
        transitions=transitions,
        trainable_batch=trainable_batch,
        rejections=rejections,
        reward_audit=reward_audit,
        sampling_audit=sampling_audit,
        model_audit=model_audit,
        reason_codes=_unique(reason_codes),
    )


def _collect_episode(
    *,
    scenario: dict[str, Any],
    scenario_index: int,
    source: dict[str, Any],
    slice_by_id: dict[str, dict[str, Any]],
    obstacle_source_linkage: dict[str, Any] | None,
    hf_config: dict[str, Any],
    config: dict[str, Any],
    profile: Any,
    model: Any,
    xunce_config: dict[str, int],
    repo_root: Path,
    output_root: Path,
    validation_cache: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    scenario_id = str(scenario.get("scenario_id", f"scenario-{scenario_index:04d}"))
    slice_row = slice_by_id.get(scenario_id, {})
    roi_group = hf.resolve_roi_group(scenario, slice_row)
    denominator_context = hf.resolve_coverage_denominator(scenario, slice_row, hf_config, repo_root)
    denominator = float(denominator_context["legacy_coverage_denominator_cells"])
    radius = int(hf_config["coverage_radius_cells"])
    start_cell = hf._cell_tuple(scenario.get("start_cell")) or (0, 0)
    covered_cells = set(hf._footprint(start_cell, radius=radius))
    current_cell = start_cell
    current_theta_deg = float(config.get("initial_theta_deg", 0.0))
    coverage_rates = [len(covered_cells) / denominator]
    path_cost_total = 0.0
    soft_risk_exposure_total = 0.0
    hard_risk_violation_count = 0

    transitions: list[dict[str, Any]] = []
    trainable_batch: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    reward_audit: list[dict[str, Any]] = []
    sampling_audit: list[dict[str, Any]] = []
    reason_codes: list[str] = []
    pending: dict[str, Any] | None = None
    executed_step_count = 0
    terminal_reason: str | None = None

    for step_index in range(int(hf_config["rollout_steps"])):
        cell_before = current_cell
        candidate_batch = hf._candidate_rows_for_step(
            scenario,
            current_cell=current_cell,
            current_theta_deg=current_theta_deg,
            covered_cells=covered_cells,
            step_index=step_index,
            config=hf_config,
            scenario_id=scenario_id,
            slice_row=slice_row,
            repo_root=repo_root,
            output_root=output_root,
            validation_cache=validation_cache,
        )
        candidates = candidate_batch["candidates"]
        candidate_set_hash_value = candidate_batch["candidate_set_hash"]
        candidate_set_id = candidate_batch["candidate_set_id"]
        covered_hash = hf.covered_cells_hash(covered_cells)
        if not candidates:
            terminal_reason = "candidate_generation_exhausted"
            reason_codes.append(terminal_reason)
            _finalize_pending(
                pending,
                transitions,
                trainable_batch,
                done=True,
                next_observation=None,
                next_xunce_batch=None,
            )
            pending = None
            rejections.append(
                _rejection_row(
                    scenario_id,
                    step_index,
                    "candidate_generation_exhausted",
                    candidate_set_hash_value=candidate_set_hash_value,
                    covered_cells_hash_value=covered_hash,
                )
            )
            break

        scenario_state = dict(scenario)
        scenario_state["coverage_rate"] = coverage_rates[-1]
        scenario_state["coverage_rate_delta"] = transitions[-1]["info"]["coverage_rate_delta"] if transitions else 0.0
        adapter = hf._scenario_to_model_inputs(
            scenario_state,
            candidates,
            scenario_index + step_index,
            xunce_config,
        )
        observation_payload = _observation_to_dict(adapter["incumbent_observation"])
        xunce_batch_payload = _xunce_batch_to_dict(adapter["xunce_batch"])
        observation_payload["candidate_cells"] = candidate_observation_cells(candidates)
        observation_payload.update(theta_metadata(candidates))
        slope_theta_metadata = _slope_obstacle_theta_metadata(
            candidates,
            current_cell=cell_before,
            covered_cells=covered_cells,
            config=hf_config,
            obstacle_source_linkage=obstacle_source_linkage,
        )
        observation_payload.update(slope_theta_metadata)
        synthetic_terrain_metadata = _synthetic_terrain_metadata(
            slice_row,
            config=hf_config,
            obstacle_source_linkage=obstacle_source_linkage,
        )
        observation_payload.update(synthetic_terrain_metadata)
        hybrid_path_metadata = (
            {}
            if continuous_theta_enabled(config)
            else _hybrid_astar_path_cost_metadata(
                candidates,
                current_cell=cell_before,
                current_theta_deg=current_theta_deg,
                candidate_set_hash_value=candidate_set_hash_value,
                config=hf_config,
                slice_row=slice_row,
                platform_contract_hash=slope_theta_metadata.get("platform_contract_hash"),
            )
        )
        continuous_hybrid_probe_metadata = (
            _continuous_theta_reachability_probe_metadata(
                candidates,
                current_cell=cell_before,
                current_theta_deg=current_theta_deg,
                candidate_set_hash_value=candidate_set_hash_value,
                config=hf_config,
                slice_row=slice_row,
                platform_contract_hash=slope_theta_metadata.get("platform_contract_hash"),
            )
            if continuous_theta_enabled(config)
            else {}
        )
        observation_payload.update(hybrid_path_metadata)
        action_mask = tuple(bool(value) for value in adapter["action_mask"])
        hard_risk_clean_mask = _hard_risk_clean_mask(
            candidates,
            allow_open_grid_fallback=bool(hf_config["allow_open_grid_fallback"]),
        )
        hybrid_reachable_mask = (
            _hybrid_astar_reachable_mask(continuous_hybrid_probe_metadata, candidate_count=len(candidates))
            if continuous_theta_enabled(config)
            else _hybrid_astar_reachable_mask(hybrid_path_metadata, candidate_count=len(candidates))
        )
        sampling_mask = tuple(
            bool(valid) and bool(clean) and bool(hybrid_reachable)
            for valid, clean, hybrid_reachable in zip(action_mask, hard_risk_clean_mask, hybrid_reachable_mask)
        )

        if pending is not None:
            _finalize_pending(
                pending,
                transitions,
                trainable_batch,
                done=False,
                next_observation=observation_payload,
                next_xunce_batch=xunce_batch_payload,
            )
            pending = None

        if not any(sampling_mask):
            terminal_reason = "no_hard_risk_clean_action"
            reason_codes.append(terminal_reason)
            rejections.append(
                _rejection_row(
                    scenario_id,
                    step_index,
                    terminal_reason,
                    candidate_set_hash_value=candidate_set_hash_value,
                    covered_cells_hash_value=covered_hash,
                    action_mask=list(action_mask),
                    hard_risk_clean_mask=list(hard_risk_clean_mask),
                    sampling_mask=list(sampling_mask),
                )
            )
            break

        try:
            detail = _sample_xunce_action(
                model,
                adapter["xunce_batch"],
                sampling_mask=sampling_mask,
                temperature=float(config["sampling_temperature"]),
                continuous_theta_action_space_enabled=continuous_theta_enabled(config),
            )
        except Exception as exc:  # pragma: no cover
            terminal_reason = f"model_sampling_failure:{type(exc).__name__}"
            reason_codes.append("model_sampling_failure")
            rejections.append(
                _rejection_row(
                    scenario_id,
                    step_index,
                    terminal_reason,
                    candidate_set_hash_value=candidate_set_hash_value,
                    covered_cells_hash_value=covered_hash,
                )
            )
            break

        selected_index = detail["action_index"]
        selected_candidate = hf._candidate_at(candidates, selected_index)
        if selected_candidate is not None and continuous_theta_enabled(config):
            selected_candidate = _selected_continuous_theta_candidate(
                selected_candidate,
                selected_index=selected_index,
                detail=detail,
                candidate_set_hash_value=candidate_set_hash_value,
                sampling_seed=int(config["sampling_seed"]),
            )
            candidates[selected_index] = selected_candidate
            slope_theta_metadata = _slope_obstacle_theta_metadata(
                candidates,
                current_cell=cell_before,
                covered_cells=covered_cells,
                config=hf_config,
                obstacle_source_linkage=obstacle_source_linkage,
            )
            observation_payload.update(slope_theta_metadata)
            synthetic_terrain_metadata = _synthetic_terrain_metadata(
                slice_row,
                config=hf_config,
                obstacle_source_linkage=obstacle_source_linkage,
            )
            observation_payload.update(synthetic_terrain_metadata)
            hybrid_path_metadata = _hybrid_astar_path_cost_metadata(
                candidates,
                current_cell=cell_before,
                current_theta_deg=current_theta_deg,
                candidate_set_hash_value=candidate_set_hash_value,
                config=hf_config,
                slice_row=slice_row,
                platform_contract_hash=slope_theta_metadata.get("platform_contract_hash"),
            )
            observation_payload.update(hybrid_path_metadata)
            selected_candidate = _apply_selected_hybrid_path_cost(selected_candidate, selected_index, hybrid_path_metadata)
            candidates[selected_index] = selected_candidate
        selected_hybrid_unreachable = (
            continuous_theta_enabled(config)
            and bool(hf_config.get("hybrid_astar_pose_path_cost_enabled", False))
            and (
                selected_candidate is None
                or selected_candidate.get("path_cost_source") != HYBRID_ASTAR_PATH_COST_SOURCE
                or selected_candidate.get("hybrid_astar_reachable") is not True
                or _finite(selected_candidate.get("hybrid_astar_path_cost")) is None
                or not selected_candidate.get("hybrid_astar_pose_path_hash")
            )
        )
        selected_cell = hf._cell_tuple(hf._candidate_cell(selected_candidate)) if selected_candidate is not None else None
        selected_cost = hf._candidate_cost(selected_candidate) if selected_candidate is not None else None
        selected_hard_risk_violation = bool(selected_candidate is not None and _candidate_hard_risk_violation(selected_candidate, allow_open_grid_fallback=bool(hf_config["allow_open_grid_fallback"])))
        mask_violation = hf._mask_violation(action_mask, selected_index)
        if selected_hard_risk_violation:
            hard_risk_violation_count += 1
        if (
            mask_violation
            or selected_hard_risk_violation
            or selected_hybrid_unreachable
            or selected_candidate is None
            or selected_cell is None
            or selected_cost is None
        ):
            terminal_reason = "selected_action_not_trainable"
            if selected_hybrid_unreachable:
                terminal_reason = "selected_continuous_theta_hybrid_astar_unreachable"
                _mark_latest_transition_terminal(
                    transitions,
                    trainable_batch,
                    scenario_id=scenario_id,
                    terminal_reason=terminal_reason,
                )
            reason_codes.append(terminal_reason)
            rejections.append(
                _rejection_row(
                    scenario_id,
                    step_index,
                    terminal_reason,
                    candidate_set_hash_value=candidate_set_hash_value,
                    covered_cells_hash_value=covered_hash,
                    action_index=selected_index,
                    mask_violation=mask_violation,
                    selected_hard_risk_violation=selected_hard_risk_violation,
                    selected_hybrid_astar_failure_reason=(
                        selected_candidate.get("hybrid_astar_failure_reason")
                        if isinstance(selected_candidate, dict)
                        else None
                    ),
                )
            )
            break

        footprint = hf._candidate_coverage_cells(
            start=cell_before,
            end=selected_cell,
            candidate=selected_candidate,
            config=hf_config,
            obstacle_source_linkage=obstacle_source_linkage,
        )
        new_cells = footprint - covered_cells
        revisited_cells = footprint & covered_cells
        covered_cells.update(footprint)
        current_cell = selected_cell
        selected_theta = hf._finite_or_none(selected_candidate.get("candidate_theta_deg"))
        if selected_theta is not None:
            current_theta_deg = float(selected_theta)
        path_cost_total += float(selected_cost)
        soft_risk_exposure = hf._candidate_soft_risk_exposure(selected_candidate)
        soft_risk_exposure_total += soft_risk_exposure
        coverage_delta = len(new_cells) / denominator
        coverage_rates.append(len(covered_cells) / denominator)
        reward_result = _compute_collector_reward(
            selected_candidate,
            coverage_delta=coverage_delta,
            new_cell_count=len(new_cells),
            path_cost=float(selected_cost),
            soft_risk_exposure=soft_risk_exposure,
            profile=profile,
        )
        if continuous_theta_enabled(config):
            log_prob_error = abs(
                float(detail["old_log_prob"])
                - (float(detail.get("old_point_log_prob", 0.0)) + float(detail.get("old_theta_log_prob", 0.0)))
            )
        else:
            log_prob_error = abs(_recompute_log_prob(detail["old_sampling_logits"], selected_index) - float(detail["old_log_prob"]))
        transition_id = f"{scenario_id}:step-{step_index}:sample-{selected_index}"
        transition = {
            "schema_version": "xunce-stage21-1-ppo-transition/v1",
            "transition_id": transition_id,
            "scenario_id": scenario_id,
            "roi_group": roi_group,
            "split": slice_row.get("split"),
            "step_index": step_index,
            "observation": observation_payload,
            "xunce_batch": xunce_batch_payload,
            "action_index": int(selected_index),
            "action_space_type": detail.get("action_space_type"),
            "selected_base_candidate_index": detail.get("selected_base_candidate_index"),
            "selected_theta_rad": detail.get("selected_theta_rad"),
            "selected_theta_deg": detail.get("selected_theta_deg"),
            "old_point_log_prob": detail.get("old_point_log_prob"),
            "old_theta_log_prob": detail.get("old_theta_log_prob"),
            "base_candidate_set_hash": candidate_set_hash_value,
            "action_sample_hash": selected_candidate.get("action_sample_hash"),
            "old_log_prob": float(detail["old_log_prob"]),
            "old_value": float(detail["old_value"]),
            "reward": float(reward_result["reward"]),
            "reward_components": reward_result["components"],
            "reward_profile_id": reward_result["profile_id"],
            "reward_profile_version": reward_result["profile_version"],
            "reward_profile_hash": reward_result["profile_hash"],
            "next_observation": None,
            "next_xunce_batch": None,
            "done": False,
            "trainable": True,
            "info": {
                "scenario_id": scenario_id,
                "step_index": step_index,
                "current_cell_before": list(cell_before),
                "selected_cell": list(selected_cell),
                "selected_viewpoint": selected_candidate.get("candidate_viewpoint"),
                "action_space_type": detail.get("action_space_type"),
                "selected_base_candidate_index": detail.get("selected_base_candidate_index"),
                "selected_theta_rad": detail.get("selected_theta_rad"),
                "selected_theta_deg": selected_candidate.get("candidate_theta_deg"),
                "base_candidate_set_hash": candidate_set_hash_value,
                "action_sample_hash": selected_candidate.get("action_sample_hash"),
                "old_point_log_prob": detail.get("old_point_log_prob"),
                "old_theta_log_prob": detail.get("old_theta_log_prob"),
                "theta_mu_rad": detail.get("theta_mu_rad"),
                "theta_kappa": detail.get("theta_kappa"),
                "candidate_cells": candidate_observation_cells(candidates),
                **theta_metadata(candidates),
                **slope_theta_metadata,
                **synthetic_terrain_metadata,
                **hybrid_path_metadata,
                "candidate_set_id": candidate_set_id,
                "candidate_set_hash": candidate_set_hash_value,
                "covered_cells_hash": covered_hash,
                "action_mask": list(action_mask),
                "sampling_mask": list(sampling_mask),
                "hard_risk_clean_mask": list(hard_risk_clean_mask),
                "sampling_seed": int(config["sampling_seed"]),
                "sampling_temperature": float(config["sampling_temperature"]),
                "old_logits": detail["old_logits"],
                "old_masked_logits": detail["old_masked_logits"],
                "old_sampling_logits": detail["old_sampling_logits"],
                "old_action_probs": detail["old_action_probs"],
                "argmax_action_index": detail["argmax_action_index"],
                "selected_probability": detail["selected_probability"],
                "action_entropy": detail["action_entropy"],
                "old_log_prob_recompute_abs_error": log_prob_error,
                "path_cost": float(selected_cost),
                "soft_risk_exposure": float(soft_risk_exposure),
                "hard_risk_violation": selected_hard_risk_violation,
                "coverage_rate_delta": float(coverage_delta),
                "new_covered_cell_count": len(new_cells),
                "revisited_cell_count": len(revisited_cells),
                "final_coverage_rate_after_step": coverage_rates[-1],
                "candidate_generation_source": candidate_batch["candidate_generation_source"],
                "dynamic_proposal_count": candidate_batch["dynamic_proposal_count"],
                "dynamic_validated_candidate_count": candidate_batch["dynamic_validated_candidate_count"],
            },
        }
        pending = transition
        executed_step_count += 1
        reward_audit.append(
            {
                "schema_version": "xunce-stage21-1-reward-audit/v1",
                "transition_id": transition_id,
                "scenario_id": scenario_id,
                "step_index": step_index,
                "reward": reward_result["reward"],
                "reward_components": reward_result["components"],
                "metrics": reward_result["metrics"],
                "profile_id": reward_result["profile_id"],
                "profile_version": reward_result["profile_version"],
                "profile_hash": reward_result["profile_hash"],
                "reward_contract": "canonical_v3_interim_collector_reward",
            }
        )
        sampling_audit.append(
            {
                "schema_version": "xunce-stage21-1-sampling-audit/v1",
                "transition_id": transition_id,
                "scenario_id": scenario_id,
                "step_index": step_index,
                "action_index": int(selected_index),
                "argmax_action_index": detail["argmax_action_index"],
                "sampled_action_equals_argmax": int(selected_index) == int(detail["argmax_action_index"]),
                "old_log_prob": detail["old_log_prob"],
                "old_point_log_prob": detail.get("old_point_log_prob"),
                "old_theta_log_prob": detail.get("old_theta_log_prob"),
                "selected_theta_rad": detail.get("selected_theta_rad"),
                "selected_theta_deg": detail.get("selected_theta_deg"),
                "action_space_type": detail.get("action_space_type"),
                "theta_mu_rad": detail.get("theta_mu_rad"),
                "theta_kappa": detail.get("theta_kappa"),
                "old_value": detail["old_value"],
                "old_log_prob_recompute_abs_error": log_prob_error,
                "action_mask": list(action_mask),
                "hard_risk_clean_mask": list(hard_risk_clean_mask),
                "sampling_mask": list(sampling_mask),
                "sampled_from": (
                    "Categorical(point_logits)+VonMises(theta)"
                    if continuous_theta_enabled(config)
                    else "torch.distributions.Categorical(logits=old_sampling_logits)"
                ),
            }
        )

    if pending is not None:
        _finalize_pending(
            pending,
            transitions,
            trainable_batch,
            done=True,
            next_observation=None,
            next_xunce_batch=None,
        )
    if terminal_reason is None:
        terminal_reason = "rollout_steps_exhausted"
    episode = {
        "schema_version": "xunce-stage21-1-ppo-rollout-episode/v1",
        "scenario_id": scenario_id,
        "roi_group": roi_group,
        "split": slice_row.get("split"),
        "rollout_steps": int(hf_config["rollout_steps"]),
        "executed_step_count": executed_step_count,
        "terminal_reason": terminal_reason,
        "transition_ids": [row["transition_id"] for row in transitions],
        "trainable_transition_count": sum(1 for row in transitions if row.get("trainable") is True),
        "initial_coverage_rate": coverage_rates[0],
        "final_coverage_rate": coverage_rates[-1],
        "coverage_rate_delta": coverage_rates[-1] - coverage_rates[0],
        "coverage_curve_auc": hf._coverage_curve_auc(coverage_rates, int(hf_config["rollout_steps"])),
        "path_cost_total_m": path_cost_total,
        "soft_risk_exposure_total": soft_risk_exposure_total,
        "hard_risk_violation_count": hard_risk_violation_count,
        "reason_codes": _unique(reason_codes),
    }
    return {
        "episode": episode,
        "transitions": transitions,
        "trainable_batch": trainable_batch,
        "rejections": rejections,
        "reward_audit": reward_audit,
        "sampling_audit": sampling_audit,
        "reason_codes": reason_codes,
    }


def _sample_xunce_action(
    model: Any,
    xunce_batch: dict[str, torch.Tensor],
    *,
    sampling_mask: tuple[bool, ...],
    temperature: float,
    continuous_theta_action_space_enabled: bool = False,
) -> dict[str, Any]:
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("sampling_temperature must be finite and > 0")
    started = time.perf_counter()
    with torch.no_grad():
        output = model(**xunce_batch)
    latency_ms = (time.perf_counter() - started) * 1000.0
    logits = output.logits[0].detach().cpu()
    masked_logits = output.masked_logits[0].detach().cpu()
    value = output.value[0].detach().cpu()
    sampling_logits = masked_logits / float(temperature)
    mask_tensor = torch.tensor(list(sampling_mask), dtype=torch.bool)
    sampling_logits = sampling_logits.masked_fill(~mask_tensor, -1.0e9)
    finite_outputs = bool(
        torch.isfinite(logits).all()
        and torch.isfinite(masked_logits).all()
        and torch.isfinite(sampling_logits).all()
        and torch.isfinite(value)
    )
    if continuous_theta_action_space_enabled:
        detail = sample_continuous_theta_action(
            point_logits=masked_logits,
            theta_mu_rad=output.theta_mu_rad[0].detach().cpu(),
            theta_kappa=output.theta_kappa[0].detach().cpu(),
            sampling_mask=mask_tensor,
            temperature=temperature,
        )
        detail.update(
            {
                "old_logits": _float_list(logits),
                "old_masked_logits": _float_list(masked_logits),
                "old_value": float(value),
                "theta_mu_rad": _float_list(output.theta_mu_rad[0].detach().cpu()),
                "theta_kappa": _float_list(output.theta_kappa[0].detach().cpu()),
                "finite_outputs": bool(
                    finite_outputs
                    and torch.isfinite(output.theta_mu_rad[0]).all()
                    and torch.isfinite(output.theta_kappa[0]).all()
                ),
                "latency_ms": float(latency_ms),
            }
        )
        return detail
    distribution = torch.distributions.Categorical(logits=sampling_logits)
    action = distribution.sample()
    action_index = int(action.item())
    probs = torch.softmax(sampling_logits, dim=-1)
    argmax_action_index = int(torch.argmax(probs).item())
    return {
        "old_logits": _float_list(logits),
        "old_masked_logits": _float_list(masked_logits),
        "old_sampling_logits": _float_list(sampling_logits),
        "old_action_probs": _float_list(probs),
        "old_value": float(value),
        "old_log_prob": float(distribution.log_prob(action).item()),
        "action_index": action_index,
        "argmax_action_index": argmax_action_index,
        "selected_probability": float(probs[action_index]),
        "action_entropy": float(distribution.entropy().item()),
        "finite_outputs": finite_outputs,
        "latency_ms": float(latency_ms),
    }


def _slope_obstacle_theta_metadata(
    candidates: list[dict[str, Any]],
    *,
    current_cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    config: dict[str, Any],
    obstacle_source_linkage: dict[str, Any] | None,
) -> dict[str, Any]:
    if not (
        bool(config.get("slope_obstacle_aware_theta_reward_enabled", False))
        and bool(config.get("obstacle_occlusion_enabled", False))
    ):
        return {}
    has_theta = any(candidate.get("candidate_theta_deg") is not None for candidate in candidates)
    if not has_theta:
        return {}

    counts: list[int | None] = []
    hashes: list[str | None] = []
    gains: list[float | None] = []
    strict_count = 0
    for candidate in candidates:
        cell = hf._cell_tuple(hf._candidate_cell(candidate))
        if cell is None or candidate.get("candidate_theta_deg") is None:
            counts.append(None)
            hashes.append(None)
            gains.append(None)
            continue
        footprint = hf._candidate_coverage_cells(
            start=current_cell,
            end=cell,
            candidate=candidate,
            config=config,
            obstacle_source_linkage=obstacle_source_linkage,
        )
        new_count = len(footprint - covered_cells)
        counts.append(int(new_count))
        hashes.append(obstacle_aware_theta_coverage_hash(footprint))
        path_cost = hf._candidate_cost(candidate)
        gains.append((float(new_count) / float(path_cost)) if path_cost not in (None, 0) else None)
        strict_count += 1

    source_hash = None
    source_kind = None
    if isinstance(obstacle_source_linkage, dict):
        source_hash = obstacle_source_linkage.get("obstacle_source_hash")
        source_kind = obstacle_source_linkage.get("obstacle_source_kind")
    return {
        "obstacle_aware_new_visible_cell_counts": counts,
        "obstacle_aware_theta_coverage_hashes": hashes,
        "obstacle_aware_theta_coverage_gain_per_path_costs": gains,
        "slope_obstacle_source_hash": source_hash,
        "slope_obstacle_source_hashes": [source_hash for _ in candidates],
        "platform_contract_hash": config.get("platform_contract_hash"),
        "platform_contract_hashes": [config.get("platform_contract_hash") for _ in candidates],
        "platform_contract_id": config.get("platform_contract_id"),
        "max_traversable_slope_deg": float(config.get("max_traversable_slope_deg", 30.0)),
        "slope_blocked_source_kind": source_kind,
        "strict_obstacle_aware_new_visible_cell_count": strict_count == len(candidates) and strict_count > 0,
        "slope_obstacle_aware_coverage_source": "endpoint_theta_slope_obstacle_los/v1",
    }


def _synthetic_terrain_metadata(
    slice_row: dict[str, Any],
    *,
    config: dict[str, Any],
    obstacle_source_linkage: dict[str, Any] | None,
) -> dict[str, Any]:
    if not bool(config.get("synthetic_terrain_contract_enabled", False)):
        return {}
    sidecar_path = _resolved_file(slice_row.get("sidecar"))
    sidecar: dict[str, Any] = {}
    if sidecar_path is not None and sidecar_path.is_file():
        try:
            sidecar = _read_json(sidecar_path)
        except Exception:
            sidecar = {}
    synthetic_hash = str(config.get("synthetic_terrain_hash") or sidecar.get("synthetic_terrain_hash") or "")
    source_kind = str(config.get("synthetic_source_kind") or sidecar.get("synthetic_source_kind") or "")
    model_id = str(config.get("synthetic_terrain_model_id") or sidecar.get("synthetic_terrain_model_id") or "")
    hard_cells = _cell_list_payload(sidecar.get("synthetic_hard_obstacle_cells"))
    los_cells = _cell_list_payload(sidecar.get("synthetic_los_blocker_cells"))
    high_risk_cells = _cell_list_payload(sidecar.get("synthetic_high_risk_cells"))
    physical_written = bool(sidecar.get("physical_obstacle_cells_written_by_synthetic", False))
    obstacle_kind = None
    obstacle_hash = None
    if isinstance(obstacle_source_linkage, dict):
        obstacle_kind = obstacle_source_linkage.get("obstacle_source_kind")
        obstacle_hash = obstacle_source_linkage.get("obstacle_source_hash")
    return {
        "synthetic_terrain_model_id": model_id,
        "synthetic_terrain_hash": synthetic_hash,
        "synthetic_source_kind": source_kind,
        "synthetic_hard_obstacle_cells_used": bool(hard_cells),
        "synthetic_los_blocker_cells_used": bool(los_cells),
        "synthetic_high_risk_cells_available": bool(high_risk_cells),
        "synthetic_hard_obstacle_cell_count": len(hard_cells),
        "synthetic_los_blocker_cell_count": len(los_cells),
        "synthetic_high_risk_cell_count": len(high_risk_cells),
        "physical_obstacle_cells_written": physical_written,
        "effective_hard_obstacle_source": [
            source
            for source in (
                "physical_obstacle_cells",
                "slope_blocked_cells",
                "blocked_cells",
                "synthetic_hard_obstacle_cells" if hard_cells else None,
            )
            if source
        ],
        "effective_los_blocker_source": [
            source
            for source in (
                "physical_obstacle_cells",
                "slope_blocked_cells",
                "synthetic_los_blocker_cells" if los_cells else None,
            )
            if source
        ],
        "synthetic_obstacle_source_hash": obstacle_hash if obstacle_kind == "synthetic_terrain_obstacle_proxy/v1" else None,
        "synthetic_obstacle_source_kind": obstacle_kind if obstacle_kind == "synthetic_terrain_obstacle_proxy/v1" else None,
        "synthetic_terrain_contract_enabled": True,
    }


def _selected_continuous_theta_candidate(
    candidate: dict[str, Any],
    *,
    selected_index: int,
    detail: dict[str, Any],
    candidate_set_hash_value: str,
    sampling_seed: int,
) -> dict[str, Any]:
    updated = dict(candidate)
    cell = hf._cell_tuple(hf._candidate_cell(updated))
    theta_rad = float(detail["selected_theta_rad"])
    theta_deg = float(detail["selected_theta_deg"])
    updated["candidate_theta_deg"] = theta_deg
    updated["candidate_theta_rad"] = theta_rad
    updated["selected_theta_rad"] = theta_rad
    updated["candidate_viewpoint"] = [cell[0], cell[1], theta_deg] if cell is not None else None
    updated["base_candidate_index"] = int(selected_index)
    updated["selected_base_candidate_index"] = int(selected_index)
    updated["base_candidate_set_hash"] = str(candidate_set_hash_value)
    updated["action_sample_hash"] = action_sample_hash(
        base_candidate_set_hash=str(candidate_set_hash_value),
        action_index=int(selected_index),
        theta_rad=theta_rad,
        sampling_seed=int(sampling_seed),
    )
    updated["action_space_type"] = CONTINUOUS_THETA_ACTION_SPACE
    return updated


def _apply_selected_hybrid_path_cost(
    candidate: dict[str, Any],
    selected_index: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(candidate)
    if not metadata:
        return updated
    index = int(selected_index)
    for target, source in (
        ("path_cost_source", "path_cost_sources"),
        ("hybrid_astar_path_cost", "hybrid_astar_path_costs"),
        ("hybrid_astar_pose_path_hash", "hybrid_astar_pose_path_hashes"),
        ("hybrid_astar_trajectory_kind", "hybrid_astar_trajectory_kinds"),
        ("hybrid_astar_reachable", "hybrid_astar_reachable_flags"),
        ("hybrid_astar_failure_reason", "hybrid_astar_failure_reasons"),
        ("legacy_grid_astar_path_cost", "legacy_grid_astar_path_costs"),
        ("hybrid_vs_grid_path_cost_delta", "hybrid_vs_grid_path_cost_deltas"),
        ("default_astar_replaced", "default_astar_replaced_flags"),
        ("hybrid_astar_ackermann_feasible_claimed", "hybrid_astar_ackermann_feasible_claimed_flags"),
    ):
        values = metadata.get(source)
        if isinstance(values, list) and 0 <= index < len(values):
            updated[target] = values[index]
    if updated.get("path_cost_source") == HYBRID_ASTAR_PATH_COST_SOURCE and updated.get("hybrid_astar_path_cost") is not None:
        updated["path_cost"] = updated["hybrid_astar_path_cost"]
    if metadata.get("hybrid_astar_current_pose") is not None:
        updated["hybrid_astar_current_pose"] = metadata.get("hybrid_astar_current_pose")
    if metadata.get("hybrid_astar_current_pose_provenance") is not None:
        updated["hybrid_astar_current_pose_provenance"] = metadata.get("hybrid_astar_current_pose_provenance")
    return updated


def _continuous_theta_reachability_probe_metadata(
    candidates: list[dict[str, Any]],
    *,
    current_cell: tuple[int, int],
    current_theta_deg: float,
    candidate_set_hash_value: str,
    config: dict[str, Any],
    slice_row: dict[str, Any],
    platform_contract_hash: str | None,
) -> dict[str, Any]:
    if not bool(config.get("hybrid_astar_pose_path_cost_enabled", False)):
        return {}
    probe_candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        probe = dict(candidate)
        cell = hf._cell_tuple(hf._candidate_cell(probe))
        if cell is not None:
            probe["candidate_theta_deg"] = float(current_theta_deg)
            probe["candidate_viewpoint"] = [int(cell[0]), int(cell[1]), float(current_theta_deg)]
        probe["candidate_index"] = int(index)
        probe["candidate_set_hash"] = candidate_set_hash_value
        probe_candidates.append(probe)
    metadata = _hybrid_astar_path_cost_metadata(
        probe_candidates,
        current_cell=current_cell,
        current_theta_deg=current_theta_deg,
        candidate_set_hash_value=candidate_set_hash_value,
        config=config,
        slice_row=slice_row,
        platform_contract_hash=platform_contract_hash,
    )
    metadata["hybrid_astar_reachability_probe_theta_deg"] = float(current_theta_deg)
    metadata["hybrid_astar_reachability_probe_provenance"] = (
        "continuous_theta_base_candidate_current_heading_probe/v1"
    )
    return metadata


def _evaluate_hybrid_astar_candidate_path_cost_worker(args: tuple[Any, ...]) -> tuple[int, dict[str, Any] | None, str | None]:
    (
        index,
        grid,
        current_pose,
        payload,
        platform_hash,
        max_slope,
        planner_options,
    ) = args
    try:
        row = evaluate_hybrid_astar_candidate_path_cost(
            grid=grid,
            current_pose=current_pose,
            candidate=payload,
            platform_contract_hash=platform_hash,
            max_traversable_slope_deg=float(max_slope),
            theta_bin_count=int(planner_options["theta_bin_count"]),
            goal_position_tolerance_m=planner_options["goal_position_tolerance_m"],
            goal_theta_tolerance_deg=float(planner_options["goal_theta_tolerance_deg"]),
            max_iterations=int(planner_options["max_iterations"]),
            primitive_duration_s=float(planner_options["primitive_duration_s"]),
            integration_dt_s=float(planner_options["integration_dt_s"]),
            max_speed_mps=float(planner_options["max_speed_mps"]),
            max_angular_speed_degps=float(planner_options["max_angular_speed_degps"]),
            rotation_cost_weight=float(planner_options["rotation_cost_weight"]),
            reverse_penalty_weight=float(planner_options["reverse_penalty_weight"]),
            turn_penalty_weight=float(planner_options["turn_penalty_weight"]),
        )
        return int(index), row, None
    except Exception as exc:  # pragma: no cover - represented as per-candidate failure.
        return int(index), None, f"{type(exc).__name__}:{exc}"


def _hybrid_astar_failed_candidate_row(candidate: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "path_cost_source_recommendation": HYBRID_ASTAR_PATH_COST_SOURCE,
        "hybrid_astar_reachable": False,
        "hybrid_astar_trajectory_kind": "hybrid_astar_pose_path",
        "hybrid_astar_path_cost": None,
        "hybrid_astar_pose_path_hash": None,
        "hybrid_astar_failure_reason": reason,
        "legacy_grid_astar_path_cost": hf._candidate_cost(candidate),
        "hybrid_vs_grid_path_cost_delta": None,
        "default_astar_replaced": False,
        "hybrid_astar_ackermann_feasible_claimed": False,
    }


def _hybrid_astar_path_cost_metadata(
    candidates: list[dict[str, Any]],
    *,
    current_cell: tuple[int, int],
    current_theta_deg: float,
    candidate_set_hash_value: str,
    config: dict[str, Any],
    slice_row: dict[str, Any],
    platform_contract_hash: str | None,
) -> dict[str, Any]:
    if not bool(config.get("hybrid_astar_pose_path_cost_enabled", False)):
        return {}
    if not candidates:
        return {}
    sidecar_path = _resolved_file(slice_row.get("sidecar"))
    if sidecar_path is None or not sidecar_path.is_file():
        return _hybrid_path_unavailable_metadata(
            candidates,
            reason="sidecar_missing",
            platform_contract_hash=platform_contract_hash,
            max_traversable_slope_deg=config.get("max_traversable_slope_deg", 30.0),
        )
    try:
        sidecar = _read_json(sidecar_path)
        grid = build_cost_grid_from_sidecar(sidecar)
        current_world = _hybrid_cell_center_world(grid.spec, HYBRID_CELL(int(current_cell[0]), int(current_cell[1])))
    except Exception:
        return _hybrid_path_unavailable_metadata(
            candidates,
            reason="sidecar_decode_or_grid_build_failed",
            platform_contract_hash=platform_contract_hash,
            max_traversable_slope_deg=config.get("max_traversable_slope_deg", 30.0),
        )
    current_pose = [float(current_world.x), float(current_world.y), math.radians(float(current_theta_deg))]
    platform_hash = str(platform_contract_hash or config.get("platform_contract_hash") or sidecar.get("platform_contract_hash") or "")
    max_slope = float(config.get("max_traversable_slope_deg") or sidecar.get("max_traversable_slope_deg") or 30.0)
    worker_requested = _positive_int(config.get("hybrid_astar_candidate_eval_workers", 1), "hybrid_astar_candidate_eval_workers")
    worker_effective = worker_requested if worker_requested > 1 and len(candidates) > 1 else 1
    parallel_enabled = worker_effective > 1
    planner_options = {
        "theta_bin_count": int(config.get("hybrid_astar_theta_bin_count", 72)),
        "goal_position_tolerance_m": (
            float(config["hybrid_astar_goal_position_tolerance_m"])
            if config.get("hybrid_astar_goal_position_tolerance_m") is not None
            else None
        ),
        "goal_theta_tolerance_deg": float(config.get("hybrid_astar_goal_theta_tolerance_deg", 5.0)),
        "max_iterations": int(config.get("hybrid_astar_max_iterations", 100_000)),
        "primitive_duration_s": float(config.get("hybrid_astar_primitive_duration_s", 1.0)),
        "integration_dt_s": float(config.get("hybrid_astar_integration_dt_s", 0.25)),
        "max_speed_mps": float(config.get("hybrid_astar_max_speed_mps", 1.0)),
        "max_angular_speed_degps": float(config.get("hybrid_astar_max_angular_speed_degps", 45.0)),
        "rotation_cost_weight": float(config.get("hybrid_astar_rotation_cost_weight", 0.2)),
        "reverse_penalty_weight": float(config.get("hybrid_astar_reverse_penalty_weight", 0.5)),
        "turn_penalty_weight": float(config.get("hybrid_astar_turn_penalty_weight", 0.05)),
    }

    def payload_for(index: int, candidate: dict[str, Any]) -> dict[str, Any]:
        payload = dict(candidate)
        payload.setdefault("candidate_index", index)
        payload.setdefault("candidate_set_hash", candidate_set_hash_value)
        return payload

    started = time.perf_counter()
    rows_by_index: dict[int, dict[str, Any]] = {}
    failed_indices: set[int] = set()
    if parallel_enabled:
        with ProcessPoolExecutor(max_workers=worker_effective) as executor:
            futures = {
                executor.submit(
                    _evaluate_hybrid_astar_candidate_path_cost_worker,
                    (
                        index,
                        grid,
                        current_pose,
                        payload_for(index, candidate),
                        platform_hash,
                        max_slope,
                        planner_options,
                    ),
                ): index
                for index, candidate in enumerate(candidates)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    row_index, row, error = future.result()
                except Exception:
                    failed_indices.add(index)
                    continue
                if error is not None or row is None:
                    failed_indices.add(int(row_index))
                else:
                    rows_by_index[int(row_index)] = row
    else:
        for index, candidate in enumerate(candidates):
            try:
                rows_by_index[index] = evaluate_hybrid_astar_candidate_path_cost(
                    grid=grid,
                    current_pose=current_pose,
                    candidate=payload_for(index, candidate),
                    platform_contract_hash=platform_hash,
                    max_traversable_slope_deg=max_slope,
                    theta_bin_count=int(planner_options["theta_bin_count"]),
                    goal_position_tolerance_m=planner_options["goal_position_tolerance_m"],
                    goal_theta_tolerance_deg=float(planner_options["goal_theta_tolerance_deg"]),
                    max_iterations=int(planner_options["max_iterations"]),
                    primitive_duration_s=float(planner_options["primitive_duration_s"]),
                    integration_dt_s=float(planner_options["integration_dt_s"]),
                    max_speed_mps=float(planner_options["max_speed_mps"]),
                    max_angular_speed_degps=float(planner_options["max_angular_speed_degps"]),
                    rotation_cost_weight=float(planner_options["rotation_cost_weight"]),
                    reverse_penalty_weight=float(planner_options["reverse_penalty_weight"]),
                    turn_penalty_weight=float(planner_options["turn_penalty_weight"]),
                )
            except Exception:
                failed_indices.add(index)

    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        row = rows_by_index.get(index)
        if row is None or index in failed_indices:
            row = _hybrid_astar_failed_candidate_row(candidate, reason="hybrid_astar_evaluation_failed")
        rows.append(row)
    duration_s = max(0.0, time.perf_counter() - started)
    failed_count = len(failed_indices)
    path_sources = [
        HYBRID_ASTAR_PATH_COST_SOURCE if row.get("path_cost_source_recommendation") == HYBRID_ASTAR_PATH_COST_SOURCE else None
        for row in rows
    ]
    return {
        "path_cost_source": HYBRID_ASTAR_PATH_COST_SOURCE,
        "path_cost_sources": path_sources,
        "hybrid_astar_path_costs": [row.get("hybrid_astar_path_cost") for row in rows],
        "hybrid_astar_pose_path_hashes": [row.get("hybrid_astar_pose_path_hash") for row in rows],
        "hybrid_astar_trajectory_kinds": [row.get("hybrid_astar_trajectory_kind") for row in rows],
        "hybrid_astar_reachable_flags": [row.get("hybrid_astar_reachable") is True for row in rows],
        "hybrid_astar_failure_reasons": [row.get("hybrid_astar_failure_reason") for row in rows],
        "legacy_grid_astar_path_costs": [row.get("legacy_grid_astar_path_cost") for row in rows],
        "hybrid_vs_grid_path_cost_deltas": [row.get("hybrid_vs_grid_path_cost_delta") for row in rows],
        "default_astar_replaced": False,
        "default_astar_replaced_flags": [row.get("default_astar_replaced") is True for row in rows],
        "hybrid_astar_ackermann_feasible_claimed": False,
        "hybrid_astar_ackermann_feasible_claimed_flags": [
            row.get("hybrid_astar_ackermann_feasible_claimed") is True for row in rows
        ],
        "hybrid_astar_current_pose": current_pose,
        "hybrid_astar_current_pose_provenance": "stage21_1_current_cell_plus_previous_selected_theta/v1",
        "hybrid_astar_path_cost_source": HYBRID_ASTAR_PATH_COST_SOURCE,
        "hybrid_astar_candidate_eval_parallel_enabled": parallel_enabled,
        "hybrid_astar_candidate_eval_workers_requested": worker_requested,
        "hybrid_astar_candidate_eval_workers_effective": worker_effective,
        "hybrid_astar_candidate_eval_submitted_count": len(candidates),
        "hybrid_astar_candidate_eval_failed_count": failed_count,
        "hybrid_astar_candidate_eval_duration_s": duration_s,
    }


def _hybrid_astar_reachable_mask(metadata: dict[str, Any], *, candidate_count: int) -> tuple[bool, ...]:
    flags = metadata.get("hybrid_astar_reachable_flags")
    if not isinstance(flags, list) or len(flags) != candidate_count:
        return tuple(True for _ in range(candidate_count))
    costs = metadata.get("hybrid_astar_path_costs")
    hashes = metadata.get("hybrid_astar_pose_path_hashes")
    kinds = metadata.get("hybrid_astar_trajectory_kinds")
    sources = metadata.get("path_cost_sources")
    mask: list[bool] = []
    for index, value in enumerate(flags):
        valid = value is True
        if isinstance(costs, list) and len(costs) == candidate_count:
            cost = _finite(costs[index])
            valid = valid and cost is not None and cost > 0.0
        if isinstance(hashes, list) and len(hashes) == candidate_count:
            path_hash = hashes[index]
            valid = valid and isinstance(path_hash, str) and bool(path_hash.strip())
        if isinstance(kinds, list) and len(kinds) == candidate_count:
            valid = valid and kinds[index] == "hybrid_astar_pose_path"
        if isinstance(sources, list) and len(sources) == candidate_count:
            valid = valid and sources[index] == HYBRID_ASTAR_PATH_COST_SOURCE
        mask.append(bool(valid))
    return tuple(mask)


def _hybrid_cell_center_world(spec: Any, cell: HYBRID_CELL) -> HYBRID_WORLD_POINT:
    return HYBRID_WORLD_POINT(
        float(spec.origin[0]) + (float(cell.x) + 0.5) * float(spec.resolution),
        float(spec.origin[1]) + (float(cell.y) + 0.5) * float(spec.resolution),
    )


def _hybrid_path_unavailable_metadata(
    candidates: list[dict[str, Any]],
    *,
    reason: str,
    platform_contract_hash: str | None,
    max_traversable_slope_deg: Any,
) -> dict[str, Any]:
    count = len(candidates)
    legacy = [hf._candidate_cost(candidate) for candidate in candidates]
    return {
        "path_cost_source": HYBRID_ASTAR_PATH_COST_SOURCE,
        "path_cost_sources": [None for _ in candidates],
        "hybrid_astar_path_costs": [None for _ in candidates],
        "hybrid_astar_pose_path_hashes": [None for _ in candidates],
        "hybrid_astar_trajectory_kinds": ["hybrid_astar_pose_path" for _ in candidates],
        "hybrid_astar_reachable_flags": [False for _ in candidates],
        "hybrid_astar_failure_reasons": [reason for _ in candidates],
        "legacy_grid_astar_path_costs": legacy,
        "hybrid_vs_grid_path_cost_deltas": [None for _ in candidates],
        "default_astar_replaced": False,
        "default_astar_replaced_flags": [False for _ in candidates],
        "hybrid_astar_ackermann_feasible_claimed": False,
        "hybrid_astar_ackermann_feasible_claimed_flags": [False for _ in candidates],
        "platform_contract_hash": platform_contract_hash,
        "platform_contract_hashes": [platform_contract_hash for _ in candidates],
        "max_traversable_slope_deg": max_traversable_slope_deg,
        "hybrid_astar_unavailable_reason": reason,
        "hybrid_astar_unavailable_candidate_count": count,
    }


def _hard_risk_clean_sampling_mask(
    candidates: list[dict[str, Any]],
    action_mask: tuple[bool, ...],
    *,
    allow_open_grid_fallback: bool,
) -> tuple[bool, ...]:
    hard_risk_clean = _hard_risk_clean_mask(candidates, allow_open_grid_fallback=allow_open_grid_fallback)
    return tuple(bool(valid) and bool(clean) for valid, clean in zip(action_mask, hard_risk_clean))


def _hard_risk_clean_mask(
    candidates: list[dict[str, Any]],
    *,
    allow_open_grid_fallback: bool,
) -> tuple[bool, ...]:
    return tuple(
        not _candidate_hard_risk_violation(candidate, allow_open_grid_fallback=allow_open_grid_fallback)
        for candidate in candidates
    )


def _candidate_hard_risk_violation(candidate: dict[str, Any], *, allow_open_grid_fallback: bool) -> bool:
    if hf._candidate_hard_risk_violation(candidate):
        return True
    if not allow_open_grid_fallback and candidate.get("open_grid_fallback_used") is True:
        return True
    return False


def _compute_collector_reward(
    candidate: dict[str, Any],
    *,
    coverage_delta: float,
    new_cell_count: int,
    path_cost: float,
    soft_risk_exposure: float,
    profile: Any,
) -> dict[str, Any]:
    roi_value = hf._finite_or_none(candidate.get("roi_weighted_coverage_delta"))
    if roi_value is None:
        candidate_value = hf._finite_or_none(candidate.get("value"))
        roi_value = float(new_cell_count) * (candidate_value if candidate_value is not None else 1.0)
    metrics = {
        "coverage_gain_rate": float(coverage_delta),
        "roi_coverage": float(roi_value),
        "information_gain": 0.0,
        "path_cost_m": float(path_cost),
        "soft_risk_exposure": float(soft_risk_exposure),
        "fallback_used": bool(candidate.get("open_grid_fallback_used", False)),
        "hard_risk_violation": hf._candidate_hard_risk_violation(candidate),
        "failure": False,
    }
    result = compute_canonical_reward_components(metrics, profile)
    payload = result.to_dict()
    payload["metrics"] = metrics
    return payload


def _finalize_pending(
    pending: dict[str, Any] | None,
    transitions: list[dict[str, Any]],
    trainable_batch: list[dict[str, Any]],
    *,
    done: bool,
    next_observation: dict[str, Any] | None,
    next_xunce_batch: dict[str, Any] | None,
) -> None:
    if pending is None:
        return
    pending["done"] = bool(done)
    pending["next_observation"] = next_observation
    pending["next_xunce_batch"] = next_xunce_batch
    transitions.append(pending)
    if pending.get("trainable") is True:
        trainable_batch.append(pending)


def _mark_latest_transition_terminal(
    transitions: list[dict[str, Any]],
    trainable_batch: list[dict[str, Any]],
    *,
    scenario_id: str,
    terminal_reason: str,
) -> None:
    for row in reversed(transitions):
        if row.get("scenario_id") == scenario_id:
            row["done"] = True
            info = row.get("info")
            if isinstance(info, dict):
                info["terminal_reason"] = terminal_reason
            break
    for row in reversed(trainable_batch):
        if row.get("scenario_id") == scenario_id:
            row["done"] = True
            info = row.get("info")
            if isinstance(info, dict):
                info["terminal_reason"] = terminal_reason
            break


def _contract_counts(collection: CollectionResult) -> dict[str, Any]:
    trainable = collection.trainable_batch
    log_errors = [
        float(row["info"].get("old_log_prob_recompute_abs_error", 0.0))
        for row in trainable
        if _finite(row["info"].get("old_log_prob_recompute_abs_error")) is not None
    ]
    return {
        "episode_count": len(collection.episodes),
        "sampled_transition_count": len(collection.transitions),
        "trainable_transition_count": len(trainable),
        "mask_violation_count": sum(1 for row in trainable if row["info"]["action_mask"][int(row["action_index"])] is not True),
        "hard_risk_violation_count": sum(1 for row in trainable if row["info"].get("hard_risk_violation") is True),
        "non_finite_old_log_prob_count": sum(1 for row in trainable if _finite(row.get("old_log_prob")) is None),
        "non_finite_old_value_count": sum(1 for row in trainable if _finite(row.get("old_value")) is None),
        "non_finite_reward_count": sum(1 for row in trainable if _finite(row.get("reward")) is None),
        "old_log_prob_recompute_max_abs_error": max(log_errors) if log_errors else 0.0,
        "terminal_transition_count": sum(1 for row in trainable if row.get("done") is True),
        "transition_with_next_observation_count": sum(1 for row in trainable if row.get("next_observation") is not None),
        "sampled_action_equals_argmax_count": sum(
            1 for row in trainable if int(row["action_index"]) == int(row["info"].get("argmax_action_index", -1))
        ),
    }


def _write_outputs(
    *,
    config: dict[str, Any],
    profile: Any,
    output_root: Path,
    config_path: Path,
    collection: CollectionResult,
    blocking_reason_codes: list[str],
    route: str,
) -> dict[str, Any]:
    counts = _contract_counts(collection)
    status = "passed" if route == ROUTE_STAGE21_2 else "failed"
    generated_at = _utc_now()
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "primary_route": route,
        "stage21_1_authorized": False,
        "stage21_2_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "next_required_change": route,
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
        "collector_policy": "xunce_on_policy_stochastic",
        "sampling_method": "torch.distributions.Categorical(logits=old_sampling_logits)",
        "sampling_seed": int(config["sampling_seed"]),
        "sampling_temperature": float(config["sampling_temperature"]),
        "required_scenario_count": int(config["required_scenario_count"]),
        "rollout_steps": int(config["rollout_steps"]),
        "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
        "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
        "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
        "blocking_reason_codes": _unique(blocking_reason_codes),
        "reason_codes": _unique(blocking_reason_codes + collection.reason_codes),
        **counts,
        "stage21_1_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "summary": str((output_root / SUMMARY_FILE).resolve()),
        "episodes": str((output_root / EPISODES_FILE).resolve()),
        "transitions": str((output_root / TRANSITIONS_FILE).resolve()),
        "trainable_batch": str((output_root / TRAINABLE_BATCH_FILE).resolve()),
        "rejection_report": str((output_root / REJECTION_FILE).resolve()),
        "reward_audit": str((output_root / REWARD_AUDIT_FILE).resolve()),
        "sampling_audit": str((output_root / SAMPLING_AUDIT_FILE).resolve()),
        "routing": str((output_root / ROUTING_FILE).resolve()),
        "report": str((output_root / REPORT_FILE).resolve()),
        "manifest": str((output_root / MANIFEST_FILE).resolve()),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config": str(config_path.resolve()),
        "artifacts": {
            "summary": summary["summary"],
            "episodes": summary["episodes"],
            "transitions": summary["transitions"],
            "trainable_batch": summary["trainable_batch"],
            "rejection_report": summary["rejection_report"],
            "reward_audit": summary["reward_audit"],
            "sampling_audit": summary["sampling_audit"],
            "routing": summary["routing"],
            "report": summary["report"],
        },
        "model_audit": collection.model_audit,
        "summary_status": status,
        "next_required_change": route,
    }
    _write_jsonl(output_root / EPISODES_FILE, collection.episodes)
    _write_jsonl(output_root / TRANSITIONS_FILE, collection.transitions)
    _write_jsonl(output_root / TRAINABLE_BATCH_FILE, collection.trainable_batch)
    _write_jsonl(output_root / REJECTION_FILE, collection.rejections)
    _write_jsonl(output_root / REWARD_AUDIT_FILE, collection.reward_audit)
    _write_jsonl(output_root / SAMPLING_AUDIT_FILE, collection.sampling_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _load_config(path: Path, *, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    string_paths = (
        "high_fidelity_config",
        "stage21_0_readiness_root",
        "canonical_reward_profile",
        "dynamic_validation_work_root",
    )
    for key in string_paths:
        if not isinstance(config.get(key), str) or not str(config[key]).strip():
            raise ConfigError(f"{key} must be a non-empty path string")
        if key.endswith("_root"):
            config[key] = str(_resolve_path(Path(config[key]), repo_root))
        else:
            config[key] = str(_resolve_path(Path(config[key]), repo_root))
    if config.get("source_roi_expansion_root") is not None:
        if not isinstance(config.get("source_roi_expansion_root"), str) or not str(config["source_roi_expansion_root"]).strip():
            raise ConfigError("source_roi_expansion_root must be a non-empty path string when provided")
        config["source_roi_expansion_root"] = str(_resolve_path(Path(config["source_roi_expansion_root"]), repo_root))
    config["required_scenario_count"] = _positive_int(config.get("required_scenario_count", 2), "required_scenario_count")
    config["rollout_steps"] = _positive_int(config.get("rollout_steps", 4), "rollout_steps")
    config["dynamic_max_candidates_per_step"] = _positive_int(
        config.get("dynamic_max_candidates_per_step", 36),
        "dynamic_max_candidates_per_step",
    )
    config["dynamic_proposal_pool_limit_per_step"] = _positive_int(
        config.get("dynamic_proposal_pool_limit_per_step", 288),
        "dynamic_proposal_pool_limit_per_step",
    )
    config["min_trainable_transition_count"] = _nonnegative_int(
        config.get("min_trainable_transition_count", 1),
        "min_trainable_transition_count",
    )
    config["sampling_seed"] = _nonnegative_int(config.get("sampling_seed", 2101), "sampling_seed")
    config["sampling_temperature"] = _positive_float(config.get("sampling_temperature", 1.0), "sampling_temperature")
    config["theta_aware_candidate_viewpoints_enabled"] = bool(config.get("theta_aware_candidate_viewpoints_enabled", False))
    config["continuous_theta_action_space_enabled"] = continuous_theta_enabled(config)
    if config["continuous_theta_action_space_enabled"]:
        config["theta_aware_candidate_viewpoints_enabled"] = False
        config["action_space_type"] = CONTINUOUS_THETA_ACTION_SPACE
    config["theta_bin_count"] = _positive_int(config.get("theta_bin_count", 8), "theta_bin_count")
    config["theta_step_deg"] = _positive_int(config.get("theta_step_deg", 45), "theta_step_deg")
    config["sensor_model_id"] = str(config.get("sensor_model_id") or "theta-fov-90-range-radius/v1")
    config["sensor_fov_deg"] = _positive_float(config.get("sensor_fov_deg", 90.0), "sensor_fov_deg")
    if config.get("sensor_range_cells") is None:
        config.pop("sensor_range_cells", None)
    else:
        config["sensor_range_cells"] = _nonnegative_int(config.get("sensor_range_cells"), "sensor_range_cells")
    config["slope_obstacle_aware_theta_reward_enabled"] = bool(
        config.get("slope_obstacle_aware_theta_reward_enabled", False)
    )
    config["synthetic_terrain_contract_enabled"] = bool(config.get("synthetic_terrain_contract_enabled", False))
    for key in ("synthetic_terrain_model_id", "synthetic_terrain_hash", "synthetic_source_kind"):
        if config.get(key) is not None:
            config[key] = str(config[key])
    config["hybrid_astar_pose_path_cost_enabled"] = bool(
        config.get("hybrid_astar_pose_path_cost_enabled", False)
    )
    config["hybrid_astar_candidate_eval_workers"] = _positive_int(
        config.get("hybrid_astar_candidate_eval_workers", 1),
        "hybrid_astar_candidate_eval_workers",
    )
    config["initial_theta_deg"] = float(config.get("initial_theta_deg", 0.0))
    config["hybrid_astar_theta_bin_count"] = _positive_int(
        config.get("hybrid_astar_theta_bin_count", 72),
        "hybrid_astar_theta_bin_count",
    )
    config["hybrid_astar_goal_theta_tolerance_deg"] = _positive_float(
        config.get("hybrid_astar_goal_theta_tolerance_deg", 5.0),
        "hybrid_astar_goal_theta_tolerance_deg",
    )
    if config.get("hybrid_astar_goal_position_tolerance_m") is not None:
        config["hybrid_astar_goal_position_tolerance_m"] = _positive_float(
            config.get("hybrid_astar_goal_position_tolerance_m"),
            "hybrid_astar_goal_position_tolerance_m",
        )
    config["hybrid_astar_max_iterations"] = _positive_int(
        config.get("hybrid_astar_max_iterations", 100000),
        "hybrid_astar_max_iterations",
    )
    config["obstacle_occlusion_enabled"] = bool(config.get("obstacle_occlusion_enabled", False))
    config["derive_slope_blocked_cells_from_sidecar_dem"] = bool(
        config.get("derive_slope_blocked_cells_from_sidecar_dem", False)
    )
    config["no_go_blocks_los"] = bool(config.get("no_go_blocks_los", False))
    if config.get("max_traversable_slope_deg") is not None:
        config["max_traversable_slope_deg"] = _positive_float(
            config.get("max_traversable_slope_deg"),
            "max_traversable_slope_deg",
        )
    if config.get("platform_max_climb_deg") is not None:
        config["platform_max_climb_deg"] = _positive_float(config.get("platform_max_climb_deg"), "platform_max_climb_deg")
    config["max_log_prob_recompute_abs_error"] = _positive_float(
        config.get("max_log_prob_recompute_abs_error", 1.0e-6),
        "max_log_prob_recompute_abs_error",
    )
    config["canary_traffic_fraction"] = _nonnegative_float(config.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    for field in BOUNDARY_FIELDS:
        config.setdefault(field, False)
    return config


def _load_high_fidelity_config(config: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    overrides = {
        "canonical_reward_profile": config["canonical_reward_profile"],
        "required_scenario_count": config["required_scenario_count"],
        "rollout_steps": config["rollout_steps"],
        "dynamic_max_candidates_per_step": config["dynamic_max_candidates_per_step"],
        "dynamic_proposal_pool_limit_per_step": config["dynamic_proposal_pool_limit_per_step"],
        "dynamic_validation_work_root": config["dynamic_validation_work_root"],
        "include_oracle_baselines": False,
        "include_canonical_reward_rerank_oracle": False,
        "emit_on_policy_oracle_teacher_labels": False,
        "emit_candidate_metric_audit": False,
    }
    for key in (
        "theta_aware_candidate_viewpoints_enabled",
        "continuous_theta_action_space_enabled",
        "action_space_type",
        "source_roi_expansion_root",
        "theta_bin_count",
        "theta_step_deg",
        "sensor_model_id",
        "sensor_fov_deg",
        "sensor_range_cells",
        "slope_obstacle_aware_theta_reward_enabled",
        "synthetic_terrain_contract_enabled",
        "synthetic_terrain_model_id",
        "synthetic_terrain_hash",
        "synthetic_source_kind",
        "hybrid_astar_pose_path_cost_enabled",
        "hybrid_astar_candidate_eval_workers",
        "initial_theta_deg",
        "hybrid_astar_theta_bin_count",
        "hybrid_astar_goal_position_tolerance_m",
        "hybrid_astar_goal_theta_tolerance_deg",
        "hybrid_astar_max_iterations",
        "hybrid_astar_primitive_duration_s",
        "hybrid_astar_integration_dt_s",
        "hybrid_astar_max_speed_mps",
        "hybrid_astar_max_angular_speed_degps",
        "hybrid_astar_rotation_cost_weight",
        "hybrid_astar_reverse_penalty_weight",
        "hybrid_astar_turn_penalty_weight",
        "obstacle_occlusion_enabled",
        "derive_slope_blocked_cells_from_sidecar_dem",
        "no_go_blocks_los",
        "platform_contract_id",
        "platform_contract_hash",
        "platform_max_climb_deg",
        "max_traversable_slope_deg",
        "slope_sensitivity_thresholds_deg",
    ):
        if key in config:
            overrides[key] = config[key]
    return hf._load_config(Path(config["high_fidelity_config"]), repo_root, config_overrides=overrides)


def _input_rejections(config: dict[str, Any], profile: Any) -> list[str]:
    reasons: list[str] = []
    if profile.profile_version != "v3":
        reasons.append("stage21_1_requires_canonical_reward_profile_v3")
    stage21_0_summary = Path(config["stage21_0_readiness_root"]) / "xunce-stage21-0-pure-ppo-readiness-summary.json"
    if not stage21_0_summary.is_file():
        reasons.append("missing_stage21_0_readiness_summary")
    else:
        payload = _read_json(stage21_0_summary)
        if payload.get("status") != "passed":
            reasons.append("stage21_0_readiness_not_passed")
        if payload.get("next_required_change") != "implement_stage21_1_xunce_on_policy_ppo_rollout_collector":
            reasons.append("stage21_0_route_not_stage21_1_collector")
        if payload.get("profile_hash") != profile.profile_hash:
            reasons.append("stage21_0_profile_hash_mismatch")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in BOUNDARY_FIELDS:
        if config.get(field) is not False:
            reasons.append(field)
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _recompute_log_prob(logits: list[float], action_index: int) -> float:
    tensor = torch.tensor(logits, dtype=torch.float32)
    distribution = torch.distributions.Categorical(logits=tensor)
    return float(distribution.log_prob(torch.tensor(int(action_index))).item())


def _observation_to_dict(observation: Any) -> dict[str, Any]:
    return {
        "candidate_feature_names": list(observation.candidate_feature_names),
        "candidate_features": [list(row) for row in observation.candidate_features],
        "global_feature_names": list(observation.global_feature_names),
        "global_features": list(observation.global_features),
        "action_mask": [bool(value) for value in observation.action_mask],
        "candidate_cells": [_cell_to_list(cell) for cell in observation.candidate_cells],
        "candidate_missing_feature_names": [list(row) for row in observation.candidate_missing_feature_names],
        "candidate_missing_indicator_names": list(observation.candidate_missing_indicator_names),
        "candidate_missing_indicators": [list(row) for row in observation.candidate_missing_indicators],
    }


def _xunce_batch_to_dict(batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "values": value.detach().cpu().tolist(),
        }
        for key, value in batch.items()
    }


def _cell_to_list(cell: Any) -> list[int] | None:
    normalized = hf._cell_tuple(cell)
    if normalized is None:
        return None
    return [int(normalized[0]), int(normalized[1])]


def _cell_list_payload(value: Any) -> list[list[int]]:
    if not isinstance(value, list):
        return []
    cells: list[list[int]] = []
    for item in value:
        cell = _cell_to_list(item)
        if cell is not None:
            cells.append(cell)
    return cells


def _float_list(tensor: torch.Tensor) -> list[float]:
    return [float(item) for item in tensor.detach().cpu().tolist()]


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _rejection_row(
    scenario_id: str,
    step_index: int,
    reason: str,
    *,
    candidate_set_hash_value: str,
    covered_cells_hash_value: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage21-1-rejection/v1",
        "scenario_id": scenario_id,
        "step_index": int(step_index),
        "reason": reason,
        "candidate_set_hash": candidate_set_hash_value,
        "covered_cells_hash": covered_cells_hash_value,
        **extra,
    }


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage 21.1 Xunce On-Policy PPO Rollout Collector",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- sampled_transition_count: `{summary['sampled_transition_count']}`",
            f"- trainable_transition_count: `{summary['trainable_transition_count']}`",
            f"- mask_violation_count: `{summary['mask_violation_count']}`",
            f"- hard_risk_violation_count: `{summary['hard_risk_violation_count']}`",
            f"- old_log_prob_recompute_max_abs_error: `{summary['old_log_prob_recompute_max_abs_error']}`",
            "",
            "Stage 21.1 only collects PPO rollout data. It does not train, publish, replace the default policy, connect a real executor, or start canary traffic.",
        ]
    ) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"JSON file is invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"JSON root must be an object: {path}")
    return payload


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _resolved_file(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else path.resolve()


def _positive_int(value: Any, field: str) -> int:
    parsed = _nonnegative_int(value, field)
    if parsed <= 0:
        raise ConfigError(f"{field} must be > 0")
    return parsed


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be an integer") from exc
    if parsed < 0:
        raise ConfigError(f"{field} must be >= 0")
    return parsed


def _positive_float(value: Any, field: str) -> float:
    parsed = _nonnegative_float(value, field)
    if parsed <= 0.0:
        raise ConfigError(f"{field} must be > 0")
    return parsed


def _nonnegative_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ConfigError(f"{field} must be finite and >= 0")
    return parsed


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
