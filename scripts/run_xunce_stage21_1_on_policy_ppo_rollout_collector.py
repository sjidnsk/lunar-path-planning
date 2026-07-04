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
    build_derived_high_res_planning_proxy_grid,
    build_cost_grid_from_sidecar,
    evaluate_hybrid_astar_candidate_path_cost,
)
from xunce_continuous_theta_action import (
    CONTINUOUS_THETA_ACTION_SPACE,
    action_sample_hash,
    continuous_theta_log_prob,
    continuous_theta_enabled,
    sample_continuous_theta_action,
)
from xunce_synthetic_exploration_credit import (
    BEHAVIOR_POLICY_ID as SYNTHETIC_CREDIT_BEHAVIOR_POLICY_ID,
    SYNTHETIC_CREDIT_SCORE_V1,
    apply_feature_rows_to_xunce_batch,
    build_synthetic_credit_feature_rows,
    candidate_pressure_values,
    feature_semantic_map,
    select_reachable_synthetic_credit_theta,
    select_synthetic_credit_target,
    synthetic_credit_behavior_logprob,
)
import xunce_artifact_io as artifact_io
from xunce_artifact_paths import (
    STAGE21_1_EPISODES,
    STAGE21_1_REJECTIONS,
    STAGE21_1_SUMMARY,
    STAGE21_1_TRAINABLE,
    STAGE21_1_TRANSITIONS,
    artifact_path,
    write_json_artifact,
    write_jsonl_artifact,
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

REACHABILITY_GUARD_BEHAVIOR_POLICY_ID = "continuous_theta_reachability_guard_policy/v1"
REACHABILITY_GUARD_THETA_POLICY_ID = "reachable_theta_proposal/v1"

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
    artifact_io.make_dirs(output_root)

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
    terminal_no_reachable_reasons = {
        "no_hybrid_reachable_candidate_terminal",
        "no_selected_reachable_pose_candidate_terminal",
    }
    terminal_no_reachable_allowed = (
        any(reason in collection.reason_codes for reason in terminal_no_reachable_reasons)
        and counts["trainable_transition_count"] >= int(config["min_trainable_transition_count"])
        and counts["mask_violation_count"] == 0
        and counts["hard_risk_violation_count"] == 0
        and not any(
            row.get("reason") in {"path_planning_failure", "open_grid_fallback"}
            or "path_planning" in str(row.get("reason") or row.get("reason_code") or "")
            for row in collection.rejections
        )
    )
    if terminal_no_reachable_allowed:
        non_blocking_reason_codes.update(terminal_no_reachable_reasons)
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
    scenario_diversity_metadata = {
        "scenario_seed": scenario.get("scenario_seed"),
        "scenario_start_cell": scenario.get("scenario_start_cell") or list(start_cell),
        "scenario_start_cell_source": scenario.get("scenario_start_cell_source"),
        "scenario_roi_id": scenario.get("scenario_roi_id"),
        "scenario_candidate_seed": scenario.get("scenario_candidate_seed"),
        "scenario_diversity_source": scenario.get("scenario_diversity_source")
        or hf_config.get("scenario_diversity_source"),
        "scenario_diversity_signature_hash": scenario.get("scenario_diversity_signature_hash"),
        "scenario_diversity_content_hash": scenario.get("scenario_diversity_content_hash"),
    }
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
        observation_payload["candidate_cells"] = candidate_observation_cells(candidates)
        observation_payload.update(theta_metadata(candidates))
        continuous_probe_candidates = (
            _continuous_theta_probe_candidates(
                candidates,
                current_theta_deg=current_theta_deg,
                candidate_set_hash_value=candidate_set_hash_value,
            )
            if continuous_theta_enabled(config)
            else candidates
        )
        slope_theta_metadata = _slope_obstacle_theta_metadata(
            continuous_probe_candidates,
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
        if bool(config.get("synthetic_credit_feature_exposure_enabled", False)):
            synthetic_terrain_metadata.update(
                _synthetic_candidate_pressure_metadata(
                    continuous_probe_candidates,
                    current_cell=cell_before,
                    config=hf_config,
                    slice_row=slice_row,
                )
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
                continuous_probe_candidates,
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
        synthetic_credit_metadata = _synthetic_credit_step_metadata(
            xunce_batch=adapter["xunce_batch"],
            observation_payload=observation_payload,
            slope_theta_metadata=slope_theta_metadata,
            hybrid_path_metadata=hybrid_path_metadata or continuous_hybrid_probe_metadata,
            synthetic_terrain_metadata=synthetic_terrain_metadata,
            action_mask=action_mask,
            sampling_mask=sampling_mask,
            hard_risk_clean_mask=hard_risk_clean_mask,
            enabled=bool(config.get("synthetic_credit_feature_exposure_enabled", False)),
            score_version=str(config.get("synthetic_credit_score_version") or SYNTHETIC_CREDIT_SCORE_V1),
            path_efficiency_max_cost_norm=float(config.get("path_efficiency_max_cost_norm", 0.70)),
            theta_step_deg=float(config.get("theta_step_deg", 45.0)),
        )
        adapter["xunce_batch"] = synthetic_credit_metadata["updated_xunce_batch"]
        if synthetic_credit_metadata["enabled"]:
            observation_payload.update(
                {
                    "xunce_batch_feature_semantic_map": synthetic_credit_metadata["xunce_batch_feature_semantic_map"],
                    "synthetic_credit_feature_rows": synthetic_credit_metadata["feature_rows"],
                    "synthetic_los_blocker_candidate_counts": synthetic_credit_metadata[
                        "synthetic_los_blocker_candidate_counts"
                    ],
                    "synthetic_hard_obstacle_candidate_counts": synthetic_credit_metadata[
                        "synthetic_hard_obstacle_candidate_counts"
                    ],
                    "synthetic_credit_target_index": synthetic_credit_metadata["synthetic_credit_target_index"],
                    "synthetic_credit_target_theta_deg": synthetic_credit_metadata.get(
                        "synthetic_credit_target_theta_deg"
                    ),
                    "synthetic_credit_theta_policy_id": synthetic_credit_metadata.get(
                        "synthetic_credit_theta_policy_id"
                    ),
                    "synthetic_credit_theta_proposals_deg": synthetic_credit_metadata.get(
                        "synthetic_credit_theta_proposals_deg"
                    ),
                    "synthetic_credit_theta_selected_proposal_index": synthetic_credit_metadata.get(
                        "synthetic_credit_theta_selected_proposal_index"
                    ),
                    "synthetic_credit_theta_reachable_proposal_count": synthetic_credit_metadata.get(
                        "synthetic_credit_theta_reachable_proposal_count"
                    ),
                    "old_behavior_theta_log_prob": synthetic_credit_metadata.get("old_behavior_theta_log_prob"),
                    "synthetic_credit_score": synthetic_credit_metadata["synthetic_credit_target_score"],
                    "synthetic_credit_score_version": synthetic_credit_metadata["synthetic_credit_score_version"],
                    "path_efficiency_filter_relaxed": synthetic_credit_metadata["path_efficiency_filter_relaxed"],
                    "selected_target_hybrid_cost_norm": synthetic_credit_metadata["selected_target_hybrid_cost_norm"],
                    "selected_target_gain_per_cost_norm": synthetic_credit_metadata["selected_target_gain_per_cost_norm"],
                    "synthetic_credit_target_reason": synthetic_credit_metadata["synthetic_credit_target_reason"],
                }
            )
        xunce_batch_payload = _xunce_batch_to_dict(adapter["xunce_batch"])

        if not any(sampling_mask):
            planning_proxy_rejection_fields = _planning_proxy_rejection_fields(
                continuous_hybrid_probe_metadata or hybrid_path_metadata
            )
            if pending is not None:
                _finalize_pending(
                    pending,
                    transitions,
                    trainable_batch,
                    done=True,
                    next_observation=observation_payload,
                    next_xunce_batch=xunce_batch_payload,
                )
                pending = None
            terminal_reason = _no_sampling_candidate_reason(
                action_mask=action_mask,
                hard_risk_clean_mask=hard_risk_clean_mask,
                hybrid_reachable_mask=hybrid_reachable_mask,
            )
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
                    hybrid_astar_reachable_mask=list(hybrid_reachable_mask),
                    sampling_mask=list(sampling_mask),
                    action_mask_true_count=sum(1 for value in action_mask if value),
                    hard_risk_clean_mask_true_count=sum(1 for value in hard_risk_clean_mask if value),
                    hybrid_astar_reachable_count=sum(1 for value in hybrid_reachable_mask if value),
                    sampling_mask_true_count=sum(1 for value in sampling_mask if value),
                    **planning_proxy_rejection_fields,
                )
            )
            break

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

        try:
            detail = _sample_xunce_action(
                model,
                adapter["xunce_batch"],
                sampling_mask=sampling_mask,
                temperature=float(config["sampling_temperature"]),
                continuous_theta_action_space_enabled=continuous_theta_enabled(config),
                synthetic_credit_target_index=(
                    _int_or_none(synthetic_credit_metadata.get("synthetic_credit_target_index"))
                    if bool(config.get("synthetic_exploration_credit_enabled", False))
                    else None
                ),
                synthetic_credit_target_theta_deg=(
                    synthetic_credit_metadata.get("synthetic_credit_target_theta_deg")
                    if bool(config.get("synthetic_exploration_credit_enabled", False))
                    else None
                ),
                synthetic_credit_theta_proposals_deg=(
                    synthetic_credit_metadata.get("synthetic_credit_theta_proposals_deg")
                    if bool(config.get("synthetic_exploration_credit_enabled", False))
                    else None
                ),
                synthetic_credit_theta_selected_proposal_index=(
                    synthetic_credit_metadata.get("synthetic_credit_theta_selected_proposal_index")
                    if bool(config.get("synthetic_exploration_credit_enabled", False))
                    else None
                ),
                synthetic_credit_theta_reachable_proposal_count=(
                    synthetic_credit_metadata.get("synthetic_credit_theta_reachable_proposal_count")
                    if bool(config.get("synthetic_exploration_credit_enabled", False))
                    else None
                ),
                synthetic_credit_theta_policy_id=(
                    synthetic_credit_metadata.get("synthetic_credit_theta_policy_id")
                    if bool(config.get("synthetic_exploration_credit_enabled", False))
                    else None
                ),
                synthetic_credit_behavior_theta_log_prob=(
                    synthetic_credit_metadata.get("old_behavior_theta_log_prob")
                    if bool(config.get("synthetic_exploration_credit_enabled", False))
                    else None
                ),
                synthetic_credit_mixture_probability=(
                    float(config.get("synthetic_credit_mixture_probability", 0.0))
                    if bool(config.get("synthetic_exploration_credit_enabled", False))
                    else 0.0
                ),
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
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                )
            )
            break

        if continuous_theta_enabled(config) and bool(config.get("selected_continuous_theta_reachability_guard_enabled", False)):
            detail = _apply_selected_continuous_theta_reachability_guard(
                detail,
                sampling_mask=sampling_mask,
                hybrid_path_metadata=continuous_hybrid_probe_metadata,
                sampling_seed=int(config["sampling_seed"]) + scenario_index + step_index,
            )

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
            if bool(config.get("synthetic_credit_feature_exposure_enabled", False)):
                synthetic_terrain_metadata.update(
                    _synthetic_candidate_pressure_metadata(
                        candidates,
                        current_cell=cell_before,
                        config=hf_config,
                        slice_row=slice_row,
                    )
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
                guard_failure_reason = detail.get("selected_pose_reachability_guard_failure_reason")
                terminal_reason = (
                    str(guard_failure_reason)
                    if guard_failure_reason == "no_selected_reachable_pose_candidate_terminal"
                    else "selected_continuous_theta_hybrid_astar_unreachable"
                )
                _mark_latest_transition_terminal(
                    transitions,
                    trainable_batch,
                    scenario_id=scenario_id,
                    terminal_reason=terminal_reason,
                )
            reason_codes.append(terminal_reason)
            planning_proxy_rejection_fields = _planning_proxy_rejection_fields(
                continuous_hybrid_probe_metadata or hybrid_path_metadata
            )
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
                    behavior_policy_id=detail.get("behavior_policy_id"),
                    synthetic_credit_target_index=synthetic_credit_metadata.get("synthetic_credit_target_index"),
                    synthetic_credit_target_theta_deg=synthetic_credit_metadata.get("synthetic_credit_target_theta_deg"),
                    synthetic_credit_theta_policy_id=detail.get("synthetic_credit_theta_policy_id"),
                    synthetic_credit_theta_proposals_deg=detail.get("synthetic_credit_theta_proposals_deg"),
                    synthetic_credit_theta_selected_proposal_index=detail.get(
                        "synthetic_credit_theta_selected_proposal_index"
                    ),
                    synthetic_credit_theta_reachable_proposal_count=detail.get(
                        "synthetic_credit_theta_reachable_proposal_count"
                    ),
                    old_behavior_theta_log_prob=detail.get("old_behavior_theta_log_prob"),
                    synthetic_credit_target_selected=detail.get("synthetic_credit_target_selected"),
                    synthetic_credit_target_score=synthetic_credit_metadata.get("synthetic_credit_target_score"),
                    selected_theta_deg=detail.get("selected_theta_deg"),
                    selected_continuous_theta_reachability_guard_enabled=detail.get(
                        "selected_continuous_theta_reachability_guard_enabled"
                    ),
                    selected_continuous_theta_unreachable_attempted=detail.get(
                        "selected_continuous_theta_unreachable_attempted"
                    ),
                    selected_pose_reachability_guard_failed=detail.get("selected_pose_reachability_guard_failed"),
                    selected_pose_reachability_guard_failure_reason=detail.get(
                        "selected_pose_reachability_guard_failure_reason"
                    ),
                    **planning_proxy_rejection_fields,
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
        elif detail.get("behavior_policy_id") == SYNTHETIC_CREDIT_BEHAVIOR_POLICY_ID:
            log_prob_error = abs(float(detail["old_log_prob"]) - float(detail.get("old_behavior_log_prob", 0.0)))
        else:
            log_prob_error = abs(_recompute_log_prob(detail["old_sampling_logits"], selected_index) - float(detail["old_log_prob"]))
        transition_id = f"{scenario_id}:step-{step_index}:sample-{selected_index}"
        transition = {
            "schema_version": "xunce-stage21-1-ppo-transition/v1",
            "transition_id": transition_id,
            "scenario_id": scenario_id,
            **scenario_diversity_metadata,
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
            "old_policy_point_log_prob": detail.get("old_policy_point_log_prob"),
            "old_policy_log_prob": detail.get("old_policy_log_prob"),
            "old_policy_theta_log_prob": detail.get("old_policy_theta_log_prob"),
            "old_behavior_point_log_prob": detail.get("old_behavior_point_log_prob"),
            "old_behavior_theta_log_prob": detail.get("old_behavior_theta_log_prob"),
            "old_behavior_log_prob": detail.get("old_behavior_log_prob"),
            "behavior_policy_id": detail.get("behavior_policy_id"),
            "synthetic_credit_mixture_probability": detail.get("synthetic_credit_mixture_probability"),
            "synthetic_credit_target_index": detail.get("synthetic_credit_target_index"),
            "synthetic_credit_target_theta_deg": synthetic_credit_metadata.get("synthetic_credit_target_theta_deg"),
            "synthetic_credit_theta_policy_id": detail.get("synthetic_credit_theta_policy_id"),
            "synthetic_credit_theta_proposals_deg": detail.get("synthetic_credit_theta_proposals_deg"),
            "synthetic_credit_theta_selected_proposal_index": detail.get(
                "synthetic_credit_theta_selected_proposal_index"
            ),
            "synthetic_credit_theta_reachable_proposal_count": detail.get(
                "synthetic_credit_theta_reachable_proposal_count"
            ),
            "synthetic_credit_target_selected": detail.get("synthetic_credit_target_selected"),
            "synthetic_credit_score": synthetic_credit_metadata.get("synthetic_credit_target_score"),
            "synthetic_credit_score_version": synthetic_credit_metadata.get("synthetic_credit_score_version"),
            "path_efficiency_filter_relaxed": synthetic_credit_metadata.get("path_efficiency_filter_relaxed"),
            "selected_target_hybrid_cost_norm": synthetic_credit_metadata.get("selected_target_hybrid_cost_norm"),
            "selected_target_gain_per_cost_norm": synthetic_credit_metadata.get("selected_target_gain_per_cost_norm"),
            "selected_continuous_theta_reachability_guard_enabled": detail.get(
                "selected_continuous_theta_reachability_guard_enabled"
            ),
            "selected_continuous_theta_unreachable_attempted": detail.get(
                "selected_continuous_theta_unreachable_attempted"
            ),
            "selected_continuous_theta_unreachable_original_index": detail.get(
                "selected_continuous_theta_unreachable_original_index"
            ),
            "selected_continuous_theta_unreachable_original_theta_deg": detail.get(
                "selected_continuous_theta_unreachable_original_theta_deg"
            ),
            "selected_continuous_theta_unreachable_reachable_candidate_count": detail.get(
                "selected_continuous_theta_unreachable_reachable_candidate_count"
            ),
            "selected_theta_resampled_for_reachability": detail.get("selected_theta_resampled_for_reachability"),
            "selected_candidate_resampled_for_reachability": detail.get(
                "selected_candidate_resampled_for_reachability"
            ),
            "selected_continuous_theta_resample_policy": detail.get("selected_continuous_theta_resample_policy"),
            "selected_continuous_theta_resample_point_log_prob_source": detail.get(
                "selected_continuous_theta_resample_point_log_prob_source"
            ),
            "selected_continuous_theta_reachable_candidate_count": detail.get(
                "selected_continuous_theta_reachable_candidate_count"
            ),
            "selected_continuous_theta_reachable_candidate_indices": detail.get(
                "selected_continuous_theta_reachable_candidate_indices"
            ),
            "selected_continuous_theta_guard_original_behavior_policy_id": detail.get(
                "selected_continuous_theta_guard_original_behavior_policy_id"
            ),
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
                **scenario_diversity_metadata,
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
                "old_policy_point_log_prob": detail.get("old_policy_point_log_prob"),
                "old_policy_log_prob": detail.get("old_policy_log_prob"),
                "old_policy_theta_log_prob": detail.get("old_policy_theta_log_prob"),
                "old_behavior_point_log_prob": detail.get("old_behavior_point_log_prob"),
                "old_behavior_theta_log_prob": detail.get("old_behavior_theta_log_prob"),
                "old_behavior_log_prob": detail.get("old_behavior_log_prob"),
                "behavior_policy_id": detail.get("behavior_policy_id"),
                "synthetic_credit_mixture_probability": detail.get("synthetic_credit_mixture_probability"),
                "synthetic_credit_target_index": detail.get("synthetic_credit_target_index"),
                "synthetic_credit_target_theta_deg": synthetic_credit_metadata.get("synthetic_credit_target_theta_deg"),
                "synthetic_credit_theta_policy_id": detail.get("synthetic_credit_theta_policy_id"),
                "synthetic_credit_theta_proposals_deg": detail.get("synthetic_credit_theta_proposals_deg"),
                "synthetic_credit_theta_selected_proposal_index": detail.get(
                    "synthetic_credit_theta_selected_proposal_index"
                ),
                "synthetic_credit_theta_reachable_proposal_count": detail.get(
                    "synthetic_credit_theta_reachable_proposal_count"
                ),
                "synthetic_credit_target_selected": detail.get("synthetic_credit_target_selected"),
                "synthetic_credit_score": synthetic_credit_metadata.get("synthetic_credit_target_score"),
                "synthetic_credit_score_version": synthetic_credit_metadata.get("synthetic_credit_score_version"),
                "path_efficiency_filter_relaxed": synthetic_credit_metadata.get("path_efficiency_filter_relaxed"),
                "selected_target_hybrid_cost_norm": synthetic_credit_metadata.get("selected_target_hybrid_cost_norm"),
                "selected_target_gain_per_cost_norm": synthetic_credit_metadata.get("selected_target_gain_per_cost_norm"),
                "selected_continuous_theta_reachability_guard_enabled": detail.get(
                    "selected_continuous_theta_reachability_guard_enabled"
                ),
                "selected_continuous_theta_unreachable_attempted": detail.get(
                    "selected_continuous_theta_unreachable_attempted"
                ),
                "selected_continuous_theta_unreachable_original_index": detail.get(
                    "selected_continuous_theta_unreachable_original_index"
                ),
                "selected_continuous_theta_unreachable_original_theta_deg": detail.get(
                    "selected_continuous_theta_unreachable_original_theta_deg"
                ),
                "selected_continuous_theta_unreachable_reachable_candidate_count": detail.get(
                    "selected_continuous_theta_unreachable_reachable_candidate_count"
                ),
                "selected_theta_resampled_for_reachability": detail.get("selected_theta_resampled_for_reachability"),
                "selected_candidate_resampled_for_reachability": detail.get(
                    "selected_candidate_resampled_for_reachability"
                ),
                "selected_continuous_theta_resample_policy": detail.get("selected_continuous_theta_resample_policy"),
                "selected_continuous_theta_resample_point_log_prob_source": detail.get(
                    "selected_continuous_theta_resample_point_log_prob_source"
                ),
                "selected_continuous_theta_reachable_candidate_count": detail.get(
                    "selected_continuous_theta_reachable_candidate_count"
                ),
                "selected_continuous_theta_reachable_candidate_indices": detail.get(
                    "selected_continuous_theta_reachable_candidate_indices"
                ),
                "selected_continuous_theta_guard_original_behavior_policy_id": detail.get(
                    "selected_continuous_theta_guard_original_behavior_policy_id"
                ),
                "xunce_batch_feature_semantic_map": synthetic_credit_metadata.get("xunce_batch_feature_semantic_map"),
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
                "old_policy_point_log_prob": detail.get("old_policy_point_log_prob"),
                "old_policy_log_prob": detail.get("old_policy_log_prob"),
                "old_behavior_point_log_prob": detail.get("old_behavior_point_log_prob"),
                "old_behavior_log_prob": detail.get("old_behavior_log_prob"),
                "behavior_policy_id": detail.get("behavior_policy_id"),
                "synthetic_credit_mixture_probability": detail.get("synthetic_credit_mixture_probability"),
                "synthetic_credit_target_index": detail.get("synthetic_credit_target_index"),
                "synthetic_credit_target_theta_deg": synthetic_credit_metadata.get("synthetic_credit_target_theta_deg"),
                "synthetic_credit_target_selected": detail.get("synthetic_credit_target_selected"),
                "synthetic_credit_score": synthetic_credit_metadata.get("synthetic_credit_target_score"),
                "synthetic_credit_score_version": synthetic_credit_metadata.get("synthetic_credit_score_version"),
                "path_efficiency_filter_relaxed": synthetic_credit_metadata.get("path_efficiency_filter_relaxed"),
                "selected_target_hybrid_cost_norm": synthetic_credit_metadata.get("selected_target_hybrid_cost_norm"),
                "selected_target_gain_per_cost_norm": synthetic_credit_metadata.get("selected_target_gain_per_cost_norm"),
                "selected_continuous_theta_reachability_guard_enabled": detail.get(
                    "selected_continuous_theta_reachability_guard_enabled"
                ),
                "selected_continuous_theta_unreachable_attempted": detail.get(
                    "selected_continuous_theta_unreachable_attempted"
                ),
                "selected_theta_resampled_for_reachability": detail.get("selected_theta_resampled_for_reachability"),
                "selected_candidate_resampled_for_reachability": detail.get(
                    "selected_candidate_resampled_for_reachability"
                ),
                "selected_continuous_theta_resample_policy": detail.get("selected_continuous_theta_resample_policy"),
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
        **scenario_diversity_metadata,
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
    synthetic_credit_target_index: int | None = None,
    synthetic_credit_target_theta_deg: Any = None,
    synthetic_credit_theta_proposals_deg: Any = None,
    synthetic_credit_theta_selected_proposal_index: Any = None,
    synthetic_credit_theta_reachable_proposal_count: Any = None,
    synthetic_credit_theta_policy_id: Any = None,
    synthetic_credit_behavior_theta_log_prob: Any = None,
    synthetic_credit_mixture_probability: float = 0.0,
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
    policy_probs = torch.softmax(sampling_logits, dim=-1)
    credit_target = _valid_synthetic_credit_target(
        synthetic_credit_target_index,
        mask_tensor=mask_tensor,
        candidate_count=int(policy_probs.shape[0]),
    )
    credit_p = float(synthetic_credit_mixture_probability)
    use_credit_target = (
        credit_target is not None
        and credit_p > 0.0
        and (
            credit_p >= 1.0
            or float(torch.rand((), dtype=torch.float32).item()) < credit_p
        )
    )
    finite_outputs = bool(
        torch.isfinite(logits).all()
        and torch.isfinite(masked_logits).all()
        and torch.isfinite(sampling_logits).all()
        and torch.isfinite(value)
    )
    if continuous_theta_action_space_enabled:
        theta_mu = output.theta_mu_rad[0].detach().cpu()
        theta_kappa = output.theta_kappa[0].detach().cpu()
        if use_credit_target and credit_target is not None:
            theta_distribution = torch.distributions.VonMises(theta_mu[credit_target], theta_kappa[credit_target])
            target_theta_deg = _finite(synthetic_credit_target_theta_deg)
            policy_theta_sample = theta_distribution.sample()
            reachability_theta_policy_enabled = (
                target_theta_deg is not None
                and str(synthetic_credit_theta_policy_id or "") == "reachability_theta_proposal_mixture/v1"
            )
            theta_sample = (
                torch.tensor(math.radians(float(target_theta_deg)), dtype=theta_mu.dtype)
                if reachability_theta_policy_enabled and target_theta_deg is not None
                else policy_theta_sample
            )
            theta_behavior = (
                str(synthetic_credit_theta_policy_id)
                if reachability_theta_policy_enabled
                else "policy_von_mises_sample/v1"
            )
            detail = continuous_theta_log_prob(
                point_logits=sampling_logits,
                theta_mu_rad=theta_mu,
                theta_kappa=theta_kappa,
                action_index=credit_target,
                theta_rad=theta_sample,
            )
            detail.update(
                {
                    "action_index": credit_target,
                    "selected_base_candidate_index": credit_target,
                    "old_action_probs": _float_list(policy_probs),
                    "old_sampling_logits": _float_list(sampling_logits),
                    "argmax_action_index": int(torch.argmax(policy_probs).item()),
                    "selected_probability": float(policy_probs[credit_target]),
                    "point_action_entropy": float(torch.distributions.Categorical(logits=sampling_logits).entropy().item()),
                    "theta_action_entropy": None,
                    "action_entropy": float(torch.distributions.Categorical(logits=sampling_logits).entropy().item()),
                    "synthetic_credit_target_theta_deg": target_theta_deg,
                    "synthetic_credit_theta_behavior": theta_behavior,
                    "synthetic_credit_theta_policy_id": theta_behavior,
                    "synthetic_credit_theta_proposals_deg": (
                        list(synthetic_credit_theta_proposals_deg)
                        if isinstance(synthetic_credit_theta_proposals_deg, list)
                        else None
                    ),
                    "synthetic_credit_theta_selected_proposal_index": _int_or_none(
                        synthetic_credit_theta_selected_proposal_index
                    ),
                    "synthetic_credit_theta_reachable_proposal_count": _int_or_none(
                        synthetic_credit_theta_reachable_proposal_count
                    ),
                    "old_policy_theta_log_prob": float(detail["old_theta_log_prob"]),
                    "old_behavior_theta_log_prob": _finite(synthetic_credit_behavior_theta_log_prob),
                }
            )
        else:
            detail = sample_continuous_theta_action(
                point_logits=masked_logits,
                theta_mu_rad=theta_mu,
                theta_kappa=theta_kappa,
                sampling_mask=mask_tensor,
                temperature=temperature,
            )
        detail = _apply_synthetic_credit_behavior_logprob(
            detail,
            policy_probs=_float_list(policy_probs),
            target_index=credit_target,
            mixture_probability=credit_p,
            behavior_theta_log_prob=detail.get("old_behavior_theta_log_prob"),
        )
        detail.update(
            {
                "old_logits": _float_list(logits),
                "old_masked_logits": _float_list(masked_logits),
                "old_value": float(value),
                "theta_mu_rad": _float_list(theta_mu),
                "theta_kappa": _float_list(theta_kappa),
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
    action = (
        torch.tensor(int(credit_target), dtype=torch.long)
        if use_credit_target and credit_target is not None
        else distribution.sample()
    )
    action_index = int(action.item())
    argmax_action_index = int(torch.argmax(policy_probs).item())
    detail = {
        "old_logits": _float_list(logits),
        "old_masked_logits": _float_list(masked_logits),
        "old_sampling_logits": _float_list(sampling_logits),
        "old_action_probs": _float_list(policy_probs),
        "old_value": float(value),
        "old_log_prob": float(distribution.log_prob(action).item()),
        "action_index": action_index,
        "argmax_action_index": argmax_action_index,
        "selected_probability": float(policy_probs[action_index]),
        "action_entropy": float(distribution.entropy().item()),
        "finite_outputs": finite_outputs,
        "latency_ms": float(latency_ms),
    }
    return _apply_synthetic_credit_behavior_logprob(
        detail,
        policy_probs=_float_list(policy_probs),
        target_index=credit_target,
        mixture_probability=credit_p,
    )


def _valid_synthetic_credit_target(
    target_index: int | None,
    *,
    mask_tensor: torch.Tensor,
    candidate_count: int,
) -> int | None:
    if target_index is None:
        return None
    target = int(target_index)
    if target < 0 or target >= int(candidate_count):
        return None
    if not bool(mask_tensor[target].item()):
        return None
    return target


def _apply_synthetic_credit_behavior_logprob(
    detail: dict[str, Any],
    *,
    policy_probs: list[float],
    target_index: int | None,
    mixture_probability: float,
    behavior_theta_log_prob: Any = None,
) -> dict[str, Any]:
    if target_index is None or float(mixture_probability) <= 0.0:
        return detail
    theta_log_prob = _finite(detail.get("old_theta_log_prob"))
    behavior = synthetic_credit_behavior_logprob(
        policy_probs=policy_probs,
        action_index=int(detail["action_index"]),
        target_index=target_index,
        mixture_probability=float(mixture_probability),
        theta_log_prob=theta_log_prob,
        behavior_theta_log_prob=behavior_theta_log_prob,
    )
    detail["behavior_policy_id"] = SYNTHETIC_CREDIT_BEHAVIOR_POLICY_ID
    detail["synthetic_credit_mixture_probability"] = float(mixture_probability)
    detail["synthetic_credit_target_index"] = target_index
    detail["synthetic_credit_target_selected"] = bool(behavior["synthetic_credit_target_selected"])
    detail["old_policy_point_log_prob"] = float(behavior["old_policy_point_log_prob"])
    detail["old_policy_log_prob"] = float(behavior["old_policy_log_prob"])
    detail["old_policy_theta_log_prob"] = theta_log_prob
    detail["old_behavior_point_log_prob"] = float(behavior["old_behavior_point_log_prob"])
    detail["old_behavior_theta_log_prob"] = float(behavior["old_behavior_theta_log_prob"])
    detail["old_behavior_log_prob"] = float(behavior["old_behavior_log_prob"])
    detail["old_log_prob"] = float(behavior["old_log_prob"])
    if detail.get("old_point_log_prob") is not None:
        detail["old_point_log_prob"] = float(behavior["old_behavior_point_log_prob"])
    if behavior_theta_log_prob is not None and detail.get("old_theta_log_prob") is not None:
        detail["old_theta_log_prob"] = float(behavior["old_behavior_theta_log_prob"])
    return detail


def _synthetic_credit_step_metadata(
    *,
    xunce_batch: dict[str, torch.Tensor],
    observation_payload: dict[str, Any],
    slope_theta_metadata: dict[str, Any],
    hybrid_path_metadata: dict[str, Any],
    synthetic_terrain_metadata: dict[str, Any],
    action_mask: tuple[bool, ...],
    sampling_mask: tuple[bool, ...],
    hard_risk_clean_mask: tuple[bool, ...],
    enabled: bool,
    score_version: str = SYNTHETIC_CREDIT_SCORE_V1,
    path_efficiency_max_cost_norm: float = 0.70,
    theta_step_deg: float = 45.0,
) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "updated_xunce_batch": xunce_batch,
            "feature_rows": [],
            "synthetic_credit_target_index": None,
            "synthetic_credit_target_theta_deg": None,
            "synthetic_credit_theta_policy_id": None,
            "synthetic_credit_theta_proposals_deg": [],
            "synthetic_credit_theta_selected_proposal_index": None,
            "synthetic_credit_theta_reachable_proposal_count": 0,
            "synthetic_credit_target_skipped_unreachable": False,
            "old_behavior_theta_log_prob": None,
            "synthetic_credit_target_score": None,
            "synthetic_credit_score_version": score_version,
            "path_efficiency_filter_relaxed": False,
            "selected_target_hybrid_cost_norm": None,
            "selected_target_gain_per_cost_norm": None,
            "synthetic_credit_target_reason": "synthetic_credit_disabled",
            "xunce_batch_feature_semantic_map": None,
        }
    candidate_count = int(xunce_batch["candidate_features"].shape[1])
    los_counts = _repeated_synthetic_pressure(
        synthetic_terrain_metadata,
        candidate_count,
        keys=("synthetic_los_blocker_candidate_counts", "synthetic_los_blocker_cell_count"),
    )
    hard_counts = _repeated_synthetic_pressure(
        synthetic_terrain_metadata,
        candidate_count,
        keys=("synthetic_hard_obstacle_candidate_counts", "synthetic_hard_obstacle_cell_count"),
    )
    feature_rows = build_synthetic_credit_feature_rows(
        candidate_count=candidate_count,
        relative_distances=_observation_feature_column(observation_payload, "relative_distance", candidate_count),
        hybrid_reachable_flags=hybrid_path_metadata.get("hybrid_astar_reachable_flags"),
        obstacle_aware_new_visible_cell_counts=slope_theta_metadata.get("obstacle_aware_new_visible_cell_counts"),
        obstacle_aware_gain_per_hybrid_costs=slope_theta_metadata.get(
            "obstacle_aware_theta_coverage_gain_per_path_costs"
        ),
        hybrid_astar_path_costs=hybrid_path_metadata.get("hybrid_astar_path_costs"),
        synthetic_los_blocker_candidate_counts=los_counts,
        synthetic_hard_obstacle_candidate_counts=hard_counts,
        risk_values=_observation_feature_column(observation_payload, "risk", candidate_count),
    )
    working_sampling_mask = list(sampling_mask)
    target: dict[str, Any] = {}
    target_index = None
    target_theta_deg = None
    theta_choice: dict[str, Any] = {}
    for _ in range(candidate_count):
        target = select_synthetic_credit_target(
            feature_rows,
            action_mask=action_mask,
            sampling_mask=working_sampling_mask,
            hard_risk_clean_mask=hard_risk_clean_mask,
            hybrid_reachable_flags=hybrid_path_metadata.get("hybrid_astar_reachable_flags"),
            score_version=score_version,
            path_efficiency_max_cost_norm=path_efficiency_max_cost_norm,
        )
        target_index = target.get("synthetic_credit_target_index")
        if target_index is None:
            break
        theta_choice = _synthetic_credit_theta_choice_for_target(
            hybrid_path_metadata,
            target_index=int(target_index),
            theta_step_deg=theta_step_deg,
        )
        target_theta_deg = _finite(theta_choice.get("synthetic_credit_target_theta_deg"))
        if not bool(theta_choice.get("synthetic_credit_target_skipped_unreachable")):
            break
        working_sampling_mask[int(target_index)] = False
        target_index = None
        target_theta_deg = None
    updated_batch = apply_feature_rows_to_xunce_batch(xunce_batch, feature_rows)
    semantic_map = feature_semantic_map()
    semantic_map["feature_contract_id"] = "synthetic_credit_candidate_features/v1"
    semantic_map["source"] = "stage26_6_synthetic_exploration_credit_assignment"
    return {
        "enabled": True,
        "updated_xunce_batch": updated_batch,
        "feature_rows": feature_rows,
        "synthetic_los_blocker_candidate_counts": los_counts,
        "synthetic_hard_obstacle_candidate_counts": hard_counts,
        "synthetic_credit_target_index": target_index,
        "synthetic_credit_target_theta_deg": target_theta_deg,
        "synthetic_credit_theta_policy_id": theta_choice.get("synthetic_credit_theta_policy_id"),
        "synthetic_credit_theta_proposals_deg": theta_choice.get("synthetic_credit_theta_proposals_deg", []),
        "synthetic_credit_theta_selected_proposal_index": theta_choice.get(
            "synthetic_credit_theta_selected_proposal_index"
        ),
        "synthetic_credit_theta_reachable_proposal_count": theta_choice.get(
            "synthetic_credit_theta_reachable_proposal_count",
            0,
        ),
        "synthetic_credit_target_skipped_unreachable": bool(
            theta_choice.get("synthetic_credit_target_skipped_unreachable", False)
        ),
        "old_behavior_theta_log_prob": theta_choice.get("old_behavior_theta_log_prob"),
        "synthetic_credit_target_score": target.get("synthetic_credit_target_score"),
        "synthetic_credit_score_version": target.get("synthetic_credit_score_version"),
        "path_efficiency_filter_relaxed": target.get("path_efficiency_filter_relaxed"),
        "selected_target_hybrid_cost_norm": target.get("selected_target_hybrid_cost_norm"),
        "selected_target_gain_per_cost_norm": target.get("selected_target_gain_per_cost_norm"),
        "synthetic_credit_target_reason": target.get("synthetic_credit_target_reason"),
        "xunce_batch_feature_semantic_map": semantic_map,
    }


def _synthetic_credit_theta_choice_for_target(
    hybrid_path_metadata: dict[str, Any],
    *,
    target_index: int,
    theta_step_deg: float,
) -> dict[str, Any]:
    proposals = _nested_float_list(
        hybrid_path_metadata.get("hybrid_astar_theta_proposals_deg_by_candidate"),
        target_index,
    )
    reachable = _nested_float_list(
        hybrid_path_metadata.get("hybrid_astar_reachable_theta_degs_by_candidate"),
        target_index,
    )
    probe_theta = _finite(hybrid_path_metadata.get("hybrid_astar_reachability_probe_theta_deg"))
    if not proposals:
        probe_theta = _finite(hybrid_path_metadata.get("hybrid_astar_reachability_probe_theta_deg"))
        proposals = [] if probe_theta is None else [probe_theta]
        flags = hybrid_path_metadata.get("hybrid_astar_reachable_flags")
        reachable = proposals if isinstance(flags, list) and target_index < len(flags) and flags[target_index] is True else []
    return select_reachable_synthetic_credit_theta(
        policy_theta_deg=None,
        reachability_probe_theta_deg=probe_theta,
        current_theta_deg=probe_theta,
        theta_step_deg=theta_step_deg,
        reachable_theta_degs=reachable,
        proposal_theta_degs=proposals,
    )


def _apply_selected_continuous_theta_reachability_guard(
    detail: dict[str, Any],
    *,
    sampling_mask: tuple[bool, ...],
    hybrid_path_metadata: dict[str, Any],
    sampling_seed: int,
) -> dict[str, Any]:
    selected_index = _int_or_none(detail.get("action_index"))
    selected_theta = _finite(detail.get("selected_theta_deg"))
    if selected_index is None or selected_theta is None:
        return detail
    reachable_by_candidate = hybrid_path_metadata.get("hybrid_astar_reachable_theta_degs_by_candidate")
    if not isinstance(reachable_by_candidate, list):
        return detail
    selected_reachable = _theta_in_reachable_set(selected_theta, _nested_float_list(reachable_by_candidate, selected_index))
    if selected_reachable:
        updated = dict(detail)
        updated["selected_continuous_theta_reachability_guard_enabled"] = True
        updated["selected_theta_resampled_for_reachability"] = False
        updated["selected_candidate_resampled_for_reachability"] = False
        return updated

    reachable_candidates = [
        index
        for index, mask in enumerate(sampling_mask)
        if mask and _nested_float_list(reachable_by_candidate, index)
    ]
    updated = dict(detail)
    updated["selected_continuous_theta_reachability_guard_enabled"] = True
    updated["selected_continuous_theta_unreachable_attempted"] = True
    updated["selected_continuous_theta_unreachable_original_index"] = int(selected_index)
    updated["selected_continuous_theta_unreachable_original_theta_deg"] = float(selected_theta)
    updated["selected_continuous_theta_unreachable_reachable_candidate_count"] = len(reachable_candidates)
    if not reachable_candidates:
        updated["selected_pose_reachability_guard_failed"] = True
        updated["selected_pose_reachability_guard_failure_reason"] = "no_selected_reachable_pose_candidate_terminal"
        return updated

    chosen_index = (
        selected_index
        if selected_index in reachable_candidates
        else _uniform_reachable_candidate_index(reachable_candidates, sampling_seed)
    )
    theta_choice = _reachable_guard_theta_choice(
        hybrid_path_metadata,
        candidate_index=chosen_index,
        selection_seed=int(sampling_seed),
    )
    target_theta_deg = _finite(theta_choice.get("selected_theta_deg"))
    if target_theta_deg is None:
        updated["selected_pose_reachability_guard_failed"] = True
        updated["selected_pose_reachability_guard_failure_reason"] = "selected_candidate_reachable_theta_missing"
        return updated

    policy_point_log_prob, policy_theta_log_prob = _policy_log_probs_for_continuous_theta(
        detail,
        action_index=chosen_index,
        theta_deg=float(target_theta_deg),
    )
    if policy_point_log_prob is None or policy_theta_log_prob is None:
        updated["selected_pose_reachability_guard_failed"] = True
        updated["selected_pose_reachability_guard_failure_reason"] = "policy_logprob_recompute_failed"
        return updated

    candidate_changed = int(chosen_index) != int(selected_index)
    existing_behavior_point = _finite(detail.get("old_behavior_point_log_prob"))
    existing_point = _finite(detail.get("old_point_log_prob"))
    if candidate_changed:
        behavior_point_log_prob = -math.log(max(1, len(reachable_candidates)))
        point_source = "uniform_reachable_candidate_support"
    else:
        behavior_point_log_prob = (
            existing_behavior_point
            if existing_behavior_point is not None
            else existing_point
            if existing_point is not None
            else policy_point_log_prob
        )
        point_source = "existing_behavior_point" if existing_behavior_point is not None else "policy_point"
    behavior_theta_log_prob = float(theta_choice["old_behavior_theta_log_prob"])
    selected_theta_rad = math.radians(float(target_theta_deg))
    old_policy_log_prob = float(policy_point_log_prob + policy_theta_log_prob)
    old_behavior_log_prob = float(behavior_point_log_prob + behavior_theta_log_prob)
    old_action_probs = detail.get("old_action_probs")
    selected_probability = (
        float(old_action_probs[chosen_index])
        if isinstance(old_action_probs, list)
        and 0 <= int(chosen_index) < len(old_action_probs)
        and _finite(old_action_probs[chosen_index]) is not None
        else detail.get("selected_probability")
    )
    updated.update(
        {
            "action_index": int(chosen_index),
            "selected_base_candidate_index": int(chosen_index),
            "selected_probability": selected_probability,
            "selected_theta_deg": float(target_theta_deg),
            "selected_theta_rad": selected_theta_rad,
            "old_policy_point_log_prob": float(policy_point_log_prob),
            "old_policy_theta_log_prob": float(policy_theta_log_prob),
            "old_policy_log_prob": old_policy_log_prob,
            "old_behavior_point_log_prob": float(behavior_point_log_prob),
            "old_behavior_theta_log_prob": behavior_theta_log_prob,
            "old_behavior_log_prob": old_behavior_log_prob,
            "old_point_log_prob": float(behavior_point_log_prob),
            "old_theta_log_prob": behavior_theta_log_prob,
            "old_log_prob": old_behavior_log_prob,
            "behavior_policy_id": REACHABILITY_GUARD_BEHAVIOR_POLICY_ID,
            "selected_theta_resampled_for_reachability": True,
            "selected_candidate_resampled_for_reachability": bool(candidate_changed),
            "selected_continuous_theta_resample_policy": REACHABILITY_GUARD_THETA_POLICY_ID,
            "selected_continuous_theta_resample_point_log_prob_source": point_source,
            "selected_continuous_theta_reachable_candidate_count": len(reachable_candidates),
            "selected_continuous_theta_reachable_candidate_indices": reachable_candidates,
            "selected_continuous_theta_guard_original_behavior_policy_id": detail.get("behavior_policy_id"),
            "synthetic_credit_target_selected": False if candidate_changed else detail.get("synthetic_credit_target_selected"),
            "synthetic_credit_theta_policy_id": REACHABILITY_GUARD_THETA_POLICY_ID,
            "synthetic_credit_theta_behavior": REACHABILITY_GUARD_THETA_POLICY_ID,
            "synthetic_credit_theta_proposals_deg": theta_choice["proposals"],
            "synthetic_credit_theta_selected_proposal_index": theta_choice["selected_proposal_index"],
            "synthetic_credit_theta_reachable_proposal_count": theta_choice["reachable_count"],
        }
    )
    return updated


def _theta_in_reachable_set(theta_deg: float, reachable: list[float]) -> bool:
    return any(_angle_deg_delta_abs(theta_deg, candidate) <= 1.0e-6 for candidate in reachable)


def _angle_deg_delta_abs(lhs: float, rhs: float) -> float:
    return abs(((float(lhs) - float(rhs) + 180.0) % 360.0) - 180.0)


def _uniform_reachable_candidate_index(indices: list[int], selection_seed: int) -> int:
    ordered = sorted(int(index) for index in indices)
    return ordered[abs(int(selection_seed)) % len(ordered)]


def _reachable_guard_theta_choice(
    hybrid_path_metadata: dict[str, Any],
    *,
    candidate_index: int,
    selection_seed: int,
) -> dict[str, Any]:
    proposals = _nested_float_list(hybrid_path_metadata.get("hybrid_astar_theta_proposals_deg_by_candidate"), candidate_index)
    reachable = _nested_float_list(hybrid_path_metadata.get("hybrid_astar_reachable_theta_degs_by_candidate"), candidate_index)
    flags = hybrid_path_metadata.get("hybrid_astar_theta_probe_reachable_flags_by_candidate")
    costs = hybrid_path_metadata.get("hybrid_astar_theta_probe_path_costs_by_candidate")
    flag_row = flags[candidate_index] if isinstance(flags, list) and 0 <= candidate_index < len(flags) and isinstance(flags[candidate_index], list) else []
    cost_row = costs[candidate_index] if isinstance(costs, list) and 0 <= candidate_index < len(costs) and isinstance(costs[candidate_index], list) else []
    reachable_entries: list[tuple[int, float, float]] = []
    for proposal_index, theta in enumerate(proposals):
        flag_reachable = bool(proposal_index < len(flag_row) and flag_row[proposal_index] is True)
        set_reachable = _theta_in_reachable_set(theta, reachable)
        if not flag_reachable and not set_reachable:
            continue
        cost = _finite(cost_row[proposal_index]) if proposal_index < len(cost_row) else None
        reachable_entries.append((proposal_index, float(theta), float("inf") if cost is None else float(cost)))
    if not reachable_entries:
        return {"selected_theta_deg": None}
    selected = reachable_entries[abs(int(selection_seed)) % len(reachable_entries)]
    return {
        "selected_proposal_index": int(selected[0]),
        "selected_theta_deg": float(selected[1]),
        "reachable_count": len(reachable_entries),
        "old_behavior_theta_log_prob": -math.log(max(1, len(reachable_entries))),
        "proposals": proposals,
    }


def _policy_log_probs_for_continuous_theta(
    detail: dict[str, Any],
    *,
    action_index: int,
    theta_deg: float,
) -> tuple[float | None, float | None]:
    logits = detail.get("old_sampling_logits")
    theta_mu = detail.get("theta_mu_rad")
    theta_kappa = detail.get("theta_kappa")
    if not isinstance(logits, list) or not isinstance(theta_mu, list) or not isinstance(theta_kappa, list):
        return None, None
    if action_index < 0 or action_index >= len(logits) or action_index >= len(theta_mu) or action_index >= len(theta_kappa):
        return None, None
    try:
        logits_tensor = torch.tensor([float(value) for value in logits], dtype=torch.float64)
        point_log_prob = torch.distributions.Categorical(logits=logits_tensor).log_prob(
            torch.tensor(int(action_index), dtype=torch.long)
        )
        theta_dist = torch.distributions.VonMises(
            torch.tensor(float(theta_mu[action_index]), dtype=torch.float64),
            torch.tensor(float(theta_kappa[action_index]), dtype=torch.float64),
        )
        theta_log_prob = theta_dist.log_prob(torch.tensor(math.radians(float(theta_deg)), dtype=torch.float64))
    except (TypeError, ValueError, RuntimeError):
        return None, None
    return float(point_log_prob.item()), float(theta_log_prob.item())


def _nested_float_list(value: Any, index: int) -> list[float]:
    if not isinstance(value, list) or index < 0 or index >= len(value):
        return []
    row = value[index]
    if not isinstance(row, list):
        return []
    result: list[float] = []
    for item in row:
        number = _finite(item)
        if number is not None:
            result.append(float(number))
    return result


def _observation_feature_column(observation_payload: dict[str, Any], name: str, count: int) -> list[float | None]:
    names = observation_payload.get("candidate_feature_names")
    rows = observation_payload.get("candidate_features")
    if not isinstance(names, list) or name not in names or not isinstance(rows, list):
        return [None for _ in range(count)]
    column = int(names.index(name))
    values: list[float | None] = []
    for index in range(count):
        row = rows[index] if index < len(rows) and isinstance(rows[index], list) else []
        values.append(_finite(row[column]) if column < len(row) else None)
    return values


def _repeated_synthetic_pressure(
    synthetic_terrain_metadata: dict[str, Any],
    count: int,
    *,
    keys: tuple[str, str],
) -> list[float | None]:
    return candidate_pressure_values(synthetic_terrain_metadata.get(keys[0]), count)


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
    if sidecar_path is not None and artifact_io.path_is_file(sidecar_path):
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


def _synthetic_candidate_pressure_metadata(
    candidates: list[dict[str, Any]],
    *,
    current_cell: tuple[int, int],
    config: dict[str, Any],
    slice_row: dict[str, Any],
) -> dict[str, Any]:
    if not bool(config.get("synthetic_terrain_contract_enabled", False)):
        return {}
    sidecar_path = _resolved_file(slice_row.get("sidecar"))
    if sidecar_path is None or not artifact_io.path_is_file(sidecar_path):
        return {}
    try:
        sidecar = _read_json(sidecar_path)
    except Exception:
        return {}
    los_cells = {tuple(cell) for cell in _cell_list_payload(sidecar.get("synthetic_los_blocker_cells"))}
    hard_cells = {tuple(cell) for cell in _cell_list_payload(sidecar.get("synthetic_hard_obstacle_cells"))}
    if not los_cells and not hard_cells:
        return {
            "synthetic_los_blocker_candidate_counts": [0 for _ in candidates],
            "synthetic_hard_obstacle_candidate_counts": [0 for _ in candidates],
        }
    pressure_config = dict(config)
    pressure_config["obstacle_occlusion_enabled"] = False
    los_counts: list[int] = []
    hard_counts: list[int] = []
    for candidate in candidates:
        cell = hf._cell_tuple(hf._candidate_cell(candidate))
        if cell is None or candidate.get("candidate_theta_deg") is None:
            los_counts.append(0)
            hard_counts.append(0)
            continue
        footprint = hf._candidate_coverage_cells(
            start=current_cell,
            end=cell,
            candidate=candidate,
            config=pressure_config,
            obstacle_source_linkage=None,
        )
        los_counts.append(len(footprint & los_cells))
        hard_counts.append(len(footprint & hard_cells))
    return {
        "synthetic_los_blocker_candidate_counts": los_counts,
        "synthetic_hard_obstacle_candidate_counts": hard_counts,
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
    theta_step_deg = float(config.get("theta_step_deg", 45.0) or 45.0)
    probe_candidates, proposal_sets = _continuous_theta_probe_candidate_sets(
        candidates,
        current_theta_deg=current_theta_deg,
        theta_step_deg=theta_step_deg,
        candidate_set_hash_value=candidate_set_hash_value,
    )
    flat_metadata = _hybrid_astar_path_cost_metadata(
        probe_candidates,
        current_cell=current_cell,
        current_theta_deg=current_theta_deg,
        candidate_set_hash_value=candidate_set_hash_value,
        config=config,
        slice_row=slice_row,
        platform_contract_hash=platform_contract_hash,
    )
    metadata = _aggregate_continuous_theta_probe_metadata(flat_metadata, proposal_sets)
    metadata["hybrid_astar_reachability_probe_theta_deg"] = float(current_theta_deg)
    metadata["hybrid_astar_reachability_probe_provenance"] = (
        "continuous_theta_multi_proposal_probe/v1"
    )
    return metadata


def _continuous_theta_probe_candidates(
    candidates: list[dict[str, Any]],
    *,
    current_theta_deg: float,
    candidate_set_hash_value: str,
) -> list[dict[str, Any]]:
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
    return probe_candidates


def _continuous_theta_probe_candidate_sets(
    candidates: list[dict[str, Any]],
    *,
    current_theta_deg: float,
    theta_step_deg: float,
    candidate_set_hash_value: str,
) -> tuple[list[dict[str, Any]], list[list[float]]]:
    probe_candidates: list[dict[str, Any]] = []
    proposal_sets: list[list[float]] = []
    for index, candidate in enumerate(candidates):
        proposals = _continuous_theta_proposal_degs(
            candidate,
            current_theta_deg=current_theta_deg,
            theta_step_deg=theta_step_deg,
        )
        proposal_sets.append(proposals)
        cell = hf._cell_tuple(hf._candidate_cell(candidate))
        for theta_deg in proposals:
            probe = dict(candidate)
            if cell is not None:
                probe["candidate_theta_deg"] = float(theta_deg)
                probe["candidate_viewpoint"] = [int(cell[0]), int(cell[1]), float(theta_deg)]
            probe["candidate_index"] = int(index)
            probe["candidate_set_hash"] = candidate_set_hash_value
            probe_candidates.append(probe)
    return probe_candidates, proposal_sets


def _continuous_theta_proposal_degs(
    candidate: dict[str, Any],
    *,
    current_theta_deg: float,
    theta_step_deg: float,
) -> list[float]:
    viewpoint = candidate.get("candidate_viewpoint")
    viewpoint_theta = None
    if isinstance(viewpoint, list) and len(viewpoint) >= 3:
        viewpoint_theta = _finite(viewpoint[2])
    base = float(current_theta_deg)
    step = float(theta_step_deg) if math.isfinite(float(theta_step_deg)) and float(theta_step_deg) > 0.0 else 45.0
    return _unique_theta_degs(
        [
            candidate.get("candidate_theta_deg"),
            viewpoint_theta,
            base,
            base + step,
            base - step,
        ]
    )


def _aggregate_continuous_theta_probe_metadata(
    flat_metadata: dict[str, Any],
    proposal_sets: list[list[float]],
) -> dict[str, Any]:
    best_indices: list[int] = []
    reachable_sets: list[list[float]] = []
    reachable_flags_by_candidate: list[list[bool]] = []
    costs_by_candidate: list[list[float | None]] = []
    offset = 0
    for proposals in proposal_sets:
        count = len(proposals)
        indices = list(range(offset, offset + count))
        flags = [
            _flat_bool(flat_metadata.get("hybrid_astar_reachable_flags"), flat_index)
            and _flat_value(flat_metadata.get("hybrid_astar_path_costs"), flat_index) is not None
            and _flat_value(flat_metadata.get("hybrid_astar_pose_path_hashes"), flat_index) is not None
            for flat_index in indices
        ]
        costs = [_finite(_flat_value(flat_metadata.get("hybrid_astar_path_costs"), flat_index)) for flat_index in indices]
        reachable_offsets = [local for local, flag in enumerate(flags) if flag]
        reachable_sets.append([float(proposals[local]) for local in reachable_offsets])
        reachable_flags_by_candidate.append(flags)
        costs_by_candidate.append(costs)
        if reachable_offsets:
            best_local = min(
                reachable_offsets,
                key=lambda local: (
                    float("inf") if costs[local] is None else float(costs[local]),
                    local,
                ),
            )
            best_indices.append(indices[best_local])
        elif indices:
            best_indices.append(indices[0])
        else:
            best_indices.append(-1)
        offset += count
    metadata = {
        "path_cost_source": HYBRID_ASTAR_PATH_COST_SOURCE,
        "path_cost_sources": [_flat_value(flat_metadata.get("path_cost_sources"), index) for index in best_indices],
        "hybrid_astar_path_costs": [_flat_value(flat_metadata.get("hybrid_astar_path_costs"), index) for index in best_indices],
        "hybrid_astar_pose_path_hashes": [
            _flat_value(flat_metadata.get("hybrid_astar_pose_path_hashes"), index) for index in best_indices
        ],
        "hybrid_astar_trajectory_kinds": [
            _flat_value(flat_metadata.get("hybrid_astar_trajectory_kinds"), index) for index in best_indices
        ],
        "hybrid_astar_reachable_flags": [bool(items) for items in reachable_sets],
        "hybrid_astar_failure_reasons": [
            _flat_value(flat_metadata.get("hybrid_astar_failure_reasons"), index) for index in best_indices
        ],
        "legacy_grid_astar_path_costs": [
            _flat_value(flat_metadata.get("legacy_grid_astar_path_costs"), index) for index in best_indices
        ],
        "hybrid_vs_grid_path_cost_deltas": [
            _flat_value(flat_metadata.get("hybrid_vs_grid_path_cost_deltas"), index) for index in best_indices
        ],
        "default_astar_replaced": False,
        "default_astar_replaced_flags": [
            _flat_value(flat_metadata.get("default_astar_replaced_flags"), index) is True for index in best_indices
        ],
        "hybrid_astar_ackermann_feasible_claimed": False,
        "hybrid_astar_ackermann_feasible_claimed_flags": [
            _flat_value(flat_metadata.get("hybrid_astar_ackermann_feasible_claimed_flags"), index) is True
            for index in best_indices
        ],
        "platform_contract_hash": flat_metadata.get("platform_contract_hash"),
        "hybrid_astar_current_pose": flat_metadata.get("hybrid_astar_current_pose"),
        "hybrid_astar_current_pose_provenance": flat_metadata.get("hybrid_astar_current_pose_provenance"),
        "hybrid_astar_planning_grid_source": flat_metadata.get("hybrid_astar_planning_grid_source"),
        "planner_grid_resolution_m": flat_metadata.get("planner_grid_resolution_m"),
        "source_grid_resolution_m": flat_metadata.get("source_grid_resolution_m"),
        "planning_proxy_hash": flat_metadata.get("planning_proxy_hash"),
        "closed_key_xy_resolution_m": flat_metadata.get("closed_key_xy_resolution_m"),
        "planning_proxy_candidate_binding": flat_metadata.get("planning_proxy_candidate_binding"),
        "hybrid_astar_theta_proposals_deg_by_candidate": proposal_sets,
        "hybrid_astar_reachable_theta_degs_by_candidate": reachable_sets,
        "hybrid_astar_theta_probe_reachable_flags_by_candidate": reachable_flags_by_candidate,
        "hybrid_astar_theta_probe_path_costs_by_candidate": costs_by_candidate,
    }
    return metadata


def _unique_theta_degs(values: list[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        number = _finite(value)
        if number is None:
            continue
        normalized = float(number) % 360.0
        if not any(abs(((normalized - existing + 180.0) % 360.0) - 180.0) <= 1.0e-6 for existing in result):
            result.append(normalized)
    return result


def _flat_value(values: Any, index: int) -> Any:
    if isinstance(values, list) and 0 <= index < len(values):
        return values[index]
    return None


def _flat_bool(values: Any, index: int) -> bool:
    return _flat_value(values, index) is True


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
            closed_key_xy_resolution_m=planner_options["closed_key_xy_resolution_m"],
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
    if sidecar_path is None or not artifact_io.path_is_file(sidecar_path):
        return _hybrid_path_unavailable_metadata(
            candidates,
            reason="sidecar_missing",
            platform_contract_hash=platform_contract_hash,
            max_traversable_slope_deg=config.get("max_traversable_slope_deg", 30.0),
        )
    try:
        sidecar = _read_json(sidecar_path)
        source_grid = build_cost_grid_from_sidecar(sidecar)
        grid = _hybrid_astar_planning_grid(source_grid, config)
        current_world = _hybrid_cell_center_world(source_grid.spec, HYBRID_CELL(int(current_cell[0]), int(current_cell[1])))
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
        "closed_key_xy_resolution_m": (
            float(config["hybrid_astar_closed_key_xy_resolution_m"])
            if config.get("hybrid_astar_closed_key_xy_resolution_m") is not None
            else None
        ),
    }
    proxy_enabled = grid.metadata.get("planning_grid_source") == "derived_high_res_planning_proxy/v1"

    def payload_for(index: int, candidate: dict[str, Any]) -> dict[str, Any]:
        payload = dict(candidate)
        payload.setdefault("candidate_index", index)
        payload.setdefault("candidate_set_hash", candidate_set_hash_value)
        if proxy_enabled:
            world_pose = _candidate_goal_world_pose_from_source_grid(source_grid, candidate)
            if world_pose is not None:
                payload["candidate_goal_world_pose"] = world_pose
                payload["planning_proxy_candidate_binding"] = "coarse_candidate_cell_center_world_pose/v1"
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
                    closed_key_xy_resolution_m=planner_options["closed_key_xy_resolution_m"],
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
        "hybrid_astar_planning_grid_source": grid.metadata.get("planning_grid_source"),
        "planner_grid_resolution_m": grid.metadata.get("planner_grid_resolution_m"),
        "source_grid_resolution_m": grid.metadata.get("source_grid_resolution_m"),
        "planning_proxy_hash": grid.metadata.get("planning_proxy_hash"),
        "closed_key_xy_resolution_m": planner_options["closed_key_xy_resolution_m"],
        "planning_proxy_candidate_binding": (
            "coarse_candidate_cell_center_world_pose/v1" if proxy_enabled else "native_candidate_grid_cell/v1"
        ),
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


def _no_sampling_candidate_reason(
    *,
    action_mask: tuple[bool, ...],
    hard_risk_clean_mask: tuple[bool, ...],
    hybrid_reachable_mask: tuple[bool, ...],
) -> str:
    if not any(action_mask):
        return "no_action_mask_candidate"
    if not any(bool(valid) and bool(clean) for valid, clean in zip(action_mask, hard_risk_clean_mask)):
        return "no_hard_risk_clean_candidate"
    if not any(
        bool(valid) and bool(clean) and bool(reachable)
        for valid, clean, reachable in zip(action_mask, hard_risk_clean_mask, hybrid_reachable_mask)
    ):
        return "no_hybrid_reachable_candidate_terminal"
    return "no_sampling_candidate_unknown"


def _hybrid_cell_center_world(spec: Any, cell: HYBRID_CELL) -> HYBRID_WORLD_POINT:
    return HYBRID_WORLD_POINT(
        float(spec.origin[0]) + (float(cell.x) + 0.5) * float(spec.resolution),
        float(spec.origin[1]) + (float(cell.y) + 0.5) * float(spec.resolution),
    )


def _hybrid_astar_planning_grid(source_grid: Any, config: dict[str, Any]) -> Any:
    if config.get("hybrid_astar_planning_grid_source") != "derived_high_res_planning_proxy/v1":
        return source_grid
    return build_derived_high_res_planning_proxy_grid(
        source_grid,
        float(config.get("planner_grid_resolution_m") or 1.0),
    )


def _candidate_goal_world_pose_from_source_grid(source_grid: Any, candidate: dict[str, Any]) -> list[float] | None:
    viewpoint = candidate.get("candidate_viewpoint")
    if not isinstance(viewpoint, (list, tuple)) or len(viewpoint) < 2:
        return None
    try:
        cell = HYBRID_CELL(int(viewpoint[0]), int(viewpoint[1]))
    except (TypeError, ValueError):
        return None
    if not source_grid.spec.in_bounds(cell):
        return None
    world = _hybrid_cell_center_world(source_grid.spec, cell)
    return [float(world.x), float(world.y)]


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
    trainable_by_scenario: dict[str, int] = {}
    for row in trainable:
        scenario_id = str(row.get("scenario_id") or row.get("info", {}).get("scenario_id") or "")
        if scenario_id:
            trainable_by_scenario[scenario_id] = trainable_by_scenario.get(scenario_id, 0) + 1
    terminal_step_histogram: dict[str, int] = {}
    for row in collection.rejections:
        if row.get("reason") in {
            "selected_continuous_theta_hybrid_astar_unreachable",
            "no_selected_reachable_pose_candidate_terminal",
            "no_hybrid_reachable_candidate_terminal",
        }:
            step_index = _int_or_none(row.get("step_index"))
            bucket = str(step_index if step_index is not None else "unknown")
            terminal_step_histogram[bucket] = terminal_step_histogram.get(bucket, 0) + 1
    return {
        "episode_count": len(collection.episodes),
        "sampled_transition_count": len(collection.transitions),
        "trainable_transition_count": len(trainable),
        "trainable_transition_count_by_scenario": trainable_by_scenario,
        "scenario_early_terminal_step_histogram": terminal_step_histogram,
        "mask_violation_count": sum(1 for row in trainable if row["info"]["action_mask"][int(row["action_index"])] is not True),
        "hard_risk_violation_count": sum(1 for row in trainable if row["info"].get("hard_risk_violation") is True),
        "non_finite_old_log_prob_count": sum(1 for row in trainable if _finite(row.get("old_log_prob")) is None),
        "non_finite_old_value_count": sum(1 for row in trainable if _finite(row.get("old_value")) is None),
        "non_finite_reward_count": sum(1 for row in trainable if _finite(row.get("reward")) is None),
        "old_log_prob_recompute_max_abs_error": max(log_errors) if log_errors else 0.0,
        "terminal_transition_count": sum(1 for row in trainable if row.get("done") is True),
        "transition_with_next_observation_count": sum(1 for row in trainable if row.get("next_observation") is not None),
        "selected_continuous_theta_unreachable_attempt_count": sum(
            1 for row in trainable if row.get("selected_continuous_theta_unreachable_attempted") is True
        ),
        "selected_continuous_theta_resample_success_count": sum(
            1 for row in trainable if row.get("selected_theta_resampled_for_reachability") is True
        ),
        "selected_candidate_resample_success_count": sum(
            1 for row in trainable if row.get("selected_candidate_resampled_for_reachability") is True
        ),
        "selected_pose_unreachable_terminal_count": sum(
            1 for row in collection.rejections if row.get("reason") == "no_selected_reachable_pose_candidate_terminal"
        ),
        "synthetic_credit_target_selected_count_from_transition_rows": sum(
            1 for row in trainable if row.get("synthetic_credit_target_selected") is True
        ),
        "no_hybrid_reachable_candidate_terminal_count": sum(
            1 for row in collection.rejections if row.get("reason") == "no_hybrid_reachable_candidate_terminal"
        ),
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
        "selected_continuous_theta_reachability_guard_enabled": bool(
            config.get("selected_continuous_theta_reachability_guard_enabled", False)
        ),
        "selected_continuous_theta_unreachable_resample_policy": str(
            config.get("selected_continuous_theta_unreachable_resample_policy") or "terminal/v1"
        ),
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
    canonical_artifacts = {
        "summary": str(artifact_path(output_root, STAGE21_1_SUMMARY).resolve()),
        "episodes": str(artifact_path(output_root, STAGE21_1_EPISODES).resolve()),
        "transitions": str(artifact_path(output_root, STAGE21_1_TRANSITIONS).resolve()),
        "trainable_batch": str(artifact_path(output_root, STAGE21_1_TRAINABLE).resolve()),
        "rejection_report": str(artifact_path(output_root, STAGE21_1_REJECTIONS).resolve()),
    }
    legacy_artifacts = {
        "summary": summary["summary"],
        "episodes": summary["episodes"],
        "transitions": summary["transitions"],
        "trainable_batch": summary["trainable_batch"],
        "rejection_report": summary["rejection_report"],
    }
    summary["canonical_artifacts"] = canonical_artifacts
    summary["legacy_artifacts"] = legacy_artifacts
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
        "canonical_artifacts": canonical_artifacts,
        "legacy_artifacts": legacy_artifacts,
        "model_audit": collection.model_audit,
        "summary_status": status,
        "next_required_change": route,
    }
    write_jsonl_artifact(output_root, STAGE21_1_EPISODES, collection.episodes)
    write_jsonl_artifact(output_root, STAGE21_1_TRANSITIONS, collection.transitions)
    write_jsonl_artifact(output_root, STAGE21_1_TRAINABLE, collection.trainable_batch)
    write_jsonl_artifact(output_root, STAGE21_1_REJECTIONS, collection.rejections)
    _write_jsonl(output_root / REWARD_AUDIT_FILE, collection.reward_audit)
    _write_jsonl(output_root / SAMPLING_AUDIT_FILE, collection.sampling_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    write_json_artifact(output_root, STAGE21_1_SUMMARY, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    _write_text(output_root / REPORT_FILE, _render_report(summary))
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
    config["synthetic_credit_feature_exposure_enabled"] = bool(
        config.get("synthetic_credit_feature_exposure_enabled", False)
    )
    config["synthetic_exploration_credit_enabled"] = bool(
        config.get("synthetic_exploration_credit_enabled", False)
    )
    config["synthetic_credit_mixture_probability"] = _fraction_float(
        config.get("synthetic_credit_mixture_probability", 0.35),
        "synthetic_credit_mixture_probability",
    )
    config["selected_continuous_theta_reachability_guard_enabled"] = bool(
        config.get("selected_continuous_theta_reachability_guard_enabled", False)
    )
    config["selected_continuous_theta_unreachable_resample_policy"] = str(
        config.get("selected_continuous_theta_unreachable_resample_policy") or "terminal/v1"
    )
    if config["selected_continuous_theta_reachability_guard_enabled"]:
        if config["selected_continuous_theta_unreachable_resample_policy"] != REACHABILITY_GUARD_THETA_POLICY_ID:
            raise ConfigError(
                "selected_continuous_theta_unreachable_resample_policy must be "
                f"{REACHABILITY_GUARD_THETA_POLICY_ID} when the reachability guard is enabled"
            )
        if not config["continuous_theta_action_space_enabled"]:
            raise ConfigError("selected_continuous_theta_reachability_guard_enabled requires continuous theta action space")
        if not bool(config.get("hybrid_astar_pose_path_cost_enabled", False)):
            raise ConfigError("selected_continuous_theta_reachability_guard_enabled requires Hybrid A* path cost")
    config["hybrid_astar_pose_path_cost_enabled"] = bool(
        config.get("hybrid_astar_pose_path_cost_enabled", False)
    )
    config["hybrid_astar_candidate_eval_workers"] = _positive_int(
        config.get("hybrid_astar_candidate_eval_workers", 1),
        "hybrid_astar_candidate_eval_workers",
    )
    if config.get("hybrid_astar_planning_grid_source") is not None:
        config["hybrid_astar_planning_grid_source"] = str(config["hybrid_astar_planning_grid_source"])
        if config["hybrid_astar_planning_grid_source"] != "derived_high_res_planning_proxy/v1":
            raise ConfigError("hybrid_astar_planning_grid_source must be derived_high_res_planning_proxy/v1 when provided")
    if config.get("planner_grid_resolution_m") is not None:
        config["planner_grid_resolution_m"] = _positive_float(
            config.get("planner_grid_resolution_m"),
            "planner_grid_resolution_m",
        )
    if config.get("hybrid_astar_closed_key_xy_resolution_m") is not None:
        config["hybrid_astar_closed_key_xy_resolution_m"] = _positive_float(
            config.get("hybrid_astar_closed_key_xy_resolution_m"),
            "hybrid_astar_closed_key_xy_resolution_m",
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
        "hybrid_astar_planning_grid_source",
        "planner_grid_resolution_m",
        "hybrid_astar_closed_key_xy_resolution_m",
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
    if not artifact_io.path_is_file(stage21_0_summary):
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


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


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


def _planning_proxy_rejection_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "hybrid_astar_planning_grid_source": metadata.get("hybrid_astar_planning_grid_source"),
        "planner_grid_resolution_m": metadata.get("planner_grid_resolution_m"),
        "source_grid_resolution_m": metadata.get("source_grid_resolution_m"),
        "closed_key_xy_resolution_m": metadata.get("closed_key_xy_resolution_m"),
        "planning_proxy_hash": metadata.get("planning_proxy_hash"),
        "planning_proxy_candidate_binding": metadata.get("planning_proxy_candidate_binding"),
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
    artifact_io.write_json(path, payload)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    artifact_io.write_jsonl(path, rows)


def _write_text(path: Path, text: str) -> None:
    artifact_io.write_text(path, text)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = artifact_io.read_json(path)
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


def _fraction_float(value: Any, field: str) -> float:
    parsed = _nonnegative_float(value, field)
    if parsed > 1.0:
        raise ConfigError(f"{field} must be <= 1")
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
