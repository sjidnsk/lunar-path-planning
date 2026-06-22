from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
MODEL_EXPLORER_SRC = SCRIPT_DIR.parent / "model-explorer" / "src"
if str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))

from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now
from global_99_governance_common import global_99_boundary_defaults
from model_explorer.policy.canonical_reward import (
    CanonicalRewardGuardProfile,
    compute_canonical_reward_components,
    load_canonical_reward_profile,
)


CONFIG_SCHEMA_VERSION = "xunce-stage18-11-path-cost-weight-calibration-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage18-11-path-cost-weight-calibration-summary/v1"
REPLAY_ROW_SCHEMA_VERSION = "xunce-stage18-11-weight-replay-row/v1"
MATRIX_SCHEMA_VERSION = "xunce-stage18-11-count-weight-matrix/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage18-11-next-stage-routing/v1"
COMMAND_PLAN_SCHEMA_VERSION = "xunce-stage18-11-reward-rerank-command-plan/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage18-11-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage18_11_path_cost_weight_calibration_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage18_11_path_cost_weight_calibration/"
    "outputs/path_feedback_batch_xunce_stage18_11_path_cost_weight_calibration_v1"
)

COVERAGE_SUMMARY = "xunce-exploration-coverage-comparison-summary.json"
COVERAGE_AGGREGATE = "xunce-exploration-coverage-comparison-aggregate.json"
COVERAGE_MANIFEST = "xunce-exploration-coverage-comparison-manifest.json"
COVERAGE_EPISODES = "xunce-exploration-coverage-episodes.jsonl"
COVERAGE_PAIRS = "xunce-exploration-coverage-comparison-pairs.jsonl"
PAIRED_DECISION_AUDIT = "xunce-exploration-coverage-paired-decision-audit.jsonl"
CANDIDATE_METRIC_AUDIT = "xunce-exploration-coverage-candidate-metric-audit.jsonl"

SUMMARY_FILE = "xunce-stage18-11-path-cost-weight-calibration-summary.json"
REPLAY_RESULTS_FILE = "xunce-stage18-11-weight-replay-results.jsonl"
MATRIX_FILE = "xunce-stage18-11-count-weight-matrix.json"
COMMAND_PLAN_FILE = "xunce-stage18-11-reward-rerank-command-plan.json"
ROUTING_FILE = "xunce-stage18-11-next-stage-routing.json"
REPORT_FILE = "xunce-stage18-11-report.md"
MANIFEST_FILE = "xunce-stage18-11-manifest.json"

ROUTE_BOUNDARY = "resolve_stage18_11_boundary_rejections"
ROUTE_INPUTS = "rerun_stage18_11_required_inputs"
ROUTE_RISK = "repair_path_risk_boundary_filtering"
ROUTE_DIAGNOSTIC = "run_stage18_11_reward_rerank_diagnostic_rollouts"
ROUTE_STAGE18_12 = "stage18_12_rollout_horizon_or_mission_budget_scaling_for_99pct_coverage"
ROUTE_CONTINUE = "continue_path_cost_weight_calibration_at_99pct_coverage"
ROUTE_PREFLIGHT = "prepare_stage19_evaluator_critic_preflight"

BOUNDARY_FIELDS = tuple(global_99_boundary_defaults()) + (
    "stage19_authorized",
    "default_policy_replacement_approved",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
    "runs_new_ppo_update",
    "real_world_release_approved",
    "real_world_performance_claimed",
    "modifies_network",
    "modifies_action_space",
    "modifies_default_astar",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate Xunce Stage 18.11 path cost reward weights.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_stage18_11_path_cost_weight_calibration(
            config_path=resolve_path(Path(args.config), repo_root).resolve(),
            output_root=resolve_path(Path(args.output_root), repo_root).resolve(),
            repo_root=repo_root,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": summary["status"],
                "next_required_change": summary["next_required_change"],
                "best_replay_candidate_count": summary.get("best_replay_candidate_count"),
                "best_replay_path_cost_weight": summary.get("best_replay_path_cost_weight"),
                "stage19_authorized": summary["stage19_authorized"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] in {"passed", "partial"} else 1


def run_xunce_stage18_11_path_cost_weight_calibration(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path=config_path, repo_root=repo_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    observed_profile = load_canonical_reward_profile(config["canonical_observed_profile"])
    if observed_profile.profile_version != "v3":
        raise ConfigError("Stage 18.11 requires canonical observed profile_version v3")
    variants = [load_canonical_reward_profile(path) for path in config["profile_variants"]]
    _validate_profile_variants(observed_profile, variants)
    expected_identity = _profile_identity(observed_profile)

    blocking: list[str] = []
    diagnostic: list[str] = []
    blocking.extend(_boundary_violations(config))

    sweeps = [_load_sweep(sweep, expected_identity) for sweep in config["sweeps"]]
    for sweep in sweeps:
        blocking.extend(sweep["blocking_reason_codes"])
        diagnostic.extend(sweep["diagnostic_reason_codes"])

    replay_rows, matrix = _weight_replay_matrix(sweeps=sweeps, variants=variants)
    best_replay = _best_replay_combo(matrix["rows"])
    command_plan = _command_plan(config=config, matrix_rows=matrix["rows"], best_replay=best_replay, repo_root=repo_root)
    diagnostic_rollouts = _diagnostic_rollouts(config=config, command_plan=command_plan)
    diagnostic_summary = _diagnostic_rollout_summary(diagnostic_rollouts)

    observed_best = _observed_best_coverage(sweeps)
    hard_risk_violation_count = _hard_risk_violation_count(sweeps, matrix["rows"], diagnostic_summary)
    target = float(config["target_final_coverage_rate"])
    route = _route(
        blocking=unique_sorted(blocking),
        hard_risk_violation_count=hard_risk_violation_count,
        observed_best=observed_best,
        diagnostic_summary=diagnostic_summary,
        target_final_coverage_rate=target,
    )
    status = "partial" if route == ROUTE_DIAGNOSTIC else ("passed" if route == ROUTE_PREFLIGHT else "failed")

    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "primary_route": route,
        "stage19_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": status,
        "reason_codes": unique_sorted([*blocking, *diagnostic, *diagnostic_summary["reason_codes"]]),
        "blocking_reason_codes": unique_sorted(blocking),
        "diagnostic_reason_codes": unique_sorted([*diagnostic, *diagnostic_summary["reason_codes"]]),
        "canonical_observed_profile": str(Path(config["canonical_observed_profile"]).resolve()),
        "profile_id": observed_profile.profile_id,
        "profile_version": observed_profile.profile_version,
        "profile_hash": observed_profile.profile_hash,
        "profile_variant_count": len(variants),
        "profile_variants": [_profile_variant_row(profile, path) for profile, path in zip(variants, config["profile_variants"])],
        "target_final_coverage_rate": target,
        "observed_best_final_coverage_rate_mean": observed_best["mean"],
        "observed_best_final_coverage_rate_max": observed_best["max"],
        "best_replay_candidate_count": best_replay.get("candidate_count") if best_replay else None,
        "best_replay_path_cost_weight": best_replay.get("path_cost_weight") if best_replay else None,
        "best_replay_profile_hash": best_replay.get("profile_hash") if best_replay else None,
        "best_diagnostic_rollout_candidate_count": diagnostic_summary["best_candidate_count"],
        "best_diagnostic_rollout_path_cost_weight": diagnostic_summary["best_path_cost_weight"],
        "best_diagnostic_final_coverage_rate_mean": diagnostic_summary["best_final_coverage_rate_mean"],
        "best_diagnostic_final_coverage_rate_max": diagnostic_summary["best_final_coverage_rate_max"],
        "hard_risk_violation_count": hard_risk_violation_count,
        "sweep_count": len(sweeps),
        "complete_sweep_count": sum(1 for sweep in sweeps if sweep["input_valid"]),
        "weight_replay_row_count": len(replay_rows),
        "count_weight_matrix": matrix,
        "diagnostic_rollout_summary": diagnostic_summary,
        "next_required_change": route,
        "next_stage_routing": routing,
        "stage19_authorized": False,
        "stage19_readiness": {
            "schema_version": "xunce-stage18-11-stage19-readiness/v1",
            "readiness": "ready_for_stage19_preflight_human_review_only"
            if route == ROUTE_PREFLIGHT
            else "not_authorized",
            "authorized": False,
        },
        "counterfactual_type": "one_step_observed_candidate_set_replay",
        "does_not_claim_trajectory_outcome": True,
        "does_not_claim_xunce_policy_changed": True,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "default_policy_replacement_approved": False,
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
        "summary": str(output_root / SUMMARY_FILE),
    }

    _write_jsonl(output_root / REPLAY_RESULTS_FILE, replay_rows)
    _write_json(output_root / MATRIX_FILE, matrix)
    _write_json(output_root / COMMAND_PLAN_FILE, command_plan)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    _write_json(
        output_root / MANIFEST_FILE,
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generated_at": summary["generated_at"],
            "summary": str(output_root / SUMMARY_FILE),
            "artifacts": [
                SUMMARY_FILE,
                REPLAY_RESULTS_FILE,
                MATRIX_FILE,
                COMMAND_PLAN_FILE,
                ROUTING_FILE,
                REPORT_FILE,
                MANIFEST_FILE,
            ],
            "stage19_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
        },
    )
    return summary


def _load_config(*, config_path: Path, repo_root: Path) -> dict[str, Any]:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")

    config = dict(payload)
    for key in ("artifact_workspace_root", "canonical_observed_profile", "diagnostic_rollout_workspace_root"):
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{key} must be a non-empty path string")
        config[key] = str(resolve_path(Path(value), repo_root).resolve())

    raw_variants = config.get("profile_variants")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ConfigError("profile_variants must be a non-empty array")
    config["profile_variants"] = [
        str(resolve_path(Path(_require_string(path, f"profile_variants[{index}]")), repo_root).resolve())
        for index, path in enumerate(raw_variants)
    ]

    raw_sweeps = config.get("sweeps")
    if not isinstance(raw_sweeps, list) or not raw_sweeps:
        raise ConfigError("sweeps must be a non-empty array")
    sweeps: list[dict[str, Any]] = []
    for index, item in enumerate(raw_sweeps):
        if not isinstance(item, dict):
            raise ConfigError(f"sweeps[{index}] must be an object")
        sweeps.append(
            {
                "candidate_count": _positive_int(item.get("candidate_count"), f"sweeps[{index}].candidate_count"),
                "proposal_pool_limit": _positive_int(item.get("proposal_pool_limit"), f"sweeps[{index}].proposal_pool_limit"),
                "coverage_comparison_root": str(
                    resolve_path(Path(_require_string(item.get("coverage_comparison_root"), f"sweeps[{index}].coverage_comparison_root")), repo_root).resolve()
                ),
            }
        )
    config["sweeps"] = sweeps

    target = _finite(config.get("target_final_coverage_rate"))
    if target is None or target <= 0.0 or target > 1.0:
        raise ConfigError("target_final_coverage_rate must be in (0, 1]")
    config["target_final_coverage_rate"] = target
    config["rollout_steps"] = _positive_int(config.get("rollout_steps", 40), "rollout_steps")
    config["required_scenario_count"] = _positive_int(config.get("required_scenario_count", 24), "required_scenario_count")
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        raise ConfigError("canary_traffic_fraction must be 0.0")
    return config


def _load_sweep(sweep: dict[str, Any], expected_identity: dict[str, str]) -> dict[str, Any]:
    root = Path(sweep["coverage_comparison_root"])
    blocking: list[str] = []
    diagnostic: list[str] = []
    summary = _read_json(root / COVERAGE_SUMMARY, blocking, "missing_coverage_summary")
    aggregate = _read_json(root / COVERAGE_AGGREGATE, blocking, "missing_coverage_aggregate")
    manifest = _read_json(root / COVERAGE_MANIFEST, blocking, "missing_coverage_manifest")
    episodes = _read_jsonl(root / COVERAGE_EPISODES, blocking, "missing_coverage_episodes")
    pairs = _read_jsonl(root / COVERAGE_PAIRS, blocking, "missing_coverage_pairs")
    paired = _read_jsonl(root / PAIRED_DECISION_AUDIT, blocking, "missing_paired_decision_audit")
    candidates = _read_jsonl(root / CANDIDATE_METRIC_AUDIT, blocking, "missing_candidate_metric_audit")

    for payload in (summary, aggregate, manifest):
        if payload:
            _validate_profile_lineage(payload, expected_identity, blocking)
            blocking.extend(_boundary_violations(payload))
    _validate_candidate_metric_lineage(candidates, expected_identity, blocking)
    _validate_paired_candidate_key_coverage(paired, candidates, blocking)

    observed = _observed_coverage_stats(episodes)
    return {
        "candidate_count": sweep["candidate_count"],
        "proposal_pool_limit": sweep["proposal_pool_limit"],
        "coverage_comparison_root": str(root.resolve()),
        "input_valid": not blocking,
        "summary": summary,
        "aggregate": aggregate,
        "manifest": manifest,
        "episodes": episodes,
        "pairs": pairs,
        "paired_decision_rows": paired,
        "candidate_metric_rows": candidates,
        "coverage_denominator_cells": _coverage_denominator(summary, manifest),
        "observed_final_coverage_rate_mean": observed["mean"],
        "observed_final_coverage_rate_max": observed["max"],
        "hard_risk_violation_count": _sum_numeric(episodes, "hard_risk_violation_count"),
        "blocking_reason_codes": unique_sorted(blocking),
        "diagnostic_reason_codes": unique_sorted(diagnostic),
    }


def _weight_replay_matrix(
    *,
    sweeps: list[dict[str, Any]],
    variants: list[CanonicalRewardGuardProfile],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    replay_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    for sweep in sweeps:
        if not sweep["input_valid"]:
            continue
        candidates_by_key = _candidate_rows_by_key(sweep["candidate_metric_rows"])
        for profile in variants:
            profile_rows: list[dict[str, Any]] = []
            for paired_row in sweep["paired_decision_rows"]:
                key = _decision_key(paired_row)
                candidate_rows = candidates_by_key.get(key, [])
                if not candidate_rows:
                    continue
                row = _replay_step(
                    sweep=sweep,
                    profile=profile,
                    paired_row=paired_row,
                    candidate_rows=candidate_rows,
                )
                replay_rows.append(row)
                profile_rows.append(row)
            matrix_rows.append(_matrix_row(sweep, profile, profile_rows))
    return replay_rows, {"schema_version": MATRIX_SCHEMA_VERSION, "rows": matrix_rows}


def _replay_step(
    *,
    sweep: dict[str, Any],
    profile: CanonicalRewardGuardProfile,
    paired_row: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    denominator = max(float(sweep["coverage_denominator_cells"]), 1.0)
    scored: list[tuple[float, float, int, dict[str, Any], dict[str, Any]]] = []
    hard_rejected = 0
    for candidate in candidate_rows:
        hard_risk = _candidate_hard_risk(candidate)
        if hard_risk:
            hard_rejected += 1
            continue
        if candidate.get("action_mask_valid") is False:
            continue
        metrics = _candidate_reward_metrics(candidate, denominator=denominator)
        result = compute_canonical_reward_components(metrics, profile)
        coverage = _finite(candidate.get("expected_new_coverage_cell_count")) or 0.0
        index = int(candidate.get("candidate_index", 0) or 0)
        scored.append((result.reward, coverage, -index, candidate, result.components))

    selected_candidate: dict[str, Any] | None = None
    selected_components: dict[str, float] = {}
    selected_reward: float | None = None
    if scored:
        reward, _coverage, _neg_index, selected_candidate, selected_components = max(scored)
        selected_reward = reward

    xunce_candidate = _candidate_by_index(candidate_rows, paired_row.get("xunce_selected_action_index"))
    incumbent_candidate = _candidate_by_index(candidate_rows, paired_row.get("incumbent_selected_action_index"))
    return {
        "schema_version": REPLAY_ROW_SCHEMA_VERSION,
        "counterfactual_type": "one_step_observed_candidate_set_replay",
        "does_not_claim_trajectory_outcome": True,
        "does_not_claim_xunce_policy_changed": True,
        "candidate_count": sweep["candidate_count"],
        "proposal_pool_limit": sweep["proposal_pool_limit"],
        "path_cost_weight": profile.soft_reward_components["path_cost_weight"],
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
        "scenario_id": paired_row.get("scenario_id"),
        "executing_policy": paired_row.get("executing_policy"),
        "step_index": paired_row.get("step_index"),
        "candidate_set_hash": paired_row.get("candidate_set_hash"),
        "covered_cells_hash": paired_row.get("covered_cells_hash"),
        "xunce_selected_action_index": paired_row.get("xunce_selected_action_index"),
        "incumbent_selected_action_index": paired_row.get("incumbent_selected_action_index"),
        "reward_rerank_selected_action_index": selected_candidate.get("candidate_index") if selected_candidate else None,
        "reward_rerank_reward": selected_reward,
        "reward_rerank_components": selected_components,
        "reward_rerank_expected_new_coverage_cell_count": _metric_from_candidate(selected_candidate, "expected_new_coverage_cell_count"),
        "reward_rerank_path_cost": _metric_from_candidate(selected_candidate, "path_cost"),
        "reward_rerank_soft_risk_exposure": _soft_risk_exposure(selected_candidate or {}),
        "xunce_selected_expected_new_coverage_cell_count": _metric_from_candidate(xunce_candidate, "expected_new_coverage_cell_count"),
        "xunce_selected_path_cost": _metric_from_candidate(xunce_candidate, "path_cost"),
        "incumbent_selected_expected_new_coverage_cell_count": _metric_from_candidate(incumbent_candidate, "expected_new_coverage_cell_count"),
        "incumbent_selected_path_cost": _metric_from_candidate(incumbent_candidate, "path_cost"),
        "hard_risk_rejected_candidate_count": hard_rejected,
    }


def _matrix_row(
    sweep: dict[str, Any],
    profile: CanonicalRewardGuardProfile,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "candidate_count": sweep["candidate_count"],
        "proposal_pool_limit": sweep["proposal_pool_limit"],
        "path_cost_weight": profile.soft_reward_components["path_cost_weight"],
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
        "replay_step_count": len(rows),
        "counterfactual_type": "one_step_observed_candidate_set_replay",
        "does_not_claim_trajectory_outcome": True,
        "does_not_claim_xunce_policy_changed": True,
        "reward_rerank_coverage_proxy_sum": _sum_field(rows, "reward_rerank_expected_new_coverage_cell_count"),
        "reward_rerank_path_cost_proxy_sum": _sum_field(rows, "reward_rerank_path_cost"),
        "reward_rerank_soft_risk_exposure_proxy_sum": _sum_field(rows, "reward_rerank_soft_risk_exposure"),
        "xunce_selected_coverage_proxy_sum": _sum_field(rows, "xunce_selected_expected_new_coverage_cell_count"),
        "xunce_selected_path_cost_proxy_sum": _sum_field(rows, "xunce_selected_path_cost"),
        "incumbent_selected_coverage_proxy_sum": _sum_field(rows, "incumbent_selected_expected_new_coverage_cell_count"),
        "incumbent_selected_path_cost_proxy_sum": _sum_field(rows, "incumbent_selected_path_cost"),
        "hard_risk_rejected_candidate_count": int(_sum_field(rows, "hard_risk_rejected_candidate_count")),
        "observed_final_coverage_rate_mean": sweep["observed_final_coverage_rate_mean"],
        "observed_final_coverage_rate_max": sweep["observed_final_coverage_rate_max"],
    }


def _command_plan(
    *,
    config: dict[str, Any],
    matrix_rows: list[dict[str, Any]],
    best_replay: dict[str, Any] | None,
    repo_root: Path,
) -> dict[str, Any]:
    selected = _diagnostic_combo_selection(config, matrix_rows, best_replay)
    commands = []
    workspace = Path(config["diagnostic_rollout_workspace_root"])
    for combo in selected:
        count = int(combo["candidate_count"])
        weight_label = _weight_label(float(combo["path_cost_weight"]))
        profile_path = combo["profile_path"]
        output_root = workspace / f"count_{count:03d}_{weight_label}" / "stage18_4e"
        commands.append(
            {
                "name": combo["name"],
                "candidate_count": count,
                "proposal_pool_limit": int(combo["proposal_pool_limit"]),
                "path_cost_weight": float(combo["path_cost_weight"]),
                "canonical_reward_rerank_profile": str(profile_path),
                "output_root": str(output_root),
                "command": [
                    "python",
                    "scripts/run_xunce_high_fidelity_exploration_coverage_comparison.py",
                    "--config",
                    "configs/xunce_high_fidelity_exploration_coverage_comparison_stage18_9_strict_v3.json",
                    "--output-root",
                    str(output_root),
                    "--repo-root",
                    ".",
                    "--rollout-steps",
                    str(config["rollout_steps"]),
                    "--candidate-refresh-mode",
                    "dynamic_frontier_nbv_in_process",
                    "--dynamic-candidate-validation-mode",
                    "in_process_path_planner_astar_batch",
                    "--dynamic-candidate-generation-mode",
                    "map_aware_coverage_frontier_nbv",
                    "--dynamic-candidate-selection-mode",
                    "validated_pareto_diverse",
                    "--coverage-metric-mode",
                    "path_line_plus_endpoint",
                    "--dynamic-max-candidates-per-step",
                    str(count),
                    "--dynamic-proposal-pool-limit-per-step",
                    str(combo["proposal_pool_limit"]),
                    "--include-oracle-baselines",
                    "--include-roi-weighted-coverage",
                    "--emit-candidate-metric-audit",
                    "--include-canonical-reward-rerank-oracle",
                    "--canonical-reward-rerank-profile",
                    str(Path(profile_path).resolve()),
                ],
            }
        )
    return {
        "schema_version": COMMAND_PLAN_SCHEMA_VERSION,
        "generated_for_repo_root": str(repo_root.resolve()),
        "diagnostic_rollout_count": len(commands),
        "commands": commands,
    }


def _diagnostic_combo_selection(
    config: dict[str, Any],
    matrix_rows: list[dict[str, Any]],
    best_replay: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    profiles_by_weight = {
        float(load_canonical_reward_profile(path).soft_reward_components["path_cost_weight"]): path
        for path in config["profile_variants"]
    }
    sweeps_by_count = {int(sweep["candidate_count"]): sweep for sweep in config["sweeps"]}
    selected: list[dict[str, Any]] = []

    def add(name: str, count: int, weight: float) -> None:
        if count not in sweeps_by_count or weight not in profiles_by_weight:
            return
        if any(item["candidate_count"] == count and item["path_cost_weight"] == weight for item in selected):
            return
        selected.append(
            {
                "name": name,
                "candidate_count": count,
                "proposal_pool_limit": sweeps_by_count[count]["proposal_pool_limit"],
                "path_cost_weight": weight,
                "profile_path": profiles_by_weight[weight],
            }
        )

    add("mandatory_coverage_reference", 6, 0.0)
    add("mandatory_current_best_efficiency_reference", 24, 0.2)
    ranked = _ranked_replay_combos(matrix_rows)
    if ranked:
        top_coverage = float(ranked[0]["reward_rerank_coverage_proxy_sum"] or 0.0)
        for row in ranked:
            add("top_replay_combo_1" if len(selected) < 3 else "top_replay_combo_2", int(row["candidate_count"]), float(row["path_cost_weight"]))
            if len(selected) >= 3:
                break
        for row in sorted(
            [row for row in ranked if float(row["reward_rerank_coverage_proxy_sum"] or 0.0) >= top_coverage * 0.95],
            key=lambda row: (float(row["reward_rerank_path_cost_proxy_sum"] or 0.0), -float(row["reward_rerank_coverage_proxy_sum"] or 0.0)),
        ):
            add("top_replay_combo_2", int(row["candidate_count"]), float(row["path_cost_weight"]))
            if len(selected) >= 4:
                break
        for row in ranked:
            add("top_replay_combo_fill", int(row["candidate_count"]), float(row["path_cost_weight"]))
            if len(selected) >= 4:
                break
    return selected[:4]


def _diagnostic_rollouts(config: dict[str, Any], command_plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for command in command_plan["commands"]:
        root = Path(command["output_root"])
        summary = _read_json(root / COVERAGE_SUMMARY, [], "missing_diagnostic_rollout_summary")
        episodes = _read_jsonl(root / COVERAGE_EPISODES, [], "missing_diagnostic_rollout_episodes")
        complete = bool(summary and episodes)
        oracle_episodes = [
            row for row in episodes if (row.get("policy") or row.get("policy_name")) == "canonical_reward_rerank_oracle"
        ]
        rows.append(
            {
                "name": command["name"],
                "candidate_count": command["candidate_count"],
                "proposal_pool_limit": command["proposal_pool_limit"],
                "path_cost_weight": command["path_cost_weight"],
                "root": str(root),
                "complete": complete,
                "final_coverage_rate_mean": _mean(
                    [_finite(row.get("final_coverage_rate")) for row in oracle_episodes]
                ),
                "final_coverage_rate_max": _max(
                    [_finite(row.get("final_coverage_rate")) for row in oracle_episodes]
                ),
                "hard_risk_violation_count": _sum_numeric(oracle_episodes, "hard_risk_violation_count"),
                "path_cost_total_m_mean": _mean([_finite(row.get("path_cost_total_m")) for row in oracle_episodes]),
                "soft_risk_exposure_total_mean": _mean(
                    [_finite(row.get("soft_risk_exposure_total")) for row in oracle_episodes]
                ),
            }
        )
    return rows


def _diagnostic_rollout_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row["complete"]]
    best = _best_diagnostic_row(complete)
    reasons = []
    if len(complete) < len(rows):
        reasons.append("diagnostic_reward_rerank_rollouts_missing")
    return {
        "diagnostic_rollout_count": len(rows),
        "complete_diagnostic_rollout_count": len(complete),
        "reason_codes": reasons,
        "rows": rows,
        "best_candidate_count": best.get("candidate_count") if best else None,
        "best_path_cost_weight": best.get("path_cost_weight") if best else None,
        "best_final_coverage_rate_mean": best.get("final_coverage_rate_mean") if best else None,
        "best_final_coverage_rate_max": best.get("final_coverage_rate_max") if best else None,
        "best_hard_risk_violation_count": best.get("hard_risk_violation_count") if best else None,
        "best_path_cost_total_m_mean": best.get("path_cost_total_m_mean") if best else None,
        "best_soft_risk_exposure_total_mean": best.get("soft_risk_exposure_total_mean") if best else None,
    }


def _route(
    *,
    blocking: list[str],
    hard_risk_violation_count: float,
    observed_best: dict[str, float | None],
    diagnostic_summary: dict[str, Any],
    target_final_coverage_rate: float,
) -> str:
    blockers = set(blocking)
    if blockers.intersection(BOUNDARY_FIELDS) or "canary_traffic_fraction" in blockers:
        return ROUTE_BOUNDARY
    if blockers:
        return ROUTE_INPUTS
    if hard_risk_violation_count > 0:
        return ROUTE_RISK
    if diagnostic_summary["complete_diagnostic_rollout_count"] < diagnostic_summary["diagnostic_rollout_count"]:
        return ROUTE_DIAGNOSTIC

    best_mean = max(
        _finite(observed_best.get("mean")) or 0.0,
        _finite(diagnostic_summary.get("best_final_coverage_rate_mean")) or 0.0,
    )
    best_max = max(
        _finite(observed_best.get("max")) or 0.0,
        _finite(diagnostic_summary.get("best_final_coverage_rate_max")) or 0.0,
    )
    if best_mean < target_final_coverage_rate and best_max < target_final_coverage_rate:
        return ROUTE_STAGE18_12
    if (
        (_finite(diagnostic_summary.get("best_hard_risk_violation_count")) or 0.0) > 0.0
        or (_finite(diagnostic_summary.get("best_path_cost_total_m_mean")) or 0.0) <= 0.0
    ):
        return ROUTE_CONTINUE
    return ROUTE_PREFLIGHT


def _observed_best_coverage(sweeps: list[dict[str, Any]]) -> dict[str, float | None]:
    means = [_finite(sweep.get("observed_final_coverage_rate_mean")) for sweep in sweeps if sweep["input_valid"]]
    maxes = [_finite(sweep.get("observed_final_coverage_rate_max")) for sweep in sweeps if sweep["input_valid"]]
    return {"mean": _max(means), "max": _max(maxes)}


def _hard_risk_violation_count(
    sweeps: list[dict[str, Any]],
    matrix_rows: list[dict[str, Any]],
    diagnostic_summary: dict[str, Any],
) -> float:
    return (
        sum(float(sweep.get("hard_risk_violation_count", 0.0) or 0.0) for sweep in sweeps if sweep["input_valid"])
        + sum(float(row.get("hard_risk_rejected_candidate_count", 0.0) or 0.0) for row in matrix_rows)
        + sum(float(row.get("hard_risk_violation_count", 0.0) or 0.0) for row in diagnostic_summary["rows"] if row["complete"])
    )


def _candidate_reward_metrics(candidate: Mapping[str, Any], *, denominator: float) -> dict[str, Any]:
    coverage = _finite(candidate.get("expected_new_coverage_cell_count")) or 0.0
    return {
        "coverage_gain_rate": coverage / denominator,
        "roi_coverage": _finite(candidate.get("roi_weighted_coverage_delta")) or coverage,
        "information_gain": 0.0,
        "path_cost_m": _finite(candidate.get("path_cost")) or 0.0,
        "soft_risk_exposure": _soft_risk_exposure(candidate),
        "hard_risk_violation": _candidate_hard_risk(candidate),
        "fallback_used": False,
        "failure": False,
    }


def _candidate_hard_risk(candidate: Mapping[str, Any]) -> bool:
    return candidate.get("path_allowed_by_risk") is False or bool(candidate.get("hard_risk_flags") or [])


def _soft_risk_exposure(candidate: Mapping[str, Any]) -> float:
    for field in ("soft_risk_exposure", "path_risk_exposure", "risk_cost_weighted"):
        value = _finite(candidate.get(field))
        if value is not None:
            return value
    return 0.0


def _candidate_rows_by_key(rows: Iterable[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_decision_key(row)].append(row)
    return grouped


def _decision_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("scenario_id"),
        row.get("executing_policy") or row.get("policy"),
        row.get("step_index"),
        row.get("candidate_set_hash"),
        row.get("covered_cells_hash"),
    )


def _candidate_by_index(rows: list[dict[str, Any]], index: Any) -> dict[str, Any] | None:
    parsed = _finite(index)
    if parsed is None:
        return None
    for row in rows:
        row_index = _finite(row.get("candidate_index"))
        if row_index is not None and int(row_index) == int(parsed):
            return row
    return None


def _metric_from_candidate(candidate: Mapping[str, Any] | None, field: str) -> float | None:
    return _finite(candidate.get(field)) if candidate else None


def _validate_profile_variants(
    observed_profile: CanonicalRewardGuardProfile,
    variants: list[CanonicalRewardGuardProfile],
) -> None:
    base = observed_profile.to_content_dict()
    base_components = dict(base["soft_reward_components"])
    for profile in variants:
        if profile.profile_version != "v3":
            raise ConfigError("all Stage 18.11 profile variants must use profile_version v3")
        payload = profile.to_content_dict()
        components = dict(payload["soft_reward_components"])
        for key, value in base_components.items():
            if key == "path_cost_weight":
                continue
            if components.get(key) != value:
                raise ConfigError(f"profile variant {profile.profile_id} changes unsupported component {key}")
        for section in ("normalizers", "risk_policy", "trajectory_guards"):
            if payload.get(section) != base.get(section):
                raise ConfigError(f"profile variant {profile.profile_id} changes unsupported section {section}")


def _validate_profile_lineage(payload: Mapping[str, Any], expected: dict[str, str], blocking: list[str]) -> None:
    mismatched = False
    for field, expected_value in expected.items():
        value = payload.get(field) or payload.get(f"canonical_guard_{field}")
        if value != expected_value:
            mismatched = True
            blocking.append(f"{field}_mismatch")
    if mismatched:
        blocking.append("profile_lineage_mismatch")


def _validate_candidate_metric_lineage(rows: list[dict[str, Any]], expected: dict[str, str], blocking: list[str]) -> None:
    if not rows:
        return
    for row in rows:
        for field, expected_value in expected.items():
            if row.get(field) != expected_value:
                blocking.append(f"candidate_metric_{field}_mismatch")
                blocking.append("profile_lineage_mismatch")
                return


def _validate_paired_candidate_key_coverage(
    paired_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    blocking: list[str],
) -> None:
    if not paired_rows or not candidate_rows:
        return
    paired_keys = {_decision_key(row) for row in paired_rows}
    candidate_keys = {_decision_key(row) for row in candidate_rows}
    missing = paired_keys - candidate_keys
    if missing:
        blocking.append("candidate_metric_audit_missing_paired_keys")


def _profile_identity(profile: CanonicalRewardGuardProfile) -> dict[str, str]:
    return {"profile_id": profile.profile_id, "profile_version": profile.profile_version, "profile_hash": profile.profile_hash}


def _profile_variant_row(profile: CanonicalRewardGuardProfile, path: str) -> dict[str, Any]:
    return {
        "profile_path": str(path),
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
        "path_cost_weight": profile.soft_reward_components["path_cost_weight"],
    }


def _ranked_replay_combos(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row.get("replay_step_count", 0) and row.get("hard_risk_rejected_candidate_count", 0) == 0]
    if not eligible:
        eligible = [row for row in rows if row.get("replay_step_count", 0)]
    return sorted(
        eligible,
        key=lambda row: (
            -float(row.get("reward_rerank_coverage_proxy_sum", 0.0) or 0.0),
            float(row.get("reward_rerank_path_cost_proxy_sum", 0.0) or 0.0),
            float(row.get("path_cost_weight", 0.0) or 0.0),
        ),
    )


def _best_replay_combo(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    ranked = _ranked_replay_combos(rows)
    return ranked[0] if ranked else None


def _best_diagnostic_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if _finite(row.get("final_coverage_rate_mean")) is not None]
    if not valid:
        return None
    return sorted(
        valid,
        key=lambda row: (
            -float(row.get("final_coverage_rate_mean") or 0.0),
            float(row.get("path_cost_total_m_mean") or 0.0),
            float(row.get("path_cost_weight") or 0.0),
        ),
    )[0]


def _observed_coverage_stats(episodes: list[dict[str, Any]]) -> dict[str, float | None]:
    xunce = [row for row in episodes if (row.get("policy") or row.get("policy_name")) == "xunce"]
    values = [_finite(row.get("final_coverage_rate")) for row in xunce]
    return {"mean": _mean(values), "max": _max(values)}


def _coverage_denominator(summary: dict[str, Any], manifest: dict[str, Any]) -> float:
    for source in (
        summary,
        manifest.get("normalized_config") if isinstance(manifest.get("normalized_config"), dict) else {},
    ):
        value = _finite(source.get("coverage_denominator_cells")) if isinstance(source, dict) else None
        if value is not None and value > 0:
            return value
    return 1000.0


def _read_json(path: Path, reasons: list[str], missing_code: str) -> dict[str, Any]:
    if not path.is_file():
        reasons.append(missing_code)
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        reasons.append("invalid_json")
        return {}
    if not isinstance(value, dict):
        reasons.append("invalid_json")
        return {}
    return value


def _read_jsonl(path: Path, reasons: list[str], missing_code: str) -> list[dict[str, Any]]:
    if not path.is_file():
        reasons.append(missing_code)
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    except json.JSONDecodeError:
        reasons.append("invalid_jsonl")
        return []
    return rows


def _boundary_violations(*payloads: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for field in BOUNDARY_FIELDS:
            if payload.get(field) is True:
                violations.append(field)
        if float(payload.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
            violations.append("canary_traffic_fraction")
    return unique_sorted(violations)


def _sum_field(rows: list[Mapping[str, Any]], field: str) -> float:
    return sum(_finite(row.get(field)) or 0.0 for row in rows)


def _sum_numeric(rows: Iterable[Mapping[str, Any]], field: str) -> float:
    return sum(_finite(row.get(field)) or 0.0 for row in rows)


def _mean(values: Iterable[float | None]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    return sum(nums) / len(nums) if nums else None


def _max(values: Iterable[float | None]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    return max(nums) if nums else None


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a positive integer") from exc
    if parsed <= 0:
        raise ConfigError(f"{label} must be a positive integer")
    return parsed


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label} must be a non-empty string")
    return value


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed not in {float("inf"), float("-inf")} else None


def _weight_label(weight: float) -> str:
    return f"w{int(round(weight * 100)):03d}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Stage 18.11 Path Cost Weight Calibration",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- target_final_coverage_rate: `{summary['target_final_coverage_rate']}`",
            f"- observed_best_final_coverage_rate_mean: `{summary['observed_best_final_coverage_rate_mean']}`",
            f"- observed_best_final_coverage_rate_max: `{summary['observed_best_final_coverage_rate_max']}`",
            f"- best_replay_candidate_count: `{summary['best_replay_candidate_count']}`",
            f"- best_replay_path_cost_weight: `{summary['best_replay_path_cost_weight']}`",
            f"- stage19_authorized: `{summary['stage19_authorized']}`",
            "",
            "Stage 18.11 performs one-step observed candidate-set replay and optional reward-rerank oracle diagnostics. It does not claim a changed Xunce policy, does not claim trajectory success from one-step replay, and does not authorize PPO, checkpoint publishing, default policy replacement, executor connection, or canary traffic.",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
