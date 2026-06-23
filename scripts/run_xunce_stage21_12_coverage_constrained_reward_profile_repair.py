from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_EXPLORER_SRC = str(REPO_ROOT / "model-explorer" / "src")
if MODEL_EXPLORER_SRC not in sys.path:
    sys.path.insert(0, MODEL_EXPLORER_SRC)

from model_explorer.policy.coverage_first_reward import (  # noqa: E402
    SCHEMA_VERSION,
    SCHEMA_VERSION_V2,
    compute_coverage_first_reward_components,
    load_coverage_first_reward_profile,
)


CONFIG_SCHEMA_VERSION = "xunce-stage21-12-coverage-constrained-reward-profile-repair-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage21-12-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage21-12-manifest/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage21-12-next-stage-routing/v1"

DEFAULT_CONFIG = "configs/xunce_stage21_12_coverage_constrained_reward_profile_repair_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/outputs/"
    "path_feedback_batch_xunce_stage21_12_coverage_constrained_reward_profile_repair_v1"
)

SUMMARY_FILE = "xunce-stage21-12-summary.json"
COMPARISON_FILE = "xunce-stage21-12-reward-profile-comparison.json"
REPLAY_FILE = "xunce-stage21-12-reward-replay.jsonl"
ADVANTAGE_FILE = "xunce-stage21-12-advantage-replay-audit.json"
RECOMMENDED_STAGE21_2_CONFIG_FILE = "xunce-stage21-12-recommended-stage21-2-config.json"
RECOMMENDED_STAGE21_6_CONFIG_FILE = "xunce-stage21-12-recommended-stage21-6-config.json"
ROUTING_FILE = "xunce-stage21-12-next-stage-routing.json"
REPORT_FILE = "xunce-stage21-12-report.md"
MANIFEST_FILE = "xunce-stage21-12-manifest.json"

ROUTE_INPUTS = "rerun_stage21_12_required_inputs"
ROUTE_BOUNDARY = "resolve_stage21_12_boundary_rejections"
ROUTE_CONTINUE = "continue_stage21_12_reward_profile_repair"
ROUTE_ADVANTAGE = "repair_stage21_return_advantage_credit_assignment"
ROUTE_STAGE21_13 = "run_stage21_13_coverage_constrained_multi_seed_ppo_smoke"

BOUNDARY_FIELDS = (
    "stage21_12_authorized",
    "training_or_release_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 21.12 coverage-constrained reward profile repair.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        summary = run_xunce_stage21_12_coverage_constrained_reward_profile_repair(
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
                "primary_reason": summary["primary_reason"],
                "v1_reward_best_matches_coverage_per_cost_rate": summary.get(
                    "v1_reward_best_matches_coverage_per_cost_rate"
                ),
                "v2_reward_best_matches_coverage_per_cost_rate": summary.get(
                    "v2_reward_best_matches_coverage_per_cost_rate"
                ),
                "v2_advantage_coverage_per_cost_correlation": summary.get(
                    "v2_advantage_coverage_per_cost_correlation"
                ),
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] in {"passed", "partial"} else 1


def run_xunce_stage21_12_coverage_constrained_reward_profile_repair(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    config_path = _resolve_path(config_path, repo_root)
    config = _load_config(config_path, repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    profile_v1 = load_coverage_first_reward_profile(config["reward_profile_v1"])
    profile_v2 = load_coverage_first_reward_profile(config["reward_profile_v2"])
    stage21_11_root = Path(config["stage21_11_root"])
    stage21_11_summary = _read_json_or_empty(stage21_11_root / "xunce-stage21-11-summary.json")
    stage21_6_root = _stage21_6_root(config, stage21_11_summary)
    stage21_6_summary = _read_json_or_empty(stage21_6_root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json")
    stage21_6_aggregate = _read_json_or_empty(stage21_6_root / "xunce-stage21-6-aggregate-metrics.json")
    stage21_6_lineage = _read_json_or_empty(stage21_6_root / "xunce-stage21-6-lineage-audit.json")
    seed_rows = _read_jsonl(stage21_6_root / "xunce-stage21-6-seed-results.jsonl")

    input_reasons = _input_reasons(stage21_11_summary, stage21_6_summary, stage21_6_aggregate, stage21_6_lineage, seed_rows, profile_v1, profile_v2)
    boundary_reasons = _boundary_reasons(config)
    replay_rows: list[dict[str, Any]] = []
    if not input_reasons and not boundary_reasons:
        for seed_row in seed_rows:
            replay_rows.extend(_reward_replay_seed(seed_row, config, profile_v1, profile_v2))

    comparison = _reward_profile_comparison(replay_rows, config, stage21_11_summary, profile_v1, profile_v2)
    advantage = _advantage_replay_audit(replay_rows, config)
    recommended_stage21_2_config = _recommended_stage21_2_config(config, profile_v2)
    recommended_stage21_6_config = _recommended_stage21_6_config(
        config=config,
        stage21_11_summary=stage21_11_summary,
        output_root=output_root,
        stage21_2_config_path=output_root / RECOMMENDED_STAGE21_2_CONFIG_FILE,
    )
    status, route, primary_reason = _route(
        input_reasons=input_reasons,
        boundary_reasons=boundary_reasons,
        comparison=comparison,
        advantage=advantage,
        stage21_6_aggregate=stage21_6_aggregate,
    )
    return _write_outputs(
        config=config,
        config_path=config_path,
        output_root=output_root,
        profile_v1=profile_v1,
        profile_v2=profile_v2,
        stage21_11_summary=stage21_11_summary,
        stage21_6_root=stage21_6_root,
        stage21_6_summary=stage21_6_summary,
        stage21_6_aggregate=stage21_6_aggregate,
        replay_rows=replay_rows,
        comparison=comparison,
        advantage=advantage,
        recommended_stage21_2_config=recommended_stage21_2_config,
        recommended_stage21_6_config=recommended_stage21_6_config,
        status=status,
        route=route,
        primary_reason=primary_reason,
        input_reasons=input_reasons,
        boundary_reasons=boundary_reasons,
    )


def _reward_replay_seed(
    seed_row: dict[str, Any],
    config: dict[str, Any],
    profile_v1: Any,
    profile_v2: Any,
) -> list[dict[str, Any]]:
    stage21_3_root = Path(str(seed_row["stage21_3_root"]))
    batch_rows = _read_jsonl(stage21_3_root / "xunce-stage21-3-ppo-trainable-batch.jsonl")
    stage21_2_root = _stage21_2_root(seed_row)
    reward_rows = _read_jsonl(stage21_2_root / "xunce-stage21-2-reward-contract-evaluation.jsonl")
    reward_by_transition = _index_by_transition_id(reward_rows)
    return [
        _reward_replay_transition(row, seed_row, config, profile_v1, profile_v2, reward_by_transition)
        for row in batch_rows
    ]


def _reward_replay_transition(
    row: dict[str, Any],
    seed_row: dict[str, Any],
    config: dict[str, Any],
    profile_v1: Any,
    profile_v2: Any,
    reward_by_transition: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    observation = row.get("observation") if isinstance(row.get("observation"), dict) else {}
    action_mask = _bool_list(info.get("action_mask") or observation.get("action_mask"))
    sampling_mask = _bool_list(info.get("sampling_mask") or observation.get("sampling_mask"))
    hard_risk_clean_mask = _bool_list(info.get("hard_risk_clean_mask") or observation.get("hard_risk_clean_mask"))
    candidate_rows = _candidate_rows(row, action_mask, sampling_mask, hard_risk_clean_mask, config, profile_v1, profile_v2)
    valid_rows = [candidate for candidate in candidate_rows if candidate["valid"]]
    action_index = int(row.get("action_index", -1))
    selected = candidate_rows[action_index] if 0 <= action_index < len(candidate_rows) else {}
    best_cpc = _best_by(valid_rows, "coverage_per_cost")
    best_coverage = _best_by(valid_rows, "coverage_gain_rate")
    best_v1 = _best_by(valid_rows, "v1_reward")
    best_v2 = _best_by(valid_rows, "v2_reward")
    selected_metrics = _selected_transition_metrics(row, selected)
    selected_v1 = compute_coverage_first_reward_components(selected_metrics, profile_v1)
    selected_v2 = compute_coverage_first_reward_components(selected_metrics, profile_v2)
    stage21_2_reward_row = reward_by_transition.get(str(row.get("transition_id"))) or {}
    coverage_priority_violation = _coverage_priority_violation(best_v2, best_coverage, config)
    return {
        "schema_version": "xunce-stage21-12-reward-replay-row/v1",
        "seed": seed_row.get("seed"),
        "transition_id": row.get("transition_id"),
        "scenario_id": row.get("scenario_id") or (row.get("info") or {}).get("scenario_id"),
        "step_index": row.get("step_index") if row.get("step_index") is not None else (row.get("info") or {}).get("step_index"),
        "candidate_set_hash": (row.get("info") or {}).get("candidate_set_hash"),
        "action_index": action_index,
        "selected": selected,
        "best_coverage_per_cost": best_cpc,
        "best_coverage": best_coverage,
        "best_v1_reward": best_v1,
        "best_v2_reward": best_v2,
        "v1_reward_best_matches_coverage_per_cost": bool(best_v1 and best_cpc and best_v1["candidate_index"] == best_cpc["candidate_index"]),
        "v2_reward_best_matches_coverage_per_cost": bool(best_v2 and best_cpc and best_v2["candidate_index"] == best_cpc["candidate_index"]),
        "v2_best_preserves_coverage_priority": not coverage_priority_violation,
        "coverage_priority_violation": coverage_priority_violation,
        "selected_is_best_coverage_per_cost": bool(selected and best_cpc and selected["candidate_index"] == best_cpc["candidate_index"]),
        "selected_v1_reward": selected_v1.reward,
        "selected_v2_reward": selected_v2.reward,
        "stage21_2_v1_reward": stage21_2_reward_row.get("reward"),
        "stage21_2_v1_profile_hash": stage21_2_reward_row.get("profile_hash"),
        "v1_selected_reward_matches_stage21_2": _numbers_close(selected_v1.reward, stage21_2_reward_row.get("reward")),
        "v1_profile_hash_matches_stage21_2": stage21_2_reward_row.get("profile_hash") == profile_v1.profile_hash,
        "selected_v1_components": selected_v1.components,
        "selected_v2_components": selected_v2.components,
        "selected_v2_trainable": selected_v2.trainable and bool(row.get("trainable", True)),
        "selected_v2_reason_codes": selected_v2.reason_codes,
        "selected_coverage_per_cost": selected.get("coverage_per_cost"),
        "selected_coverage_gain_rate": selected.get("coverage_gain_rate"),
        "selected_path_cost_proxy": selected.get("path_cost_proxy"),
        "old_value": _float(row.get("value")) or _float(row.get("old_value")) or 0.0,
        "stage21_3_split": row.get("stage21_3_split") or row.get("split"),
        "hard_risk_rejected": "hard_risk_rejected" in selected_v2.reason_codes,
    }


def _candidate_rows(
    row: dict[str, Any],
    action_mask: list[bool],
    sampling_mask: list[bool],
    hard_risk_clean_mask: list[bool],
    config: dict[str, Any],
    profile_v1: Any,
    profile_v2: Any,
) -> list[dict[str, Any]]:
    observation = row.get("observation") if isinstance(row.get("observation"), dict) else {}
    names = list(observation.get("candidate_feature_names") or [])
    feature_rows = observation.get("candidate_features") or []
    if not feature_rows:
        batch = row.get("xunce_batch", {}).get("candidate_features", {})
        values = batch.get("values") if isinstance(batch, dict) else None
        if values and values[0]:
            feature_rows = values[0]
            names = [
                "cell_x",
                "cell_y",
                "relative_dx",
                "relative_dy",
                "relative_distance",
                "utility",
                "reachable",
                "expected_coverage_rate_delta",
            ]
    candidate_cells = (row.get("info") or {}).get("candidate_cells") or observation.get("candidate_cells") or []
    candidates: list[dict[str, Any]] = []
    for idx, features in enumerate(feature_rows):
        action_valid = bool(action_mask[idx]) if idx < len(action_mask) else True
        sampling_valid = bool(sampling_mask[idx]) if idx < len(sampling_mask) else action_valid
        hard_risk_clean = bool(hard_risk_clean_mask[idx]) if idx < len(hard_risk_clean_mask) else sampling_valid
        valid = action_valid and sampling_valid and hard_risk_clean
        coverage = _feature(features, names, "expected_coverage_rate_delta")
        if coverage is None:
            coverage = _feature(features, names, "information_gain")
        path_cost = _feature(features, names, "path_cost")
        if path_cost is None:
            path_cost = _feature(features, names, "relative_distance")
        soft_risk = _feature(features, names, "risk")
        coverage_value = max(0.0, _finite_or_zero(coverage))
        path_cost_value = max(_finite_or_zero(path_cost), float(config["candidate_path_cost_floor"]))
        soft_risk_value = max(0.0, _finite_or_zero(soft_risk))
        coverage_per_cost = coverage_value / path_cost_value
        metrics = {
            "coverage_rate_delta": coverage_value,
            "coverage_progress_rate": _finite_or_zero((row.get("info") or {}).get("final_coverage_rate_after_step")),
            "final_coverage_rate": _finite_or_zero((row.get("info") or {}).get("final_coverage_rate_after_step")),
            "coverage_per_cost": coverage_per_cost,
            "path_cost_m": path_cost_value,
            "soft_risk_exposure": soft_risk_value,
            "done": bool(row.get("done", False)),
            "path_allowed_by_risk": hard_risk_clean,
            "hard_risk_flags": [] if hard_risk_clean else ["hard_risk_clean_mask_false"],
        }
        v1 = compute_coverage_first_reward_components(metrics, profile_v1)
        v2 = compute_coverage_first_reward_components(metrics, profile_v2)
        candidates.append(
            {
                "candidate_index": idx,
                "candidate_cell": candidate_cells[idx] if idx < len(candidate_cells) else None,
                "valid": valid,
                "action_mask_valid": action_valid,
                "sampling_mask_valid": sampling_valid,
                "hard_risk_clean": hard_risk_clean,
                "coverage_gain_rate": coverage_value,
                "path_cost_proxy": path_cost_value,
                "soft_risk_proxy": soft_risk_value,
                "coverage_per_cost": coverage_per_cost,
                "v1_reward": v1.reward,
                "v2_reward": v2.reward,
                "v2_components": v2.components,
            }
        )
    return candidates


def _selected_transition_metrics(row: dict[str, Any], selected: Mapping[str, Any]) -> dict[str, Any]:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    coverage_delta = _float(info.get("coverage_rate_delta"))
    if coverage_delta is None:
        coverage_delta = _float(selected.get("coverage_gain_rate")) or 0.0
    path_cost = _float(info.get("path_cost_m"))
    if path_cost is None:
        path_cost = _float(info.get("path_cost"))
    if path_cost is None:
        path_cost = _float(selected.get("path_cost_proxy")) or 0.0
    soft_risk = _float(info.get("soft_risk_exposure"))
    if soft_risk is None:
        soft_risk = _float(info.get("risk_cost_weighted"))
    if soft_risk is None:
        soft_risk = _float(selected.get("soft_risk_proxy")) or 0.0
    final_coverage = _float(info.get("final_coverage_rate_after_step"))
    if final_coverage is None:
        final_coverage = _float(info.get("final_coverage_rate")) or 0.0
    selected_hard_risk_clean = selected.get("hard_risk_clean")
    hard_risk_violation = bool(info.get("hard_risk_violation", False))
    hard_risk_flags = list(info.get("hard_risk_flags") or [])
    if selected_hard_risk_clean is False and "hard_risk_clean_mask_false" not in hard_risk_flags:
        hard_risk_flags.append("hard_risk_clean_mask_false")
        hard_risk_violation = True
    if hard_risk_violation and not hard_risk_flags:
        hard_risk_flags.append("hard_risk_violation")
    return {
        "coverage_rate_delta": max(0.0, coverage_delta),
        "coverage_progress_rate": max(0.0, final_coverage),
        "final_coverage_rate": max(0.0, final_coverage),
        "coverage_per_cost": _float(selected.get("coverage_per_cost")) or 0.0,
        "path_cost_m": max(0.0, path_cost),
        "soft_risk_exposure": max(0.0, soft_risk),
        "done": bool(row.get("done", False) or info.get("done", False)),
        "failure": row.get("failure", False) or info.get("failure", False),
        "failure_reason": info.get("failure_reason") or row.get("failure_reason"),
        "path_allowed_by_risk": bool(info.get("path_allowed_by_risk", not hard_risk_violation)) and selected_hard_risk_clean is not False,
        "open_grid_fallback_used": info.get("open_grid_fallback_used", False),
        "hard_risk_flags": hard_risk_flags,
        "hard_risk_violation_count": max(int(info.get("hard_risk_violation_count", 0) or 0), 1 if hard_risk_violation else 0),
    }


def _reward_profile_comparison(
    replay_rows: list[dict[str, Any]],
    config: dict[str, Any],
    stage21_11_summary: dict[str, Any],
    profile_v1: Any,
    profile_v2: Any,
) -> dict[str, Any]:
    rows = [row for row in replay_rows if row.get("best_coverage_per_cost")]
    v1_matches = [1.0 if row["v1_reward_best_matches_coverage_per_cost"] else 0.0 for row in rows]
    v2_matches = [1.0 if row["v2_reward_best_matches_coverage_per_cost"] else 0.0 for row in rows]
    selected_matches = [1.0 if row["selected_is_best_coverage_per_cost"] else 0.0 for row in rows]
    coverage_priority_violations = [1.0 if row["coverage_priority_violation"] else 0.0 for row in rows]
    v1_rate = _avg(v1_matches)
    v2_rate = _avg(v2_matches)
    improvement = v2_rate - v1_rate
    violation_rate = _avg(coverage_priority_violations)
    reason_codes: list[str] = []
    if rows and v2_rate < float(config["min_v2_reward_best_matches_coverage_per_cost_rate"]):
        reason_codes.append("v2_reward_best_still_not_aligned_with_coverage_per_cost")
    if rows and improvement < float(config["min_v2_alignment_improvement"]):
        reason_codes.append("v2_reward_alignment_improvement_too_small")
    if rows and violation_rate > float(config["max_coverage_priority_violation_rate"]):
        reason_codes.append("v2_path_cost_pressure_sacrifices_coverage_gain")
    hard_risk_trainable_count = sum(1 for row in rows if row["hard_risk_rejected"] and row["selected_v2_trainable"])
    v1_reward_mismatch_count = sum(1 for row in rows if not row.get("v1_selected_reward_matches_stage21_2"))
    v1_profile_mismatch_count = sum(1 for row in rows if not row.get("v1_profile_hash_matches_stage21_2"))
    if hard_risk_trainable_count:
        reason_codes.append("v2_hard_risk_transition_marked_trainable")
    if v1_reward_mismatch_count:
        reason_codes.append("v1_replay_reward_mismatch_with_stage21_2")
    if v1_profile_mismatch_count:
        reason_codes.append("v1_replay_profile_hash_mismatch_with_stage21_2")
    return {
        "schema_version": "xunce-stage21-12-reward-profile-comparison/v1",
        "transition_count": len(rows),
        "profile_v1_id": profile_v1.profile_id,
        "profile_v1_version": profile_v1.profile_version,
        "profile_v1_hash": profile_v1.profile_hash,
        "profile_v2_id": profile_v2.profile_id,
        "profile_v2_version": profile_v2.profile_version,
        "profile_v2_hash": profile_v2.profile_hash,
        "stage21_11_reward_best_matches_coverage_per_cost_rate": stage21_11_summary.get("reward_best_matches_coverage_per_cost_rate"),
        "v1_reward_best_matches_coverage_per_cost_rate": v1_rate,
        "v2_reward_best_matches_coverage_per_cost_rate": v2_rate,
        "v2_alignment_improvement": improvement,
        "selected_best_coverage_per_cost_rate": _avg(selected_matches),
        "v2_coverage_priority_violation_rate": violation_rate,
        "v2_hard_risk_trainable_count": hard_risk_trainable_count,
        "v1_reward_mismatch_count": v1_reward_mismatch_count,
        "v1_profile_mismatch_count": v1_profile_mismatch_count,
        "reason_codes": reason_codes,
        "diagnostic_note": "Candidate replay uses observed Stage21 candidate features. Path cost may be normalized/proxy when raw meters are unavailable.",
    }


def _advantage_replay_audit(replay_rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    train_rows = [
        row
        for row in replay_rows
        if row.get("selected_v2_trainable") and str(row.get("stage21_3_split")) == "train"
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in train_rows:
        grouped.setdefault((str(row.get("seed")), str(row.get("scenario_id"))), []).append(row)
    gamma = float(config["discount_factor"])
    raw_advantages: list[float] = []
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item.get("step_index", 0) or 0), reverse=True)
        running_return = 0.0
        for row in rows:
            running_return = float(row["selected_v2_reward"]) + gamma * running_return
            raw_advantage = running_return - float(row.get("old_value", 0.0) or 0.0)
            row["v2_return"] = running_return
            row["v2_raw_advantage"] = raw_advantage
            raw_advantages.append(raw_advantage)
    mean_adv = mean(raw_advantages) if raw_advantages else 0.0
    std_adv = _std(raw_advantages) or 0.0
    for row in train_rows:
        raw = float(row.get("v2_raw_advantage", 0.0) or 0.0)
        row["v2_normalized_advantage"] = (raw - mean_adv) / std_adv if std_adv > 0.0 else 0.0
    normalized = [float(row.get("v2_normalized_advantage", 0.0) or 0.0) for row in train_rows]
    cpc_values = [_float((row.get("selected") or {}).get("coverage_per_cost")) or 0.0 for row in train_rows]
    coverage_values = [_float((row.get("selected") or {}).get("coverage_gain_rate")) or 0.0 for row in train_rows]
    path_cost_values = [_float((row.get("selected") or {}).get("path_cost_proxy")) or 0.0 for row in train_rows]
    cpc_corr = _pearson(normalized, cpc_values)
    coverage_corr = _pearson(normalized, coverage_values)
    path_corr = _pearson(normalized, path_cost_values)
    reason_codes: list[str] = []
    if std_adv < float(config["min_v2_advantage_std"]):
        reason_codes.append("v2_advantage_replay_flat")
    if cpc_corr is None or cpc_corr < float(config["min_v2_advantage_coverage_per_cost_correlation"]):
        reason_codes.append("v2_advantage_not_positive_for_coverage_per_cost")
    return {
        "schema_version": "xunce-stage21-12-advantage-replay-audit/v1",
        "transition_count": len(train_rows),
        "v2_advantage_std": std_adv,
        "v2_advantage_coverage_per_cost_correlation": cpc_corr,
        "v2_advantage_coverage_gain_correlation": coverage_corr,
        "v2_advantage_path_cost_correlation": path_corr,
        "positive_v2_advantage_count": sum(1 for value in normalized if value > 0.0),
        "negative_v2_advantage_count": sum(1 for value in normalized if value < 0.0),
        "reason_codes": reason_codes,
    }


def _coverage_priority_violation(best_v2: dict[str, Any] | None, best_coverage: dict[str, Any] | None, config: dict[str, Any]) -> bool:
    if not best_v2 or not best_coverage:
        return False
    best_v2_cov = _float(best_v2.get("coverage_gain_rate")) or 0.0
    best_cov = _float(best_coverage.get("coverage_gain_rate")) or 0.0
    return best_v2_cov + float(config["coverage_close_tolerance_rate"]) < best_cov


def _recommended_stage21_2_config(config: dict[str, Any], profile_v2: Any) -> dict[str, Any]:
    base = _read_json_or_empty(Path(str(config["stage21_2_base_config"])))
    recommended = dict(base)
    recommended["coverage_first_reward_profile"] = str(config["reward_profile_v2"])
    recommended["recommended_by_stage21_12"] = True
    recommended["profile_id"] = profile_v2.profile_id
    recommended["profile_version"] = profile_v2.profile_version
    recommended["profile_hash"] = profile_v2.profile_hash
    recommended["runs_new_ppo_update"] = False
    recommended["publishes_checkpoint"] = False
    recommended["replaces_default_policy"] = False
    recommended["connects_real_executor"] = False
    recommended["starts_online_canary"] = False
    recommended["canary_traffic_fraction"] = 0.0
    return recommended


def _recommended_stage21_6_config(
    *,
    config: dict[str, Any],
    stage21_11_summary: dict[str, Any],
    output_root: Path,
    stage21_2_config_path: Path,
) -> dict[str, Any]:
    base_path = Path(str(stage21_11_summary.get("recommended_stage21_6_config") or ""))
    base = _read_json_or_empty(base_path)
    if not base:
        base = {
            "schema_version": "xunce-stage21-6-multi-seed-ppo-pilot-config/v1",
            "stage_id": "xunce-stage21-6-multi-seed-ppo-pilot",
            "stage_name": "Stage 21.6 Multi-Seed PPO Pilot",
        }
    recommended = dict(base)
    recommended.update(
        {
            "objective_mode": "coverage_constrained_path_cost_v2",
            "stage21_12_recommended": True,
            "stage21_12_generated_recommendation_only": True,
            "recommended_config_requires_separate_human_execution": True,
            "stage21_2_base_config": str(stage21_2_config_path),
            "reward_profile": str(config["reward_profile_v2"]),
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return recommended


def _route(
    *,
    input_reasons: list[str],
    boundary_reasons: list[str],
    comparison: dict[str, Any],
    advantage: dict[str, Any],
    stage21_6_aggregate: dict[str, Any],
) -> tuple[str, str, str]:
    if input_reasons:
        return "failed", ROUTE_INPUTS, "stage21_12_required_inputs_missing_or_untrusted"
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "stage21_12_boundary_rejection"
    execution_reasons = _stage21_6_execution_boundary_reasons(stage21_6_aggregate)
    if execution_reasons:
        return "failed", ROUTE_BOUNDARY, "stage21_6_execution_boundary_rejection"
    if comparison.get("reason_codes"):
        return "failed", ROUTE_CONTINUE, "reward_v2_not_sufficiently_aligned"
    if advantage.get("reason_codes"):
        return "failed", ROUTE_ADVANTAGE, "advantage_replay_not_aligned"
    return "passed", ROUTE_STAGE21_13, "reward_and_advantage_replay_passed"


def _write_outputs(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    profile_v1: Any,
    profile_v2: Any,
    stage21_11_summary: dict[str, Any],
    stage21_6_root: Path,
    stage21_6_summary: dict[str, Any],
    stage21_6_aggregate: dict[str, Any],
    replay_rows: list[dict[str, Any]],
    comparison: dict[str, Any],
    advantage: dict[str, Any],
    recommended_stage21_2_config: dict[str, Any],
    recommended_stage21_6_config: dict[str, Any],
    status: str,
    route: str,
    primary_reason: str,
    input_reasons: list[str],
    boundary_reasons: list[str],
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_json(output_root / COMPARISON_FILE, comparison)
    _write_jsonl(output_root / REPLAY_FILE, replay_rows)
    _write_json(output_root / ADVANTAGE_FILE, advantage)
    _write_json(output_root / RECOMMENDED_STAGE21_2_CONFIG_FILE, recommended_stage21_2_config)
    _write_json(output_root / RECOMMENDED_STAGE21_6_CONFIG_FILE, recommended_stage21_6_config)
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "primary_reason": primary_reason,
        "input_reason_codes": input_reasons,
        "boundary_reason_codes": sorted(set(boundary_reasons + _stage21_6_execution_boundary_reasons(stage21_6_aggregate))),
        "stage21_12_authorized": False,
        "training_or_release_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    _write_json(output_root / ROUTING_FILE, routing)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "next_required_change": route,
        "primary_reason": primary_reason,
        "stage21_11_root": str(config["stage21_11_root"]),
        "stage21_11_status": stage21_11_summary.get("status"),
        "stage21_11_next_required_change": stage21_11_summary.get("next_required_change"),
        "stage21_6_root": str(stage21_6_root),
        "stage21_6_status": stage21_6_summary.get("status"),
        "transition_count": comparison.get("transition_count", 0),
        "trainable_transition_count_total": stage21_6_aggregate.get("trainable_transition_count_total"),
        "mean_final_coverage_delta": stage21_6_aggregate.get("final_coverage_delta_mean"),
        "mean_coverage_auc_delta": stage21_6_aggregate.get("coverage_auc_delta_mean"),
        "profile_v1_id": profile_v1.profile_id,
        "profile_v1_version": profile_v1.profile_version,
        "profile_v1_hash": profile_v1.profile_hash,
        "profile_v2_id": profile_v2.profile_id,
        "profile_v2_version": profile_v2.profile_version,
        "profile_v2_hash": profile_v2.profile_hash,
        "v1_reward_best_matches_coverage_per_cost_rate": comparison.get("v1_reward_best_matches_coverage_per_cost_rate"),
        "v2_reward_best_matches_coverage_per_cost_rate": comparison.get("v2_reward_best_matches_coverage_per_cost_rate"),
        "v2_alignment_improvement": comparison.get("v2_alignment_improvement"),
        "v2_coverage_priority_violation_rate": comparison.get("v2_coverage_priority_violation_rate"),
        "v2_hard_risk_trainable_count": comparison.get("v2_hard_risk_trainable_count"),
        "v2_advantage_std": advantage.get("v2_advantage_std"),
        "v2_advantage_coverage_per_cost_correlation": advantage.get("v2_advantage_coverage_per_cost_correlation"),
        "reward_reason_codes": comparison.get("reason_codes", []),
        "advantage_reason_codes": advantage.get("reason_codes", []),
        "input_reason_codes": input_reasons,
        "boundary_reason_codes": sorted(set(boundary_reasons + _stage21_6_execution_boundary_reasons(stage21_6_aggregate))),
        "summary": str(output_root / SUMMARY_FILE),
        "reward_profile_comparison": str(output_root / COMPARISON_FILE),
        "reward_replay": str(output_root / REPLAY_FILE),
        "advantage_replay_audit": str(output_root / ADVANTAGE_FILE),
        "recommended_stage21_2_config": str(output_root / RECOMMENDED_STAGE21_2_CONFIG_FILE),
        "recommended_stage21_6_config": str(output_root / RECOMMENDED_STAGE21_6_CONFIG_FILE),
        "routing": str(output_root / ROUTING_FILE),
        "manifest": str(output_root / MANIFEST_FILE),
        "stage21_12_authorized": False,
        "training_or_release_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_report(output_root / REPORT_FILE, summary, comparison, advantage)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "stage21_11_root": str(config["stage21_11_root"]),
        "artifacts": {
            "summary": str(output_root / SUMMARY_FILE),
            "reward_profile_comparison": str(output_root / COMPARISON_FILE),
            "reward_replay": str(output_root / REPLAY_FILE),
            "advantage_replay_audit": str(output_root / ADVANTAGE_FILE),
            "recommended_stage21_2_config": str(output_root / RECOMMENDED_STAGE21_2_CONFIG_FILE),
            "recommended_stage21_6_config": str(output_root / RECOMMENDED_STAGE21_6_CONFIG_FILE),
            "routing": str(output_root / ROUTING_FILE),
            "report": str(output_root / REPORT_FILE),
        },
    }
    _write_json(output_root / MANIFEST_FILE, manifest)
    return summary


def _write_report(path: Path, summary: dict[str, Any], comparison: dict[str, Any], advantage: dict[str, Any]) -> None:
    lines = [
        "# Xunce Stage 21.12 Coverage-Constrained Reward Profile Repair",
        "",
        f"- status: `{summary['status']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- primary_reason: `{summary['primary_reason']}`",
        f"- v1 reward best matches coverage-per-cost: `{comparison.get('v1_reward_best_matches_coverage_per_cost_rate')}`",
        f"- v2 reward best matches coverage-per-cost: `{comparison.get('v2_reward_best_matches_coverage_per_cost_rate')}`",
        f"- v2 alignment improvement: `{comparison.get('v2_alignment_improvement')}`",
        f"- v2 coverage-priority violation rate: `{comparison.get('v2_coverage_priority_violation_rate')}`",
        f"- v2 hard-risk trainable count: `{comparison.get('v2_hard_risk_trainable_count')}`",
        f"- v2 advantage coverage-per-cost correlation: `{advantage.get('v2_advantage_coverage_per_cost_correlation')}`",
        f"- runs_new_ppo_update: `{summary.get('runs_new_ppo_update')}`",
        f"- publishes_checkpoint: `{summary.get('publishes_checkpoint')}`",
        f"- replaces_default_policy: `{summary.get('replaces_default_policy')}`",
        f"- connects_real_executor: `{summary.get('connects_real_executor')}`",
        f"- starts_online_canary: `{summary.get('starts_online_canary')}`",
        "",
        "## Interpretation",
        "",
        "Stage21.11 showed the old reward often ranked a different candidate than the best coverage-per-cost candidate. "
        "Stage21.12 adds a coverage-per-cost component while keeping coverage gain first below the 99% target. "
        "The recommended Stage21.6 config is recommendation-only and must be executed separately.",
        "",
        "This stage does not run PPO, publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _input_reasons(
    stage21_11_summary: dict[str, Any],
    stage21_6_summary: dict[str, Any],
    stage21_6_aggregate: dict[str, Any],
    stage21_6_lineage: dict[str, Any],
    seed_rows: list[dict[str, Any]],
    profile_v1: Any,
    profile_v2: Any,
) -> list[str]:
    reasons: list[str] = []
    if not stage21_11_summary:
        reasons.append("missing_stage21_11_summary")
        return reasons
    if stage21_11_summary.get("status") != "failed":
        reasons.append("stage21_11_status_not_failed_repair_input")
    if stage21_11_summary.get("next_required_change") != "repair_stage21_coverage_constrained_reward_profile":
        reasons.append("stage21_11_route_not_reward_profile_repair")
    if not stage21_6_summary:
        reasons.append("missing_stage21_6_summary")
    if not stage21_6_aggregate:
        reasons.append("missing_stage21_6_aggregate")
    if stage21_6_lineage.get("passed") is not True:
        reasons.append("stage21_6_lineage_not_passed")
    if not seed_rows:
        reasons.append("missing_stage21_6_seed_results")
    if profile_v1.schema_version != SCHEMA_VERSION:
        reasons.append("reward_profile_v1_schema_invalid")
    if profile_v2.schema_version != SCHEMA_VERSION_V2:
        reasons.append("reward_profile_v2_schema_invalid")
    for row in seed_rows:
        seed = row.get("seed")
        stage21_3_root = Path(str(row.get("stage21_3_root", "")))
        stage21_2_root = _stage21_2_root(row)
        batch_path = stage21_3_root / "xunce-stage21-3-ppo-trainable-batch.jsonl"
        reward_path = stage21_2_root / "xunce-stage21-2-reward-contract-evaluation.jsonl"
        if not batch_path.is_file():
            reasons.append(f"seed_{seed}_missing_stage21_3_trainable_batch")
        if not reward_path.is_file():
            reasons.append(f"seed_{seed}_missing_stage21_2_reward_contract_evaluation")
        if not (stage21_2_root / "xunce-stage21-2-coverage-first-reward-summary.json").is_file():
            reasons.append(f"seed_{seed}_missing_stage21_2_reward_summary")
        if batch_path.is_file() and reward_path.is_file():
            reasons.extend(_stage21_2_contract_reasons(seed, batch_path, reward_path))
    return sorted(set(reasons))


def _stage21_2_contract_reasons(seed: Any, batch_path: Path, reward_path: Path) -> list[str]:
    reasons: list[str] = []
    batch_rows = _read_jsonl(batch_path)
    reward_rows = _read_jsonl(reward_path)
    batch_ids, batch_duplicates, batch_missing_ids = _transition_id_audit(batch_rows)
    reward_ids, reward_duplicates, reward_missing_ids = _transition_id_audit(reward_rows)
    if batch_missing_ids:
        reasons.append(f"seed_{seed}_stage21_3_missing_transition_id")
    if reward_missing_ids:
        reasons.append(f"seed_{seed}_stage21_2_missing_transition_id")
    if batch_duplicates:
        reasons.append(f"seed_{seed}_stage21_3_duplicate_transition_id")
    if reward_duplicates:
        reasons.append(f"seed_{seed}_stage21_2_duplicate_transition_id")
    missing_reward_ids = sorted(batch_ids - reward_ids)
    extra_reward_ids = sorted(reward_ids - batch_ids)
    if missing_reward_ids:
        reasons.append(f"seed_{seed}_stage21_2_missing_reward_rows_for_batch")
    if extra_reward_ids:
        reasons.append(f"seed_{seed}_stage21_2_extra_reward_rows")
    for row in batch_rows:
        split = row.get("stage21_3_split")
        if split is None:
            reasons.append(f"seed_{seed}_stage21_3_missing_split")
            break
        if str(split) not in {"train", "validation"}:
            reasons.append(f"seed_{seed}_stage21_3_invalid_split")
            break
    return reasons


def _transition_id_audit(rows: list[dict[str, Any]]) -> tuple[set[str], set[str], int]:
    ids: set[str] = set()
    duplicates: set[str] = set()
    missing = 0
    for row in rows:
        transition_id = row.get("transition_id")
        if transition_id is None or str(transition_id) == "":
            missing += 1
            continue
        key = str(transition_id)
        if key in ids:
            duplicates.add(key)
        ids.add(key)
    return ids, duplicates, missing


def _index_by_transition_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        transition_id = row.get("transition_id")
        if transition_id is None or str(transition_id) == "":
            continue
        key = str(transition_id)
        if key not in indexed:
            indexed[key] = row
    return indexed


def _stage21_2_root(seed_row: Mapping[str, Any]) -> Path:
    if seed_row.get("stage21_2_root"):
        return Path(str(seed_row["stage21_2_root"]))
    if seed_row.get("seed_root"):
        return Path(str(seed_row["seed_root"])) / "stage21_2"
    stage21_3_root = Path(str(seed_row.get("stage21_3_root", "")))
    return stage21_3_root.parent / "stage21_2"


def _boundary_reasons(config: dict[str, Any]) -> list[str]:
    reasons = [f"config_{field}_true" for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) > 0.0:
        reasons.append("config_canary_traffic_fraction_nonzero")
    return sorted(set(reasons))


def _stage21_6_execution_boundary_reasons(aggregate: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in (
        "hard_risk_violation_total",
        "model_inference_mask_violation_total",
        "model_inference_failure_total",
        "path_planning_failure_total",
        "open_grid_fallback_total",
        "safety_boundary_violation_total",
        "execution_boundary_violation_total",
    ):
        if int(aggregate.get(field, 0) or 0) > 0:
            reasons.append(f"stage21_6_{field}_nonzero")
    return reasons


def _load_config(config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"expected schema_version {CONFIG_SCHEMA_VERSION}")
    defaults = {
        "discount_factor": 0.99,
        "candidate_path_cost_floor": 0.01,
        "min_v2_reward_best_matches_coverage_per_cost_rate": 0.25,
        "min_v2_alignment_improvement": 0.1,
        "min_v2_advantage_coverage_per_cost_correlation": 0.0,
        "min_v2_advantage_std": 1.0e-6,
        "max_coverage_priority_violation_rate": 0.05,
        "coverage_close_tolerance_rate": 0.005,
        "stage21_12_authorized": False,
        "training_or_release_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    merged = {**defaults, **config}
    for key in ("stage21_11_root", "reward_profile_v1", "reward_profile_v2", "stage21_2_base_config"):
        if not merged.get(key):
            raise ConfigError(f"missing required config field: {key}")
        merged[key] = str(_resolve_path(Path(str(merged[key])), repo_root))
    if merged.get("stage21_10_stage21_6_root"):
        merged["stage21_10_stage21_6_root"] = str(_resolve_path(Path(str(merged["stage21_10_stage21_6_root"])), repo_root))
    return merged


def _stage21_6_root(config: dict[str, Any], stage21_11_summary: dict[str, Any]) -> Path:
    if config.get("stage21_10_stage21_6_root"):
        return Path(str(config["stage21_10_stage21_6_root"]))
    if stage21_11_summary.get("stage21_6_root"):
        return Path(str(stage21_11_summary["stage21_6_root"]))
    return Path(str(config["stage21_11_root"])) / "s6"


def _feature(features: list[Any], names: list[str], name: str) -> float | None:
    try:
        index = names.index(name)
    except ValueError:
        return None
    if index >= len(features):
        return None
    return _float(features[index])


def _best_by(rows: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    usable = [row for row in rows if _float(row.get(field)) is not None]
    if not usable:
        return None
    return max(usable, key=lambda row: float(row[field]))


def _pearson(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x_values = [x for x, _ in pairs]
    y_values = [y for _, y in pairs]
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    x_var = sum((x - x_mean) ** 2 for x in x_values)
    y_var = sum((y - y_mean) ** 2 for y in y_values)
    if x_var <= 0.0 or y_var <= 0.0:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    return cov / math.sqrt(x_var * y_var)


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def _avg(values: list[float | None]) -> float:
    usable = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return mean(usable) if usable else 0.0


def _finite_or_zero(value: Any) -> float:
    finite = _float(value)
    return finite if finite is not None else 0.0


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool_list(value: Any) -> list[bool]:
    if not isinstance(value, list):
        return []
    return [bool(item) for item in value]


def _numbers_close(left: Any, right: Any, *, tolerance: float = 1.0e-9) -> bool:
    left_value = _float(left)
    right_value = _float(right)
    if left_value is None or right_value is None:
        return False
    return abs(left_value - right_value) <= tolerance


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return _read_json(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
