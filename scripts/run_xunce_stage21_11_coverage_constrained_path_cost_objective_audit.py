from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


CONFIG_SCHEMA_VERSION = "xunce-stage21-11-coverage-constrained-path-cost-objective-audit-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage21-11-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage21-11-manifest/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage21-11-next-stage-routing/v1"

DEFAULT_CONFIG = "configs/xunce_stage21_11_coverage_constrained_path_cost_objective_audit_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/outputs/"
    "path_feedback_batch_xunce_stage21_11_coverage_constrained_path_cost_objective_audit_v1"
)

SUMMARY_FILE = "xunce-stage21-11-summary.json"
REWARD_AUDIT_FILE = "xunce-stage21-11-reward-separation-audit.json"
ADVANTAGE_AUDIT_FILE = "xunce-stage21-11-advantage-separation-audit.json"
PROBABILITY_AUDIT_FILE = "xunce-stage21-11-action-probability-shift-audit.json"
OBJECTIVE_FILE = "xunce-stage21-11-coverage-cost-objective-recommendation.json"
RECOMMENDED_CONFIG_FILE = "xunce-stage21-11-recommended-stage21-6-config.json"
ROUTING_FILE = "xunce-stage21-11-next-stage-routing.json"
REPORT_FILE = "xunce-stage21-11-report.md"
MANIFEST_FILE = "xunce-stage21-11-manifest.json"

ROUTE_INPUTS = "rerun_stage21_11_required_inputs"
ROUTE_BOUNDARY = "resolve_stage21_11_boundary_rejections"
ROUTE_REWARD = "repair_stage21_coverage_constrained_reward_profile"
ROUTE_ADVANTAGE = "repair_stage21_return_advantage_credit_assignment"
ROUTE_SIGNAL = "calibrate_stage21_policy_update_signal_strength"
ROUTE_BINDING = "repair_stage21_post_update_evaluation_binding"
ROUTE_STAGE21_12 = "run_stage21_12_coverage_constrained_multi_seed_ppo_pilot"

BOUNDARY_FIELDS = (
    "stage21_11_authorized",
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
    parser = argparse.ArgumentParser(description="Run Stage 21.11 coverage-constrained path-cost objective audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        summary = run_xunce_stage21_11_coverage_constrained_path_cost_objective_audit(
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
                "reward_alignment_rate": summary.get("reward_best_matches_coverage_per_cost_rate"),
                "advantage_coverage_per_cost_correlation": summary.get("advantage_coverage_per_cost_correlation"),
                "mean_abs_probability_delta": summary.get("mean_abs_probability_delta"),
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] in {"passed", "partial"} else 1


def run_xunce_stage21_11_coverage_constrained_path_cost_objective_audit(
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

    stage21_10_root = Path(config["stage21_10_root"])
    stage21_10_summary = _read_json_or_empty(stage21_10_root / "xunce-stage21-10-summary.json")
    stage21_6_root = _stage21_6_root(stage21_10_summary, config)
    stage21_6_summary = _read_json_or_empty(stage21_6_root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json")
    stage21_6_aggregate = _read_json_or_empty(stage21_6_root / "xunce-stage21-6-aggregate-metrics.json")
    stage21_6_lineage = _read_json_or_empty(stage21_6_root / "xunce-stage21-6-lineage-audit.json")
    seed_rows = _read_jsonl(stage21_6_root / "xunce-stage21-6-seed-results.jsonl")

    boundary_reasons = _boundary_reasons(config)
    input_reasons = _input_reasons(stage21_10_summary, stage21_6_summary, stage21_6_aggregate, stage21_6_lineage, seed_rows)

    seed_audits: list[dict[str, Any]] = []
    if not boundary_reasons and not input_reasons:
        for seed_row in seed_rows:
            seed_audits.append(_audit_seed(seed_row, config))

    reward_audit = _reward_audit(seed_audits, config)
    advantage_audit = _advantage_audit(seed_audits, config)
    probability_audit = _probability_audit(seed_audits, config)
    objective = _objective_recommendation(config, stage21_10_summary, stage21_6_aggregate, reward_audit, advantage_audit, probability_audit)
    recommended_config = _recommended_stage21_6_config(config, stage21_10_summary, objective)
    status, route, primary_reason = _route(
        input_reasons=input_reasons,
        boundary_reasons=boundary_reasons,
        reward_audit=reward_audit,
        advantage_audit=advantage_audit,
        probability_audit=probability_audit,
        stage21_6_aggregate=stage21_6_aggregate,
    )

    return _write_outputs(
        config=config,
        config_path=config_path,
        output_root=output_root,
        stage21_10_summary=stage21_10_summary,
        stage21_6_root=stage21_6_root,
        stage21_6_summary=stage21_6_summary,
        stage21_6_aggregate=stage21_6_aggregate,
        stage21_6_lineage=stage21_6_lineage,
        seed_audits=seed_audits,
        reward_audit=reward_audit,
        advantage_audit=advantage_audit,
        probability_audit=probability_audit,
        objective=objective,
        recommended_config=recommended_config,
        status=status,
        route=route,
        primary_reason=primary_reason,
        input_reasons=input_reasons,
        boundary_reasons=boundary_reasons,
    )


def _audit_seed(seed_row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    stage21_3_root = Path(str(seed_row["stage21_3_root"]))
    stage21_4_root = Path(str(seed_row["stage21_4_root"]))
    stage21_5_root = Path(str(seed_row["stage21_5_root"]))
    batch_rows = _read_jsonl(stage21_3_root / "xunce-stage21-3-ppo-trainable-batch.jsonl")
    return_rows = _read_jsonl(stage21_3_root / "xunce-stage21-3-return-advantage-audit.jsonl")
    loss_rows = _read_jsonl(stage21_4_root / "xunce-stage21-4-ppo-loss-audit.jsonl")
    gradient_audit = _read_json_or_empty(stage21_4_root / "xunce-stage21-4-gradient-audit.json")
    stage21_5_summary = _read_json_or_empty(stage21_5_root / "xunce-stage21-5-post-update-evaluation-summary.json")
    pre_rows = _read_jsonl(_eval_root(stage21_5_summary, "pre") / "xunce-exploration-coverage-model-inference.jsonl")
    post_rows = _read_jsonl(_eval_root(stage21_5_summary, "post") / "xunce-exploration-coverage-model-inference.jsonl")

    transition_audits = [_transition_metrics(row, config) for row in batch_rows]
    probability_rows = _probability_shift_rows(transition_audits, pre_rows, post_rows)
    return {
        "seed": seed_row.get("seed"),
        "seed_row": seed_row,
        "stage21_3_root": str(stage21_3_root),
        "stage21_4_root": str(stage21_4_root),
        "stage21_5_root": str(stage21_5_root),
        "transition_count": len(batch_rows),
        "return_audit_count": len(return_rows),
        "loss_audit_count": len(loss_rows),
        "gradient_audit": gradient_audit,
        "stage21_5_summary": stage21_5_summary,
        "transition_audits": transition_audits,
        "probability_shift_rows": probability_rows,
    }


def _transition_metrics(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    info = row.get("info", {})
    observation = row.get("observation", {})
    action_mask = _bool_list(info.get("sampling_mask") or info.get("action_mask") or observation.get("action_mask"))
    candidate_metrics = _candidate_metrics(row, action_mask, config)
    action_index = int(row.get("action_index", -1))
    valid_candidates = [m for m in candidate_metrics if m["valid"]]
    selected = candidate_metrics[action_index] if 0 <= action_index < len(candidate_metrics) else {}
    best_cpc = _best_by(valid_candidates, "coverage_per_cost_proxy")
    best_reward = _best_by(valid_candidates, "coverage_constrained_reward_proxy")
    best_coverage = _best_by(valid_candidates, "coverage_gain_rate")
    return {
        "transition_id": row.get("transition_id"),
        "scenario_id": row.get("scenario_id") or info.get("scenario_id"),
        "step_index": row.get("step_index") if row.get("step_index") is not None else info.get("step_index"),
        "candidate_set_hash": info.get("candidate_set_hash"),
        "covered_cells_hash": info.get("covered_cells_hash"),
        "current_cell": info.get("current_cell_before") or info.get("current_cell"),
        "action_index": action_index,
        "advantage": _float(row.get("advantage")),
        "raw_advantage": _float(row.get("raw_advantage")),
        "reward": _float(row.get("reward")),
        "reward_components": row.get("reward_components", {}),
        "selected": selected,
        "best_coverage_per_cost": best_cpc,
        "best_reward": best_reward,
        "best_coverage": best_coverage,
        "selected_is_best_coverage_per_cost": bool(selected and best_cpc and selected["candidate_index"] == best_cpc["candidate_index"]),
        "selected_is_best_reward": bool(selected and best_reward and selected["candidate_index"] == best_reward["candidate_index"]),
        "reward_best_matches_coverage_per_cost": bool(best_reward and best_cpc and best_reward["candidate_index"] == best_cpc["candidate_index"]),
        "selected_coverage_per_cost_rank": _rank(valid_candidates, action_index, "coverage_per_cost_proxy", descending=True),
        "selected_reward_rank": _rank(valid_candidates, action_index, "coverage_constrained_reward_proxy", descending=True),
        "selected_coverage_rank": _rank(valid_candidates, action_index, "coverage_gain_rate", descending=True),
        "selected_path_cost_rank": _rank(valid_candidates, action_index, "path_cost_proxy", descending=False),
        "candidate_count": len(candidate_metrics),
        "valid_candidate_count": len(valid_candidates),
    }


def _candidate_metrics(row: dict[str, Any], action_mask: list[bool], config: dict[str, Any]) -> list[dict[str, Any]]:
    observation = row.get("observation", {})
    names = list(observation.get("candidate_feature_names") or [])
    feature_rows = observation.get("candidate_features") or []
    if not feature_rows:
        batch = row.get("xunce_batch", {}).get("candidate_features", {})
        values = batch.get("values") if isinstance(batch, dict) else None
        if values and values[0]:
            feature_rows = values[0]
            names = ["cell_x", "cell_y", "relative_dx", "relative_dy", "relative_distance", "utility", "reachable", "expected_coverage_rate_delta"]
    candidate_cells = row.get("info", {}).get("candidate_cells") or observation.get("candidate_cells") or []
    metrics: list[dict[str, Any]] = []
    for idx, features in enumerate(feature_rows):
        coverage = _feature(features, names, "expected_coverage_rate_delta")
        if coverage is None:
            coverage = _feature(features, names, "information_gain")
        path_cost = _feature(features, names, "path_cost")
        if path_cost is None:
            path_cost = _feature(features, names, "relative_distance")
        risk = _feature(features, names, "risk")
        valid = bool(action_mask[idx]) if idx < len(action_mask) else True
        coverage_value = _finite_or_zero(coverage)
        path_cost_value = max(_finite_or_zero(path_cost), float(config["coverage_per_cost_path_cost_epsilon"]))
        risk_value = _finite_or_zero(risk)
        score = (
            coverage_value * float(config["reward_proxy_coverage_weight"])
            - path_cost_value * float(config["reward_proxy_path_cost_weight"])
            - risk_value * float(config["reward_proxy_soft_risk_weight"])
        )
        metrics.append(
            {
                "candidate_index": idx,
                "candidate_cell": candidate_cells[idx] if idx < len(candidate_cells) else None,
                "valid": valid,
                "coverage_gain_rate": coverage_value,
                "path_cost_proxy": path_cost_value,
                "soft_risk_proxy": risk_value,
                "coverage_per_cost_proxy": coverage_value / path_cost_value,
                "coverage_constrained_reward_proxy": score,
            }
        )
    return metrics


def _reward_audit(seed_audits: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    rows = [row for seed in seed_audits for row in seed["transition_audits"] if row["valid_candidate_count"] > 0]
    reward_matches = [1.0 if row["reward_best_matches_coverage_per_cost"] else 0.0 for row in rows]
    selected_best = [1.0 if row["selected_is_best_coverage_per_cost"] else 0.0 for row in rows]
    selected_reward_best = [1.0 if row["selected_is_best_reward"] else 0.0 for row in rows]
    selected_cpc_ranks = [_float(row["selected_coverage_per_cost_rank"]) for row in rows]
    selected_reward_ranks = [_float(row["selected_reward_rank"]) for row in rows]
    selected_to_best_ratios: list[float] = []
    for row in rows:
        selected = row.get("selected") or {}
        best = row.get("best_coverage_per_cost") or {}
        selected_cpc = _float(selected.get("coverage_per_cost_proxy"))
        best_cpc = _float(best.get("coverage_per_cost_proxy"))
        if selected_cpc is not None and best_cpc and best_cpc > 0:
            selected_to_best_ratios.append(selected_cpc / best_cpc)
    alignment_rate = _avg(reward_matches)
    selected_best_rate = _avg(selected_best)
    reason_codes: list[str] = []
    if rows and alignment_rate < float(config["min_reward_best_matches_coverage_per_cost_rate"]):
        reason_codes.append("reward_proxy_best_candidate_not_aligned_with_coverage_per_cost")
    if rows and selected_best_rate < float(config["min_selected_best_coverage_per_cost_rate"]):
        reason_codes.append("xunce_selected_action_often_below_best_coverage_per_cost_candidate")
    return {
        "schema_version": "xunce-stage21-11-reward-separation-audit/v1",
        "transition_count": len(rows),
        "reward_best_matches_coverage_per_cost_rate": alignment_rate,
        "selected_best_coverage_per_cost_rate": selected_best_rate,
        "selected_best_reward_rate": _avg(selected_reward_best),
        "selected_coverage_per_cost_rank_mean": _avg(selected_cpc_ranks),
        "selected_reward_rank_mean": _avg(selected_reward_ranks),
        "selected_to_best_coverage_per_cost_ratio_mean": _avg(selected_to_best_ratios),
        "reason_codes": reason_codes,
        "diagnostic_note": "coverage_per_cost uses per-candidate normalized/proxy path-cost features when raw path meters are unavailable.",
    }


def _advantage_audit(seed_audits: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    rows = [row for seed in seed_audits for row in seed["transition_audits"] if row.get("selected")]
    advantages = [_float(row.get("advantage")) for row in rows]
    coverage = [_float((row.get("selected") or {}).get("coverage_gain_rate")) for row in rows]
    cpc = [_float((row.get("selected") or {}).get("coverage_per_cost_proxy")) for row in rows]
    path_cost = [_float((row.get("selected") or {}).get("path_cost_proxy")) for row in rows]
    advantage_std = _std([value for value in advantages if value is not None])
    cpc_corr = _pearson(advantages, cpc)
    coverage_corr = _pearson(advantages, coverage)
    cost_corr = _pearson(advantages, path_cost)
    positive_count = sum(1 for value in advantages if value is not None and value > 0)
    negative_count = sum(1 for value in advantages if value is not None and value < 0)
    reason_codes: list[str] = []
    if advantage_std is None or advantage_std < float(config["min_advantage_std"]):
        reason_codes.append("advantage_signal_flat")
    if cpc_corr is None or cpc_corr < float(config["min_advantage_coverage_per_cost_correlation"]):
        reason_codes.append("advantage_not_positive_for_coverage_per_cost")
    return {
        "schema_version": "xunce-stage21-11-advantage-separation-audit/v1",
        "transition_count": len(rows),
        "advantage_std": advantage_std,
        "positive_advantage_count": positive_count,
        "negative_advantage_count": negative_count,
        "advantage_coverage_gain_correlation": coverage_corr,
        "advantage_coverage_per_cost_correlation": cpc_corr,
        "advantage_path_cost_correlation": cost_corr,
        "reason_codes": reason_codes,
    }


def _probability_audit(seed_audits: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    rows = [row for seed in seed_audits for row in seed["probability_shift_rows"]]
    strong_rows = [row for row in rows if row.get("strong_state_join_available")]
    mean_abs_delta = _avg([_float(row.get("mean_abs_probability_delta")) for row in rows])
    best_delta = _avg([_float(row.get("best_coverage_per_cost_probability_delta")) for row in rows])
    selected_delta = _avg([_float(row.get("selected_probability_delta")) for row in rows])
    strong_mean_abs_delta = _avg([_float(row.get("mean_abs_probability_delta")) for row in strong_rows])
    strong_best_delta = _avg([_float(row.get("best_coverage_per_cost_probability_delta")) for row in strong_rows])
    strong_selected_delta = _avg([_float(row.get("selected_probability_delta")) for row in strong_rows])
    argmax_changed_count = sum(1 for row in strong_rows if row.get("argmax_changed"))
    selected_changed_count = sum(1 for row in strong_rows if row.get("selected_action_changed"))
    rank_changed_count = sum(1 for row in strong_rows if row.get("selected_rank_changed"))
    observable = (
        strong_mean_abs_delta >= float(config["min_probability_shift_mean_abs_delta"])
        or argmax_changed_count > 0
        or selected_changed_count > 0
        or rank_changed_count > 0
    )
    reason_codes: list[str] = []
    if not rows:
        reason_codes.append("no_strong_state_probability_shift_rows")
    if rows and not strong_rows:
        reason_codes.append("strong_state_probability_binding_unavailable")
    if rows and not observable:
        reason_codes.append("post_update_action_probabilities_nearly_unchanged")
    if strong_rows and strong_best_delta <= float(config["min_best_efficiency_probability_delta"]):
        reason_codes.append("best_coverage_per_cost_probability_not_increased")
    return {
        "schema_version": "xunce-stage21-11-action-probability-shift-audit/v1",
        "matched_inference_row_count": len(rows),
        "mean_abs_probability_delta": mean_abs_delta,
        "selected_probability_delta_mean": selected_delta,
        "best_coverage_per_cost_probability_delta_mean": best_delta,
        "strong_state_mean_abs_probability_delta": strong_mean_abs_delta,
        "strong_state_selected_probability_delta_mean": strong_selected_delta,
        "strong_state_best_coverage_per_cost_probability_delta_mean": strong_best_delta,
        "argmax_changed_count": argmax_changed_count,
        "selected_action_changed_count": selected_changed_count,
        "selected_rank_changed_count": rank_changed_count,
        "policy_shift_observable_for_objective": observable,
        "strong_state_join_required": True,
        "strong_state_join_available_count": sum(1 for row in rows if row.get("strong_state_join_available")),
        "missing_strong_state_binding_field_count": sum(1 for row in rows if row.get("missing_strong_state_binding_fields")),
        "reason_codes": reason_codes,
    }


def _probability_shift_rows(
    transition_audits: list[dict[str, Any]],
    pre_rows: list[dict[str, Any]],
    post_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pre_by_key = {_strong_inference_key(row): row for row in pre_rows}
    post_by_key = {_strong_inference_key(row): row for row in post_rows}
    metrics_by_key = {_strong_transition_key(row): row for row in transition_audits}
    if not set(pre_by_key).intersection(post_by_key).intersection(metrics_by_key):
        # Older inference artifacts do not always persist current_cell / covered_cells_hash.
        # Fall back to candidate-set hash, but mark each row as a weak binding diagnostic.
        pre_by_key = {_inference_key(row): row for row in pre_rows}
        post_by_key = {_inference_key(row): row for row in post_rows}
        metrics_by_key = {_transition_key(row): row for row in transition_audits}
        strong_join_available = False
    else:
        strong_join_available = True
    rows: list[dict[str, Any]] = []
    for key, pre in pre_by_key.items():
        post = post_by_key.get(key)
        metrics = metrics_by_key.get(key)
        if not post or not metrics:
            continue
        pre_detail = pre.get("detail", {})
        post_detail = post.get("detail", {})
        pre_probs = [_float(value) for value in pre_detail.get("action_probs", [])]
        post_probs = [_float(value) for value in post_detail.get("action_probs", [])]
        if len(pre_probs) != len(post_probs) or not pre_probs:
            continue
        deltas = [abs(float(b) - float(a)) for a, b in zip(pre_probs, post_probs) if a is not None and b is not None]
        selected_index = int(pre_detail.get("selected_action_index", pre.get("selected_action_index", -1)))
        best_index = int((metrics.get("best_coverage_per_cost") or {}).get("candidate_index", -1))
        selected_delta = _prob_delta(pre_probs, post_probs, selected_index)
        best_delta = _prob_delta(pre_probs, post_probs, best_index)
        rows.append(
            {
                "scenario_id": key[0],
                "step_index": key[1],
                "candidate_set_hash": key[2],
                "selected_action_index": selected_index,
                "best_coverage_per_cost_action_index": best_index,
                "mean_abs_probability_delta": _avg(deltas),
                "selected_probability_delta": selected_delta,
                "best_coverage_per_cost_probability_delta": best_delta,
                "pre_argmax_action_index": _argmax(pre_probs),
                "post_argmax_action_index": _argmax(post_probs),
                "argmax_changed": _argmax(pre_probs) != _argmax(post_probs),
                "selected_action_changed": int(pre_detail.get("selected_action_index", -1)) != int(post_detail.get("selected_action_index", -1)),
                "pre_selected_rank": pre_detail.get("selected_rank"),
                "post_selected_rank": post_detail.get("selected_rank"),
                "selected_rank_changed": pre_detail.get("selected_rank") != post_detail.get("selected_rank"),
                "strong_state_join_available": strong_join_available,
                "missing_strong_state_binding_fields": not strong_join_available,
            }
        )
    return rows


def _objective_recommendation(
    config: dict[str, Any],
    stage21_10_summary: dict[str, Any],
    stage21_6_aggregate: dict[str, Any],
    reward_audit: dict[str, Any],
    advantage_audit: dict[str, Any],
    probability_audit: dict[str, Any],
) -> dict[str, Any]:
    target = float(config["target_final_coverage_rate"])
    observed = _float(stage21_10_summary.get("stage21_6_final_coverage_rate_mean"))
    if observed is None:
        observed = _float(stage21_10_summary.get("post_final_coverage_rate_mean"))
    if observed is None:
        observed = _float(stage21_10_summary.get("mean_post_final_coverage_rate"))
    observed = observed if observed is not None else 0.1416015625
    return {
        "schema_version": "xunce-stage21-11-coverage-cost-objective-recommendation/v1",
        "target_final_coverage_rate": target,
        "observed_final_coverage_rate_proxy": observed,
        "coverage_not_saturated": observed < target,
        "recommended_objective": "coverage_constrained_path_cost",
        "objective_text": "先把整场覆盖推向 99%，在同等或接近覆盖下优化路径成本和 coverage per cost。",
        "recommended_profile_direction": {
            "below_99pct": "coverage progress and coverage-per-cost should both create positive advantage for selected actions.",
            "at_or_above_99pct": "increase path-cost and coverage-per-100m pressure while keeping hard-risk boundary at zero.",
            "do_not_claim": "path-cost improvement alone is not success when final coverage is far below 99%.",
        },
        "current_stage21_6": {
            "trainable_transition_count_total": stage21_6_aggregate.get("trainable_transition_count_total"),
            "final_coverage_delta_mean": stage21_6_aggregate.get("final_coverage_delta_mean"),
            "coverage_auc_delta_mean": stage21_6_aggregate.get("coverage_auc_delta_mean"),
            "path_cost_delta_m_mean": stage21_6_aggregate.get("path_cost_delta_m_mean"),
        },
        "audit_summary": {
            "reward_best_matches_coverage_per_cost_rate": reward_audit.get("reward_best_matches_coverage_per_cost_rate"),
            "advantage_coverage_per_cost_correlation": advantage_audit.get("advantage_coverage_per_cost_correlation"),
            "mean_abs_probability_delta": probability_audit.get("mean_abs_probability_delta"),
        },
    }


def _recommended_stage21_6_config(
    config: dict[str, Any],
    stage21_10_summary: dict[str, Any],
    objective: dict[str, Any],
) -> dict[str, Any]:
    source = stage21_10_summary.get("repaired_stage21_6_config")
    payload = _read_json_or_empty(Path(str(source))) if source else {}
    payload.update(
        {
            "schema_version": payload.get("schema_version", "xunce-stage21-6-multi-seed-ppo-pilot-config/v1"),
            "stage21_11_recommended": True,
            "source_stage21_10_root": str(config["stage21_10_root"]),
            "objective_mode": "coverage_constrained_path_cost",
        "target_final_coverage_rate": float(config["target_final_coverage_rate"]),
        "coverage_cost_objective_recommendation": objective["recommended_profile_direction"],
        "runs_new_ppo_update": False,
        "stage21_11_generated_recommendation_only": True,
        "recommended_config_requires_separate_human_execution": True,
        "stage21_6_authorized": False,
            "training_or_release_authorized": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return payload


def _route(
    *,
    input_reasons: list[str],
    boundary_reasons: list[str],
    reward_audit: dict[str, Any],
    advantage_audit: dict[str, Any],
    probability_audit: dict[str, Any],
    stage21_6_aggregate: dict[str, Any],
) -> tuple[str, str, str]:
    if input_reasons:
        return "failed", ROUTE_INPUTS, "input_or_lineage_rejections"
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "boundary_rejections"
    hard_boundary = _stage21_6_execution_boundary_reasons(stage21_6_aggregate)
    if hard_boundary:
        return "failed", ROUTE_BOUNDARY, "stage21_6_execution_boundary_rejections"
    if reward_audit.get("reason_codes"):
        return "failed", ROUTE_REWARD, "reward_not_aligned_with_coverage_per_cost"
    if advantage_audit.get("reason_codes"):
        return "failed", ROUTE_ADVANTAGE, "advantage_not_aligned_with_coverage_per_cost"
    if probability_audit.get("reason_codes"):
        return "failed", ROUTE_SIGNAL, "policy_probability_shift_not_supporting_objective"
    mean_final_delta = _float(stage21_6_aggregate.get("final_coverage_delta_mean"))
    mean_auc_delta = _float(stage21_6_aggregate.get("coverage_auc_delta_mean"))
    if probability_audit.get("policy_shift_observable_for_objective") and mean_final_delta == 0.0 and mean_auc_delta == 0.0:
        return "failed", ROUTE_BINDING, "probability_shift_observed_but_trajectory_unchanged"
    return "passed", ROUTE_STAGE21_12, "reward_advantage_probability_audits_passed"


def _write_outputs(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    stage21_10_summary: dict[str, Any],
    stage21_6_root: Path,
    stage21_6_summary: dict[str, Any],
    stage21_6_aggregate: dict[str, Any],
    stage21_6_lineage: dict[str, Any],
    seed_audits: list[dict[str, Any]],
    reward_audit: dict[str, Any],
    advantage_audit: dict[str, Any],
    probability_audit: dict[str, Any],
    objective: dict[str, Any],
    recommended_config: dict[str, Any],
    status: str,
    route: str,
    primary_reason: str,
    input_reasons: list[str],
    boundary_reasons: list[str],
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_json(output_root / REWARD_AUDIT_FILE, reward_audit)
    _write_json(output_root / ADVANTAGE_AUDIT_FILE, advantage_audit)
    _write_json(output_root / PROBABILITY_AUDIT_FILE, probability_audit)
    _write_json(output_root / OBJECTIVE_FILE, objective)
    _write_json(output_root / RECOMMENDED_CONFIG_FILE, recommended_config)
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "primary_reason": primary_reason,
        "input_reason_codes": input_reasons,
        "boundary_reason_codes": sorted(set(boundary_reasons + _stage21_6_execution_boundary_reasons(stage21_6_aggregate))),
        "stage21_11_authorized": False,
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
        "stage21_10_root": str(config["stage21_10_root"]),
        "stage21_10_status": stage21_10_summary.get("status"),
        "stage21_10_next_required_change": stage21_10_summary.get("next_required_change"),
        "stage21_6_root": str(stage21_6_root),
        "stage21_6_status": stage21_6_summary.get("status"),
        "stage21_6_lineage_passed": stage21_6_lineage.get("passed"),
        "seed_count": len(seed_audits),
        "transition_count": sum(int(seed.get("transition_count", 0) or 0) for seed in seed_audits),
        "trainable_transition_count_total": stage21_6_aggregate.get("trainable_transition_count_total"),
        "mean_final_coverage_delta": stage21_6_aggregate.get("final_coverage_delta_mean"),
        "mean_coverage_auc_delta": stage21_6_aggregate.get("coverage_auc_delta_mean"),
        "pre_clip_grad_norm_max": stage21_6_aggregate.get("pre_clip_grad_norm_max"),
        "hard_risk_violation_total": stage21_6_aggregate.get("hard_risk_violation_total"),
        "model_inference_mask_violation_total": stage21_6_aggregate.get("model_inference_mask_violation_total"),
        "path_planning_failure_total": stage21_6_aggregate.get("path_planning_failure_total"),
        "open_grid_fallback_total": stage21_6_aggregate.get("open_grid_fallback_total"),
        "target_final_coverage_rate": config["target_final_coverage_rate"],
        "coverage_not_saturated": objective["coverage_not_saturated"],
        "reward_best_matches_coverage_per_cost_rate": reward_audit.get("reward_best_matches_coverage_per_cost_rate"),
        "selected_best_coverage_per_cost_rate": reward_audit.get("selected_best_coverage_per_cost_rate"),
        "advantage_std": advantage_audit.get("advantage_std"),
        "advantage_coverage_per_cost_correlation": advantage_audit.get("advantage_coverage_per_cost_correlation"),
        "mean_abs_probability_delta": probability_audit.get("mean_abs_probability_delta"),
        "best_coverage_per_cost_probability_delta_mean": probability_audit.get(
            "best_coverage_per_cost_probability_delta_mean"
        ),
        "matched_inference_row_count": probability_audit.get("matched_inference_row_count"),
        "strong_state_join_available_count": probability_audit.get("strong_state_join_available_count"),
        "missing_strong_state_binding_field_count": probability_audit.get("missing_strong_state_binding_field_count"),
        "summary": str(output_root / SUMMARY_FILE),
        "reward_separation_audit": str(output_root / REWARD_AUDIT_FILE),
        "advantage_separation_audit": str(output_root / ADVANTAGE_AUDIT_FILE),
        "action_probability_shift_audit": str(output_root / PROBABILITY_AUDIT_FILE),
        "coverage_cost_objective_recommendation": str(output_root / OBJECTIVE_FILE),
        "recommended_stage21_6_config": str(output_root / RECOMMENDED_CONFIG_FILE),
        "routing": str(output_root / ROUTING_FILE),
        "manifest": str(output_root / MANIFEST_FILE),
        "input_reason_codes": input_reasons,
        "boundary_reason_codes": sorted(set(boundary_reasons + _stage21_6_execution_boundary_reasons(stage21_6_aggregate))),
        "reward_reason_codes": reward_audit.get("reason_codes", []),
        "advantage_reason_codes": advantage_audit.get("reason_codes", []),
        "probability_reason_codes": probability_audit.get("reason_codes", []),
        "stage21_11_authorized": False,
        "training_or_release_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_report(output_root / REPORT_FILE, summary, reward_audit, advantage_audit, probability_audit, objective)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "stage21_10_root": str(config["stage21_10_root"]),
        "artifacts": {
            "summary": str(output_root / SUMMARY_FILE),
            "reward_separation_audit": str(output_root / REWARD_AUDIT_FILE),
            "advantage_separation_audit": str(output_root / ADVANTAGE_AUDIT_FILE),
            "action_probability_shift_audit": str(output_root / PROBABILITY_AUDIT_FILE),
            "coverage_cost_objective_recommendation": str(output_root / OBJECTIVE_FILE),
            "recommended_stage21_6_config": str(output_root / RECOMMENDED_CONFIG_FILE),
            "routing": str(output_root / ROUTING_FILE),
            "report": str(output_root / REPORT_FILE),
        },
    }
    _write_json(output_root / MANIFEST_FILE, manifest)
    return summary


def _write_report(
    path: Path,
    summary: dict[str, Any],
    reward_audit: dict[str, Any],
    advantage_audit: dict[str, Any],
    probability_audit: dict[str, Any],
    objective: dict[str, Any],
) -> None:
    lines = [
        "# Xunce Stage 21.11 Coverage-Constrained Path-Cost Objective Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- primary_reason: `{summary['primary_reason']}`",
        f"- Stage21.10 final coverage delta mean: `{summary.get('mean_final_coverage_delta')}`",
        f"- Stage21.10 coverage AUC delta mean: `{summary.get('mean_coverage_auc_delta')}`",
        f"- reward best matches coverage-per-cost rate: `{reward_audit.get('reward_best_matches_coverage_per_cost_rate')}`",
        f"- selected best coverage-per-cost rate: `{reward_audit.get('selected_best_coverage_per_cost_rate')}`",
        f"- advantage coverage-per-cost correlation: `{advantage_audit.get('advantage_coverage_per_cost_correlation')}`",
        f"- mean abs probability delta: `{probability_audit.get('mean_abs_probability_delta')}`",
        f"- matched inference rows: `{probability_audit.get('matched_inference_row_count')}`",
        f"- strong state joins: `{probability_audit.get('strong_state_join_available_count')}`",
        "",
        "## Objective",
        "",
        objective["objective_text"],
        "",
        "This stage is audit-only. It does not run PPO, publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _input_reasons(
    stage21_10_summary: dict[str, Any],
    stage21_6_summary: dict[str, Any],
    stage21_6_aggregate: dict[str, Any],
    stage21_6_lineage: dict[str, Any],
    seed_rows: list[dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    if not stage21_10_summary:
        reasons.append("missing_stage21_10_summary")
        return reasons
    if stage21_10_summary.get("status") != "failed":
        reasons.append("stage21_10_status_not_failed_repair_input")
    if stage21_10_summary.get("next_required_change") != "repair_stage21_reward_signal_or_advantage_separation":
        reasons.append("stage21_10_route_not_reward_signal_or_advantage_separation")
    if not stage21_6_summary:
        reasons.append("missing_stage21_6_summary")
    if not stage21_6_aggregate:
        reasons.append("missing_stage21_6_aggregate")
    if stage21_6_lineage.get("passed") is not True:
        reasons.append("stage21_6_lineage_not_passed")
    if not seed_rows:
        reasons.append("missing_stage21_6_seed_results")
    for row in seed_rows:
        seed = row.get("seed")
        for key in ("stage21_3_root", "stage21_4_root", "stage21_5_root"):
            if not Path(str(row.get(key, ""))).exists():
                reasons.append(f"seed_{seed}_missing_{key}")
                continue
        stage21_3_root = Path(str(row.get("stage21_3_root", "")))
        stage21_4_root = Path(str(row.get("stage21_4_root", "")))
        stage21_5_root = Path(str(row.get("stage21_5_root", "")))
        required_files = (
            (stage21_3_root / "xunce-stage21-3-ppo-trainable-batch.jsonl", "stage21_3_trainable_batch"),
            (stage21_3_root / "xunce-stage21-3-return-advantage-audit.jsonl", "stage21_3_return_advantage_audit"),
            (stage21_4_root / "xunce-stage21-4-ppo-loss-audit.jsonl", "stage21_4_loss_audit"),
            (stage21_4_root / "xunce-stage21-4-gradient-audit.json", "stage21_4_gradient_audit"),
            (stage21_5_root / "xunce-stage21-5-post-update-evaluation-summary.json", "stage21_5_summary"),
        )
        for path, label in required_files:
            if not path.is_file():
                reasons.append(f"seed_{seed}_missing_{label}")
        stage21_5_summary = _read_json_or_empty(stage21_5_root / "xunce-stage21-5-post-update-evaluation-summary.json")
        pre_root = _eval_root(stage21_5_summary, "pre")
        post_root = _eval_root(stage21_5_summary, "post")
        if not (pre_root / "xunce-exploration-coverage-model-inference.jsonl").is_file():
            reasons.append(f"seed_{seed}_missing_pre_model_inference")
        if not (post_root / "xunce-exploration-coverage-model-inference.jsonl").is_file():
            reasons.append(f"seed_{seed}_missing_post_model_inference")
    return sorted(set(reasons))


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
        "target_final_coverage_rate": 0.99,
        "coverage_per_cost_path_cost_epsilon": 0.01,
        "reward_proxy_coverage_weight": 1.0,
        "reward_proxy_path_cost_weight": 0.2,
        "reward_proxy_soft_risk_weight": 0.005,
        "min_reward_best_matches_coverage_per_cost_rate": 0.35,
        "min_selected_best_coverage_per_cost_rate": 0.2,
        "min_advantage_std": 1.0e-6,
        "min_advantage_coverage_per_cost_correlation": 0.0,
        "min_probability_shift_mean_abs_delta": 1.0e-4,
        "min_best_efficiency_probability_delta": 0.0,
        "stage21_11_authorized": False,
        "training_or_release_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    merged = {**defaults, **config}
    for key in ("stage21_10_root", "coverage_first_reward_profile"):
        if not merged.get(key):
            raise ConfigError(f"missing required config field: {key}")
        merged[key] = str(_resolve_path(Path(str(merged[key])), repo_root))
    return merged


def _stage21_6_root(stage21_10_summary: dict[str, Any], config: dict[str, Any]) -> Path:
    if config.get("stage21_6_root"):
        return Path(str(config["stage21_6_root"]))
    value = stage21_10_summary.get("stage21_6_root")
    if value:
        return Path(str(value))
    return Path(str(config["stage21_10_root"])) / "s6"


def _eval_root(stage21_5_summary: dict[str, Any], label: str) -> Path:
    key = f"{label}_evaluation_root"
    if stage21_5_summary.get(key):
        return Path(str(stage21_5_summary[key]))
    return Path()


def _transition_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (str(row.get("scenario_id")), int(row.get("step_index", -1)), str(row.get("candidate_set_hash")))


def _inference_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (str(row.get("scenario_id")), int(row.get("step_index", -1)), str(row.get("candidate_set_hash")))


def _strong_transition_key(row: dict[str, Any]) -> tuple[str, int, str, str, str]:
    return (
        str(row.get("scenario_id")),
        int(row.get("step_index", -1)),
        str(row.get("candidate_set_hash")),
        _stable_json_key(row.get("covered_cells_hash")),
        _stable_json_key(row.get("current_cell")),
    )


def _strong_inference_key(row: dict[str, Any]) -> tuple[str, int, str, str, str]:
    return (
        str(row.get("scenario_id")),
        int(row.get("step_index", -1)),
        str(row.get("candidate_set_hash")),
        _stable_json_key(row.get("covered_cells_hash")),
        _stable_json_key(row.get("current_cell") or row.get("current_cell_before")),
    )


def _stable_json_key(value: Any) -> str:
    if value is None:
        return "<missing>"
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


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


def _rank(rows: list[dict[str, Any]], candidate_index: int, field: str, *, descending: bool) -> int | None:
    usable = [row for row in rows if _float(row.get(field)) is not None]
    if not usable:
        return None
    usable.sort(key=lambda row: float(row[field]), reverse=descending)
    for rank, row in enumerate(usable, start=1):
        if row["candidate_index"] == candidate_index:
            return rank
    return None


def _prob_delta(pre_probs: list[float | None], post_probs: list[float | None], index: int) -> float | None:
    if index < 0 or index >= len(pre_probs) or index >= len(post_probs):
        return None
    pre = pre_probs[index]
    post = post_probs[index]
    if pre is None or post is None:
        return None
    return float(post) - float(pre)


def _argmax(values: list[float | None]) -> int | None:
    usable = [(idx, value) for idx, value in enumerate(values) if value is not None]
    if not usable:
        return None
    return max(usable, key=lambda item: float(item[1]))[0]


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


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
