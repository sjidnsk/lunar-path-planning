from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


STAGE_ID = "xunce-stage26-5-synthetic-discrete-margin-crossing-calibration"
CONFIG_SCHEMA_VERSION = "xunce-stage26-5-synthetic-discrete-margin-crossing-calibration-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-5-summary/v1"
MARGIN_SCHEMA_VERSION = "xunce-stage26-5-margin-crossing-audit/v1"
CREDIT_SCHEMA_VERSION = "xunce-stage26-5-ppo-credit-binding-audit/v1"
FEATURE_SCHEMA_VERSION = "xunce-stage26-5-candidate-feature-signal-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-5-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-5-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_5_synthetic_discrete_margin_crossing_calibration_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_5_synthetic_discrete_margin_crossing_calibration_v1"
)
DEFAULT_STAGE26_4_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_4_synthetic_policy_update_signal_strength_repair_v1"
)

SYNTHETIC_HASH = "e628d24c6fb0f961c50294ea0a018a2558a79a0b30ad21dd31b6e34fc0867e85"
SYNTHETIC_SOURCE_KIND = "synthetic_terrain_obstacle_proxy/v1"
COVERAGE_SOURCE = "endpoint_theta_slope_obstacle_los/v1"
PATH_COST_SOURCE = "hybrid_astar_pose_path/v1"

SUMMARY_FILE = "xunce-stage26-5-summary.json"
MARGIN_AUDIT_FILE = "xunce-stage26-5-margin-crossing-audit.json"
TREND_FILE = "xunce-stage26-5-logit-rank-trend.jsonl"
CREDIT_AUDIT_FILE = "xunce-stage26-5-ppo-credit-binding-audit.json"
FEATURE_AUDIT_FILE = "xunce-stage26-5-candidate-feature-signal-audit.json"
RECOMMENDED_CONFIG_FILE = "xunce-stage26-5-recommended-next-config.json"
ROUTING_FILE = "xunce-stage26-5-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-5-report.md"
MANIFEST_FILE = "xunce-stage26-5-manifest.json"

STAGE26_4_SUMMARY_FILE = "xunce-stage26-4-summary.json"
STAGE26_4_SWEEP_FILE = "xunce-stage26-4-sweep-results.jsonl"
STAGE26_4_COLLECTOR_CONFIG_FILE = "xunce-stage26-4-stage26-1-config.json"
MODEL_INFERENCE_FILE = "xunce-exploration-coverage-model-inference.jsonl"

ROUTE_INPUTS = "rerun_stage26_5_required_inputs"
ROUTE_PARALLEL_COLLECTOR = "rerun_stage26_4_with_parallel_collector"
ROUTE_BINDING = "repair_stage26_5_margin_audit_binding"
ROUTE_EXPLORATION_CREDIT = "repair_stage26_synthetic_exploration_credit_assignment"
ROUTE_FEATURE_EXPOSURE = "repair_stage26_synthetic_candidate_feature_exposure"
ROUTE_ITERATIVE = "run_stage26_6_iterative_synthetic_margin_crossing_probe"
ROUTE_GRADIENT_DIRECTION = "repair_stage26_synthetic_policy_gradient_direction"
ROUTE_CREDIT = "repair_stage26_synthetic_credit_assignment"
ROUTE_STAGE26_7 = "run_stage26_7_synthetic_terrain_multi_seed_ppo_pilot"
ROUTE_BOUNDARY = "resolve_stage26_5_boundary_rejections"
ROUTE_FROM_STAGE26_4 = "calibrate_stage26_synthetic_discrete_margin_crossing"

BOUNDARY_FIELDS = (
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.5 synthetic discrete margin crossing calibration.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_5_synthetic_discrete_margin_crossing_calibration(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_5_synthetic_discrete_margin_crossing_calibration(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(config_path, repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage26_4_root = Path(config["stage26_4_root"])
    stage26_4_summary = _read_json_if_exists(stage26_4_root / STAGE26_4_SUMMARY_FILE)
    sweep_rows = _read_jsonl_if_exists(stage26_4_root / STAGE26_4_SWEEP_FILE)
    worker_count = _stage26_4_worker_count(stage26_4_root, repo_root)

    boundary_reasons = _boundary_rejections(config)
    input_reasons = _input_rejections(stage26_4_summary)
    worker_reasons = [] if worker_count == int(config["required_hybrid_astar_candidate_eval_workers"]) else ["stage26_4_worker_count_not_4"]

    trend_rows: list[dict[str, Any]] = []
    if not boundary_reasons and not input_reasons and not worker_reasons:
        trend_rows = _build_trend_rows(sweep_rows)
    margin_audit = _margin_crossing_audit(trend_rows)
    credit_audit = _credit_binding_audit(stage26_4_root, trend_rows)
    feature_audit = _candidate_feature_signal_audit(trend_rows)

    status, route, route_reason = _route(
        boundary_reasons=boundary_reasons,
        input_reasons=input_reasons,
        worker_reasons=worker_reasons,
        margin_audit=margin_audit,
        credit_audit=credit_audit,
        feature_audit=feature_audit,
        stage26_4_summary=stage26_4_summary,
    )
    blocking = _blocking_reasons(route, boundary_reasons, input_reasons, worker_reasons, margin_audit, credit_audit, feature_audit)
    recommended = _recommended_next_config(route, stage26_4_root, margin_audit, credit_audit, feature_audit)

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": blocking,
        "stage26_4_root": str(stage26_4_root),
        "stage26_4_status": stage26_4_summary.get("status"),
        "stage26_4_next_required_change": stage26_4_summary.get("next_required_change"),
        "stage26_4_hybrid_astar_candidate_eval_workers": worker_count,
        "combo_count": margin_audit["combo_count"],
        "strong_state_join_available_count": margin_audit["strong_state_join_available_count"],
        "mean_selected_to_best_probability_margin_pre": margin_audit["mean_selected_to_best_probability_margin_pre"],
        "mean_selected_to_best_probability_margin_post": margin_audit["mean_selected_to_best_probability_margin_post"],
        "mean_probability_gap_closure": margin_audit["mean_probability_gap_closure"],
        "mean_best_rank_gap_closure": margin_audit["mean_best_rank_gap_closure"],
        "mean_estimated_updates_to_cross_margin": margin_audit["mean_estimated_updates_to_cross_margin"],
        "counterfactual_best_not_directly_credited": credit_audit["counterfactual_best_not_directly_credited"],
        "candidate_feature_signal_missing": feature_audit["candidate_feature_signal_missing"],
        "coverage_source": COVERAGE_SOURCE,
        "path_cost_source": PATH_COST_SOURCE,
        "synthetic_terrain_hash": stage26_4_summary.get("synthetic_terrain_hash") or SYNTHETIC_HASH,
        "synthetic_source_kind": SYNTHETIC_SOURCE_KIND,
        "max_traversable_slope_deg": 30.0,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "summary": str(output_root / SUMMARY_FILE),
        "margin_crossing_audit": str(output_root / MARGIN_AUDIT_FILE),
        "logit_rank_trend": str(output_root / TREND_FILE),
        "ppo_credit_binding_audit": str(output_root / CREDIT_AUDIT_FILE),
        "candidate_feature_signal_audit": str(output_root / FEATURE_AUDIT_FILE),
        "recommended_next_config": str(output_root / RECOMMENDED_CONFIG_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }

    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MARGIN_AUDIT_FILE, margin_audit)
    _write_jsonl(output_root / TREND_FILE, trend_rows)
    _write_json(output_root / CREDIT_AUDIT_FILE, credit_audit)
    _write_json(output_root / FEATURE_AUDIT_FILE, feature_audit)
    _write_json(output_root / RECOMMENDED_CONFIG_FILE, recommended)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _build_trend_rows(sweep_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for combo in sweep_rows:
        combo_root_value = combo.get("combo_root")
        if not combo_root_value:
            continue
        combo_root = Path(str(combo_root_value))
        pre_rows = _read_jsonl_if_exists(combo_root / "s3" / "pre" / MODEL_INFERENCE_FILE)
        post_rows = _read_jsonl_if_exists(combo_root / "s3" / "post" / MODEL_INFERENCE_FILE)
        pre_index = _index_rows(pre_rows)
        post_index = _index_rows(post_rows)
        for key, pre in pre_index.items():
            post = post_index.get(key)
            if post is None:
                continue
            row = _trend_row(str(combo.get("combo_id") or ""), key, pre, post)
            if row:
                rows.append(row)
    return rows


def _trend_row(combo_id: str, key: str, pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any] | None:
    pre_probs = _float_list(pre.get("action_probs"))
    post_probs = _float_list(post.get("action_probs"))
    pre_logits = _float_list(pre.get("logits"))
    post_logits = _float_list(post.get("logits"))
    cpc = _float_list(_first_present(post.get("theta_coverage_gain_per_path_costs"), post.get("coverage_gain_per_path_costs")))
    selected_index = _int_or_none(post.get("selected_action_index"))
    if selected_index is None or not pre_probs or not post_probs or not pre_logits or not post_logits or not cpc:
        return None
    n = min(len(pre_probs), len(post_probs), len(pre_logits), len(post_logits), len(cpc))
    if selected_index < 0 or selected_index >= n:
        return None
    valid = _valid_indices(post.get("action_mask"), n)
    if selected_index not in valid:
        return None
    best_index = max(valid, key=lambda index: cpc[index])
    pre_rank = _rank(pre_probs[:n], valid, selected_index)
    post_rank = _rank(post_probs[:n], valid, selected_index)
    pre_best_rank = _rank(pre_probs[:n], valid, best_index)
    post_best_rank = _rank(post_probs[:n], valid, best_index)
    if None in (pre_rank, post_rank, pre_best_rank, post_best_rank):
        return None
    pre_prob_gap = pre_probs[best_index] - pre_probs[selected_index]
    post_prob_gap = post_probs[best_index] - post_probs[selected_index]
    pre_logit_gap = pre_logits[best_index] - pre_logits[selected_index]
    post_logit_gap = post_logits[best_index] - post_logits[selected_index]
    prob_closure = post_prob_gap - pre_prob_gap
    logit_closure = post_logit_gap - pre_logit_gap
    pre_rank_gap = float(pre_rank - pre_best_rank)
    post_rank_gap = float(post_rank - post_best_rank)
    rank_closure = post_rank_gap - pre_rank_gap
    estimated = _estimated_updates_to_cross(post_prob_gap, prob_closure)
    return {
        "combo_id": combo_id,
        "strong_key": key,
        "selected_action_index": selected_index,
        "best_synthetic_candidate_index": best_index,
        "best_coverage_per_cost_candidate_index": best_index,
        "best_candidate_differs_from_selected": best_index != selected_index,
        "selected_pre_probability": pre_probs[selected_index],
        "selected_post_probability": post_probs[selected_index],
        "best_pre_probability": pre_probs[best_index],
        "best_post_probability": post_probs[best_index],
        "selected_probability_delta": post_probs[selected_index] - pre_probs[selected_index],
        "best_probability_delta": post_probs[best_index] - pre_probs[best_index],
        "selected_pre_logit": pre_logits[selected_index],
        "selected_post_logit": post_logits[selected_index],
        "best_pre_logit": pre_logits[best_index],
        "best_post_logit": post_logits[best_index],
        "selected_logit_delta": post_logits[selected_index] - pre_logits[selected_index],
        "best_logit_delta": post_logits[best_index] - pre_logits[best_index],
        "selected_pre_rank": pre_rank,
        "selected_post_rank": post_rank,
        "best_pre_rank": pre_best_rank,
        "best_post_rank": post_best_rank,
        "pre_probability_margin": pre_prob_gap,
        "post_probability_margin": post_prob_gap,
        "probability_gap_closure": prob_closure,
        "pre_logit_gap": pre_logit_gap,
        "post_logit_gap": post_logit_gap,
        "logit_gap_closure": logit_closure,
        "pre_rank_gap": pre_rank_gap,
        "post_rank_gap": post_rank_gap,
        "rank_gap_closure": rank_closure,
        "estimated_updates_to_cross_margin": estimated,
        "has_theta_coverage_gain_per_path_costs": bool(_float_list(post.get("theta_coverage_gain_per_path_costs"))),
        "has_hybrid_astar_path_costs": bool(_float_list(post.get("hybrid_astar_path_costs"))),
        "has_synthetic_los_candidate_features": isinstance(post.get("synthetic_los_blocker_candidate_counts"), list),
        "has_synthetic_hard_candidate_features": isinstance(post.get("synthetic_hard_obstacle_candidate_counts"), list),
    }


def _margin_crossing_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    estimates = [_finite_float(row.get("estimated_updates_to_cross_margin")) for row in rows]
    estimates = [value for value in estimates if value is not None]
    return {
        "schema_version": MARGIN_SCHEMA_VERSION,
        "combo_count": len({row.get("combo_id") for row in rows}),
        "strong_state_join_available_count": len(rows),
        "best_candidate_differs_from_selected_count": sum(1 for row in rows if row.get("best_candidate_differs_from_selected") is True),
        "best_probability_increased_count": sum(1 for row in rows if float(row.get("best_probability_delta") or 0.0) > 0.0),
        "selected_probability_decreased_count": sum(1 for row in rows if float(row.get("selected_probability_delta") or 0.0) < 0.0),
        "best_rank_closing_count": sum(1 for row in rows if float(row.get("rank_gap_closure") or 0.0) > 0.0),
        "mean_selected_to_best_probability_margin_pre": _mean(row.get("pre_probability_margin") for row in rows),
        "mean_selected_to_best_probability_margin_post": _mean(row.get("post_probability_margin") for row in rows),
        "mean_probability_gap_closure": _mean(row.get("probability_gap_closure") for row in rows),
        "mean_logit_gap_closure": _mean(row.get("logit_gap_closure") for row in rows),
        "mean_best_rank_gap_closure": _mean(row.get("rank_gap_closure") for row in rows),
        "mean_estimated_updates_to_cross_margin": mean(estimates) if estimates else None,
        "rows": rows,
    }


def _credit_binding_audit(stage26_4_root: Path, trend_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_rows = _read_train_rows(stage26_4_root)
    selected_pairs = {
        (
            str(row.get("scenario_id")),
            _int_or_none(row.get("step_index")) if _int_or_none(row.get("step_index")) is not None else -1,
            _int_or_none(_first_present(row.get("selected_action_index"), row.get("action_index"))),
        )
        for row in train_rows
    }
    best_selected = 0
    best_differs = 0
    for row in trend_rows:
        if row.get("best_candidate_differs_from_selected") is True:
            best_differs += 1
        scenario_id, step_index = _key_parts(str(row.get("strong_key") or ""))
        pair = (scenario_id, step_index, _int_or_none(row.get("best_synthetic_candidate_index")))
        if pair in selected_pairs:
            best_selected += 1
    advantages = [_finite_float(_first_present(row.get("advantage"), row.get("gae_advantage"), row.get("normalized_advantage"))) for row in train_rows]
    advantages = [value for value in advantages if value is not None]
    return {
        "schema_version": CREDIT_SCHEMA_VERSION,
        "trainable_batch_row_count": len(train_rows),
        "best_candidate_differs_from_selected_count": best_differs,
        "best_candidate_selected_in_train_count": best_selected,
        "counterfactual_best_not_directly_credited": best_differs > 0 and best_selected == 0,
        "selected_action_advantage_mean": mean(advantages) if advantages else None,
        "selected_action_advantage_min": min(advantages) if advantages else None,
        "selected_action_advantage_max": max(advantages) if advantages else None,
    }


def _candidate_feature_signal_audit(trend_rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing: set[str] = set()
    if not any(row.get("has_theta_coverage_gain_per_path_costs") for row in trend_rows):
        missing.add("theta_coverage_gain_per_path_costs")
    if not any(row.get("has_hybrid_astar_path_costs") for row in trend_rows):
        missing.add("hybrid_astar_path_costs")
    if not any(row.get("has_synthetic_los_candidate_features") for row in trend_rows):
        missing.add("synthetic_los_blocker_candidate_counts")
    if not any(row.get("has_synthetic_hard_candidate_features") for row in trend_rows):
        missing.add("synthetic_hard_obstacle_candidate_counts")
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "candidate_feature_signal_missing": bool(missing),
        "missing_candidate_feature_fields": sorted(missing),
        "checked_join_row_count": len(trend_rows),
    }


def _route(
    *,
    boundary_reasons: list[str],
    input_reasons: list[str],
    worker_reasons: list[str],
    margin_audit: dict[str, Any],
    credit_audit: dict[str, Any],
    feature_audit: dict[str, Any],
    stage26_4_summary: dict[str, Any],
) -> tuple[str, str, str]:
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "boundary fields are open"
    if input_reasons:
        return "failed", ROUTE_INPUTS, "Stage26.4 inputs are missing or not routed to discrete margin calibration"
    if worker_reasons:
        return "failed", ROUTE_PARALLEL_COLLECTOR, "Stage26.4 did not use the required parallel collector worker count"
    if int(margin_audit.get("strong_state_join_available_count") or 0) <= 0:
        return "failed", ROUTE_BINDING, "pre/post margin audit rows could not be strongly joined"
    if credit_audit.get("counterfactual_best_not_directly_credited") is True:
        return "failed", ROUTE_EXPLORATION_CREDIT, "best synthetic candidate was not sampled as a trainable selected action"
    if feature_audit.get("candidate_feature_signal_missing") is True:
        return "failed", ROUTE_FEATURE_EXPOSURE, "policy candidate features do not expose required synthetic/path signals"
    if int(margin_audit.get("best_rank_closing_count") or 0) > 0:
        return "failed", ROUTE_ITERATIVE, "best candidate rank is approaching but has not crossed the discrete boundary"
    if _mean(row.get("best_probability_delta") for row in margin_audit.get("rows", [])) <= 0.0:
        return "failed", ROUTE_GRADIENT_DIRECTION, "best candidate probability is not increasing"
    if int(stage26_4_summary.get("max_selected_viewpoint_changed_count") or 0) > 0:
        return "failed", ROUTE_CREDIT, "viewpoint changed but Stage26.4 did not show coverage/path-cost improvement"
    return "failed", ROUTE_ITERATIVE, "margin remains too large for one bounded update"


def _blocking_reasons(
    route: str,
    boundary_reasons: list[str],
    input_reasons: list[str],
    worker_reasons: list[str],
    margin_audit: dict[str, Any],
    credit_audit: dict[str, Any],
    feature_audit: dict[str, Any],
) -> list[str]:
    if route == ROUTE_BOUNDARY:
        return boundary_reasons
    if route == ROUTE_INPUTS:
        return input_reasons
    if route == ROUTE_PARALLEL_COLLECTOR:
        return worker_reasons
    if route == ROUTE_BINDING:
        return ["stage26_5_margin_strong_join_unavailable"]
    if route == ROUTE_EXPLORATION_CREDIT:
        return ["counterfactual_best_candidate_not_directly_sampled_for_ppo"]
    if route == ROUTE_FEATURE_EXPOSURE:
        return [f"candidate_feature_missing:{field}" for field in feature_audit.get("missing_candidate_feature_fields", [])]
    if route == ROUTE_ITERATIVE:
        return ["best_candidate_rank_closing_without_crossing_discrete_boundary"]
    if route == ROUTE_GRADIENT_DIRECTION:
        return ["best_candidate_probability_not_increasing"]
    return []


def _recommended_next_config(route: str, stage26_4_root: Path, margin: dict[str, Any], credit: dict[str, Any], feature: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage26-5-recommended-next-config/v1",
        "source_stage26_4_root": str(stage26_4_root),
        "recommended_route": route,
        "hybrid_astar_candidate_eval_workers": 4,
        "coverage_source": COVERAGE_SOURCE,
        "path_cost_source": PATH_COST_SOURCE,
        "synthetic_terrain_hash": SYNTHETIC_HASH,
        "synthetic_source_kind": SYNTHETIC_SOURCE_KIND,
        "mean_estimated_updates_to_cross_margin": margin.get("mean_estimated_updates_to_cross_margin"),
        "counterfactual_best_not_directly_credited": credit.get("counterfactual_best_not_directly_credited"),
        "candidate_feature_signal_missing": feature.get("candidate_feature_signal_missing"),
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _input_rejections(summary: dict[str, Any]) -> list[str]:
    if not summary:
        return ["missing_stage26_4_summary"]
    reasons: list[str] = []
    if summary.get("status") != "failed":
        reasons.append("stage26_4_status_not_failed")
    if summary.get("next_required_change") != ROUTE_FROM_STAGE26_4:
        reasons.append("stage26_4_route_not_discrete_margin")
    if summary.get("coverage_source") != COVERAGE_SOURCE:
        reasons.append("stage26_4_coverage_source_mismatch")
    if summary.get("path_cost_source") != PATH_COST_SOURCE:
        reasons.append("stage26_4_path_cost_source_mismatch")
    if summary.get("synthetic_source_kind") != SYNTHETIC_SOURCE_KIND:
        reasons.append("stage26_4_synthetic_source_kind_mismatch")
    if summary.get("synthetic_terrain_hash") != SYNTHETIC_HASH:
        reasons.append("stage26_4_synthetic_hash_mismatch")
    if _finite_float(summary.get("max_traversable_slope_deg")) != 30.0:
        reasons.append("stage26_4_max_traversable_slope_not_30")
    for field in ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary"):
        if summary.get(field) is True:
            reasons.append(f"stage26_4_{field}_enabled")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        reasons.append("canary_traffic_fraction_nonzero")
    return reasons


def _stage26_4_worker_count(stage26_4_root: Path, repo_root: Path) -> int:
    payload = _read_json_if_exists(stage26_4_root / STAGE26_4_COLLECTOR_CONFIG_FILE)
    value = payload.get("hybrid_astar_candidate_eval_workers")
    parsed = _int_or_none(value)
    if parsed is not None:
        return parsed
    config_payload = _read_json_if_exists(repo_root / "configs" / "xunce_stage26_4_synthetic_policy_update_signal_strength_repair_v1.json")
    parsed = _int_or_none(config_payload.get("hybrid_astar_candidate_eval_workers"))
    return parsed if parsed is not None else 0


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("policy") not in (None, "xunce"):
            continue
        key = _strong_key(row)
        if key and key not in indexed:
            indexed[key] = row
    return indexed


def _strong_key(row: dict[str, Any]) -> str | None:
    fields = ("scenario_id", "step_index", "current_cell", "covered_cells_hash", "candidate_set_hash", "synthetic_terrain_hash")
    if any(row.get(field) is None for field in fields):
        return None
    return "|".join(_normalize_json_value(row[field]) for field in fields)


def _valid_indices(mask: Any, n: int) -> list[int]:
    if isinstance(mask, list):
        return [index for index in range(n) if index < len(mask) and mask[index] is True]
    return list(range(n))


def _rank(values: list[float], valid_indices: list[int], index: int) -> int | None:
    if index not in valid_indices:
        return None
    ordered = sorted(valid_indices, key=lambda item: values[item], reverse=True)
    return ordered.index(index) + 1


def _estimated_updates_to_cross(post_gap: float, closure: float) -> float | None:
    if post_gap >= 0.0:
        return 0.0
    if closure <= 0.0:
        return None
    return abs(post_gap) / closure


def _read_train_rows(stage26_4_root: Path) -> list[dict[str, Any]]:
    root = stage26_4_root / "s26_1" / "s21_3"
    rows: list[dict[str, Any]] = []
    for path in root.glob("*trainable*batch*.jsonl"):
        rows.extend(_read_jsonl_if_exists(path))
    return rows


def _key_parts(strong_key: str) -> tuple[str, int]:
    parts = strong_key.split("|")
    if len(parts) < 2:
        return "", -1
    scenario = _decode_json_part(parts[0])
    step = _decode_json_part(parts[1])
    parsed_step = _int_or_none(step)
    return str(scenario), parsed_step if parsed_step is not None else -1


def _decode_json_part(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"config schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    config = {
        "stage26_4_root": DEFAULT_STAGE26_4_ROOT,
        "required_hybrid_astar_candidate_eval_workers": 4,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    } | payload
    config["stage26_4_root"] = str(_resolve_path(Path(str(config["stage26_4_root"])), repo_root))
    config["required_hybrid_astar_candidate_eval_workers"] = _positive_int(
        config["required_hybrid_astar_candidate_eval_workers"],
        "required_hybrid_astar_candidate_eval_workers",
    )
    return config


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    output: list[float] = []
    for item in value:
        parsed = _finite_float(item)
        if parsed is None:
            return []
        output.append(parsed)
    return output


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _mean(values: Any) -> float:
    parsed = [_finite_float(value) for value in values]
    parsed = [value for value in parsed if value is not None]
    return mean(parsed) if parsed else 0.0


def _normalize_json_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.5 Synthetic Discrete Margin Crossing Calibration",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- worker_count: `{summary['stage26_4_hybrid_astar_candidate_eval_workers']}`",
            f"- strong_state_join_available_count: `{summary['strong_state_join_available_count']}`",
            f"- mean_probability_gap_closure: `{summary['mean_probability_gap_closure']}`",
            f"- mean_best_rank_gap_closure: `{summary['mean_best_rank_gap_closure']}`",
            f"- counterfactual_best_not_directly_credited: `{summary['counterfactual_best_not_directly_credited']}`",
            f"- candidate_feature_signal_missing: `{summary['candidate_feature_signal_missing']}`",
            "",
            "Stage26.5 is a read-only diagnostic. It does not run PPO, publish checkpoints, replace default policy, connect an executor, start canary traffic, regenerate synthetic terrain, or change planner/reward/network semantics.",
            "",
        ]
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
