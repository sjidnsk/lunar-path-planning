from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import xunce_artifact_io as artifact_io


STAGE_ID = "xunce-stage26-8f-scenario-diversity-and-policy-margin-audit"
CONFIG_SCHEMA_VERSION = "xunce-stage26-8f-scenario-diversity-and-policy-margin-audit-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-8f-summary/v1"
SOURCE_INDEX_SCHEMA_VERSION = "xunce-stage26-8f-completed-eval-source-index/v1"
SCENARIO_AUDIT_SCHEMA_VERSION = "xunce-stage26-8f-scenario-diversity-audit/v1"
MARGIN_AUDIT_SCHEMA_VERSION = "xunce-stage26-8f-policy-margin-crossing-audit/v1"
OPPORTUNITY_AUDIT_SCHEMA_VERSION = "xunce-stage26-8f-candidate-opportunity-audit/v1"
SENSITIVITY_AUDIT_SCHEMA_VERSION = "xunce-stage26-8f-coverage-metric-sensitivity-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-8f-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-8f-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_8f_scenario_diversity_and_policy_margin_audit_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit_v1"
)
DEFAULT_STAGE26_8D_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8d_resumable_seed_horizon_execution_v1"
)

SUMMARY_FILE = "xunce-stage26-8f-summary.json"
SOURCE_INDEX_FILE = "xunce-stage26-8f-completed-eval-source-index.json"
SCENARIO_AUDIT_FILE = "xunce-stage26-8f-scenario-diversity-audit.json"
MARGIN_AUDIT_FILE = "xunce-stage26-8f-policy-margin-crossing-audit.json"
OPPORTUNITY_AUDIT_FILE = "xunce-stage26-8f-candidate-opportunity-audit.json"
SENSITIVITY_AUDIT_FILE = "xunce-stage26-8f-coverage-metric-sensitivity-audit.json"
ROUTING_FILE = "xunce-stage26-8f-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-8f-report.md"
MANIFEST_FILE = "xunce-stage26-8f-manifest.json"

COVERAGE_DENOMINATOR_SOURCE = "main_coverable_cells/v1"
POST_UPDATE_SUCCESS_METRIC = "main_coverable_coverage_efficiency/v1"
COVERAGE_SOURCE = "endpoint_theta_slope_obstacle_los/v1"
PATH_COST_SOURCE = "hybrid_astar_pose_path/v1"
SYNTHETIC_SOURCE_KIND = "synthetic_terrain_obstacle_proxy/v1"
ACTION_SPACE_TYPE = "hybrid_discrete_xy_continuous_theta/v1"

ROUTE_INPUTS = "rerun_stage26_8f_required_inputs"
ROUTE_CONTINUE_JOBS = "continue_stage26_8d_seed_horizon_jobs"
ROUTE_AUDIT_BINDING = "repair_stage26_8f_audit_binding"
ROUTE_SCENARIO_DIVERSITY = "repair_stage26_synthetic_scenario_diversity"
ROUTE_CANDIDATE_LOGGING = "repair_stage26_8f_candidate_metric_logging"
ROUTE_METRIC_SENSITIVITY = "repair_stage26_synthetic_candidate_generation_or_metric_sensitivity"
ROUTE_POLICY_SIGNAL = "repair_stage26_synthetic_policy_update_signal_strength"
ROUTE_MARGIN_CROSSING = "calibrate_stage26_synthetic_discrete_margin_crossing_after_credit"
ROUTE_CREDIT = "repair_stage26_synthetic_exploration_credit_assignment"
ROUTE_COUNTERFACTUAL = "run_stage26_8g_counterfactual_action_replay_probe"
ROUTE_BOUNDARY = "resolve_stage26_8f_boundary_rejections"

BOUNDARY_FIELDS = (
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)
STRONG_JOIN_FIELDS = (
    "scenario_id",
    "step_index",
    "current_cell",
    "covered_cells_hash",
    "candidate_set_hash",
    "synthetic_terrain_hash",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.8F scenario diversity and policy margin audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "next_required_change": summary["next_required_change"],
                "diagnostic_conclusion": summary.get("diagnostic_conclusion"),
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(config_path, repo_root)
    output_root = _resolve_path(output_root, repo_root)
    artifact_io.make_dirs(output_root)

    stage26_8d_root = Path(config["stage26_8d_root"])
    stage26_8d_summary = _read_json_if_exists(stage26_8d_root / "xunce-stage26-8d-summary.json")
    job_rows = _read_jsonl_if_exists(stage26_8d_root / "xunce-stage26-8d-job-state.jsonl")
    boundary_rejections = _boundary_rejections(config)
    input_rejections = _input_rejections(stage26_8d_summary)
    source_index = _completed_eval_source_index(job_rows, min_count=int(config["min_completed_eval_count"]))

    if boundary_rejections or input_rejections or len(source_index["completed_eval_sources"]) < int(config["min_completed_eval_count"]):
        scenario_audit = _empty_scenario_audit()
        margin_audit = _empty_margin_audit()
        opportunity_audit = _empty_opportunity_audit()
        sensitivity_audit = _empty_sensitivity_audit()
    else:
        evals = [_load_eval_source(row) for row in source_index["completed_eval_sources"]]
        scenario_audit = _scenario_diversity_audit(evals, duplicate_ratio_threshold=float(config["scenario_duplicate_ratio_threshold"]))
        margin_audit = _policy_margin_audit(
            evals,
            near_boundary_threshold=float(config["near_boundary_probability_margin_threshold"]),
            margin_delta_ratio_for_too_small=float(config["margin_delta_ratio_for_too_small"]),
        )
        opportunity_audit = _candidate_opportunity_audit(evals)
        sensitivity_audit = _coverage_metric_sensitivity_audit(
            evals,
            proxy_range_epsilon=float(config["candidate_proxy_range_epsilon"]),
        )

    route = _route(
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        source_index=source_index,
        min_completed_eval_count=int(config["min_completed_eval_count"]),
        scenario_audit=scenario_audit,
        margin_audit=margin_audit,
        opportunity_audit=opportunity_audit,
        sensitivity_audit=sensitivity_audit,
    )
    status = "failed" if route in {ROUTE_INPUTS, ROUTE_AUDIT_BINDING, ROUTE_BOUNDARY} else "passed"
    diagnostic_conclusion = _diagnostic_conclusion(route)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "diagnostic_conclusion": diagnostic_conclusion,
        "stage26_8d_root": str(stage26_8d_root),
        "stage26_8d_status": stage26_8d_summary.get("status"),
        "stage26_8d_next_required_change": stage26_8d_summary.get("next_required_change"),
        "completed_eval_source_count": len(source_index["completed_eval_sources"]),
        "min_completed_eval_count": int(config["min_completed_eval_count"]),
        "scenario_diversity_suspect": bool(scenario_audit.get("scenario_diversity_suspect")),
        "strong_state_join_available_count": int(margin_audit.get("strong_state_join_available_count") or 0),
        "selected_action_changed_count": int(margin_audit.get("selected_action_changed_count") or 0),
        "margin_evaluable_join_count": int(margin_audit.get("margin_evaluable_join_count") or 0),
        "margin_required_field_missing_count": int(margin_audit.get("margin_required_field_missing_count") or 0),
        "mean_abs_probability_delta": _float(margin_audit.get("mean_abs_probability_delta")),
        "mean_top1_probability_margin": _float(margin_audit.get("mean_top1_probability_margin")),
        "policy_delta_too_small_for_margin": bool(margin_audit.get("policy_delta_too_small_for_margin")),
        "candidate_opportunity_fields_missing_count": int(opportunity_audit.get("candidate_opportunity_fields_missing_count") or 0),
        "best_gain_per_cost_selected_count": int(opportunity_audit.get("best_gain_per_cost_selected_count") or 0),
        "candidate_proxy_variance_row_count": int(sensitivity_audit.get("candidate_proxy_variance_row_count") or 0),
        "selected_vs_best_proxy_diff_count": int(sensitivity_audit.get("selected_vs_best_proxy_diff_count") or 0),
        "coverage_denominator_source": COVERAGE_DENOMINATOR_SOURCE,
        "post_update_success_metric": POST_UPDATE_SUCCESS_METRIC,
        "coverage_source": COVERAGE_SOURCE,
        "path_cost_source": PATH_COST_SOURCE,
        "synthetic_source_kind": SYNTHETIC_SOURCE_KIND,
        "action_space_type": ACTION_SPACE_TYPE,
        "max_traversable_slope_deg": 30.0,
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "summary": str(output_root / SUMMARY_FILE),
    }
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "diagnostic_conclusion": diagnostic_conclusion,
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "summary_status": status,
        "next_required_change": route,
        "artifacts": {
            "summary": str(output_root / SUMMARY_FILE),
            "completed_eval_source_index": str(output_root / SOURCE_INDEX_FILE),
            "scenario_diversity_audit": str(output_root / SCENARIO_AUDIT_FILE),
            "policy_margin_crossing_audit": str(output_root / MARGIN_AUDIT_FILE),
            "candidate_opportunity_audit": str(output_root / OPPORTUNITY_AUDIT_FILE),
            "coverage_metric_sensitivity_audit": str(output_root / SENSITIVITY_AUDIT_FILE),
            "routing": str(output_root / ROUTING_FILE),
            "report": str(output_root / REPORT_FILE),
        },
    }
    _write_json(output_root / SOURCE_INDEX_FILE, source_index)
    _write_json(output_root / SCENARIO_AUDIT_FILE, scenario_audit)
    _write_json(output_root / MARGIN_AUDIT_FILE, margin_audit)
    _write_json(output_root / OPPORTUNITY_AUDIT_FILE, opportunity_audit)
    _write_json(output_root / SENSITIVITY_AUDIT_FILE, sensitivity_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    artifact_io.write_text(
        output_root / REPORT_FILE,
        _render_report(summary, scenario_audit, margin_audit, opportunity_audit, sensitivity_audit),
    )
    return summary


def _completed_eval_source_index(job_rows: list[dict[str, Any]], *, min_count: int) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    pending_sources: list[dict[str, Any]] = []
    for row in job_rows:
        if row.get("status") == "pending":
            pending_sources.append(
                {
                    "job_id": row.get("job_id"),
                    "horizon_steps": row.get("horizon_steps"),
                    "seed": row.get("seed"),
                    "phase": row.get("phase"),
                    "summary_path": row.get("summary_path"),
                    "resume_decision": row.get("resume_decision"),
                }
            )
        if row.get("status") != "complete" or row.get("phase") != "complete":
            continue
        stage26_3_root = Path(str(row.get("stage26_3_root") or ""))
        summary_path = Path(str(row.get("summary_path") or "")) if row.get("summary_path") else stage26_3_root / "xunce-stage26-3-summary.json"
        if not artifact_io.path_is_dir(stage26_3_root) or not artifact_io.path_is_file(summary_path):
            continue
        sources.append(
            {
                "job_id": row.get("job_id"),
                "horizon_steps": row.get("horizon_steps"),
                "seed": row.get("seed"),
                "seed_index": row.get("seed_index"),
                "stage26_3_root": str(stage26_3_root),
                "summary_path": str(summary_path),
                "stage26_3_source": row.get("stage26_3_source"),
                "resume_decision": row.get("resume_decision"),
                "main_coverage_per_100m_delta": _float(row.get("main_coverage_per_100m_delta")),
            }
        )
    return {
        "schema_version": SOURCE_INDEX_SCHEMA_VERSION,
        "min_completed_eval_count": min_count,
        "completed_eval_source_count": len(sources),
        "completed_eval_sources": sources,
        "pending_job_count": len(pending_sources),
        "pending_sources": pending_sources,
        "insufficient_completed_eval_sources": len(sources) < min_count,
    }


def _load_eval_source(source: dict[str, Any]) -> dict[str, Any]:
    root = Path(source["stage26_3_root"])
    pre_dir = root / "pre"
    post_dir = root / "post"
    return {
        "source": source,
        "summary": _read_json_if_exists(Path(source["summary_path"])),
        "action_audit": _read_json_if_exists(root / "xunce-stage26-3-synthetic-action-change-audit.json"),
        "trajectory_audit": _read_json_if_exists(root / "xunce-stage26-3-trajectory-delta-audit.json"),
        "pre_inference": _read_jsonl_if_exists(pre_dir / "xunce-exploration-coverage-model-inference.jsonl"),
        "post_inference": _read_jsonl_if_exists(post_dir / "xunce-exploration-coverage-model-inference.jsonl"),
        "pre_episodes": _read_jsonl_if_exists(pre_dir / "xunce-exploration-coverage-episodes.jsonl"),
        "post_episodes": _read_jsonl_if_exists(post_dir / "xunce-exploration-coverage-episodes.jsonl"),
        "pre_steps": _read_jsonl_if_exists(pre_dir / "xunce-exploration-coverage-steps.jsonl"),
        "post_steps": _read_jsonl_if_exists(post_dir / "xunce-exploration-coverage-steps.jsonl"),
    }


def _scenario_diversity_audit(evals: list[dict[str, Any]], *, duplicate_ratio_threshold: float) -> dict[str, Any]:
    signatures: list[dict[str, Any]] = []
    for payload in evals:
        source = payload["source"]
        episodes = {str(row.get("scenario_id")): row for row in payload["pre_episodes"]}
        by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in payload["pre_inference"]:
            by_scenario[str(row.get("scenario_id"))].append(row)
        for scenario_id, rows in sorted(by_scenario.items()):
            rows = sorted(rows, key=lambda item: int(item.get("step_index") or 0))
            episode = episodes.get(scenario_id, {})
            candidate_sequence = [row.get("candidate_set_hash") for row in rows]
            covered_sequence = [row.get("covered_cells_hash") for row in rows]
            selected_action_sequence = [row.get("selected_action_index") for row in rows]
            selected_viewpoint_sequence = [row.get("selected_viewpoint") for row in rows]
            metrics = {
                "final_coverage_rate": _float(episode.get("final_coverage_rate")),
                "coverage_curve_auc": _float(episode.get("coverage_curve_auc")),
                "coverage_per_100m": _float(episode.get("coverage_per_100m")),
                "path_cost_total_m": _float(episode.get("path_cost_total_m")),
                "new_covered_cell_count": _float(episode.get("new_covered_cell_count")),
            }
            signature = {
                "job_id": source.get("job_id"),
                "horizon_steps": source.get("horizon_steps"),
                "seed": source.get("seed"),
                "scenario_id": scenario_id,
                "source_summary_path": source.get("summary_path"),
                "first_current_cell": rows[0].get("current_cell") if rows else None,
                "synthetic_terrain_hash": rows[0].get("synthetic_terrain_hash") if rows else "",
                "candidate_set_sequence_hash": _stable_hash(candidate_sequence),
                "covered_cells_sequence_hash": _stable_hash(covered_sequence),
                "selected_action_sequence_hash": _stable_hash(selected_action_sequence),
                "selected_viewpoint_sequence_hash": _stable_hash(selected_viewpoint_sequence),
                "episode_metrics_hash": _stable_hash(metrics),
                "scenario_context_key": _stable_hash(
                    {
                        "job_id": source.get("job_id"),
                        "horizon_steps": source.get("horizon_steps"),
                        "seed": source.get("seed"),
                        "source_summary_path": source.get("summary_path"),
                        "scenario_id": scenario_id,
                    }
                ),
                "scenario_content_signature_hash": "",
                "scenario_within_job_signature_hash": "",
                "step_count": len(rows),
                **metrics,
            }
            content_payload = {
                "first_current_cell": signature["first_current_cell"],
                "synthetic_terrain_hash": signature["synthetic_terrain_hash"],
                "candidate_set_sequence_hash": signature["candidate_set_sequence_hash"],
                "covered_cells_sequence_hash": signature["covered_cells_sequence_hash"],
                "selected_action_sequence_hash": signature["selected_action_sequence_hash"],
                "selected_viewpoint_sequence_hash": signature["selected_viewpoint_sequence_hash"],
                "episode_metrics_hash": signature["episode_metrics_hash"],
            }
            signature["scenario_content_signature_hash"] = _stable_hash(content_payload)
            signature["scenario_within_job_signature_hash"] = _stable_hash(
                {
                    "job_id": source.get("job_id"),
                    "horizon_steps": source.get("horizon_steps"),
                    "seed": source.get("seed"),
                    **content_payload,
                }
            )
            signatures.append(signature)

    scenario_count = len(signatures)
    candidate_pairs = _duplicate_pair_count([row["candidate_set_sequence_hash"] for row in signatures])
    action_pairs = _duplicate_pair_count([row["selected_action_sequence_hash"] for row in signatures])
    episode_pairs = _duplicate_pair_count([row["episode_metrics_hash"] for row in signatures])
    within_job_counter = Counter(row["scenario_within_job_signature_hash"] for row in signatures)
    content_counter = Counter(row["scenario_content_signature_hash"] for row in signatures)
    within_job_duplicate_groups = [key for key, count in within_job_counter.items() if count > 1]
    content_duplicate_groups = [key for key, count in content_counter.items() if count > 1]
    within_job_duplicate_rows = sum(within_job_counter[key] for key in within_job_duplicate_groups)
    content_duplicate_rows = sum(content_counter[key] for key in content_duplicate_groups)
    duplicate_ratio = within_job_duplicate_rows / scenario_count if scenario_count else 0.0
    return {
        "schema_version": SCENARIO_AUDIT_SCHEMA_VERSION,
        "scenario_signature_count": scenario_count,
        "scenario_signatures": signatures,
        "identical_candidate_sequence_pair_count": candidate_pairs,
        "identical_selected_action_sequence_pair_count": action_pairs,
        "identical_episode_metric_pair_count": episode_pairs,
        "scenario_signature_duplicate_group_count": len(within_job_duplicate_groups),
        "scenario_signature_duplicate_row_count": within_job_duplicate_rows,
        "cross_job_content_duplicate_group_count": len(content_duplicate_groups),
        "cross_job_content_duplicate_row_count": content_duplicate_rows,
        "scenario_signature_duplicate_ratio": duplicate_ratio,
        "scenario_diversity_suspect": bool(scenario_count) and duplicate_ratio >= duplicate_ratio_threshold,
        "duplicate_ratio_threshold": duplicate_ratio_threshold,
    }


def _policy_margin_audit(evals: list[dict[str, Any]], *, near_boundary_threshold: float, margin_delta_ratio_for_too_small: float) -> dict[str, Any]:
    joined_rows: list[dict[str, Any]] = []
    missing_field_count = 0
    duplicate_key_count = 0
    unmatched_pre_count = 0
    unmatched_post_count = 0
    strong_join_count = 0
    margin_field_reasons: Counter[str] = Counter()
    for payload in evals:
        pre_index: dict[tuple[Any, ...], dict[str, Any]] = {}
        post_index: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in payload["pre_inference"]:
            key = _strong_key(row)
            if key is None:
                missing_field_count += 1
                continue
            if key in pre_index:
                duplicate_key_count += 1
            pre_index[key] = row
        for row in payload["post_inference"]:
            key = _strong_key(row)
            if key is None:
                missing_field_count += 1
                continue
            if key in post_index:
                duplicate_key_count += 1
            post_index[key] = row
        unmatched_pre_count += len([key for key in pre_index if key not in post_index])
        unmatched_post_count += len([key for key in post_index if key not in pre_index])
        for key in sorted(set(pre_index) & set(post_index), key=str):
            strong_join_count += 1
            margin_rejections = _margin_field_rejections(pre_index[key], post_index[key])
            if margin_rejections:
                margin_field_reasons.update(margin_rejections)
                continue
            row = _joined_margin_row(payload["source"], pre_index[key], post_index[key])
            joined_rows.append(row)

    mean_abs_delta = _mean([row["mean_abs_probability_delta"] for row in joined_rows])
    mean_margin = _mean([row["pre_top1_probability_margin"] for row in joined_rows])
    mean_closure = _mean([row["probability_margin_closure_rate"] for row in joined_rows if row["probability_margin_closure_rate"] is not None])
    changed_count = len([row for row in joined_rows if row["selected_action_changed"]])
    near_boundary_count = len([row for row in joined_rows if row["pre_top1_probability_margin"] <= near_boundary_threshold])
    return {
        "schema_version": MARGIN_AUDIT_SCHEMA_VERSION,
        "strong_state_join_available_count": strong_join_count,
        "margin_evaluable_join_count": len(joined_rows),
        "pre_post_required_field_missing_count": missing_field_count,
        "margin_required_field_missing_count": sum(margin_field_reasons.values()),
        "margin_required_field_missing_reasons": dict(sorted(margin_field_reasons.items())),
        "duplicate_strong_key_count": duplicate_key_count,
        "unmatched_pre_row_count": unmatched_pre_count,
        "unmatched_post_row_count": unmatched_post_count,
        "selected_action_changed_count": changed_count,
        "selected_viewpoint_changed_count": len([row for row in joined_rows if row["selected_viewpoint_changed"]]),
        "mean_abs_probability_delta": mean_abs_delta,
        "mean_top1_probability_margin": mean_margin,
        "mean_margin_closure_rate": mean_closure,
        "near_boundary_probability_margin_threshold": near_boundary_threshold,
        "near_boundary_row_count": near_boundary_count,
        "policy_delta_too_small_for_margin": bool(joined_rows) and mean_abs_delta < mean_margin * margin_delta_ratio_for_too_small,
        "margin_delta_ratio_for_too_small": margin_delta_ratio_for_too_small,
        "estimated_updates_to_cross_margin_mean": _mean(
            [row["estimated_updates_to_cross_margin"] for row in joined_rows if row["estimated_updates_to_cross_margin"] is not None]
        ),
        "sample_rows": joined_rows[:50],
    }


def _joined_margin_row(source: dict[str, Any], pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    pre_probs = [_float(value) for value in pre.get("action_probs") or []]
    post_probs = [_float(value) for value in post.get("action_probs") or []]
    pre_logits = [_float(value) for value in pre.get("logits") or []]
    post_logits = [_float(value) for value in post.get("logits") or []]
    pre_selected = _int(pre.get("selected_action_index"))
    post_selected = _int(post.get("selected_action_index"))
    pre_top = _top_two(pre_probs)
    post_top = _top_two(post_probs)
    pre_logit_top = _top_two(pre_logits)
    post_logit_top = _top_two(post_logits)
    pre_margin = pre_top["top1_value"] - pre_top["top2_value"]
    post_margin = post_top["top1_value"] - post_top["top2_value"]
    margin_closure = pre_margin - post_margin
    closure_rate = margin_closure / pre_margin if pre_margin > 0.0 else None
    mean_abs_delta = _mean([abs(after - before) for before, after in zip(pre_probs, post_probs)])
    selected_delta = _indexed(post_probs, pre_selected) - _indexed(pre_probs, pre_selected)
    estimated_updates = pre_margin / margin_closure if margin_closure > 1.0e-12 else None
    return {
        "job_id": source.get("job_id"),
        "horizon_steps": source.get("horizon_steps"),
        "seed": source.get("seed"),
        "scenario_id": pre.get("scenario_id"),
        "step_index": pre.get("step_index"),
        "current_cell": pre.get("current_cell"),
        "candidate_set_hash": pre.get("candidate_set_hash"),
        "pre_selected_action_index": pre_selected,
        "post_selected_action_index": post_selected,
        "selected_action_changed": pre_selected != post_selected,
        "selected_viewpoint_changed": pre.get("selected_viewpoint") != post.get("selected_viewpoint"),
        "pre_selected_probability": _indexed(pre_probs, pre_selected),
        "post_preselected_probability": _indexed(post_probs, pre_selected),
        "selected_action_probability_delta": selected_delta,
        "mean_abs_probability_delta": mean_abs_delta,
        "pre_top1_index": pre_top["top1_index"],
        "pre_top2_index": pre_top["top2_index"],
        "post_top1_index": post_top["top1_index"],
        "post_top2_index": post_top["top2_index"],
        "pre_top1_probability_margin": pre_margin,
        "post_top1_probability_margin": post_margin,
        "pre_top1_logit_margin": pre_logit_top["top1_value"] - pre_logit_top["top2_value"],
        "post_top1_logit_margin": post_logit_top["top1_value"] - post_logit_top["top2_value"],
        "probability_margin_closure": margin_closure,
        "probability_margin_closure_rate": closure_rate,
        "estimated_updates_to_cross_margin": estimated_updates,
    }


def _margin_field_rejections(pre: dict[str, Any], post: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for prefix, row in (("pre", pre), ("post", post)):
        selected = _int(row.get("selected_action_index"))
        probs = row.get("action_probs")
        logits = row.get("logits")
        if selected is None:
            reasons.append(f"{prefix}_selected_action_index_missing")
        if not isinstance(probs, list) or not probs:
            reasons.append(f"{prefix}_action_probs_missing")
        elif selected is not None and (selected < 0 or selected >= len(probs)):
            reasons.append(f"{prefix}_selected_action_index_out_of_probability_range")
        if not isinstance(logits, list) or not logits:
            reasons.append(f"{prefix}_logits_missing")
        elif selected is not None and (selected < 0 or selected >= len(logits)):
            reasons.append(f"{prefix}_selected_action_index_out_of_logit_range")
    pre_probs = pre.get("action_probs")
    post_probs = post.get("action_probs")
    if isinstance(pre_probs, list) and isinstance(post_probs, list) and len(pre_probs) != len(post_probs):
        reasons.append("probability_vector_length_mismatch")
    pre_logits = pre.get("logits")
    post_logits = post.get("logits")
    if isinstance(pre_logits, list) and isinstance(post_logits, list) and len(pre_logits) != len(post_logits):
        reasons.append("logit_vector_length_mismatch")
    return reasons


def _candidate_opportunity_audit(evals: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fields_missing = 0
    synthetic_credit_feature_missing = 0
    for payload in evals:
        for row in payload["pre_inference"]:
            selected = _int(row.get("selected_action_index"))
            probs = [_float(value) for value in row.get("action_probs") or []]
            gain_per_cost = row.get("theta_coverage_gain_per_path_costs")
            new_visible = row.get("theta_new_visible_cell_counts")
            if row.get("synthetic_credit_feature_rows") is None:
                synthetic_credit_feature_missing += 1
            if not isinstance(gain_per_cost, list) or not isinstance(new_visible, list):
                fields_missing += 1
                continue
            best_gpc = _max_index(gain_per_cost)
            best_visible = _max_index(new_visible)
            if best_gpc is None or best_visible is None:
                fields_missing += 1
                continue
            prob_ranks = _probability_ranks(probs)
            rows.append(
                {
                    "job_id": payload["source"].get("job_id"),
                    "horizon_steps": payload["source"].get("horizon_steps"),
                    "seed": payload["source"].get("seed"),
                    "scenario_id": row.get("scenario_id"),
                    "step_index": row.get("step_index"),
                    "selected_action_index": selected,
                    "best_gain_per_cost_index": best_gpc,
                    "best_new_visible_index": best_visible,
                    "best_gain_per_cost_differs_from_selected": best_gpc != selected,
                    "best_new_visible_differs_from_selected": best_visible != selected,
                    "selected_gain_per_cost": _indexed(gain_per_cost, selected),
                    "best_gain_per_cost": _indexed(gain_per_cost, best_gpc),
                    "selected_new_visible": _indexed(new_visible, selected),
                    "best_new_visible": _indexed(new_visible, best_visible),
                    "selected_probability_rank": prob_ranks.get(selected),
                    "best_gain_per_cost_probability_rank": prob_ranks.get(best_gpc),
                    "best_gain_per_cost_probability": _indexed(probs, best_gpc),
                    "selected_probability": _indexed(probs, selected),
                }
            )
    best_gpc_selected = len([row for row in rows if row["best_gain_per_cost_index"] == row["selected_action_index"]])
    return {
        "schema_version": OPPORTUNITY_AUDIT_SCHEMA_VERSION,
        "candidate_opportunity_row_count": len(rows),
        "candidate_opportunity_fields_missing_count": fields_missing,
        "synthetic_credit_feature_rows_missing_count": synthetic_credit_feature_missing,
        "best_gain_per_cost_selected_count": best_gpc_selected,
        "best_gain_per_cost_differs_from_selected_count": len([row for row in rows if row["best_gain_per_cost_differs_from_selected"]]),
        "best_new_visible_differs_from_selected_count": len([row for row in rows if row["best_new_visible_differs_from_selected"]]),
        "mean_selected_vs_best_gain_per_cost_gap": _mean([row["best_gain_per_cost"] - row["selected_gain_per_cost"] for row in rows]),
        "mean_selected_vs_best_new_visible_gap": _mean([row["best_new_visible"] - row["selected_new_visible"] for row in rows]),
        "best_candidate_never_selected": bool(rows) and best_gpc_selected == 0,
        "sample_rows": rows[:50],
    }


def _coverage_metric_sensitivity_audit(evals: list[dict[str, Any]], *, proxy_range_epsilon: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fields_missing = 0
    for payload in evals:
        for row in payload["pre_inference"]:
            selected = _int(row.get("selected_action_index"))
            gain_per_cost = row.get("theta_coverage_gain_per_path_costs")
            new_visible = row.get("theta_new_visible_cell_counts")
            if not isinstance(gain_per_cost, list) or not isinstance(new_visible, list):
                fields_missing += 1
                continue
            gpc_range = _range(gain_per_cost)
            visible_range = _range(new_visible)
            best_gpc = _max_index(gain_per_cost)
            best_visible = _max_index(new_visible)
            selected_best_diff = (best_gpc is not None and best_gpc != selected) or (best_visible is not None and best_visible != selected)
            rows.append(
                {
                    "job_id": payload["source"].get("job_id"),
                    "horizon_steps": payload["source"].get("horizon_steps"),
                    "seed": payload["source"].get("seed"),
                    "scenario_id": row.get("scenario_id"),
                    "step_index": row.get("step_index"),
                    "gain_per_cost_range": gpc_range,
                    "new_visible_range": visible_range,
                    "candidate_proxy_has_variance": gpc_range > proxy_range_epsilon or visible_range > proxy_range_epsilon,
                    "selected_vs_best_proxy_differs": selected_best_diff,
                }
            )
    variance_rows = [row for row in rows if row["candidate_proxy_has_variance"]]
    diff_rows = [row for row in rows if row["selected_vs_best_proxy_differs"]]
    return {
        "schema_version": SENSITIVITY_AUDIT_SCHEMA_VERSION,
        "coverage_metric_proxy_row_count": len(rows),
        "coverage_metric_proxy_fields_missing_count": fields_missing,
        "candidate_proxy_range_epsilon": proxy_range_epsilon,
        "candidate_proxy_variance_row_count": len(variance_rows),
        "selected_vs_best_proxy_diff_count": len(diff_rows),
        "candidate_proxy_has_material_variance": bool(variance_rows),
        "selected_vs_best_proxy_has_material_difference": bool(diff_rows),
        "counterfactual_trajectory_replay_executed": False,
        "counterfactual_trajectory_replay_recommendation": "run_stage26_8g_counterfactual_action_replay_probe" if diff_rows else "",
        "sample_rows": rows[:50],
    }


def _route(
    *,
    boundary_rejections: list[str],
    input_rejections: list[str],
    source_index: dict[str, Any],
    min_completed_eval_count: int,
    scenario_audit: dict[str, Any],
    margin_audit: dict[str, Any],
    opportunity_audit: dict[str, Any],
    sensitivity_audit: dict[str, Any],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections:
        return ROUTE_INPUTS
    if int(source_index.get("completed_eval_source_count") or 0) < min_completed_eval_count:
        return ROUTE_CONTINUE_JOBS
    if bool(scenario_audit.get("scenario_diversity_suspect")):
        return ROUTE_SCENARIO_DIVERSITY
    if (
        int(margin_audit.get("pre_post_required_field_missing_count") or 0) > 0
        or int(margin_audit.get("margin_required_field_missing_count") or 0) > 0
        or int(margin_audit.get("duplicate_strong_key_count") or 0) > 0
        or int(margin_audit.get("strong_state_join_available_count") or 0) == 0
        or int(margin_audit.get("margin_evaluable_join_count") or 0) == 0
    ):
        return ROUTE_AUDIT_BINDING
    if int(opportunity_audit.get("candidate_opportunity_fields_missing_count") or 0) > 0:
        return ROUTE_CANDIDATE_LOGGING
    if not bool(sensitivity_audit.get("candidate_proxy_has_material_variance")):
        return ROUTE_METRIC_SENSITIVITY
    if bool(opportunity_audit.get("best_candidate_never_selected")):
        return ROUTE_CREDIT
    selected_changed = int(margin_audit.get("selected_action_changed_count") or 0)
    if selected_changed == 0 and bool(margin_audit.get("policy_delta_too_small_for_margin")):
        return ROUTE_POLICY_SIGNAL
    if selected_changed == 0:
        return ROUTE_MARGIN_CROSSING
    if selected_changed > 0 and not bool(sensitivity_audit.get("selected_vs_best_proxy_has_material_difference")):
        return ROUTE_COUNTERFACTUAL
    return ROUTE_CONTINUE_JOBS


def _input_rejections(stage26_8d_summary: dict[str, Any]) -> list[str]:
    if not stage26_8d_summary:
        return ["missing_stage26_8d_summary"]
    reasons: list[str] = []
    if stage26_8d_summary.get("schema_version") != "xunce-stage26-8d-summary/v1":
        reasons.append("stage26_8d_schema_version_mismatch")
    if stage26_8d_summary.get("stage_id") != "xunce-stage26-8d-resumable-seed-horizon-execution":
        reasons.append("stage26_8d_stage_id_mismatch")
    if stage26_8d_summary.get("status") != "passed":
        reasons.append("stage26_8d_status_not_passed")
    if stage26_8d_summary.get("next_required_change") not in {
        "continue_stage26_8d_seed_horizon_jobs",
        "repair_stage26_synthetic_policy_update_signal_strength",
        "run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot",
    }:
        reasons.append("stage26_8d_route_not_accepted")
    if stage26_8d_summary.get("coverage_denominator_source") != COVERAGE_DENOMINATOR_SOURCE:
        reasons.append("stage26_8d_coverage_denominator_source_mismatch")
    if stage26_8d_summary.get("coverage_source") != COVERAGE_SOURCE:
        reasons.append("stage26_8d_coverage_source_mismatch")
    if stage26_8d_summary.get("path_cost_source") != PATH_COST_SOURCE:
        reasons.append("stage26_8d_path_cost_source_mismatch")
    if stage26_8d_summary.get("synthetic_source_kind") != SYNTHETIC_SOURCE_KIND:
        reasons.append("stage26_8d_synthetic_source_kind_mismatch")
    if stage26_8d_summary.get("action_space_type") != ACTION_SPACE_TYPE:
        reasons.append("stage26_8d_action_space_type_mismatch")
    for field in BOUNDARY_FIELDS:
        if stage26_8d_summary.get(field) is True:
            reasons.append(f"stage26_8d_{field}")
    if _float(stage26_8d_summary.get("canary_traffic_fraction")) > 0.0:
        reasons.append("stage26_8d_canary_traffic_fraction")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    if _float(config.get("canary_traffic_fraction")) > 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _diagnostic_conclusion(route: str) -> str:
    return {
        ROUTE_INPUTS: "Stage26.8D 输入不可信，无法诊断。",
        ROUTE_CONTINUE_JOBS: "已完成 eval 数不足或审计建议继续可恢复队列。",
        ROUTE_AUDIT_BINDING: "pre/post 强绑定字段或 join 不完整。",
        ROUTE_SCENARIO_DIVERSITY: "scenario/candidate/action/episode signature 存在大量重复，优先修 scenario 多样性。",
        ROUTE_CANDIDATE_LOGGING: "候选级机会字段缺失，无法判断 best candidate。",
        ROUTE_METRIC_SENSITIVITY: "候选覆盖效率 proxy 本身缺少差异，优先修 candidate generation 或 metric sensitivity。",
        ROUTE_POLICY_SIGNAL: "pre/post action 未变且 policy delta 远小于 argmax margin，优先修 policy update signal strength。",
        ROUTE_MARGIN_CROSSING: "pre/post action 未变但已接近边界，优先校准 discrete margin crossing。",
        ROUTE_CREDIT: "best candidate 没有成为 selected action，优先修 exploration credit assignment。",
        ROUTE_COUNTERFACTUAL: "动作变化但 proxy/metric 不分辨，建议做 counterfactual action replay。",
        ROUTE_BOUNDARY: "release/default-policy/executor/canary 边界被打开。",
    }[route]


def _strong_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    values = []
    for field in STRONG_JOIN_FIELDS:
        value = row.get(field)
        if value in (None, ""):
            return None
        values.append(_canonical(value))
    return tuple(values)


def _top_two(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"top1_index": None, "top2_index": None, "top1_value": 0.0, "top2_value": 0.0}
    ranked = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else (-1, 0.0)
    return {"top1_index": top1[0], "top2_index": top2[0], "top1_value": top1[1], "top2_value": top2[1]}


def _probability_ranks(values: list[float]) -> dict[int, int]:
    return {index: rank + 1 for rank, (index, _value) in enumerate(sorted(enumerate(values), key=lambda item: item[1], reverse=True))}


def _max_index(values: list[Any]) -> int | None:
    if not values:
        return None
    parsed = [_float(value) for value in values]
    return max(range(len(parsed)), key=lambda index: parsed[index])


def _indexed(values: list[Any], index: int | None) -> float:
    if index is None or index < 0 or index >= len(values):
        return 0.0
    return _float(values[index])


def _range(values: list[Any]) -> float:
    parsed = [_float(value) for value in values]
    if not parsed:
        return 0.0
    return max(parsed) - min(parsed)


def _duplicate_pair_count(values: list[str]) -> int:
    total = 0
    for count in Counter(values).values():
        if count > 1:
            total += count * (count - 1) // 2
    return total


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _empty_scenario_audit() -> dict[str, Any]:
    return {
        "schema_version": SCENARIO_AUDIT_SCHEMA_VERSION,
        "scenario_signature_count": 0,
        "scenario_signatures": [],
        "identical_candidate_sequence_pair_count": 0,
        "identical_selected_action_sequence_pair_count": 0,
        "identical_episode_metric_pair_count": 0,
        "scenario_signature_duplicate_group_count": 0,
        "scenario_signature_duplicate_row_count": 0,
        "scenario_signature_duplicate_ratio": 0.0,
        "scenario_diversity_suspect": False,
    }


def _empty_margin_audit() -> dict[str, Any]:
    return {
        "schema_version": MARGIN_AUDIT_SCHEMA_VERSION,
        "strong_state_join_available_count": 0,
        "margin_evaluable_join_count": 0,
        "pre_post_required_field_missing_count": 0,
        "margin_required_field_missing_count": 0,
        "margin_required_field_missing_reasons": {},
        "duplicate_strong_key_count": 0,
        "unmatched_pre_row_count": 0,
        "unmatched_post_row_count": 0,
        "selected_action_changed_count": 0,
        "mean_abs_probability_delta": 0.0,
        "mean_top1_probability_margin": 0.0,
        "mean_margin_closure_rate": 0.0,
        "near_boundary_row_count": 0,
        "policy_delta_too_small_for_margin": False,
        "sample_rows": [],
    }


def _empty_opportunity_audit() -> dict[str, Any]:
    return {
        "schema_version": OPPORTUNITY_AUDIT_SCHEMA_VERSION,
        "candidate_opportunity_row_count": 0,
        "candidate_opportunity_fields_missing_count": 0,
        "synthetic_credit_feature_rows_missing_count": 0,
        "best_gain_per_cost_selected_count": 0,
        "best_gain_per_cost_differs_from_selected_count": 0,
        "best_candidate_never_selected": False,
        "sample_rows": [],
    }


def _empty_sensitivity_audit() -> dict[str, Any]:
    return {
        "schema_version": SENSITIVITY_AUDIT_SCHEMA_VERSION,
        "coverage_metric_proxy_row_count": 0,
        "coverage_metric_proxy_fields_missing_count": 0,
        "candidate_proxy_variance_row_count": 0,
        "selected_vs_best_proxy_diff_count": 0,
        "candidate_proxy_has_material_variance": False,
        "selected_vs_best_proxy_has_material_difference": False,
        "counterfactual_trajectory_replay_executed": False,
        "sample_rows": [],
    }


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    defaults = {
        "stage26_8d_root": DEFAULT_STAGE26_8D_ROOT,
        "min_completed_eval_count": 2,
        "scenario_duplicate_ratio_threshold": 0.50,
        "near_boundary_probability_margin_threshold": 5.0e-5,
        "margin_delta_ratio_for_too_small": 0.25,
        "candidate_proxy_range_epsilon": 1.0e-9,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    config["stage26_8d_root"] = str(_resolve_path(Path(str(config["stage26_8d_root"])), repo_root))
    if int(config["min_completed_eval_count"]) < 1:
        raise ValueError("min_completed_eval_count must be >= 1")
    return config


def _read_json(path: Path) -> dict[str, Any]:
    return artifact_io.read_json(path)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not artifact_io.path_is_file(path):
        return {}
    return _read_json(path)


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return artifact_io.read_jsonl(path) if artifact_io.path_is_file(path) else []


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    artifact_io.write_json(path, payload)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isfinite(parsed):
        return parsed
    return 0.0


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[Any]) -> float:
    parsed = [_float(value) for value in values]
    if not parsed:
        return 0.0
    return sum(parsed) / len(parsed)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _render_report(
    summary: dict[str, Any],
    scenario_audit: dict[str, Any],
    margin_audit: dict[str, Any],
    opportunity_audit: dict[str, Any],
    sensitivity_audit: dict[str, Any],
) -> str:
    lines = [
        "# Stage26.8F Scenario Diversity And Policy Margin Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- diagnostic_conclusion: `{summary['diagnostic_conclusion']}`",
        f"- completed_eval_source_count: `{summary['completed_eval_source_count']}`",
        f"- scenario_diversity_suspect: `{scenario_audit.get('scenario_diversity_suspect')}`",
        f"- duplicate_signature_groups: `{scenario_audit.get('scenario_signature_duplicate_group_count')}`",
        f"- strong_state_join_available_count: `{margin_audit.get('strong_state_join_available_count')}`",
        f"- selected_action_changed_count: `{margin_audit.get('selected_action_changed_count')}`",
        f"- mean_abs_probability_delta: `{margin_audit.get('mean_abs_probability_delta')}`",
        f"- mean_top1_probability_margin: `{margin_audit.get('mean_top1_probability_margin')}`",
        f"- policy_delta_too_small_for_margin: `{margin_audit.get('policy_delta_too_small_for_margin')}`",
        f"- best_gain_per_cost_selected_count: `{opportunity_audit.get('best_gain_per_cost_selected_count')}`",
        f"- best_gain_per_cost_differs_from_selected_count: `{opportunity_audit.get('best_gain_per_cost_differs_from_selected_count')}`",
        f"- synthetic_credit_feature_rows_missing_count: `{opportunity_audit.get('synthetic_credit_feature_rows_missing_count')}`",
        f"- candidate_proxy_variance_row_count: `{sensitivity_audit.get('candidate_proxy_variance_row_count')}`",
        f"- selected_vs_best_proxy_diff_count: `{sensitivity_audit.get('selected_vs_best_proxy_diff_count')}`",
        "",
        "This stage is read-only. It does not run collector, PPO update, trajectory eval, checkpoint publication, executor connection, or canary traffic.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
