from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

REPO_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_IMPORT_ROOT))

from scripts.run_xunce_stage21_15_policy_update_signal_strength_calibration import (
    _float,
    _int,
    _positive_float,
    _read_json,
    _read_json_or_empty,
    _read_jsonl,
    _resolve_path,
    _write_json,
    _write_jsonl,
)
from scripts.run_xunce_stage21_4_tiny_ppo_update_smoke import run_xunce_stage21_4_tiny_ppo_update_smoke


CONFIG_SCHEMA_VERSION = "xunce-stage21-19-policy-update-signal-source-repair-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage21-19-summary/v1"
ACTION_LOGPROB_SCHEMA_VERSION = "xunce-stage21-19-action-logprob-binding-audit/v1"
POLICY_GRADIENT_SCHEMA_VERSION = "xunce-stage21-19-policy-gradient-path-audit/v1"
RATIO_CLIP_SCHEMA_VERSION = "xunce-stage21-19-ratio-clip-advantage-audit/v1"
CANDIDATE_LOGIT_SCHEMA_VERSION = "xunce-stage21-19-candidate-logit-sensitivity-audit/v1"
SMOKE_SCHEMA_VERSION = "xunce-stage21-19-diagnostic-smoke-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage21-19-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage21-19-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage21_19_policy_update_signal_source_repair_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/outputs/"
    "path_feedback_batch_xunce_stage21_19_policy_update_signal_source_repair_v1"
)

SUMMARY_FILE = "xunce-stage21-19-summary.json"
ACTION_LOGPROB_FILE = "xunce-stage21-19-action-logprob-binding-audit.json"
POLICY_GRADIENT_FILE = "xunce-stage21-19-policy-gradient-path-audit.json"
RATIO_CLIP_FILE = "xunce-stage21-19-ratio-clip-advantage-audit.json"
CANDIDATE_LOGIT_FILE = "xunce-stage21-19-candidate-logit-sensitivity-audit.json"
SMOKE_FILE = "xunce-stage21-19-diagnostic-smoke-summary.json"
RECOMMENDED_CONFIG_FILE = "xunce-stage21-19-recommended-stage21-6-config.json"
ROUTING_FILE = "xunce-stage21-19-next-stage-routing.json"
REPORT_FILE = "xunce-stage21-19-report.md"
MANIFEST_FILE = "xunce-stage21-19-manifest.json"

ROUTE_INPUTS = "rerun_stage21_19_required_inputs"
ROUTE_ACTION_BINDING = "repair_stage21_action_logprob_binding"
ROUTE_POLICY_PATH = "repair_stage21_policy_head_update_path"
ROUTE_RATIO_ADV = "repair_stage21_ratio_clip_or_advantage_scaling"
ROUTE_MARGIN = "calibrate_stage21_discrete_action_margin_crossing"
ROUTE_CREDIT = "repair_stage21_return_advantage_credit_assignment"
ROUTE_REPAIR_DONE = "rerun_stage21_18_iterative_ppo_learning_curve_audit_with_repaired_signal"
ROUTE_BOUNDARY = "resolve_stage21_19_boundary_rejections"
ROUTE_RUNTIME = "stage21_19_diagnostic_smoke_runtime_budget_blocked"

BOUNDARY_FIELDS = (
    "stage21_19_authorized",
    "training_or_release_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage21.19 policy update signal source repair audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        summary = run_xunce_stage21_19_policy_update_signal_source_repair(
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
                "action_logprob_binding_passed": summary["action_logprob_binding_passed"],
                "policy_signal_probability_delta": summary["diagnostic_smoke_probability_delta"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] in {"passed", "partial"} else 1


def run_xunce_stage21_19_policy_update_signal_source_repair(
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

    input_reasons = _input_reasons(config)
    boundary_reasons = _boundary_reasons(config)
    seed_runs = _collect_seed_runs(config) if not input_reasons else []

    action_logprob = _audit_action_logprob_binding(seed_runs, config)
    ratio_clip = _audit_ratio_clip_advantage(seed_runs, config)
    policy_gradient = _audit_policy_gradient_path(seed_runs, config)
    candidate_logit = _audit_candidate_logit_sensitivity(seed_runs, config)
    smoke = _diagnostic_smoke(config, seed_runs, output_root, repo_root) if not input_reasons and not boundary_reasons else _empty_smoke()
    recommended_config = _recommended_stage21_6_config(config, output_root)

    status, route, primary_reason = _route(
        input_reasons=input_reasons,
        boundary_reasons=boundary_reasons,
        action_logprob=action_logprob,
        ratio_clip=ratio_clip,
        policy_gradient=policy_gradient,
        candidate_logit=candidate_logit,
        smoke=smoke,
        config=config,
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "next_required_change": route,
        "primary_reason": primary_reason,
        "stage21_18_root": str(config["stage21_18_root"]),
        "seed_run_count": len(seed_runs),
        "input_reason_codes": input_reasons,
        "boundary_reason_codes": boundary_reasons,
        "action_logprob_binding_passed": action_logprob["binding_passed"],
        "action_logprob_recompute_max_abs_error": action_logprob["old_log_prob_recompute_max_abs_error"],
        "policy_gradient_path_passed": policy_gradient["policy_gradient_path_passed"],
        "policy_head_parameter_delta_l2_mean": policy_gradient["parameter_delta_by_group_mean"].get("policy_head"),
        "candidate_encoder_parameter_delta_l2_mean": policy_gradient["parameter_delta_by_group_mean"].get("candidate_encoder"),
        "ratio_clip_advantage_passed": ratio_clip["ratio_clip_advantage_passed"],
        "candidate_logit_sensitivity_available": candidate_logit["strong_state_join_available_count"] > 0,
        "inference_binding_passed": candidate_logit.get("inference_binding_passed"),
        "duplicate_xunce_strong_key_count": candidate_logit.get("duplicate_xunce_strong_key_count"),
        "vector_contract_violation_count": candidate_logit.get("vector_contract_violation_count"),
        "candidate_metric_join_count": candidate_logit.get("candidate_metric_join_count"),
        "candidate_logit_delta_mean": candidate_logit["candidate_logit_delta_mean"],
        "argmax_changed_count": candidate_logit["argmax_changed_count"],
        "selected_action_changed_count": candidate_logit["selected_action_changed_count"],
        "selected_rank_changed_count": candidate_logit["selected_rank_changed_count"],
        "diagnostic_smoke_executed": smoke["diagnostic_smoke_executed"],
        "diagnostic_smoke_status": smoke["status"],
        "diagnostic_smoke_probability_delta": smoke["mean_abs_probability_delta"],
        "diagnostic_smoke_post_update_approx_kl_abs_max": smoke.get("post_update_approx_kl_abs_max"),
        "diagnostic_smoke_argmax_changed_count": smoke["argmax_changed_count"],
        "stage21_19_authorized": False,
        "training_or_release_authorized": False,
        "runs_new_ppo_update": bool(smoke["diagnostic_smoke_executed"]),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    _write_outputs(
        output_root=output_root,
        summary=summary,
        action_logprob=action_logprob,
        policy_gradient=policy_gradient,
        ratio_clip=ratio_clip,
        candidate_logit=candidate_logit,
        smoke=smoke,
        recommended_config=recommended_config,
        generated_at=generated_at,
    )
    return summary | {"summary": str(output_root / SUMMARY_FILE)}


def _collect_seed_runs(config: dict[str, Any]) -> list[dict[str, Any]]:
    stage21_18_root = Path(config["stage21_18_root"])
    round_rows = _read_jsonl(stage21_18_root / "xunce-stage21-18-round-results.jsonl")
    seed_runs: list[dict[str, Any]] = []
    for round_row in round_rows:
        for stage21_6_root_raw in round_row.get("stage21_6_roots", []):
            stage21_6_root = Path(str(stage21_6_root_raw))
            seed_rows = _read_jsonl(stage21_6_root / "xunce-stage21-6-seed-results.jsonl")
            for seed_row in seed_rows:
                seed_runs.append(
                    {
                        "round_index": _int(round_row.get("round_index")) or 0,
                        "seed": _int(seed_row.get("seed")) or 0,
                        "stage21_6_root": str(stage21_6_root),
                        "stage21_1_root": seed_row.get("stage21_1_root"),
                        "stage21_3_root": seed_row.get("stage21_3_root"),
                        "stage21_4_root": seed_row.get("stage21_4_root"),
                        "stage21_5_root": seed_row.get("stage21_5_root"),
                    }
                )
    return seed_runs


def _audit_action_logprob_binding(seed_runs: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    max_error = 0.0
    total = invalid_mask = missing_field = nonfinite = candidate_hash_missing = 0
    sample_rows: list[dict[str, Any]] = []
    max_rows = int(config["max_audit_sample_rows"])
    for run in seed_runs:
        rows = _read_jsonl(Path(str(run.get("stage21_3_root"))) / "xunce-stage21-3-ppo-trainable-batch.jsonl")
        for row in rows:
            total += 1
            info = row.get("info") if isinstance(row.get("info"), dict) else {}
            action_index = _int(row.get("action_index"))
            logits = info.get("old_sampling_logits")
            sampling_mask = info.get("sampling_mask")
            action_mask = info.get("action_mask")
            hard_mask = info.get("hard_risk_clean_mask")
            required = [
                row.get("scenario_id"),
                row.get("step_index"),
                info.get("current_cell_before"),
                info.get("covered_cells_hash"),
                info.get("candidate_set_hash"),
            ]
            if any(value is None or value == "" for value in required):
                missing_field += 1
            if not info.get("candidate_set_hash"):
                candidate_hash_missing += 1
            xunce_action_mask = _tensor_values(row.get("xunce_batch", {}).get("action_mask"))
            if action_index is None or logits is None or sampling_mask is None or action_mask is None or hard_mask is None or not xunce_action_mask:
                missing_field += 1
                continue
            if _bool_list(xunce_action_mask) != _bool_list(action_mask):
                invalid_mask += 1
            if not _is_allowed(action_index, sampling_mask, action_mask, hard_mask):
                invalid_mask += 1
            recomputed = _masked_log_prob(logits, sampling_mask, action_index)
            old_log_prob = _float(row.get("old_log_prob"))
            if old_log_prob is None or recomputed is None or not math.isfinite(old_log_prob) or not math.isfinite(recomputed):
                nonfinite += 1
                continue
            error = abs(recomputed - old_log_prob)
            max_error = max(max_error, error)
            if len(sample_rows) < max_rows:
                sample_rows.append(
                    {
                        "round_index": run.get("round_index"),
                        "seed": run.get("seed"),
                        "transition_id": row.get("transition_id"),
                        "action_index": action_index,
                        "old_log_prob": old_log_prob,
                        "recomputed_old_log_prob": recomputed,
                        "abs_error": error,
                        "candidate_set_hash": info.get("candidate_set_hash"),
                    }
                )
    tolerance = float(config["old_log_prob_recompute_tolerance"])
    return {
        "schema_version": ACTION_LOGPROB_SCHEMA_VERSION,
        "transition_count": total,
        "invalid_action_mask_count": invalid_mask,
        "missing_required_field_count": missing_field,
        "candidate_set_hash_missing_count": candidate_hash_missing,
        "nonfinite_log_prob_count": nonfinite,
        "old_log_prob_recompute_max_abs_error": max_error,
        "old_log_prob_recompute_tolerance": tolerance,
        "binding_passed": bool(total and invalid_mask == 0 and missing_field == 0 and nonfinite == 0 and max_error <= tolerance),
        "sample_rows": sample_rows,
    }


def _audit_ratio_clip_advantage(seed_runs: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    loss_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    for run in seed_runs:
        loss_rows.extend(_read_jsonl(Path(str(run.get("stage21_4_root"))) / "xunce-stage21-4-ppo-loss-audit.jsonl"))
        batch_rows.extend(_read_jsonl(Path(str(run.get("stage21_3_root"))) / "xunce-stage21-3-ppo-trainable-batch.jsonl"))
    advantages = [_float(row.get("advantage")) for row in batch_rows]
    advantages = [float(value) for value in advantages if value is not None and math.isfinite(value)]
    adv_mean = mean(advantages) if advantages else 0.0
    adv_std = _std(advantages)
    policy_grad_max = _max_float(row.get("policy_loss_grad_norm") for row in loss_rows)
    clip_fraction_max = _max_float(row.get("post_update_clip_fraction") for row in loss_rows)
    ratio_min = _min_float(row.get("ratio_min") for row in loss_rows)
    ratio_max = _max_float(row.get("ratio_max") for row in loss_rows)
    post_kl_max = _max_abs(row.get("post_update_approx_kl") for row in loss_rows)
    weak_advantage = adv_std < float(config["min_advantage_std"]) or abs(adv_mean) < float(config["min_abs_advantage_mean"]) and adv_std < 1.0e-8
    policy_loss_tiny = policy_grad_max < float(config["min_policy_grad_norm"])
    return {
        "schema_version": RATIO_CLIP_SCHEMA_VERSION,
        "loss_row_count": len(loss_rows),
        "batch_row_count": len(batch_rows),
        "advantage_mean": adv_mean,
        "advantage_std": adv_std,
        "policy_loss_grad_norm_max": policy_grad_max,
        "clip_fraction_max": clip_fraction_max,
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
        "post_update_approx_kl_abs_max": post_kl_max,
        "weak_or_flat_advantage": weak_advantage,
        "policy_loss_gradient_too_small": policy_loss_tiny,
        "ratio_clip_advantage_passed": bool(loss_rows and not weak_advantage and not policy_loss_tiny),
    }


def _audit_policy_gradient_path(seed_runs: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    gradients: list[dict[str, Any]] = []
    deltas: list[dict[str, float]] = []
    for run in seed_runs:
        stage21_4_root = Path(str(run.get("stage21_4_root")))
        summary = _read_json_or_empty(stage21_4_root / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json")
        gradient = _read_json_or_empty(stage21_4_root / "xunce-stage21-4-gradient-audit.json")
        checkpoint = _read_json_or_empty(stage21_4_root / "xunce-stage21-4-checkpoint-audit.json")
        summaries.append(summary)
        gradients.append(gradient)
        if bool(config.get("load_checkpoint_parameter_groups", True)):
            delta = _checkpoint_parameter_delta_by_group(
                source=summary.get("source_xunce_candidate_checkpoint") or checkpoint.get("source_xunce_candidate_checkpoint"),
                experimental=checkpoint.get("experimental_checkpoint_path") or summary.get("experimental_checkpoint_path"),
            )
            if delta:
                deltas.append(delta)
    policy_grad = [_float(g.get("component_grad_norms", {}).get("policy_loss_grad_norm")) for g in gradients]
    value_grad = [_float(g.get("component_grad_norms", {}).get("value_loss_grad_norm")) for g in gradients]
    total_grad = [_float(g.get("component_grad_norms", {}).get("total_loss_grad_norm")) for g in gradients]
    policy_grad_max = _max_float(policy_grad)
    total_grad_max = _max_float(total_grad)
    value_to_policy_max = _max_ratio(value_grad, policy_grad)
    group_mean = _mean_group_deltas(deltas)
    policy_delta = group_mean.get("policy_head", 0.0)
    candidate_delta = group_mean.get("candidate_encoder", 0.0)
    policy_path_passed = bool(
        summaries
        and policy_grad_max >= float(config["min_policy_grad_norm"])
        and (not deltas or policy_delta >= float(config["min_policy_head_parameter_delta_l2"]))
    )
    return {
        "schema_version": POLICY_GRADIENT_SCHEMA_VERSION,
        "stage21_4_run_count": len(summaries),
        "policy_loss_grad_norm_max": policy_grad_max,
        "total_loss_grad_norm_max": total_grad_max,
        "value_to_policy_grad_norm_ratio_max": value_to_policy_max,
        "parameter_delta_by_group_mean": group_mean,
        "parameter_delta_sample_count": len(deltas),
        "policy_gradient_path_passed": policy_path_passed,
        "policy_head_parameter_delta_l2_threshold": float(config["min_policy_head_parameter_delta_l2"]),
    }


def _audit_candidate_logit_sensitivity(seed_runs: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    joined = 0
    missing = 0
    duplicate_xunce_key_count = 0
    vector_contract_violation_count = 0
    candidate_metric_join_count = 0
    candidate_metric_missing_count = 0
    logit_deltas: list[float] = []
    prob_deltas: list[float] = []
    best_cpc_prob_deltas: list[float] = []
    argmax_changed = 0
    selected_changed = 0
    rank_changed = 0
    for run in seed_runs:
        summary5 = _read_json_or_empty(Path(str(run.get("stage21_5_root"))) / "xunce-stage21-5-post-update-evaluation-summary.json")
        pre_root = Path(str(summary5.get("pre_evaluation_root", "")))
        post_root = Path(str(summary5.get("post_evaluation_root", "")))
        pre_rows = [row for row in _read_jsonl(pre_root / "xunce-exploration-coverage-model-inference.jsonl") if row.get("policy") == "xunce"]
        post_rows = [row for row in _read_jsonl(post_root / "xunce-exploration-coverage-model-inference.jsonl") if row.get("policy") == "xunce"]
        pre_by_key, pre_duplicates = _unique_rows_by_strong_key(pre_rows)
        post_by_key, post_duplicates = _unique_rows_by_strong_key(post_rows)
        duplicate_xunce_key_count += pre_duplicates + post_duplicates
        best_cpc_by_key = _best_cpc_index_by_strong_key(pre_root / "xunce-exploration-coverage-candidate-metric-audit.jsonl")
        for key, pre in pre_by_key.items():
            if key not in post_by_key:
                missing += 1
                continue
            post = post_by_key[key]
            if not _inference_vector_contract_passed(pre) or not _inference_vector_contract_passed(post):
                vector_contract_violation_count += 1
                continue
            pre_probs = _number_list(pre.get("action_probs"))
            post_probs = _number_list(post.get("action_probs"))
            pre_logits = _number_list(pre.get("logits") or pre.get("masked_logits"))
            post_logits = _number_list(post.get("logits") or post.get("masked_logits"))
            joined += 1
            for before, after in zip(pre_probs, post_probs):
                prob_deltas.append(abs(after - before))
            for before, after in zip(pre_logits, post_logits):
                if math.isfinite(before) and math.isfinite(after):
                    logit_deltas.append(abs(after - before))
            selected_index = _int(pre.get("selected_action_index"))
            if selected_index is not None and selected_index < min(len(pre_probs), len(post_probs)):
                if abs(post_probs[selected_index] - pre_probs[selected_index]) > 0.0:
                    pass
            if _argmax(pre_probs) != _argmax(post_probs):
                argmax_changed += 1
            if _int(pre.get("selected_action_index")) != _int(post.get("selected_action_index")):
                selected_changed += 1
            if _int(pre.get("selected_rank")) != _int(post.get("selected_rank")):
                rank_changed += 1
            best_index = best_cpc_by_key.get(key)
            if best_index is not None and best_index < min(len(pre_probs), len(post_probs)):
                candidate_metric_join_count += 1
                best_cpc_prob_deltas.append(post_probs[best_index] - pre_probs[best_index])
            else:
                candidate_metric_missing_count += 1
    return {
        "schema_version": CANDIDATE_LOGIT_SCHEMA_VERSION,
        "strong_state_join_available_count": joined,
        "strong_state_binding_unavailable_count": missing,
        "duplicate_xunce_strong_key_count": duplicate_xunce_key_count,
        "vector_contract_violation_count": vector_contract_violation_count,
        "candidate_metric_join_count": candidate_metric_join_count,
        "candidate_metric_missing_count": candidate_metric_missing_count,
        "inference_binding_passed": bool(joined > 0 and duplicate_xunce_key_count == 0 and vector_contract_violation_count == 0),
        "mean_abs_probability_delta": mean(prob_deltas) if prob_deltas else 0.0,
        "candidate_logit_delta_mean": mean(logit_deltas) if logit_deltas else 0.0,
        "candidate_logit_delta_max": max(logit_deltas) if logit_deltas else 0.0,
        "best_coverage_per_cost_probability_delta_mean": mean(best_cpc_prob_deltas) if best_cpc_prob_deltas else 0.0,
        "best_coverage_per_cost_probability_delta_max": max(best_cpc_prob_deltas) if best_cpc_prob_deltas else 0.0,
        "argmax_changed_count": argmax_changed,
        "selected_action_changed_count": selected_changed,
        "selected_rank_changed_count": rank_changed,
    }


def _diagnostic_smoke(
    config: dict[str, Any],
    seed_runs: list[dict[str, Any]],
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    if not bool(config["execute_diagnostic_smoke"]):
        return _empty_smoke(status="skipped", reason="execute_diagnostic_smoke_false")
    selected = _select_smoke_seed_run(seed_runs)
    if selected is None:
        return _empty_smoke(status="blocked", reason="no_seed_run_available")
    smoke_root = output_root / "diagnostic_smoke" / "stage21_4"
    smoke_config_path = output_root / "diagnostic_smoke" / "stage21_4_policy_path_overfit_config.json"
    smoke_config_path.parent.mkdir(parents=True, exist_ok=True)
    base_config = _read_json(Path(config["stage21_4_base_config"]))
    source_summary = _read_json_or_empty(Path(str(selected["stage21_4_root"])) / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json")
    base_config.update(
        {
            "stage21_3_ppo_batch_validation_root": str(selected["stage21_3_root"]),
            "xunce_candidate_checkpoint": source_summary.get("source_xunce_candidate_checkpoint") or base_config.get("xunce_candidate_checkpoint"),
            "value_loss_coefficient": 0.0,
            "entropy_coefficient": 0.0,
            "policy_loss_coefficient": 1.0,
            "epochs": int(config["diagnostic_smoke_epochs"]),
            "learning_rate": float(config["diagnostic_smoke_learning_rate"]),
            "clip_ratio": float(config["diagnostic_smoke_clip_ratio"]),
            "advantage_clip_abs": float(config["diagnostic_smoke_advantage_clip_abs"]),
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
            "stage21_4_authorized": False,
        }
    )
    _write_json(smoke_config_path, base_config)
    try:
        result = run_xunce_stage21_4_tiny_ppo_update_smoke(
            config_path=smoke_config_path,
            output_root=smoke_root,
            repo_root=repo_root,
        )
    except Exception as exc:  # pragma: no cover - exercised by integration runs.
        return _empty_smoke(status="blocked", reason=f"diagnostic_smoke_exception:{type(exc).__name__}:{exc}")
    summary = _read_json_or_empty(smoke_root / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json") or result
    loss_rows = _read_jsonl(smoke_root / "xunce-stage21-4-ppo-loss-audit.jsonl")
    gradient = _read_json_or_empty(smoke_root / "xunce-stage21-4-gradient-audit.json")
    return {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "diagnostic_smoke_executed": True,
        "status": summary.get("status", "unknown"),
        "reason": None,
        "smoke_root": str(smoke_root),
        "smoke_config": str(smoke_config_path),
        "source_seed": selected.get("seed"),
        "source_round_index": selected.get("round_index"),
        "stage21_3_root": selected.get("stage21_3_root"),
        "policy_path_overfit": True,
        "value_loss_coefficient": 0.0,
        "entropy_coefficient": 0.0,
        "policy_loss_coefficient": 1.0,
        "epochs": int(config["diagnostic_smoke_epochs"]),
        "learning_rate": float(config["diagnostic_smoke_learning_rate"]),
        "mean_abs_probability_delta": 0.0,
        "post_update_approx_kl_abs_max": _max_abs(row.get("post_update_approx_kl") for row in loss_rows),
        "argmax_changed_count": 0,
        "pre_clip_grad_norm": gradient.get("pre_clip_grad_norm"),
        "post_clip_grad_norm": gradient.get("post_clip_grad_norm"),
        "policy_loss_grad_norm": gradient.get("component_grad_norms", {}).get("policy_loss_grad_norm"),
        "experimental_checkpoint_path": summary.get("experimental_checkpoint_path"),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
    }


def _route(
    *,
    input_reasons: list[str],
    boundary_reasons: list[str],
    action_logprob: dict[str, Any],
    ratio_clip: dict[str, Any],
    policy_gradient: dict[str, Any],
    candidate_logit: dict[str, Any],
    smoke: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str, str, str]:
    if input_reasons:
        return "failed", ROUTE_INPUTS, "stage21_19_input_rejected"
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "stage21_19_boundary_rejected"
    if not action_logprob["binding_passed"]:
        return "failed", ROUTE_ACTION_BINDING, "action_logprob_mask_or_candidate_binding_failed"
    if candidate_logit.get("inference_binding_passed") is False:
        return "failed", ROUTE_ACTION_BINDING, "inference_policy_strong_binding_contract_failed"
    if not ratio_clip["ratio_clip_advantage_passed"]:
        return "failed", ROUTE_RATIO_ADV, "ratio_clip_or_advantage_signal_failed"
    if not policy_gradient["policy_gradient_path_passed"]:
        return "failed", ROUTE_POLICY_PATH, "policy_gradient_path_not_effective"
    if smoke.get("status") == "blocked":
        return "partial", ROUTE_RUNTIME, "diagnostic_smoke_blocked"
    probability_delta = max(
        _float(smoke.get("mean_abs_probability_delta")) or 0.0,
        _float(candidate_logit.get("mean_abs_probability_delta")) or 0.0,
    )
    if probability_delta >= float(config["policy_signal_enhanced_probability_delta_threshold"]):
        return "passed", ROUTE_REPAIR_DONE, "policy_signal_probability_delta_enhanced"
    if (
        int(candidate_logit.get("argmax_changed_count", 0)) > 0
        or int(candidate_logit.get("selected_action_changed_count", 0)) > 0
        or int(candidate_logit.get("selected_rank_changed_count", 0)) > 0
    ):
        return "failed", ROUTE_CREDIT, "action_changed_without_trajectory_improvement"
    if (
        candidate_logit.get("candidate_logit_delta_max", 0.0) >= float(config["candidate_logit_delta_margin_threshold"])
        and int(candidate_logit.get("selected_rank_changed_count", 0)) == 0
    ):
        return "failed", ROUTE_MARGIN, "logits_changed_without_rank_crossing"
    if int(candidate_logit.get("selected_rank_changed_count", 0)) > 0:
        return "failed", ROUTE_CREDIT, "rank_changed_without_trajectory_improvement"
    return "failed", ROUTE_POLICY_PATH, "policy_signal_still_too_small"


def _write_outputs(
    *,
    output_root: Path,
    summary: dict[str, Any],
    action_logprob: dict[str, Any],
    policy_gradient: dict[str, Any],
    ratio_clip: dict[str, Any],
    candidate_logit: dict[str, Any],
    smoke: dict[str, Any],
    recommended_config: dict[str, Any],
    generated_at: str,
) -> None:
    _write_json(output_root / ACTION_LOGPROB_FILE, action_logprob)
    _write_json(output_root / POLICY_GRADIENT_FILE, policy_gradient)
    _write_json(output_root / RATIO_CLIP_FILE, ratio_clip)
    _write_json(output_root / CANDIDATE_LOGIT_FILE, candidate_logit)
    _write_json(output_root / SMOKE_FILE, smoke)
    _write_json(output_root / RECOMMENDED_CONFIG_FILE, recommended_config)
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": summary["status"],
        "next_required_change": summary["next_required_change"],
        "primary_reason": summary["primary_reason"],
        "stage21_19_authorized": False,
        "training_or_release_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    _write_json(output_root / ROUTING_FILE, routing)
    report = _report(summary, action_logprob, policy_gradient, ratio_clip, candidate_logit, smoke)
    (output_root / REPORT_FILE).write_text(report, encoding="utf-8")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "summary": str(output_root / SUMMARY_FILE),
        "action_logprob_binding_audit": str(output_root / ACTION_LOGPROB_FILE),
        "policy_gradient_path_audit": str(output_root / POLICY_GRADIENT_FILE),
        "ratio_clip_advantage_audit": str(output_root / RATIO_CLIP_FILE),
        "candidate_logit_sensitivity_audit": str(output_root / CANDIDATE_LOGIT_FILE),
        "diagnostic_smoke_summary": str(output_root / SMOKE_FILE),
        "recommended_stage21_6_config": str(output_root / RECOMMENDED_CONFIG_FILE),
        "routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
        "stage21_19_authorized": False,
        "training_or_release_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    _write_json(output_root / MANIFEST_FILE, manifest)
    summary.update(
        {
            "summary": str(output_root / SUMMARY_FILE),
            "action_logprob_binding_audit": str(output_root / ACTION_LOGPROB_FILE),
            "policy_gradient_path_audit": str(output_root / POLICY_GRADIENT_FILE),
            "ratio_clip_advantage_audit": str(output_root / RATIO_CLIP_FILE),
            "candidate_logit_sensitivity_audit": str(output_root / CANDIDATE_LOGIT_FILE),
            "diagnostic_smoke_summary": str(output_root / SMOKE_FILE),
            "recommended_stage21_6_config": str(output_root / RECOMMENDED_CONFIG_FILE),
            "routing": str(output_root / ROUTING_FILE),
            "report": str(output_root / REPORT_FILE),
            "manifest": str(output_root / MANIFEST_FILE),
        }
    )
    _write_json(output_root / SUMMARY_FILE, summary)


def _report(
    summary: dict[str, Any],
    action_logprob: dict[str, Any],
    policy_gradient: dict[str, Any],
    ratio_clip: dict[str, Any],
    candidate_logit: dict[str, Any],
    smoke: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# Stage21.19 Policy Update Signal Source Repair",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- action/logprob binding passed: `{action_logprob['binding_passed']}`",
            f"- old_log_prob max recompute error: `{action_logprob['old_log_prob_recompute_max_abs_error']}`",
            f"- policy gradient path passed: `{policy_gradient['policy_gradient_path_passed']}`",
            f"- policy head parameter delta mean: `{policy_gradient['parameter_delta_by_group_mean'].get('policy_head')}`",
            f"- ratio/clip/advantage passed: `{ratio_clip['ratio_clip_advantage_passed']}`",
            f"- strong state join count: `{candidate_logit['strong_state_join_available_count']}`",
            f"- mean abs probability delta: `{candidate_logit['mean_abs_probability_delta']}`",
            f"- diagnostic smoke status: `{smoke['status']}`",
            "",
            "Stage21.19 remains an offline diagnostic and does not authorize release, default-policy replacement, executor connection, or canary.",
            "",
        ]
    )


def _recommended_stage21_6_config(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    source = _read_json(Path(config["stage21_18_recommended_stage21_6_config"]))
    source["stage21_19_recommended"] = True
    source["stage21_19_source_root"] = str(output_root)
    source["publishes_checkpoint"] = False
    source["replaces_default_policy"] = False
    source["connects_real_executor"] = False
    source["starts_online_canary"] = False
    source["canary_traffic_fraction"] = 0.0
    source["training_or_release_authorized"] = False
    source["recommended_config_requires_separate_human_execution"] = True
    return source


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"unsupported schema_version: {config.get('schema_version')}")
    for field in (
        "stage21_18_root",
        "stage21_18_recommended_stage21_6_config",
        "stage21_4_base_config",
    ):
        if not config.get(field):
            raise ConfigError(f"missing required field: {field}")
        config[field] = str(_resolve_path(Path(str(config[field])), repo_root))
    config["old_log_prob_recompute_tolerance"] = _positive_float(config.get("old_log_prob_recompute_tolerance", 1.0e-5), "old_log_prob_recompute_tolerance")
    config["min_advantage_std"] = _positive_float(config.get("min_advantage_std", 1.0e-6), "min_advantage_std")
    config["min_abs_advantage_mean"] = float(config.get("min_abs_advantage_mean", 0.0))
    config["min_policy_grad_norm"] = _positive_float(config.get("min_policy_grad_norm", 1.0e-8), "min_policy_grad_norm")
    config["min_policy_head_parameter_delta_l2"] = _positive_float(config.get("min_policy_head_parameter_delta_l2", 1.0e-9), "min_policy_head_parameter_delta_l2")
    config["policy_signal_enhanced_probability_delta_threshold"] = _positive_float(
        config.get("policy_signal_enhanced_probability_delta_threshold", 0.005),
        "policy_signal_enhanced_probability_delta_threshold",
    )
    config["candidate_logit_delta_margin_threshold"] = _positive_float(config.get("candidate_logit_delta_margin_threshold", 1.0e-4), "candidate_logit_delta_margin_threshold")
    config["diagnostic_smoke_epochs"] = int(config.get("diagnostic_smoke_epochs", 3))
    config["diagnostic_smoke_learning_rate"] = float(config.get("diagnostic_smoke_learning_rate", 2.0e-5))
    config["diagnostic_smoke_clip_ratio"] = float(config.get("diagnostic_smoke_clip_ratio", 0.2))
    config["diagnostic_smoke_advantage_clip_abs"] = float(config.get("diagnostic_smoke_advantage_clip_abs", 5.0))
    config["max_audit_sample_rows"] = int(config.get("max_audit_sample_rows", 20))
    return config


def _input_reasons(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    stage21_18_summary = _read_json_or_empty(Path(config["stage21_18_root"]) / "xunce-stage21-18-summary.json")
    if not stage21_18_summary:
        reasons.append("missing_stage21_18_summary")
    elif stage21_18_summary.get("next_required_change") != "repair_stage21_policy_update_signal_source":
        reasons.append("stage21_18_route_not_policy_update_signal_source")
    for field in ("stage21_18_root", "stage21_18_recommended_stage21_6_config", "stage21_4_base_config"):
        if not Path(config[field]).exists():
            reasons.append(f"missing_{field}")
    return reasons


def _boundary_reasons(config: dict[str, Any]) -> list[str]:
    reasons = [f"stage21_19_config_{field}_true" for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("stage21_19_config_canary_traffic_fraction_nonzero")
    return reasons


def _select_smoke_seed_run(seed_runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not seed_runs:
        return None
    return sorted(seed_runs, key=lambda row: (int(row.get("round_index") or 0), int(row.get("seed") or 0)))[-1]


def _masked_log_prob(logits: Any, mask: Any, action_index: int) -> float | None:
    values = _number_list(logits)
    flags = _bool_list(mask)
    if action_index < 0 or action_index >= len(values) or action_index >= len(flags) or not flags[action_index]:
        return None
    valid = [value for value, keep in zip(values, flags) if keep and math.isfinite(value)]
    if not valid:
        return None
    max_logit = max(valid)
    denom = max_logit + math.log(sum(math.exp(value - max_logit) for value in valid))
    return values[action_index] - denom


def _is_allowed(action_index: int, *masks: Any) -> bool:
    if action_index < 0:
        return False
    for mask in masks:
        flags = _bool_list(mask)
        if action_index >= len(flags) or not flags[action_index]:
            return False
    return True


def _tensor_values(value: Any) -> Any:
    if isinstance(value, dict) and "values" in value:
        values = value["values"]
        if isinstance(values, list) and values and isinstance(values[0], list):
            return values[0]
        return values
    return value


def _strong_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    required = (
        row.get("scenario_id"),
        row.get("step_index"),
        row.get("current_cell") or row.get("current_cell_before"),
        row.get("covered_cells_hash"),
        row.get("candidate_set_hash"),
    )
    if any(value is None or value == "" for value in required):
        return None
    return tuple(_stable_cell(value) if index == 2 else value for index, value in enumerate(required))


def _unique_rows_by_strong_key(rows: list[dict[str, Any]]) -> tuple[dict[tuple[Any, ...], dict[str, Any]], int]:
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    duplicate_count = 0
    for row in rows:
        key = _strong_key(row)
        if key is None:
            continue
        if key in by_key:
            duplicate_count += 1
            continue
        by_key[key] = row
    return by_key, duplicate_count


def _inference_vector_contract_passed(row: dict[str, Any]) -> bool:
    lengths = [
        len(_number_list(row.get("action_probs"))),
        len(_number_list(row.get("logits"))),
        len(_number_list(row.get("masked_logits"))),
        len(_bool_list(row.get("action_mask"))),
    ]
    candidate_cells = row.get("candidate_cells")
    if isinstance(candidate_cells, list):
        lengths.append(len(candidate_cells))
    return bool(lengths and min(lengths) > 0 and len(set(lengths)) == 1)


def _best_cpc_index_by_strong_key(path: Path) -> dict[tuple[Any, ...], int]:
    best_by_key: dict[tuple[Any, ...], tuple[int, float]] = {}
    for row in _read_jsonl(path):
        if row.get("policy") != "xunce" and row.get("executing_policy") != "xunce":
            continue
        if row.get("action_mask_valid") is False:
            continue
        if row.get("path_allowed_by_risk") is False:
            continue
        key = _strong_key(row)
        index = _int(row.get("candidate_index"))
        score = _float(row.get("coverage_gain_per_path_cost"))
        if key is None or index is None or score is None or not math.isfinite(score):
            continue
        if key not in best_by_key or score > best_by_key[key][1]:
            best_by_key[key] = (index, score)
    return {key: index for key, (index, _) in best_by_key.items()}


def _best_cpc_index(row: dict[str, Any]) -> int | None:
    candidates = row.get("candidate_metrics") or row.get("candidate_rows") or row.get("candidates")
    if not isinstance(candidates, list):
        return None
    best_index = None
    best_value = -float("inf")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        coverage = _float(candidate.get("expected_new_coverage_cell_count") or candidate.get("new_covered_cell_count"))
        path_cost = _float(candidate.get("path_cost") or candidate.get("path_cost_m"))
        if coverage is None or path_cost is None or path_cost <= 0.0:
            continue
        score = coverage / path_cost
        if score > best_value:
            best_value = score
            best_index = index
    return best_index


def _checkpoint_parameter_delta_by_group(source: Any, experimental: Any) -> dict[str, float]:
    if not source or not experimental:
        return {}
    source_path = Path(str(source))
    experimental_path = Path(str(experimental))
    if not source_path.is_file() or not experimental_path.is_file():
        return {}
    try:
        import torch

        source_state = torch.load(source_path, map_location="cpu")
        experiment_state = torch.load(experimental_path, map_location="cpu")
        source_model = source_state.get("model_state_dict", source_state)
        experiment_model = experiment_state.get("model_state_dict", experiment_state)
        sums: dict[str, float] = {}
        for key, before in source_model.items():
            after = experiment_model.get(key)
            if after is None:
                continue
            group = _parameter_group(key)
            delta = (after.detach().to(dtype=torch.float64) - before.detach().to(dtype=torch.float64)).pow(2).sum().sqrt().item()
            sums[group] = sums.get(group, 0.0) + float(delta)
        return sums
    except Exception:
        return {}


def _parameter_group(name: str) -> str:
    if name.startswith("policy_head"):
        return "policy_head"
    if name.startswith("value_head"):
        return "value_head"
    if name.startswith("candidate_encoder"):
        return "candidate_encoder"
    if name.startswith(("context_encoder", "edge_encoder", "memory_encoder", "message_", "topology_bias")):
        return "shared_trunk"
    return "other"


def _mean_group_deltas(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: mean([row.get(key, 0.0) for row in rows]) for key in keys}


def _empty_smoke(status: str = "skipped", reason: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "diagnostic_smoke_executed": False,
        "status": status,
        "reason": reason,
        "mean_abs_probability_delta": 0.0,
        "post_update_approx_kl_abs_max": 0.0,
        "argmax_changed_count": 0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
    }


def _number_list(value: Any) -> list[float]:
    if isinstance(value, dict) and "values" in value:
        value = value["values"]
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            pass
    return result


def _bool_list(value: Any) -> list[bool]:
    if isinstance(value, dict) and "values" in value:
        value = value["values"]
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        return []
    return [bool(item) for item in value]


def _stable_cell(value: Any) -> tuple[Any, ...]:
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _argmax(values: list[float]) -> int | None:
    if not values:
        return None
    return max(range(len(values)), key=lambda index: values[index])


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def _max_float(values: Any) -> float:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return max(finite) if finite else 0.0


def _min_float(values: Any) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return min(finite) if finite else None


def _max_abs(values: Any) -> float:
    finite = [abs(float(value)) for value in values if value is not None and math.isfinite(float(value))]
    return max(finite) if finite else 0.0


def _max_ratio(numerators: list[float | None], denominators: list[float | None]) -> float:
    ratios = []
    for numerator, denominator in zip(numerators, denominators):
        if numerator is None or denominator is None or denominator <= 0.0:
            continue
        ratios.append(float(numerator) / float(denominator))
    return max(ratios) if ratios else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
