from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

REPO_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_IMPORT_ROOT))

from scripts.run_xunce_stage21_15_policy_update_signal_strength_calibration import (
    _action_probs,
    _argmax,
    _combo_boundary_reasons,
    _distribution,
    _float,
    _format_lr_for_id,
    _inference_probability_contract_valid,
    _int,
    _is_xunce_row,
    _positive_float,
    _positive_int,
    _prob_delta,
    _read_json,
    _read_json_or_empty,
    _read_jsonl,
    _resolve_path,
    _selected_index,
    _selected_rank,
    _strong_key,
    _unique_strong_map,
    _write_json,
    _write_jsonl,
)


CONFIG_SCHEMA_VERSION = "xunce-stage21-16-policy-signal-margin-credit-attribution-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage21-16-summary/v1"
SWEEP_ROW_SCHEMA_VERSION = "xunce-stage21-16-sweep-result/v1"
UPDATE_AUDIT_SCHEMA_VERSION = "xunce-stage21-16-update-strength-audit/v1"
SEPARATION_AUDIT_SCHEMA_VERSION = "xunce-stage21-16-reward-advantage-separation-audit/v1"
MARGIN_AUDIT_SCHEMA_VERSION = "xunce-stage21-16-discrete-action-margin-audit/v1"
GRADIENT_AUDIT_SCHEMA_VERSION = "xunce-stage21-16-loss-gradient-attribution/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage21-16-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage21-16-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage21_16_policy_signal_margin_credit_attribution_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/outputs/"
    "path_feedback_batch_xunce_stage21_16_policy_signal_margin_credit_attribution_v1"
)

SUMMARY_FILE = "xunce-stage21-16-summary.json"
UPDATE_AUDIT_FILE = "xunce-stage21-16-update-strength-audit.json"
SEPARATION_AUDIT_FILE = "xunce-stage21-16-reward-advantage-separation-audit.json"
MARGIN_AUDIT_FILE = "xunce-stage21-16-discrete-action-margin-audit.json"
GRADIENT_AUDIT_FILE = "xunce-stage21-16-loss-gradient-attribution.json"
SWEEP_RESULTS_FILE = "xunce-stage21-16-sweep-results.jsonl"
RECOMMENDED_CONFIG_FILE = "xunce-stage21-16-recommended-stage21-6-config.json"
ROUTING_FILE = "xunce-stage21-16-next-stage-routing.json"
REPORT_FILE = "xunce-stage21-16-report.md"
MANIFEST_FILE = "xunce-stage21-16-manifest.json"

ROUTE_INPUTS = "rerun_stage21_16_required_inputs"
ROUTE_BOUNDARY = "resolve_stage21_16_boundary_rejections"
ROUTE_BINDING = "repair_stage21_5_inference_binding_contract"
ROUTE_NUMERICS = "continue_stage21_9_gradient_normalization_loss_scaling_repair"
ROUTE_SIGNAL = "increase_stage21_policy_update_signal_strength"
ROUTE_MARGIN = "calibrate_stage21_discrete_action_margin_crossing"
ROUTE_CREDIT = "repair_stage21_return_advantage_credit_assignment"
ROUTE_LOSS_BALANCE = "repair_stage21_policy_value_loss_balance"
ROUTE_SCALE = "scale_stage21_ppo_pilot_scenarios_and_horizon"
ROUTE_RUNTIME = "stage21_16_calibration_runtime_budget_blocked"

BOUNDARY_FIELDS = (
    "stage21_16_authorized",
    "training_or_release_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)

AGGREGATE_HARD_FAIL_FIELDS = (
    "hard_risk_violation_total",
    "safety_boundary_violation_total",
    "execution_boundary_violation_total",
    "model_inference_failure_total",
    "model_inference_mask_violation_total",
    "open_grid_fallback_total",
    "path_planning_failure_total",
    "unreachable_selected_total",
    "checkpoint_reload_failed_count",
    "checkpoint_not_experimental_only_count",
    "seed_release_boundary_violation_total",
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 21.16 policy signal, margin, and credit attribution.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        summary = run_xunce_stage21_16_policy_signal_margin_credit_attribution(
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
                "mean_abs_probability_delta_max": summary["mean_abs_probability_delta_max"],
                "best_candidate_probability_margin_mean": summary["best_candidate_probability_margin_mean"],
                "value_to_policy_grad_norm_ratio_max": summary["value_to_policy_grad_norm_ratio_max"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] in {"passed", "partial"} else 1


def run_xunce_stage21_16_policy_signal_margin_credit_attribution(
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
    combos = _build_combos(config)
    rows: list[dict[str, Any]] = []
    if not input_reasons and not boundary_reasons:
        rows = _collect_existing_results(config, output_root, combos)
        _execute_missing_combos(config, output_root, combos, rows, repo_root)
        rows = _collect_existing_results(config, output_root, combos)

    update_audit = _update_strength_audit(rows)
    separation_audit = _separation_audit(rows)
    margin_audit = _margin_audit(rows)
    gradient_audit = _gradient_attribution(rows)
    status, route, primary_reason = _route(
        input_reasons=input_reasons,
        boundary_reasons=boundary_reasons,
        rows=rows,
        update_audit=update_audit,
        separation_audit=separation_audit,
        margin_audit=margin_audit,
        gradient_audit=gradient_audit,
        probability_delta_threshold=float(config["probability_delta_threshold"]),
        large_probability_margin_threshold=float(config["large_probability_margin_threshold"]),
        min_reward_cpc_correlation=float(config["min_reward_cpc_correlation"]),
        min_advantage_cpc_correlation=float(config["min_advantage_cpc_correlation"]),
        min_policy_grad_ratio=float(config["min_policy_grad_ratio"]),
        max_value_to_policy_grad_ratio=float(config["max_value_to_policy_grad_ratio"]),
    )
    recommended = _select_recommendation(config, combos, rows, route)
    recommended_config = _write_recommended_config(config, output_root, recommended)
    return _write_outputs(
        config=config,
        config_path=config_path,
        output_root=output_root,
        combos=combos,
        rows=rows,
        update_audit=update_audit,
        separation_audit=separation_audit,
        margin_audit=margin_audit,
        gradient_audit=gradient_audit,
        recommended=recommended,
        recommended_config=recommended_config,
        status=status,
        route=route,
        primary_reason=primary_reason,
        input_reasons=input_reasons,
        boundary_reasons=boundary_reasons,
    )


def _input_reasons(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    summary = _read_json_or_empty(Path(config["stage21_15_root"]) / "xunce-stage21-15-summary.json")
    if not summary:
        reasons.append("missing_stage21_15_summary")
    else:
        if summary.get("next_required_change") != "increase_stage21_policy_update_signal_strength":
            reasons.append("stage21_15_route_not_increase_policy_update_signal_strength")
        if _int(summary.get("strong_state_join_available_count")) <= 0:
            reasons.append("stage21_15_no_strong_state_join")
        if _int(summary.get("strong_state_binding_unavailable_count")) > 0:
            reasons.append("stage21_15_strong_state_binding_incomplete")
    for field in ("stage21_6_base_config", "stage21_4_base_config"):
        if not Path(config[field]).is_file():
            reasons.append(f"missing_{field}")
    return sorted(set(reasons))


def _boundary_reasons(config: dict[str, Any]) -> list[str]:
    reasons = [f"stage21_16_config_{field}_true" for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) > 0.0:
        reasons.append("stage21_16_config_canary_traffic_fraction_nonzero")
    return sorted(set(reasons))


def _build_combos(config: dict[str, Any]) -> list[dict[str, Any]]:
    configured = config.get("priority_combinations")
    if not isinstance(configured, list) or not configured:
        raise ConfigError("priority_combinations must be a non-empty list")
    combos: list[dict[str, Any]] = []
    for index, combo in enumerate(configured):
        epochs = _positive_int(combo.get("epochs"), "epochs")
        learning_rate = _positive_float(combo.get("learning_rate"), "learning_rate")
        clip_ratio = _positive_float(combo.get("clip_ratio"), "clip_ratio")
        loss_scale = _positive_float(combo.get("loss_scale", config["loss_scale"]), "loss_scale")
        combos.append(
            {
                "combo_index": index,
                "combo_id": str(combo.get("combo_id") or f"e{epochs}_lr{learning_rate:g}_c{clip_ratio:g}_loss{loss_scale:g}"),
                "epochs": epochs,
                "learning_rate": learning_rate,
                "clip_ratio": clip_ratio,
                "loss_scale": loss_scale,
                "stage21_4_max_grad_norm": float(config["stage21_4_max_grad_norm"]),
                "stage21_6_pre_clip_grad_norm_gate": float(config["stage21_6_pre_clip_grad_norm_gate"]),
            }
        )
    return combos


def _collect_existing_results(config: dict[str, Any], output_root: Path, combos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for combo in combos:
        combo_root = _combo_root(config, output_root, combo)
        stage21_6_root = combo_root / "stage21_6"
        row = _read_combo_result(combo, stage21_6_root)
        if row:
            rows.append(row)
            continue
        blocker = _read_json_or_empty(combo_root / "runtime-blocker.json")
        if blocker:
            rows.append(_runtime_blocker_row(combo, stage21_6_root, blocker))
    return rows


def _execute_missing_combos(
    config: dict[str, Any],
    output_root: Path,
    combos: list[dict[str, Any]],
    existing_rows: list[dict[str, Any]],
    repo_root: Path,
) -> None:
    if not config["execute_sweep"]:
        return
    done = {row["combo_id"] for row in existing_rows if not row.get("runtime_blocked")}
    executed = 0
    for combo in combos:
        if combo["combo_id"] in done:
            continue
        if executed >= int(config["max_new_combinations_to_execute"]):
            return
        _run_combo(config, output_root, combo, repo_root)
        executed += 1


def _run_combo(config: dict[str, Any], output_root: Path, combo: dict[str, Any], repo_root: Path) -> None:
    combo_root = _combo_root(config, output_root, combo)
    combo_root.mkdir(parents=True, exist_ok=True)
    stage21_4_config = _read_json(Path(config["stage21_4_base_config"]))
    stage21_4_config.update(
        {
            "epochs": combo["epochs"],
            "learning_rate": combo["learning_rate"],
            "clip_ratio": combo["clip_ratio"],
            "max_grad_norm": float(config["stage21_4_max_grad_norm"]),
            "loss_scale": float(combo["loss_scale"]),
            "value_loss_coefficient": float(config["value_loss_coefficient"]),
            "advantage_clip_abs": float(config["advantage_clip_abs"]),
            "normalize_minibatch_advantages": bool(config["normalize_minibatch_advantages"]),
        }
    )
    stage21_4_path = combo_root / "stage21_4_config.json"
    _write_json(stage21_4_path, stage21_4_config)

    stage21_6_config = _read_json(Path(config["stage21_6_base_config"]))
    stage21_6_config.update(
        {
            "stage21_4_base_config": str(stage21_4_path),
            "seed_list": config["seed_list"],
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "max_grad_norm": float(config["stage21_6_pre_clip_grad_norm_gate"]),
            "execute_seed_pipeline": True,
            "stage21_6_authorized": False,
            "training_or_release_authorized": False,
            "runs_new_ppo_update": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
            "stage21_16_generated": True,
        }
    )
    stage21_6_path = combo_root / "stage21_6_config.json"
    _write_json(stage21_6_path, stage21_6_config)
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_xunce_stage21_6_multi_seed_ppo_pilot.py"),
        "--config",
        str(stage21_6_path),
        "--output-root",
        str(combo_root / "stage21_6"),
        "--repo-root",
        str(repo_root),
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=int(config["per_combo_timeout_seconds"]),
        )
    except subprocess.TimeoutExpired as exc:
        _write_json(
            combo_root / "runtime-blocker.json",
            {
                "status": "blocked",
                "reason_code": ROUTE_RUNTIME,
                "combo_id": combo["combo_id"],
                "timeout_seconds": int(config["per_combo_timeout_seconds"]),
                "stdout": exc.stdout,
                "stderr": exc.stderr,
            },
        )
        return
    _write_json(
        combo_root / "stage21_6-subprocess-result.json",
        {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "cmd": cmd},
    )
    if completed.returncode != 0:
        _write_json(
            combo_root / "runtime-blocker.json",
            {
                "status": "blocked",
                "reason_code": "stage21_16_calibration_subprocess_failed",
                "combo_id": combo["combo_id"],
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )


def _read_combo_result(combo: dict[str, Any], stage21_6_root: Path) -> dict[str, Any]:
    summary = _read_json_or_empty(stage21_6_root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json")
    aggregate = _read_json_or_empty(stage21_6_root / "xunce-stage21-6-aggregate-metrics.json")
    lineage = _read_json_or_empty(stage21_6_root / "xunce-stage21-6-lineage-audit.json")
    seed_rows = _read_jsonl(stage21_6_root / "xunce-stage21-6-seed-results.jsonl")
    if not summary or not aggregate or not seed_rows:
        return {}
    boundary_reasons = _combo_boundary_reasons(summary, aggregate, seed_rows)
    signal = _combo_policy_signal(seed_rows)
    separation = _combo_reward_advantage_separation(seed_rows)
    gradient = _combo_gradient_attribution(seed_rows)
    capped = _capped_aggregate(seed_rows)
    pre_clip = _float(aggregate.get("pre_clip_grad_norm_max"))
    gate = float(combo["stage21_6_pre_clip_grad_norm_gate"])
    post_clip = _max([_float(row.get("post_clip_grad_norm")) for row in seed_rows])
    kl = _float(aggregate.get("max_post_update_approx_kl_mean"))
    entropy = _float(aggregate.get("min_entropy_mean"))
    numerically_stable = bool(
        not boundary_reasons
        and lineage.get("passed") is True
        and pre_clip is not None
        and pre_clip <= gate
        and post_clip is not None
        and post_clip <= float(combo["stage21_4_max_grad_norm"]) + 1.0e-5
        and kl is not None
        and abs(kl) <= 0.5
        and entropy is not None
        and entropy >= 0.0
        and all(row.get("grad_norm_finite") is True for row in seed_rows)
    )
    final_delta = _float(aggregate.get("final_coverage_delta_mean"))
    auc_delta = _float(aggregate.get("coverage_auc_delta_mean"))
    final_min = _float(aggregate.get("final_coverage_delta_min"))
    auc_min = _float(aggregate.get("coverage_auc_delta_min"))
    capped_final = _float(capped.get("final_coverage_rate_capped_delta_mean"))
    capped_auc = _float(capped.get("coverage_curve_auc_capped_delta_mean"))
    capped_final_min = _float(capped.get("final_coverage_rate_capped_delta_min"))
    capped_auc_min = _float(capped.get("coverage_curve_auc_capped_delta_min"))
    return {
        "schema_version": SWEEP_ROW_SCHEMA_VERSION,
        "combo_id": combo["combo_id"],
        "combo_index": combo["combo_index"],
        "stage21_6_root": str(stage21_6_root),
        "epochs": combo["epochs"],
        "learning_rate": combo["learning_rate"],
        "clip_ratio": combo["clip_ratio"],
        "loss_scale": combo["loss_scale"],
        "status": summary.get("status"),
        "stage21_6_next_required_change": summary.get("next_required_change"),
        "boundary_reason_codes": boundary_reasons,
        "lineage_passed": lineage.get("passed"),
        "trainable_transition_count_total": _int(aggregate.get("trainable_transition_count_total")),
        "pre_clip_grad_norm_max": pre_clip,
        "post_clip_grad_norm_max": post_clip,
        "post_update_approx_kl_mean": kl,
        "min_entropy_mean": entropy,
        "parameter_delta_l2_mean": _mean([_float(row.get("parameter_delta_l2")) for row in seed_rows]),
        "final_coverage_delta_mean": final_delta,
        "final_coverage_delta_min": final_min,
        "coverage_auc_delta_mean": auc_delta,
        "coverage_auc_delta_min": auc_min,
        "final_coverage_rate_capped_delta_mean": capped_final,
        "coverage_curve_auc_capped_delta_mean": capped_auc,
        "final_coverage_rate_capped_delta_min": capped_final_min,
        "coverage_curve_auc_capped_delta_min": capped_auc_min,
        "coverage_auc_improved": all(
            value is not None and value > 0.0
            for value in (final_delta, auc_delta, final_min, auc_min, capped_final, capped_auc, capped_final_min, capped_auc_min)
        ),
        "coverage_auc_not_regressed": all(
            value is not None and value >= 0.0
            for value in (final_delta, auc_delta, final_min, auc_min, capped_final, capped_auc, capped_final_min, capped_auc_min)
        ),
        "numerically_stable": numerically_stable,
        **signal,
        **separation,
        **gradient,
    }


def _combo_policy_signal(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = 0
    unavailable = 0
    duplicate_count = 0
    argmax_changed = 0
    selected_action_changed = 0
    selected_rank_changed = 0
    best_not_top = 0
    mean_abs_deltas: list[float] = []
    selected_deltas: list[float] = []
    best_deltas: list[float] = []
    probability_margins: list[float] = []
    logit_margins: list[float] = []
    rank_gaps: list[float] = []
    for row in seed_rows:
        stage21_5_root = Path(str(row.get("stage21_5_root", "")))
        summary5 = _read_json_or_empty(stage21_5_root / "xunce-stage21-5-post-update-evaluation-summary.json")
        pre_root = Path(str(summary5.get("pre_evaluation_root") or stage21_5_root / "pre_ppo_xunce"))
        post_root = Path(str(summary5.get("post_evaluation_root") or stage21_5_root / "post_ppo_xunce"))
        pre_rows = [item for item in _read_jsonl(pre_root / "xunce-exploration-coverage-model-inference.jsonl") if _is_xunce_row(item)]
        post_rows = [item for item in _read_jsonl(post_root / "xunce-exploration-coverage-model-inference.jsonl") if _is_xunce_row(item)]
        best_by_key, scores_by_key = _best_candidate_by_key(_read_jsonl(pre_root / "xunce-exploration-coverage-candidate-metric-audit.jsonl"))
        pre_map, pre_dupes = _unique_strong_map(pre_rows)
        post_map, post_dupes = _unique_strong_map(post_rows)
        duplicate_count += pre_dupes + post_dupes
        if pre_dupes or post_dupes:
            unavailable += len(pre_rows)
            continue
        for pre in pre_rows:
            key = _strong_key(pre)
            if key is None:
                unavailable += 1
                continue
            post = post_map.get(key)
            if post is None or pre_map.get(key) is not pre:
                unavailable += 1
                continue
            pre_probs = _action_probs(pre)
            post_probs = _action_probs(post)
            if (
                not _inference_probability_contract_valid(pre, pre_probs)
                or not _inference_probability_contract_valid(post, post_probs)
                or len(pre_probs) != len(post_probs)
            ):
                unavailable += 1
                continue
            matched += 1
            deltas = [abs(float(b) - float(a)) for a, b in zip(pre_probs, post_probs)]
            mean_abs_deltas.append(mean(deltas))
            selected = _selected_index(pre)
            selected_delta = _prob_delta(pre_probs, post_probs, selected)
            if selected_delta is not None:
                selected_deltas.append(selected_delta)
            best = best_by_key.get(key)
            best_delta = _prob_delta(pre_probs, post_probs, best) if best is not None else None
            if best_delta is not None:
                best_deltas.append(best_delta)
            if _argmax(pre_probs) != _argmax(post_probs):
                argmax_changed += 1
            if _selected_index(pre) != _selected_index(post):
                selected_action_changed += 1
            if _selected_rank(pre) != _selected_rank(post):
                selected_rank_changed += 1
            top = _argmax(pre_probs)
            if best is not None and top is not None and 0 <= best < len(pre_probs) and best != top:
                best_not_top += 1
                probability_margins.append(float(pre_probs[top]) - float(pre_probs[best]))
                logits = _logits(pre)
                if len(logits) == len(pre_probs):
                    logit_margins.append(float(logits[top]) - float(logits[best]))
                rank_gaps.append(float(_rank_of(pre_probs, best) - _rank_of(pre_probs, top)))
    return {
        "strong_state_join_available_count": matched,
        "strong_state_binding_unavailable_count": unavailable,
        "duplicate_strong_state_key_count": duplicate_count,
        "mean_abs_probability_delta": _mean(mean_abs_deltas),
        "selected_action_probability_delta_mean": _mean(selected_deltas),
        "selected_action_probability_abs_delta_mean": _mean([abs(value) for value in selected_deltas]),
        "best_coverage_per_cost_probability_delta_mean": _mean(best_deltas),
        "argmax_changed_count": argmax_changed,
        "selected_action_changed_count": selected_action_changed,
        "selected_rank_changed_count": selected_rank_changed,
        "best_candidate_not_top_count": best_not_top,
        "best_candidate_not_top_rate": (best_not_top / matched) if matched else None,
        "probability_margin_mean": _mean(probability_margins),
        "probability_margin_max": _max(probability_margins),
        "logit_margin_mean": _mean(logit_margins),
        "logit_margin_max": _max(logit_margins),
        "best_candidate_rank_gap_mean": _mean(rank_gaps),
        "best_candidate_rank_gap_max": _max(rank_gaps),
    }


def _best_candidate_by_key(rows: list[dict[str, Any]]) -> tuple[dict[tuple[Any, ...], int], dict[tuple[Any, ...], dict[int, float]]]:
    best: dict[tuple[Any, ...], tuple[int, float]] = {}
    scores: dict[tuple[Any, ...], dict[int, float]] = {}
    for row in rows:
        if row.get("action_mask_valid") is False or row.get("path_allowed_by_risk") is False:
            continue
        key = _strong_key(row)
        index = _int(row.get("candidate_index"))
        score = _coverage_per_cost_score(row)
        if key is None or score is None:
            continue
        scores.setdefault(key, {})[index] = score
        if key not in best or score > best[key][1]:
            best[key] = (index, score)
    return {key: value[0] for key, value in best.items()}, scores


def _coverage_per_cost_score(row: dict[str, Any]) -> float | None:
    score = _float(row.get("coverage_gain_per_path_cost"))
    if score is not None:
        return score
    coverage = _float(row.get("expected_new_coverage_cell_count"))
    path_cost = _float(row.get("path_cost"))
    if coverage is not None and path_cost is not None and path_cost > 0.0:
        return coverage / path_cost
    return None


def _combo_reward_advantage_separation(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    reward_pairs: list[tuple[float, float]] = []
    advantage_pairs: list[tuple[float, float]] = []
    reward_gaps: list[float] = []
    advantage_gaps: list[float] = []
    for seed in seed_rows:
        stage21_2_root = Path(str(seed.get("stage21_2_root") or Path(str(seed.get("seed_root", ""))) / "stage21_2"))
        stage21_3_root = Path(str(seed.get("stage21_3_root", "")))
        reward_rows = _read_jsonl(stage21_2_root / "xunce-stage21-2-reward-contract-evaluation.jsonl")
        cpc_by_transition: dict[str, float] = {}
        for row in reward_rows:
            if row.get("trainable") is False or row.get("hard_risk_rejected") is True:
                continue
            cpc = _cpc_value(row)
            reward = _float(row.get("reward"))
            transition_id = str(row.get("transition_id") or "")
            if cpc is not None and reward is not None:
                reward_pairs.append((cpc, reward))
                if transition_id:
                    cpc_by_transition[transition_id] = cpc
        reward_gaps.append(_upper_lower_gap(reward_pairs))
        for row in _read_jsonl(stage21_3_root / "xunce-stage21-3-return-advantage-audit.jsonl"):
            if row.get("stage21_3_split") not in (None, "", "train"):
                continue
            cpc = _cpc_value(row)
            if cpc is None:
                cpc = cpc_by_transition.get(str(row.get("transition_id") or ""))
            advantage = _float(row.get("advantage", row.get("normalized_advantage")))
            if cpc is not None and advantage is not None:
                advantage_pairs.append((cpc, advantage))
        advantage_gaps.append(_upper_lower_gap(advantage_pairs))
    return {
        "reward_cpc_observation_count": len(reward_pairs),
        "advantage_cpc_observation_count": len(advantage_pairs),
        "reward_coverage_per_cost_correlation": _correlation([x for x, _ in reward_pairs], [y for _, y in reward_pairs]),
        "advantage_coverage_per_cost_correlation": _correlation([x for x, _ in advantage_pairs], [y for _, y in advantage_pairs]),
        "reward_high_vs_low_cpc_gap_mean": _mean([value for value in reward_gaps if value is not None]),
        "advantage_high_vs_low_cpc_gap_mean": _mean([value for value in advantage_gaps if value is not None]),
    }


def _combo_gradient_attribution(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    policy: list[float] = []
    value: list[float] = []
    entropy: list[float] = []
    total: list[float] = []
    policy_ratios: list[float] = []
    value_ratios: list[float] = []
    entropy_ratios: list[float] = []
    value_to_policy: list[float] = []
    for seed in seed_rows:
        stage21_4_root = Path(str(seed.get("stage21_4_root", "")))
        audit = _read_json_or_empty(stage21_4_root / "xunce-stage21-4-gradient-audit.json")
        components = audit.get("component_grad_norms") if isinstance(audit.get("component_grad_norms"), dict) else {}
        p = _float(components.get("policy_loss_grad_norm") or audit.get("policy_loss_grad_norm"))
        v = _float(components.get("value_loss_grad_norm") or audit.get("value_loss_grad_norm"))
        e = _float(components.get("entropy_loss_grad_norm") or audit.get("entropy_loss_grad_norm"))
        t = _float(components.get("total_loss_grad_norm") or audit.get("pre_clip_grad_norm") or audit.get("grad_norm"))
        if p is not None:
            policy.append(abs(p))
        if v is not None:
            value.append(abs(v))
        if e is not None:
            entropy.append(abs(e))
        if t is not None:
            total.append(abs(t))
        if t is not None and t > 0.0:
            if p is not None:
                policy_ratios.append(abs(p) / t)
            if v is not None:
                value_ratios.append(abs(v) / t)
            if e is not None:
                entropy_ratios.append(abs(e) / t)
        if p is not None and abs(p) > 0.0 and v is not None:
            value_to_policy.append(abs(v) / abs(p))
    return {
        "component_gradient_audit_available": bool(policy or value or entropy),
        "policy_loss_grad_norm_mean": _mean(policy),
        "value_loss_grad_norm_mean": _mean(value),
        "entropy_loss_grad_norm_mean": _mean(entropy),
        "total_loss_grad_norm_mean": _mean(total),
        "policy_grad_ratio_mean": _mean(policy_ratios),
        "policy_grad_ratio_min": _min(policy_ratios),
        "value_grad_ratio_mean": _mean(value_ratios),
        "entropy_grad_ratio_mean": _mean(entropy_ratios),
        "value_to_policy_grad_norm_ratio_max": _max(value_to_policy),
        "value_to_policy_grad_norm_ratio_mean": _mean(value_to_policy),
    }


def _cpc_value(row: dict[str, Any]) -> float | None:
    direct = _float(row.get("coverage_per_cost") or row.get("coverage_gain_per_path_cost"))
    if direct is not None:
        return direct
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    direct = _float(info.get("coverage_per_cost") or info.get("coverage_gain_per_path_cost"))
    if direct is not None:
        return direct
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    direct = _float(metrics.get("coverage_per_cost") or metrics.get("coverage_gain_per_path_cost"))
    if direct is not None:
        return direct
    coverage = _float(metrics.get("coverage_rate_delta") or metrics.get("coverage_gain_rate") or info.get("coverage_rate_delta"))
    path_cost = _float(metrics.get("path_cost_m") or info.get("path_cost"))
    if coverage is not None and path_cost is not None and path_cost > 0.0:
        return coverage / path_cost
    components = row.get("components") if isinstance(row.get("components"), dict) else {}
    return _float(components.get("coverage_per_cost_component"))


def _upper_lower_gap(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    sorted_pairs = sorted(pairs, key=lambda item: item[0])
    midpoint = len(sorted_pairs) // 2
    low = [value for _, value in sorted_pairs[:midpoint]]
    high = [value for _, value in sorted_pairs[midpoint:]]
    low_mean = _mean(low)
    high_mean = _mean(high)
    if low_mean is None or high_mean is None:
        return None
    return high_mean - low_mean


def _update_strength_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stable = [row for row in rows if row.get("numerically_stable") and not row.get("runtime_blocked")]
    return {
        "schema_version": UPDATE_AUDIT_SCHEMA_VERSION,
        "completed_combo_count": len(rows),
        "stable_combo_count": len(stable),
        "mean_abs_probability_delta_max": max((_float(row.get("mean_abs_probability_delta")) or 0.0 for row in stable), default=0.0),
        "best_coverage_per_cost_probability_delta_max": max(
            (_float(row.get("best_coverage_per_cost_probability_delta_mean")) or 0.0 for row in stable),
            default=0.0,
        ),
        "parameter_delta_l2_max": max((_float(row.get("parameter_delta_l2_mean")) or 0.0 for row in stable), default=0.0),
        "post_update_approx_kl_abs_max": max((abs(_float(row.get("post_update_approx_kl_mean")) or 0.0) for row in stable), default=0.0),
        "argmax_changed_count": sum(_int(row.get("argmax_changed_count")) for row in stable),
        "selected_rank_changed_count": sum(_int(row.get("selected_rank_changed_count")) for row in stable),
        "selected_action_changed_count": sum(_int(row.get("selected_action_changed_count")) for row in stable),
        "strong_state_join_available_count": sum(_int(row.get("strong_state_join_available_count")) for row in stable),
        "strong_state_binding_unavailable_count": sum(_int(row.get("strong_state_binding_unavailable_count")) for row in stable),
    }


def _separation_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SEPARATION_AUDIT_SCHEMA_VERSION,
        "reward_cpc_observation_count": sum(_int(row.get("reward_cpc_observation_count")) for row in rows),
        "advantage_cpc_observation_count": sum(_int(row.get("advantage_cpc_observation_count")) for row in rows),
        "reward_coverage_per_cost_correlation": _mean([_float(row.get("reward_coverage_per_cost_correlation")) for row in rows]),
        "advantage_coverage_per_cost_correlation": _mean([_float(row.get("advantage_coverage_per_cost_correlation")) for row in rows]),
        "reward_high_vs_low_cpc_gap_mean": _mean([_float(row.get("reward_high_vs_low_cpc_gap_mean")) for row in rows]),
        "advantage_high_vs_low_cpc_gap_mean": _mean([_float(row.get("advantage_high_vs_low_cpc_gap_mean")) for row in rows]),
    }


def _margin_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = sum(_int(row.get("strong_state_join_available_count")) for row in rows)
    not_top = sum(_int(row.get("best_candidate_not_top_count")) for row in rows)
    return {
        "schema_version": MARGIN_AUDIT_SCHEMA_VERSION,
        "strong_state_join_available_count": matched,
        "best_candidate_not_top_count": not_top,
        "best_candidate_not_top_rate": (not_top / matched) if matched else None,
        "probability_margin_mean": _mean([_float(row.get("probability_margin_mean")) for row in rows]),
        "probability_margin_max": _max([_float(row.get("probability_margin_max")) for row in rows]),
        "logit_margin_mean": _mean([_float(row.get("logit_margin_mean")) for row in rows]),
        "logit_margin_max": _max([_float(row.get("logit_margin_max")) for row in rows]),
        "best_candidate_rank_gap_mean": _mean([_float(row.get("best_candidate_rank_gap_mean")) for row in rows]),
        "best_candidate_rank_gap_max": _max([_float(row.get("best_candidate_rank_gap_max")) for row in rows]),
    }


def _gradient_attribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": GRADIENT_AUDIT_SCHEMA_VERSION,
        "component_gradient_audit_available": any(row.get("component_gradient_audit_available") for row in rows),
        "policy_loss_grad_norm_mean": _mean([_float(row.get("policy_loss_grad_norm_mean")) for row in rows]),
        "value_loss_grad_norm_mean": _mean([_float(row.get("value_loss_grad_norm_mean")) for row in rows]),
        "entropy_loss_grad_norm_mean": _mean([_float(row.get("entropy_loss_grad_norm_mean")) for row in rows]),
        "total_loss_grad_norm_mean": _mean([_float(row.get("total_loss_grad_norm_mean")) for row in rows]),
        "policy_grad_ratio_mean": _mean([_float(row.get("policy_grad_ratio_mean")) for row in rows]),
        "policy_grad_ratio_min": _min([_float(row.get("policy_grad_ratio_min")) for row in rows]),
        "value_grad_ratio_mean": _mean([_float(row.get("value_grad_ratio_mean")) for row in rows]),
        "entropy_grad_ratio_mean": _mean([_float(row.get("entropy_grad_ratio_mean")) for row in rows]),
        "value_to_policy_grad_norm_ratio_mean": _mean([_float(row.get("value_to_policy_grad_norm_ratio_mean")) for row in rows]),
        "value_to_policy_grad_norm_ratio_max": _max([_float(row.get("value_to_policy_grad_norm_ratio_max")) for row in rows]),
    }


def _route(
    *,
    input_reasons: list[str],
    boundary_reasons: list[str],
    rows: list[dict[str, Any]],
    update_audit: dict[str, Any],
    separation_audit: dict[str, Any],
    margin_audit: dict[str, Any],
    gradient_audit: dict[str, Any],
    probability_delta_threshold: float,
    large_probability_margin_threshold: float,
    min_reward_cpc_correlation: float,
    min_advantage_cpc_correlation: float,
    min_policy_grad_ratio: float,
    max_value_to_policy_grad_ratio: float,
) -> tuple[str, str, str]:
    if input_reasons:
        return "failed", ROUTE_INPUTS, "stage21_16_input_rejected"
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "stage21_16_boundary_rejected"
    if not rows:
        return "partial", ROUTE_RUNTIME, "no_completed_stage21_16_sweep_results"
    completed = [row for row in rows if not row.get("runtime_blocked")]
    if not completed:
        return "partial", ROUTE_RUNTIME, "stage21_16_sweep_runtime_blocked"
    stable = [row for row in completed if row.get("numerically_stable")]
    if not stable:
        return "failed", ROUTE_NUMERICS, "no_numerically_stable_combo"
    if any(_int(row.get("strong_state_join_available_count")) == 0 or _int(row.get("strong_state_binding_unavailable_count")) > 0 for row in stable):
        return "failed", ROUTE_BINDING, "strong_state_binding_contract_incomplete"
    if any(row.get("coverage_auc_improved") for row in stable):
        return "passed", ROUTE_SCALE, "policy_signal_smoke_improved"

    signal = _float(update_audit.get("mean_abs_probability_delta_max")) or 0.0
    if signal < probability_delta_threshold:
        return "failed", ROUTE_SIGNAL, "probability_delta_below_threshold"

    policy_ratio = _float(gradient_audit.get("policy_grad_ratio_mean"))
    value_to_policy = _float(gradient_audit.get("value_to_policy_grad_norm_ratio_max"))
    if gradient_audit.get("component_gradient_audit_available") and (
        (policy_ratio is not None and policy_ratio < min_policy_grad_ratio)
        or (value_to_policy is not None and value_to_policy > max_value_to_policy_grad_ratio)
    ):
        return "failed", ROUTE_LOSS_BALANCE, "policy_value_gradient_balance_failed"

    reward_corr = _float(separation_audit.get("reward_coverage_per_cost_correlation"))
    advantage_corr = _float(separation_audit.get("advantage_coverage_per_cost_correlation"))
    reward_gap = _float(separation_audit.get("reward_high_vs_low_cpc_gap_mean"))
    advantage_gap = _float(separation_audit.get("advantage_high_vs_low_cpc_gap_mean"))
    if (
        (_int(separation_audit.get("reward_cpc_observation_count")) >= 2 and reward_corr is not None and reward_corr < min_reward_cpc_correlation)
        or (_int(separation_audit.get("advantage_cpc_observation_count")) >= 2 and advantage_corr is not None and advantage_corr < min_advantage_cpc_correlation)
        or (reward_gap is not None and reward_gap < 0.0)
        or (advantage_gap is not None and advantage_gap < 0.0)
    ):
        return "failed", ROUTE_CREDIT, "reward_or_advantage_not_separated_for_coverage_per_cost"

    if _int(update_audit.get("argmax_changed_count")) > 0 or _int(update_audit.get("selected_rank_changed_count")) > 0:
        return "failed", ROUTE_CREDIT, "rank_or_argmax_changed_without_trajectory_gain"

    margin_large = max(
        _float(margin_audit.get("probability_margin_mean")) or 0.0,
        _float(margin_audit.get("probability_margin_max")) or 0.0,
    ) >= large_probability_margin_threshold
    if (
        (_float(update_audit.get("best_coverage_per_cost_probability_delta_max")) or 0.0) > 0.0
        and (_int(margin_audit.get("best_candidate_not_top_count")) > 0)
        and margin_large
    ):
        return "failed", ROUTE_MARGIN, "probability_moved_but_discrete_action_margin_not_crossed"

    return "failed", ROUTE_SIGNAL, "policy_update_signal_inconclusive"


def _select_recommendation(
    config: dict[str, Any],
    combos: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    route: str,
) -> dict[str, Any]:
    stable = [
        row
        for row in rows
        if row.get("numerically_stable")
        and not row.get("runtime_blocked")
        and row.get("coverage_auc_not_regressed")
        and _int(row.get("strong_state_join_available_count")) > 0
        and _int(row.get("strong_state_binding_unavailable_count")) == 0
    ]
    completed_ids = {str(row.get("combo_id")) for row in rows if not row.get("runtime_blocked")}
    if route == ROUTE_SIGNAL:
        for combo in combos:
            if str(combo["combo_id"]) not in completed_ids:
                return {**combo, "recommendation_type": "next_unrun_stage21_16_combo"}
        if stable:
            stable.sort(
                key=lambda row: (
                    -float(_float(row.get("mean_abs_probability_delta")) or 0.0),
                    abs(_float(row.get("post_update_approx_kl_mean")) or 0.0),
                )
            )
            best = stable[0]
            next_lr = min(float(_float(best.get("learning_rate")) or 0.0) * 2.0, 5e-5)
            return {
                "combo_index": len(combos),
                "combo_id": f"e{int(best['epochs'])}_lr{_format_lr_for_id(next_lr)}_c{_clip_for_id(float(best['clip_ratio']))}_loss{_loss_for_id(float(best['loss_scale']))}",
                "epochs": int(best["epochs"]),
                "learning_rate": next_lr,
                "clip_ratio": float(best["clip_ratio"]),
                "loss_scale": float(best["loss_scale"]),
                "stage21_4_max_grad_norm": float(config["stage21_4_max_grad_norm"]),
                "stage21_6_pre_clip_grad_norm_gate": float(config["stage21_6_pre_clip_grad_norm_gate"]),
                "recommendation_type": "next_extrapolated_policy_signal_combo",
            }
        return combos[0] if combos else {}
    if stable:
        stable.sort(
            key=lambda row: (
                -float(_float(row.get("mean_abs_probability_delta")) or 0.0),
                abs(_float(row.get("post_update_approx_kl_mean")) or 0.0),
            )
        )
        best = stable[0]
        return {**best, "recommendation_type": f"diagnostic_baseline_for_{route}"}
    return {}


def _write_recommended_config(config: dict[str, Any], output_root: Path, recommended: dict[str, Any]) -> Path | None:
    if not recommended:
        return None
    stage21_4 = _read_json(Path(config["stage21_4_base_config"]))
    stage21_4.update(
        {
            "epochs": int(recommended["epochs"]),
            "learning_rate": float(recommended["learning_rate"]),
            "clip_ratio": float(recommended["clip_ratio"]),
            "max_grad_norm": float(config["stage21_4_max_grad_norm"]),
            "loss_scale": float(recommended.get("loss_scale", config["loss_scale"])),
            "value_loss_coefficient": float(config["value_loss_coefficient"]),
            "advantage_clip_abs": float(config["advantage_clip_abs"]),
            "normalize_minibatch_advantages": bool(config["normalize_minibatch_advantages"]),
        }
    )
    stage21_4_path = output_root / "xunce-stage21-16-recommended-stage21-4-config.json"
    _write_json(stage21_4_path, stage21_4)
    stage21_6 = _read_json(Path(config["stage21_6_base_config"]))
    stage21_6.update(
        {
            "stage21_4_base_config": str(stage21_4_path),
            "seed_list": config["seed_list"],
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "max_grad_norm": float(config["stage21_6_pre_clip_grad_norm_gate"]),
            "stage21_6_authorized": False,
            "training_or_release_authorized": False,
            "runs_new_ppo_update": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    path = output_root / RECOMMENDED_CONFIG_FILE
    _write_json(path, stage21_6)
    return path


def _write_outputs(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    combos: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    update_audit: dict[str, Any],
    separation_audit: dict[str, Any],
    margin_audit: dict[str, Any],
    gradient_audit: dict[str, Any],
    recommended: dict[str, Any],
    recommended_config: Path | None,
    status: str,
    route: str,
    primary_reason: str,
    input_reasons: list[str],
    boundary_reasons: list[str],
) -> dict[str, Any]:
    _write_jsonl(output_root / SWEEP_RESULTS_FILE, rows)
    _write_json(output_root / UPDATE_AUDIT_FILE, update_audit)
    _write_json(output_root / SEPARATION_AUDIT_FILE, separation_audit)
    _write_json(output_root / MARGIN_AUDIT_FILE, margin_audit)
    _write_json(output_root / GRADIENT_AUDIT_FILE, gradient_audit)
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "primary_reason": primary_reason,
        "stage21_authorized": False,
        "training_or_release_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    _write_json(output_root / ROUTING_FILE, routing)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "next_required_change": route,
        "primary_reason": primary_reason,
        "stage21_15_root": config["stage21_15_root"],
        "planned_combo_count": len(combos),
        "completed_stage21_16_sweep_combo_count": sum(1 for row in rows if not row.get("runtime_blocked")),
        "input_reason_codes": input_reasons,
        "boundary_reason_codes": boundary_reasons,
        "strong_state_join_available_count": update_audit["strong_state_join_available_count"],
        "strong_state_binding_unavailable_count": update_audit["strong_state_binding_unavailable_count"],
        "mean_abs_probability_delta_max": update_audit["mean_abs_probability_delta_max"],
        "best_coverage_per_cost_probability_delta_max": update_audit["best_coverage_per_cost_probability_delta_max"],
        "argmax_changed_count": update_audit["argmax_changed_count"],
        "selected_rank_changed_count": update_audit["selected_rank_changed_count"],
        "selected_action_changed_count": update_audit["selected_action_changed_count"],
        "best_candidate_probability_margin_mean": margin_audit["probability_margin_mean"],
        "best_candidate_logit_margin_mean": margin_audit["logit_margin_mean"],
        "best_candidate_rank_gap_mean": margin_audit["best_candidate_rank_gap_mean"],
        "reward_coverage_per_cost_correlation": separation_audit["reward_coverage_per_cost_correlation"],
        "advantage_coverage_per_cost_correlation": separation_audit["advantage_coverage_per_cost_correlation"],
        "policy_grad_ratio_mean": gradient_audit["policy_grad_ratio_mean"],
        "value_to_policy_grad_norm_ratio_max": gradient_audit["value_to_policy_grad_norm_ratio_max"],
        "recommendation_type": recommended.get("recommendation_type"),
        "recommended_stage21_6_config": str(recommended_config) if recommended_config else None,
        "stage21_16_authorized": False,
        "training_or_release_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "summary": str(output_root / SUMMARY_FILE),
        "manifest": str(output_root / MANIFEST_FILE),
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "config": str(config_path),
        "output_root": str(output_root),
        "artifacts": {
            "summary": str(output_root / SUMMARY_FILE),
            "update_strength_audit": str(output_root / UPDATE_AUDIT_FILE),
            "reward_advantage_separation_audit": str(output_root / SEPARATION_AUDIT_FILE),
            "discrete_action_margin_audit": str(output_root / MARGIN_AUDIT_FILE),
            "loss_gradient_attribution": str(output_root / GRADIENT_AUDIT_FILE),
            "sweep_results": str(output_root / SWEEP_RESULTS_FILE),
            "recommended_stage21_6_config": str(recommended_config) if recommended_config else None,
            "routing": str(output_root / ROUTING_FILE),
            "report": str(output_root / REPORT_FILE),
        },
        "boundary": {
            "stage21_16_authorized": False,
            "training_or_release_authorized": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    }
    _write_json(output_root / MANIFEST_FILE, manifest)
    _write_report(output_root / REPORT_FILE, summary)
    return summary


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage21.16 Policy Signal Margin And Credit Attribution",
        "",
        "Stage21.16 separates four possible reasons why PPO did not change selected actions: update strength, reward/advantage separation, discrete action margin, and policy/value/entropy gradient balance.",
        "",
        f"- status: `{summary['status']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- primary_reason: `{summary['primary_reason']}`",
        f"- strong_state_join_available_count: `{summary['strong_state_join_available_count']}`",
        f"- mean_abs_probability_delta_max: `{summary['mean_abs_probability_delta_max']}`",
        f"- best_coverage_per_cost_probability_delta_max: `{summary['best_coverage_per_cost_probability_delta_max']}`",
        f"- best_candidate_probability_margin_mean: `{summary['best_candidate_probability_margin_mean']}`",
        f"- best_candidate_logit_margin_mean: `{summary['best_candidate_logit_margin_mean']}`",
        f"- reward_coverage_per_cost_correlation: `{summary['reward_coverage_per_cost_correlation']}`",
        f"- advantage_coverage_per_cost_correlation: `{summary['advantage_coverage_per_cost_correlation']}`",
        f"- policy_grad_ratio_mean: `{summary['policy_grad_ratio_mean']}`",
        f"- value_to_policy_grad_norm_ratio_max: `{summary['value_to_policy_grad_norm_ratio_max']}`",
        "",
        "This is bounded offline calibration evidence only. It does not publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _runtime_blocker_row(combo: dict[str, Any], stage21_6_root: Path, blocker: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SWEEP_ROW_SCHEMA_VERSION,
        "combo_id": combo["combo_id"],
        "combo_index": combo["combo_index"],
        "stage21_6_root": str(stage21_6_root),
        "epochs": combo["epochs"],
        "learning_rate": combo["learning_rate"],
        "clip_ratio": combo["clip_ratio"],
        "loss_scale": combo["loss_scale"],
        "status": "blocked",
        "runtime_blocked": True,
        "reason_codes": [blocker.get("reason_code", ROUTE_RUNTIME)],
        "numerically_stable": False,
        "coverage_auc_improved": False,
        "strong_state_join_available_count": 0,
        "mean_abs_probability_delta": None,
    }


def _capped_aggregate(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = []
    for row in seed_rows:
        stage21_5_root = Path(str(row.get("stage21_5_root", "")))
        summaries.append(_read_json_or_empty(stage21_5_root / "xunce-stage21-5-post-update-evaluation-summary.json"))
    return {
        **_distribution("final_coverage_rate_capped_delta", [_float(item.get("final_coverage_rate_capped_delta")) for item in summaries]),
        **_distribution("coverage_curve_auc_capped_delta", [_float(item.get("coverage_curve_auc_capped_delta")) for item in summaries]),
    }


def _combo_root(config: dict[str, Any], output_root: Path, combo: dict[str, Any]) -> Path:
    base = Path(str(config.get("sweep_work_root") or (output_root / "sweep_runs")))
    return base / str(combo["combo_id"])


def _rank_of(values: list[float], index: int) -> int:
    ordered = sorted(range(len(values)), key=lambda item: values[item], reverse=True)
    return ordered.index(index) + 1 if index in ordered else len(values) + 1


def _logits(row: dict[str, Any]) -> list[float]:
    values = row.get("logits")
    if not isinstance(values, list):
        detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
        values = detail.get("logits")
    return [float(value) for value in values or [] if _float(value) is not None]


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if _float(x) is not None and _float(y) is not None]
    if len(pairs) < 2:
        return None
    mx = mean([x for x, _ in pairs])
    my = mean([y for _, y in pairs])
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def _mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return mean(finite) if finite else None


def _max(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return max(finite) if finite else None


def _min(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return min(finite) if finite else None


def _clip_for_id(value: float) -> str:
    return str(value).replace(".", "p")


def _loss_for_id(value: float) -> str:
    return str(value).replace(".", "p")


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    for field in ("stage21_15_root", "stage21_6_base_config", "stage21_4_base_config"):
        if field not in config:
            raise ConfigError(f"missing required config field: {field}")
        config[field] = str(_resolve_path(Path(str(config[field])), repo_root))
    config["execute_sweep"] = bool(config.get("execute_sweep", True))
    config["sweep_work_root"] = str(_resolve_path(Path(str(config.get("sweep_work_root") or "stage21_16_sweep_runs_v1")), repo_root))
    config["seed_list"] = [int(value) for value in config.get("seed_list", [2101, 2102, 2103])]
    for field in (
        "required_scenario_count",
        "rollout_steps",
        "dynamic_max_candidates_per_step",
        "dynamic_proposal_pool_limit_per_step",
        "max_new_combinations_to_execute",
        "per_combo_timeout_seconds",
    ):
        config[field] = _positive_int(config.get(field), field)
    for field in (
        "stage21_6_pre_clip_grad_norm_gate",
        "stage21_4_max_grad_norm",
        "loss_scale",
        "value_loss_coefficient",
        "advantage_clip_abs",
        "probability_delta_threshold",
        "large_probability_margin_threshold",
        "min_reward_cpc_correlation",
        "min_advantage_cpc_correlation",
        "min_policy_grad_ratio",
        "max_value_to_policy_grad_ratio",
    ):
        config[field] = _nonnegative_float(config.get(field), field)
    for field in ("stage21_6_pre_clip_grad_norm_gate", "stage21_4_max_grad_norm", "loss_scale", "max_value_to_policy_grad_ratio"):
        if config[field] <= 0.0:
            raise ConfigError(f"{field} must be positive")
    config["normalize_minibatch_advantages"] = bool(config.get("normalize_minibatch_advantages", True))
    config["canary_traffic_fraction"] = _nonnegative_float(config.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return config


def _nonnegative_float(value: Any, field: str) -> float:
    parsed = _float(value)
    if parsed is None or parsed < 0.0:
        raise ConfigError(f"{field} must be a non-negative finite float")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
