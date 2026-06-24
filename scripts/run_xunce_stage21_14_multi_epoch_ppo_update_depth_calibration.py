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


CONFIG_SCHEMA_VERSION = "xunce-stage21-14-multi-epoch-ppo-update-depth-calibration-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage21-14-summary/v1"
SWEEP_ROW_SCHEMA_VERSION = "xunce-stage21-14-sweep-result/v1"
ACTION_AUDIT_SCHEMA_VERSION = "xunce-stage21-14-action-rank-shift-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage21-14-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage21-14-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage21_14_multi_epoch_ppo_update_depth_calibration_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/outputs/"
    "path_feedback_batch_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration_v1"
)

SUMMARY_FILE = "xunce-stage21-14-summary.json"
SWEEP_RESULTS_FILE = "xunce-stage21-14-sweep-results.jsonl"
ACTION_AUDIT_FILE = "xunce-stage21-14-action-rank-shift-audit.json"
RECOMMENDED_CONFIG_FILE = "xunce-stage21-14-recommended-stage21-6-config.json"
ROUTING_FILE = "xunce-stage21-14-next-stage-routing.json"
REPORT_FILE = "xunce-stage21-14-report.md"
MANIFEST_FILE = "xunce-stage21-14-manifest.json"

ROUTE_INPUTS = "rerun_stage21_14_required_inputs"
ROUTE_BOUNDARY = "resolve_stage21_14_boundary_rejections"
ROUTE_NUMERICS = "continue_stage21_9_gradient_normalization_loss_scaling_repair"
ROUTE_POLICY_SIGNAL = "calibrate_stage21_policy_update_signal_strength"
ROUTE_CREDIT = "repair_stage21_return_advantage_credit_assignment"
ROUTE_SCALE = "scale_stage21_ppo_pilot_scenarios_and_horizon"
ROUTE_RUNTIME = "stage21_14_calibration_runtime_budget_blocked"

BOUNDARY_FIELDS = (
    "stage21_14_authorized",
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
    parser = argparse.ArgumentParser(description="Run Stage 21.14 multi-epoch PPO update depth calibration.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        summary = run_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration(
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
                "completed_stage21_14_sweep_combo_count": summary["completed_stage21_14_sweep_combo_count"],
                "observable_action_shift_combo_count": summary["observable_action_shift_combo_count"],
                "best_combo_id": summary.get("best_combo_id"),
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] in {"passed", "partial"} else 1


def run_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration(
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

    action_audit = _action_rank_shift_audit(rows)
    recommended = _select_recommendation(rows)
    recommended_config = _write_recommended_config(config, output_root, recommended)
    status, route, primary_reason = _route(
        input_reasons=input_reasons,
        boundary_reasons=boundary_reasons,
        rows=rows,
        recommended=recommended,
        required_core_combo_count=int(config["required_core_combo_count"]),
    )
    return _write_outputs(
        config=config,
        config_path=config_path,
        output_root=output_root,
        combos=combos,
        rows=rows,
        action_audit=action_audit,
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
    stage21_13_summary = _read_json_or_empty(Path(config["stage21_13_root"]) / "xunce-stage21-13-summary.json")
    if not stage21_13_summary:
        reasons.append("missing_stage21_13_summary")
    else:
        if stage21_13_summary.get("status") != "failed":
            reasons.append("stage21_13_status_not_failed")
        if stage21_13_summary.get("next_required_change") != ROUTE_CREDIT:
            reasons.append("stage21_13_route_not_return_advantage_credit_assignment")
        if _int(stage21_13_summary.get("trainable_transition_count_total")) < 200:
            reasons.append("stage21_13_trainable_transition_count_below_200")
    stage21_12_summary = _read_json_or_empty(Path(config["stage21_12_root"]) / "xunce-stage21-12-summary.json")
    if not stage21_12_summary:
        reasons.append("missing_stage21_12_summary")
    elif stage21_12_summary.get("status") != "passed":
        reasons.append("stage21_12_not_passed")
    for field in ("stage21_6_base_config", "stage21_4_base_config"):
        if not Path(config[field]).is_file():
            reasons.append(f"missing_{field}")
    return sorted(set(reasons))


def _boundary_reasons(config: dict[str, Any]) -> list[str]:
    reasons = [f"stage21_14_config_{field}_true" for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) > 0.0:
        reasons.append("stage21_14_config_canary_traffic_fraction_nonzero")
    return sorted(set(reasons))


def _build_combos(config: dict[str, Any]) -> list[dict[str, Any]]:
    configured = config.get("priority_combinations")
    if not isinstance(configured, list) or not configured:
        raise ConfigError("priority_combinations must be a non-empty list")
    combos = []
    for index, combo in enumerate(configured):
        epochs = _positive_int(combo.get("epochs"), "epochs")
        learning_rate = _positive_float(combo.get("learning_rate"), "learning_rate")
        clip_ratio = _positive_float(combo.get("clip_ratio"), "clip_ratio")
        combo_id = str(combo.get("combo_id") or f"e{epochs}_lr{learning_rate:g}_c{clip_ratio:g}")
        combos.append(
            {
                "combo_index": index,
                "combo_id": combo_id,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "clip_ratio": clip_ratio,
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
            drift_reasons = _combo_config_drift_reasons(config, combo, combo_root)
            row["combo_config_drift_reason_codes"] = drift_reasons
            row["numerically_stable"] = bool(row.get("numerically_stable")) and not drift_reasons
            if drift_reasons and config.get("execute_sweep"):
                continue
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
    required_core_combo_count = int(config["required_core_combo_count"])
    core_combo_ids = {combo["combo_id"] for combo in combos[:required_core_combo_count]}
    core_done = done & core_combo_ids
    if len(core_done) >= required_core_combo_count:
        return
    executed = 0
    for combo in combos:
        if combo["combo_id"] not in core_combo_ids:
            continue
        if combo["combo_id"] in done:
            continue
        if executed >= int(config["max_new_combinations_to_execute"]):
            return
        _run_combo(config, output_root, combo, repo_root)
        executed += 1
        core_done.add(combo["combo_id"])
        if len(core_done) >= required_core_combo_count:
            return


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
            "loss_scale": float(config["loss_scale"]),
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
            "stage21_14_generated": True,
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
                "reason_code": "stage21_14_calibration_subprocess_failed",
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
    shift = _combo_action_shift(seed_rows)
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
        "stage21_4_max_grad_norm": combo["stage21_4_max_grad_norm"],
        "stage21_6_pre_clip_grad_norm_gate": combo["stage21_6_pre_clip_grad_norm_gate"],
        "status": summary.get("status"),
        "stage21_6_next_required_change": summary.get("next_required_change"),
        "boundary_reason_codes": boundary_reasons,
        "reason_codes": list(summary.get("reason_codes") or []),
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
        **shift,
    }


def _combo_config_drift_reasons(config: dict[str, Any], combo: dict[str, Any], combo_root: Path) -> list[str]:
    reasons: list[str] = []
    stage21_4_path = combo_root / "stage21_4_config.json"
    stage21_6_path = combo_root / "stage21_6_config.json"
    if not stage21_4_path.exists() and not stage21_6_path.exists():
        return []
    stage21_4 = _read_json_or_empty(stage21_4_path)
    stage21_6 = _read_json_or_empty(stage21_6_path)
    if not stage21_4:
        reasons.append("missing_generated_stage21_4_config")
    else:
        expected_stage21_4 = {
            "epochs": int(combo["epochs"]),
            "learning_rate": float(combo["learning_rate"]),
            "clip_ratio": float(combo["clip_ratio"]),
            "max_grad_norm": float(config["stage21_4_max_grad_norm"]),
            "loss_scale": float(config["loss_scale"]),
            "value_loss_coefficient": float(config["value_loss_coefficient"]),
            "advantage_clip_abs": float(config["advantage_clip_abs"]),
        }
        for field, expected in expected_stage21_4.items():
            if not _same_scalar(stage21_4.get(field), expected):
                reasons.append(f"generated_stage21_4_{field}_drift")
        if stage21_4.get("normalize_minibatch_advantages") is not bool(config["normalize_minibatch_advantages"]):
            reasons.append("generated_stage21_4_normalize_minibatch_advantages_drift")
    if not stage21_6:
        reasons.append("missing_generated_stage21_6_config")
    else:
        expected_stage21_6 = {
            "stage21_4_base_config": str(stage21_4_path),
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "max_grad_norm": float(config["stage21_6_pre_clip_grad_norm_gate"]),
        }
        for field, expected in expected_stage21_6.items():
            if not _same_scalar(stage21_6.get(field), expected):
                reasons.append(f"generated_stage21_6_{field}_drift")
        if stage21_6.get("seed_list") != config["seed_list"]:
            reasons.append("generated_stage21_6_seed_list_drift")
        if stage21_6.get("runs_new_ppo_update") is not True:
            reasons.append("generated_stage21_6_runs_new_ppo_update_not_true")
        for field in ("stage21_6_authorized", "training_or_release_authorized", "publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary"):
            if stage21_6.get(field) is True:
                reasons.append(f"generated_stage21_6_{field}_true")
        if float(stage21_6.get("canary_traffic_fraction", 0.0) or 0.0) > 0.0:
            reasons.append("generated_stage21_6_canary_traffic_fraction_nonzero")
    return sorted(set(reasons))


def _combo_boundary_reasons(summary: dict[str, Any], aggregate: dict[str, Any], seed_rows: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for field in ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary"):
        if summary.get(field) is True:
            reasons.append(f"stage21_6_summary_{field}_true")
    if float(summary.get("canary_traffic_fraction", 0.0) or 0.0) > 0.0:
        reasons.append("stage21_6_summary_canary_traffic_fraction_nonzero")
    for field in AGGREGATE_HARD_FAIL_FIELDS:
        if _int(aggregate.get(field)) > 0:
            reasons.append(f"stage21_6_{field}_nonzero")
    for row in seed_rows:
        seed = row.get("seed")
        for field in ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary", "training_or_release_authorized"):
            if row.get(field) is True:
                reasons.append(f"seed_{seed}_{field}_true")
    return sorted(set(reasons))


def _combo_action_shift(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = 0
    strong_unavailable = 0
    duplicate_strong_key_count = 0
    argmax_changed = 0
    selected_action_changed = 0
    selected_rank_changed = 0
    mean_abs_deltas: list[float] = []
    selected_deltas: list[float] = []
    best_deltas: list[float] = []
    for row in seed_rows:
        stage21_5_root = Path(str(row.get("stage21_5_root", "")))
        summary5 = _read_json_or_empty(stage21_5_root / "xunce-stage21-5-post-update-evaluation-summary.json")
        pre_root = Path(str(summary5.get("pre_evaluation_root") or stage21_5_root / "pre_ppo_xunce"))
        post_root = Path(str(summary5.get("post_evaluation_root") or stage21_5_root / "post_ppo_xunce"))
        pre_rows = [item for item in _read_jsonl(pre_root / "xunce-exploration-coverage-model-inference.jsonl") if _is_xunce_row(item)]
        post_rows = [item for item in _read_jsonl(post_root / "xunce-exploration-coverage-model-inference.jsonl") if _is_xunce_row(item)]
        candidate_rows = _read_jsonl(pre_root / "xunce-exploration-coverage-candidate-metric-audit.jsonl")
        best_by_key = _best_candidate_by_key(candidate_rows)
        post_by_strong, post_duplicates = _unique_strong_map(post_rows)
        pre_by_strong, pre_duplicates = _unique_strong_map(pre_rows)
        duplicate_strong_key_count += post_duplicates + pre_duplicates
        if post_duplicates or pre_duplicates:
            strong_unavailable += len(pre_rows)
            continue
        for pre in pre_rows:
            key = _strong_key(pre)
            if key is None:
                strong_unavailable += 1
                continue
            post = post_by_strong.get(key)
            if post is None:
                strong_unavailable += 1
                continue
            if pre_by_strong.get(key) is not pre:
                strong_unavailable += 1
                continue
            matched += 1
            pre_detail = pre.get("detail") if isinstance(pre.get("detail"), dict) else {}
            post_detail = post.get("detail") if isinstance(post.get("detail"), dict) else {}
            pre_probs = [_float(value) for value in pre_detail.get("action_probs", [])]
            post_probs = [_float(value) for value in post_detail.get("action_probs", [])]
            if len(pre_probs) == len(post_probs) and pre_probs:
                clean_deltas = [abs(float(b) - float(a)) for a, b in zip(pre_probs, post_probs) if a is not None and b is not None]
                if clean_deltas:
                    mean_abs_deltas.append(mean(clean_deltas))
                selected = _int(pre_detail.get("selected_action_index", pre.get("selected_action_index", -1)))
                selected_delta = _prob_delta(pre_probs, post_probs, selected)
                if selected_delta is not None:
                    selected_deltas.append(abs(selected_delta))
                best = best_by_key.get(key)
                if best is not None:
                    best_delta = _prob_delta(pre_probs, post_probs, best)
                    if best_delta is not None:
                        best_deltas.append(best_delta)
                if _argmax(pre_probs) != _argmax(post_probs):
                    argmax_changed += 1
            if _int(pre_detail.get("selected_action_index", pre.get("selected_action_index", -1))) != _int(
                post_detail.get("selected_action_index", post.get("selected_action_index", -1))
            ):
                selected_action_changed += 1
            if pre_detail.get("selected_rank") != post_detail.get("selected_rank"):
                selected_rank_changed += 1
    observable = bool(
        matched > 0
        and duplicate_strong_key_count == 0
        and (argmax_changed > 0 or selected_action_changed > 0 or selected_rank_changed > 0 or (_mean(mean_abs_deltas) or 0.0) >= 0.005)
    )
    return {
        "policy_shift_matched_step_count": matched,
        "strong_state_binding_unavailable_count": strong_unavailable,
        "duplicate_strong_state_key_count": duplicate_strong_key_count,
        "strong_state_join_available": strong_unavailable == 0 and duplicate_strong_key_count == 0 and matched > 0,
        "argmax_changed_count": argmax_changed,
        "selected_action_changed_count": selected_action_changed,
        "selected_rank_changed_count": selected_rank_changed,
        "mean_abs_probability_delta": _mean(mean_abs_deltas),
        "selected_probability_abs_delta_mean": _mean(selected_deltas),
        "best_coverage_per_cost_probability_delta_mean": _mean(best_deltas),
        "action_rank_shift_observable": observable,
    }


def _best_candidate_by_key(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], int]:
    best: dict[tuple[Any, ...], tuple[int, float]] = {}
    for row in rows:
        if row.get("action_mask_valid") is False:
            continue
        key = _strong_key(row)
        if key is None:
            continue
        score = _float(row.get("coverage_gain_per_path_cost"))
        index = _int(row.get("candidate_index"))
        if score is None:
            continue
        if key not in best or score > best[key][1]:
            best[key] = (index, score)
    return {key: value[0] for key, value in best.items()}


def _unique_strong_map(rows: list[dict[str, Any]]) -> tuple[dict[tuple[Any, ...], dict[str, Any]], int]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    duplicates = 0
    for row in rows:
        key = _strong_key(row)
        if key is None:
            continue
        if key in result:
            duplicates += 1
            continue
        result[key] = row
    return result, duplicates


def _is_xunce_row(row: dict[str, Any]) -> bool:
    policy = row.get("policy")
    return policy in (None, "", "xunce")


def _capped_aggregate(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = []
    for row in seed_rows:
        stage21_5_root = Path(str(row.get("stage21_5_root", "")))
        summaries.append(_read_json_or_empty(stage21_5_root / "xunce-stage21-5-post-update-evaluation-summary.json"))
    return {
        **_distribution("final_coverage_rate_capped_delta", [_float(item.get("final_coverage_rate_capped_delta")) for item in summaries]),
        **_distribution("coverage_curve_auc_capped_delta", [_float(item.get("coverage_curve_auc_capped_delta")) for item in summaries]),
    }


def _runtime_blocker_row(combo: dict[str, Any], stage21_6_root: Path, blocker: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SWEEP_ROW_SCHEMA_VERSION,
        "combo_id": combo["combo_id"],
        "combo_index": combo["combo_index"],
        "stage21_6_root": str(stage21_6_root),
        "epochs": combo["epochs"],
        "learning_rate": combo["learning_rate"],
        "clip_ratio": combo["clip_ratio"],
        "status": "blocked",
        "runtime_blocked": True,
        "reason_codes": [blocker.get("reason_code", ROUTE_RUNTIME)],
        "numerically_stable": False,
        "coverage_auc_improved": False,
        "action_rank_shift_observable": False,
    }


def _select_recommendation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row.get("numerically_stable")
        and not row.get("runtime_blocked")
        and not row.get("boundary_reason_codes")
        and row.get("coverage_auc_not_regressed")
    ]
    if not candidates:
        return {}
    candidates.sort(
        key=lambda row: (
            not bool(row.get("coverage_auc_improved")),
            not bool(row.get("action_rank_shift_observable")),
            abs(_float(row.get("post_update_approx_kl_mean")) or 0.0),
            int(row.get("epochs") or 999),
        )
    )
    return candidates[0]


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
            "loss_scale": float(config["loss_scale"]),
            "value_loss_coefficient": float(config["value_loss_coefficient"]),
            "advantage_clip_abs": float(config["advantage_clip_abs"]),
            "normalize_minibatch_advantages": bool(config["normalize_minibatch_advantages"]),
        }
    )
    stage21_4_path = output_root / "xunce-stage21-14-recommended-stage21-4-config.json"
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


def _action_rank_shift_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": ACTION_AUDIT_SCHEMA_VERSION,
        "completed_combo_count": len(rows),
        "observable_action_shift_combo_count": sum(1 for row in rows if row.get("action_rank_shift_observable")),
        "max_argmax_changed_count": max((_int(row.get("argmax_changed_count")) for row in rows), default=0),
        "max_selected_action_changed_count": max((_int(row.get("selected_action_changed_count")) for row in rows), default=0),
        "max_selected_rank_changed_count": max((_int(row.get("selected_rank_changed_count")) for row in rows), default=0),
        "max_mean_abs_probability_delta": max((_float(row.get("mean_abs_probability_delta")) or 0.0 for row in rows), default=0.0),
        "strong_state_binding_unavailable_combo_count": sum(1 for row in rows if _int(row.get("strong_state_binding_unavailable_count")) > 0),
        "duplicate_strong_state_key_combo_count": sum(1 for row in rows if _int(row.get("duplicate_strong_state_key_count")) > 0),
    }


def _route(
    *,
    input_reasons: list[str],
    boundary_reasons: list[str],
    rows: list[dict[str, Any]],
    recommended: dict[str, Any],
    required_core_combo_count: int,
) -> tuple[str, str, str]:
    if input_reasons:
        return "failed", ROUTE_INPUTS, "stage21_14_input_rejected"
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "stage21_14_boundary_rejected"
    if not rows:
        return "partial", ROUTE_RUNTIME, "no_completed_stage21_14_sweep_results"
    completed = [row for row in rows if not row.get("runtime_blocked")]
    if not completed:
        return "partial", ROUTE_RUNTIME, "stage21_14_sweep_runtime_blocked"
    core_completed = [row for row in completed if _int(row.get("combo_index")) < required_core_combo_count]
    if len(core_completed) < required_core_combo_count:
        core_runtime = [
            row
            for row in rows
            if row.get("runtime_blocked") and _int(row.get("combo_index")) < required_core_combo_count
        ]
        if core_runtime:
            return "partial", ROUTE_RUNTIME, "required_core_stage21_14_sweep_runtime_blocked"
        return "partial", ROUTE_RUNTIME, "required_core_stage21_14_sweep_incomplete"
    stable = [row for row in completed if row.get("numerically_stable")]
    if not stable:
        return "failed", ROUTE_NUMERICS, "no_numerically_stable_combo"
    improved = [row for row in stable if row.get("coverage_auc_improved")]
    if improved and recommended:
        return "passed", ROUTE_SCALE, "multi_epoch_smoke_improved"
    shifted = [row for row in stable if row.get("action_rank_shift_observable")]
    if not shifted:
        if all(_int(row.get("policy_shift_matched_step_count")) == 0 and _int(row.get("strong_state_binding_unavailable_count")) > 0 for row in stable):
            return "failed", ROUTE_POLICY_SIGNAL, "strong_state_binding_unavailable_for_action_shift_audit"
        return "failed", ROUTE_POLICY_SIGNAL, "stable_but_action_shift_too_small"
    return "failed", ROUTE_CREDIT, "action_shift_without_trajectory_improvement"


def _write_outputs(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    combos: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    action_audit: dict[str, Any],
    recommended: dict[str, Any],
    recommended_config: Path | None,
    status: str,
    route: str,
    primary_reason: str,
    input_reasons: list[str],
    boundary_reasons: list[str],
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_jsonl(output_root / SWEEP_RESULTS_FILE, rows)
    _write_json(output_root / ACTION_AUDIT_FILE, action_audit)
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "primary_reason": primary_reason,
        "input_reason_codes": input_reasons,
        "boundary_reason_codes": boundary_reasons,
        "stage21_14_authorized": False,
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
        "generated_at": generated_at,
        "status": status,
        "next_required_change": route,
        "primary_reason": primary_reason,
        "input_reason_codes": input_reasons,
        "boundary_reason_codes": boundary_reasons,
        "planned_combo_count": len(combos),
        "completed_stage21_14_sweep_combo_count": sum(1 for row in rows if not row.get("runtime_blocked")),
        "runtime_blocked_stage21_14_sweep_combo_count": sum(1 for row in rows if row.get("runtime_blocked")),
        "observable_action_shift_combo_count": action_audit["observable_action_shift_combo_count"],
        "max_argmax_changed_count": action_audit["max_argmax_changed_count"],
        "max_selected_action_changed_count": action_audit["max_selected_action_changed_count"],
        "max_selected_rank_changed_count": action_audit["max_selected_rank_changed_count"],
        "max_mean_abs_probability_delta": action_audit["max_mean_abs_probability_delta"],
        "strong_state_binding_unavailable_combo_count": action_audit["strong_state_binding_unavailable_combo_count"],
        "best_combo_id": recommended.get("combo_id"),
        "recommended_stage21_6_config": str(recommended_config) if recommended_config else None,
        "stage21_13_root": str(config["stage21_13_root"]),
        "stage21_12_root": str(config["stage21_12_root"]),
        "stage21_14_authorized": False,
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
    (output_root / REPORT_FILE).write_text(_report(summary), encoding="utf-8")
    _write_json(
        output_root / MANIFEST_FILE,
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generated_at": generated_at,
            "config": str(config_path),
            "artifacts": {
                "summary": str(output_root / SUMMARY_FILE),
                "sweep_results": str(output_root / SWEEP_RESULTS_FILE),
                "action_rank_shift_audit": str(output_root / ACTION_AUDIT_FILE),
                "recommended_stage21_6_config": str(recommended_config) if recommended_config else None,
                "routing": str(output_root / ROUTING_FILE),
                "report": str(output_root / REPORT_FILE),
                "manifest": str(output_root / MANIFEST_FILE),
            },
            "summary_status": status,
            "next_required_change": route,
        },
    )
    return summary


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage 21.14 Multi-Epoch PPO Update Depth Calibration",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- primary_reason: `{summary['primary_reason']}`",
            f"- completed_stage21_14_sweep_combo_count: `{summary['completed_stage21_14_sweep_combo_count']}`",
            f"- observable_action_shift_combo_count: `{summary['observable_action_shift_combo_count']}`",
            f"- max_argmax_changed_count: `{summary['max_argmax_changed_count']}`",
            f"- max_selected_rank_changed_count: `{summary['max_selected_rank_changed_count']}`",
            f"- max_mean_abs_probability_delta: `{summary['max_mean_abs_probability_delta']}`",
            f"- best_combo_id: `{summary.get('best_combo_id')}`",
            "",
            "Stage21.14 is bounded offline calibration evidence only. It does not publish checkpoints, replace default policy, connect an executor, or start canary traffic.",
            "",
        ]
    )


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    for field in ("stage21_13_root", "stage21_12_root", "stage21_6_base_config", "stage21_4_base_config"):
        if field not in config:
            raise ConfigError(f"missing required config field: {field}")
        config[field] = str(_resolve_path(Path(str(config[field])), repo_root))
    config["execute_sweep"] = bool(config.get("execute_sweep", True))
    config["sweep_work_root"] = str(_resolve_path(Path(str(config.get("sweep_work_root") or "stage21_14_sweep_runs_v1")), repo_root))
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
    config["required_core_combo_count"] = _positive_int(
        config.get("required_core_combo_count", min(3, len(config.get("priority_combinations") or []))),
        "required_core_combo_count",
    )
    for field in (
        "stage21_6_pre_clip_grad_norm_gate",
        "stage21_4_max_grad_norm",
        "loss_scale",
        "value_loss_coefficient",
        "advantage_clip_abs",
    ):
        config[field] = _nonnegative_float(config.get(field), field)
    if config["stage21_6_pre_clip_grad_norm_gate"] <= 0.0 or config["stage21_4_max_grad_norm"] <= 0.0 or config["loss_scale"] <= 0.0:
        raise ConfigError("stage21_6_pre_clip_grad_norm_gate, stage21_4_max_grad_norm and loss_scale must be positive")
    config["normalize_minibatch_advantages"] = bool(config.get("normalize_minibatch_advantages", True))
    config["canary_traffic_fraction"] = _nonnegative_float(config.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return config


def _combo_root(config: dict[str, Any], output_root: Path, combo: dict[str, Any]) -> Path:
    base = Path(str(config.get("sweep_work_root") or (output_root / "sweep_runs")))
    return base / str(combo["combo_id"])


def _strong_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    required = ("scenario_id", "step_index", "current_cell", "covered_cells_hash", "candidate_set_hash")
    if any(row.get(field) in (None, "") for field in required):
        return None
    return (
        row.get("scenario_id"),
        row.get("step_index"),
        _cell_key(row.get("current_cell")),
        row.get("covered_cells_hash"),
        row.get("candidate_set_hash"),
    )


def _weak_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("scenario_id"), row.get("step_index"), row.get("candidate_set_hash"))


def _cell_key(value: Any) -> tuple[Any, ...]:
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, tuple):
        return value
    return (value,)


def _prob_delta(pre_probs: list[float | None], post_probs: list[float | None], index: int) -> float | None:
    if index < 0 or index >= len(pre_probs) or index >= len(post_probs):
        return None
    if pre_probs[index] is None or post_probs[index] is None:
        return None
    return float(post_probs[index]) - float(pre_probs[index])


def _argmax(values: list[float | None]) -> int | None:
    clean = [(index, value) for index, value in enumerate(values) if value is not None]
    if not clean:
        return None
    return max(clean, key=lambda item: item[1])[0]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _same_scalar(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        parsed = _float(actual)
        return parsed is not None and abs(parsed - expected) <= 1.0e-12
    return actual == expected


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    try:
        return _read_json(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _positive_int(value: Any, field: str) -> int:
    parsed = _int(value)
    if parsed <= 0:
        raise ConfigError(f"{field} must be positive")
    return parsed


def _positive_float(value: Any, field: str) -> float:
    parsed = _float(value)
    if parsed is None or parsed <= 0.0:
        raise ConfigError(f"{field} must be positive")
    return parsed


def _nonnegative_float(value: Any, field: str) -> float:
    parsed = _float(value)
    if parsed is None or parsed < 0.0:
        raise ConfigError(f"{field} must be nonnegative")
    return parsed


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return mean(clean) if clean else None


def _max(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return max(clean) if clean else None


def _distribution(prefix: str, values: list[float | None]) -> dict[str, float | None]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {f"{prefix}_mean": None, f"{prefix}_min": None, f"{prefix}_max": None}
    return {f"{prefix}_mean": mean(clean), f"{prefix}_min": min(clean), f"{prefix}_max": max(clean)}


if __name__ == "__main__":
    raise SystemExit(main())
