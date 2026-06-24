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

import scripts.run_xunce_stage21_16_policy_signal_margin_credit_attribution as s16
from scripts.run_xunce_stage21_15_policy_update_signal_strength_calibration import (
    _distribution,
    _float,
    _int,
    _positive_float,
    _positive_int,
    _read_json,
    _read_json_or_empty,
    _read_jsonl,
    _resolve_path,
    _write_json,
    _write_jsonl,
)


CONFIG_SCHEMA_VERSION = "xunce-stage21-17-policy-signal-amplification-value-balance-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage21-17-summary/v1"
SWEEP_ROW_SCHEMA_VERSION = "xunce-stage21-17-sweep-result/v1"
BALANCE_AUDIT_SCHEMA_VERSION = "xunce-stage21-17-policy-value-balance-audit/v1"
SIGNAL_AUDIT_SCHEMA_VERSION = "xunce-stage21-17-action-signal-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage21-17-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage21-17-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage21_17_policy_signal_amplification_value_balance_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/outputs/"
    "path_feedback_batch_xunce_stage21_17_policy_signal_amplification_value_balance_v1"
)

SUMMARY_FILE = "xunce-stage21-17-summary.json"
SWEEP_RESULTS_FILE = "xunce-stage21-17-sweep-results.jsonl"
BALANCE_AUDIT_FILE = "xunce-stage21-17-policy-value-balance-audit.json"
SIGNAL_AUDIT_FILE = "xunce-stage21-17-action-signal-audit.json"
RECOMMENDED_CONFIG_FILE = "xunce-stage21-17-recommended-stage21-6-config.json"
ROUTING_FILE = "xunce-stage21-17-next-stage-routing.json"
REPORT_FILE = "xunce-stage21-17-report.md"
MANIFEST_FILE = "xunce-stage21-17-manifest.json"

ROUTE_INPUTS = "rerun_stage21_17_required_inputs"
ROUTE_BOUNDARY = "resolve_stage21_17_boundary_rejections"
ROUTE_NUMERICS = "continue_stage21_9_gradient_normalization_loss_scaling_repair"
ROUTE_VALUE_BALANCE = "repair_stage21_policy_value_loss_balance"
ROUTE_POLICY_MULTIPLIER = "increase_stage21_policy_update_signal_strength_with_policy_loss_multiplier"
ROUTE_MARGIN = "calibrate_stage21_discrete_action_margin_crossing"
ROUTE_CREDIT = "repair_stage21_return_advantage_credit_assignment"
ROUTE_SCALE = "scale_stage21_ppo_pilot_scenarios_and_horizon"
ROUTE_BINDING = "repair_stage21_5_inference_binding_contract"
ROUTE_RUNTIME = "stage21_17_calibration_runtime_budget_blocked"

BOUNDARY_FIELDS = (
    "stage21_17_authorized",
    "training_or_release_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 21.17 policy signal amplification and value balance.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        summary = run_xunce_stage21_17_policy_signal_amplification_value_balance(
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
                "policy_grad_ratio_mean": summary["policy_grad_ratio_mean"],
                "value_to_policy_grad_norm_ratio_max": summary["value_to_policy_grad_norm_ratio_max"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] in {"passed", "partial"} else 1


def run_xunce_stage21_17_policy_signal_amplification_value_balance(
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

    balance_audit = _balance_audit(rows)
    signal_audit = _signal_audit(rows)
    status, route, primary_reason = _route(
        input_reasons=input_reasons,
        boundary_reasons=boundary_reasons,
        rows=rows,
        balance_audit=balance_audit,
        signal_audit=signal_audit,
        probability_delta_threshold=float(config["probability_delta_threshold"]),
        min_policy_grad_ratio=float(config["min_policy_grad_ratio"]),
        max_value_to_policy_grad_ratio=float(config["max_value_to_policy_grad_ratio"]),
    )
    recommended = _select_recommendation(config, combos, rows, route)
    best_balanced = _best_balanced_row(
        rows,
        min_policy_grad_ratio=float(config["min_policy_grad_ratio"]),
        max_value_to_policy_grad_ratio=float(config["max_value_to_policy_grad_ratio"]),
    )
    recommended_config = _write_recommended_config(config, output_root, recommended)
    return _write_outputs(
        config=config,
        config_path=config_path,
        output_root=output_root,
        combos=combos,
        rows=rows,
        balance_audit=balance_audit,
        signal_audit=signal_audit,
        best_balanced=best_balanced,
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
    summary = _read_json_or_empty(Path(config["stage21_16_root"]) / "xunce-stage21-16-summary.json")
    if not summary:
        reasons.append("missing_stage21_16_summary")
    else:
        if summary.get("next_required_change") not in {
            "increase_stage21_policy_update_signal_strength",
            "repair_stage21_policy_value_loss_balance",
        }:
            reasons.append("stage21_16_route_not_policy_signal_or_value_balance")
        if _int(summary.get("strong_state_join_available_count")) <= 0:
            reasons.append("stage21_16_no_strong_state_join")
        if _int(summary.get("strong_state_binding_unavailable_count")) > 0:
            reasons.append("stage21_16_strong_state_binding_incomplete")
    for field in ("stage21_6_base_config", "stage21_4_base_config"):
        if not Path(config[field]).is_file():
            reasons.append(f"missing_{field}")
    return sorted(set(reasons))


def _boundary_reasons(config: dict[str, Any]) -> list[str]:
    reasons = [f"stage21_17_config_{field}_true" for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) > 0.0:
        reasons.append("stage21_17_config_canary_traffic_fraction_nonzero")
    return sorted(set(reasons))


def _build_combos(config: dict[str, Any]) -> list[dict[str, Any]]:
    configured = config.get("priority_combinations")
    if not isinstance(configured, list) or not configured:
        raise ConfigError("priority_combinations must be a non-empty list")
    combos: list[dict[str, Any]] = []
    for index, combo in enumerate(configured):
        value_coef = _nonnegative_float(combo.get("value_loss_coefficient"), "value_loss_coefficient")
        policy_coef = _nonnegative_float(combo.get("policy_loss_coefficient", 1.0), "policy_loss_coefficient")
        if policy_coef <= 0.0:
            raise ConfigError("policy_loss_coefficient must be positive")
        loss_scale = _positive_float(combo.get("loss_scale", config["loss_scale"]), "loss_scale")
        epochs = _positive_int(combo.get("epochs"), "epochs")
        lr = _positive_float(combo.get("learning_rate"), "learning_rate")
        clip = _positive_float(combo.get("clip_ratio"), "clip_ratio")
        combos.append(
            {
                "combo_index": index,
                "combo_id": str(combo.get("combo_id") or f"v{_id_float(value_coef)}_p{_id_float(policy_coef)}_loss{_id_float(loss_scale)}"),
                "epochs": epochs,
                "learning_rate": lr,
                "clip_ratio": clip,
                "loss_scale": loss_scale,
                "value_loss_coefficient": value_coef,
                "policy_loss_coefficient": policy_coef,
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
    stage21_4 = _read_json(Path(config["stage21_4_base_config"]))
    stage21_4.update(
        {
            "epochs": int(combo["epochs"]),
            "learning_rate": float(combo["learning_rate"]),
            "clip_ratio": float(combo["clip_ratio"]),
            "max_grad_norm": float(config["stage21_4_max_grad_norm"]),
            "loss_scale": float(combo["loss_scale"]),
            "value_loss_coefficient": float(combo["value_loss_coefficient"]),
            "policy_loss_coefficient": float(combo["policy_loss_coefficient"]),
            "advantage_clip_abs": float(config["advantage_clip_abs"]),
            "normalize_minibatch_advantages": bool(config["normalize_minibatch_advantages"]),
        }
    )
    stage21_4_path = combo_root / "stage21_4_config.json"
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
            "execute_seed_pipeline": True,
            "stage21_6_authorized": False,
            "training_or_release_authorized": False,
            "runs_new_ppo_update": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
            "stage21_17_generated": True,
        }
    )
    stage21_6_path = combo_root / "stage21_6_config.json"
    _write_json(stage21_6_path, stage21_6)
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
    _write_json(combo_root / "stage21_6-subprocess-result.json", {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "cmd": cmd})
    if completed.returncode != 0:
        _write_json(
            combo_root / "runtime-blocker.json",
            {
                "status": "blocked",
                "reason_code": "stage21_17_calibration_subprocess_failed",
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
    boundary_reasons = s16._combo_boundary_reasons(summary, aggregate, seed_rows)
    signal = s16._combo_policy_signal(seed_rows)
    separation = s16._combo_reward_advantage_separation(seed_rows)
    gradient = s16._combo_gradient_attribution(seed_rows)
    capped = s16._capped_aggregate(seed_rows)
    pre_clip = _float(aggregate.get("pre_clip_grad_norm_max"))
    post_clip = s16._max([_float(row.get("post_clip_grad_norm")) for row in seed_rows])
    kl = _float(aggregate.get("max_post_update_approx_kl_mean"))
    entropy = _float(aggregate.get("min_entropy_mean"))
    numerically_stable = bool(
        not boundary_reasons
        and lineage.get("passed") is True
        and pre_clip is not None
        and pre_clip <= float(combo["stage21_6_pre_clip_grad_norm_gate"])
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
        "value_loss_coefficient": combo["value_loss_coefficient"],
        "policy_loss_coefficient": combo["policy_loss_coefficient"],
        "status": summary.get("status"),
        "stage21_6_next_required_change": summary.get("next_required_change"),
        "boundary_reason_codes": boundary_reasons,
        "lineage_passed": lineage.get("passed"),
        "trainable_transition_count_total": _int(aggregate.get("trainable_transition_count_total")),
        "pre_clip_grad_norm_max": pre_clip,
        "post_clip_grad_norm_max": post_clip,
        "post_update_approx_kl_mean": kl,
        "min_entropy_mean": entropy,
        "parameter_delta_l2_mean": s16._mean([_float(row.get("parameter_delta_l2")) for row in seed_rows]),
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


def _balance_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": BALANCE_AUDIT_SCHEMA_VERSION,
        "component_gradient_audit_available": any(row.get("component_gradient_audit_available") for row in rows),
        "policy_loss_grad_norm_mean": s16._mean([_float(row.get("policy_loss_grad_norm_mean")) for row in rows]),
        "value_loss_grad_norm_mean": s16._mean([_float(row.get("value_loss_grad_norm_mean")) for row in rows]),
        "entropy_loss_grad_norm_mean": s16._mean([_float(row.get("entropy_loss_grad_norm_mean")) for row in rows]),
        "total_loss_grad_norm_mean": s16._mean([_float(row.get("total_loss_grad_norm_mean")) for row in rows]),
        "policy_grad_ratio_mean": s16._mean([_float(row.get("policy_grad_ratio_mean")) for row in rows]),
        "policy_grad_ratio_min": s16._min([_float(row.get("policy_grad_ratio_min")) for row in rows]),
        "value_grad_ratio_mean": s16._mean([_float(row.get("value_grad_ratio_mean")) for row in rows]),
        "entropy_grad_ratio_mean": s16._mean([_float(row.get("entropy_grad_ratio_mean")) for row in rows]),
        "value_to_policy_grad_norm_ratio_mean": s16._mean([_float(row.get("value_to_policy_grad_norm_ratio_mean")) for row in rows]),
        "value_to_policy_grad_norm_ratio_max": s16._max([_float(row.get("value_to_policy_grad_norm_ratio_max")) for row in rows]),
    }


def _signal_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stable = [row for row in rows if row.get("numerically_stable") and not row.get("runtime_blocked")]
    return {
        "schema_version": SIGNAL_AUDIT_SCHEMA_VERSION,
        "completed_combo_count": len(rows),
        "stable_combo_count": len(stable),
        "mean_abs_probability_delta_max": max((_float(row.get("mean_abs_probability_delta")) or 0.0 for row in stable), default=0.0),
        "best_coverage_per_cost_probability_delta_max": max((_float(row.get("best_coverage_per_cost_probability_delta_mean")) or 0.0 for row in stable), default=0.0),
        "argmax_changed_count": sum(_int(row.get("argmax_changed_count")) for row in stable),
        "selected_rank_changed_count": sum(_int(row.get("selected_rank_changed_count")) for row in stable),
        "selected_action_changed_count": sum(_int(row.get("selected_action_changed_count")) for row in stable),
        "strong_state_join_available_count": sum(_int(row.get("strong_state_join_available_count")) for row in stable),
        "strong_state_binding_unavailable_count": sum(_int(row.get("strong_state_binding_unavailable_count")) for row in stable),
    }


def _route(
    *,
    input_reasons: list[str],
    boundary_reasons: list[str],
    rows: list[dict[str, Any]],
    balance_audit: dict[str, Any],
    signal_audit: dict[str, Any],
    probability_delta_threshold: float,
    min_policy_grad_ratio: float,
    max_value_to_policy_grad_ratio: float,
) -> tuple[str, str, str]:
    if input_reasons:
        return "failed", ROUTE_INPUTS, "stage21_17_input_rejected"
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "stage21_17_boundary_rejected"
    if not rows:
        return "partial", ROUTE_RUNTIME, "no_completed_stage21_17_sweep_results"
    completed = [row for row in rows if not row.get("runtime_blocked")]
    if not completed:
        return "partial", ROUTE_RUNTIME, "stage21_17_sweep_runtime_blocked"
    stable = [row for row in completed if row.get("numerically_stable")]
    if not stable:
        return "failed", ROUTE_NUMERICS, "no_numerically_stable_combo"
    if any(_int(row.get("strong_state_join_available_count")) == 0 or _int(row.get("strong_state_binding_unavailable_count")) > 0 for row in stable):
        return "failed", ROUTE_BINDING, "strong_state_binding_contract_incomplete"
    if any(row.get("coverage_auc_improved") for row in stable):
        return "passed", ROUTE_SCALE, "policy_signal_smoke_improved"

    balanced = _balanced_rows(
        stable,
        min_policy_grad_ratio=min_policy_grad_ratio,
        max_value_to_policy_grad_ratio=max_value_to_policy_grad_ratio,
    )
    if balance_audit.get("component_gradient_audit_available") and not balanced:
        return "failed", ROUTE_VALUE_BALANCE, "policy_value_gradient_balance_failed"

    evidence_rows = balanced or stable
    signal = max((_float(row.get("mean_abs_probability_delta")) or 0.0 for row in evidence_rows), default=0.0)
    if signal < probability_delta_threshold:
        return "failed", ROUTE_POLICY_MULTIPLIER, "probability_delta_below_threshold_after_value_balance"

    if any(_int(row.get("argmax_changed_count")) > 0 or _int(row.get("selected_rank_changed_count")) > 0 for row in evidence_rows):
        return "failed", ROUTE_CREDIT, "rank_or_argmax_changed_without_trajectory_gain"
    return "failed", ROUTE_MARGIN, "probability_moved_but_discrete_action_margin_not_crossed"


def _balanced_rows(
    rows: list[dict[str, Any]],
    *,
    min_policy_grad_ratio: float,
    max_value_to_policy_grad_ratio: float,
) -> list[dict[str, Any]]:
    balanced: list[dict[str, Any]] = []
    for row in rows:
        policy_ratio = _float(row.get("policy_grad_ratio_mean"))
        value_to_policy = _float(row.get("value_to_policy_grad_norm_ratio_max"))
        policy_ok = policy_ratio is not None and policy_ratio >= min_policy_grad_ratio
        value_ok = value_to_policy is None or value_to_policy <= max_value_to_policy_grad_ratio
        if policy_ok and value_ok:
            balanced.append(row)
    return balanced


def _best_balanced_row(
    rows: list[dict[str, Any]],
    *,
    min_policy_grad_ratio: float,
    max_value_to_policy_grad_ratio: float,
) -> dict[str, Any] | None:
    balanced = _balanced_rows(
        [row for row in rows if row.get("numerically_stable") and not row.get("runtime_blocked")],
        min_policy_grad_ratio=min_policy_grad_ratio,
        max_value_to_policy_grad_ratio=max_value_to_policy_grad_ratio,
    )
    if not balanced:
        return None
    balanced.sort(
        key=lambda row: (
            -float(_float(row.get("mean_abs_probability_delta")) or 0.0),
            abs(_float(row.get("post_update_approx_kl_mean")) or 0.0),
        )
    )
    return balanced[0]


def _select_recommendation(
    config: dict[str, Any],
    combos: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    route: str,
) -> dict[str, Any]:
    completed_ids = {str(row.get("combo_id")) for row in rows if not row.get("runtime_blocked")}
    if route in {ROUTE_VALUE_BALANCE, ROUTE_POLICY_MULTIPLIER, ROUTE_RUNTIME}:
        for combo in combos:
            if str(combo["combo_id"]) not in completed_ids:
                return {**combo, "recommendation_type": "next_unrun_stage21_17_combo"}
    stable = [
        row
        for row in rows
        if row.get("numerically_stable")
        and not row.get("runtime_blocked")
        and row.get("coverage_auc_not_regressed")
        and _int(row.get("strong_state_join_available_count")) > 0
        and _int(row.get("strong_state_binding_unavailable_count")) == 0
    ]
    if stable:
        stable.sort(
            key=lambda row: (
                float(_float(row.get("value_to_policy_grad_norm_ratio_mean")) or 999999.0),
                -float(_float(row.get("mean_abs_probability_delta")) or 0.0),
            )
        )
        return {**stable[0], "recommendation_type": f"diagnostic_baseline_for_{route}"}
    return combos[0] if combos else {}


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
            "value_loss_coefficient": float(recommended["value_loss_coefficient"]),
            "policy_loss_coefficient": float(recommended["policy_loss_coefficient"]),
            "advantage_clip_abs": float(config["advantage_clip_abs"]),
            "normalize_minibatch_advantages": bool(config["normalize_minibatch_advantages"]),
        }
    )
    stage21_4_path = output_root / "xunce-stage21-17-recommended-stage21-4-config.json"
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
    balance_audit: dict[str, Any],
    signal_audit: dict[str, Any],
    best_balanced: dict[str, Any] | None,
    recommended: dict[str, Any],
    recommended_config: Path | None,
    status: str,
    route: str,
    primary_reason: str,
    input_reasons: list[str],
    boundary_reasons: list[str],
) -> dict[str, Any]:
    _write_jsonl(output_root / SWEEP_RESULTS_FILE, rows)
    _write_json(output_root / BALANCE_AUDIT_FILE, balance_audit)
    _write_json(output_root / SIGNAL_AUDIT_FILE, signal_audit)
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "primary_reason": primary_reason,
        "stage21_17_authorized": False,
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
        "stage21_16_root": config["stage21_16_root"],
        "planned_combo_count": len(combos),
        "completed_stage21_17_sweep_combo_count": sum(1 for row in rows if not row.get("runtime_blocked")),
        "input_reason_codes": input_reasons,
        "boundary_reason_codes": boundary_reasons,
        "strong_state_join_available_count": signal_audit["strong_state_join_available_count"],
        "strong_state_binding_unavailable_count": signal_audit["strong_state_binding_unavailable_count"],
        "mean_abs_probability_delta_max": signal_audit["mean_abs_probability_delta_max"],
        "best_coverage_per_cost_probability_delta_max": signal_audit["best_coverage_per_cost_probability_delta_max"],
        "argmax_changed_count": signal_audit["argmax_changed_count"],
        "selected_rank_changed_count": signal_audit["selected_rank_changed_count"],
        "selected_action_changed_count": signal_audit["selected_action_changed_count"],
        "policy_grad_ratio_mean": balance_audit["policy_grad_ratio_mean"],
        "value_to_policy_grad_norm_ratio_max": balance_audit["value_to_policy_grad_norm_ratio_max"],
        "best_balanced_combo_id": best_balanced.get("combo_id") if best_balanced else None,
        "best_balanced_mean_abs_probability_delta": best_balanced.get("mean_abs_probability_delta") if best_balanced else None,
        "best_balanced_policy_grad_ratio_mean": best_balanced.get("policy_grad_ratio_mean") if best_balanced else None,
        "best_balanced_value_to_policy_grad_norm_ratio_max": best_balanced.get("value_to_policy_grad_norm_ratio_max") if best_balanced else None,
        "recommendation_type": recommended.get("recommendation_type"),
        "recommended_stage21_6_config": str(recommended_config) if recommended_config else None,
        "stage21_17_authorized": False,
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
            "sweep_results": str(output_root / SWEEP_RESULTS_FILE),
            "policy_value_balance_audit": str(output_root / BALANCE_AUDIT_FILE),
            "action_signal_audit": str(output_root / SIGNAL_AUDIT_FILE),
            "recommended_stage21_6_config": str(recommended_config) if recommended_config else None,
            "routing": str(output_root / ROUTING_FILE),
            "report": str(output_root / REPORT_FILE),
        },
        "boundary": {
            "stage21_17_authorized": False,
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
        "# Stage21.17 Policy Signal Amplification Value Balance",
        "",
        "Stage21.17 calibrates whether reducing value-loss pressure or increasing policy-loss pressure produces observable action-probability movement while keeping PPO numerically stable.",
        "",
        f"- status: `{summary['status']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- primary_reason: `{summary['primary_reason']}`",
        f"- strong_state_join_available_count: `{summary['strong_state_join_available_count']}`",
        f"- mean_abs_probability_delta_max: `{summary['mean_abs_probability_delta_max']}`",
        f"- best_coverage_per_cost_probability_delta_max: `{summary['best_coverage_per_cost_probability_delta_max']}`",
        f"- policy_grad_ratio_mean: `{summary['policy_grad_ratio_mean']}`",
        f"- value_to_policy_grad_norm_ratio_max: `{summary['value_to_policy_grad_norm_ratio_max']}`",
        f"- best_balanced_combo_id: `{summary['best_balanced_combo_id']}`",
        f"- best_balanced_mean_abs_probability_delta: `{summary['best_balanced_mean_abs_probability_delta']}`",
        f"- best_balanced_policy_grad_ratio_mean: `{summary['best_balanced_policy_grad_ratio_mean']}`",
        f"- best_balanced_value_to_policy_grad_norm_ratio_max: `{summary['best_balanced_value_to_policy_grad_norm_ratio_max']}`",
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
        "value_loss_coefficient": combo["value_loss_coefficient"],
        "policy_loss_coefficient": combo["policy_loss_coefficient"],
        "status": "blocked",
        "runtime_blocked": True,
        "reason_codes": [blocker.get("reason_code", ROUTE_RUNTIME)],
        "numerically_stable": False,
        "coverage_auc_improved": False,
        "strong_state_join_available_count": 0,
        "mean_abs_probability_delta": None,
    }


def _combo_root(config: dict[str, Any], output_root: Path, combo: dict[str, Any]) -> Path:
    base = Path(str(config.get("sweep_work_root") or (output_root / "sweep_runs")))
    return base / str(combo["combo_id"])


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    for field in ("stage21_16_root", "stage21_6_base_config", "stage21_4_base_config"):
        if field not in config:
            raise ConfigError(f"missing required config field: {field}")
        config[field] = str(_resolve_path(Path(str(config[field])), repo_root))
    config["execute_sweep"] = bool(config.get("execute_sweep", True))
    config["sweep_work_root"] = str(_resolve_path(Path(str(config.get("sweep_work_root") or "stage21_17_sweep_runs_v1")), repo_root))
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
        "advantage_clip_abs",
        "probability_delta_threshold",
        "large_probability_margin_threshold",
        "min_policy_grad_ratio",
        "max_value_to_policy_grad_ratio",
    ):
        default = 0.25 if field == "loss_scale" else None
        config[field] = _nonnegative_float(config.get(field, default), field)
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


def _id_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    raise SystemExit(main())
