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


CONFIG_SCHEMA_VERSION = "xunce-stage21-8-ppo-update-strength-calibration-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage21-8-calibration-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage21-8-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage21-8-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage21_8_ppo_update_strength_calibration_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/"
    "outputs/path_feedback_batch_xunce_stage21_8_ppo_update_strength_calibration_v1"
)

SUMMARY_FILE = "xunce-stage21-8-calibration-summary.json"
SWEEP_RESULTS_FILE = "xunce-stage21-8-sweep-results.jsonl"
GRADIENT_AUDIT_FILE = "xunce-stage21-8-gradient-stability-audit.json"
POLICY_AUDIT_FILE = "xunce-stage21-8-policy-shift-audit.json"
RECOMMENDED_CONFIG_FILE = "xunce-stage21-8-recommended-stage21-6-config.json"
ROUTING_FILE = "xunce-stage21-8-next-stage-routing.json"
REPORT_FILE = "xunce-stage21-8-report.md"
MANIFEST_FILE = "xunce-stage21-8-manifest.json"

ROUTE_INPUTS = "rerun_stage21_8_required_inputs"
ROUTE_GRAD = "repair_stage21_4_gradient_normalization_or_loss_scaling"
ROUTE_LARGER = "calibrate_stage21_ppo_update_strength_with_slightly_larger_step"
ROUTE_RERUN_21_6 = "rerun_stage21_6_multi_seed_pilot_with_stage21_8_recommended_config"
ROUTE_RUNTIME = "stage21_8_calibration_runtime_budget_blocked"
ROUTE_CONTINUE = "continue_stage21_8_ppo_update_strength_calibration"

BOUNDARY_FIELDS = (
    "stage21_8_authorized",
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
    "scenario_regression_total",
    "scenario_safety_boundary_regression_total",
    "checkpoint_reload_failed_count",
    "checkpoint_not_experimental_only_count",
    "seed_release_boundary_violation_total",
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 21.8 PPO update strength calibration.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        summary = run_xunce_stage21_8_ppo_update_strength_calibration(
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
                "completed_combo_count": summary["completed_combo_count"],
                "stable_combo_count": summary["stable_combo_count"],
                "recommended_combo_id": summary.get("recommended_combo_id"),
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] in {"passed", "partial"} else 1


def run_xunce_stage21_8_ppo_update_strength_calibration(
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
    boundary_reasons = _config_boundary_reasons(config)
    combos = _build_combos(config)
    result_rows: list[dict[str, Any]] = []

    if not input_reasons and not boundary_reasons:
        result_rows = _collect_existing_results(config, output_root, combos, repo_root)
        _execute_missing_combos(config, output_root, combos, result_rows, repo_root)
        result_rows = _collect_existing_results(config, output_root, combos, repo_root)

    gradient_audit = _gradient_audit(result_rows)
    policy_audit = _policy_audit(result_rows)
    recommended = _select_recommendation(config, result_rows)
    recommended_config = _write_recommended_config(config, output_root, recommended)
    status, route, primary_reason = _route(input_reasons, boundary_reasons, result_rows, recommended)

    return _write_outputs(
        config=config,
        config_path=config_path,
        output_root=output_root,
        combos=combos,
        result_rows=result_rows,
        gradient_audit=gradient_audit,
        policy_audit=policy_audit,
        recommended=recommended,
        recommended_config=recommended_config,
        status=status,
        route=route,
        primary_reason=primary_reason,
        input_reasons=input_reasons,
        boundary_reasons=boundary_reasons,
    )


def _build_combos(config: dict[str, Any]) -> list[dict[str, Any]]:
    configured = config.get("priority_combinations")
    if not isinstance(configured, list) or not configured:
        raise ConfigError("priority_combinations must be a non-empty list")
    combos = []
    for index, combo in enumerate(configured):
        learning_rate = _positive_float(combo.get("learning_rate"), "learning_rate")
        epochs = _positive_int(combo.get("epochs"), "epochs")
        clip_ratio = _positive_float(combo.get("clip_ratio"), "clip_ratio")
        pre_clip_gate = _positive_float(
            combo.get("stage21_6_pre_clip_grad_norm_gate", combo.get("max_grad_norm_gate")),
            "stage21_6_pre_clip_grad_norm_gate",
        )
        stage21_4_max_grad_norm = _positive_float(
            combo.get("stage21_4_max_grad_norm"),
            "stage21_4_max_grad_norm",
        )
        combo_id = str(
            combo.get("combo_id")
            or f"lr{learning_rate:g}_e{epochs}_c{clip_ratio:g}_clip{stage21_4_max_grad_norm:g}_gate{pre_clip_gate:g}"
        )
        combos.append(
            {
                "combo_index": index,
                "combo_id": combo_id,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "clip_ratio": clip_ratio,
                "stage21_4_max_grad_norm": stage21_4_max_grad_norm,
                "stage21_6_pre_clip_grad_norm_gate": pre_clip_gate,
                "max_grad_norm_gate": pre_clip_gate,
            }
        )
    return combos


def _collect_existing_results(
    config: dict[str, Any],
    output_root: Path,
    combos: list[dict[str, Any]],
    repo_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    baseline = _read_combo_result(
        combo={
            "combo_index": -1,
            "combo_id": "stage21_7_repaired_baseline",
            "learning_rate": None,
            "epochs": None,
            "clip_ratio": None,
            "stage21_4_max_grad_norm": None,
            "stage21_6_pre_clip_grad_norm_gate": _float(config.get("stage21_7_repaired_baseline_max_grad_norm_gate")),
            "max_grad_norm_gate": _float(config.get("stage21_7_repaired_baseline_max_grad_norm_gate")),
        },
        root=Path(config["stage21_7_repaired_stage21_6_root"]),
        source_type="stage21_7_repaired_baseline",
        repo_root=repo_root,
    )
    if baseline:
        rows.append(baseline)
    for combo in combos:
        combo_root = _combo_root(config, output_root, combo) / "stage21_6"
        existing = _read_combo_result(combo=combo, root=combo_root, source_type="stage21_8_sweep", repo_root=repo_root)
        if existing:
            rows.append(existing)
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
    retry_runtime_blocked = bool(config.get("retry_runtime_blocked_combinations", False))
    completed_ids = {
        row["combo_id"] for row in existing_rows
        if row.get("source_type") == "stage21_8_sweep"
        and (retry_runtime_blocked is False or not row.get("runtime_blocked"))
    }
    executed = 0
    for combo in combos:
        if combo["combo_id"] in completed_ids:
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
            "learning_rate": combo["learning_rate"],
            "epochs": combo["epochs"],
            "clip_ratio": combo["clip_ratio"],
            "max_grad_norm": combo["stage21_4_max_grad_norm"],
        }
    )
    stage21_4_config_path = combo_root / "stage21_4_config.json"
    _write_json(stage21_4_config_path, stage21_4_config)

    stage21_6_config = _read_json(Path(config["stage21_6_base_config"]))
    stage21_6_config.update(
        {
            "stage21_4_base_config": str(stage21_4_config_path),
            "seed_list": config["seed_list"],
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "max_grad_norm": combo["stage21_6_pre_clip_grad_norm_gate"],
            "execute_seed_pipeline": True,
            "stage21_6_authorized": False,
            "training_or_release_authorized": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    stage21_6_config_path = combo_root / "stage21_6_config.json"
    _write_json(stage21_6_config_path, stage21_6_config)
    stage21_6_root = combo_root / "stage21_6"
    timeout_path = combo_root / "runtime-blocker.json"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_xunce_stage21_6_multi_seed_ppo_pilot.py"),
        "--config",
        str(stage21_6_config_path),
        "--output-root",
        str(stage21_6_root),
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
            timeout_path,
            {
                "status": "blocked",
                "reason_code": "stage21_8_calibration_runtime_budget_blocked",
                "combo_id": combo["combo_id"],
                "timeout_seconds": int(config["per_combo_timeout_seconds"]),
                "stdout": exc.stdout,
                "stderr": exc.stderr,
            },
        )
        return
    _write_json(
        combo_root / "stage21_6-subprocess-result.json",
        {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "cmd": cmd,
        },
    )
    if completed.returncode != 0:
        _write_json(
            combo_root / "runtime-blocker.json",
            {
                "status": "blocked",
                "reason_code": "stage21_8_calibration_subprocess_failed",
                "combo_id": combo["combo_id"],
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )


def _read_combo_result(combo: dict[str, Any], root: Path, source_type: str, repo_root: Path) -> dict[str, Any]:
    summary = _read_json_or_empty(root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json")
    aggregate = _read_json_or_empty(root / "xunce-stage21-6-aggregate-metrics.json")
    seed_rows = _read_jsonl(root / "xunce-stage21-6-seed-results.jsonl")
    if not summary or not aggregate:
        blocker = _read_json_or_empty(root.parent / "runtime-blocker.json")
        if blocker:
            return _runtime_blocker_row(combo, root, source_type, blocker)
        return {}
    boundary_reasons = [
        field for field in AGGREGATE_HARD_FAIL_FIELDS
        if _int(aggregate.get(field) or summary.get(field)) > 0
    ]
    reason_codes = list(summary.get("reason_codes") or [])
    grad_reason_codes = [reason for reason in reason_codes if "grad_unstable" in str(reason)]
    kl_reason_codes = [reason for reason in reason_codes if "kl_unstable" in str(reason)]
    entropy_reason_codes = [reason for reason in reason_codes if "entropy_unstable" in str(reason)]
    policy = _policy_shift_from_seed_rows(seed_rows)
    pre_clip_max = _float(aggregate.get("pre_clip_grad_norm_max"))
    pre_clip_mean = _float(aggregate.get("pre_clip_grad_norm_mean"))
    gate = combo.get("stage21_6_pre_clip_grad_norm_gate", combo.get("max_grad_norm_gate"))
    post_clip_max = _mean_or_max(seed_rows, "post_clip_grad_norm", mode="max")
    clip_limit = _float(combo.get("stage21_4_max_grad_norm"))
    post_clip_within_clip_norm = (
        post_clip_max is not None
        and clip_limit is not None
        and post_clip_max <= clip_limit + 1.0e-5
    )
    stage_status_reasons = _stage_status_reasons(seed_rows)
    fingerprint_reasons = _fingerprint_reasons(seed_rows)
    gradient_stable = bool(not boundary_reasons and pre_clip_max is not None and gate is not None and pre_clip_max <= float(gate))
    numerically_stable = bool(
        gradient_stable
        and post_clip_within_clip_norm
        and not stage_status_reasons
        and not fingerprint_reasons
        and not grad_reason_codes
        and not kl_reason_codes
        and not entropy_reason_codes
        and _int(aggregate.get("checkpoint_reload_failed_count")) == 0
        and _int(aggregate.get("checkpoint_not_experimental_only_count")) == 0
    )
    coverage_auc_not_regressed = (
        (_float(aggregate.get("coverage_auc_delta_min")) is not None and float(aggregate["coverage_auc_delta_min"]) >= 0.0)
        and (_float(aggregate.get("final_coverage_delta_min")) is not None and float(aggregate["final_coverage_delta_min"]) >= 0.0)
    )
    policy_shift_observable = bool(
        (_float(policy.get("selected_action_change_rate_mean")) or 0.0) > 0.0
        or (_float(policy.get("selected_probability_abs_delta_mean")) or 0.0) >= 0.001
    )
    return {
        "schema_version": "xunce-stage21-8-sweep-result/v1",
        "combo_id": combo["combo_id"],
        "combo_index": combo["combo_index"],
        "source_type": source_type,
        "stage21_6_root": str(root),
        "learning_rate": combo.get("learning_rate"),
        "epochs": combo.get("epochs"),
        "clip_ratio": combo.get("clip_ratio"),
        "stage21_4_max_grad_norm": combo.get("stage21_4_max_grad_norm"),
        "stage21_6_pre_clip_grad_norm_gate": gate,
        "max_grad_norm_gate": gate,
        "status": summary.get("status"),
        "stage21_6_next_required_change": summary.get("next_required_change"),
        "reason_codes": reason_codes,
        "boundary_reason_codes": boundary_reasons + stage_status_reasons + fingerprint_reasons,
        "pre_clip_grad_norm_mean": pre_clip_mean,
        "pre_clip_grad_norm_max": pre_clip_max,
        "post_clip_grad_norm_max": post_clip_max,
        "post_clip_within_stage21_4_max_grad_norm": post_clip_within_clip_norm,
        "post_update_approx_kl_mean": _float(aggregate.get("max_post_update_approx_kl_mean")),
        "min_entropy_mean": _float(aggregate.get("min_entropy_mean")),
        "parameter_delta_l2_mean": _mean([_float(row.get("parameter_delta_l2")) for row in seed_rows]),
        "trainable_transition_count_total": _int(aggregate.get("trainable_transition_count_total")),
        "final_coverage_delta_mean": _float(aggregate.get("final_coverage_delta_mean")),
        "final_coverage_delta_min": _float(aggregate.get("final_coverage_delta_min")),
        "coverage_auc_delta_mean": _float(aggregate.get("coverage_auc_delta_mean")),
        "coverage_auc_delta_min": _float(aggregate.get("coverage_auc_delta_min")),
        "path_cost_delta_m_mean": _float(aggregate.get("path_cost_delta_m_mean")),
        "soft_risk_exposure_delta_mean": _float(aggregate.get("soft_risk_exposure_delta_mean")),
        "gradient_stable": gradient_stable,
        "numerically_stable": numerically_stable,
        "calibration_stable_only": numerically_stable,
        "coverage_auc_not_regressed": bool(coverage_auc_not_regressed),
        "policy_shift_observable": policy_shift_observable,
        **policy,
    }


def _runtime_blocker_row(combo: dict[str, Any], root: Path, source_type: str, blocker: dict[str, Any]) -> dict[str, Any]:
    gate = combo.get("stage21_6_pre_clip_grad_norm_gate", combo.get("max_grad_norm_gate"))
    return {
        "schema_version": "xunce-stage21-8-sweep-result/v1",
        "combo_id": combo["combo_id"],
        "combo_index": combo["combo_index"],
        "source_type": source_type,
        "stage21_6_root": str(root),
        "learning_rate": combo.get("learning_rate"),
        "epochs": combo.get("epochs"),
        "clip_ratio": combo.get("clip_ratio"),
        "stage21_4_max_grad_norm": combo.get("stage21_4_max_grad_norm"),
        "stage21_6_pre_clip_grad_norm_gate": gate,
        "max_grad_norm_gate": gate,
        "status": "blocked",
        "stage21_6_next_required_change": ROUTE_RUNTIME,
        "reason_codes": [blocker.get("reason_code", "stage21_8_calibration_runtime_blocked")],
        "boundary_reason_codes": [],
        "runtime_blocked": True,
        "subprocess_returncode": blocker.get("returncode"),
        "gradient_stable": False,
        "numerically_stable": False,
        "calibration_stable_only": False,
        "coverage_auc_not_regressed": False,
        "policy_shift_observable": False,
    }


def _combo_root(config: dict[str, Any], output_root: Path, combo: dict[str, Any]) -> Path:
    base = Path(str(config.get("sweep_work_root") or (output_root / "sweep_runs")))
    return base / combo["combo_id"]


def _stage_status_reasons(seed_rows: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    if not seed_rows:
        return ["missing_seed_results"]
    for row in seed_rows:
        seed = row.get("seed")
        for field in ("stage21_1_status", "stage21_3_status", "stage21_4_status", "stage21_5_status"):
            if row.get(field) != "passed":
                reasons.append(f"seed_{seed}_{field}_not_passed")
    return reasons


def _fingerprint_reasons(seed_rows: list[dict[str, Any]]) -> list[str]:
    reasons = []
    batch = []
    transitions = []
    for row in seed_rows:
        seed = row.get("seed")
        batch_value = row.get("stage21_3_batch_fingerprint")
        transition_value = row.get("transition_id_fingerprint")
        if not batch_value:
            reasons.append(f"seed_{seed}_missing_stage21_3_batch_fingerprint")
        else:
            batch.append(batch_value)
        if not transition_value:
            reasons.append(f"seed_{seed}_missing_transition_id_fingerprint")
        else:
            transitions.append(transition_value)
    if len(batch) != len(set(batch)):
        reasons.append("duplicate_stage21_3_batch_fingerprint")
    if len(transitions) != len(set(transitions)):
        reasons.append("duplicate_transition_id_fingerprint")
    return reasons


def _policy_shift_from_seed_rows(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    action_rates = []
    prob_deltas = []
    matched_counts = []
    for row in seed_rows:
        stage21_5_root = Path(str(row.get("stage21_5_root", "")))
        summary5 = _read_json_or_empty(stage21_5_root / "xunce-stage21-5-post-update-evaluation-summary.json")
        pre_rows = _read_jsonl(Path(str(summary5.get("pre_evaluation_root", ""))) / "xunce-exploration-coverage-model-inference.jsonl")
        post_rows = _read_jsonl(Path(str(summary5.get("post_evaluation_root", ""))) / "xunce-exploration-coverage-model-inference.jsonl")
        shift = _compare_policy_inference(pre_rows, post_rows)
        if shift["selected_action_change_rate"] is not None:
            action_rates.append(shift["selected_action_change_rate"])
        if shift["selected_probability_abs_delta_mean"] is not None:
            prob_deltas.append(shift["selected_probability_abs_delta_mean"])
        matched_counts.append(shift["policy_shift_matched_step_count"])
    return {
        "policy_shift_matched_step_count_total": sum(matched_counts),
        "selected_action_change_rate_mean": _mean(action_rates),
        "selected_probability_abs_delta_mean": _mean(prob_deltas),
    }


def _compare_policy_inference(pre_rows: list[dict[str, Any]], post_rows: list[dict[str, Any]]) -> dict[str, Any]:
    post_by_key = {
        (row.get("scenario_id"), row.get("step_index"), row.get("candidate_set_hash")): row
        for row in post_rows
    }
    matched = 0
    action_changes = 0
    prob_deltas = []
    for pre in pre_rows:
        post = post_by_key.get((pre.get("scenario_id"), pre.get("step_index"), pre.get("candidate_set_hash")))
        if not post:
            continue
        matched += 1
        pre_detail = pre.get("detail") if isinstance(pre.get("detail"), dict) else {}
        post_detail = post.get("detail") if isinstance(post.get("detail"), dict) else {}
        if pre_detail.get("selected_action_index") != post_detail.get("selected_action_index"):
            action_changes += 1
        pre_prob = _float(pre_detail.get("selected_probability"))
        post_prob = _float(post_detail.get("selected_probability"))
        if pre_prob is not None and post_prob is not None:
            prob_deltas.append(abs(post_prob - pre_prob))
    return {
        "policy_shift_matched_step_count": matched,
        "selected_action_change_rate": action_changes / matched if matched else None,
        "selected_probability_abs_delta_mean": _mean(prob_deltas),
    }


def _gradient_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage21-8-gradient-stability-audit/v1",
        "completed_combo_count": len(rows),
        "stable_combo_count": sum(1 for row in rows if row.get("numerically_stable")),
        "pre_clip_grad_norm_max_observed": max((_float(row.get("pre_clip_grad_norm_max")) or 0.0 for row in rows), default=None),
        "combo_ids_with_grad_unstable": [
            row["combo_id"] for row in rows if any("grad_unstable" in str(reason) for reason in row.get("reason_codes", []))
        ],
    }


def _policy_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage21-8-policy-shift-audit/v1",
        "observable_policy_shift_combo_count": sum(1 for row in rows if row.get("policy_shift_observable")),
        "selected_action_change_rate_max": max(
            (_float(row.get("selected_action_change_rate_mean")) or 0.0 for row in rows),
            default=None,
        ),
        "selected_probability_abs_delta_max": max(
            (_float(row.get("selected_probability_abs_delta_mean")) or 0.0 for row in rows),
            default=None,
        ),
    }


def _select_recommendation(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        row for row in rows
        if row.get("source_type") == "stage21_8_sweep"
        and row.get("status") == "passed"
        and row.get("numerically_stable")
        and not row.get("runtime_blocked")
        and row.get("coverage_auc_not_regressed")
        and not row.get("boundary_reason_codes")
    ]
    if not candidates:
        return {}
    candidates.sort(
        key=lambda row: (
            not bool(row.get("policy_shift_observable")),
            abs(_float(row.get("post_update_approx_kl_mean")) or 0.0),
            _float(row.get("pre_clip_grad_norm_max")) or float("inf"),
        )
    )
    return candidates[0]


def _write_recommended_config(config: dict[str, Any], output_root: Path, recommended: dict[str, Any]) -> Path | None:
    if not recommended:
        return None
    payload = _read_json(Path(config["stage21_6_base_config"]))
    stage21_4_config = _read_json(Path(config["stage21_4_base_config"]))
    stage21_4_config.update(
        {
            "learning_rate": recommended["learning_rate"],
            "epochs": recommended["epochs"],
            "clip_ratio": recommended["clip_ratio"],
            "max_grad_norm": recommended["stage21_4_max_grad_norm"],
        }
    )
    stage21_4_path = output_root / "xunce-stage21-8-recommended-stage21-4-config.json"
    _write_json(stage21_4_path, stage21_4_config)
    payload.update(
        {
            "stage21_4_base_config": str(stage21_4_path),
            "seed_list": config["seed_list"],
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "max_grad_norm": recommended["max_grad_norm_gate"],
            "stage21_6_authorized": False,
            "training_or_release_authorized": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    path = output_root / RECOMMENDED_CONFIG_FILE
    _write_json(path, payload)
    return path


def _route(
    input_reasons: list[str],
    boundary_reasons: list[str],
    rows: list[dict[str, Any]],
    recommended: dict[str, Any],
) -> tuple[str, str, str]:
    if input_reasons or boundary_reasons:
        return "failed", ROUTE_INPUTS, "input_or_boundary"
    if not rows:
        return "partial", ROUTE_RUNTIME, "no_completed_calibration_results"
    runtime_blocked_rows = [
        row for row in rows
        if row.get("source_type") == "stage21_8_sweep" and row.get("runtime_blocked")
    ]
    completed_sweep_rows = [
        row for row in rows
        if row.get("source_type") == "stage21_8_sweep" and not row.get("runtime_blocked")
    ]
    if runtime_blocked_rows and not completed_sweep_rows:
        return "partial", ROUTE_RUNTIME, "stage21_8_sweep_runtime_blocked"
    if not completed_sweep_rows:
        return "partial", ROUTE_RUNTIME, "no_completed_stage21_8_sweep_results"
    stable_rows = [row for row in completed_sweep_rows if row.get("numerically_stable")]
    if not stable_rows:
        return "failed", ROUTE_GRAD, "no_numerically_stable_combo"
    shifted_rows = [row for row in stable_rows if row.get("policy_shift_observable")]
    if not shifted_rows:
        return "partial", ROUTE_LARGER, "stable_but_policy_shift_too_small"
    if recommended:
        return "passed", ROUTE_RERUN_21_6, "stable_recommended_config"
    return "partial", ROUTE_CONTINUE, "stable_but_no_recommendation"


def _write_outputs(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    combos: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    gradient_audit: dict[str, Any],
    policy_audit: dict[str, Any],
    recommended: dict[str, Any],
    recommended_config: Path | None,
    status: str,
    route: str,
    primary_reason: str,
    input_reasons: list[str],
    boundary_reasons: list[str],
) -> dict[str, Any]:
    paths = {
        "summary": output_root / SUMMARY_FILE,
        "sweep_results": output_root / SWEEP_RESULTS_FILE,
        "gradient": output_root / GRADIENT_AUDIT_FILE,
        "policy": output_root / POLICY_AUDIT_FILE,
        "routing": output_root / ROUTING_FILE,
        "report": output_root / REPORT_FILE,
        "manifest": output_root / MANIFEST_FILE,
    }
    _write_jsonl(paths["sweep_results"], result_rows)
    _write_json(paths["gradient"], gradient_audit)
    _write_json(paths["policy"], policy_audit)
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "primary_reason": primary_reason,
        "input_reason_codes": input_reasons,
        "boundary_reason_codes": boundary_reasons,
        "stage21_8_authorized": False,
        "training_or_release_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    _write_json(paths["routing"], routing)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "next_required_change": route,
        "primary_reason": primary_reason,
        "input_reason_codes": input_reasons,
        "boundary_reason_codes": boundary_reasons,
        "planned_combo_count": len(combos),
        "completed_combo_count": len(result_rows),
        "completed_stage21_8_sweep_combo_count": sum(1 for row in result_rows if row.get("source_type") == "stage21_8_sweep"),
        "runtime_blocked_stage21_8_sweep_combo_count": sum(
            1 for row in result_rows if row.get("source_type") == "stage21_8_sweep" and row.get("runtime_blocked")
        ),
        "retry_runtime_blocked_combinations": bool(config.get("retry_runtime_blocked_combinations", False)),
        "stable_combo_count": gradient_audit["stable_combo_count"],
        "recommended_combo_id": recommended.get("combo_id"),
        "recommended_stage21_6_config": str(recommended_config) if recommended_config else None,
        "stage21_7_root": str(config["stage21_7_root"]),
        "stage21_7_repaired_stage21_6_root": str(config["stage21_7_repaired_stage21_6_root"]),
        "stage21_8_authorized": False,
        "training_or_release_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
    }
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_report(summary, gradient_audit, policy_audit, recommended), encoding="utf-8")
    _write_json(
        paths["manifest"],
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generated_at": summary["generated_at"],
            "config": str(config_path),
            "artifacts": {key: str(path) for key, path in paths.items()},
            "recommended_stage21_6_config": str(recommended_config) if recommended_config else None,
            "summary_status": status,
            "next_required_change": route,
        },
    )
    return summary


def _report(summary: dict[str, Any], gradient_audit: dict[str, Any], policy_audit: dict[str, Any], recommended: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage 21.8 PPO Update Strength Calibration",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- primary_reason: `{summary['primary_reason']}`",
            f"- completed_stage21_8_sweep_combo_count: `{summary['completed_stage21_8_sweep_combo_count']}`",
            f"- runtime_blocked_stage21_8_sweep_combo_count: `{summary['runtime_blocked_stage21_8_sweep_combo_count']}`",
            f"- retry_runtime_blocked_combinations: `{summary['retry_runtime_blocked_combinations']}`",
            f"- stable_combo_count: `{summary['stable_combo_count']}`",
            f"- recommended_combo_id: `{summary['recommended_combo_id']}`",
            f"- combo_ids_with_grad_unstable: `{gradient_audit['combo_ids_with_grad_unstable']}`",
            f"- observable_policy_shift_combo_count: `{policy_audit['observable_policy_shift_combo_count']}`",
            "",
            "Stage 21.8 is bounded calibration evidence only. It does not publish checkpoints, replace default policy, connect an executor, or start canary traffic.",
            "",
        ]
    )


def _input_reasons(config: dict[str, Any]) -> list[str]:
    reasons = []
    stage21_7_summary = _read_json_or_empty(Path(config["stage21_7_root"]) / "xunce-stage21-7-diagnostic-summary.json")
    if not stage21_7_summary:
        reasons.append("missing_stage21_7_summary")
    elif stage21_7_summary.get("next_required_change") != "calibrate_stage21_ppo_update_strength":
        reasons.append("stage21_7_route_not_calibrate_ppo_update_strength")
    if not (Path(config["stage21_7_repaired_stage21_6_root"]) / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json").is_file():
        reasons.append("missing_stage21_7_repaired_stage21_6_summary")
    if not Path(config["stage21_6_base_config"]).is_file():
        reasons.append("missing_stage21_6_base_config")
    if not Path(config["stage21_4_base_config"]).is_file():
        reasons.append("missing_stage21_4_base_config")
    return reasons


def _config_boundary_reasons(config: dict[str, Any]) -> list[str]:
    reasons = [f"stage21_8_config_{field}_true" for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) > 0.0:
        reasons.append("stage21_8_config_canary_traffic_fraction_nonzero")
    return sorted(set(reasons))


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    for field in ("stage21_7_root", "stage21_7_repaired_stage21_6_root", "stage21_6_base_config", "stage21_4_base_config"):
        config[field] = str(_resolve_path(Path(str(config[field])), repo_root))
    if config.get("sweep_work_root"):
        config["sweep_work_root"] = str(_resolve_path(Path(str(config["sweep_work_root"])), repo_root))
    config["execute_sweep"] = bool(config.get("execute_sweep", True))
    config["retry_runtime_blocked_combinations"] = bool(config.get("retry_runtime_blocked_combinations", False))
    config["seed_list"] = [int(value) for value in config.get("seed_list", [2101, 2102, 2103])]
    for field in (
        "required_scenario_count",
        "rollout_steps",
        "dynamic_max_candidates_per_step",
        "dynamic_proposal_pool_limit_per_step",
        "max_new_combinations_to_execute",
        "per_combo_timeout_seconds",
    ):
        config[field] = _positive_int(config[field], field)
    config["stage21_7_repaired_baseline_max_grad_norm_gate"] = _positive_float(
        config.get("stage21_7_repaired_baseline_max_grad_norm_gate", 25.0),
        "stage21_7_repaired_baseline_max_grad_norm_gate",
    )
    return config


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


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


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return mean(clean) if clean else None


def _mean_or_max(rows: list[dict[str, Any]], field: str, *, mode: str) -> float | None:
    values = [_float(row.get(field)) for row in rows]
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    if mode == "max":
        return max(clean)
    return mean(clean)


if __name__ == "__main__":
    raise SystemExit(main())
