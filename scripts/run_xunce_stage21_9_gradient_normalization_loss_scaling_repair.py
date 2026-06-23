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


CONFIG_SCHEMA_VERSION = "xunce-stage21-9-gradient-normalization-loss-scaling-repair-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage21-9-diagnostic-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage21-9-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage21-9-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage21_9_gradient_normalization_loss_scaling_repair_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/"
    "outputs/path_feedback_batch_xunce_stage21_9_gradient_normalization_loss_scaling_repair_v1"
)

SUMMARY_FILE = "xunce-stage21-9-diagnostic-summary.json"
SEED_DIAGNOSTICS_FILE = "xunce-stage21-9-seed-loss-scale-diagnostics.jsonl"
ADVANTAGE_AUDIT_FILE = "xunce-stage21-9-advantage-scale-audit.json"
LOSS_GRADIENT_AUDIT_FILE = "xunce-stage21-9-loss-component-gradient-audit.json"
REPAIRED_STAGE21_4_CONFIG_FILE = "xunce-stage21-9-repaired-stage21-4-config.json"
REPAIRED_STAGE21_8_CONFIG_FILE = "xunce-stage21-9-repaired-stage21-8-config.json"
REPAIRED_SMOKE_SUMMARY_FILE = "xunce-stage21-9-repaired-smoke-summary.json"
ROUTING_FILE = "xunce-stage21-9-next-stage-routing.json"
REPORT_FILE = "xunce-stage21-9-report.md"
MANIFEST_FILE = "xunce-stage21-9-manifest.json"

ROUTE_INPUTS = "rerun_stage21_9_required_inputs"
ROUTE_ADVANTAGE = "repair_stage21_3_advantage_normalization_contract"
ROUTE_LOSS = "repair_stage21_4_loss_component_scaling"
ROUTE_CONTINUE = "continue_stage21_9_gradient_normalization_loss_scaling_repair"
ROUTE_RERUN_21_6 = "rerun_stage21_6_multi_seed_pilot_with_stage21_9_repaired_config"
ROUTE_RUNTIME = "stage21_9_repaired_smoke_runtime_budget_blocked"
ROUTE_CLIP = "repair_stage21_4_gradient_clipping_application"
ROUTE_WEAK_SHIFT = "calibrate_stage21_4_loss_scale_or_learning_rate_for_policy_shift"
ROUTE_NUMERICS = "repair_stage21_4_ppo_update_numerics"

BOUNDARY_FIELDS = (
    "stage21_9_authorized",
    "training_or_release_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 21.9 gradient normalization and loss scaling repair.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        summary = run_xunce_stage21_9_gradient_normalization_loss_scaling_repair(
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
                "repaired_smoke_status": summary.get("repaired_smoke_status"),
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] in {"passed", "partial"} else 1


def run_xunce_stage21_9_gradient_normalization_loss_scaling_repair(
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
    seed_rows: list[dict[str, Any]] = []
    advantage_audit: dict[str, Any] = _empty_advantage_audit()
    loss_gradient_audit: dict[str, Any] = _empty_loss_gradient_audit()
    repaired_stage21_4_config: Path | None = None
    repaired_stage21_8_config: Path | None = None
    repaired_smoke_summary: dict[str, Any] = {}
    repaired_smoke_result: dict[str, Any] = {}
    repaired_loss_gradient_audit: dict[str, Any] = _empty_loss_gradient_audit()
    runtime_reason_codes: list[str] = []

    if not input_reasons and not boundary_reasons:
        seed_rows = _seed_diagnostics(Path(config["stage21_8_completed_combo_root"]))
        advantage_audit = _advantage_audit(seed_rows)
        loss_gradient_audit = _loss_gradient_audit(seed_rows)
        repaired_stage21_4_config = _write_repaired_stage21_4_config(config, output_root)
        repaired_stage21_8_config = _write_repaired_stage21_8_config(config, output_root, repaired_stage21_4_config)
        if config["execute_repaired_smoke"]:
            runtime_reason_codes = _execute_repaired_smoke(config, repaired_stage21_8_config, output_root, repo_root)
        repaired_smoke_summary = _read_repaired_smoke_summary(config)
        repaired_smoke_result = _read_repaired_smoke_result(config)
        if repaired_smoke_result.get("stage21_6_root"):
            repaired_loss_gradient_audit = _loss_gradient_audit(_seed_diagnostics(Path(str(repaired_smoke_result["stage21_6_root"]))))

    status, route, primary_reason = _route(
        input_reasons=input_reasons,
        boundary_reasons=boundary_reasons,
        advantage_audit=advantage_audit,
        loss_gradient_audit=loss_gradient_audit,
        repaired_smoke_summary=repaired_smoke_summary,
        repaired_smoke_result=repaired_smoke_result,
        runtime_reason_codes=runtime_reason_codes,
    )

    return _write_outputs(
        config=config,
        config_path=config_path,
        output_root=output_root,
        seed_rows=seed_rows,
        advantage_audit=advantage_audit,
        loss_gradient_audit=loss_gradient_audit,
        repaired_loss_gradient_audit=repaired_loss_gradient_audit,
        repaired_smoke_summary=repaired_smoke_summary,
        repaired_smoke_result=repaired_smoke_result,
        repaired_stage21_4_config=repaired_stage21_4_config,
        repaired_stage21_8_config=repaired_stage21_8_config,
        status=status,
        route=route,
        primary_reason=primary_reason,
        input_reasons=input_reasons,
        boundary_reasons=boundary_reasons,
        runtime_reason_codes=runtime_reason_codes,
    )


def _seed_diagnostics(stage21_6_root: Path) -> list[dict[str, Any]]:
    rows = []
    for seed_row in _read_jsonl(stage21_6_root / "xunce-stage21-6-seed-results.jsonl"):
        stage21_3_root = Path(str(seed_row.get("stage21_3_root", "")))
        stage21_4_root = Path(str(seed_row.get("stage21_4_root", "")))
        loss_rows = _read_jsonl(stage21_4_root / "xunce-stage21-4-ppo-loss-audit.jsonl")
        gradient = _read_json_or_empty(stage21_4_root / "xunce-stage21-4-gradient-audit.json")
        batch_rows = _read_jsonl(stage21_3_root / "xunce-stage21-3-ppo-trainable-batch.jsonl")
        return_rows = _read_jsonl(stage21_3_root / "xunce-stage21-3-return-advantage-audit.jsonl")
        final_loss = loss_rows[-1] if loss_rows else {}
        dominant = _dominant_component(final_loss)
        train_batch = [row for row in batch_rows if row.get("stage21_3_split") == "train"]
        train_return = [row for row in return_rows if row.get("stage21_3_split") == "train"]
        normalized_flags = [row.get("advantage_normalization_applied") for row in train_batch]
        rows.append(
            {
                "schema_version": "xunce-stage21-9-seed-loss-scale-diagnostic/v1",
                "seed": seed_row.get("seed"),
                "stage21_3_root": str(stage21_3_root),
                "stage21_4_root": str(stage21_4_root),
                "stage21_5_root": seed_row.get("stage21_5_root"),
                "trainable_transition_count": seed_row.get("trainable_transition_count"),
                "pre_clip_grad_norm": _float(seed_row.get("pre_clip_grad_norm")),
                "post_clip_grad_norm": _float(seed_row.get("post_clip_grad_norm")),
                "grad_norm_finite": bool(seed_row.get("grad_norm_finite")),
                "stage21_3_train_batch_row_count": len(train_batch),
                "stage21_3_train_return_audit_row_count": len(train_return),
                "advantage_normalization_applied_all_train": bool(train_batch) and all(value is True for value in normalized_flags),
                "advantage_normalization_applied_any_train": any(value is True for value in normalized_flags),
                "raw_advantage_mean": _mean([_float(row.get("raw_advantage")) for row in train_batch]),
                "raw_advantage_std": _std([_float(row.get("raw_advantage")) for row in train_batch]),
                "normalized_advantage_mean": _mean([_float(row.get("advantage")) for row in train_batch]),
                "normalized_advantage_std": _std([_float(row.get("advantage")) for row in train_batch]),
                "total_loss": _float(final_loss.get("total_loss")),
                "policy_loss": _float(final_loss.get("policy_loss")),
                "value_loss": _float(final_loss.get("value_loss")),
                "entropy": _float(final_loss.get("entropy")),
                "total_loss_grad_norm": _float(final_loss.get("total_loss_grad_norm")),
                "policy_loss_grad_norm": _float(final_loss.get("policy_loss_grad_norm")),
                "value_loss_grad_norm": _float(final_loss.get("value_loss_grad_norm")),
                "entropy_loss_grad_norm": _float(final_loss.get("entropy_loss_grad_norm")),
                "dominant_loss_component": dominant["component"],
                "dominant_loss_component_value": dominant["value"],
                "gradient_audit_component_grad_norms": gradient.get("component_grad_norms"),
            }
        )
    return rows


def _dominant_component(loss_row: dict[str, Any]) -> dict[str, Any]:
    candidates = {
        "policy": abs(_float(loss_row.get("policy_loss_grad_norm")) or _float(loss_row.get("policy_loss")) or 0.0),
        "value": abs(_float(loss_row.get("value_loss_grad_norm")) or _float(loss_row.get("value_loss")) or 0.0),
        "entropy": abs(_float(loss_row.get("entropy_loss_grad_norm")) or _float(loss_row.get("entropy")) or 0.0),
    }
    component, value = max(candidates.items(), key=lambda item: item[1])
    return {"component": component, "value": value}


def _advantage_audit(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage21-9-advantage-scale-audit/v1",
        "seed_count": len(seed_rows),
        "all_train_advantages_normalized": bool(seed_rows) and all(row.get("advantage_normalization_applied_all_train") for row in seed_rows),
        "any_train_advantages_normalized": any(row.get("advantage_normalization_applied_any_train") for row in seed_rows),
        "normalized_advantage_std_mean": _mean([_float(row.get("normalized_advantage_std")) for row in seed_rows]),
        "raw_advantage_std_mean": _mean([_float(row.get("raw_advantage_std")) for row in seed_rows]),
    }


def _loss_gradient_audit(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    dominant_counts: dict[str, int] = {}
    for row in seed_rows:
        dominant_counts[str(row.get("dominant_loss_component"))] = dominant_counts.get(str(row.get("dominant_loss_component")), 0) + 1
    return {
        "schema_version": "xunce-stage21-9-loss-component-gradient-audit/v1",
        "seed_count": len(seed_rows),
        "pre_clip_grad_norm_max": max((_float(row.get("pre_clip_grad_norm")) or 0.0 for row in seed_rows), default=None),
        "pre_clip_grad_norm_mean": _mean([_float(row.get("pre_clip_grad_norm")) for row in seed_rows]),
        "post_clip_grad_norm_max": max((_float(row.get("post_clip_grad_norm")) or 0.0 for row in seed_rows), default=None),
        "dominant_loss_component_counts": dominant_counts,
        "total_loss_grad_norm_mean": _mean([_float(row.get("total_loss_grad_norm")) for row in seed_rows]),
        "policy_loss_grad_norm_mean": _mean([_float(row.get("policy_loss_grad_norm")) for row in seed_rows]),
        "value_loss_grad_norm_mean": _mean([_float(row.get("value_loss_grad_norm")) for row in seed_rows]),
        "entropy_loss_grad_norm_mean": _mean([_float(row.get("entropy_loss_grad_norm")) for row in seed_rows]),
        "component_gradient_audit_available": any(row.get("policy_loss_grad_norm") is not None for row in seed_rows),
    }


def _write_repaired_stage21_4_config(config: dict[str, Any], output_root: Path) -> Path:
    payload = _read_json(Path(config["stage21_4_base_config"]))
    payload.update(
        {
            "advantage_clip_abs": float(config["advantage_clip_abs"]),
            "normalize_minibatch_advantages": bool(config["normalize_minibatch_advantages"]),
            "loss_scale": float(config["loss_scale"]),
            "value_loss_coefficient": float(config["value_loss_coefficient"]),
            "entropy_coefficient": float(config["entropy_coefficient"]),
            "stage21_4_authorized": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    path = output_root / REPAIRED_STAGE21_4_CONFIG_FILE
    _write_json(path, payload)
    return path


def _write_repaired_stage21_8_config(config: dict[str, Any], output_root: Path, stage21_4_config: Path) -> Path:
    payload = _read_json(Path(config["stage21_8_base_config"]))
    payload.update(
        {
            "stage21_4_base_config": str(stage21_4_config),
            "execute_sweep": bool(config["execute_repaired_smoke"]),
            "retry_runtime_blocked_combinations": True,
            "sweep_work_root": str(config["repaired_stage21_8_sweep_work_root"]),
            "max_new_combinations_to_execute": 1,
            "per_combo_timeout_seconds": int(config["repaired_smoke_timeout_seconds"]),
            "priority_combinations": [
                {
                    "combo_id": "stage21_9_repaired_loss_scale_smoke",
                    "learning_rate": float(config["learning_rate"]),
                    "epochs": int(config["epochs"]),
                    "clip_ratio": float(config["clip_ratio"]),
                    "stage21_4_max_grad_norm": float(config["stage21_4_max_grad_norm"]),
                    "stage21_6_pre_clip_grad_norm_gate": float(config["stage21_6_pre_clip_grad_norm_gate"]),
                }
            ],
            "stage21_8_authorized": False,
            "training_or_release_authorized": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    path = output_root / REPAIRED_STAGE21_8_CONFIG_FILE
    _write_json(path, payload)
    return path


def _execute_repaired_smoke(config: dict[str, Any], repaired_stage21_8_config: Path, output_root: Path, repo_root: Path) -> list[str]:
    smoke_root = Path(config["repaired_stage21_8_output_root"])
    smoke_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_xunce_stage21_8_ppo_update_strength_calibration.py"),
        "--config",
        str(repaired_stage21_8_config),
        "--output-root",
        str(smoke_root),
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
            timeout=int(config["repaired_smoke_timeout_seconds"]) + 300,
        )
    except subprocess.TimeoutExpired as exc:
        _write_json(
            output_root / "xunce-stage21-9-repaired-smoke-runtime-blocker.json",
            {
                "status": "blocked",
                "reason_code": ROUTE_RUNTIME,
                "stdout": exc.stdout,
                "stderr": exc.stderr,
            },
        )
        return [ROUTE_RUNTIME]
    _write_json(
        output_root / "xunce-stage21-9-repaired-smoke-subprocess-result.json",
        {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "cmd": cmd},
    )
    return ["stage21_9_repaired_smoke_subprocess_failed"] if completed.returncode not in (0, 1) else []


def _read_repaired_smoke_summary(config: dict[str, Any]) -> dict[str, Any]:
    return _read_json_or_empty(Path(config["repaired_stage21_8_output_root"]) / "xunce-stage21-8-calibration-summary.json")


def _read_repaired_smoke_result(config: dict[str, Any]) -> dict[str, Any]:
    rows = _read_jsonl(Path(config["repaired_stage21_8_output_root"]) / "xunce-stage21-8-sweep-results.jsonl")
    for row in rows:
        if row.get("combo_id") == "stage21_9_repaired_loss_scale_smoke":
            return row
    return {}


def _route(
    *,
    input_reasons: list[str],
    boundary_reasons: list[str],
    advantage_audit: dict[str, Any],
    loss_gradient_audit: dict[str, Any],
    repaired_smoke_summary: dict[str, Any],
    repaired_smoke_result: dict[str, Any],
    runtime_reason_codes: list[str],
) -> tuple[str, str, str]:
    if input_reasons or boundary_reasons:
        return "failed", ROUTE_INPUTS, "input_or_boundary"
    if not advantage_audit.get("all_train_advantages_normalized"):
        return "failed", ROUTE_ADVANTAGE, "advantage_normalization_not_confirmed"
    if runtime_reason_codes:
        return "partial", ROUTE_RUNTIME, "repaired_smoke_runtime_blocked"
    if not repaired_smoke_summary:
        return "partial", ROUTE_LOSS, "repaired_smoke_not_executed"
    if not repaired_smoke_result:
        return "partial", ROUTE_LOSS, "repaired_smoke_result_missing"
    if repaired_smoke_result.get("runtime_blocked"):
        return "partial", ROUTE_RUNTIME, "repaired_smoke_runtime_blocked"
    boundary = repaired_smoke_result.get("boundary_reason_codes")
    if isinstance(boundary, list) and boundary:
        return "failed", ROUTE_INPUTS, "repaired_smoke_boundary_or_lineage_failed"
    pre_clip = _float(repaired_smoke_result.get("pre_clip_grad_norm_max"))
    gate = _float(repaired_smoke_result.get("stage21_6_pre_clip_grad_norm_gate"))
    if pre_clip is None or gate is None or pre_clip > gate:
        return "failed", ROUTE_CONTINUE, "repaired_smoke_still_grad_unstable"
    if repaired_smoke_result.get("post_clip_within_stage21_4_max_grad_norm") is False:
        return "failed", ROUTE_CLIP, "repaired_smoke_post_clip_outside_limit"
    kl = _float(repaired_smoke_result.get("post_update_approx_kl_mean"))
    entropy = _float(repaired_smoke_result.get("min_entropy_mean"))
    parameter_delta = _float(repaired_smoke_result.get("parameter_delta_l2_mean"))
    if kl is None or entropy is None or parameter_delta is None:
        return "failed", ROUTE_NUMERICS, "repaired_smoke_numerics_missing"
    if parameter_delta <= 0.0:
        return "partial", ROUTE_WEAK_SHIFT, "repaired_smoke_parameter_shift_zero"
    if repaired_smoke_result.get("policy_shift_observable") is False:
        return "partial", ROUTE_WEAK_SHIFT, "repaired_smoke_policy_shift_too_small"
    if repaired_smoke_result.get("numerically_stable") and repaired_smoke_result.get("coverage_auc_not_regressed"):
        return "passed", ROUTE_RERUN_21_6, "repaired_smoke_stable"
    if loss_gradient_audit.get("component_gradient_audit_available") is False:
        return "failed", ROUTE_LOSS, "component_gradient_audit_missing_before_repair"
    return "partial", ROUTE_LOSS, "loss_component_scaling_repair_required"


def _write_outputs(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    seed_rows: list[dict[str, Any]],
    advantage_audit: dict[str, Any],
    loss_gradient_audit: dict[str, Any],
    repaired_loss_gradient_audit: dict[str, Any],
    repaired_smoke_summary: dict[str, Any],
    repaired_smoke_result: dict[str, Any],
    repaired_stage21_4_config: Path | None,
    repaired_stage21_8_config: Path | None,
    status: str,
    route: str,
    primary_reason: str,
    input_reasons: list[str],
    boundary_reasons: list[str],
    runtime_reason_codes: list[str],
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    paths = {
        "summary": output_root / SUMMARY_FILE,
        "seed_diagnostics": output_root / SEED_DIAGNOSTICS_FILE,
        "advantage": output_root / ADVANTAGE_AUDIT_FILE,
        "loss_gradient": output_root / LOSS_GRADIENT_AUDIT_FILE,
        "smoke": output_root / REPAIRED_SMOKE_SUMMARY_FILE,
        "routing": output_root / ROUTING_FILE,
        "report": output_root / REPORT_FILE,
        "manifest": output_root / MANIFEST_FILE,
    }
    _write_jsonl(paths["seed_diagnostics"], seed_rows)
    _write_json(paths["advantage"], advantage_audit)
    _write_json(paths["loss_gradient"], {**loss_gradient_audit, "repaired_smoke": repaired_loss_gradient_audit})
    _write_json(
        paths["smoke"],
        {
            "stage21_8_summary": repaired_smoke_summary or {"status": "not_executed"},
            "stage21_8_repaired_smoke_result": repaired_smoke_result or {},
        },
    )
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "primary_reason": primary_reason,
        "input_reason_codes": input_reasons,
        "boundary_reason_codes": boundary_reasons,
        "runtime_reason_codes": runtime_reason_codes,
        "stage21_9_authorized": False,
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
        "generated_at": generated_at,
        "status": status,
        "next_required_change": route,
        "primary_reason": primary_reason,
        "input_reason_codes": input_reasons,
        "boundary_reason_codes": boundary_reasons,
        "runtime_reason_codes": runtime_reason_codes,
        "seed_diagnostic_count": len(seed_rows),
        "pre_clip_grad_norm_max": loss_gradient_audit.get("pre_clip_grad_norm_max"),
        "total_loss_grad_norm_mean": loss_gradient_audit.get("total_loss_grad_norm_mean"),
        "policy_loss_grad_norm_mean": loss_gradient_audit.get("policy_loss_grad_norm_mean"),
        "value_loss_grad_norm_mean": loss_gradient_audit.get("value_loss_grad_norm_mean"),
        "entropy_loss_grad_norm_mean": loss_gradient_audit.get("entropy_loss_grad_norm_mean"),
        "dominant_loss_component_counts": loss_gradient_audit.get("dominant_loss_component_counts"),
        "all_train_advantages_normalized": advantage_audit.get("all_train_advantages_normalized"),
        "component_gradient_audit_available": loss_gradient_audit.get("component_gradient_audit_available"),
        "repaired_stage21_4_config": str(repaired_stage21_4_config) if repaired_stage21_4_config else None,
        "repaired_stage21_8_config": str(repaired_stage21_8_config) if repaired_stage21_8_config else None,
        "repaired_smoke_status": repaired_smoke_summary.get("status"),
        "repaired_smoke_next_required_change": repaired_smoke_summary.get("next_required_change"),
        "repaired_smoke_combo_id": repaired_smoke_result.get("combo_id"),
        "repaired_pre_clip_grad_norm_max": repaired_smoke_result.get("pre_clip_grad_norm_max"),
        "repaired_stage21_6_pre_clip_grad_norm_gate": repaired_smoke_result.get("stage21_6_pre_clip_grad_norm_gate"),
        "repaired_post_clip_grad_norm_max": repaired_smoke_result.get("post_clip_grad_norm_max"),
        "repaired_post_clip_within_stage21_4_max_grad_norm": repaired_smoke_result.get("post_clip_within_stage21_4_max_grad_norm"),
        "repaired_post_update_approx_kl_mean": repaired_smoke_result.get("post_update_approx_kl_mean"),
        "repaired_min_entropy_mean": repaired_smoke_result.get("min_entropy_mean"),
        "repaired_parameter_delta_l2_mean": repaired_smoke_result.get("parameter_delta_l2_mean"),
        "repaired_policy_shift_observable": repaired_smoke_result.get("policy_shift_observable"),
        "repaired_coverage_auc_not_regressed": repaired_smoke_result.get("coverage_auc_not_regressed"),
        "repaired_numerically_stable": repaired_smoke_result.get("numerically_stable"),
        "repaired_total_loss_grad_norm_mean": repaired_loss_gradient_audit.get("total_loss_grad_norm_mean"),
        "repaired_policy_loss_grad_norm_mean": repaired_loss_gradient_audit.get("policy_loss_grad_norm_mean"),
        "repaired_value_loss_grad_norm_mean": repaired_loss_gradient_audit.get("value_loss_grad_norm_mean"),
        "repaired_entropy_loss_grad_norm_mean": repaired_loss_gradient_audit.get("entropy_loss_grad_norm_mean"),
        "repaired_dominant_loss_component_counts": repaired_loss_gradient_audit.get("dominant_loss_component_counts"),
        "repaired_component_gradient_audit_available": repaired_loss_gradient_audit.get("component_gradient_audit_available"),
        "repaired_stage21_6_status": repaired_smoke_result.get("status"),
        "repaired_stage21_6_next_required_change": repaired_smoke_result.get("stage21_6_next_required_change"),
        "repaired_smoke_root": str(config["repaired_stage21_8_output_root"]),
        "stage21_9_authorized": False,
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
    paths["report"].write_text(_report(summary), encoding="utf-8")
    _write_json(
        paths["manifest"],
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generated_at": generated_at,
            "config": str(config_path),
            "artifacts": {key: str(path) for key, path in paths.items()},
            "summary_status": status,
            "next_required_change": route,
        },
    )
    return summary


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage 21.9 Gradient Normalization Loss Scaling Repair",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- primary_reason: `{summary['primary_reason']}`",
            f"- pre_clip_grad_norm_max: `{summary['pre_clip_grad_norm_max']}`",
            f"- dominant_loss_component_counts: `{summary['dominant_loss_component_counts']}`",
            f"- total_loss_grad_norm_mean: `{summary.get('total_loss_grad_norm_mean')}`",
            f"- value_loss_grad_norm_mean: `{summary.get('value_loss_grad_norm_mean')}`",
            f"- repaired_smoke_status: `{summary['repaired_smoke_status']}`",
            f"- repaired_smoke_next_required_change: `{summary['repaired_smoke_next_required_change']}`",
            f"- repaired_pre_clip_grad_norm_max: `{summary.get('repaired_pre_clip_grad_norm_max')}`",
            f"- repaired_stage21_6_pre_clip_grad_norm_gate: `{summary.get('repaired_stage21_6_pre_clip_grad_norm_gate')}`",
            f"- repaired_policy_shift_observable: `{summary.get('repaired_policy_shift_observable')}`",
            f"- repaired_total_loss_grad_norm_mean: `{summary.get('repaired_total_loss_grad_norm_mean')}`",
            f"- repaired_value_loss_grad_norm_mean: `{summary.get('repaired_value_loss_grad_norm_mean')}`",
            f"- repaired_dominant_loss_component_counts: `{summary.get('repaired_dominant_loss_component_counts')}`",
            "",
            "Baseline loss-component attribution may be inferred from legacy loss magnitudes when `component_gradient_audit_available=false`; repaired smoke attribution uses Stage21.4 component gradient norms directly.",
            "",
            "Stage 21.9 is an offline repair smoke only. It does not publish checkpoints, replace default policy, connect an executor, or start canary traffic.",
            "",
        ]
    )


def _input_reasons(config: dict[str, Any]) -> list[str]:
    reasons = []
    stage21_8_summary = _read_json_or_empty(Path(config["stage21_8_root"]) / "xunce-stage21-8-calibration-summary.json")
    if not stage21_8_summary:
        reasons.append("missing_stage21_8_summary")
    elif stage21_8_summary.get("next_required_change") != "repair_stage21_4_gradient_normalization_or_loss_scaling":
        reasons.append("stage21_8_route_not_gradient_normalization_loss_scaling")
    combo_root = Path(config["stage21_8_completed_combo_root"])
    if not (combo_root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json").is_file():
        reasons.append("missing_stage21_8_completed_combo_stage21_6_summary")
    if not Path(config["stage21_4_base_config"]).is_file():
        reasons.append("missing_stage21_4_base_config")
    if not Path(config["stage21_8_base_config"]).is_file():
        reasons.append("missing_stage21_8_base_config")
    return reasons


def _boundary_reasons(config: dict[str, Any]) -> list[str]:
    reasons = [f"stage21_9_config_{field}_true" for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) > 0.0:
        reasons.append("stage21_9_config_canary_traffic_fraction_nonzero")
    return sorted(set(reasons))


def _empty_advantage_audit() -> dict[str, Any]:
    return {"schema_version": "xunce-stage21-9-advantage-scale-audit/v1", "seed_count": 0}


def _empty_loss_gradient_audit() -> dict[str, Any]:
    return {"schema_version": "xunce-stage21-9-loss-component-gradient-audit/v1", "seed_count": 0}


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    for field in (
        "stage21_8_root",
        "stage21_8_completed_combo_root",
        "stage21_4_base_config",
        "stage21_8_base_config",
        "repaired_stage21_8_output_root",
    ):
        config[field] = str(_resolve_path(Path(str(config[field])), repo_root))
    sweep_root = config.get("repaired_stage21_8_sweep_work_root")
    if sweep_root:
        config["repaired_stage21_8_sweep_work_root"] = str(_resolve_path(Path(str(sweep_root)), repo_root))
    else:
        config["repaired_stage21_8_sweep_work_root"] = (
            "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/"
            "stage21_9_repaired_stage21_8_sweep_runs_v1"
        )
    config["execute_repaired_smoke"] = bool(config.get("execute_repaired_smoke", True))
    config["repaired_smoke_timeout_seconds"] = _positive_int(config.get("repaired_smoke_timeout_seconds", 7200), "repaired_smoke_timeout_seconds")
    config["learning_rate"] = _positive_float(config.get("learning_rate", 2.0e-6), "learning_rate")
    config["epochs"] = _positive_int(config.get("epochs", 1), "epochs")
    config["clip_ratio"] = _positive_float(config.get("clip_ratio", 0.2), "clip_ratio")
    config["stage21_4_max_grad_norm"] = _positive_float(config.get("stage21_4_max_grad_norm", 1.0), "stage21_4_max_grad_norm")
    config["stage21_6_pre_clip_grad_norm_gate"] = _positive_float(config.get("stage21_6_pre_clip_grad_norm_gate", 25.0), "stage21_6_pre_clip_grad_norm_gate")
    config["advantage_clip_abs"] = _nonnegative_float(config.get("advantage_clip_abs", 5.0), "advantage_clip_abs")
    config["normalize_minibatch_advantages"] = bool(config.get("normalize_minibatch_advantages", True))
    config["loss_scale"] = _positive_float(config.get("loss_scale", 0.25), "loss_scale")
    config["value_loss_coefficient"] = _nonnegative_float(config.get("value_loss_coefficient", 0.1), "value_loss_coefficient")
    config["entropy_coefficient"] = _nonnegative_float(config.get("entropy_coefficient", 0.01), "entropy_coefficient")
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


def _positive_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be positive") from exc
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
    clean = [float(value) for value in values if value is not None]
    return mean(clean) if clean else None


def _std(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    avg = mean(clean)
    return (sum((value - avg) ** 2 for value in clean) / len(clean)) ** 0.5


if __name__ == "__main__":
    raise SystemExit(main())
