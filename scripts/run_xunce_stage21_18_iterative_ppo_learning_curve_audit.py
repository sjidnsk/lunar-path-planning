from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_IMPORT_ROOT))

import scripts.run_xunce_stage21_16_policy_signal_margin_credit_attribution as s16
from scripts.run_xunce_stage21_15_policy_update_signal_strength_calibration import (
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
from scripts.run_xunce_stage21_6_multi_seed_ppo_pilot import run_xunce_stage21_6_multi_seed_ppo_pilot


CONFIG_SCHEMA_VERSION = "xunce-stage21-18-iterative-ppo-learning-curve-audit-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage21-18-summary/v1"
ROUND_ROW_SCHEMA_VERSION = "xunce-stage21-18-round-result/v1"
LINEAGE_SCHEMA_VERSION = "xunce-stage21-18-checkpoint-lineage/v1"
ACTION_TREND_SCHEMA_VERSION = "xunce-stage21-18-action-signal-trend/v1"
COVERAGE_TREND_SCHEMA_VERSION = "xunce-stage21-18-coverage-trend/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage21-18-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage21-18-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage21_18_iterative_ppo_learning_curve_audit_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/outputs/"
    "path_feedback_batch_xunce_stage21_18_iterative_ppo_learning_curve_audit_v1"
)

SUMMARY_FILE = "xunce-stage21-18-summary.json"
ROUND_RESULTS_FILE = "xunce-stage21-18-round-results.jsonl"
CHECKPOINT_LINEAGE_FILE = "xunce-stage21-18-checkpoint-lineage.json"
ACTION_TREND_FILE = "xunce-stage21-18-action-signal-trend.json"
COVERAGE_TREND_FILE = "xunce-stage21-18-coverage-trend.json"
RECOMMENDED_CONFIG_FILE = "xunce-stage21-18-recommended-stage21-6-config.json"
ROUTING_FILE = "xunce-stage21-18-next-stage-routing.json"
REPORT_FILE = "xunce-stage21-18-report.md"
MANIFEST_FILE = "xunce-stage21-18-manifest.json"

ROUTE_INPUTS = "rerun_stage21_18_required_inputs"
ROUTE_CHECKPOINT = "repair_stage21_iterative_checkpoint_lineage"
ROUTE_BOUNDARY = "resolve_stage21_18_boundary_rejections"
ROUTE_NUMERICS = "continue_stage21_9_gradient_normalization_loss_scaling_repair"
ROUTE_POLICY_SOURCE = "repair_stage21_policy_update_signal_source"
ROUTE_MARGIN = "calibrate_stage21_discrete_action_margin_crossing"
ROUTE_CREDIT = "repair_stage21_return_advantage_credit_assignment"
ROUTE_SCALE = "scale_stage21_ppo_pilot_scenarios_and_horizon"
ROUTE_RUNTIME = "stage21_18_iterative_smoke_runtime_budget_blocked"

BOUNDARY_FIELDS = (
    "stage21_18_authorized",
    "training_or_release_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage21.18 iterative PPO learning curve audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        summary = run_xunce_stage21_18_iterative_ppo_learning_curve_audit(
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
                "completed_round_count": summary["completed_round_count"],
                "mean_abs_probability_delta_max": summary["mean_abs_probability_delta_max"],
                "coverage_auc_delta_max": summary["coverage_auc_delta_max"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] in {"passed", "partial"} else 1


def run_xunce_stage21_18_iterative_ppo_learning_curve_audit(
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
    rows: list[dict[str, Any]] = []
    if not input_reasons and not boundary_reasons:
        rows = _collect_round_results(config)
        _execute_missing_rounds(config, rows, repo_root)
        rows = _collect_round_results(config)

    lineage = _checkpoint_lineage(rows)
    action_trend = _action_trend(rows)
    coverage_trend = _coverage_trend(rows)
    status, route, primary_reason = _route(
        config=config,
        input_reasons=input_reasons,
        boundary_reasons=boundary_reasons,
        rows=rows,
        lineage=lineage,
        action_trend=action_trend,
        coverage_trend=coverage_trend,
    )
    recommended_config = _write_recommended_config(config, output_root, rows, repo_root)
    return _write_outputs(
        config=config,
        config_path=config_path,
        output_root=output_root,
        rows=rows,
        lineage=lineage,
        action_trend=action_trend,
        coverage_trend=coverage_trend,
        recommended_config=recommended_config,
        status=status,
        route=route,
        primary_reason=primary_reason,
        input_reasons=input_reasons,
        boundary_reasons=boundary_reasons,
    )


def _execute_missing_rounds(config: dict[str, Any], rows: list[dict[str, Any]], repo_root: Path) -> None:
    if not bool(config["execute_rounds"]):
        return
    executed = 0
    existing_by_round = _existing_seed_rounds(config)
    for round_index in range(1, int(config["iterative_round_count"]) + 1):
        if all(seed in existing_by_round.get(round_index, set()) for seed in config["seed_list"]):
            continue
        if executed >= int(config["max_new_rounds_to_execute"]):
            return
        for seed in config["seed_list"]:
            if seed in existing_by_round.get(round_index, set()):
                continue
            source_checkpoint = _source_checkpoint_for_seed_round(config, seed, round_index)
            if source_checkpoint is None:
                _write_json(
                    _chain_round_root(config, seed, round_index) / "runtime-blocker.json",
                    {"status": "blocked", "reason_code": ROUTE_CHECKPOINT, "round_index": round_index, "seed": seed},
                )
                continue
            _execute_seed_round(config, repo_root=repo_root, round_index=round_index, seed=seed, source_checkpoint=source_checkpoint)
        executed += 1
        existing_by_round = _existing_seed_rounds(config)


def _execute_seed_round(config: dict[str, Any], *, repo_root: Path, round_index: int, seed: int, source_checkpoint: Path) -> None:
    root = _chain_round_root(config, seed, round_index)
    root.mkdir(parents=True, exist_ok=True)
    generated = root / "generated_configs"
    generated.mkdir(parents=True, exist_ok=True)

    stage21_6_base = _read_json(Path(config["stage21_6_base_config"]))
    stage21_4_base = _read_json(Path(config["stage21_4_base_config"]))
    stage21_1_base = _read_json(_resolve_path(Path(stage21_6_base["stage21_1_base_config"]), repo_root))
    stage21_5_base = _read_json(_resolve_path(Path(stage21_6_base["stage21_5_base_config"]), repo_root))
    high_fidelity_base = _read_json(_resolve_path(Path(stage21_4_base["high_fidelity_config"]), repo_root))

    high_fidelity_base["xunce_candidate_checkpoint"] = str(source_checkpoint)
    high_fidelity_path = generated / "high_fidelity_config.json"
    _write_json(high_fidelity_path, high_fidelity_base)

    stage21_1_base["high_fidelity_config"] = str(high_fidelity_path)
    stage21_1_base["dynamic_validation_work_root"] = str(root / "_xunce_dynamic_validation_work_stage21_1")
    stage21_1_path = generated / "stage21_1_config.json"
    _write_json(stage21_1_path, stage21_1_base)

    stage21_4_base["high_fidelity_config"] = str(high_fidelity_path)
    stage21_4_base["xunce_candidate_checkpoint"] = str(source_checkpoint)
    stage21_4_path = generated / "stage21_4_config.json"
    _write_json(stage21_4_path, stage21_4_base)

    stage21_5_base["high_fidelity_config"] = str(high_fidelity_path)
    stage21_5_path = generated / "stage21_5_config.json"
    _write_json(stage21_5_path, stage21_5_base)

    stage21_6_base.update(
        {
            "stage21_1_base_config": str(stage21_1_path),
            "stage21_4_base_config": str(stage21_4_path),
            "stage21_5_base_config": str(stage21_5_path),
            "seed_list": [int(seed)],
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "execute_seed_pipeline": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    stage21_6_path = generated / "stage21_6_config.json"
    _write_json(stage21_6_path, stage21_6_base)
    try:
        _run_stage21_6(config=config, config_path=stage21_6_path, output_root=root / "stage21_6", repo_root=repo_root)
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - exercised by subprocess runtime
        _write_json(
            root / "runtime-blocker.json",
            {
                "status": "blocked",
                "reason_code": ROUTE_RUNTIME,
                "round_index": round_index,
                "timeout_seconds": int(config["per_round_timeout_seconds"]),
                "error": str(exc),
            },
        )
    except Exception as exc:  # pragma: no cover - defensive runtime artifact path
        _write_json(root / "runtime-blocker.json", {"status": "blocked", "reason_code": ROUTE_RUNTIME, "round_index": round_index, "error": str(exc)})


def _run_stage21_6(*, config: dict[str, Any], config_path: Path, output_root: Path, repo_root: Path) -> None:
    if str(config.get("stage21_6_execution_mode", "subprocess")) == "in_process":
        run_xunce_stage21_6_multi_seed_ppo_pilot(config_path=config_path, output_root=output_root, repo_root=repo_root)
        return
    result = subprocess.run(
        [
            sys.executable,
            str((repo_root / "scripts" / "run_xunce_stage21_6_multi_seed_ppo_pilot.py").resolve()),
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--repo-root",
            str(repo_root),
        ],
        cwd=str(repo_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=int(config["per_round_timeout_seconds"]),
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"Stage21.6 subprocess failed with exit code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _collect_round_results(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for round_index in range(1, int(config["iterative_round_count"]) + 1):
        seed_rows = [
            _read_seed_round_result(config, seed, round_index, _chain_round_root(config, seed, round_index) / "stage21_6")
            for seed in config["seed_list"]
        ]
        seed_rows = [row for row in seed_rows if row]
        if seed_rows:
            rows.append(_aggregate_round_result(config, round_index, seed_rows))
    return rows


def _read_seed_round_result(config: dict[str, Any], seed: int, round_index: int, stage21_6_root: Path) -> dict[str, Any]:
    summary = _read_json_or_empty(stage21_6_root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json")
    aggregate = _read_json_or_empty(stage21_6_root / "xunce-stage21-6-aggregate-metrics.json")
    lineage = _read_json_or_empty(stage21_6_root / "xunce-stage21-6-lineage-audit.json")
    seed_rows = _read_jsonl(stage21_6_root / "xunce-stage21-6-seed-results.jsonl")
    if not summary or not aggregate or not seed_rows:
        return {}
    boundary_reasons = s16._combo_boundary_reasons(summary, aggregate, seed_rows)
    signal = s16._combo_policy_signal(seed_rows)
    gradient = s16._combo_gradient_attribution(seed_rows)
    capped = s16._capped_aggregate(seed_rows)
    pre_clip = _float(aggregate.get("pre_clip_grad_norm_max"))
    post_clip = s16._max([_float(row.get("post_clip_grad_norm")) for row in seed_rows])
    kl = _float(aggregate.get("max_post_update_approx_kl_mean"))
    entropy = _float(aggregate.get("min_entropy_mean"))
    selected_checkpoint = _select_checkpoint(seed_rows)
    final_delta = _float(aggregate.get("final_coverage_delta_mean"))
    auc_delta = _float(aggregate.get("coverage_auc_delta_mean"))
    capped_final = _float(capped.get("final_coverage_rate_capped_delta_mean"))
    capped_auc = _float(capped.get("coverage_curve_auc_capped_delta_mean"))
    capped_final_min = _float(capped.get("final_coverage_rate_capped_delta_min"))
    capped_auc_min = _float(capped.get("coverage_curve_auc_capped_delta_min"))
    numerically_stable = bool(
        not boundary_reasons
        and lineage.get("passed") is True
        and pre_clip is not None
        and pre_clip <= float(config["stage21_6_pre_clip_grad_norm_gate"])
        and post_clip is not None
        and post_clip <= float(config["stage21_4_max_grad_norm"]) + 1.0e-5
        and kl is not None
        and abs(kl) <= 0.5
        and entropy is not None
        and entropy >= 0.0
        and selected_checkpoint.get("checkpoint_lineage_passed") is True
        and all(row.get("grad_norm_finite") is True for row in seed_rows)
    )
    coverage_improved = all(
        value is not None and value > 0.0
        for value in (final_delta, auc_delta, capped_final, capped_auc, capped_final_min, capped_auc_min)
    )
    return {
        "schema_version": ROUND_ROW_SCHEMA_VERSION,
        "round_index": round_index,
        "seed": seed,
        "stage21_6_root": str(stage21_6_root),
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
        "coverage_auc_delta_mean": auc_delta,
        "final_coverage_rate_capped_delta_mean": capped_final,
        "coverage_curve_auc_capped_delta_mean": capped_auc,
        "final_coverage_rate_capped_delta_min": capped_final_min,
        "coverage_curve_auc_capped_delta_min": capped_auc_min,
        "coverage_auc_improved": coverage_improved,
        "coverage_auc_not_regressed": all(
            value is not None and value >= 0.0
            for value in (final_delta, auc_delta, capped_final, capped_auc, capped_final_min, capped_auc_min)
        ),
        "numerically_stable": numerically_stable,
        **selected_checkpoint,
        **signal,
        **gradient,
    }


def _aggregate_round_result(config: dict[str, Any], round_index: int, seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stable = [row for row in seed_rows if row.get("numerically_stable")]
    round_improved_without_regression = any(row.get("coverage_auc_improved") for row in seed_rows) and all(
        row.get("coverage_auc_not_regressed") for row in seed_rows
    )
    return {
        "schema_version": ROUND_ROW_SCHEMA_VERSION,
        "round_index": round_index,
        "seed_count": len(seed_rows),
        "stage21_6_roots": [row.get("stage21_6_root") for row in seed_rows],
        "status": "passed" if round_improved_without_regression else "failed",
        "boundary_reason_codes": sorted({reason for row in seed_rows for reason in row.get("boundary_reason_codes", [])}),
        "lineage_passed": all(row.get("lineage_passed") is True for row in seed_rows),
        "trainable_transition_count_total": sum(_int(row.get("trainable_transition_count_total")) for row in seed_rows),
        "pre_clip_grad_norm_max": s16._max([_float(row.get("pre_clip_grad_norm_max")) for row in seed_rows]),
        "post_clip_grad_norm_max": s16._max([_float(row.get("post_clip_grad_norm_max")) for row in seed_rows]),
        "post_update_approx_kl_mean": s16._mean([_float(row.get("post_update_approx_kl_mean")) for row in seed_rows]),
        "min_entropy_mean": s16._min([_float(row.get("min_entropy_mean")) for row in seed_rows]),
        "parameter_delta_l2_mean": s16._mean([_float(row.get("parameter_delta_l2_mean")) for row in seed_rows]),
        "final_coverage_delta_mean": s16._mean([_float(row.get("final_coverage_delta_mean")) for row in seed_rows]),
        "coverage_auc_delta_mean": s16._mean([_float(row.get("coverage_auc_delta_mean")) for row in seed_rows]),
        "final_coverage_rate_capped_delta_mean": s16._mean([_float(row.get("final_coverage_rate_capped_delta_mean")) for row in seed_rows]),
        "coverage_curve_auc_capped_delta_mean": s16._mean([_float(row.get("coverage_curve_auc_capped_delta_mean")) for row in seed_rows]),
        "final_coverage_rate_capped_delta_min": s16._min([_float(row.get("final_coverage_rate_capped_delta_min")) for row in seed_rows]),
        "coverage_curve_auc_capped_delta_min": s16._min([_float(row.get("coverage_curve_auc_capped_delta_min")) for row in seed_rows]),
        "coverage_auc_not_regressed": all(row.get("coverage_auc_not_regressed") for row in seed_rows),
        "coverage_auc_improved": round_improved_without_regression,
        "numerically_stable": len(stable) == len(config["seed_list"]),
        "checkpoint_lineage_passed": all(row.get("checkpoint_lineage_passed") is True for row in seed_rows),
        "selected_checkpoints": [
            {
                "seed": row.get("seed"),
                "path": row.get("selected_experimental_checkpoint_path"),
                "sha256": row.get("selected_experimental_checkpoint_sha256"),
                "source_path": row.get("source_checkpoint_path"),
                "source_sha256": row.get("source_checkpoint_sha256"),
            }
            for row in seed_rows
        ],
        "strong_state_join_available_count": sum(_int(row.get("strong_state_join_available_count")) for row in seed_rows),
        "strong_state_binding_unavailable_count": sum(_int(row.get("strong_state_binding_unavailable_count")) for row in seed_rows),
        "mean_abs_probability_delta": s16._mean([_float(row.get("mean_abs_probability_delta")) for row in seed_rows]),
        "selected_action_probability_delta_mean": s16._mean([_float(row.get("selected_action_probability_delta_mean")) for row in seed_rows]),
        "selected_action_probability_abs_delta_mean": s16._mean([_float(row.get("selected_action_probability_abs_delta_mean")) for row in seed_rows]),
        "best_coverage_per_cost_probability_delta_mean": s16._mean([_float(row.get("best_coverage_per_cost_probability_delta_mean")) for row in seed_rows]),
        "argmax_changed_count": sum(_int(row.get("argmax_changed_count")) for row in seed_rows),
        "selected_action_changed_count": sum(_int(row.get("selected_action_changed_count")) for row in seed_rows),
        "selected_rank_changed_count": sum(_int(row.get("selected_rank_changed_count")) for row in seed_rows),
        "policy_grad_ratio_mean": s16._mean([_float(row.get("policy_grad_ratio_mean")) for row in seed_rows]),
        "value_to_policy_grad_norm_ratio_max": s16._max([_float(row.get("value_to_policy_grad_norm_ratio_max")) for row in seed_rows]),
    }


def _select_checkpoint(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for row in seed_rows:
        stage21_4_root = Path(str(row.get("stage21_4_root", "")))
        summary4 = _read_json_or_empty(stage21_4_root / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json")
        checkpoint = _read_json_or_empty(stage21_4_root / "xunce-stage21-4-checkpoint-audit.json")
        stage21_1_root = Path(str(row.get("stage21_1_root", "")))
        summary1 = _read_json_or_empty(stage21_1_root / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json")
        manifest1 = _read_json_or_empty(stage21_1_root / "xunce-stage21-1-manifest.json")
        summary_model_audit = summary1.get("model_audit") if isinstance(summary1.get("model_audit"), dict) else {}
        manifest_model_audit = manifest1.get("model_audit") if isinstance(manifest1.get("model_audit"), dict) else {}
        model_audit = summary_model_audit or manifest_model_audit
        collector_audit = model_audit.get("xunce_checkpoint_audit", {}) if isinstance(model_audit, dict) else {}
        path = str(summary4.get("experimental_checkpoint_path") or checkpoint.get("experimental_checkpoint_path") or "")
        sha = str(summary4.get("experimental_checkpoint_sha256") or checkpoint.get("experimental_checkpoint_sha256") or "")
        metadata = checkpoint.get("metadata") if isinstance(checkpoint.get("metadata"), dict) else {}
        reload_audit = checkpoint.get("reload_audit") if isinstance(checkpoint.get("reload_audit"), dict) else {}
        actual_sha = _sha256_file(Path(path)) if path and Path(path).is_file() else None
        source_sha = summary4.get("source_xunce_checkpoint_sha256") or metadata.get("source_checkpoint_sha256")
        source_path = summary4.get("source_xunce_candidate_checkpoint") or metadata.get("source_checkpoint_path")
        collector_path = collector_audit.get("checkpoint_path")
        collector_sha = collector_audit.get("checkpoint_sha256")
        passed = bool(
            path
            and sha
            and actual_sha == sha
            and checkpoint.get("checkpoint_reload_passed") is True
            and metadata.get("experimental_only") is True
            and reload_audit.get("checkpoint_loaded") is True
            and reload_audit.get("checkpoint_path")
            and str(reload_audit.get("checkpoint_path")) == path
            and reload_audit.get("checkpoint_sha256") == sha
            and source_sha
            and source_path
            and collector_path
            and collector_audit.get("checkpoint_loaded") is True
            and collector_sha
            and source_sha == collector_sha
            and str(collector_path) == str(source_path)
        )
        if passed:
            candidates.append(
                {
                    "seed": _int(row.get("seed")),
                    "path": path,
                    "sha": sha,
                    "source_path": source_path,
                    "source_sha": source_sha,
                    "coverage_auc": _float(row.get("coverage_auc_delta")),
                    "coverage": _float(row.get("final_coverage_delta")),
                }
            )
    if not candidates:
        return {
            "checkpoint_lineage_passed": False,
            "selected_checkpoint_seed": None,
            "selected_experimental_checkpoint_path": None,
            "selected_experimental_checkpoint_sha256": None,
        }
    candidates.sort(key=lambda item: (-(item["coverage_auc"] or 0.0), -(item["coverage"] or 0.0), item["seed"] or 0))
    selected = candidates[0]
    return {
        "checkpoint_lineage_passed": True,
        "selected_checkpoint_seed": selected["seed"],
        "selected_experimental_checkpoint_path": selected["path"],
        "selected_experimental_checkpoint_sha256": selected["sha"],
        "source_checkpoint_path": selected["source_path"],
        "source_checkpoint_sha256": selected["source_sha"],
    }


def _checkpoint_lineage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected_by_round = {
        int(row.get("round_index", 0)): row.get("selected_checkpoints", [])
        for row in rows
    }
    parent_mismatches: list[dict[str, Any]] = []
    for round_index in sorted(selected_by_round):
        if round_index <= 1:
            continue
        previous = {
            int(item.get("seed")): item
            for item in selected_by_round.get(round_index - 1, [])
            if item.get("seed") is not None
        }
        for item in selected_by_round.get(round_index, []):
            seed = int(item.get("seed")) if item.get("seed") is not None else None
            prior = previous.get(seed) if seed is not None else None
            if (
                not prior
                or prior.get("sha256") != item.get("source_sha256")
                or prior.get("path") != item.get("source_path")
            ):
                parent_mismatches.append(
                    {
                        "round_index": round_index,
                        "seed": seed,
                        "expected_parent_sha256": prior.get("sha256") if prior else None,
                        "expected_parent_path": prior.get("path") if prior else None,
                        "actual_source_sha256": item.get("source_sha256"),
                        "actual_source_path": item.get("source_path"),
                    }
                )
    return {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "round_count": len(rows),
        "checkpoint_lineage_passed": bool(
            rows
            and all(row.get("checkpoint_lineage_passed") is True for row in rows)
            and not parent_mismatches
        ),
        "parent_checkpoint_mismatch_count": len(parent_mismatches),
        "parent_checkpoint_mismatches": parent_mismatches,
        "selected_checkpoints": [
            {
                "round_index": row.get("round_index"),
                "selected_checkpoints": row.get("selected_checkpoints", []),
            }
            for row in rows
        ],
    }


def _action_trend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stable = [row for row in rows if row.get("numerically_stable")]
    return {
        "schema_version": ACTION_TREND_SCHEMA_VERSION,
        "stable_round_count": len(stable),
        "strong_state_join_available_count": sum(_int(row.get("strong_state_join_available_count")) for row in stable),
        "strong_state_binding_unavailable_count": sum(_int(row.get("strong_state_binding_unavailable_count")) for row in stable),
        "mean_abs_probability_delta_max": max((_float(row.get("mean_abs_probability_delta")) or 0.0 for row in stable), default=0.0),
        "best_coverage_per_cost_probability_delta_max": max((_float(row.get("best_coverage_per_cost_probability_delta_mean")) or 0.0 for row in stable), default=0.0),
        "argmax_changed_count": sum(_int(row.get("argmax_changed_count")) for row in stable),
        "selected_rank_changed_count": sum(_int(row.get("selected_rank_changed_count")) for row in stable),
        "selected_action_changed_count": sum(_int(row.get("selected_action_changed_count")) for row in stable),
    }


def _coverage_trend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stable = [row for row in rows if row.get("numerically_stable")]
    return {
        "schema_version": COVERAGE_TREND_SCHEMA_VERSION,
        "coverage_auc_delta_max": max((_float(row.get("coverage_auc_delta_mean")) or 0.0 for row in stable), default=0.0),
        "final_coverage_delta_max": max((_float(row.get("final_coverage_delta_mean")) or 0.0 for row in stable), default=0.0),
        "coverage_auc_improved_round_count": sum(1 for row in stable if row.get("coverage_auc_improved")),
        "coverage_auc_not_regressed_round_count": sum(1 for row in stable if row.get("coverage_auc_not_regressed")),
        "final_stable_round_improved": bool(stable and stable[-1].get("coverage_auc_improved")),
    }


def _route(
    *,
    config: dict[str, Any],
    input_reasons: list[str],
    boundary_reasons: list[str],
    rows: list[dict[str, Any]],
    lineage: dict[str, Any],
    action_trend: dict[str, Any],
    coverage_trend: dict[str, Any],
) -> tuple[str, str, str]:
    if input_reasons:
        return "failed", ROUTE_INPUTS, "stage21_18_input_rejected"
    if boundary_reasons:
        return "failed", ROUTE_BOUNDARY, "stage21_18_boundary_rejected"
    if not rows or len(rows) < int(config["min_completed_round_count"]):
        return "partial", ROUTE_RUNTIME, "stage21_18_min_rounds_not_completed"
    if lineage.get("checkpoint_lineage_passed") is not True:
        return "failed", ROUTE_CHECKPOINT, "checkpoint_lineage_incomplete"
    stable = [row for row in rows if row.get("numerically_stable")]
    if len(stable) < int(config["min_completed_round_count"]):
        return "failed", ROUTE_NUMERICS, "insufficient_numerically_stable_rounds"
    if int(action_trend["strong_state_join_available_count"]) <= 0:
        return "failed", ROUTE_CHECKPOINT, "strong_state_binding_unavailable"
    if bool(coverage_trend.get("final_stable_round_improved")):
        return "passed", ROUTE_SCALE, "iterative_ppo_learning_curve_improved"
    if float(action_trend["mean_abs_probability_delta_max"]) < float(config["probability_delta_threshold"]):
        return "failed", ROUTE_POLICY_SOURCE, "multi_round_probability_delta_below_threshold"
    if int(action_trend["argmax_changed_count"]) == 0 and int(action_trend["selected_rank_changed_count"]) == 0:
        return "failed", ROUTE_MARGIN, "probability_accumulated_without_rank_crossing"
    return "failed", ROUTE_CREDIT, "rank_or_argmax_changed_without_coverage_improvement"


def _write_recommended_config(config: dict[str, Any], output_root: Path, rows: list[dict[str, Any]], repo_root: Path) -> Path:
    payload = _read_json(Path(config["stage21_6_base_config"]))
    last = rows[-1] if rows else {}
    final_checkpoints = last.get("selected_checkpoints") if isinstance(last.get("selected_checkpoints"), list) else []
    selected_final = final_checkpoints[0] if final_checkpoints else None
    if selected_final and selected_final.get("path"):
        recommended_dir = output_root / "recommended_configs"
        recommended_dir.mkdir(parents=True, exist_ok=True)
        source_checkpoint = str(selected_final["path"])
        stage21_1 = _read_json(_resolve_path(Path(payload["stage21_1_base_config"]), repo_root))
        stage21_4 = _read_json(Path(config["stage21_4_base_config"]))
        stage21_5 = _read_json(_resolve_path(Path(payload["stage21_5_base_config"]), repo_root))
        high_fidelity = _read_json(_resolve_path(Path(stage21_4["high_fidelity_config"]), repo_root))
        high_fidelity["xunce_candidate_checkpoint"] = source_checkpoint
        high_fidelity_path = recommended_dir / "high_fidelity_config.json"
        _write_json(high_fidelity_path, high_fidelity)
        stage21_1["high_fidelity_config"] = str(high_fidelity_path)
        stage21_1_path = output_root / "xunce-stage21-18-recommended-stage21-1-config.json"
        _write_json(stage21_1_path, stage21_1)
        stage21_4["high_fidelity_config"] = str(high_fidelity_path)
        stage21_4["xunce_candidate_checkpoint"] = source_checkpoint
        stage21_4_path = output_root / "xunce-stage21-18-recommended-stage21-4-config.json"
        _write_json(stage21_4_path, stage21_4)
        stage21_5["high_fidelity_config"] = str(high_fidelity_path)
        stage21_5_path = output_root / "xunce-stage21-18-recommended-stage21-5-config.json"
        _write_json(stage21_5_path, stage21_5)
        payload["stage21_1_base_config"] = str(stage21_1_path)
        payload["stage21_4_base_config"] = str(stage21_4_path)
        payload["stage21_5_base_config"] = str(stage21_5_path)
    payload["recommended_config_requires_separate_human_execution"] = True
    payload["stage21_18_recommended"] = True
    payload["stage21_18_recommended_checkpoint_selection_policy"] = "first_seed_from_final_round_selected_checkpoints"
    payload["stage21_18_recommended_checkpoint_is_representative_only"] = True
    payload["stage21_18_final_round_per_seed_checkpoints"] = final_checkpoints
    payload["publishes_checkpoint"] = False
    payload["replaces_default_policy"] = False
    payload["connects_real_executor"] = False
    payload["starts_online_canary"] = False
    payload["canary_traffic_fraction"] = 0.0
    path = output_root / RECOMMENDED_CONFIG_FILE
    _write_json(path, payload)
    return path


def _write_outputs(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    rows: list[dict[str, Any]],
    lineage: dict[str, Any],
    action_trend: dict[str, Any],
    coverage_trend: dict[str, Any],
    recommended_config: Path,
    status: str,
    route: str,
    primary_reason: str,
    input_reasons: list[str],
    boundary_reasons: list[str],
) -> dict[str, Any]:
    _write_jsonl(output_root / ROUND_RESULTS_FILE, rows)
    _write_json(output_root / CHECKPOINT_LINEAGE_FILE, lineage)
    _write_json(output_root / ACTION_TREND_FILE, action_trend)
    _write_json(output_root / COVERAGE_TREND_FILE, coverage_trend)
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "primary_reason": primary_reason,
        "stage21_18_authorized": False,
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
        "stage21_17_root": config["stage21_17_root"],
        "completed_round_count": len(rows),
        "stable_round_count": action_trend["stable_round_count"],
        "input_reason_codes": input_reasons,
        "boundary_reason_codes": boundary_reasons,
        "checkpoint_lineage_passed": lineage["checkpoint_lineage_passed"],
        "strong_state_join_available_count": action_trend["strong_state_join_available_count"],
        "strong_state_binding_unavailable_count": action_trend["strong_state_binding_unavailable_count"],
        "mean_abs_probability_delta_max": action_trend["mean_abs_probability_delta_max"],
        "best_coverage_per_cost_probability_delta_max": action_trend["best_coverage_per_cost_probability_delta_max"],
        "argmax_changed_count": action_trend["argmax_changed_count"],
        "selected_rank_changed_count": action_trend["selected_rank_changed_count"],
        "selected_action_changed_count": action_trend["selected_action_changed_count"],
        "coverage_auc_delta_max": coverage_trend["coverage_auc_delta_max"],
        "final_coverage_delta_max": coverage_trend["final_coverage_delta_max"],
        "recommended_stage21_6_config": str(recommended_config),
        "stage21_18_authorized": False,
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
            "round_results": str(output_root / ROUND_RESULTS_FILE),
            "checkpoint_lineage": str(output_root / CHECKPOINT_LINEAGE_FILE),
            "action_signal_trend": str(output_root / ACTION_TREND_FILE),
            "coverage_trend": str(output_root / COVERAGE_TREND_FILE),
            "recommended_stage21_6_config": str(recommended_config),
            "routing": str(output_root / ROUTING_FILE),
            "report": str(output_root / REPORT_FILE),
        },
    }
    _write_json(output_root / MANIFEST_FILE, manifest)
    _write_report(output_root / REPORT_FILE, summary)
    return summary


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage21.18 Iterative PPO Learning Curve Audit",
        "",
        "Stage21.18 checks whether multiple small PPO updates accumulate into action probability, rank, argmax, or coverage/AUC movement.",
        "",
        f"- status: `{summary['status']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- primary_reason: `{summary['primary_reason']}`",
        f"- completed_round_count: `{summary['completed_round_count']}`",
        f"- mean_abs_probability_delta_max: `{summary['mean_abs_probability_delta_max']}`",
        f"- best_coverage_per_cost_probability_delta_max: `{summary['best_coverage_per_cost_probability_delta_max']}`",
        f"- argmax_changed_count: `{summary['argmax_changed_count']}`",
        f"- selected_rank_changed_count: `{summary['selected_rank_changed_count']}`",
        f"- coverage_auc_delta_max: `{summary['coverage_auc_delta_max']}`",
        "",
        "This is bounded offline smoke evidence only. It does not publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _source_checkpoint_for_seed_round(config: dict[str, Any], seed: int, round_index: int) -> Path | None:
    if round_index <= 1:
        return Path(str(config["initial_xunce_candidate_checkpoint"]))
    prior = _read_seed_round_result(config, seed, round_index - 1, _chain_round_root(config, seed, round_index - 1) / "stage21_6")
    if not prior:
        return None
    value = prior.get("selected_experimental_checkpoint_path")
    return Path(str(value)) if value else None


def _chain_round_root(config: dict[str, Any], seed: int, round_index: int) -> Path:
    return Path(config["round_work_root"]) / f"seed_{int(seed)}" / f"round_{round_index:03d}"


def _existing_seed_rounds(config: dict[str, Any]) -> dict[int, set[int]]:
    existing: dict[int, set[int]] = {}
    for round_index in range(1, int(config["iterative_round_count"]) + 1):
        for seed in config["seed_list"]:
            row = _read_seed_round_result(config, seed, round_index, _chain_round_root(config, seed, round_index) / "stage21_6")
            if row:
                existing.setdefault(round_index, set()).add(seed)
    return existing


def _input_reasons(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    summary = _read_json_or_empty(Path(config["stage21_17_root"]) / "xunce-stage21-17-summary.json")
    if not summary:
        reasons.append("missing_stage21_17_summary")
    elif summary.get("next_required_change") != "increase_stage21_policy_update_signal_strength_with_policy_loss_multiplier":
        reasons.append("stage21_17_route_not_iterative_policy_signal")
    for field in ("stage21_6_base_config", "stage21_4_base_config", "initial_xunce_candidate_checkpoint"):
        if not Path(str(config[field])).is_file():
            reasons.append(f"missing_{field}")
    return sorted(set(reasons))


def _boundary_reasons(config: dict[str, Any]) -> list[str]:
    reasons = [f"stage21_18_config_{field}_true" for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        reasons.append("stage21_18_config_canary_traffic_fraction_nonzero")
    return sorted(set(reasons))


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    for field in ("stage21_17_root", "stage21_6_base_config", "stage21_4_base_config", "round_work_root"):
        if not isinstance(config.get(field), str) or not str(config[field]).strip():
            raise ConfigError(f"{field} must be a non-empty path string")
        config[field] = str(_resolve_path(Path(str(config[field])), repo_root))
    stage21_4_base = _read_json(Path(config["stage21_4_base_config"])) if Path(config["stage21_4_base_config"]).is_file() else {}
    initial = config.get("initial_xunce_candidate_checkpoint") or stage21_4_base.get("xunce_candidate_checkpoint")
    if not initial:
        raise ConfigError("initial_xunce_candidate_checkpoint is required when stage21_4_base_config has no xunce_candidate_checkpoint")
    config["initial_xunce_candidate_checkpoint"] = str(_resolve_path(Path(str(initial)), repo_root))
    config["seed_list"] = [_positive_int(seed, "seed") for seed in config.get("seed_list", [2101, 2102, 2103])]
    for field in (
        "iterative_round_count",
        "min_completed_round_count",
        "max_new_rounds_to_execute",
        "per_round_timeout_seconds",
        "required_scenario_count",
        "rollout_steps",
        "dynamic_max_candidates_per_step",
        "dynamic_proposal_pool_limit_per_step",
    ):
        if field == "max_new_rounds_to_execute":
            config[field] = _nonnegative_int(config.get(field, 1), field)
        else:
            config[field] = _positive_int(config.get(field), field)
    config["stage21_6_pre_clip_grad_norm_gate"] = _positive_float(config.get("stage21_6_pre_clip_grad_norm_gate", 25.0), "stage21_6_pre_clip_grad_norm_gate")
    config["stage21_4_max_grad_norm"] = _positive_float(config.get("stage21_4_max_grad_norm", 1.0), "stage21_4_max_grad_norm")
    config["probability_delta_threshold"] = _positive_float(config.get("probability_delta_threshold", 0.005), "probability_delta_threshold")
    config["execute_rounds"] = bool(config.get("execute_rounds", True))
    config["stage21_6_execution_mode"] = str(config.get("stage21_6_execution_mode", "subprocess"))
    if config["stage21_6_execution_mode"] not in {"subprocess", "in_process"}:
        raise ConfigError("stage21_6_execution_mode must be subprocess or in_process")
    config["checkpoint_selection_policy"] = str(config.get("checkpoint_selection_policy", "best_capped_auc_then_lowest_seed"))
    config["canary_traffic_fraction"] = _nonnegative_float(config.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    for field in BOUNDARY_FIELDS:
        config.setdefault(field, False)
    return config


def _nonnegative_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be a non-negative integer") from exc
    if parsed < 0:
        raise ConfigError(f"{field} must be a non-negative integer")
    return parsed


def _nonnegative_float(value: Any, field: str) -> float:
    parsed = _float(value)
    if parsed is None or parsed < 0.0:
        raise ConfigError(f"{field} must be a non-negative finite float")
    return parsed


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
