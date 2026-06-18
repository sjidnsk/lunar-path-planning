from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
REPO_ROOT = SCRIPT_DIR.parent
MODEL_EXPLORER_SRC = REPO_ROOT / "model-explorer" / "src"
if MODEL_EXPLORER_SRC.is_dir() and str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))

try:
    from git_provenance import git_snapshot
    from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from global_99_governance_common import global_99_boundary_defaults
    from run_xunce_high_fidelity_real_map_comparison import (
        _boundary_audit as _stage18b_boundary_audit,
        _candidate_at,
        _candidate_cell,
        _candidate_cost,
        _candidate_is_valid,
        _candidate_rows,
        _finite_or_none,
        _float_default,
        _int_value,
        _load_model_bundle,
        _load_source,
        _mask_violation,
        _model_score,
        _policy_detail_to_dict,
        _scenario_to_model_inputs,
        _score_xunce_model,
        _selected_index,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from scripts.global_99_governance_common import global_99_boundary_defaults
    from scripts.run_xunce_high_fidelity_real_map_comparison import (
        _boundary_audit as _stage18b_boundary_audit,
        _candidate_at,
        _candidate_cell,
        _candidate_cost,
        _candidate_is_valid,
        _candidate_rows,
        _finite_or_none,
        _float_default,
        _int_value,
        _load_model_bundle,
        _load_source,
        _mask_violation,
        _model_score,
        _policy_detail_to_dict,
        _scenario_to_model_inputs,
        _score_xunce_model,
        _selected_index,
    )


CONFIG_SCHEMA_VERSION = "xunce-high-fidelity-exploration-coverage-comparison-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-exploration-coverage-comparison-summary/v1"
DEFAULT_CONFIG = "configs/xunce_high_fidelity_exploration_coverage_comparison_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_high_fidelity_exploration_coverage_comparison_v1"

SUMMARY_FILE = "xunce-exploration-coverage-comparison-summary.json"
EPISODES_FILE = "xunce-exploration-coverage-episodes.jsonl"
STEPS_FILE = "xunce-exploration-coverage-steps.jsonl"
MODEL_INFERENCE_FILE = "xunce-exploration-coverage-model-inference.jsonl"
ROI_BREAKDOWN_FILE = "xunce-exploration-coverage-roi-breakdown.json"
DECISION_AUDIT_FILE = "xunce-exploration-coverage-decision-audit.json"
MANIFEST_FILE = "xunce-exploration-coverage-comparison-manifest.json"
REPORT_FILE = "xunce-exploration-coverage-comparison-report.md"
V2_SUMMARY_FILE = "xunce-exploration-coverage-comparison-v2-summary.json"
V2_EPISODES_FILE = "xunce-exploration-coverage-v2-episodes.jsonl"
V2_STEPS_FILE = "xunce-exploration-coverage-v2-steps.jsonl"

FIX_XUNCE_CHECKPOINT_NEXT_REQUIRED_CHANGE = "fix_xunce_sandbox_candidate_preflight"
FIX_INCUMBENT_CHECKPOINT_NEXT_REQUIRED_CHANGE = "fix_incumbent_policy_checkpoint"
FIX_RUNNER_NEXT_REQUIRED_CHANGE = "fix_xunce_coverage_comparison_runner"
BOUNDARY_NEXT_REQUIRED_CHANGE = "resolve_xunce_coverage_boundary_rejections"
EFFICIENCY_NEXT_REQUIRED_CHANGE = "refine_coverage_reward_and_cost_guard"
ADVANTAGE_NEXT_REQUIRED_CHANGE = "xunce_default_policy_candidate_authorization_preflight"
NO_ADVANTAGE_NEXT_REQUIRED_CHANGE = "xunce_research_iteration_required"
FIX_SOURCE_NEXT_REQUIRED_CHANGE = "fix_xunce_high_fidelity_real_map_roi_expansion"
REVIEW_COMPARISON_NEXT_REQUIRED_CHANGE = "review_xunce_incumbent_comparison_metrics"

POLICIES = ("xunce", "incumbent")
ORACLE_POLICIES = ("greedy_coverage_oracle", "cost_aware_coverage_oracle")
TOLERANCE = 1.0e-12

BOUNDARY_FIELDS = tuple(global_99_boundary_defaults()) + (
    "real_world_release_approved",
    "real_world_performance_claimed",
    "default_policy_replacement_approved",
    "real_executor_connection_approved",
    "starts_online_canary",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Xunce high-fidelity exploration coverage comparison v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--source-roi-expansion-root")
    parser.add_argument("--xunce-candidate-checkpoint")
    parser.add_argument("--incumbent-policy-checkpoint")
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--candidate-refresh-mode")
    parser.add_argument("--coverage-metric-mode")
    parser.add_argument("--include-oracle-baselines", action="store_true")
    parser.add_argument("--include-roi-weighted-coverage", action="store_true")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    overrides = {
        key: value
        for key, value in {
            "source_roi_expansion_root": args.source_roi_expansion_root,
            "xunce_candidate_checkpoint": args.xunce_candidate_checkpoint,
            "incumbent_policy_checkpoint": args.incumbent_policy_checkpoint,
            "rollout_steps": args.rollout_steps,
            "candidate_refresh_mode": args.candidate_refresh_mode,
            "coverage_metric_mode": args.coverage_metric_mode,
            "include_oracle_baselines": True if args.include_oracle_baselines else None,
            "include_roi_weighted_coverage": True if args.include_roi_weighted_coverage else None,
        }.items()
        if value is not None
    }
    try:
        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
            config_overrides=overrides,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "xunce_coverage_advantage_established": summary["xunce_coverage_advantage_established"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_high_fidelity_exploration_coverage_comparison(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path, repo_root, config_overrides=config_overrides)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_root)

    source = _load_source(config, repo_root)
    boundary = _boundary_audit(config, source)
    model_bundle = _load_model_bundle(config, source, repo_root)
    source_match = _source_match_audit(config, source)
    episodes, steps, inference_rows = _run_coverage_rollouts(config, source, model_bundle)
    comparison = _coverage_comparison_audit(episodes, steps)
    roi_breakdown = _roi_breakdown(episodes)
    model_inference = _model_inference_audit(model_bundle, inference_rows)
    decision = _decision(
        source=source,
        boundary=boundary,
        source_match=source_match,
        model_inference=model_inference,
        comparison=comparison,
    )
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=config_path,
        output_root=output_root,
        paths=paths,
        config=config,
        source=source,
        boundary=boundary,
        source_match=source_match,
        model_inference=model_inference,
        comparison=comparison,
        roi_breakdown=roi_breakdown,
        decision=decision,
        repo_root=repo_root,
    )
    manifest = {
        "schema_version": "xunce-exploration-coverage-comparison-manifest/v1",
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "artifacts": {key: str(value) for key, value in paths.items()},
        "summary_status": summary["status"],
        "next_required_change": summary["next_required_change"],
    }

    write_jsonl(paths["episodes"], [_public_episode(row) for row in episodes])
    write_jsonl(paths["steps"], steps)
    write_jsonl(paths["model_inference"], inference_rows)
    write_json(paths["roi_breakdown"], roi_breakdown)
    write_json(paths["decision_audit"], decision)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    if _v2_enabled(config):
        write_json(paths["v2_summary"], summary)
        write_jsonl(paths["v2_episodes"], [_public_episode(row) for row in episodes])
        write_jsonl(paths["v2_steps"], steps)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _load_config(
    path: Path,
    repo_root: Path,
    *,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    if config_overrides:
        payload = {**payload, **config_overrides}
    normalized = dict(payload)
    for key in ("source_roi_expansion_root", "xunce_candidate_checkpoint", "incumbent_policy_checkpoint"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(payload[key]), repo_root))
    normalized["required_scenario_count"] = _positive_int(payload.get("required_scenario_count", 24), "required_scenario_count")
    normalized["rollout_steps"] = _positive_int(payload.get("rollout_steps", 10), "rollout_steps")
    normalized["coverage_radius_cells"] = _nonnegative_int(payload.get("coverage_radius_cells", 1), "coverage_radius_cells")
    sensitivity = payload.get("coverage_radius_sensitivity", [3])
    if not isinstance(sensitivity, list) or any(not isinstance(item, int) or item < 0 for item in sensitivity):
        raise ConfigError("coverage_radius_sensitivity must be a list of non-negative integers")
    normalized["coverage_radius_sensitivity"] = list(sensitivity)
    normalized["coverage_denominator_cells"] = _positive_int(payload.get("coverage_denominator_cells", 1000), "coverage_denominator_cells")
    normalized["path_budget_m"] = _positive_float(payload.get("path_budget_m", 5000.0), "path_budget_m")
    normalized["planning_backend"] = _require_string(payload.get("planning_backend", "channel_aware_astar"), "planning_backend")
    normalized["candidate_refresh_mode"] = _require_string(payload.get("candidate_refresh_mode", "static_from_source"), "candidate_refresh_mode")
    if normalized["candidate_refresh_mode"] not in {"static_from_source", "dynamic_from_coverage_memory"}:
        raise ConfigError("candidate_refresh_mode must be static_from_source or dynamic_from_coverage_memory")
    normalized["coverage_metric_mode"] = _require_string(payload.get("coverage_metric_mode", "endpoint_footprint"), "coverage_metric_mode")
    if normalized["coverage_metric_mode"] not in {"endpoint_footprint", "path_line_plus_endpoint"}:
        raise ConfigError("coverage_metric_mode must be endpoint_footprint or path_line_plus_endpoint")
    normalized["include_oracle_baselines"] = _require_bool(payload.get("include_oracle_baselines", False), "include_oracle_baselines")
    normalized["include_roi_weighted_coverage"] = _require_bool(payload.get("include_roi_weighted_coverage", False), "include_roi_weighted_coverage")
    normalized["allow_open_grid_fallback"] = _require_bool(payload.get("allow_open_grid_fallback", False), "allow_open_grid_fallback")
    normalized["require_context_ids"] = _require_bool(payload.get("require_context_ids", True), "require_context_ids")
    normalized["require_contract_and_sidecar_paths"] = _require_bool(payload.get("require_contract_and_sidecar_paths", True), "require_contract_and_sidecar_paths")
    normalized["max_xunce_parameter_count"] = _positive_int(payload.get("max_xunce_parameter_count", 10_000_000), "max_xunce_parameter_count")
    normalized["max_median_inference_latency_ms"] = _nonnegative_float(payload.get("max_median_inference_latency_ms", 5.0), "max_median_inference_latency_ms")
    normalized["max_latency_ratio_vs_incumbent"] = _nonnegative_float(payload.get("max_latency_ratio_vs_incumbent", 10.0), "max_latency_ratio_vs_incumbent")
    normalized["canary_traffic_fraction"] = _nonnegative_float(payload.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return normalized


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "episodes": output_root / EPISODES_FILE,
        "steps": output_root / STEPS_FILE,
        "model_inference": output_root / MODEL_INFERENCE_FILE,
        "roi_breakdown": output_root / ROI_BREAKDOWN_FILE,
        "decision_audit": output_root / DECISION_AUDIT_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "report": output_root / REPORT_FILE,
        "v2_summary": output_root / V2_SUMMARY_FILE,
        "v2_episodes": output_root / V2_EPISODES_FILE,
        "v2_steps": output_root / V2_STEPS_FILE,
    }


def _v2_enabled(config: dict[str, Any]) -> bool:
    return (
        config["candidate_refresh_mode"] == "dynamic_from_coverage_memory"
        or config["coverage_metric_mode"] == "path_line_plus_endpoint"
        or bool(config["include_oracle_baselines"])
        or bool(config["include_roi_weighted_coverage"])
    )


def _run_coverage_rollouts(
    config: dict[str, Any],
    source: dict[str, Any],
    model_bundle: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    scenarios = source["path_feedback"].get("scenarios", [])
    if not isinstance(scenarios, list):
        scenarios = []
    slice_by_id = {str(row.get("scenario_id")): row for row in source["slices"] if isinstance(row, dict)}
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios[: config["required_scenario_count"]]):
        if not isinstance(scenario, dict):
            continue
        scenario_id = str(scenario.get("scenario_id", f"scenario-{scenario_index:04d}"))
        slice_row = slice_by_id.get(scenario_id, {})
        roi_group = str(slice_row.get("roi_name") or scenario.get("roi_group") or scenario.get("scenario_group") or "unknown")
        policy_names = POLICIES + (ORACLE_POLICIES if config["include_oracle_baselines"] else ())
        for policy_name in policy_names:
            episode = _run_policy_episode(
                policy_name=policy_name,
                scenario=scenario,
                scenario_index=scenario_index,
                scenario_id=scenario_id,
                roi_group=roi_group,
                split=slice_row.get("split"),
                config=config,
                model_bundle=model_bundle,
            )
            episodes.append(episode)
            steps.extend(episode.get("steps", []))
            inference_rows.extend(episode.get("inference_rows", []))
    return episodes, steps, inference_rows


def _run_policy_episode(
    *,
    policy_name: str,
    scenario: dict[str, Any],
    scenario_index: int,
    scenario_id: str,
    roi_group: str,
    split: Any,
    config: dict[str, Any],
    model_bundle: dict[str, Any],
) -> dict[str, Any]:
    denominator = float(config["coverage_denominator_cells"])
    radius = int(config["coverage_radius_cells"])
    start_cell = _cell_tuple(scenario.get("start_cell")) or (0, 0)
    covered_cells = set(_footprint(start_cell, radius=radius))
    coverage_rates = [len(covered_cells) / denominator]
    steps: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    reason_codes: list[str] = []
    path_cost_total = 0.0
    risk_total = 0.0
    energy_total = 0.0
    new_cell_total = 0
    revisited_cell_total = 0
    coverage_return = 0.0
    valuable_area_covered = 0.0
    selected_probabilities: list[float] = []
    selected_ranks: list[int] = []
    entropies: list[float] = []
    latencies: list[float] = []
    action_indices: list[int | None] = []
    mask_violation_count = 0
    unreachable_selected_count = 0
    path_planning_failure_count = 0
    open_grid_fallback_count = 0
    current_cell = start_cell

    for step_index in range(config["rollout_steps"]):
        cell_before = current_cell
        candidates = _candidate_rows_for_step(
            scenario,
            current_cell=current_cell,
            covered_cells=covered_cells,
            step_index=step_index,
            config=config,
        )
        scenario_state = dict(scenario)
        scenario_state["coverage_rate"] = coverage_rates[-1]
        scenario_state["coverage_rate_delta"] = steps[-1]["coverage_rate_delta"] if steps else 0.0
        adapter = _scenario_to_model_inputs(
            scenario_state,
            candidates,
            scenario_index + step_index,
            model_bundle.get("xunce_config") or {},
        )
        step_reasons: list[str] = []
        detail: dict[str, Any] | None = None
        true_model_inference_executed = False
        if policy_name in ORACLE_POLICIES:
            selected_index = _oracle_selected_index(
                policy_name,
                candidates,
                current_cell=current_cell,
                covered_cells=covered_cells,
                config=config,
            )
            true_model_inference_executed = False
            detail = {
                "logits": [],
                "masked_logits": [],
                "action_probs": [],
                "value": 0.0,
                "selected_action_index": selected_index,
                "selected_probability": 1.0 if selected_index is not None else 0.0,
                "selected_rank": 1 if selected_index is not None else 0,
                "finite_outputs": True,
                "latency_ms": 0.0,
            }
        elif not model_bundle["xunce_checkpoint_loaded"] or not model_bundle["incumbent_checkpoint_loaded"]:
            step_reasons.append("true_model_inference_not_executed")
        elif not adapter["has_valid_action"]:
            step_reasons.append("no_valid_action")
        else:
            try:
                if policy_name == "xunce":
                    detail = _score_xunce_model(model_bundle["xunce_model"], adapter["xunce_batch"])
                else:
                    detail = _policy_detail_to_dict(model_bundle["incumbent_scorer"].score_detail(adapter["incumbent_observation"]))
                true_model_inference_executed = True
            except Exception as exc:  # pragma: no cover
                step_reasons.append(f"true_model_inference_failed:{type(exc).__name__}")
                step_reasons.append("true_model_inference_not_executed")

        detail_payload = detail or _empty_model_detail()
        selected_index = detail["selected_action_index"] if policy_name in ORACLE_POLICIES and detail is not None else _selected_index(detail)
        selected_candidate = _candidate_at(candidates, selected_index)
        selected_cell = _cell_tuple(_candidate_cell(selected_candidate)) if selected_candidate is not None else None
        selected_cost = _candidate_cost(selected_candidate) if selected_candidate is not None else None
        selected_risk = _finite_or_none(selected_candidate.get("risk")) if selected_candidate is not None else None
        selected_energy = _finite_or_none(selected_candidate.get("energy_cost")) if selected_candidate is not None else None
        selected_value = _finite_or_none(selected_candidate.get("value")) if selected_candidate is not None else None
        mask_violation = _mask_violation(adapter["action_mask"], selected_index)
        if mask_violation:
            mask_violation_count += 1
            step_reasons.append("model_inference_mask_violation")
        if selected_candidate is None or selected_cell is None:
            path_planning_failure_count += 1
            step_reasons.append("path_planning_failure")
        elif not _candidate_is_valid(selected_candidate):
            unreachable_selected_count += 1
            step_reasons.append("unreachable_selected_candidate")
        if (scenario.get("open_grid_fallback_used") is True or (selected_candidate or {}).get("open_grid_fallback_used") is True) and not config["allow_open_grid_fallback"]:
            open_grid_fallback_count += 1
            step_reasons.append("open_grid_fallback_forbidden")
        if selected_cost is not None and path_cost_total + float(selected_cost) > float(config["path_budget_m"]):
            step_reasons.append("path_budget_exhausted")

        executed = (
            true_model_inference_executed
            and detail_payload["finite_outputs"]
            and selected_candidate is not None
            and selected_cell is not None
            and selected_cost is not None
            and not mask_violation
            and "unreachable_selected_candidate" not in step_reasons
            and "open_grid_fallback_forbidden" not in step_reasons
            and "path_budget_exhausted" not in step_reasons
        )
        new_cells: set[tuple[int, int]] = set()
        revisited_cells: set[tuple[int, int]] = set()
        coverage_delta = 0.0
        if executed and selected_cell is not None and selected_cost is not None:
            footprint = _coverage_cells(
                start=cell_before,
                end=selected_cell,
                radius=radius,
                mode=str(config["coverage_metric_mode"]),
            )
            new_cells = footprint - covered_cells
            revisited_cells = footprint & covered_cells
            covered_cells.update(footprint)
            current_cell = selected_cell
            path_cost_total += float(selected_cost)
            risk_total += _float_default(selected_risk)
            energy_total += _float_default(selected_energy)
            new_cell_total += len(new_cells)
            revisited_cell_total += len(revisited_cells)
            coverage_delta = len(new_cells) / denominator
            coverage_return += coverage_delta
            valuable_area_covered += len(new_cells) * _float_default(selected_value)
        coverage_rates.append(len(covered_cells) / denominator)
        if detail is not None:
            selected_probabilities.append(float(detail_payload["selected_probability"]))
            selected_ranks.append(int(detail_payload["selected_rank"]))
            entropies.append(_entropy(detail_payload["action_probs"]))
            latencies.append(float(detail_payload["latency_ms"]))
        action_indices.append(selected_index)

        cumulative_delta = coverage_rates[-1] - coverage_rates[0]
        step_row = {
            "schema_version": "xunce-exploration-coverage-step/v1",
            "scenario_id": scenario_id,
            "roi_group": roi_group,
            "split": split,
            "policy": policy_name,
            "step_index": step_index,
            "current_cell_before": list(cell_before),
            "selected_action_index": selected_index,
            "selected_cell": list(selected_cell) if selected_cell is not None else None,
            "candidate_cells": [_candidate_cell(candidate) for candidate in candidates],
            "selected_probability": detail_payload["selected_probability"],
            "selected_rank": detail_payload["selected_rank"],
            "action_entropy": entropies[-1] if entropies else 0.0,
            "path_cost": selected_cost,
            "risk": selected_risk,
            "energy_cost": selected_energy,
            "coverage_rate_delta": coverage_delta,
            "cumulative_coverage_rate_delta": cumulative_delta,
            "final_coverage_rate": coverage_rates[-1],
            "new_covered_cell_count": len(new_cells),
            "revisited_cell_count": len(revisited_cells),
            "coverage_gain_per_path_cost": _safe_ratio(coverage_delta, selected_cost),
            "coverage_gain_per_risk": _safe_ratio(coverage_delta, selected_risk),
            "true_model_inference_executed": true_model_inference_executed,
            "finite_outputs": detail_payload["finite_outputs"],
            "model_inference_mask_violation": mask_violation,
            "executed": executed,
            "reason_codes": unique_sorted(step_reasons),
        }
        steps.append(step_row)
        if policy_name not in ORACLE_POLICIES:
            inference_rows.append(
                {
                    "schema_version": "xunce-exploration-coverage-model-inference/v1",
                    "input_source": "high_fidelity_scenario_adapter/v1",
                    "scenario_id": scenario_id,
                    "roi_group": roi_group,
                    "policy": policy_name,
                    "step_index": step_index,
                    "candidate_cells": [_candidate_cell(candidate) for candidate in candidates],
                    "action_mask": list(adapter["action_mask"]),
                    "selected_action_index": selected_index,
                    "true_model_inference_executed": true_model_inference_executed,
                    "model_inference_mask_violation_count": int(mask_violation),
                    "detail": detail_payload,
                    "reason_codes": unique_sorted(step_reasons),
                }
            )
        reason_codes.extend(step_reasons)
        if not executed:
            break

    executed_step_count = sum(1 for row in steps if row["executed"])
    coverage_curve_auc = _coverage_curve_auc(coverage_rates, config["rollout_steps"])
    cumulative_delta = coverage_rates[-1] - coverage_rates[0]
    total_seen_cells = new_cell_total + revisited_cell_total
    return {
        "schema_version": "xunce-exploration-coverage-episode/v1",
        "scenario_id": scenario_id,
        "roi_group": roi_group,
        "split": split,
        "policy": policy_name,
        "rollout_steps": config["rollout_steps"],
        "executed_step_count": executed_step_count,
        "initial_coverage_rate": coverage_rates[0],
        "final_coverage_rate": coverage_rates[-1],
        "coverage_rate_delta": cumulative_delta,
        "cumulative_coverage_rate_delta": cumulative_delta,
        "coverage_return": coverage_return,
        "coverage_curve_auc": coverage_curve_auc,
        "new_covered_cell_count": new_cell_total,
        "revisited_cell_count": revisited_cell_total,
        "revisit_rate": _safe_ratio(revisited_cell_total, total_seen_cells) or 0.0,
        "coverage_overlap_ratio": _safe_ratio(revisited_cell_total, total_seen_cells) or 0.0,
        "min_roi_group_coverage_rate": coverage_rates[-1],
        "valuable_area_covered": valuable_area_covered,
        "path_cost": path_cost_total,
        "risk": risk_total,
        "energy_cost": energy_total,
        "coverage_gain_per_path_cost": _safe_ratio(coverage_return, path_cost_total),
        "coverage_gain_per_meter": _safe_ratio(coverage_return, path_cost_total),
        "coverage_gain_per_risk": _safe_ratio(coverage_return, risk_total),
        "coverage_gain_per_energy": _safe_ratio(coverage_return, energy_total),
        "path_budget_used_ratio": _safe_ratio(path_cost_total, config["path_budget_m"]) or 0.0,
        "latency_per_coverage_gain": _safe_ratio(sum(latencies), coverage_return),
        "selected_probability_median": statistics.median(selected_probabilities) if selected_probabilities else 0.0,
        "selected_rank_median": statistics.median(selected_ranks) if selected_ranks else 0.0,
        "action_entropy_median": statistics.median(entropies) if entropies else 0.0,
        "median_inference_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "mask_violation_count": mask_violation_count,
        "unreachable_selected_count": unreachable_selected_count,
        "path_planning_failure_count": path_planning_failure_count,
        "open_grid_fallback_count": open_grid_fallback_count,
        "reason_codes": unique_sorted(reason_codes),
        "action_indices": action_indices,
        "steps": steps,
        "inference_rows": inference_rows,
    }


def _coverage_comparison_audit(episodes: list[dict[str, Any]], steps: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for episode in episodes:
        pairs[str(episode["scenario_id"])][str(episode["policy"])] = episode
    xunce_better = 0
    xunce_worse = 0
    xunce_tie = 0
    efficiency_regression = 0
    safety_regression = 0
    risk_regression = 0
    path_cost_regression = 0
    coverage_return_deltas: list[float] = []
    coverage_auc_deltas: list[float] = []
    final_coverage_deltas: list[float] = []
    new_cell_deltas: list[float] = []
    min_roi_deltas: list[float] = []
    gain_per_cost_deltas: list[float] = []
    gain_per_risk_deltas: list[float] = []
    for policies in pairs.values():
        xunce = policies.get("xunce")
        incumbent = policies.get("incumbent")
        if not xunce or not incumbent:
            continue
        coverage_delta = float(xunce["coverage_return"]) - float(incumbent["coverage_return"])
        coverage_return_deltas.append(coverage_delta)
        coverage_auc_deltas.append(float(xunce["coverage_curve_auc"]) - float(incumbent["coverage_curve_auc"]))
        final_coverage_deltas.append(float(xunce["final_coverage_rate"]) - float(incumbent["final_coverage_rate"]))
        new_cell_deltas.append(float(xunce["new_covered_cell_count"]) - float(incumbent["new_covered_cell_count"]))
        min_roi_deltas.append(float(xunce["min_roi_group_coverage_rate"]) - float(incumbent["min_roi_group_coverage_rate"]))
        cost_delta = _none_to_zero(xunce.get("coverage_gain_per_path_cost")) - _none_to_zero(incumbent.get("coverage_gain_per_path_cost"))
        risk_delta = _none_to_zero(xunce.get("coverage_gain_per_risk")) - _none_to_zero(incumbent.get("coverage_gain_per_risk"))
        gain_per_cost_deltas.append(cost_delta)
        gain_per_risk_deltas.append(risk_delta)
        if coverage_delta > TOLERANCE:
            xunce_better += 1
        elif coverage_delta < -TOLERANCE:
            xunce_worse += 1
        else:
            xunce_tie += 1
        if float(xunce["risk"]) > float(incumbent["risk"]) + TOLERANCE:
            risk_regression += 1
        if float(xunce["path_cost"]) > float(incumbent["path_cost"]) + TOLERANCE:
            path_cost_regression += 1
        if cost_delta < -TOLERANCE or risk_delta < -TOLERANCE or float(xunce["path_cost"]) > float(incumbent["path_cost"]) + TOLERANCE:
            efficiency_regression += 1
        if (
            int(xunce["mask_violation_count"]) > int(incumbent["mask_violation_count"])
            or int(xunce["unreachable_selected_count"]) > int(incumbent["unreachable_selected_count"])
            or int(xunce["path_planning_failure_count"]) > int(incumbent["path_planning_failure_count"])
            or int(xunce["open_grid_fallback_count"]) > int(incumbent["open_grid_fallback_count"])
            or float(xunce["risk"]) > float(incumbent["risk"]) + TOLERANCE
        ):
            safety_regression += 1
    disagreement_count, useful_disagreement_count = _disagreement_counts(steps)
    xunce_episodes = [row for row in episodes if row["policy"] == "xunce"]
    incumbent_episodes = [row for row in episodes if row["policy"] == "incumbent"]
    oracle_episodes = [row for row in episodes if row["policy"] == "greedy_coverage_oracle"]
    oracle_by_scenario = {str(row["scenario_id"]): row for row in oracle_episodes}
    xunce_oracle_regrets = []
    incumbent_oracle_regrets = []
    oracle_vs_incumbent_deltas = []
    for policies in pairs.values():
        xunce = policies.get("xunce")
        incumbent = policies.get("incumbent")
        if not xunce or not incumbent:
            continue
        oracle = oracle_by_scenario.get(str(xunce["scenario_id"]))
        if not oracle:
            continue
        oracle_return = float(oracle["coverage_return"])
        xunce_oracle_regrets.append(max(0.0, oracle_return - float(xunce["coverage_return"])))
        incumbent_oracle_regrets.append(max(0.0, oracle_return - float(incumbent["coverage_return"])))
        oracle_vs_incumbent_deltas.append(oracle_return - float(incumbent["coverage_return"]))
    evaluation_task_discriminative = bool(
        oracle_vs_incumbent_deltas
        and _mean(oracle_vs_incumbent_deltas) > TOLERANCE
        and useful_disagreement_count > 0
        and sum(1 for row in episodes if row["policy"] == "greedy_coverage_oracle") > 0
    )
    return {
        "schema_version": "xunce-exploration-coverage-comparison-audit/v1",
        "scenario_count": len(pairs),
        "xunce_coverage_better_count": xunce_better,
        "xunce_coverage_worse_count": xunce_worse,
        "xunce_coverage_tie_count": xunce_tie,
        "xunce_efficiency_regression_count": efficiency_regression,
        "xunce_safety_regression_count": safety_regression,
        "risk_regression_count": risk_regression,
        "path_cost_regression_count": path_cost_regression,
        "xunce_oracle_regret": _mean(xunce_oracle_regrets),
        "incumbent_oracle_regret": _mean(incumbent_oracle_regrets),
        "oracle_vs_incumbent_coverage_delta": _mean(oracle_vs_incumbent_deltas),
        "evaluation_task_discriminative": evaluation_task_discriminative,
        "xunce_final_coverage_rate_delta_vs_incumbent": _mean(final_coverage_deltas),
        "xunce_coverage_return_delta_vs_incumbent": _mean(coverage_return_deltas),
        "xunce_coverage_curve_auc_delta_vs_incumbent": _mean(coverage_auc_deltas),
        "xunce_new_covered_cell_delta_vs_incumbent": _mean(new_cell_deltas),
        "xunce_min_roi_group_coverage_delta_vs_incumbent": _mean(min_roi_deltas),
        "coverage_gain_per_path_cost_delta_vs_incumbent": _mean(gain_per_cost_deltas),
        "coverage_gain_per_risk_delta_vs_incumbent": _mean(gain_per_risk_deltas),
        "policy_disagreement_count": disagreement_count,
        "useful_disagreement_count": useful_disagreement_count,
        "selected_probability_median": _median([row["selected_probability_median"] for row in episodes]),
        "action_entropy_median": _median([row["action_entropy_median"] for row in episodes]),
        "selected_rank_median": _median([row["selected_rank_median"] for row in episodes]),
        "xunce_median_inference_latency_ms": _median([row["median_inference_latency_ms"] for row in xunce_episodes]),
        "incumbent_median_inference_latency_ms": _median([row["median_inference_latency_ms"] for row in incumbent_episodes]),
        "xunce_open_grid_fallback_count": sum(int(row["open_grid_fallback_count"]) for row in xunce_episodes),
        "open_grid_fallback_count": sum(int(row["open_grid_fallback_count"]) for row in episodes),
        "unreachable_selected_count": sum(int(row["unreachable_selected_count"]) for row in episodes),
        "path_planning_failure_count": sum(int(row["path_planning_failure_count"]) for row in episodes),
    }


def _model_inference_audit(model_bundle: dict[str, Any], inference_rows: list[dict[str, Any]]) -> dict[str, Any]:
    reason_codes = list(model_bundle["reason_codes"])
    if not model_bundle["xunce_checkpoint_loaded"] or not model_bundle["incumbent_checkpoint_loaded"]:
        reason_codes.append("true_model_inference_not_executed")
    executed_rows = [row for row in inference_rows if row.get("true_model_inference_executed")]
    finite_count = sum(1 for row in inference_rows if row.get("detail", {}).get("finite_outputs") is True)
    mask_violation_count = sum(_int_value(row.get("model_inference_mask_violation_count")) for row in inference_rows)
    if inference_rows and len(executed_rows) != len(inference_rows):
        reason_codes.append("true_model_inference_not_executed")
    if inference_rows and finite_count != len(inference_rows):
        reason_codes.append("model_inference_non_finite_output")
    if mask_violation_count:
        reason_codes.append("model_inference_mask_violation")
    true_model_inference_executed = bool(
        inference_rows
        and len(executed_rows) == len(inference_rows)
        and model_bundle["xunce_checkpoint_loaded"]
        and model_bundle["incumbent_checkpoint_loaded"]
    )
    return {
        "schema_version": "xunce-exploration-coverage-model-inference-audit/v1",
        "true_model_inference_executed": true_model_inference_executed,
        "proxy_selection_used": False,
        "input_source": "high_fidelity_scenario_adapter/v1",
        "xunce_checkpoint_loaded": model_bundle["xunce_checkpoint_loaded"],
        "incumbent_checkpoint_loaded": model_bundle["incumbent_checkpoint_loaded"],
        "xunce_parameter_count": model_bundle["xunce_parameter_count"],
        "incumbent_parameter_count": model_bundle["incumbent_parameter_count"],
        "model_inference_finite_output_count": finite_count,
        "model_inference_mask_violation_count": mask_violation_count,
        "passed": not reason_codes,
        "reason_codes": unique_sorted(reason_codes),
    }


def _source_match_audit(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    reasons = list(source["read_reason_codes"])
    expansion = source["expansion"]
    slices = source["slices"]
    if expansion.get("status") != "passed":
        reasons.append("xunce_high_fidelity_roi_expansion_not_passed")
    if _int_value(expansion.get("slice_count")) < config["required_scenario_count"]:
        reasons.append("xunce_high_fidelity_roi_expansion_slice_count_short")
    if config["require_context_ids"]:
        missing = sum(1 for row in slices[: config["required_scenario_count"]] if not row.get("context_id"))
        if missing:
            reasons.append("xunce_high_fidelity_context_id_missing")
    if config["require_contract_and_sidecar_paths"]:
        missing_paths = sum(
            1
            for row in slices[: config["required_scenario_count"]]
            if not row.get("contract") or not row.get("sidecar")
        )
        if missing_paths:
            reasons.append("xunce_high_fidelity_contract_or_sidecar_missing")
    return {
        "schema_version": "xunce-exploration-coverage-source-match-audit/v1",
        "passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "source_slice_count": _int_value(expansion.get("slice_count")),
        "required_scenario_count": config["required_scenario_count"],
    }


def _boundary_audit(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    boundary = _stage18b_boundary_audit(config, source)
    if config["canary_traffic_fraction"] > 0:
        boundary["passed"] = False
        boundary["reason_codes"] = unique_sorted(list(boundary.get("reason_codes", [])) + ["xunce_coverage_comparison_boundary_violation"])
    return {**boundary, **_boundary_fields()}


def _decision(
    *,
    source: dict[str, Any],
    boundary: dict[str, Any],
    source_match: dict[str, Any],
    model_inference: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    reasons = unique_sorted(
        list(source_match["reason_codes"])
        + list(boundary.get("reason_codes", []))
        + list(model_inference["reason_codes"])
    )
    if "missing_xunce_candidate_checkpoint" in source["read_reason_codes"]:
        reasons.append("missing_xunce_candidate_checkpoint")
    if "missing_incumbent_policy_checkpoint" in source["read_reason_codes"]:
        reasons.append("missing_incumbent_policy_checkpoint")
    reasons = unique_sorted(reasons)
    coverage_advantage = (
        not reasons
        and comparison["xunce_coverage_return_delta_vs_incumbent"] > TOLERANCE
        and comparison["xunce_coverage_curve_auc_delta_vs_incumbent"] > TOLERANCE
        and comparison["xunce_safety_regression_count"] == 0
        and comparison["xunce_efficiency_regression_count"] == 0
        and model_inference["model_inference_mask_violation_count"] == 0
        and comparison["open_grid_fallback_count"] == 0
        and comparison["coverage_gain_per_risk_delta_vs_incumbent"] >= -TOLERANCE
        and comparison["coverage_gain_per_path_cost_delta_vs_incumbent"] >= -TOLERANCE
        and (
            comparison["incumbent_oracle_regret"] <= TOLERANCE
            or comparison["xunce_oracle_regret"] < comparison["incumbent_oracle_regret"] - TOLERANCE
        )
    )
    coverage_advantage_with_efficiency_regression = (
        not reasons
        and comparison["xunce_coverage_return_delta_vs_incumbent"] > TOLERANCE
        and comparison["xunce_efficiency_regression_count"] > 0
    )
    if reasons:
        status = "failed"
        if (
            "missing_xunce_candidate_checkpoint" in reasons
            or "invalid_xunce_candidate_checkpoint" in reasons
            or "xunce_checkpoint_state_dict_missing" in reasons
            or "xunce_checkpoint_model_config_missing" in reasons
        ):
            next_change = FIX_XUNCE_CHECKPOINT_NEXT_REQUIRED_CHANGE
        elif "missing_incumbent_policy_checkpoint" in reasons or "incumbent_checkpoint_format_unsupported" in reasons:
            next_change = FIX_INCUMBENT_CHECKPOINT_NEXT_REQUIRED_CHANGE
        elif "true_model_inference_not_executed" in reasons:
            next_change = FIX_RUNNER_NEXT_REQUIRED_CHANGE
        elif "model_inference_mask_violation" in reasons or any("boundary" in reason for reason in reasons):
            next_change = BOUNDARY_NEXT_REQUIRED_CHANGE
        else:
            next_change = FIX_SOURCE_NEXT_REQUIRED_CHANGE
    else:
        status = "passed"
        next_change = REVIEW_COMPARISON_NEXT_REQUIRED_CHANGE
    diagnostic_reasons: list[str] = []
    if not coverage_advantage:
        diagnostic_reasons.append("xunce_coverage_advantage_not_established")
    if coverage_advantage_with_efficiency_regression:
        diagnostic_reasons.append("xunce_coverage_advantage_with_efficiency_regression")
    if comparison["coverage_gain_per_path_cost_delta_vs_incumbent"] < -TOLERANCE:
        diagnostic_reasons.append("coverage_gain_per_path_cost_regressive")
    if comparison["coverage_gain_per_risk_delta_vs_incumbent"] < -TOLERANCE:
        diagnostic_reasons.append("coverage_gain_per_risk_regressive")
    blocking_reasons = unique_sorted(reasons)
    diagnostic_reasons = unique_sorted(diagnostic_reasons)
    evidence_gate = not any(
        reason
        in {
            "missing_xunce_candidate_checkpoint",
            "invalid_xunce_candidate_checkpoint",
            "xunce_checkpoint_state_dict_missing",
            "xunce_checkpoint_model_config_missing",
            "missing_incumbent_policy_checkpoint",
            "incumbent_checkpoint_format_unsupported",
            "true_model_inference_not_executed",
            "proxy_selection_used",
        }
        for reason in blocking_reasons
    )
    candidate_gate = not any(
        reason in {"model_inference_mask_violation", "no_valid_candidates"}
        or "boundary" in reason
        or "fallback" in reason
        for reason in blocking_reasons
    )
    return {
        "schema_version": "xunce-exploration-coverage-decision-audit/v1",
        "status": status,
        "reason_codes": blocking_reasons,
        "blocking_reason_codes": blocking_reasons,
        "diagnostic_reason_codes": diagnostic_reasons,
        "diagnostic_recommended_change": _coverage_diagnostic_recommended_change(diagnostic_reasons),
        "evidence_authenticity_gate_passed": evidence_gate,
        "candidate_validity_gate_passed": candidate_gate,
        "comparison_allowed": status == "passed" and evidence_gate and candidate_gate,
        "xunce_coverage_advantage_established": coverage_advantage,
        "coverage_advantage_with_efficiency_regression": coverage_advantage_with_efficiency_regression,
        "next_required_change": next_change,
        **_boundary_fields(),
    }


def _coverage_diagnostic_recommended_change(diagnostic_reasons: list[str]) -> str:
    reason_set = set(diagnostic_reasons)
    if "xunce_coverage_advantage_with_efficiency_regression" in reason_set or "coverage_gain_per_path_cost_regressive" in reason_set:
        return EFFICIENCY_NEXT_REQUIRED_CHANGE
    if "xunce_coverage_advantage_not_established" in reason_set:
        return NO_ADVANTAGE_NEXT_REQUIRED_CHANGE
    return ""


def _summary(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    config: dict[str, Any],
    source: dict[str, Any],
    boundary: dict[str, Any],
    source_match: dict[str, Any],
    model_inference: dict[str, Any],
    comparison: dict[str, Any],
    roi_breakdown: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    xunce_latency = comparison["xunce_median_inference_latency_ms"]
    incumbent_latency = comparison["incumbent_median_inference_latency_ms"]
    latency_ratio = xunce_latency / max(incumbent_latency, 1.0e-9) if incumbent_latency else 0.0
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "blocking_reason_codes": decision["blocking_reason_codes"],
        "diagnostic_reason_codes": decision["diagnostic_reason_codes"],
        "diagnostic_recommended_change": decision["diagnostic_recommended_change"],
        "evidence_authenticity_gate_passed": decision["evidence_authenticity_gate_passed"],
        "candidate_validity_gate_passed": decision["candidate_validity_gate_passed"],
        "comparison_allowed": decision["comparison_allowed"],
        "true_model_inference_executed": model_inference["true_model_inference_executed"],
        "proxy_selection_used": model_inference["proxy_selection_used"],
        "xunce_checkpoint_loaded": model_inference["xunce_checkpoint_loaded"],
        "incumbent_checkpoint_loaded": model_inference["incumbent_checkpoint_loaded"],
        "scenario_count": comparison["scenario_count"],
        "rollout_steps": config["rollout_steps"],
        "coverage_radius_cells": config["coverage_radius_cells"],
        "coverage_radius_sensitivity": config["coverage_radius_sensitivity"],
        "planning_backend": config["planning_backend"],
        "candidate_refresh_mode": config["candidate_refresh_mode"],
        "coverage_metric_mode": config["coverage_metric_mode"],
        "include_oracle_baselines": config["include_oracle_baselines"],
        "include_roi_weighted_coverage": config["include_roi_weighted_coverage"],
        "xunce_coverage_advantage_established": decision["xunce_coverage_advantage_established"],
        "xunce_coverage_better_count": comparison["xunce_coverage_better_count"],
        "xunce_coverage_worse_count": comparison["xunce_coverage_worse_count"],
        "xunce_coverage_tie_count": comparison["xunce_coverage_tie_count"],
        "xunce_efficiency_regression_count": comparison["xunce_efficiency_regression_count"],
        "xunce_safety_regression_count": comparison["xunce_safety_regression_count"],
        "model_inference_mask_violation_count": model_inference["model_inference_mask_violation_count"],
        "model_inference_finite_output_count": model_inference["model_inference_finite_output_count"],
        "open_grid_fallback_count": comparison["open_grid_fallback_count"],
        "unreachable_selected_count": comparison["unreachable_selected_count"],
        "path_planning_failure_count": comparison["path_planning_failure_count"],
        "risk_regression_count": comparison["risk_regression_count"],
        "path_cost_regression_count": comparison["path_cost_regression_count"],
        "xunce_final_coverage_rate_delta_vs_incumbent": comparison["xunce_final_coverage_rate_delta_vs_incumbent"],
        "xunce_coverage_return_delta_vs_incumbent": comparison["xunce_coverage_return_delta_vs_incumbent"],
        "xunce_coverage_curve_auc_delta_vs_incumbent": comparison["xunce_coverage_curve_auc_delta_vs_incumbent"],
        "xunce_new_covered_cell_delta_vs_incumbent": comparison["xunce_new_covered_cell_delta_vs_incumbent"],
        "xunce_min_roi_group_coverage_delta_vs_incumbent": comparison["xunce_min_roi_group_coverage_delta_vs_incumbent"],
        "coverage_gain_per_path_cost_delta_vs_incumbent": comparison["coverage_gain_per_path_cost_delta_vs_incumbent"],
        "coverage_gain_per_risk_delta_vs_incumbent": comparison["coverage_gain_per_risk_delta_vs_incumbent"],
        "policy_disagreement_count": comparison["policy_disagreement_count"],
        "useful_disagreement_count": comparison["useful_disagreement_count"],
        "xunce_oracle_regret": comparison["xunce_oracle_regret"],
        "incumbent_oracle_regret": comparison["incumbent_oracle_regret"],
        "oracle_vs_incumbent_coverage_delta": comparison["oracle_vs_incumbent_coverage_delta"],
        "evaluation_task_discriminative": comparison["evaluation_task_discriminative"],
        "selected_probability_median": comparison["selected_probability_median"],
        "action_entropy_median": comparison["action_entropy_median"],
        "selected_rank_median": comparison["selected_rank_median"],
        "xunce_parameter_count": model_inference["xunce_parameter_count"],
        "incumbent_parameter_count": model_inference["incumbent_parameter_count"],
        "xunce_median_inference_latency_ms": xunce_latency,
        "incumbent_median_inference_latency_ms": incumbent_latency,
        "latency_ratio_vs_incumbent": latency_ratio,
        "source_match_audit_passed": source_match["passed"],
        "boundary_audit_passed": boundary["passed"],
        "roi_group_count": roi_breakdown["roi_group_count"],
        "next_required_change": decision["next_required_change"],
        "summary": str(paths["summary"]),
        "episodes": str(paths["episodes"]),
        "steps": str(paths["steps"]),
        "model_inference": str(paths["model_inference"]),
        "roi_breakdown": str(paths["roi_breakdown"]),
        "decision_audit": str(paths["decision_audit"]),
        "config": str(config_path),
        "output_root": str(output_root),
        "source_roi_expansion_root": str(source["root"]),
        "git_provenance": git_snapshot(repo_root),
        **_boundary_fields(),
    }


def _roi_breakdown(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        groups[str(episode["roi_group"])].append(episode)
    rows = []
    for roi_group, rows_for_group in sorted(groups.items()):
        xunce = [row for row in rows_for_group if row["policy"] == "xunce"]
        incumbent = [row for row in rows_for_group if row["policy"] == "incumbent"]
        rows.append(
            {
                "roi_group": roi_group,
                "episode_count": len(rows_for_group),
                "xunce_mean_coverage_return": _mean([row["coverage_return"] for row in xunce]),
                "incumbent_mean_coverage_return": _mean([row["coverage_return"] for row in incumbent]),
                "xunce_mean_final_coverage_rate": _mean([row["final_coverage_rate"] for row in xunce]),
                "incumbent_mean_final_coverage_rate": _mean([row["final_coverage_rate"] for row in incumbent]),
            }
        )
    return {"schema_version": "xunce-exploration-coverage-roi-breakdown/v1", "roi_group_count": len(rows), "families": rows}


def _public_episode(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"steps", "inference_rows", "action_indices"}}


def _disagreement_counts(steps: list[dict[str, Any]]) -> tuple[int, int]:
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in steps:
        by_key[(str(row["scenario_id"]), int(row["step_index"]))][str(row["policy"])] = row
    disagreement = 0
    useful = 0
    for policies in by_key.values():
        xunce = policies.get("xunce")
        incumbent = policies.get("incumbent")
        if not xunce or not incumbent:
            continue
        if xunce["selected_action_index"] != incumbent["selected_action_index"]:
            disagreement += 1
            if (
                float(xunce["coverage_rate_delta"]) > float(incumbent["coverage_rate_delta"]) + TOLERANCE
                and _none_to_zero(xunce.get("path_cost")) <= _none_to_zero(incumbent.get("path_cost")) + TOLERANCE
                and _none_to_zero(xunce.get("risk")) <= _none_to_zero(incumbent.get("risk")) + TOLERANCE
                and not xunce["model_inference_mask_violation"]
            ):
                useful += 1
    return disagreement, useful


def _candidate_rows_for_step(
    scenario: dict[str, Any],
    *,
    current_cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    step_index: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = [dict(candidate) for candidate in _candidate_rows(scenario)]
    if config["candidate_refresh_mode"] != "dynamic_from_coverage_memory":
        return candidates
    refreshed: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        candidate = dict(candidate)
        dynamic_cells = candidate.get("dynamic_cells")
        cell = None
        if isinstance(dynamic_cells, list) and step_index < len(dynamic_cells):
            cell = _cell_tuple(dynamic_cells[step_index])
        if cell is None:
            base_cell = _cell_tuple(_candidate_cell(candidate)) or current_cell
            offset = (step_index + 1) * (index + 1)
            cell = (base_cell[0] + offset, base_cell[1] + step_index + 1)
        candidate["base_cell"] = _candidate_cell(candidate)
        candidate["cell"] = [cell[0], cell[1]]
        candidate["dynamic_candidate_generated"] = True
        if _candidate_is_valid(candidate):
            candidate_cells = _coverage_cells(
                start=current_cell,
                end=cell,
                radius=int(config["coverage_radius_cells"]),
                mode=str(config["coverage_metric_mode"]),
            )
            new_count = len(candidate_cells - covered_cells)
            candidate["expected_new_coverage_area"] = float(new_count)
            candidate["expected_coverage_rate_delta"] = float(new_count) / float(config["coverage_denominator_cells"])
            candidate["coverage_overlap_count"] = len(candidate_cells & covered_cells)
            candidate["coverage_overlap_ratio"] = _safe_ratio(len(candidate_cells & covered_cells), len(candidate_cells)) or 0.0
        refreshed.append(candidate)
    return refreshed


def _oracle_selected_index(
    policy_name: str,
    candidates: list[dict[str, Any]],
    *,
    current_cell: tuple[int, int],
    covered_cells: set[tuple[int, int]],
    config: dict[str, Any],
) -> int | None:
    scored: list[tuple[float, float, float, int]] = []
    for index, candidate in enumerate(candidates):
        if not _candidate_is_valid(candidate):
            continue
        cell = _cell_tuple(_candidate_cell(candidate))
        if cell is None:
            continue
        coverage_cells = _coverage_cells(
            start=current_cell,
            end=cell,
            radius=int(config["coverage_radius_cells"]),
            mode=str(config["coverage_metric_mode"]),
        )
        new_count = len(coverage_cells - covered_cells)
        path_cost = _candidate_cost(candidate) or 0.0
        risk = _finite_or_none(candidate.get("risk")) or 0.0
        if policy_name == "cost_aware_coverage_oracle":
            primary = new_count / max(float(path_cost), TOLERANCE)
            secondary = float(new_count)
        else:
            primary = float(new_count)
            secondary = -float(path_cost)
        scored.append((primary, secondary, -float(risk), index))
    if not scored:
        return None
    return max(scored)[3]


def _coverage_cells(
    *,
    start: tuple[int, int],
    end: tuple[int, int],
    radius: int,
    mode: str,
) -> set[tuple[int, int]]:
    cells = set(_footprint(end, radius=radius))
    if mode == "path_line_plus_endpoint":
        for cell in _line_cells(start, end):
            cells.update(_footprint(cell, radius=radius))
    return cells


def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    cells: list[tuple[int, int]] = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        error2 = 2 * err
        if error2 > -dy:
            err -= dy
            x0 += sx
        if error2 < dx:
            err += dx
            y0 += sy
    return cells


def _footprint(cell: tuple[int, int], *, radius: int) -> set[tuple[int, int]]:
    x, y = cell
    return {(x + dx, y + dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)}


def _coverage_curve_auc(coverage_rates: list[float], rollout_steps: int) -> float:
    if len(coverage_rates) < 2 or rollout_steps <= 0:
        return coverage_rates[-1] if coverage_rates else 0.0
    area = 0.0
    for left, right in zip(coverage_rates, coverage_rates[1:]):
        area += (left + right) / 2.0
    return area / float(rollout_steps)


def _empty_model_detail() -> dict[str, Any]:
    return {
        "logits": [],
        "masked_logits": [],
        "action_probs": [],
        "value": 0.0,
        "selected_action_index": None,
        "selected_probability": 0.0,
        "selected_rank": 0,
        "finite_outputs": False,
        "latency_ms": 0.0,
    }


def _cell_tuple(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _entropy(values: Any) -> float:
    if not isinstance(values, (list, tuple)):
        return 0.0
    entropy = 0.0
    for value in values:
        probability = _float_default(value)
        if probability > 0.0:
            entropy -= probability * math.log(probability)
    return entropy


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    numeric = _finite_or_none(numerator)
    denom = _finite_or_none(denominator)
    if numeric is None or denom is None or abs(float(denom)) <= TOLERANCE:
        return None
    return float(numeric) / float(denom)


def _mean(values: list[Any]) -> float:
    numeric = [float(value) for value in values if _finite_or_none(value) is not None]
    return statistics.mean(numeric) if numeric else 0.0


def _median(values: list[Any]) -> float:
    numeric = [float(value) for value in values if _finite_or_none(value) is not None]
    return statistics.median(numeric) if numeric else 0.0


def _none_to_zero(value: Any) -> float:
    numeric = _finite_or_none(value)
    return 0.0 if numeric is None else float(numeric)


def _boundary_fields() -> dict[str, Any]:
    fields = {field: False for field in BOUNDARY_FIELDS}
    fields.update(
        {
            "canary_traffic_fraction": 0.0,
            "runs_new_training_update": False,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
            "real_world_release_approved": False,
            "real_world_performance_claimed": False,
            "default_policy_replacement_approved": False,
            "real_executor_connection_approved": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
        }
    )
    return fields


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Exploration Coverage Comparison v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- true_model_inference_executed: `{summary['true_model_inference_executed']}`",
            f"- proxy_selection_used: `{summary['proxy_selection_used']}`",
            f"- scenario_count: `{summary['scenario_count']}`",
            f"- rollout_steps: `{summary['rollout_steps']}`",
            f"- xunce_coverage_advantage_established: `{summary['xunce_coverage_advantage_established']}`",
            f"- xunce_coverage_return_delta_vs_incumbent: `{summary['xunce_coverage_return_delta_vs_incumbent']}`",
            f"- xunce_coverage_curve_auc_delta_vs_incumbent: `{summary['xunce_coverage_curve_auc_delta_vs_incumbent']}`",
            f"- xunce_efficiency_regression_count: `{summary['xunce_efficiency_regression_count']}`",
            f"- xunce_safety_regression_count: `{summary['xunce_safety_regression_count']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            "",
            "This is an offline shadow rollout comparison. It does not approve default-policy replacement, real executor connection, checkpoint publication, PPO training, online canary, or real-world performance claims.",
            "",
        ]
    )


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0.0:
        raise ConfigError(f"{name} must be a positive number")
    return float(value)


def _nonnegative_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0.0:
        raise ConfigError(f"{name} must be a non-negative number")
    return float(value)


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean")
    return bool(value)


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
