from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as stage26_8
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as stage26_8


STAGE_ID = "xunce-stage26-8c-resume-h16-h20-horizon-efficiency"
CONFIG_SCHEMA_VERSION = "xunce-stage26-8c-resume-h16-h20-horizon-efficiency-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-8c-summary/v1"
HORIZON_RESULT_SCHEMA_VERSION = "xunce-stage26-8c-horizon-result/v1"
AGGREGATE_SCHEMA_VERSION = "xunce-stage26-8c-horizon-efficiency-aggregate/v1"
TERMINAL_AUDIT_SCHEMA_VERSION = "xunce-stage26-8c-terminal-reachability-carryover-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-8c-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-8c-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_8c_resume_h16_h20_horizon_efficiency_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8c_resume_h16_h20_horizon_efficiency_v1"
)
DEFAULT_STAGE26_8B_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8b_repair_horizon_collector_terminal_reachability_v1"
)

SUMMARY_FILE = "xunce-stage26-8c-summary.json"
HORIZON_RESULTS_FILE = "xunce-stage26-8c-horizon-results.jsonl"
AGGREGATE_FILE = "xunce-stage26-8c-horizon-efficiency-aggregate.json"
TERMINAL_AUDIT_FILE = "xunce-stage26-8c-terminal-reachability-carryover-audit.json"
RECOMMENDED_CONFIG_FILE = "xunce-stage26-8c-recommended-next-config.json"
ROUTING_FILE = "xunce-stage26-8c-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-8c-report.md"
MANIFEST_FILE = "xunce-stage26-8c-manifest.json"

COVERAGE_DENOMINATOR_SOURCE = "main_coverable_cells/v1"
SUCCESS_METRIC = "main_coverable_coverage_efficiency/v1"
ROUTE_INPUTS = "rerun_stage26_8c_required_inputs"
ROUTE_BINDING_OR_SAFETY = "repair_stage26_8c_horizon_eval_binding_or_safety"
ROUTE_UPDATE = "repair_stage26_8c_update_stability"
ROUTE_STAGE26_9 = "run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot"
ROUTE_EXPAND_SEEDS = "expand_stage26_8c_seed_budget_at_best_horizon"
ROUTE_POLICY_SIGNAL = "repair_stage26_synthetic_policy_update_signal_strength"
ROUTE_CREDIT = "repair_stage26_synthetic_credit_assignment"
ROUTE_BOUNDARY = "resolve_stage26_8c_boundary_rejections"

BOUNDARY_FIELDS = (
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resume Stage26.8 H16/H20 coverage-efficiency horizons after Stage26.8B.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_8c_resume_h16_h20_horizon_efficiency(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_8c_resume_h16_h20_horizon_efficiency(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(config_path, repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage26_8b_summary = _read_json_if_exists(Path(config["stage26_8b_root"]) / "xunce-stage26-8b-summary.json")
    stage26_8a_root = Path(config["stage26_8a_root"])
    h12_summary = _read_json_if_exists(stage26_8a_root / "h12" / "s26_8" / stage26_8.SUMMARY_FILE)
    boundary_rejections = _boundary_rejections(config)
    input_rejections = _input_rejections(stage26_8b_summary)

    horizon_rows: list[dict[str, Any]] = []
    if not boundary_rejections and not input_rejections and bool(config["run_horizon_chain"]):
        for horizon in config["horizons"]:
            horizon_rows.append(
                _run_horizon(
                    config=config,
                    horizon_steps=int(horizon),
                    output_root=output_root,
                    repo_root=repo_root,
                )
            )

    aggregate = _horizon_aggregate(horizon_rows)
    terminal_audit = _terminal_reachability_audit(
        stage26_8b_summary=stage26_8b_summary,
        h12_summary=h12_summary,
        horizon_rows=horizon_rows,
    )
    route = _route(
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        horizon_rows=horizon_rows,
        aggregate=aggregate,
    )
    status = "passed" if route == ROUTE_STAGE26_9 else "failed"
    recommended = _recommended_next_config(config, horizon_rows, route)
    summary = {
        **aggregate,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "stage26_8b_root": config["stage26_8b_root"],
        "stage26_8b_status": stage26_8b_summary.get("status"),
        "stage26_8b_next_required_change": stage26_8b_summary.get("next_required_change"),
        "stage26_8a_root": config["stage26_8a_root"],
        "h12_baseline_available": bool(h12_summary),
        "h12_baseline_status": h12_summary.get("status"),
        "h12_baseline_mean_main_coverage_per_100m_delta": _float(h12_summary.get("mean_main_coverage_per_100m_delta")),
        "horizons": config["horizons"],
        "seeds": config["seeds"],
        "coverage_denominator_source": COVERAGE_DENOMINATOR_SOURCE,
        "post_update_success_metric": SUCCESS_METRIC,
        "coverage_source": stage26_8.COVERAGE_SOURCE,
        "path_cost_source": stage26_8.PATH_COST_SOURCE,
        "synthetic_source_kind": stage26_8.SYNTHETIC_SOURCE_KIND,
        "action_space_type": stage26_8.ACTION_SPACE_TYPE,
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
            "horizon_results": str(output_root / HORIZON_RESULTS_FILE),
            "horizon_efficiency_aggregate": str(output_root / AGGREGATE_FILE),
            "terminal_reachability_carryover_audit": str(output_root / TERMINAL_AUDIT_FILE),
            "recommended_next_config": str(output_root / RECOMMENDED_CONFIG_FILE),
            "routing": str(output_root / ROUTING_FILE),
            "report": str(output_root / REPORT_FILE),
        },
    }
    _write_jsonl(output_root / HORIZON_RESULTS_FILE, horizon_rows)
    _write_json(output_root / AGGREGATE_FILE, aggregate)
    _write_json(output_root / TERMINAL_AUDIT_FILE, terminal_audit)
    _write_json(output_root / RECOMMENDED_CONFIG_FILE, recommended)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary, horizon_rows), encoding="utf-8")
    return summary


def _run_horizon(*, config: dict[str, Any], horizon_steps: int, output_root: Path, repo_root: Path) -> dict[str, Any]:
    label = f"h{horizon_steps}"
    horizon_root = output_root / label
    stage26_8_config_path = horizon_root / "xunce-stage26-8c-stage26-8-config.json"
    stage26_8_output_root = horizon_root / "s26_8"
    horizon_root.mkdir(parents=True, exist_ok=True)
    stage26_8_config = _build_stage26_8_config(config, horizon_steps=horizon_steps, repo_root=repo_root)
    _write_json(stage26_8_config_path, stage26_8_config)
    summary = stage26_8.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot(
        config_path=stage26_8_config_path,
        output_root=stage26_8_output_root,
        repo_root=repo_root,
    )
    return _horizon_row(
        horizon_steps=horizon_steps,
        horizon_root=horizon_root,
        stage26_8_output_root=stage26_8_output_root,
        stage26_8_config_path=stage26_8_config_path,
        summary=summary,
    )


def _build_stage26_8_config(config: dict[str, Any], *, horizon_steps: int, repo_root: Path) -> dict[str, Any]:
    cfg = _load_json_template(config["stage26_8_base_config"], repo_root)
    cfg.update(
        {
            "stage26_8c_horizon_steps": int(horizon_steps),
            "stage26_8b_root": config["stage26_8b_root"],
            "stage26_8a_root": config["stage26_8a_root"],
            "stage26_7h_root": config["stage26_7h_root"],
            "stage26_0_root": config["stage26_0_root"],
            "seeds": list(config["seeds"]),
            "run_stage26_chain": True,
            "required_scenario_count": int(config["required_scenario_count"]),
            "collector_rollout_steps": int(horizon_steps),
            "eval_rollout_steps": int(horizon_steps),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
            "stop_on_first_seed_blocker": False,
            "reuse_existing_seed_outputs": True,
            "stage21_5_timeout_seconds": 0.0,
            "post_update_success_metric": SUCCESS_METRIC,
            "coverage_denominator_source": COVERAGE_DENOMINATOR_SOURCE,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _horizon_row(
    *,
    horizon_steps: int,
    horizon_root: Path,
    stage26_8_output_root: Path,
    stage26_8_config_path: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    seed_count = int(summary.get("seed_count") or 0)
    majority = math.floor(seed_count / 2) + 1 if seed_count else 1
    positive = int(summary.get("positive_efficiency_seed_count") or 0)
    negative = int(summary.get("negative_efficiency_seed_count") or 0)
    binding = int(summary.get("binding_or_safety_failure_seed_count") or 0)
    execution = int(summary.get("execution_failure_seed_count") or 0)
    update = int(summary.get("update_failure_seed_count") or 0)
    seed_rows = _read_jsonl_if_exists(stage26_8_output_root / stage26_8.SEED_RESULTS_FILE)
    collector_failure_count = len([row for row in seed_rows if row.get("stage26_1_status") != "passed"])
    if collector_failure_count:
        binding = max(binding, collector_failure_count)
        update = max(0, update - collector_failure_count)
    mean_per100 = _float(summary.get("mean_main_coverage_per_100m_delta"))
    return {
        "schema_version": HORIZON_RESULT_SCHEMA_VERSION,
        "horizon_label": f"h{horizon_steps}",
        "horizon_steps": int(horizon_steps),
        "horizon_root": str(horizon_root),
        "stage26_8_root": str(stage26_8_output_root),
        "stage26_8_config": str(stage26_8_config_path),
        "stage26_8_status": summary.get("status"),
        "stage26_8_next_required_change": summary.get("next_required_change"),
        "seed_count": seed_count,
        "completed_seed_count": int(summary.get("completed_seed_count") or 0),
        "clean_seed_count": int(summary.get("clean_seed_count") or 0),
        "positive_efficiency_seed_count": positive,
        "nonnegative_efficiency_seed_count": int(summary.get("nonnegative_efficiency_seed_count") or 0),
        "negative_efficiency_seed_count": negative,
        "binding_or_safety_failure_seed_count": binding,
        "execution_failure_seed_count": execution,
        "update_failure_seed_count": update,
        "collector_failure_seed_count": collector_failure_count,
        "mean_main_coverage_per_100m_delta": mean_per100,
        "mean_main_final_coverage_delta": _float(summary.get("mean_main_final_coverage_delta")),
        "mean_main_coverage_auc_delta_diagnostic": _float(summary.get("mean_main_coverage_auc_delta_diagnostic")),
        "mean_hybrid_astar_path_cost_delta_diagnostic": _float(summary.get("mean_hybrid_astar_path_cost_delta_diagnostic")),
        "majority_positive": positive >= majority,
        "single_positive_without_majority": 0 < positive < majority,
        "majority_negative": negative >= majority,
        "clean_zero_delta": binding == 0 and execution == 0 and update == 0 and positive == 0 and negative == 0 and abs(mean_per100) <= 1.0e-12,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _horizon_aggregate(horizon_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "horizon_count": len(horizon_rows),
        "completed_horizon_count": len([row for row in horizon_rows if row.get("stage26_8_status")]),
        "clean_horizon_count": len(
            [
                row
                for row in horizon_rows
                if int(row.get("binding_or_safety_failure_seed_count") or 0) == 0
                and int(row.get("execution_failure_seed_count") or 0) == 0
                and int(row.get("update_failure_seed_count") or 0) == 0
            ]
        ),
        "majority_positive_horizon_count": len([row for row in horizon_rows if row.get("majority_positive")]),
        "single_positive_horizon_count": len([row for row in horizon_rows if row.get("single_positive_without_majority")]),
        "majority_negative_horizon_count": len([row for row in horizon_rows if row.get("majority_negative")]),
        "zero_delta_horizon_count": len([row for row in horizon_rows if row.get("clean_zero_delta")]),
        "max_mean_main_coverage_per_100m_delta": max([_float(row.get("mean_main_coverage_per_100m_delta")) for row in horizon_rows], default=0.0),
        "best_horizon_steps": _best_horizon(horizon_rows).get("horizon_steps"),
        "best_horizon_label": _best_horizon(horizon_rows).get("horizon_label"),
    }


def _terminal_reachability_audit(
    *,
    stage26_8b_summary: dict[str, Any],
    h12_summary: dict[str, Any],
    horizon_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": TERMINAL_AUDIT_SCHEMA_VERSION,
        "stage26_8b_status": stage26_8b_summary.get("status"),
        "stage26_8b_next_required_change": stage26_8b_summary.get("next_required_change"),
        "stage26_8b_h16_repaired_status": stage26_8b_summary.get("h16_repaired_status"),
        "stage26_8b_h16_repaired_transition_count": int(stage26_8b_summary.get("h16_repaired_transition_count") or 0),
        "stage26_8b_h16_repaired_terminal_count": int(stage26_8b_summary.get("h16_repaired_stage21_1_terminal_count") or 0),
        "h12_baseline_available": bool(h12_summary),
        "h12_baseline_status": h12_summary.get("status"),
        "h12_baseline_next_required_change": h12_summary.get("next_required_change"),
        "h12_baseline_mean_main_coverage_per_100m_delta": _float(h12_summary.get("mean_main_coverage_per_100m_delta")),
        "h16_h20_horizon_count": len(horizon_rows),
        "h16_h20_reused_old_failed_h16_output": False,
        "total_collector_failure_seed_count": sum(int(row.get("collector_failure_seed_count") or 0) for row in horizon_rows),
        "auc_and_total_path_cost_are_diagnostic_only": True,
    }


def _route(
    *,
    boundary_rejections: list[str],
    input_rejections: list[str],
    horizon_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections or not horizon_rows:
        return ROUTE_INPUTS
    if any(int(row.get("binding_or_safety_failure_seed_count") or 0) > 0 for row in horizon_rows):
        return ROUTE_BINDING_OR_SAFETY
    if any(int(row.get("execution_failure_seed_count") or 0) > 0 for row in horizon_rows):
        return ROUTE_BINDING_OR_SAFETY
    if any(int(row.get("update_failure_seed_count") or 0) > 0 for row in horizon_rows):
        return ROUTE_UPDATE
    if any(row.get("majority_positive") for row in horizon_rows):
        return ROUTE_STAGE26_9
    if any(row.get("majority_negative") for row in horizon_rows):
        return ROUTE_CREDIT
    if any(row.get("single_positive_without_majority") for row in horizon_rows):
        return ROUTE_EXPAND_SEEDS
    if int(aggregate.get("zero_delta_horizon_count") or 0) == len(horizon_rows):
        return ROUTE_POLICY_SIGNAL
    return ROUTE_EXPAND_SEEDS


def _input_rejections(stage26_8b_summary: dict[str, Any]) -> list[str]:
    if not stage26_8b_summary:
        return ["missing_stage26_8b_summary"]
    reasons: list[str] = []
    if stage26_8b_summary.get("schema_version") != "xunce-stage26-8b-summary/v1":
        reasons.append("stage26_8b_schema_version_mismatch")
    if stage26_8b_summary.get("stage_id") != "xunce-stage26-8b-repair-horizon-collector-terminal-reachability":
        reasons.append("stage26_8b_stage_id_mismatch")
    if stage26_8b_summary.get("status") != "passed":
        reasons.append("stage26_8b_status_not_passed")
    if stage26_8b_summary.get("next_required_change") != "resume_stage26_8a_from_h16_h20":
        reasons.append("stage26_8b_route_not_resume_h16_h20")
    if stage26_8b_summary.get("h16_repaired_status") != "passed":
        reasons.append("stage26_8b_h16_repaired_status_not_passed")
    if int(stage26_8b_summary.get("h16_repaired_transition_count") or 0) <= 0:
        reasons.append("stage26_8b_h16_repaired_transition_count_missing")
    if stage26_8b_summary.get("coverage_source") != stage26_8.COVERAGE_SOURCE:
        reasons.append("stage26_8b_coverage_source_mismatch")
    if stage26_8b_summary.get("path_cost_source") != stage26_8.PATH_COST_SOURCE:
        reasons.append("stage26_8b_path_cost_source_mismatch")
    if stage26_8b_summary.get("synthetic_source_kind") != stage26_8.SYNTHETIC_SOURCE_KIND:
        reasons.append("stage26_8b_synthetic_source_kind_mismatch")
    if abs(_float(stage26_8b_summary.get("max_traversable_slope_deg")) - 30.0) > 1.0e-9:
        reasons.append("stage26_8b_max_traversable_slope_deg_mismatch")
    for field in BOUNDARY_FIELDS:
        if stage26_8b_summary.get(field) is True:
            reasons.append(f"stage26_8b_{field}")
    if _float(stage26_8b_summary.get("canary_traffic_fraction")) > 0.0:
        reasons.append("stage26_8b_canary_traffic_fraction")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    if _float(config.get("canary_traffic_fraction")) > 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _recommended_next_config(config: dict[str, Any], horizon_rows: list[dict[str, Any]], route: str) -> dict[str, Any]:
    best = _best_horizon(horizon_rows)
    recommended_seed_count = len(config["seeds"])
    recommended_new_seeds: list[int] = []
    if route == ROUTE_EXPAND_SEEDS:
        recommended_seed_count = max(len(config["seeds"]) + 2, 5)
        seed_max = max(config["seeds"])
        recommended_new_seeds = [seed_max + 1, seed_max + 2]
    return {
        "coverage_denominator_source": COVERAGE_DENOMINATOR_SOURCE,
        "post_update_success_metric": SUCCESS_METRIC,
        "recommended_route": route,
        "recommended_horizon_steps": best.get("horizon_steps"),
        "recommended_seed_count": recommended_seed_count,
        "recommended_existing_seeds": config["seeds"],
        "recommended_new_seeds": recommended_new_seeds,
        "auc_and_total_path_cost_are_diagnostic_only": True,
    }


def _best_horizon(horizon_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not horizon_rows:
        return {}
    return sorted(
        horizon_rows,
        key=lambda row: (
            int(row.get("binding_or_safety_failure_seed_count") or 0) == 0
            and int(row.get("execution_failure_seed_count") or 0) == 0
            and int(row.get("update_failure_seed_count") or 0) == 0,
            int(row.get("positive_efficiency_seed_count") or 0),
            _float(row.get("mean_main_coverage_per_100m_delta")),
            int(row.get("horizon_steps") or 0),
        ),
        reverse=True,
    )[0]


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    stage26_8_base = _load_json_template(payload.get("stage26_8_base_config", stage26_8.DEFAULT_CONFIG), repo_root)
    defaults = {
        "stage26_8b_root": DEFAULT_STAGE26_8B_ROOT,
        "stage26_8_base_config": stage26_8.DEFAULT_CONFIG,
        "stage26_8a_root": None,
        "stage26_7h_root": stage26_8_base.get("stage26_7h_root", stage26_8.DEFAULT_STAGE26_7H_ROOT),
        "stage26_0_root": stage26_8_base.get("stage26_0_root", stage26_8.DEFAULT_STAGE26_0_ROOT),
        "horizons": [16, 20],
        "seeds": stage26_8_base.get("seeds", [260801, 260802, 260803]),
        "run_horizon_chain": True,
        "required_scenario_count": stage26_8_base.get("required_scenario_count", 3),
        "dynamic_max_candidates_per_step": stage26_8_base.get("dynamic_max_candidates_per_step", 36),
        "dynamic_proposal_pool_limit_per_step": stage26_8_base.get("dynamic_proposal_pool_limit_per_step", 288),
        "hybrid_astar_candidate_eval_workers": stage26_8_base.get("hybrid_astar_candidate_eval_workers", 4),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    for field in ("stage26_8b_root", "stage26_8_base_config", "stage26_7h_root", "stage26_0_root"):
        config[field] = str(_resolve_path(Path(str(config[field])), repo_root))
    if config.get("stage26_8a_root") is None:
        stage26_8b_summary = _read_json_if_exists(Path(config["stage26_8b_root"]) / "xunce-stage26-8b-summary.json")
        config["stage26_8a_root"] = stage26_8b_summary.get("stage26_8a_root", "")
    config["stage26_8a_root"] = str(_resolve_path(Path(str(config["stage26_8a_root"])), repo_root))
    config["horizons"] = [int(value) for value in config["horizons"]]
    config["seeds"] = [int(value) for value in config["seeds"]]
    if config["horizons"] != [16, 20]:
        raise ValueError("horizons must be [16, 20] for Stage26.8C v1")
    if not config["seeds"]:
        raise ValueError("seeds must not be empty")
    return config


def _load_json_template(path: str | Path, repo_root: Path) -> dict[str, Any]:
    resolved = _resolve_path(Path(path), repo_root)
    if not resolved.is_file():
        return {}
    return _read_json(resolved)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return _read_json(path)


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


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


def _render_report(summary: dict[str, Any], horizon_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage26.8C Resume H16/H20 Horizon Efficiency",
        "",
        f"- status: `{summary['status']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- h12_baseline_mean_main_coverage_per_100m_delta: `{summary.get('h12_baseline_mean_main_coverage_per_100m_delta')}`",
        f"- horizons: `{summary['horizons']}`",
        f"- best_horizon_steps: `{summary.get('best_horizon_steps')}`",
        "",
        "Horizon results:",
    ]
    for row in horizon_rows:
        lines.append(
            f"- {row['horizon_label']}: status={row.get('stage26_8_status')}, "
            f"positive={row.get('positive_efficiency_seed_count')}, "
            f"negative={row.get('negative_efficiency_seed_count')}, "
            f"mean_per100m={row.get('mean_main_coverage_per_100m_delta')}"
        )
    lines.extend(
        [
            "",
            "H12 is a baseline audit only. AUC, total path length, and Hybrid A* path-cost deltas remain diagnostic only.",
            "This stage does not publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
