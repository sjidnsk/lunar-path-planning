from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import xunce_artifact_io as artifact_io

try:  # pragma: no cover
    import run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as stage26_8
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as stage26_8


STAGE_ID = "xunce-stage26-8d-resumable-seed-horizon-execution"
CONFIG_SCHEMA_VERSION = "xunce-stage26-8d-resumable-seed-horizon-execution-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-8d-summary/v1"
JOB_PLAN_SCHEMA_VERSION = "xunce-stage26-8d-job-plan/v1"
JOB_STATE_SCHEMA_VERSION = "xunce-stage26-8d-job-state/v1"
CARRYOVER_AUDIT_SCHEMA_VERSION = "xunce-stage26-8d-carryover-audit/v1"
AGGREGATE_SCHEMA_VERSION = "xunce-stage26-8d-horizon-efficiency-aggregate/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-8d-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-8d-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_8d_resumable_seed_horizon_execution_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8d_resumable_seed_horizon_execution_v1"
)
DEFAULT_STAGE26_8B_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8b_repair_horizon_collector_terminal_reachability_v1"
)
DEFAULT_STAGE26_8C_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8c_resume_h16_h20_horizon_efficiency_v1"
)

SUMMARY_FILE = "xunce-stage26-8d-summary.json"
JOB_PLAN_FILE = "xunce-stage26-8d-job-plan.json"
JOB_STATE_FILE = "xunce-stage26-8d-job-state.jsonl"
CARRYOVER_AUDIT_FILE = "xunce-stage26-8d-carryover-audit.json"
AGGREGATE_FILE = "xunce-stage26-8d-horizon-efficiency-aggregate.json"
ROUTING_FILE = "xunce-stage26-8d-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-8d-report.md"
MANIFEST_FILE = "xunce-stage26-8d-manifest.json"

PHASES = ("stage26_1", "stage26_2", "stage26_3")
HORIZONS = (16, 20)
ROUTE_INPUTS = "rerun_stage26_8d_required_inputs"
ROUTE_RESUME_STATE = "repair_stage26_8d_resume_state_contract"
ROUTE_BINDING_OR_SAFETY = "repair_stage26_8d_job_binding_or_safety"
ROUTE_UPDATE = "repair_stage26_8d_job_update_stability"
ROUTE_CONTINUE = "continue_stage26_8d_seed_horizon_jobs"
ROUTE_STAGE26_9 = "run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot"
ROUTE_EXPAND_SEEDS = "expand_stage26_8d_seed_budget_at_best_horizon"
ROUTE_POLICY_SIGNAL = "repair_stage26_synthetic_policy_update_signal_strength"
ROUTE_CREDIT = "repair_stage26_synthetic_credit_assignment"
ROUTE_BOUNDARY = "resolve_stage26_8d_boundary_rejections"
BOUNDARY_FIELDS = (
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one resumable Stage26.8D seed/horizon phase.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-mode", choices=("run_next", "aggregate_only", "run_job"), help="Override config run mode.")
    parser.add_argument("--horizon", type=int, help="Run only this horizon with --run-mode run_job.")
    parser.add_argument("--seed", type=int, help="Run only this seed with --run-mode run_job.")
    parser.add_argument("--phase", choices=PHASES, help="Run only this phase with --run-mode run_job.")
    parser.add_argument("--max-jobs-per-invocation", type=int, help="Override max phase executions.")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_8d_resumable_seed_horizon_execution(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
        run_mode_override=args.run_mode,
        horizon_override=args.horizon,
        seed_override=args.seed,
        phase_override=args.phase,
        max_jobs_override=args.max_jobs_per_invocation,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "next_required_change": summary["next_required_change"],
                "next_job_id": summary.get("next_job_id"),
                "next_phase": summary.get("next_phase"),
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_8d_resumable_seed_horizon_execution(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    run_mode_override: str | None = None,
    horizon_override: int | None = None,
    seed_override: int | None = None,
    phase_override: str | None = None,
    max_jobs_override: int | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(config_path, repo_root)
    if run_mode_override is not None:
        config["run_mode"] = run_mode_override
    if max_jobs_override is not None:
        config["max_jobs_per_invocation"] = int(max_jobs_override)
    output_root = _resolve_path(output_root, repo_root)
    artifact_io.make_dirs(output_root)

    stage26_8b_summary = _read_json_if_exists(Path(config["stage26_8b_root"]) / "xunce-stage26-8b-summary.json")
    boundary_rejections = _boundary_rejections(config)
    input_rejections = _input_rejections(stage26_8b_summary)
    phase_executions: list[dict[str, Any]] = []
    now = _utc_now()

    job_rows = _scan_jobs(config=config, output_root=output_root, repo_root=repo_root)
    pending_before = _next_runnable_phases(job_rows)
    if not boundary_rejections and not input_rejections and config["run_mode"] != "aggregate_only":
        selected = _select_phase_executions(
            pending_before,
            run_mode=config["run_mode"],
            horizon=horizon_override,
            seed=seed_override,
            phase=phase_override,
            limit=int(config["max_jobs_per_invocation"]),
        )
        for item in selected:
            phase_executions.append(_run_phase(item, config=config, output_root=output_root, repo_root=repo_root, started_at=now))

    job_rows = _scan_jobs(config=config, output_root=output_root, repo_root=repo_root, phase_executions=phase_executions)
    aggregate = _aggregate(job_rows)
    carryover_audit = _carryover_audit(job_rows, config)
    route = _route(
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        job_rows=job_rows,
        aggregate=aggregate,
        resume_state_rejections=_resume_state_rejections(job_rows),
    )
    status = "passed" if route in {ROUTE_CONTINUE, ROUTE_STAGE26_9} else "failed"
    next_item = _next_runnable_phases(job_rows)
    next_item = next_item[0] if next_item else {}
    summary = {
        **aggregate,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "run_mode": config["run_mode"],
        "max_jobs_per_invocation": int(config["max_jobs_per_invocation"]),
        "phase_execution_count": len(phase_executions),
        "phase_executions": phase_executions,
        "stage26_8b_root": config["stage26_8b_root"],
        "stage26_8c_root": config["stage26_8c_root"],
        "stage26_8b_status": stage26_8b_summary.get("status"),
        "stage26_8b_next_required_change": stage26_8b_summary.get("next_required_change"),
        "next_job_id": next_item.get("job_id"),
        "next_phase": next_item.get("phase"),
        "coverage_denominator_source": stage26_8.COVERAGE_DENOMINATOR_SOURCE,
        "post_update_success_metric": stage26_8.SUCCESS_METRIC,
        "coverage_source": stage26_8.COVERAGE_SOURCE,
        "path_cost_source": stage26_8.PATH_COST_SOURCE,
        "synthetic_source_kind": stage26_8.SYNTHETIC_SOURCE_KIND,
        "action_space_type": stage26_8.ACTION_SPACE_TYPE,
        "max_traversable_slope_deg": 30.0,
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
        "resume_state_rejections": _resume_state_rejections(job_rows),
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
        "resume_state_rejections": summary["resume_state_rejections"],
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
            "job_plan": str(output_root / JOB_PLAN_FILE),
            "job_state": str(output_root / JOB_STATE_FILE),
            "carryover_audit": str(output_root / CARRYOVER_AUDIT_FILE),
            "horizon_efficiency_aggregate": str(output_root / AGGREGATE_FILE),
            "routing": str(output_root / ROUTING_FILE),
            "report": str(output_root / REPORT_FILE),
        },
    }

    _write_json(output_root / JOB_PLAN_FILE, {"schema_version": JOB_PLAN_SCHEMA_VERSION, "jobs": _job_plan_rows(job_rows)})
    _write_jsonl(output_root / JOB_STATE_FILE, job_rows)
    _write_json(output_root / CARRYOVER_AUDIT_FILE, carryover_audit)
    _write_json(output_root / AGGREGATE_FILE, aggregate)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    artifact_io.write_text(output_root / REPORT_FILE, _render_report(summary, job_rows))
    return summary


def _run_phase(item: dict[str, Any], *, config: dict[str, Any], output_root: Path, repo_root: Path, started_at: str) -> dict[str, Any]:
    job_root = _job_output_root(output_root, int(item["horizon_steps"]), int(item["seed"]))
    artifact_io.make_dirs(job_root)
    phase = item["phase"]
    seed = int(item["seed"])
    horizon = int(item["horizon_steps"])
    stage26_8_config = _stage26_8_config(config, horizon_steps=horizon, repo_root=repo_root)
    if phase == "stage26_1":
        config_path = job_root / "xunce-stage26-8d-stage26-1-config.json"
        phase_root = job_root / "s26_1"
        _write_json(config_path, stage26_8._build_stage26_1_config(stage26_8_config, seed, job_root, repo_root))
        summary = stage26_8.stage26_1.run_xunce_stage26_1_synthetic_terrain_collector_smoke(
            config_path=config_path,
            output_root=phase_root,
            repo_root=repo_root,
        )
    elif phase == "stage26_2":
        stage26_1_root = Path(item["stage26_1_root"])
        config_path = job_root / "xunce-stage26-8d-stage26-2-config.json"
        phase_root = job_root / "s26_2"
        _write_json(config_path, stage26_8._build_stage26_2_config(stage26_8_config, seed=seed, stage26_1_root=stage26_1_root, repo_root=repo_root))
        summary = stage26_8.stage26_2.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
            config_path=config_path,
            output_root=phase_root,
            repo_root=repo_root,
        )
    elif phase == "stage26_3":
        stage26_2_root = Path(item["stage26_2_root"])
        config_path = job_root / "xunce-stage26-8d-stage26-3-config.json"
        phase_root = job_root / "s26_3"
        _write_json(config_path, stage26_8._build_stage26_3_config(stage26_8_config, seed=seed, stage26_2_root=stage26_2_root, repo_root=repo_root))
        summary = stage26_8.stage26_3.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke(
            config_path=config_path,
            output_root=phase_root,
            repo_root=repo_root,
        )
    else:  # pragma: no cover
        raise ValueError(f"unknown phase: {phase}")
    return {
        "job_id": item["job_id"],
        "phase": phase,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "summary_path": str(Path(phase_root) / _summary_file_for_phase(phase)),
        "status": summary.get("status"),
        "next_required_change": summary.get("next_required_change"),
    }


def _scan_jobs(
    *,
    config: dict[str, Any],
    output_root: Path,
    repo_root: Path,
    phase_executions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    execution_by_key = {(row["job_id"], row["phase"]): row for row in phase_executions or []}
    for horizon in config["horizons"]:
        for seed_index, seed in enumerate(config["seeds"]):
            rows.append(_job_state(config, output_root, repo_root, int(horizon), int(seed), int(seed_index), execution_by_key))
    return rows


def _job_state(
    config: dict[str, Any],
    output_root: Path,
    repo_root: Path,
    horizon: int,
    seed: int,
    seed_index: int,
    execution_by_key: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    job_id = f"h{horizon}_s{seed}"
    current_root = _job_output_root(output_root, horizon, seed)
    carryover_root = Path(config["stage26_8c_root"]) / f"h{horizon}" / "s26_8" / f"s{seed_index}"
    phases: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        current = _phase_record(phase, current_root / _phase_dir(phase), "stage26_8d_output")
        carryover = _phase_record(phase, carryover_root / _phase_dir(phase), "stage26_8c_carryover")
        record = current if current["summary_exists"] else carryover
        record["execution"] = execution_by_key.get((job_id, phase), {})
        record["complete"] = _phase_complete(phase, record["summary"])
        record["failed"] = _phase_failed(phase, record["summary"])
        phases[phase] = record
    next_phase = _next_phase(phases)
    final_summary = phases["stage26_3"]["summary"] if phases["stage26_3"]["complete"] else {}
    return {
        "schema_version": JOB_STATE_SCHEMA_VERSION,
        "job_id": job_id,
        "horizon_steps": horizon,
        "seed": seed,
        "seed_index": seed_index,
        "phase": next_phase or "complete",
        "source_root": phases[next_phase]["root"] if next_phase else phases["stage26_3"]["root"],
        "output_root": str(current_root),
        "status": "complete" if not next_phase and _job_complete(phases) else ("failed" if _job_failed(phases) else "pending"),
        "resume_decision": _resume_decision(phases),
        "started_at": _latest_execution(phases).get("started_at"),
        "finished_at": _latest_execution(phases).get("finished_at"),
        "summary_path": phases[next_phase]["summary_path"] if next_phase else phases["stage26_3"]["summary_path"],
        "blocking_reason": _blocking_reason(phases),
        "stage26_1_root": phases["stage26_1"]["root"] if phases["stage26_1"]["complete"] else "",
        "stage26_2_root": phases["stage26_2"]["root"] if phases["stage26_2"]["complete"] else "",
        "stage26_3_root": phases["stage26_3"]["root"] if phases["stage26_3"]["complete"] else "",
        "stage26_1_status": phases["stage26_1"]["summary"].get("status"),
        "stage26_2_status": phases["stage26_2"]["summary"].get("status"),
        "stage26_3_status": phases["stage26_3"]["summary"].get("status"),
        "stage26_1_source": phases["stage26_1"]["source"],
        "stage26_2_source": phases["stage26_2"]["source"],
        "stage26_3_source": phases["stage26_3"]["source"],
        "stage26_1_summary_exists": phases["stage26_1"]["summary_exists"],
        "stage26_2_summary_exists": phases["stage26_2"]["summary_exists"],
        "stage26_3_summary_exists": phases["stage26_3"]["summary_exists"],
        "main_final_coverage_delta": _float(final_summary.get("final_coverage_delta")),
        "main_coverage_auc_delta": _float(final_summary.get("coverage_auc_delta")),
        "main_coverage_per_100m_delta": _float(final_summary.get("coverage_per_100m_delta")),
        "hybrid_astar_path_cost_delta": _float(final_summary.get("hybrid_astar_path_cost_delta")),
        "coverage_denominator_source": final_summary.get("coverage_denominator_source") or stage26_8.COVERAGE_DENOMINATOR_SOURCE,
        "coverage_source": final_summary.get("coverage_source") or stage26_8.COVERAGE_SOURCE,
        "path_cost_source": final_summary.get("path_cost_source") or stage26_8.PATH_COST_SOURCE,
        "synthetic_source_kind": final_summary.get("synthetic_source_kind") or stage26_8.SYNTHETIC_SOURCE_KIND,
        "action_space_type": final_summary.get("action_space_type") or stage26_8.ACTION_SPACE_TYPE,
        "synthetic_terrain_hash": final_summary.get("synthetic_terrain_hash"),
        "platform_contract_hash": final_summary.get("platform_contract_hash"),
        "max_traversable_slope_deg": _float(final_summary.get("max_traversable_slope_deg") or 30.0),
        "lineage_mismatch": bool(final_summary) and stage26_8._lineage_mismatch(final_summary),
        "binding_or_safety_failure": _binding_or_safety_failure(phases),
        "execution_failure": bool(final_summary) and stage26_8._stage26_3_execution_failure(final_summary),
        "update_failure": phases["stage26_2"]["failed"],
        "efficiency_seed_passed": stage26_8._efficiency_seed_passed(final_summary),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _phase_record(phase: str, root: Path, source: str) -> dict[str, Any]:
    summary_path = root / _summary_file_for_phase(phase)
    summary = _read_json_if_exists(summary_path)
    return {
        "phase": phase,
        "root": str(root),
        "source": source,
        "summary_path": str(summary_path),
        "summary_exists": bool(summary),
        "summary": summary,
    }


def _phase_complete(phase: str, summary: dict[str, Any]) -> bool:
    if not summary:
        return False
    if phase in {"stage26_1", "stage26_2"}:
        return summary.get("status") == "passed"
    return not stage26_8._binding_or_safety_failure(summary) and not stage26_8._stage26_3_execution_failure(summary) and not stage26_8._lineage_mismatch(summary)


def _phase_failed(phase: str, summary: dict[str, Any]) -> bool:
    if not summary or _phase_complete(phase, summary):
        return False
    if phase == "stage26_1":
        return summary.get("status") != "passed"
    if phase == "stage26_2":
        return summary.get("status") != "passed"
    return stage26_8._binding_or_safety_failure(summary) or stage26_8._stage26_3_execution_failure(summary) or stage26_8._lineage_mismatch(summary)


def _next_phase(phases: dict[str, dict[str, Any]]) -> str | None:
    for phase in PHASES:
        if phases[phase]["failed"]:
            return None
        if not phases[phase]["complete"]:
            return phase
    return None


def _next_runnable_phases(job_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in job_rows:
        if row["status"] == "pending" and row["phase"] in PHASES:
            if row["phase"] == "stage26_1" or row.get("stage26_1_root"):
                if row["phase"] != "stage26_2" or row.get("stage26_1_root"):
                    if row["phase"] != "stage26_3" or row.get("stage26_2_root"):
                        rows.append(row)
    return rows


def _select_phase_executions(
    pending: list[dict[str, Any]],
    *,
    run_mode: str,
    horizon: int | None,
    seed: int | None,
    phase: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    if run_mode == "run_job":
        selected = [
            row
            for row in pending
            if (horizon is None or int(row["horizon_steps"]) == int(horizon))
            and (seed is None or int(row["seed"]) == int(seed))
            and (phase is None or row["phase"] == phase)
        ]
        return selected[: max(1, limit)]
    return pending[: max(1, limit)]


def _aggregate(job_rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in job_rows if row["status"] == "complete"]
    clean = [row for row in complete if not row["binding_or_safety_failure"] and not row["update_failure"] and not row["execution_failure"]]
    positive = [row for row in clean if row["efficiency_seed_passed"] and _float(row["main_coverage_per_100m_delta"]) > 0.0]
    negative = [row for row in clean if _float(row["main_coverage_per_100m_delta"]) < 0.0]
    horizon_rows = []
    for horizon in HORIZONS:
        rows = [row for row in job_rows if row["horizon_steps"] == horizon]
        completed = [row for row in rows if row["status"] == "complete"]
        h_positive = [row for row in completed if row["efficiency_seed_passed"] and _float(row["main_coverage_per_100m_delta"]) > 0.0]
        h_negative = [row for row in completed if _float(row["main_coverage_per_100m_delta"]) < 0.0]
        majority = math.floor(len(rows) / 2) + 1 if rows else 1
        horizon_rows.append(
            {
                "horizon_steps": horizon,
                "seed_count": len(rows),
                "completed_seed_count": len(completed),
                "positive_efficiency_seed_count": len(h_positive),
                "negative_efficiency_seed_count": len(h_negative),
                "majority_positive": len(h_positive) >= majority,
                "majority_negative": len(h_negative) >= majority,
                "mean_main_coverage_per_100m_delta": _mean([row["main_coverage_per_100m_delta"] for row in completed]),
            }
        )
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "job_count": len(job_rows),
        "completed_job_count": len(complete),
        "pending_job_count": len([row for row in job_rows if row["status"] == "pending"]),
        "failed_job_count": len([row for row in job_rows if row["status"] == "failed"]),
        "clean_job_count": len(clean),
        "positive_efficiency_job_count": len(positive),
        "negative_efficiency_job_count": len(negative),
        "binding_or_safety_failure_job_count": len([row for row in job_rows if row["binding_or_safety_failure"]]),
        "execution_failure_job_count": len([row for row in job_rows if row["execution_failure"]]),
        "update_failure_job_count": len([row for row in job_rows if row["update_failure"]]),
        "horizon_results": horizon_rows,
        "majority_positive_horizon_count": len([row for row in horizon_rows if row["majority_positive"]]),
        "majority_negative_horizon_count": len([row for row in horizon_rows if row["majority_negative"]]),
        "mean_main_coverage_per_100m_delta": _mean([row["main_coverage_per_100m_delta"] for row in complete]),
        "mean_main_final_coverage_delta": _mean([row["main_final_coverage_delta"] for row in complete]),
        "mean_main_coverage_auc_delta_diagnostic": _mean([row["main_coverage_auc_delta"] for row in complete]),
        "mean_hybrid_astar_path_cost_delta_diagnostic": _mean([row["hybrid_astar_path_cost_delta"] for row in complete]),
    }


def _route(
    *,
    boundary_rejections: list[str],
    input_rejections: list[str],
    job_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    resume_state_rejections: list[str],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections or not job_rows:
        return ROUTE_INPUTS
    if resume_state_rejections:
        return ROUTE_RESUME_STATE
    if int(aggregate["binding_or_safety_failure_job_count"]) > 0 or int(aggregate["execution_failure_job_count"]) > 0:
        return ROUTE_BINDING_OR_SAFETY
    if int(aggregate["update_failure_job_count"]) > 0:
        return ROUTE_UPDATE
    if int(aggregate["pending_job_count"]) > 0:
        return ROUTE_CONTINUE
    if int(aggregate["majority_positive_horizon_count"]) > 0:
        return ROUTE_STAGE26_9
    if int(aggregate["majority_negative_horizon_count"]) > 0:
        return ROUTE_CREDIT
    if int(aggregate["positive_efficiency_job_count"]) > 0:
        return ROUTE_EXPAND_SEEDS
    return ROUTE_POLICY_SIGNAL


def _carryover_audit(job_rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CARRYOVER_AUDIT_SCHEMA_VERSION,
        "stage26_8c_root": config["stage26_8c_root"],
        "carryover_phase_count": len(
            [
                (row, phase)
                for row in job_rows
                for phase in PHASES
                if row.get(f"{phase}_source") == "stage26_8c_carryover"
                and row.get(f"{phase}_summary_exists") is True
            ]
        ),
        "carried_over_completed_eval_count": len(
            [row for row in job_rows if row.get("stage26_3_source") == "stage26_8c_carryover" and row["status"] == "complete"]
        ),
        "carried_over_update_complete_next_eval_count": len(
            [
                row
                for row in job_rows
                if row.get("stage26_2_source") == "stage26_8c_carryover"
                and row["status"] == "pending"
                and row["phase"] == "stage26_3"
            ]
        ),
        "failed_or_incomplete_carryover_reused_as_success": False,
    }


def _job_plan_rows(job_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "job_id": row["job_id"],
            "horizon_steps": row["horizon_steps"],
            "seed": row["seed"],
            "status": row["status"],
            "next_phase": row["phase"],
            "resume_decision": row["resume_decision"],
            "output_root": row["output_root"],
        }
        for row in job_rows
    ]


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
    if stage26_8b_summary.get("coverage_source") != stage26_8.COVERAGE_SOURCE:
        reasons.append("stage26_8b_coverage_source_mismatch")
    if stage26_8b_summary.get("path_cost_source") != stage26_8.PATH_COST_SOURCE:
        reasons.append("stage26_8b_path_cost_source_mismatch")
    if stage26_8b_summary.get("synthetic_source_kind") != stage26_8.SYNTHETIC_SOURCE_KIND:
        reasons.append("stage26_8b_synthetic_source_kind_mismatch")
    for field in BOUNDARY_FIELDS:
        if stage26_8b_summary.get(field) is True:
            reasons.append(f"stage26_8b_{field}")
    if _float(stage26_8b_summary.get("canary_traffic_fraction")) > 0.0:
        reasons.append("stage26_8b_canary_traffic_fraction")
    return reasons


def _resume_state_rejections(job_rows: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for row in job_rows:
        if row["status"] == "complete" and not row.get("stage26_3_root"):
            reasons.append(f"{row['job_id']}_complete_without_stage26_3_root")
        if row["status"] == "pending" and row["phase"] == "stage26_2" and not row.get("stage26_1_root"):
            reasons.append(f"{row['job_id']}_stage26_2_without_stage26_1")
        if row["status"] == "pending" and row["phase"] == "stage26_3" and not row.get("stage26_2_root"):
            reasons.append(f"{row['job_id']}_stage26_3_without_stage26_2")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    if _float(config.get("canary_traffic_fraction")) > 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _binding_or_safety_failure(phases: dict[str, dict[str, Any]]) -> bool:
    if phases["stage26_1"]["failed"]:
        return True
    summary = phases["stage26_3"]["summary"]
    return bool(summary) and (stage26_8._binding_or_safety_failure(summary) or stage26_8._lineage_mismatch(summary))


def _job_complete(phases: dict[str, dict[str, Any]]) -> bool:
    return all(phases[phase]["complete"] for phase in PHASES)


def _job_failed(phases: dict[str, dict[str, Any]]) -> bool:
    return any(phases[phase]["failed"] for phase in PHASES)


def _blocking_reason(phases: dict[str, dict[str, Any]]) -> str:
    for phase in PHASES:
        if phases[phase]["failed"]:
            return f"{phase}_failed_or_invalid"
    return ""


def _resume_decision(phases: dict[str, dict[str, Any]]) -> str:
    if _job_complete(phases):
        return "complete_no_action"
    if _job_failed(phases):
        return "blocked_on_failed_or_invalid_phase"
    next_phase = _next_phase(phases)
    if next_phase == "stage26_1":
        return "run_stage26_1"
    if next_phase == "stage26_2":
        return "resume_after_stage26_1"
    if next_phase == "stage26_3":
        return "resume_after_stage26_2"
    return "unknown"


def _latest_execution(phases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    executions = [phases[phase].get("execution") or {} for phase in PHASES]
    executions = [item for item in executions if item]
    return executions[-1] if executions else {}


def _phase_dir(phase: str) -> str:
    return {"stage26_1": "s26_1", "stage26_2": "s26_2", "stage26_3": "s26_3"}[phase]


def _summary_file_for_phase(phase: str) -> str:
    return {
        "stage26_1": stage26_8.stage26_1.SUMMARY_FILE,
        "stage26_2": stage26_8.stage26_2.SUMMARY_FILE,
        "stage26_3": stage26_8.stage26_3.SUMMARY_FILE,
    }[phase]


def _job_output_root(output_root: Path, horizon: int, seed: int) -> Path:
    return output_root / f"h{horizon}" / f"s{seed}"


def _stage26_8_config(config: dict[str, Any], *, horizon_steps: int, repo_root: Path) -> dict[str, Any]:
    base = stage26_8._load_config(Path(config["stage26_8_base_config"]), repo_root)
    base.update(
        {
            "seeds": list(config["seeds"]),
            "required_scenario_count": int(config["required_scenario_count"]),
            "collector_rollout_steps": int(horizon_steps),
            "eval_rollout_steps": int(horizon_steps),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
            "stage21_5_timeout_seconds": 0.0,
            "post_update_success_metric": stage26_8.SUCCESS_METRIC,
            "coverage_denominator_source": stage26_8.COVERAGE_DENOMINATOR_SOURCE,
            "stop_on_first_seed_blocker": False,
            "reuse_existing_seed_outputs": True,
        }
    )
    return base


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    defaults = {
        "stage26_8b_root": DEFAULT_STAGE26_8B_ROOT,
        "stage26_8c_root": DEFAULT_STAGE26_8C_ROOT,
        "stage26_8_base_config": stage26_8.DEFAULT_CONFIG,
        "horizons": [16, 20],
        "seeds": [260801, 260802, 260803],
        "run_mode": "run_next",
        "max_jobs_per_invocation": 1,
        "required_scenario_count": 3,
        "dynamic_max_candidates_per_step": 36,
        "dynamic_proposal_pool_limit_per_step": 288,
        "hybrid_astar_candidate_eval_workers": 4,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    for field in ("stage26_8b_root", "stage26_8c_root", "stage26_8_base_config"):
        config[field] = str(_resolve_path(Path(str(config[field])), repo_root))
    config["horizons"] = [int(value) for value in config["horizons"]]
    config["seeds"] = [int(value) for value in config["seeds"]]
    if tuple(config["horizons"]) != HORIZONS:
        raise ValueError("horizons must be [16, 20] for Stage26.8D v1")
    if not config["seeds"]:
        raise ValueError("seeds must not be empty")
    if config["run_mode"] not in {"run_next", "aggregate_only", "run_job"}:
        raise ValueError("run_mode must be run_next, aggregate_only, or run_job")
    if int(config["max_jobs_per_invocation"]) < 1:
        raise ValueError("max_jobs_per_invocation must be >= 1")
    return config


def _read_json(path: Path) -> dict[str, Any]:
    return artifact_io.read_json(path)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not artifact_io.path_is_file(path):
        return {}
    return _read_json(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    artifact_io.write_json(path, payload)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    artifact_io.write_jsonl(path, rows)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isfinite(parsed):
        return parsed
    return 0.0


def _mean(values: list[Any]) -> float:
    parsed = [_float(value) for value in values]
    if not parsed:
        return 0.0
    return sum(parsed) / len(parsed)


def _render_report(summary: dict[str, Any], job_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage26.8D Resumable Seed/Horizon Execution",
        "",
        f"- status: `{summary['status']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- completed_job_count: `{summary['completed_job_count']}` / `{summary['job_count']}`",
        f"- pending_job_count: `{summary['pending_job_count']}`",
        f"- next_job_id: `{summary.get('next_job_id')}`",
        f"- next_phase: `{summary.get('next_phase')}`",
        "",
        "Jobs:",
    ]
    for row in job_rows:
        lines.append(
            f"- {row['job_id']}: status={row['status']}, phase={row['phase']}, "
            f"resume={row['resume_decision']}, per100m={row['main_coverage_per_100m_delta']}"
        )
    lines.extend(
        [
            "",
            "This stage only advances resumable experiment phases. AUC, total path length, and Hybrid A* path-cost deltas are diagnostic only.",
            "It does not publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
