from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_high_fidelity_exploration_coverage_comparison as hf
    import run_xunce_stage21_5_post_update_offline_trajectory_evaluation as stage21_5
    import run_xunce_stage26_1_synthetic_terrain_collector_smoke as stage26_1
    import run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as stage26_2
    import run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as stage26_3
    import run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as stage26_8
    import run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair as stage26_8i
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_high_fidelity_exploration_coverage_comparison as hf
    import scripts.run_xunce_stage21_5_post_update_offline_trajectory_evaluation as stage21_5
    import scripts.run_xunce_stage26_1_synthetic_terrain_collector_smoke as stage26_1
    import scripts.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as stage26_2
    import scripts.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as stage26_3
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as stage26_8
    import scripts.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair as stage26_8i


STAGE_ID = "xunce-stage26-8m-generalized-resumable-training-pipeline"
CONFIG_SCHEMA_VERSION = "xunce-stage26-8m-generalized-resumable-training-pipeline-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-8m-summary/v1"
JOB_PLAN_SCHEMA_VERSION = "xunce-stage26-8m-job-plan/v1"
JOB_STATE_SCHEMA_VERSION = "xunce-stage26-8m-job-state/v1"
EFFICIENCY_AGGREGATE_SCHEMA_VERSION = "xunce-stage26-8m-efficiency-aggregate/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-8m-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-8m-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_8m_generalized_resumable_training_pipeline_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8m_generalized_resumable_training_pipeline_v1"
)

SUMMARY_FILE = "xunce-stage26-8m-summary.json"
JOB_PLAN_FILE = "xunce-stage26-8m-job-plan.json"
JOB_STATE_FILE = "xunce-stage26-8m-job-state.jsonl"
EFFICIENCY_AGGREGATE_FILE = "xunce-stage26-8m-efficiency-aggregate.json"
ROUTING_FILE = "xunce-stage26-8m-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-8m-report.md"
MANIFEST_FILE = "xunce-stage26-8m-manifest.json"
JOB_SUMMARY_FILE = "job-summary.json"
JOB_STATE_LOCAL_FILE = "job-state.jsonl"

PHASES = ("collector", "update", "eval_pre", "eval_post", "aggregate")
RUN_MODES = ("run_next", "aggregate_only", "run_job", "run_phase")
BOUNDARY_FIELDS = ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary")

ROUTE_INPUTS = "rerun_stage26_8m_required_inputs"
ROUTE_RESUME_STATE = "repair_stage26_8m_resume_state_contract"
ROUTE_COLLECTOR = "repair_stage26_8m_collector_binding_or_safety"
ROUTE_UPDATE = "repair_stage26_8m_update_stability"
ROUTE_EVAL = "repair_stage26_8m_eval_binding_or_safety"
ROUTE_CONTINUE = "continue_stage26_8m_jobs"
ROUTE_INCREASE = "increase_stage26_synthetic_update_strength_or_sample_count"
ROUTE_CREDIT = "repair_stage26_synthetic_credit_assignment"
ROUTE_RESUME_DIVERSE = "resume_stage26_8d_seed_horizon_jobs_with_diverse_scenarios"
ROUTE_STAGE26_9 = "run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot"
ROUTE_BOUNDARY = "resolve_stage26_8m_boundary_rejections"

COUNT_FIELDS_REQUIRING_ZERO = (
    "synthetic_inference_required_field_missing_count",
    "hybrid_path_missing_provenance_count",
    "hybrid_path_contract_mismatch_count",
    "explicit_unreachable_selected_provenance_count",
    "pre_unreachable_selected_count",
    "post_unreachable_selected_count",
    "hard_risk_violation_count",
    "mask_violation_count",
    "path_planning_failure_count",
    "open_grid_fallback_count",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the generalized resumable Stage26.8M training pipeline.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-mode", choices=RUN_MODES)
    parser.add_argument("--job-id")
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--max-jobs-per-invocation", type=int)
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
        run_mode_override=args.run_mode,
        job_id_override=args.job_id,
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


def run_xunce_stage26_8m_generalized_resumable_training_pipeline(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    run_mode_override: str | None = None,
    job_id_override: str | None = None,
    phase_override: str | None = None,
    max_jobs_override: int | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(config_path, repo_root)
    if run_mode_override is not None:
        config["run_mode"] = run_mode_override
    if job_id_override is not None:
        config["job_id"] = job_id_override
    if phase_override is not None:
        config["phase"] = phase_override
    if max_jobs_override is not None:
        config["max_jobs_per_invocation"] = _positive_int(max_jobs_override, "max_jobs_per_invocation")
    _validate_run_selection(config)

    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    boundary_rejections = _boundary_rejections(config)
    input_rejections = _input_rejections(config)
    phase_executions: list[dict[str, Any]] = []
    job_rows = _scan_jobs(config=config, output_root=output_root, repo_root=repo_root)
    if not boundary_rejections and not input_rejections and config["run_mode"] != "aggregate_only":
        selected = _select_phase_executions(job_rows, config)
        for row in selected:
            phase_executions.append(_run_phase(row, config=config, output_root=output_root, repo_root=repo_root))

    job_rows = _scan_jobs(config=config, output_root=output_root, repo_root=repo_root, phase_executions=phase_executions)
    aggregate = _efficiency_aggregate(job_rows)
    resume_state_rejections = _resume_state_rejections(job_rows)
    route = _route(
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        resume_state_rejections=resume_state_rejections,
        job_rows=job_rows,
        aggregate=aggregate,
    )
    status = "passed" if route in {ROUTE_CONTINUE, ROUTE_INCREASE, ROUTE_RESUME_DIVERSE, ROUTE_STAGE26_9} else "failed"
    next_item = _first_pending(job_rows)
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
        "next_job_id": next_item.get("job_id"),
        "next_phase": next_item.get("phase"),
        "config_hash": _experiment_config_hash(config),
        "source_scenario_fixture_root": config["source_scenario_fixture_root"],
        "state_files_are_under_output_root": True,
        "coverage_denominator_source": config["coverage_denominator_source"],
        "post_update_success_metric": config["post_update_success_metric"],
        "coverage_source": config["coverage_source"],
        "path_cost_source": config["path_cost_source"],
        "synthetic_source_kind": config["synthetic_source_kind"],
        "action_space_type": config["action_space_type"],
        "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
        "max_traversable_slope_deg": float(config["max_traversable_slope_deg"]),
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
        "resume_state_rejections": resume_state_rejections,
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
        "resume_state_rejections": resume_state_rejections,
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
            "efficiency_aggregate": str(output_root / EFFICIENCY_AGGREGATE_FILE),
            "routing": str(output_root / ROUTING_FILE),
            "report": str(output_root / REPORT_FILE),
        },
    }

    _write_json(output_root / JOB_PLAN_FILE, {"schema_version": JOB_PLAN_SCHEMA_VERSION, "jobs": _job_plan_rows(job_rows)})
    _write_jsonl(output_root / JOB_STATE_FILE, job_rows)
    _write_per_job_state(job_rows)
    _write_json(output_root / EFFICIENCY_AGGREGATE_FILE, aggregate)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary, job_rows), encoding="utf-8")
    return summary


def _run_phase(row: dict[str, Any], *, config: dict[str, Any], output_root: Path, repo_root: Path) -> dict[str, Any]:
    started_at = _utc_now()
    job = _job_by_id(config, str(row["job_id"]))
    phase = str(row["phase"])
    job_root = _job_output_root(output_root, job)
    try:
        if phase == "collector":
            root = job_root / "collector"
            cfg_path = job_root / "xunce-stage26-8m-collector-config.json"
            _write_json(cfg_path, _build_collector_config(config, job, job_root, repo_root))
            summary = stage26_1.run_xunce_stage26_1_synthetic_terrain_collector_smoke(
                config_path=cfg_path,
                output_root=root,
                repo_root=repo_root,
            )
            _ensure_summary(root / stage26_1.SUMMARY_FILE, summary)
        elif phase == "update":
            root = job_root / "update"
            cfg_path = job_root / "xunce-stage26-8m-update-config.json"
            _write_json(cfg_path, _build_update_config(config, job, job_root, repo_root))
            summary = stage26_2.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
                config_path=cfg_path,
                output_root=root,
                repo_root=repo_root,
            )
            _ensure_summary(root / stage26_2.SUMMARY_FILE, summary)
        elif phase in {"eval_pre", "eval_post"}:
            root = job_root / "eval" / ("pre" if phase == "eval_pre" else "post")
            cfg = _build_stage21_5_eval_config(config, job, job_root, repo_root)
            cfg_path = job_root / "xunce-stage26-8m-stage21-5-config.json"
            _write_json(cfg_path, cfg)
            checkpoint = _checkpoint_for_eval(job_root / "update", label="pre" if phase == "eval_pre" else "post")
            stage21_5._run_high_fidelity_eval(
                cfg,
                repo_root=repo_root,
                output_root=root,
                checkpoint_path=checkpoint,
                label="pre" if phase == "eval_pre" else "post",
            )
            summary = _read_json_if_exists(root / hf.SUMMARY_FILE)
        elif phase == "aggregate":
            root = job_root / "aggregate"
            cfg_path = job_root / "xunce-stage26-8m-aggregate-config.json"
            _write_json(cfg_path, _build_aggregate_config(config, job, job_root, repo_root))
            summary = stage26_3.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke(
                config_path=cfg_path,
                output_root=root,
                repo_root=repo_root,
            )
            _ensure_summary(root / stage26_3.SUMMARY_FILE, summary)
        else:  # pragma: no cover
            raise ValueError(f"unknown phase: {phase}")
        status = summary.get("status")
        blocking_reason = "" if summary else "summary_missing_after_execution"
    except Exception as exc:  # pragma: no cover - exercised through tests by monkeypatch
        root = _phase_root(job_root, phase)
        status = "failed"
        blocking_reason = f"{type(exc).__name__}:{exc}"
        summary = {}
    return {
        "job_id": job["job_id"],
        "phase": phase,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "output_root": str(root),
        "summary_path": str(_phase_summary_path(job_root, phase)),
        "status": status,
        "blocking_reason": blocking_reason,
        "next_required_change": summary.get("next_required_change"),
    }


def _scan_jobs(
    *,
    config: dict[str, Any],
    output_root: Path,
    repo_root: Path,
    phase_executions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    execution_by_key = {(str(row["job_id"]), str(row["phase"])): row for row in phase_executions or []}
    existing_state = _existing_state_by_key(output_root)
    rows: list[dict[str, Any]] = []
    for job in _expand_jobs(config, output_root, repo_root):
        job_root = _job_output_root(output_root, job)
        phase_records = {phase: _phase_record(job_root, phase, job, existing_state, execution_by_key) for phase in PHASES}
        next_phase = _next_phase(phase_records)
        job_status = _job_status(phase_records)
        for phase in PHASES:
            record = phase_records[phase]
            rows.append(
                {
                    "schema_version": JOB_STATE_SCHEMA_VERSION,
                    "job_id": job["job_id"],
                    "phase": phase,
                    "subphase": "",
                    "status": record["status"],
                    "resume_decision": _resume_decision(record, phase, next_phase, job_status),
                    "source_root": record["source_root"],
                    "output_root": record["output_root"],
                    "summary_path": record["summary_path"],
                    "blocking_reason": record["blocking_reason"],
                    "started_at": record.get("started_at"),
                    "finished_at": record.get("finished_at"),
                    "horizon": job["horizon"],
                    "seed": job["seed"],
                    "scenario_count": job["scenario_count"],
                    "collector_rollout_steps": job["collector_rollout_steps"],
                    "eval_rollout_steps": job["eval_rollout_steps"],
                    "update_combo_id": job["update_combo_id"],
                    "config_hash": job["config_hash"],
                    "input_hash": job["input_hash"],
                    "source_scenario_fixture_root": config["source_scenario_fixture_root"],
                    "summary_status": record["summary"].get("status"),
                    "next_required_change": record["summary"].get("next_required_change"),
                    "lineage_mismatch": _lineage_mismatch(record["summary"]) if phase == "aggregate" and record["summary"] else False,
                    "binding_or_safety_failure": _binding_or_safety_failure(record["summary"]),
                    "execution_failure": _aggregate_execution_failure(record["summary"]) if phase == "aggregate" else False,
                    "policy_delta_too_small_for_margin": _policy_delta_too_small(record["summary"]) if phase == "aggregate" else False,
                    "selected_action_changed_count": int(record["summary"].get("selected_action_changed_count") or 0),
                    "main_coverage_per_100m_delta": _coverage_per_100m_delta(record["summary"]) if phase == "aggregate" else 0.0,
                    "main_final_coverage_delta": _first_float(record["summary"].get("main_final_coverage_delta"), record["summary"].get("final_coverage_delta")) if phase == "aggregate" else 0.0,
                    "coverage_denominator_source": config["coverage_denominator_source"],
                    "coverage_source": config["coverage_source"],
                    "path_cost_source": config["path_cost_source"],
                    "synthetic_source_kind": config["synthetic_source_kind"],
                    "action_space_type": config["action_space_type"],
                    "max_traversable_slope_deg": float(config["max_traversable_slope_deg"]),
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "connects_real_executor": False,
                    "starts_online_canary": False,
                    "canary_traffic_fraction": 0.0,
                }
            )
    return rows


def _phase_record(
    job_root: Path,
    phase: str,
    job: dict[str, Any],
    existing_state: dict[tuple[str, str], dict[str, Any]],
    execution_by_key: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    root = _phase_root(job_root, phase)
    summary_path = _phase_summary_path(job_root, phase)
    summary = _read_json_if_exists(summary_path)
    execution = execution_by_key.get((job["job_id"], phase), {})
    existing = existing_state.get((job["job_id"], phase), {})
    hash_mismatch = bool(existing.get("config_hash")) and str(existing.get("config_hash")) != str(job["config_hash"])
    complete = _phase_complete(phase, root, summary, job["max_abs_approx_kl"])
    failed = _phase_failed(phase, root, summary, job["max_abs_approx_kl"])
    if hash_mismatch:
        status = "failed"
        reason = "config_hash_mismatch"
    elif complete:
        status = "complete"
        reason = ""
    elif failed:
        status = "failed"
        reason = _phase_blocking_reason(phase, root, summary, job["max_abs_approx_kl"])
    else:
        status = "pending"
        reason = _phase_blocking_reason(phase, root, summary, job["max_abs_approx_kl"])
    if execution.get("blocking_reason"):
        reason = str(execution["blocking_reason"])
    return {
        "status": status,
        "source_root": str(root),
        "output_root": str(root),
        "summary_path": str(summary_path),
        "summary": summary,
        "blocking_reason": reason,
        "started_at": execution.get("started_at"),
        "finished_at": execution.get("finished_at"),
    }


def _phase_complete(phase: str, root: Path, summary: dict[str, Any], max_abs_approx_kl: float) -> bool:
    if not summary:
        return False
    if phase == "collector":
        return summary.get("status") == "passed" and not _binding_or_safety_failure(summary)
    if phase == "update":
        return stage26_8i._stage26_2_is_stable(summary, max_abs_approx_kl)
    if phase in {"eval_pre", "eval_post"}:
        return (
            summary.get("status") == "passed"
            and (root / hf.MODEL_INFERENCE_FILE).is_file()
            and (root / hf.EPISODES_FILE).is_file()
            and _jsonl_count(root / hf.MODEL_INFERENCE_FILE) > 0
            and _jsonl_count(root / hf.EPISODES_FILE) > 0
        )
    if phase == "aggregate":
        return summary.get("schema_version") == stage26_3.SUMMARY_SCHEMA_VERSION
    return False


def _phase_failed(phase: str, root: Path, summary: dict[str, Any], max_abs_approx_kl: float) -> bool:
    if not summary:
        return False
    if phase == "collector":
        return summary.get("status") == "failed" or _binding_or_safety_failure(summary)
    if phase == "update":
        return summary.get("status") == "failed" or (summary.get("status") == "passed" and not _phase_complete(phase, root, summary, max_abs_approx_kl))
    if phase in {"eval_pre", "eval_post"}:
        return summary.get("status") == "failed"
    return False


def _next_phase(records: dict[str, dict[str, Any]]) -> str | None:
    for phase in PHASES:
        if records[phase]["status"] == "failed":
            return None
        if records[phase]["status"] != "complete":
            return phase
    return None


def _job_status(records: dict[str, dict[str, Any]]) -> str:
    if any(records[phase]["status"] == "failed" for phase in PHASES):
        return "failed"
    if all(records[phase]["status"] == "complete" for phase in PHASES):
        return "complete"
    return "pending"


def _select_phase_executions(job_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    mode = str(config["run_mode"])
    if mode == "aggregate_only":
        return []
    candidates = _pending_runnable_rows(job_rows)
    if mode in {"run_job", "run_phase"}:
        candidates = [row for row in candidates if row["job_id"] == config.get("job_id")]
    if mode == "run_phase":
        candidates = [row for row in candidates if row["phase"] == config.get("phase")]
    return candidates[: int(config["max_jobs_per_invocation"])]


def _pending_runnable_rows(job_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_job: dict[str, list[dict[str, Any]]] = {}
    for row in job_rows:
        by_job.setdefault(str(row["job_id"]), []).append(row)
    runnable: list[dict[str, Any]] = []
    phase_rank = {phase: index for index, phase in enumerate(PHASES)}
    for rows in by_job.values():
        ordered = sorted(rows, key=lambda row: phase_rank[str(row["phase"])])
        for row in ordered:
            if row["status"] == "failed":
                break
            if row["status"] == "pending":
                prior_complete = all(prev["status"] == "complete" for prev in ordered[: phase_rank[str(row["phase"])]])
                if prior_complete:
                    runnable.append(row)
                break
    return sorted(runnable, key=lambda row: str(row["job_id"]))


def _build_collector_config(config: dict[str, Any], job: dict[str, Any], job_root: Path, repo_root: Path) -> dict[str, Any]:
    cfg = _read_json(Path(config["base_stage26_1_config"]))
    stage21_1_base = _read_json_if_exists(_resolve_path(Path(str(cfg.get("stage21_1_base_config", ""))), repo_root))
    if stage21_1_base:
        stage21_1_base["sampling_seed"] = int(job["seed"])
        stage21_1_base["continuous_theta_head_init_seed"] = int(job["seed"])
        generated_stage21_1_base = job_root / "xunce-stage26-8m-stage21-1-base-config.json"
        _write_json(generated_stage21_1_base, stage21_1_base)
        cfg["stage21_1_base_config"] = str(generated_stage21_1_base)
    cfg.update(
        {
            "stage26_8m_job_id": job["job_id"],
            "stage26_8m_seed": int(job["seed"]),
            "required_scenario_count": int(job["scenario_count"]),
            "rollout_steps": int(job["collector_rollout_steps"]),
            "action_space_type": config["action_space_type"],
            "continuous_theta_action_space_enabled": True,
            "synthetic_credit_feature_exposure_enabled": True,
            "synthetic_exploration_credit_enabled": True,
            "allow_synthetic_credit_behavior_policy": True,
            "scenario_diversity_contract_enabled": True,
            "scenario_diversity_source": "synthetic_roi_start_seed_matrix/v1",
            "source_scenario_fixture_root": config["source_scenario_fixture_root"],
            "stage26_8m_source_scenario_fixture_root": config["source_scenario_fixture_root"],
            "scenario_seed_base": int(job["seed"]),
            "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
            "max_traversable_slope_deg": float(config["max_traversable_slope_deg"]),
            "stage26_1_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _build_update_config(config: dict[str, Any], job: dict[str, Any], job_root: Path, repo_root: Path) -> dict[str, Any]:
    combo = job["update_combo"]
    cfg = _read_json(Path(config["base_stage26_2_config"]))
    cfg.update(
        {
            "stage26_8m_job_id": job["job_id"],
            "stage26_8m_update_combo_id": job["update_combo_id"],
            "stage26_1_root": str(job_root / "collector"),
            "epochs": int(combo["epochs"]),
            "learning_rate": float(combo["learning_rate"]),
            "clip_ratio": float(config.get("clip_ratio", 0.2)),
            "policy_loss_coefficient": float(combo["policy_loss_coefficient"]),
            "value_loss_coefficient": float(combo["value_loss_coefficient"]),
            "entropy_coefficient": float(combo["entropy_coefficient"]),
            "loss_scale": float(combo["loss_scale"]),
            "max_grad_norm": float(config.get("max_grad_norm", 1.0)),
            "max_abs_approx_kl": float(config["max_abs_approx_kl"]),
            "stage26_2_authorized": False,
            "runs_new_ppo_update": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _build_aggregate_config(config: dict[str, Any], job: dict[str, Any], job_root: Path, repo_root: Path) -> dict[str, Any]:
    cfg = _build_stage26_3_base_config(config, job, job_root, repo_root)
    cfg.update(
        {
            "execute_high_fidelity_evaluations": False,
            "pre_ppo_evaluation_root": str(job_root / "eval" / "pre"),
            "post_ppo_evaluation_root": str(job_root / "eval" / "post"),
        }
    )
    return cfg


def _build_stage26_3_base_config(config: dict[str, Any], job: dict[str, Any], job_root: Path, repo_root: Path) -> dict[str, Any]:
    cfg = _read_json(Path(config["base_stage26_3_config"]))
    cfg.update(
        {
            "stage26_8m_job_id": job["job_id"],
            "stage26_2_root": str(job_root / "update"),
            "required_scenario_count": int(job["scenario_count"]),
            "rollout_steps": int(job["eval_rollout_steps"]),
            "coverage_denominator_mode": "main_coverable_cells",
            "coverage_denominator_source": config["coverage_denominator_source"],
            "post_update_success_metric": config["post_update_success_metric"],
            "coverage_source": config["coverage_source"],
            "path_cost_source": config["path_cost_source"],
            "synthetic_source_kind": config["synthetic_source_kind"],
            "action_space_type": config["action_space_type"],
            "continuous_theta_action_space_enabled": True,
            "synthetic_credit_feature_exposure_enabled": True,
            "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
            "stage21_5_timeout_seconds": 0.0,
            "stage26_3_authorized": False,
            "release_or_training_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _build_stage21_5_eval_config(config: dict[str, Any], job: dict[str, Any], job_root: Path, repo_root: Path) -> dict[str, Any]:
    stage26_3_config = stage26_3._load_config(_write_temp_stage26_3_config(config, job, job_root, repo_root), repo_root)
    stage26_2_summary = _read_json_if_exists(job_root / "update" / stage26_2.SUMMARY_FILE)
    stage26_1_summary = stage26_3._stage26_1_summary(stage26_2_summary)
    source_roi_root = stage26_3._source_roi_expansion_root(stage26_1_summary, stage26_2_summary)
    stage21_5_cfg = _read_json(Path(stage26_3_config["stage21_5_base_config"]))
    high_fidelity_cfg = _read_json(Path(stage26_3_config["high_fidelity_config"]))
    common = {
        "theta_aware_candidate_viewpoints_enabled": True,
        "slope_obstacle_aware_theta_reward_enabled": True,
        "hybrid_astar_pose_path_cost_enabled": True,
        "synthetic_terrain_contract_enabled": True,
        "obstacle_occlusion_enabled": True,
        "coverage_source": config["coverage_source"],
        "path_cost_source": config["path_cost_source"],
        "coverage_denominator_mode": "main_coverable_cells",
        "coverage_denominator_source": config["coverage_denominator_source"],
        "post_update_success_metric": config["post_update_success_metric"],
        "execute_high_fidelity_evaluations": True,
        "pre_ppo_evaluation_root": str(job_root / "eval" / "pre"),
        "post_ppo_evaluation_root": str(job_root / "eval" / "post"),
        "synthetic_terrain_model_id": stage26_3.SYNTHETIC_MODEL_ID,
        "synthetic_terrain_hash": str(stage26_2_summary.get("synthetic_terrain_hash") or stage26_3_config.get("synthetic_terrain_hash") or ""),
        "synthetic_source_kind": config["synthetic_source_kind"],
        "synthetic_los_blocker_cells_used": True,
        "synthetic_hard_obstacle_cells_used": True,
        "physical_obstacle_cells_written": False,
        "max_traversable_slope_deg": float(config["max_traversable_slope_deg"]),
        "required_scenario_count": int(job["scenario_count"]),
        "rollout_steps": int(job["eval_rollout_steps"]),
        "dynamic_max_candidates_per_step": int(stage26_3_config["dynamic_max_candidates_per_step"]),
        "dynamic_proposal_pool_limit_per_step": int(stage26_3_config["dynamic_proposal_pool_limit_per_step"]),
        "theta_bin_count": int(stage26_3_config["theta_bin_count"]),
        "theta_step_deg": int(stage26_3_config["theta_step_deg"]),
        "sensor_model_id": str(stage26_3_config["sensor_model_id"]),
        "sensor_fov_deg": float(stage26_3_config["sensor_fov_deg"]),
        "sensor_range_cells": int(stage26_3_config["sensor_range_cells"]),
        **{field: stage26_3_config[field] for field in stage26_3.HYBRID_ASTAR_PLANNER_FIELDS},
        "default_astar_replaced": False,
        "hybrid_astar_ackermann_feasible_claimed": False,
        "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
        "continuous_theta_action_space_enabled": True,
        "action_space_type": config["action_space_type"],
        "synthetic_credit_feature_exposure_enabled": True,
        "source_scenario_fixture_root": config["source_scenario_fixture_root"],
        "stage26_8m_source_scenario_fixture_root": config["source_scenario_fixture_root"],
        "include_oracle_baselines": False,
        "include_canonical_reward_rerank_oracle": False,
        "xunce_only_evaluation": True,
        "emit_candidate_metric_audit": True,
        "stage21_5_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    high_fidelity_cfg.update(
        {
            **common,
            "source_roi_expansion_root": source_roi_root,
            "dynamic_validation_work_root": str(job_root / "eval" / "_xunce_dynamic_validation_work"),
        }
    )
    high_fidelity_path = job_root / "xunce-stage26-8m-high-fidelity-config.json"
    _write_json(high_fidelity_path, high_fidelity_cfg)
    stage21_5_cfg.update(
        {
            **common,
            "stage21_4_tiny_ppo_update_smoke_root": str(job_root / "update" / "s21_4"),
            "high_fidelity_config": str(high_fidelity_path),
        }
    )
    return stage21_5_cfg


def _write_temp_stage26_3_config(config: dict[str, Any], job: dict[str, Any], job_root: Path, repo_root: Path) -> Path:
    path = job_root / "xunce-stage26-8m-stage26-3-base-config.json"
    _write_json(path, _build_stage26_3_base_config(config, job, job_root, repo_root))
    return path


def _checkpoint_for_eval(update_root: Path, *, label: str) -> Path:
    stage21_4_root = update_root / "s21_4"
    summary = _read_json_if_exists(stage21_4_root / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json")
    key = "source_xunce_candidate_checkpoint" if label == "pre" else "experimental_checkpoint_path"
    checkpoint = summary.get(key)
    if not checkpoint:
        raise RuntimeError(f"missing {key} in Stage21.4 summary")
    return Path(str(checkpoint))


def _expand_jobs(config: dict[str, Any], output_root: Path, repo_root: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for horizon in config["horizons"]:
        for seed in config["seeds"]:
            for scenario_count in config["scenario_counts"]:
                for collector_steps in config["collector_rollout_steps"]:
                    for eval_steps in config["eval_rollout_steps"]:
                        for combo in config["update_combos"]:
                            job = {
                                "horizon": int(horizon),
                                "seed": int(seed),
                                "scenario_count": int(scenario_count),
                                "collector_rollout_steps": int(collector_steps),
                                "eval_rollout_steps": int(eval_steps),
                                "update_combo_id": str(combo["combo_id"]),
                                "update_combo": combo,
                                "max_abs_approx_kl": float(config["max_abs_approx_kl"]),
                            }
                            job["job_id"] = _job_id(job)
                            job["config_hash"] = _job_config_hash(config, job)
                            job["input_hash"] = _input_hash(config)
                            jobs.append(job)
    return jobs


def _job_id(job: dict[str, Any]) -> str:
    return (
        f"h{job['horizon']}_s{job['seed']}_sc{job['scenario_count']}_"
        f"cr{job['collector_rollout_steps']}_er{job['eval_rollout_steps']}_u_{job['update_combo_id']}"
    )


def _job_output_root(output_root: Path, job: dict[str, Any]) -> Path:
    return output_root / "jobs" / str(job["job_id"])


def _phase_root(job_root: Path, phase: str) -> Path:
    if phase == "collector":
        return job_root / "collector"
    if phase == "update":
        return job_root / "update"
    if phase == "eval_pre":
        return job_root / "eval" / "pre"
    if phase == "eval_post":
        return job_root / "eval" / "post"
    if phase == "aggregate":
        return job_root / "aggregate"
    raise ValueError(f"unknown phase: {phase}")


def _phase_summary_path(job_root: Path, phase: str) -> Path:
    root = _phase_root(job_root, phase)
    if phase == "collector":
        return root / stage26_1.SUMMARY_FILE
    if phase == "update":
        return root / stage26_2.SUMMARY_FILE
    if phase in {"eval_pre", "eval_post"}:
        return root / hf.SUMMARY_FILE
    if phase == "aggregate":
        return root / stage26_3.SUMMARY_FILE
    raise ValueError(f"unknown phase: {phase}")


def _job_by_id(config: dict[str, Any], job_id: str) -> dict[str, Any]:
    for job in _expand_jobs(config, Path(config.get("_output_root_hint", DEFAULT_OUTPUT_ROOT)), Path(".")):
        if job["job_id"] == job_id:
            return job
    raise ValueError(f"unknown job_id: {job_id}")


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    if payload.get("stage_id") != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID}")
    defaults = {
        "run_mode": "run_next",
        "max_jobs_per_invocation": 1,
        "base_stage26_1_config": "configs/xunce_stage26_1_synthetic_terrain_collector_smoke_v1.json",
        "base_stage26_2_config": "configs/xunce_stage26_2_synthetic_terrain_ppo_update_smoke_v1.json",
        "base_stage26_3_config": "configs/xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke_v1.json",
        "source_scenario_fixture_root": "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/outputs/path_feedback_batch_xunce_stage26_8g_repair_synthetic_scenario_diversity_v1",
        "horizons": [16],
        "seeds": [260801],
        "scenario_counts": [3],
        "collector_rollout_steps": [16],
        "eval_rollout_steps": [16],
        "update_combos": [
            {
                "combo_id": "policy_amp_x2",
                "epochs": 16,
                "learning_rate": 3.0e-5,
                "policy_loss_coefficient": 3.0,
                "value_loss_coefficient": 0.01,
                "entropy_coefficient": 0.005,
                "loss_scale": 0.5,
            }
        ],
        "clip_ratio": 0.2,
        "max_grad_norm": 1.0,
        "max_abs_approx_kl": 1.5,
        "coverage_denominator_source": stage26_8.COVERAGE_DENOMINATOR_SOURCE,
        "post_update_success_metric": stage26_8.SUCCESS_METRIC,
        "coverage_source": stage26_8.COVERAGE_SOURCE,
        "path_cost_source": stage26_8.PATH_COST_SOURCE,
        "synthetic_source_kind": stage26_8.SYNTHETIC_SOURCE_KIND,
        "action_space_type": stage26_8.ACTION_SPACE_TYPE,
        "hybrid_astar_candidate_eval_workers": 4,
        "max_traversable_slope_deg": 30.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    config["run_mode"] = str(config["run_mode"])
    if config["run_mode"] not in RUN_MODES:
        raise ValueError(f"run_mode must be one of {RUN_MODES}")
    config["max_jobs_per_invocation"] = _positive_int(config["max_jobs_per_invocation"], "max_jobs_per_invocation")
    config["horizons"] = _positive_int_list(config["horizons"], "horizons")
    config["seeds"] = _positive_int_list(config["seeds"], "seeds")
    config["scenario_counts"] = _positive_int_list(config["scenario_counts"], "scenario_counts")
    config["collector_rollout_steps"] = _positive_int_list(config["collector_rollout_steps"], "collector_rollout_steps")
    config["eval_rollout_steps"] = _positive_int_list(config["eval_rollout_steps"], "eval_rollout_steps")
    config["update_combos"] = _update_combos(config["update_combos"])
    config["max_abs_approx_kl"] = float(config["max_abs_approx_kl"])
    if not math.isfinite(config["max_abs_approx_kl"]) or config["max_abs_approx_kl"] > 1.5:
        raise ValueError("max_abs_approx_kl must be finite and must not exceed 1.5")
    config["hybrid_astar_candidate_eval_workers"] = _positive_int(config["hybrid_astar_candidate_eval_workers"], "hybrid_astar_candidate_eval_workers")
    config["max_traversable_slope_deg"] = float(config["max_traversable_slope_deg"])
    for field in ("base_stage26_1_config", "base_stage26_2_config", "base_stage26_3_config", "source_scenario_fixture_root"):
        config[field] = str(_resolve_path(Path(str(config[field])), repo_root))
    for field in BOUNDARY_FIELDS:
        config[field] = bool(config.get(field, False))
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0))
    return config


def _update_combos(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("update_combos must be a non-empty array")
    combos: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError("update_combos entries must be objects")
        combo_id = str(item.get("combo_id") or f"combo_{index:02d}")
        if combo_id in seen:
            raise ValueError(f"duplicate update combo_id: {combo_id}")
        seen.add(combo_id)
        combos.append(
            {
                "combo_id": combo_id,
                "epochs": _positive_int(item.get("epochs", 1), "epochs"),
                "learning_rate": _positive_float(item.get("learning_rate", 1.0e-5), "learning_rate"),
                "policy_loss_coefficient": _nonnegative_float(item.get("policy_loss_coefficient", 1.0), "policy_loss_coefficient"),
                "value_loss_coefficient": _nonnegative_float(item.get("value_loss_coefficient", 0.02), "value_loss_coefficient"),
                "entropy_coefficient": _nonnegative_float(item.get("entropy_coefficient", 0.01), "entropy_coefficient"),
                "loss_scale": _positive_float(item.get("loss_scale", 0.25), "loss_scale"),
            }
        )
    return combos


def _validate_run_selection(config: dict[str, Any]) -> None:
    mode = str(config["run_mode"])
    if mode in {"run_job", "run_phase"} and not config.get("job_id"):
        raise ValueError(f"{mode} requires --job-id or config.job_id")
    if mode == "run_phase" and not config.get("phase"):
        raise ValueError("run_phase requires --phase or config.phase")
    if config.get("phase") and str(config["phase"]) not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")


def _route(
    *,
    boundary_rejections: list[str],
    input_rejections: list[str],
    resume_state_rejections: list[str],
    job_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections:
        return ROUTE_INPUTS
    if resume_state_rejections:
        return ROUTE_RESUME_STATE
    if any(row["phase"] == "collector" and row["status"] == "failed" for row in job_rows):
        return ROUTE_COLLECTOR
    if any(row["phase"] == "update" and row["status"] == "failed" for row in job_rows):
        return ROUTE_UPDATE
    if any(row["phase"] in {"eval_pre", "eval_post", "aggregate"} and row["status"] == "failed" for row in job_rows):
        return ROUTE_EVAL
    if any(row["status"] == "pending" for row in job_rows):
        return ROUTE_CONTINUE
    if int(aggregate.get("majority_positive_job_count") or 0) > 0:
        return ROUTE_STAGE26_9
    if int(aggregate.get("clean_action_changed_nonnegative_efficiency_count") or 0) > 0:
        return ROUTE_RESUME_DIVERSE
    if int(aggregate.get("clean_action_changed_negative_efficiency_count") or 0) > 0:
        return ROUTE_CREDIT
    return ROUTE_INCREASE


def _efficiency_aggregate(job_rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate_rows = [row for row in job_rows if row["phase"] == "aggregate" and row["status"] == "complete"]
    completed_job_count = len(aggregate_rows)
    positive_rows = [row for row in aggregate_rows if float(row.get("main_coverage_per_100m_delta") or 0.0) > 0.0]
    changed_rows = [row for row in aggregate_rows if int(row.get("selected_action_changed_count") or 0) > 0]
    clean_changed_nonnegative = [
        row for row in changed_rows if float(row.get("main_coverage_per_100m_delta") or 0.0) >= 0.0 and not row.get("binding_or_safety_failure")
    ]
    clean_changed_negative = [
        row for row in changed_rows if float(row.get("main_coverage_per_100m_delta") or 0.0) < 0.0 and not row.get("binding_or_safety_failure")
    ]
    total_jobs = len({row["job_id"] for row in job_rows})
    return {
        "schema_version": EFFICIENCY_AGGREGATE_SCHEMA_VERSION,
        "job_count": total_jobs,
        "completed_job_count": completed_job_count,
        "pending_job_count": len({row["job_id"] for row in job_rows if row["status"] == "pending"}),
        "failed_job_count": len({row["job_id"] for row in job_rows if row["status"] == "failed"}),
        "positive_efficiency_job_count": len(positive_rows),
        "majority_positive_job_count": 1 if completed_job_count > 0 and len(positive_rows) > completed_job_count / 2.0 else 0,
        "clean_action_changed_nonnegative_efficiency_count": len(clean_changed_nonnegative),
        "clean_action_changed_negative_efficiency_count": len(clean_changed_negative),
        "mean_main_coverage_per_100m_delta": _mean([row.get("main_coverage_per_100m_delta") for row in aggregate_rows]),
        "completed_job_ids": [row["job_id"] for row in aggregate_rows],
    }


def _input_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in ("base_stage26_1_config", "base_stage26_2_config", "base_stage26_3_config"):
        if not Path(str(config[field])).is_file():
            reasons.append(f"{field}_missing")
    if config.get("source_scenario_fixture_root") and not Path(str(config["source_scenario_fixture_root"])).exists():
        reasons.append("source_scenario_fixture_root_missing")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _resume_state_rejections(job_rows: list[dict[str, Any]]) -> list[str]:
    return _unique([str(row["blocking_reason"]) for row in job_rows if row["blocking_reason"] == "config_hash_mismatch"])


def _phase_blocking_reason(phase: str, root: Path, summary: dict[str, Any], max_abs_approx_kl: float) -> str:
    if not summary:
        return "summary_missing"
    if phase == "update" and not stage26_8i._stage26_2_is_stable(summary, max_abs_approx_kl):
        return "update_not_stable"
    if phase in {"eval_pre", "eval_post"}:
        if summary.get("status") != "passed":
            return "summary_not_passed"
        if not (root / hf.MODEL_INFERENCE_FILE).is_file():
            return "model_inference_missing"
        if not (root / hf.EPISODES_FILE).is_file():
            return "episodes_missing"
    if phase == "collector" and summary.get("status") != "passed":
        return "summary_not_passed"
    return ""


def _resume_decision(record: dict[str, Any], phase: str, next_phase: str | None, job_status: str) -> str:
    if record["status"] == "complete":
        return "reuse_completed_artifact"
    if record["status"] == "failed":
        return "blocked_requires_repair_or_new_root"
    if phase == next_phase and job_status == "pending":
        return "run_next_pending_phase"
    return "wait_for_dependency"


def _first_pending(job_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = _pending_runnable_rows(job_rows)
    return rows[0] if rows else {}


def _job_plan_rows(job_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for row in job_rows:
        if row["job_id"] in seen:
            continue
        seen.add(str(row["job_id"]))
        rows.append(
            {
                "job_id": row["job_id"],
                "horizon": row["horizon"],
                "seed": row["seed"],
                "scenario_count": row["scenario_count"],
                "collector_rollout_steps": row["collector_rollout_steps"],
                "eval_rollout_steps": row["eval_rollout_steps"],
                "update_combo_id": row["update_combo_id"],
                "output_root": row["output_root"],
            }
        )
    return rows


def _write_per_job_state(job_rows: list[dict[str, Any]]) -> None:
    by_job: dict[str, list[dict[str, Any]]] = {}
    for row in job_rows:
        by_job.setdefault(str(row["job_id"]), []).append(row)
    for rows in by_job.values():
        job_root = Path(str(rows[0]["output_root"])).parents[0]
        _write_jsonl(job_root / JOB_STATE_LOCAL_FILE, rows)
        aggregate = [row for row in rows if row["phase"] == "aggregate"]
        _write_json(
            job_root / JOB_SUMMARY_FILE,
            {
                "schema_version": "xunce-stage26-8m-job-summary/v1",
                "job_id": rows[0]["job_id"],
                "status": "complete" if aggregate and aggregate[0]["status"] == "complete" else ("failed" if any(row["status"] == "failed" for row in rows) else "pending"),
                "next_phase": next((row["phase"] for row in rows if row["status"] == "pending"), None),
            },
        )


def _existing_state_by_key(output_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = _read_jsonl_if_exists(output_root / JOB_STATE_FILE)
    return {(str(row.get("job_id")), str(row.get("phase"))): row for row in rows if row.get("job_id") and row.get("phase")}


def _job_config_hash(config: dict[str, Any], job: dict[str, Any]) -> str:
    payload = {
        "job": {key: job[key] for key in ("horizon", "seed", "scenario_count", "collector_rollout_steps", "eval_rollout_steps", "update_combo_id")},
        "combo": job["update_combo"],
        "contracts": {
            key: config[key]
            for key in (
                "coverage_denominator_source",
                "post_update_success_metric",
                "coverage_source",
                "path_cost_source",
                "synthetic_source_kind",
                "action_space_type",
                "hybrid_astar_candidate_eval_workers",
                "max_traversable_slope_deg",
                "max_abs_approx_kl",
            )
        },
    }
    return _stable_hash(payload)


def _experiment_config_hash(config: dict[str, Any]) -> str:
    return _stable_hash(
        {
            key: config[key]
            for key in (
                "horizons",
                "seeds",
                "scenario_counts",
                "collector_rollout_steps",
                "eval_rollout_steps",
                "update_combos",
                "max_abs_approx_kl",
            )
        }
    )


def _input_hash(config: dict[str, Any]) -> str:
    return _stable_hash({"source_scenario_fixture_root": config.get("source_scenario_fixture_root")})


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _ensure_summary(path: Path, payload: dict[str, Any]) -> None:
    if payload and not path.is_file():
        _write_json(path, payload)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


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


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _positive_float(value: Any, field: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{field} must be a positive finite number")
    return parsed


def _nonnegative_float(value: Any, field: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{field} must be a nonnegative finite number")
    return parsed


def _positive_int_list(value: Any, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty array")
    return [_positive_int(item, field) for item in value]


def _jsonl_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())


def _mean(values: list[Any]) -> float:
    parsed = [_float(value) for value in values]
    return sum(parsed) / len(parsed) if parsed else 0.0


def _float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _first_float(*values: Any) -> float:
    for value in values:
        if value is not None:
            return _float(value)
    return 0.0


def _coverage_per_100m_delta(summary: dict[str, Any]) -> float:
    return _first_float(summary.get("main_coverage_per_100m_delta"), summary.get("coverage_per_100m_delta"))


def _binding_or_safety_failure(summary: dict[str, Any]) -> bool:
    if not summary:
        return False
    return any(int(summary.get(field) or 0) != 0 for field in COUNT_FIELDS_REQUIRING_ZERO)


def _aggregate_execution_failure(summary: dict[str, Any]) -> bool:
    if not summary:
        return False
    return (
        summary.get("stage21_5_runtime_blocker") is True
        or summary.get("stage21_5_execution_incomplete") is True
        or summary.get("next_required_change") in {"rerun_stage26_3_required_inputs", "repair_stage26_8h_eval_binding_or_safety"}
    )


def _lineage_mismatch(summary: dict[str, Any]) -> bool:
    if not summary:
        return False
    return (
        summary.get("coverage_denominator_source") != stage26_8.COVERAGE_DENOMINATOR_SOURCE
        or summary.get("post_update_success_metric") != stage26_8.SUCCESS_METRIC
        or summary.get("coverage_source") != stage26_8.COVERAGE_SOURCE
        or summary.get("path_cost_source") != stage26_8.PATH_COST_SOURCE
        or summary.get("synthetic_source_kind") != stage26_8.SYNTHETIC_SOURCE_KIND
        or summary.get("action_space_type") != stage26_8.ACTION_SPACE_TYPE
        or abs(_float(summary.get("max_traversable_slope_deg")) - 30.0) > 1.0e-9
    )


def _policy_delta_too_small(summary: dict[str, Any]) -> bool:
    if not summary:
        return False
    return int(summary.get("selected_action_changed_count") or 0) == 0


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _render_report(summary: dict[str, Any], job_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage26.8M Generalized Resumable Training Pipeline",
        "",
        f"- status: `{summary.get('status')}`",
        f"- next_required_change: `{summary.get('next_required_change')}`",
        f"- job_count: `{summary.get('job_count')}`",
        f"- completed_job_count: `{summary.get('completed_job_count')}`",
        f"- pending_job_count: `{summary.get('pending_job_count')}`",
        f"- next_job_id: `{summary.get('next_job_id')}`",
        f"- next_phase: `{summary.get('next_phase')}`",
        "",
        "State is artifact-driven and lives under this D-drive output root. This stage only orchestrates existing collector/update/eval stages.",
    ]
    return "\n".join(lines) + "\n"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
