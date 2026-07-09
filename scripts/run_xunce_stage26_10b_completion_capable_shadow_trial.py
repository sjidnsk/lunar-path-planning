from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import run_xunce_stage26_8m_generalized_resumable_training_pipeline as stage26_8m
    import run_coverage_memory_replanning_loop as coverage_memory_loop
    import xunce_artifact_io as artifact_io
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage26_8m_generalized_resumable_training_pipeline as stage26_8m
    import scripts.run_coverage_memory_replanning_loop as coverage_memory_loop
    import scripts.xunce_artifact_io as artifact_io


STAGE_ID = "xunce-stage26-10b-completion-capable-shadow-trial"
CONFIG_SCHEMA_VERSION = "xunce-stage26-10b-completion-capable-shadow-trial-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-10b-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-10b-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-10b-manifest/v1"
MATRIX_ROW_SCHEMA_VERSION = "xunce-stage26-10b-matrix-row/v1"
JOB_STATE_SCHEMA_VERSION = "xunce-stage26-10b-job-state/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_10b_completion_capable_shadow_trial_v1.json"
DEFAULT_OUTPUT_ROOT = "D:/xunce/out/s26_10b"
SUMMARY_FILE = "xunce-stage26-10b-summary.json"
ROUTING_FILE = "xunce-stage26-10b-routing.json"
MANIFEST_FILE = "xunce-stage26-10b-manifest.json"
MATRIX_FILE = "xunce-stage26-10b-matrix.jsonl"
JOB_STATE_FILE = "xunce-stage26-10b-job-state.jsonl"
REPORT_FILE = "xunce-stage26-10b-report.md"

ROUTE_BOUNDARY = "resolve_stage26_10b_boundary_rejections"
ROUTE_INPUTS = "repair_stage26_10b_required_inputs"
ROUTE_REACHABILITY = "repair_stage26_10b_completion_reachability"
ROUTE_TERMINAL_CREDIT = "repair_stage26_terminal_credit_assignment"
ROUTE_SHORT_SIGHTED = "repair_stage26_10b_short_sighted_efficiency"
ROUTE_METRIC = "repair_stage26_10b_completion_metric_binding"
ROUTE_EVAL = "repair_stage26_10b_eval_binding_or_safety"
ROUTE_REVIEW = "review_stage26_10b_completion_shadow_readiness"
ROUTE_CONTINUE = "continue_stage26_10b_completion_shadow_trial"

SOURCE_POLICY_CHECKPOINT_PATH = (
    "outputs/path_feedback_batch_xunce_sandbox_candidate_preflight_v1/"
    "sandbox_package/xunce-controlled-training-candidate.pt"
)
SOURCE_POLICY_SHA256 = "366bc8f004879813a3500205ac048e4db73a77a68a7e326862098641880ecf3b"
SOURCE_POLICY_IS_RANDOM_UNTRAINED = False
SOURCE_POLICY_NETWORK_ARCHITECTURE = "xunce_full_network_v1"

BOUNDARY_FIELDS = (
    "release_or_training_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)
HARD_BOUNDARY_FIELDS = (
    "modifies_ppo_loss",
    "modifies_network",
    "modifies_action_space",
    "modifies_hybrid_astar_pose_gate",
    "modifies_candidate_generation",
)
STAGE26_10A_VARIANTS = ("balanced_high", "terminal_heavy", "dead_end_high")
COMPARISON_PROFILES = ("terminal_off_control", "balanced_high", "baseline_v3", "terminal_heavy", "dead_end_high")
STAGE21_3_BATCH_FILE = "xunce-stage21-3-ppo-trainable-batch.jsonl"
STAGE21_3_RETURN_AUDIT_FILE = "xunce-stage21-3-return-advantage-audit.jsonl"
TERMINAL_REWARD_COMPONENTS = (
    "success_99pct_bonus_component",
    "incomplete_terminal_penalty_component",
    "dead_end_penalty_component",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.10B completion-capable shadow trial orchestration.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_10b_completion_capable_shadow_trial(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_10b_completion_capable_shadow_trial(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root)
    output_root = _resolve_path(output_root, repo_root)
    artifact_io.make_dirs(output_root)
    profiles_dir = output_root / "profiles"
    configs_dir = output_root / "stage26_8m_configs"
    jobs_dir = output_root / "stage26_8m_jobs"
    artifact_io.make_dirs(profiles_dir)
    artifact_io.make_dirs(configs_dir)
    artifact_io.make_dirs(jobs_dir)
    config["_output_root_hint"] = str(output_root)

    boundary_rejections = _boundary_rejections(config)
    profile_rows, profile_hashes = _materialize_profiles(config, profiles_dir, repo_root)
    input_rejections = [] if boundary_rejections else _input_rejections(config, repo_root, profile_rows)
    oracle_summary = (
        {"completion_reachable": False, "skipped": True, "skip_reason": "boundary_or_input_rejections"}
        if boundary_rejections or input_rejections
        else _load_l0_oracle_summary(config, repo_root)
    )
    terminal_credit_audit = _load_terminal_credit_audit(config, repo_root)
    matrix_rows = _load_existing_matrix(output_root)

    if not boundary_rejections and not input_rejections and _oracle_reachable(oracle_summary):
        _materialize_stage26_8m_configs(config, profile_rows, configs_dir, repo_root)
        matrix_rows = _ensure_matrix_rows(matrix_rows, profile_rows)
        route_before_run = _route(boundary_rejections, input_rejections, oracle_summary, matrix_rows)
        if route_before_run == ROUTE_CONTINUE:
            _run_one_pending_job(config, matrix_rows, configs_dir, jobs_dir, repo_root, terminal_credit_audit)

    matrix_rows = _apply_terminal_credit_audit(matrix_rows, terminal_credit_audit)
    route = _route(boundary_rejections, input_rejections, oracle_summary, matrix_rows)
    status = "passed" if route in {ROUTE_CONTINUE, ROUTE_REVIEW} else "failed"
    summary = _summary(
        config=config,
        output_root=output_root,
        status=status,
        route=route,
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        oracle_summary=oracle_summary,
        matrix_rows=matrix_rows,
        profile_hashes=profile_hashes,
    )
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
        **_boundary_false_payload(),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "summary_status": status,
        "next_required_change": route,
        "source_policy": _source_policy_identity(config),
        "artifacts": {
            "summary": str(output_root / SUMMARY_FILE),
            "routing": str(output_root / ROUTING_FILE),
            "manifest": str(output_root / MANIFEST_FILE),
            "matrix": str(output_root / MATRIX_FILE),
            "job_state": str(output_root / JOB_STATE_FILE),
            "report": str(output_root / REPORT_FILE),
            "profiles": str(profiles_dir),
            "stage26_8m_configs": str(configs_dir),
            "stage26_8m_jobs": str(jobs_dir),
        },
        **_boundary_false_payload(),
        **_hard_boundary_false_payload(),
    }
    _write_jsonl(output_root / MATRIX_FILE, matrix_rows)
    _write_jsonl(output_root / JOB_STATE_FILE, _job_state_rows(matrix_rows))
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    artifact_io.write_text(output_root / REPORT_FILE, _render_report(summary))
    return summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    if not artifact_io.path_is_file(path):
        raise FileNotFoundError(f"Stage26.10B config not found: {path}")
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    if payload.get("stage_id") != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID}")
    defaults = {
        "default_output_root": DEFAULT_OUTPUT_ROOT,
        "base_stage26_8m_config": "configs/xunce_stage26_8m_generalized_resumable_training_pipeline_v1.json",
        "baseline_reward_profile": "configs/xunce_stage26_10_terminal_aware_ppo_reward_profile_v3.json",
        "stage26_10a_root": "D:/xunce/out/s26_10a",
        "source_oracle_summary": "",
        "l0_fixture_catalog": "",
        "l0_oracle_proxy_scenario_count": 11,
        "l0_target_coverage_rate": 0.99,
        "l0_path_budget_m": 5000.0,
        "l0_coverage_radius_cells": 30,
        "l0_replanning_cycle_limit": 1024,
        "l0_segment_step_limit": 64,
        "terminal_credit_audit": "",
        "source_policy_checkpoint_path": SOURCE_POLICY_CHECKPOINT_PATH,
        "source_policy_sha256": SOURCE_POLICY_SHA256,
        "source_policy_is_random_untrained": SOURCE_POLICY_IS_RANDOM_UNTRAINED,
        "source_policy_network_architecture": SOURCE_POLICY_NETWORK_ARCHITECTURE,
        "horizons": [1024],
        "seeds": [260801, 260802, 260803],
        "scenario_counts": [11],
        "collector_rollout_steps": [1024],
        "eval_rollout_steps": [1024],
        "update_combos": [
            {
                "combo_id": "depth24_lr3e5_policy3",
                "epochs": 24,
                "learning_rate": 0.00003,
                "policy_loss_coefficient": 3.0,
                "value_loss_coefficient": 0.01,
                "entropy_coefficient": 0.003,
                "loss_scale": 0.5,
            },
            {
                "combo_id": "lr5e5_policy3",
                "epochs": 16,
                "learning_rate": 0.00005,
                "policy_loss_coefficient": 3.0,
                "value_loss_coefficient": 0.01,
                "entropy_coefficient": 0.003,
                "loss_scale": 0.5,
            },
        ],
        "coverage_denominator_source": "main_coverable_cells/v1",
        "post_update_success_metric": "main_coverable_coverage_efficiency/v1",
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
        "hybrid_astar_candidate_eval_workers": 4,
        "candidate_reachability_gate_source": "hybrid_astar_pose_reachability/v1",
        "candidate_reachability_max_theta_proposals_per_candidate": 5,
        "candidate_reachability_theta_proposal_policy": "candidate_current_bearing_sweep/v1",
        "hybrid_astar_planning_grid_source": "derived_high_res_planning_proxy/v1",
        "planner_grid_resolution_m": 1.0,
        "hybrid_astar_max_iterations": 200,
        "max_jobs_per_invocation": 1,
        **_boundary_false_payload(),
        **_hard_boundary_false_payload(),
    }
    config = defaults | payload
    for field in BOUNDARY_FIELDS:
        config[field] = bool(config.get(field, False))
    for field in HARD_BOUNDARY_FIELDS:
        config[field] = bool(config.get(field, False))
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0))
    config["max_jobs_per_invocation"] = max(1, int(config.get("max_jobs_per_invocation") or 1))
    for field in (
        "base_stage26_8m_config",
        "baseline_reward_profile",
        "stage26_10a_root",
        "source_oracle_summary",
        "l0_fixture_catalog",
        "terminal_credit_audit",
    ):
        config[field] = str(config.get(field) or "")
    config["l0_oracle_proxy_scenario_count"] = max(1, int(config.get("l0_oracle_proxy_scenario_count") or 1))
    return config


def _materialize_profiles(config: dict[str, Any], profiles_dir: Path, repo_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    baseline_path = _resolve_path(Path(str(config["baseline_reward_profile"])), repo_root)
    rows: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    if not artifact_io.path_is_file(baseline_path):
        return rows, hashes
    baseline = _read_json(baseline_path)

    baseline_out = profiles_dir / "baseline_v3.json"
    _write_json(baseline_out, baseline)
    rows["baseline_v3"] = {"profile_id": "baseline_v3", "profile_path": str(baseline_out), "source": "baseline_v3"}
    hashes["baseline_v3"] = _stable_hash(baseline)

    terminal_off = json.loads(json.dumps(baseline))
    terminal_off["profile_id"] = "xunce-stage26-10b-terminal-off-control"
    for field in ("dead_end", "incomplete_terminal", "terminal_final_coverage", "success_99pct"):
        terminal_off["weights"][field] = 0.0
    terminal_off_path = profiles_dir / "terminal_off_control.json"
    _write_json(terminal_off_path, terminal_off)
    rows["terminal_off_control"] = {
        "profile_id": "terminal_off_control",
        "profile_path": str(terminal_off_path),
        "source": "terminal_off_control",
    }
    hashes["terminal_off_control"] = _stable_hash(terminal_off)

    for row in _stage26_10a_profile_rows(config, repo_root):
        variant = str(row.get("variant_id") or "")
        if variant not in STAGE26_10A_VARIANTS:
            continue
        source_path = _resolve_path(Path(str(row.get("profile_path") or "")), repo_root)
        if not artifact_io.path_is_file(source_path):
            continue
        payload = _read_json(source_path)
        out_path = profiles_dir / f"{variant}.json"
        _write_json(out_path, payload)
        rows[variant] = {"profile_id": variant, "profile_path": str(out_path), "source": "stage26_10a"}
        hashes[variant] = _stable_hash(payload)
    return rows, hashes


def _stage26_10a_profile_rows(config: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    root_raw = str(config.get("stage26_10a_root") or "")
    if not root_raw:
        return []
    root = _resolve_path(Path(root_raw), repo_root)
    matrix = root / "xunce-stage26-10a-reward-weight-sweep-matrix.jsonl"
    if not artifact_io.path_is_file(matrix):
        return []
    return _read_jsonl(matrix)


def _materialize_stage26_8m_configs(
    config: dict[str, Any],
    profile_rows: dict[str, dict[str, Any]],
    configs_dir: Path,
    repo_root: Path,
) -> None:
    base = _read_json(_resolve_path(Path(str(config["base_stage26_8m_config"])), repo_root))
    base_stage26_1_path = _resolve_path(Path(str(base["base_stage26_1_config"])), repo_root)
    base_stage26_1 = _read_json(base_stage26_1_path)
    for profile_id in COMPARISON_PROFILES:
        if profile_id not in profile_rows:
            continue
        derived_stage26_1 = dict(base_stage26_1)
        derived_stage26_1["coverage_first_reward_profile"] = str(profile_rows[profile_id]["profile_path"])
        derived_stage26_1.update({**_boundary_false_payload(), **_hard_boundary_false_payload()})
        derived_stage26_1_path = configs_dir / f"{profile_id}-stage26-1-base.json"
        _write_json(derived_stage26_1_path, derived_stage26_1)
        payload = dict(base)
        payload.update(
            {
                "run_mode": "run_next",
                "max_jobs_per_invocation": 1,
                "base_stage26_1_config": str(derived_stage26_1_path),
                "horizons": list(config["horizons"]),
                "seeds": list(config["seeds"]),
                "scenario_counts": list(config["scenario_counts"]),
                "collector_rollout_steps": list(config["collector_rollout_steps"]),
                "eval_rollout_steps": list(config["eval_rollout_steps"]),
                "update_combos": list(config["update_combos"]),
                "coverage_denominator_source": config["coverage_denominator_source"],
                "post_update_success_metric": config["post_update_success_metric"],
                "coverage_source": config["coverage_source"],
                "path_cost_source": config["path_cost_source"],
                "synthetic_source_kind": config["synthetic_source_kind"],
                "action_space_type": config["action_space_type"],
                "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
                "candidate_reachability_gate_source": config["candidate_reachability_gate_source"],
                "candidate_reachability_max_theta_proposals_per_candidate": int(
                    config["candidate_reachability_max_theta_proposals_per_candidate"]
                ),
                "candidate_reachability_theta_proposal_policy": config["candidate_reachability_theta_proposal_policy"],
                "hybrid_astar_planning_grid_source": config["hybrid_astar_planning_grid_source"],
                "planner_grid_resolution_m": float(config["planner_grid_resolution_m"]),
                "hybrid_astar_max_iterations": int(config["hybrid_astar_max_iterations"]),
                "coverage_first_reward_profile": str(profile_rows[profile_id]["profile_path"]),
                "terminal_aware_reward_profile": str(profile_rows[profile_id]["profile_path"]),
                "stage26_10b_profile_id": profile_id,
                "stage26_10b_source_policy": _source_policy_identity(config),
                **_boundary_false_payload(),
                **_hard_boundary_false_payload(),
            }
        )
        _write_json(configs_dir / f"{profile_id}.json", payload)


def _run_one_pending_job(
    config: dict[str, Any],
    matrix_rows: list[dict[str, Any]],
    configs_dir: Path,
    jobs_dir: Path,
    repo_root: Path,
    terminal_credit_audit: dict[str, Any],
) -> None:
    for row in matrix_rows:
        if row.get("status") != "pending" or row.get("level") == "L0":
            continue
        profile_id = str(row["profile_id"])
        config_path = configs_dir / f"{profile_id}.json"
        if not artifact_io.path_is_file(config_path):
            continue
        job_output_root = jobs_dir / profile_id
        summary = _invoke_stage26_8m(config_path=config_path, output_root=job_output_root, repo_root=repo_root)
        job_terminal_credit_audit = _merge_terminal_credit_audits(
            terminal_credit_audit,
            _terminal_credit_audit_from_stage26_8m_output(
                job_output_root,
                expect_terminal_reward_components=profile_id != "terminal_off_control",
            ),
        )
        row.update(_row_from_stage26_8m_summary(row, summary, job_terminal_credit_audit, job_output_root))
        return


def _row_from_stage26_8m_summary(
    row: dict[str, Any],
    summary: dict[str, Any],
    terminal_credit_audit: dict[str, Any],
    job_output_root: Path | None = None,
) -> dict[str, Any]:
    phase_execution_failed = any(row.get("status") == "failed" for row in summary.get("phase_executions", []) if isinstance(row, dict))
    status = "complete"
    if summary.get("status") == "failed" or bool(summary.get("binding_or_safety_failure")) or phase_execution_failed:
        status = "failed"
    elif int(summary.get("pending_job_count") or 0) > 0 or summary.get("next_required_change") == "continue_stage26_8m_jobs":
        status = "pending"
    completion_metrics = _completion_metrics_from_stage26_8m_output(job_output_root) if job_output_root is not None else {}
    return {
        **row,
        "status": status,
        "stage26_8m_status": summary.get("status"),
        "stage26_8m_next_required_change": summary.get("next_required_change"),
        "pending_job_count": int(summary.get("pending_job_count") or 0),
        "completed_job_count": int(summary.get("completed_job_count") or 0),
        "binding_or_safety_failure": bool(summary.get("binding_or_safety_failure") or phase_execution_failed),
        "stage26_8m_phase_execution_failure": phase_execution_failed,
        "terminal_credit_audit_missing": bool(summary.get("terminal_credit_audit_missing") or terminal_credit_audit.get("missing")),
        "terminal_credit_missing_flag_count": int(
            summary.get("terminal_credit_missing_flag_count")
            or terminal_credit_audit.get("terminal_credit_missing_flag_count")
            or 0
        ),
        "main_coverage_per_100m_delta": _float(summary.get("main_coverage_per_100m_delta")),
        "final_coverage_delta": _float(summary.get("final_coverage_delta")),
        "final_coverage_rate_delta": _optional_float(
            summary.get("final_coverage_rate_delta"),
            summary.get("main_final_coverage_delta"),
            summary.get("final_coverage_delta"),
            completion_metrics.get("final_coverage_rate_delta"),
        ),
        "success_99pct_rate": _optional_float(summary.get("success_99pct_rate"), completion_metrics.get("success_99pct_rate")),
        "median_steps_to_99pct": _optional_float(summary.get("median_steps_to_99pct"), completion_metrics.get("median_steps_to_99pct")),
        "median_path_cost_to_99pct": _optional_float(
            summary.get("median_path_cost_to_99pct"),
            completion_metrics.get("median_path_cost_to_99pct"),
        ),
        "worst_seed_final_coverage_rate_delta": _optional_float(
            summary.get("worst_seed_final_coverage_rate_delta"),
            completion_metrics.get("worst_seed_final_coverage_rate_delta"),
        ),
        "scenario_regression_rate": _optional_float(summary.get("scenario_regression_rate"), completion_metrics.get("scenario_regression_rate")),
        "success_99pct_count_delta": int(summary.get("success_99pct_count_delta") or 0),
    }


def _invoke_stage26_8m(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    return stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=config_path,
        output_root=output_root,
        repo_root=repo_root,
    )


def _ensure_matrix_rows(matrix_rows: list[dict[str, Any]], profile_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(matrix_rows)
    seen = {str(row.get("profile_id")) for row in rows}
    if "l0_oracle" not in seen:
        rows.append(_initial_matrix_row("l0_oracle", "L0", "complete"))
        seen.add("l0_oracle")
    for profile_id in COMPARISON_PROFILES:
        if profile_id not in profile_rows or profile_id in seen:
            continue
        rows.append(_initial_matrix_row(profile_id, "L1" if profile_id in {"terminal_off_control", "balanced_high"} else "L2", "pending"))
        seen.add(profile_id)
    return rows


def _initial_matrix_row(profile_id: str, level: str, status: str) -> dict[str, Any]:
    return {
        "schema_version": MATRIX_ROW_SCHEMA_VERSION,
        "profile_id": profile_id,
        "level": level,
        "status": status,
        "terminal_credit_audit_missing": False,
        "terminal_credit_missing_flag_count": 0,
        "binding_or_safety_failure": False,
        "main_coverage_per_100m_delta": 0.0,
        "final_coverage_delta": 0.0,
        "success_99pct_count_delta": 0,
    }


def _route(
    boundary_rejections: list[str],
    input_rejections: list[str],
    oracle_summary: dict[str, Any],
    matrix_rows: list[dict[str, Any]],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections:
        return ROUTE_INPUTS
    if not _oracle_reachable(oracle_summary):
        return ROUTE_REACHABILITY
    if any(row.get("binding_or_safety_failure") for row in matrix_rows):
        return ROUTE_EVAL
    if any(row.get("status") == "failed" for row in matrix_rows):
        return ROUTE_EVAL
    if any(row.get("terminal_credit_audit_missing") for row in matrix_rows):
        return ROUTE_TERMINAL_CREDIT
    if any(int(row.get("terminal_credit_missing_flag_count") or 0) != 0 for row in matrix_rows):
        return ROUTE_TERMINAL_CREDIT
    completed = [row for row in matrix_rows if row.get("status") == "complete"]
    if any(
        _float(row.get("main_coverage_per_100m_delta")) > 0.0
        and _float(row.get("final_coverage_delta")) <= 0.0
        and int(row.get("success_99pct_count_delta") or 0) <= 0
        for row in completed
    ):
        return ROUTE_SHORT_SIGHTED
    if any(row.get("completion_metric_binding_failure") for row in matrix_rows):
        return ROUTE_METRIC
    if not _required_profile_rows_complete(matrix_rows):
        return ROUTE_CONTINUE
    required_rows = [row for row in matrix_rows if row.get("profile_id") in COMPARISON_PROFILES]
    if not _completion_metrics_bound(required_rows):
        return ROUTE_METRIC
    if _short_sighted_completion_pattern(required_rows):
        return ROUTE_SHORT_SIGHTED
    if _readiness_gate_passed(required_rows):
        return ROUTE_REVIEW
    return ROUTE_TERMINAL_CREDIT


def _required_profile_rows_complete(matrix_rows: list[dict[str, Any]]) -> bool:
    by_profile = {str(row.get("profile_id")): row for row in matrix_rows}
    return all(by_profile.get(profile_id, {}).get("status") == "complete" for profile_id in COMPARISON_PROFILES)


def _completion_metrics_bound(rows: list[dict[str, Any]]) -> bool:
    required_fields = (
        "success_99pct_rate",
        "median_steps_to_99pct",
        "median_path_cost_to_99pct",
        "final_coverage_rate_delta",
        "worst_seed_final_coverage_rate_delta",
        "scenario_regression_rate",
    )
    return all(_float_or_none(row.get(field)) is not None for row in rows for field in required_fields)


def _short_sighted_completion_pattern(rows: list[dict[str, Any]]) -> bool:
    control = next((row for row in rows if row.get("profile_id") == "terminal_off_control"), None)
    if not control:
        return False
    control_success = _float(control.get("success_99pct_rate"))
    for row in rows:
        if row.get("profile_id") == "terminal_off_control":
            continue
        success_delta = _float(row.get("success_99pct_rate")) - control_success
        final_delta = _float(row.get("final_coverage_rate_delta"))
        if _float(row.get("main_coverage_per_100m_delta")) > 0.0 and success_delta <= 0.0 and final_delta <= 0.0:
            return True
    return False


def _readiness_gate_passed(rows: list[dict[str, Any]]) -> bool:
    control = next((row for row in rows if row.get("profile_id") == "terminal_off_control"), None)
    if not control:
        return False
    control_success = _float(control.get("success_99pct_rate"))
    control_steps = _float(control.get("median_steps_to_99pct"))
    control_path = _float(control.get("median_path_cost_to_99pct"))
    for row in rows:
        if row.get("profile_id") == "terminal_off_control":
            continue
        if _float(row.get("final_coverage_rate_delta")) < 0.0:
            continue
        if _float(row.get("worst_seed_final_coverage_rate_delta")) < -0.01:
            continue
        if _float(row.get("scenario_regression_rate")) > 0.25:
            continue
        success_delta = _float(row.get("success_99pct_rate")) - control_success
        if success_delta >= 0.10:
            return True
        if control_success >= 0.80 and control_steps > 0.0:
            steps_improvement = (control_steps - _float(row.get("median_steps_to_99pct"))) / control_steps
            path_worse = control_path > 0.0 and _float(row.get("median_path_cost_to_99pct")) > control_path * 1.10
            if steps_improvement >= 0.05 and not path_worse:
                return True
    return False


def _summary(
    *,
    config: dict[str, Any],
    output_root: Path,
    status: str,
    route: str,
    boundary_rejections: list[str],
    input_rejections: list[str],
    oracle_summary: dict[str, Any],
    matrix_rows: list[dict[str, Any]],
    profile_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
        "completion_reachable": _oracle_reachable(oracle_summary),
        "profile_hashes": profile_hashes,
        "matrix_row_count": len(matrix_rows),
        "summary": str(output_root / SUMMARY_FILE),
        "routing": str(output_root / ROUTING_FILE),
        "manifest": str(output_root / MANIFEST_FILE),
        "matrix": str(output_root / MATRIX_FILE),
        "job_state": str(output_root / JOB_STATE_FILE),
        "report": str(output_root / REPORT_FILE),
        "source_policy": _source_policy_identity(config),
        **_boundary_false_payload(),
        **_hard_boundary_false_payload(),
    }


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    reasons.extend(field for field in HARD_BOUNDARY_FIELDS if config.get(field) is True)
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _input_rejections(config: dict[str, Any], repo_root: Path, profile_rows: dict[str, dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for field in ("base_stage26_8m_config", "baseline_reward_profile"):
        if not artifact_io.path_is_file(_resolve_path(Path(str(config[field])), repo_root)):
            reasons.append(f"missing_{field}")
    if str(config.get("source_oracle_summary") or "") and not artifact_io.path_is_file(
        _resolve_path(Path(str(config["source_oracle_summary"])), repo_root)
    ):
        reasons.append("missing_source_oracle_summary")
    if not str(config.get("source_oracle_summary") or ""):
        l0_catalog = _l0_fixture_catalog_path(config, repo_root)
        if not artifact_io.path_is_file(l0_catalog):
            reasons.append("missing_l0_fixture_catalog")
    for variant in STAGE26_10A_VARIANTS:
        if variant not in profile_rows:
            reasons.append(f"missing_stage26_10a_profile_{variant}")
    return reasons


def _load_l0_oracle_summary(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    path_text = str(config.get("source_oracle_summary") or "")
    if not path_text:
        return _run_l0_oracle_proxy(config, repo_root, Path(str(config.get("_output_root_hint") or DEFAULT_OUTPUT_ROOT)) / "l0_oracle_proxy")
    path = _resolve_path(Path(path_text), repo_root)
    if not artifact_io.path_is_file(path):
        return {"completion_reachable": False, "missing": True, "path": str(path)}
    return _read_json(path)


def _run_l0_oracle_proxy(config: dict[str, Any], repo_root: Path, output_root: Path) -> dict[str, Any]:
    artifact_io.make_dirs(output_root)
    l0_summary_path = output_root / "l0-oracle-summary.json"
    if artifact_io.path_is_file(l0_summary_path):
        return _read_json(l0_summary_path)
    target_coverage_rate = float(config.get("l0_target_coverage_rate") or 0.99)
    path_budget_m = float(config.get("l0_path_budget_m") or 5000.0)
    coverage_radius_cells = int(config.get("l0_coverage_radius_cells") or 30)
    replanning_cycle_limit = int(config.get("l0_replanning_cycle_limit") or 1024)
    segment_step_limit = int(config.get("l0_segment_step_limit") or 64)
    fixture_rows = _read_jsonl(_l0_fixture_catalog_path(config, repo_root))[: int(config.get("l0_oracle_proxy_scenario_count") or 11)]
    scenarios = [_l0_scenario_from_fixture(row, repo_root) for row in fixture_rows]
    source_config = {
        "schema_version": "global-99-coverage-benchmark-config/v1",
        "target_coverage_rate": target_coverage_rate,
        "path_budget_m": path_budget_m,
        "scenarios": scenarios,
    }
    source_config_path = output_root / "l0-global-99-source-config.json"
    frontier_config_path = output_root / "l0-frontier-baseline-config.json"
    memory_config_path = output_root / "l0-coverage-memory-config.json"
    memory_output_root = output_root / "coverage_memory"
    _write_json(source_config_path, source_config)
    _write_json(
        frontier_config_path,
        {
            "schema_version": "frontier-coverage-planner-baseline-config/v1",
            "source_global_99_config": str(source_config_path),
            "target_coverage_rate": target_coverage_rate,
            "path_budget_m": path_budget_m,
            "coverage_radius_cells": coverage_radius_cells,
            "frontier_step_limit": replanning_cycle_limit,
            "revisit_penalty_weight": 1.0,
            "new_coverage_weight": 4.0,
        },
    )
    _write_json(
        memory_config_path,
        {
            "schema_version": "coverage-memory-replanning-loop-config/v1",
            "source_frontier_baseline_config": str(frontier_config_path),
            "source_global_99_config": str(source_config_path),
            "target_coverage_rate": target_coverage_rate,
            "path_budget_m": path_budget_m,
            "coverage_radius_cells": coverage_radius_cells,
            "replanning_cycle_limit": replanning_cycle_limit,
            "segment_step_limit": segment_step_limit,
            "memory_snapshot_interval": 1,
            "resume_from_memory_snapshot": None,
        },
    )
    memory_summary = coverage_memory_loop.run_coverage_memory_replanning_loop(
        config_path=memory_config_path,
        output_root=memory_output_root,
        repo_root=repo_root,
    )
    summary = {
        "completion_reachable": bool(memory_summary.get("coverage_target_met")),
        "source": "coverage_memory_replanning_loop_oracle_proxy/v1",
        "scenario_count": len(scenarios),
        "target_coverage_rate": target_coverage_rate,
        "achieved_coverage_rate": _float(memory_summary.get("achieved_coverage_rate")),
        "planned_path_cost_m": _float(memory_summary.get("planned_path_cost_m")),
        "path_budget_m": _float(memory_summary.get("path_budget_m")),
        "coverage_memory_summary": str(memory_output_root / coverage_memory_loop.SUMMARY_FILE),
        "source_global_99_config": str(source_config_path),
        "frontier_baseline_config": str(frontier_config_path),
        "coverage_memory_config": str(memory_config_path),
    }
    _write_json(l0_summary_path, summary)
    return summary


def _l0_scenario_from_fixture(row: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    sidecar_path = _resolve_path(Path(str(row["source_sidecar"])), repo_root)
    sidecar = _read_json(sidecar_path)
    passable_mask = sidecar.get("passable_mask")
    if not isinstance(passable_mask, list) or not passable_mask or not isinstance(passable_mask[0], list):
        raise ValueError(f"sidecar passable_mask missing or invalid: {sidecar_path}")
    height = len(passable_mask)
    width = len(passable_mask[0])
    blocked_cells = _sidecar_blocked_cells(sidecar, width, height)
    start_cell = row.get("scenario_start_cell")
    return {
        "scenario_id": str(row.get("scenario_id") or row.get("scenario_roi_id") or sidecar_path.stem),
        "grid": {"width": width, "height": height, "resolution_m": _sidecar_resolution_m(sidecar), "origin": [0.0, 0.0]},
        "start_cell": start_cell,
        "roi": {"kind": "rect", "x0": 0, "y0": 0, "x1": width, "y1": height},
        "blocked_rectangles": [{"x0": x, "y0": y, "x1": x + 1, "y1": y + 1} for x, y in sorted(blocked_cells)],
        "unsafe_rectangles": [],
        "coverage_events": [],
        "source_sidecar": str(sidecar_path),
        "l0_proxy_source": "stage26_8g_fixture_sidecar_passable_mask/v1",
    }


def _sidecar_blocked_cells(sidecar: dict[str, Any], width: int, height: int) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    passable_mask = sidecar.get("passable_mask")
    if isinstance(passable_mask, list):
        for y, row in enumerate(passable_mask):
            if not isinstance(row, list):
                continue
            for x, value in enumerate(row):
                if 0 <= x < width and 0 <= y < height and not bool(value):
                    blocked.add((x, y))
    for key in ("blocked_cells", "slope_blocked_cells", "synthetic_hard_obstacle_cells", "synthetic_pit_hard_obstacle_cells"):
        for cell in sidecar.get(key, []) or []:
            if isinstance(cell, list) and len(cell) >= 2:
                x, y = int(cell[0]), int(cell[1])
                if 0 <= x < width and 0 <= y < height:
                    blocked.add((x, y))
    return blocked


def _sidecar_resolution_m(sidecar: dict[str, Any]) -> float:
    metadata = sidecar.get("metadata") if isinstance(sidecar.get("metadata"), dict) else {}
    for value in (sidecar.get("resolution_m"), sidecar.get("resolution"), metadata.get("resolution_m"), metadata.get("resolution")):
        parsed = _float_or_none(value)
        if parsed is not None and parsed > 0.0:
            return parsed
    return 1.0


def _l0_fixture_catalog_path(config: dict[str, Any], repo_root: Path) -> Path:
    explicit = str(config.get("l0_fixture_catalog") or "")
    if explicit:
        return _resolve_path(Path(explicit), repo_root)
    base_path = _resolve_path(Path(str(config.get("base_stage26_8m_config") or "")), repo_root)
    if not artifact_io.path_is_file(base_path):
        return Path("__missing_stage26_8m_base_config__")
    base = _read_json(base_path)
    root = _resolve_path(Path(str(base.get("source_scenario_fixture_root") or "")), repo_root)
    return root / "xunce-stage26-8g-scenario-fixture-catalog.jsonl"


def _load_terminal_credit_audit(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    path_text = str(config.get("terminal_credit_audit") or "")
    if not path_text:
        return {"missing": False, "terminal_credit_missing_flag_count": 0}
    path = _resolve_path(Path(path_text), repo_root)
    if not artifact_io.path_is_file(path):
        return {"missing": True, "terminal_credit_missing_flag_count": 0}
    return _read_json(path)


def _terminal_credit_audit_from_stage26_8m_output(output_root: Path, *, expect_terminal_reward_components: bool = True) -> dict[str, Any]:
    output_root = Path(output_root)
    batch_paths = sorted(output_root.rglob(STAGE21_3_BATCH_FILE)) if artifact_io.path_exists(output_root) else []
    return_audit_paths = sorted(output_root.rglob(STAGE21_3_RETURN_AUDIT_FILE)) if artifact_io.path_exists(output_root) else []
    return_advantage_ids = _terminal_return_advantage_ids(return_audit_paths)
    terminal_rows: list[dict[str, Any]] = []
    trainable_terminal_rows: list[dict[str, Any]] = []
    component_counts = {component: 0 for component in TERMINAL_REWARD_COMPONENTS}
    terminal_return_advantage_row_count = 0

    for batch_path in batch_paths:
        for row in _read_jsonl(batch_path):
            if row.get("done") is not True:
                continue
            terminal_rows.append(row)
            if _row_trainable_for_terminal_credit(row):
                trainable_terminal_rows.append(row)
                if _terminal_row_has_return_advantage(row) and str(row.get("transition_id") or "") in return_advantage_ids:
                    terminal_return_advantage_row_count += 1
            components = row.get("reward_components")
            if not isinstance(components, dict):
                components = row.get("components")
            if not isinstance(components, dict):
                components = {}
            for component in TERMINAL_REWARD_COMPONENTS:
                if _float(components.get(component)) != 0.0:
                    component_counts[component] += 1

    missing_flags = 0
    if not batch_paths:
        missing_flags += 1
    if not return_audit_paths:
        missing_flags += 1
    if not terminal_rows:
        missing_flags += 1
    if terminal_rows and not trainable_terminal_rows:
        missing_flags += 1
    if trainable_terminal_rows and terminal_return_advantage_row_count < len(trainable_terminal_rows):
        missing_flags += 1
    if expect_terminal_reward_components and terminal_rows and sum(component_counts.values()) == 0:
        missing_flags += 1

    return {
        "missing": missing_flags > 0,
        "terminal_credit_missing_flag_count": missing_flags,
        "stage21_3_batch_file_count": len(batch_paths),
        "stage21_3_return_advantage_audit_file_count": len(return_audit_paths),
        "done_terminal_transition_count": len(terminal_rows),
        "terminal_trainable_row_count": len(trainable_terminal_rows),
        "terminal_return_advantage_row_count": terminal_return_advantage_row_count,
        "success_99pct_bonus_nonzero_count": component_counts["success_99pct_bonus_component"],
        "incomplete_terminal_penalty_nonzero_count": component_counts["incomplete_terminal_penalty_component"],
        "dead_end_penalty_nonzero_count": component_counts["dead_end_penalty_component"],
    }


def _row_trainable_for_terminal_credit(row: dict[str, Any]) -> bool:
    return row.get("transition_trainable") is not False and row.get("reward_trainable") is not False and row.get("trainable") is not False


def _terminal_row_has_return_advantage(row: dict[str, Any]) -> bool:
    return _float_or_none(row.get("return")) is not None and _float_or_none(row.get("advantage")) is not None


def _terminal_return_advantage_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        for row in _read_jsonl(path):
            transition_id = str(row.get("transition_id") or "")
            if row.get("done") is True and transition_id and _terminal_row_has_return_advantage(row):
                ids.add(transition_id)
    return ids


def _completion_metrics_from_stage26_8m_output(output_root: Path | None) -> dict[str, Any]:
    if output_root is None or not artifact_io.path_exists(output_root):
        return {}
    summaries = sorted(Path(output_root).rglob("xunce-stage26-3-summary.json"))
    rows = [_read_json(path) for path in summaries if artifact_io.path_is_file(path)]
    if not rows:
        return {}
    success_values = [_float_or_none(row.get("success_99pct_rate")) for row in rows]
    steps_values = [_float_or_none(row.get("median_steps_to_99pct")) for row in rows]
    path_values = [_float_or_none(row.get("median_path_cost_to_99pct")) for row in rows]
    final_deltas = [
        _optional_float(row.get("final_coverage_rate_delta"), row.get("main_final_coverage_delta"), row.get("final_coverage_delta"))
        for row in rows
    ]
    scenario_regressions = [_float_or_none(row.get("scenario_regression_count")) for row in rows]
    scenario_counts = [_float_or_none(row.get("required_scenario_count")) or _float_or_none(row.get("scenario_count")) for row in rows]
    return {
        "success_99pct_rate": _mean_present(success_values),
        "median_steps_to_99pct": _median_present(steps_values),
        "median_path_cost_to_99pct": _median_present(path_values),
        "final_coverage_rate_delta": _mean_present(final_deltas),
        "worst_seed_final_coverage_rate_delta": min([value for value in final_deltas if value is not None], default=None),
        "scenario_regression_rate": _scenario_regression_rate(scenario_regressions, scenario_counts),
    }


def _merge_terminal_credit_audits(*audits: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {"missing": False, "terminal_credit_missing_flag_count": 0}
    for audit in audits:
        if not audit:
            continue
        merged["missing"] = bool(merged.get("missing") or audit.get("missing"))
        merged["terminal_credit_missing_flag_count"] = int(merged.get("terminal_credit_missing_flag_count") or 0) + int(
            audit.get("terminal_credit_missing_flag_count") or 0
        )
        for key, value in audit.items():
            if key in {"missing", "terminal_credit_missing_flag_count"}:
                continue
            if isinstance(value, (int, float)):
                merged[key] = int(merged.get(key) or 0) + int(value)
            elif key not in merged:
                merged[key] = value
    return merged


def _apply_terminal_credit_audit(rows: list[dict[str, Any]], audit: dict[str, Any]) -> list[dict[str, Any]]:
    if not rows or (not audit.get("missing") and int(audit.get("terminal_credit_missing_flag_count") or 0) == 0):
        return rows
    return [
        {
            **row,
            "terminal_credit_audit_missing": bool(row.get("terminal_credit_audit_missing") or audit.get("missing")),
            "terminal_credit_missing_flag_count": int(row.get("terminal_credit_missing_flag_count") or 0)
            + int(audit.get("terminal_credit_missing_flag_count") or 0),
        }
        for row in rows
    ]


def _load_existing_matrix(output_root: Path) -> list[dict[str, Any]]:
    path = output_root / MATRIX_FILE
    return _read_jsonl(path) if artifact_io.path_is_file(path) else []


def _job_state_rows(matrix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": JOB_STATE_SCHEMA_VERSION,
            "profile_id": row.get("profile_id"),
            "level": row.get("level"),
            "status": row.get("status"),
            "next_required_change": row.get("stage26_8m_next_required_change"),
        }
        for row in matrix_rows
    ]


def _source_policy_identity(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_path": str(config.get("source_policy_checkpoint_path") or SOURCE_POLICY_CHECKPOINT_PATH),
        "sha256": str(config.get("source_policy_sha256") or SOURCE_POLICY_SHA256),
        "is_random_untrained": bool(config.get("source_policy_is_random_untrained", False)),
        "network_architecture": str(config.get("source_policy_network_architecture") or SOURCE_POLICY_NETWORK_ARCHITECTURE),
    }


def _oracle_reachable(oracle_summary: dict[str, Any]) -> bool:
    return bool(oracle_summary.get("completion_reachable"))


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.10B Completion-Capable Shadow Trial",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- completion_reachable: `{summary['completion_reachable']}`",
            f"- matrix_row_count: `{summary['matrix_row_count']}`",
            "- publishes_checkpoint: `False`",
            "- replaces_default_policy: `False`",
            "- connects_real_executor: `False`",
            "- starts_online_canary: `False`",
            "",
        ]
    )


def _boundary_false_payload() -> dict[str, Any]:
    return {
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _hard_boundary_false_payload() -> dict[str, Any]:
    return {field: False for field in HARD_BOUNDARY_FIELDS}


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(*values: Any) -> float | None:
    for value in values:
        parsed = _float_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def _mean_present(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return (sum(present) / len(present)) if present else None


def _median_present(values: list[float | None]) -> float | None:
    present = sorted(value for value in values if value is not None)
    if not present:
        return None
    midpoint = len(present) // 2
    if len(present) % 2 == 1:
        return present[midpoint]
    return (present[midpoint - 1] + present[midpoint]) / 2.0


def _scenario_regression_rate(regressions: list[float | None], counts: list[float | None]) -> float | None:
    present_regressions = [value for value in regressions if value is not None]
    if not present_regressions:
        return None
    total_regressions = sum(present_regressions)
    present_counts = [value for value in counts if value is not None and value > 0.0]
    denominator = sum(present_counts)
    return (total_regressions / denominator) if denominator > 0.0 else 0.0


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return artifact_io.read_json(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return artifact_io.read_jsonl(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    artifact_io.write_json(path, payload)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    artifact_io.write_jsonl(path, rows)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
