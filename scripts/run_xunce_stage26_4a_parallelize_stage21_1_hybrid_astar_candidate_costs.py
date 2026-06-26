from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import run_xunce_stage26_1_synthetic_terrain_collector_smoke as stage26_1
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage26_1_synthetic_terrain_collector_smoke as stage26_1

stage21_1 = stage26_1.stage21_1


STAGE_ID = "xunce-stage26-4a-parallelize-stage21-1-hybrid-astar-candidate-costs"
CONFIG_SCHEMA_VERSION = "xunce-stage26-4a-parallelize-stage21-1-hybrid-astar-candidate-costs-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-4a-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-4a-manifest/v1"
DEFAULT_CONFIG = "configs/xunce_stage26_4a_parallelize_stage21_1_hybrid_astar_candidate_costs_v1.json"
DEFAULT_STAGE26_0_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract_v1"
)
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_4a_parallelize_stage21_1_hybrid_astar_candidate_costs_v1"
)

SUMMARY_FILE = "xunce-stage26-4a-summary.json"
RUNTIME_FILE = "xunce-stage26-4a-collector-runtime-comparison.json"
EQUIVALENCE_FILE = "xunce-stage26-4a-parallel-equivalence-audit.json"
WORKER_FILE = "xunce-stage26-4a-hybrid-astar-worker-audit.json"
ROUTING_FILE = "xunce-stage26-4a-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-4a-report.md"
MANIFEST_FILE = "xunce-stage26-4a-manifest.json"

SYNTHETIC_MODEL_ID = "synthetic_rock_pit_terrain/v1"
SYNTHETIC_SOURCE_KIND = "synthetic_terrain_obstacle_proxy/v1"
COVERAGE_SOURCE = "endpoint_theta_slope_obstacle_los/v1"
PATH_COST_SOURCE = "hybrid_astar_pose_path/v1"
ROUTE_STAGE26_4 = "rerun_stage26_4_synthetic_policy_update_signal_strength_with_parallel_collector"
BOUNDARY_FIELDS = ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.4A Stage21.1 Hybrid A* candidate cost parallelization audit.")
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_4a_parallelize_stage21_1_hybrid_astar_candidate_costs(
        config_path=args.config,
        output_root=args.output_root,
        repo_root=args.repo_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "passed" else 1


def run_xunce_stage26_4a_parallelize_stage21_1_hybrid_astar_candidate_costs(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path, repo_root)
    generated_at = datetime.now(timezone.utc).isoformat()

    boundary_reasons = _boundary_rejections(config)
    input_reasons, stage26_0_summary = _input_rejections(config)
    serial_run: dict[str, Any] = {}
    parallel_run: dict[str, Any] = {}
    if not boundary_reasons and not input_reasons:
        serial_run = _run_variant(
            config,
            output_root=output_root,
            repo_root=repo_root,
            label="serial_workers_1",
            worker_count=int(config["serial_hybrid_astar_candidate_eval_workers"]),
        )
        parallel_run = _run_variant(
            config,
            output_root=output_root,
            repo_root=repo_root,
            label="parallel_workers_4",
            worker_count=int(config["parallel_hybrid_astar_candidate_eval_workers"]),
        )

    serial_rows = _read_variant_rows(serial_run)
    parallel_rows = _read_variant_rows(parallel_run)
    runtime_audit = _runtime_audit(serial_run, parallel_run, serial_rows, parallel_rows)
    equivalence_audit = _equivalence_audit(serial_rows, parallel_rows)
    worker_audit = _worker_audit(serial_rows, parallel_rows)
    binding_audit = _candidate_binding_audit(serial_rows + parallel_rows)
    status, route, reason = _route(
        boundary_reasons=boundary_reasons,
        input_reasons=input_reasons,
        serial_run=serial_run,
        parallel_run=parallel_run,
        equivalence_audit=equivalence_audit,
        worker_audit=worker_audit,
        binding_audit=binding_audit,
    )
    blocking = _unique(boundary_reasons + input_reasons + _route_blockers(route, serial_run, parallel_run, equivalence_audit, worker_audit, binding_audit))
    release_clean = not boundary_reasons and _summary_boundary_clean(serial_run.get("summary", {})) and _summary_boundary_clean(parallel_run.get("summary", {}))
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "generated_at": generated_at,
        "status": status,
        "next_required_change": route,
        "reason": reason,
        "blocking_reason_codes": blocking,
        "stage26_0_root": config["stage26_0_root"],
        "stage26_0_status": stage26_0_summary.get("status"),
        "synthetic_terrain_hash": stage26_0_summary.get("synthetic_terrain_hash"),
        "synthetic_source_kind": SYNTHETIC_SOURCE_KIND,
        "coverage_source": COVERAGE_SOURCE,
        "path_cost_source": PATH_COST_SOURCE,
        "max_traversable_slope_deg": 30.0,
        "serial_worker_count": int(config["serial_hybrid_astar_candidate_eval_workers"]),
        "parallel_worker_count": int(config["parallel_hybrid_astar_candidate_eval_workers"]),
        "serial_stage26_1_status": serial_run.get("summary", {}).get("status"),
        "parallel_stage26_1_status": parallel_run.get("summary", {}).get("status"),
        "serial_transition_count": len(serial_rows),
        "parallel_transition_count": len(parallel_rows),
        "parallel_equivalence_mismatch_count": int(equivalence_audit["mismatch_count"]),
        "candidate_path_cost_binding_mismatch_count": int(binding_audit["mismatch_count"]),
        "worker_audit_missing_count": int(worker_audit["audit_missing_count"]),
        "parallel_worker_enabled_transition_count": int(worker_audit["parallel_enabled_count"]),
        "hybrid_astar_candidate_eval_failed_count": int(worker_audit["failed_count"]),
        "runtime_wall_clock_speedup_ratio": runtime_audit.get("wall_clock_speedup_ratio"),
        "release_or_boundary_clean": release_clean,
        "stage26_4a_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "summary": str((output_root / SUMMARY_FILE).resolve()),
        "runtime_comparison": str((output_root / RUNTIME_FILE).resolve()),
        "parallel_equivalence_audit": str((output_root / EQUIVALENCE_FILE).resolve()),
        "hybrid_astar_worker_audit": str((output_root / WORKER_FILE).resolve()),
        "routing": str((output_root / ROUTING_FILE).resolve()),
        "report": str((output_root / REPORT_FILE).resolve()),
        "manifest": str((output_root / MANIFEST_FILE).resolve()),
    }
    routing = {
        "schema_version": "xunce-stage26-4a-next-stage-routing/v1",
        "status": status,
        "next_required_change": route,
        "reason": reason,
        "blocking_reason_codes": blocking,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config": str(_resolve_path(config_path, repo_root).resolve()),
        "summary_status": status,
        "next_required_change": route,
        "artifacts": {
            "summary": summary["summary"],
            "runtime_comparison": summary["runtime_comparison"],
            "parallel_equivalence_audit": summary["parallel_equivalence_audit"],
            "hybrid_astar_worker_audit": summary["hybrid_astar_worker_audit"],
            "routing": summary["routing"],
            "report": summary["report"],
        },
        "internal_roots": {
            "serial_workers_1": serial_run.get("output_root"),
            "parallel_workers_4": parallel_run.get("output_root"),
        },
    }
    _write_json(output_root / RUNTIME_FILE, runtime_audit)
    _write_json(output_root / EQUIVALENCE_FILE, equivalence_audit)
    _write_json(output_root / WORKER_FILE, worker_audit | {"candidate_binding_audit": binding_audit})
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary, runtime_audit, equivalence_audit, worker_audit), encoding="utf-8")
    return summary


def _run_variant(config: dict[str, Any], *, output_root: Path, repo_root: Path, label: str, worker_count: int) -> dict[str, Any]:
    run_root = output_root / label / "s26_1"
    cfg = _build_stage26_1_config(config, worker_count=worker_count)
    cfg_path = output_root / label / "xunce-stage26-4a-stage26-1-config.json"
    _write_json(cfg_path, cfg)
    started = time.perf_counter()
    try:
        summary = stage26_1.run_xunce_stage26_1_synthetic_terrain_collector_smoke(
            config_path=cfg_path,
            output_root=run_root,
            repo_root=repo_root,
        )
        error = None
    except Exception as exc:  # pragma: no cover - surfaced in summary and route.
        summary = {"status": "failed", "next_required_change": "repair_stage26_4a_hybrid_astar_worker_contract"}
        error = f"{type(exc).__name__}:{exc}"
    elapsed_s = max(0.0, time.perf_counter() - started)
    return {
        "label": label,
        "worker_count": int(worker_count),
        "output_root": str(run_root),
        "config_path": str(cfg_path),
        "summary": summary,
        "elapsed_s": elapsed_s,
        "error": error,
    }


def _build_stage26_1_config(config: dict[str, Any], *, worker_count: int) -> dict[str, Any]:
    base = _read_json(Path(config["stage26_1_base_config"]))
    base.update(
        {
            "stage26_0_root": config["stage26_0_root"],
            "required_scenario_count": int(config["required_scenario_count"]),
            "rollout_steps": int(config["rollout_steps"]),
            "dynamic_max_candidates_per_step": int(config["dynamic_max_candidates_per_step"]),
            "dynamic_proposal_pool_limit_per_step": int(config["dynamic_proposal_pool_limit_per_step"]),
            "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
            "theta_bin_count": int(config["theta_bin_count"]),
            "theta_step_deg": int(config["theta_step_deg"]),
            "sensor_model_id": str(config["sensor_model_id"]),
            "sensor_fov_deg": float(config["sensor_fov_deg"]),
            "sensor_range_cells": int(config["sensor_range_cells"]),
            "max_traversable_slope_deg": 30.0,
            "hybrid_astar_candidate_eval_workers": int(worker_count),
            "stage26_1_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return base


def _read_variant_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(str(run.get("output_root") or ""))
    return _read_jsonl_if_exists(root / "s21_1" / stage26_1.stage21_1.TRAINABLE_BATCH_FILE)


def _equivalence_audit(serial_rows: list[dict[str, Any]], parallel_rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        ("action_index",),
        ("info", "candidate_set_hash"),
        ("info", "selected_viewpoint"),
        ("info", "path_cost_sources"),
        ("info", "hybrid_astar_path_costs"),
        ("info", "hybrid_astar_pose_path_hashes"),
        ("info", "hybrid_astar_reachable_flags"),
        ("info", "synthetic_terrain_hash"),
        ("info", "synthetic_source_kind"),
        ("info", "platform_contract_hash"),
        ("info", "coverage_source"),
        ("info", "path_cost_source"),
    )
    mismatches: list[dict[str, Any]] = []
    if len(serial_rows) != len(parallel_rows):
        mismatches.append({"field": "transition_count", "serial": len(serial_rows), "parallel": len(parallel_rows)})
    for index, (serial, parallel) in enumerate(zip(serial_rows, parallel_rows)):
        for field_path in fields:
            a = _nested(serial, field_path)
            b = _nested(parallel, field_path)
            if not _equivalent_value(a, b):
                mismatches.append(
                    {
                        "transition_index": index,
                        "transition_id": serial.get("transition_id"),
                        "field": ".".join(field_path),
                        "serial": a,
                        "parallel": b,
                    }
                )
    return {
        "schema_version": "xunce-stage26-4a-parallel-equivalence-audit/v1",
        "serial_transition_count": len(serial_rows),
        "parallel_transition_count": len(parallel_rows),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
    }


def _worker_audit(serial_rows: list[dict[str, Any]], parallel_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = serial_rows + parallel_rows
    required = (
        "hybrid_astar_candidate_eval_parallel_enabled",
        "hybrid_astar_candidate_eval_workers_requested",
        "hybrid_astar_candidate_eval_workers_effective",
        "hybrid_astar_candidate_eval_submitted_count",
        "hybrid_astar_candidate_eval_failed_count",
        "hybrid_astar_candidate_eval_duration_s",
    )
    missing = 0
    parallel_enabled = 0
    failed_count = 0
    workers_effective: list[int] = []
    durations: list[float] = []
    for row in rows:
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        if any(key not in info for key in required):
            missing += 1
            continue
        if info.get("hybrid_astar_candidate_eval_parallel_enabled") is True:
            parallel_enabled += 1
        failed_count += int(info.get("hybrid_astar_candidate_eval_failed_count") or 0)
        workers_effective.append(int(info.get("hybrid_astar_candidate_eval_workers_effective") or 0))
        duration = _finite_or_none(info.get("hybrid_astar_candidate_eval_duration_s"))
        if duration is not None:
            durations.append(duration)
    return {
        "schema_version": "xunce-stage26-4a-hybrid-astar-worker-audit/v1",
        "transition_count": len(rows),
        "audit_missing_count": missing,
        "parallel_enabled_count": parallel_enabled,
        "failed_count": failed_count,
        "max_effective_worker_count": max(workers_effective) if workers_effective else 0,
        "total_candidate_eval_duration_s": sum(durations),
    }


def _candidate_binding_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    required_arrays = ("path_cost_sources", "hybrid_astar_path_costs", "hybrid_astar_pose_path_hashes", "hybrid_astar_reachable_flags")
    for row in rows:
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        action_index = row.get("action_index")
        for key in required_arrays:
            values = info.get(key)
            if not isinstance(values, list):
                mismatches.append({"transition_id": row.get("transition_id"), "field": key, "reason": "missing_or_not_list"})
                continue
            if not isinstance(action_index, int) or action_index < 0 or action_index >= len(values):
                mismatches.append({"transition_id": row.get("transition_id"), "field": key, "reason": "action_index_out_of_range"})
    return {"schema_version": "xunce-stage26-4a-candidate-binding-audit/v1", "mismatch_count": len(mismatches), "mismatches": mismatches[:50]}


def _runtime_audit(serial_run: dict[str, Any], parallel_run: dict[str, Any], serial_rows: list[dict[str, Any]], parallel_rows: list[dict[str, Any]]) -> dict[str, Any]:
    serial_elapsed = _finite_or_none(serial_run.get("elapsed_s")) or 0.0
    parallel_elapsed = _finite_or_none(parallel_run.get("elapsed_s")) or 0.0
    speedup = serial_elapsed / parallel_elapsed if parallel_elapsed > 0.0 else None
    return {
        "schema_version": "xunce-stage26-4a-collector-runtime-comparison/v1",
        "serial_elapsed_s": serial_elapsed,
        "parallel_elapsed_s": parallel_elapsed,
        "wall_clock_speedup_ratio": speedup,
        "serial_transition_count": len(serial_rows),
        "parallel_transition_count": len(parallel_rows),
        "runtime_is_optimization_metric_only": True,
    }


def _route(
    *,
    boundary_reasons: list[str],
    input_reasons: list[str],
    serial_run: dict[str, Any],
    parallel_run: dict[str, Any],
    equivalence_audit: dict[str, Any],
    worker_audit: dict[str, Any],
    binding_audit: dict[str, Any],
) -> tuple[str, str, str]:
    if boundary_reasons:
        return "failed", "resolve_stage26_4a_boundary_rejections", "boundary fields are open"
    if input_reasons:
        return "failed", "rerun_stage26_4a_required_inputs", "required Stage26 inputs are missing or untrusted"
    if serial_run.get("summary", {}).get("status") != "passed" or parallel_run.get("summary", {}).get("status") != "passed":
        return "failed", "repair_stage26_4a_hybrid_astar_worker_contract", "collector variant failed"
    if int(worker_audit.get("audit_missing_count", 0)) > 0 or int(worker_audit.get("parallel_enabled_count", 0)) == 0:
        return "failed", "repair_stage26_4a_hybrid_astar_worker_contract", "worker audit is missing or parallel path did not run"
    if int(binding_audit.get("mismatch_count", 0)) > 0:
        return "failed", "repair_stage26_4a_candidate_path_cost_binding", "candidate path-cost binding failed"
    if int(equivalence_audit.get("mismatch_count", 0)) > 0:
        return "failed", "repair_stage26_4a_parallel_equivalence", "serial and parallel collector outputs differ"
    return "passed", ROUTE_STAGE26_4, "parallel Stage21.1 Hybrid A* candidate cost output matches serial output"


def _route_blockers(
    route: str,
    serial_run: dict[str, Any],
    parallel_run: dict[str, Any],
    equivalence_audit: dict[str, Any],
    worker_audit: dict[str, Any],
    binding_audit: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if route == "repair_stage26_4a_hybrid_astar_worker_contract":
        if serial_run.get("summary", {}).get("status") != "passed":
            reasons.append("serial_collector_not_passed")
        if parallel_run.get("summary", {}).get("status") != "passed":
            reasons.append("parallel_collector_not_passed")
        if worker_audit.get("audit_missing_count"):
            reasons.append("hybrid_astar_worker_audit_missing")
        if int(worker_audit.get("parallel_enabled_count", 0)) == 0:
            reasons.append("parallel_worker_path_not_exercised")
    if route == "repair_stage26_4a_parallel_equivalence":
        reasons.append("serial_parallel_output_mismatch")
    if route == "repair_stage26_4a_candidate_path_cost_binding":
        reasons.append("candidate_path_cost_binding_mismatch")
    return _unique(reasons)


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"config schema_version must be {CONFIG_SCHEMA_VERSION}")
    defaults = {
        "stage26_0_root": DEFAULT_STAGE26_0_ROOT,
        "stage26_1_base_config": "configs/xunce_stage26_1_synthetic_terrain_collector_smoke_v1.json",
        "required_scenario_count": 2,
        "rollout_steps": 4,
        "dynamic_max_candidates_per_step": 36,
        "dynamic_proposal_pool_limit_per_step": 288,
        "min_trainable_transition_count": 1,
        "theta_bin_count": 8,
        "theta_step_deg": 45,
        "sensor_model_id": "theta-fov-90-range-radius/v1",
        "sensor_fov_deg": 90.0,
        "sensor_range_cells": 3,
        "serial_hybrid_astar_candidate_eval_workers": 1,
        "parallel_hybrid_astar_candidate_eval_workers": 4,
        "stage26_4a_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    for key in ("stage26_0_root", "stage26_1_base_config"):
        config[key] = str(_resolve_path(Path(str(config[key])), repo_root))
    for key in (
        "required_scenario_count",
        "rollout_steps",
        "dynamic_max_candidates_per_step",
        "dynamic_proposal_pool_limit_per_step",
        "min_trainable_transition_count",
        "theta_bin_count",
        "theta_step_deg",
        "sensor_range_cells",
        "serial_hybrid_astar_candidate_eval_workers",
        "parallel_hybrid_astar_candidate_eval_workers",
    ):
        config[key] = _positive_int(config[key], key)
    config["sensor_model_id"] = str(config["sensor_model_id"])
    config["sensor_fov_deg"] = _positive_float(config["sensor_fov_deg"], "sensor_fov_deg")
    config["canary_traffic_fraction"] = _nonnegative_float(config["canary_traffic_fraction"], "canary_traffic_fraction")
    for field in BOUNDARY_FIELDS:
        config.setdefault(field, False)
    return config


def _input_rejections(config: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    summary_path = Path(config["stage26_0_root"]) / "xunce-stage26-0-summary.json"
    reasons: list[str] = []
    summary: dict[str, Any] = {}
    if not summary_path.is_file():
        reasons.append("missing_stage26_0_summary")
        return reasons, summary
    summary = _read_json(summary_path)
    if summary.get("status") != "passed":
        reasons.append("stage26_0_not_passed")
    if summary.get("next_required_change") != "run_stage26_1_synthetic_terrain_collector_smoke":
        reasons.append("stage26_0_route_not_stage26_1")
    if summary.get("synthetic_terrain_model_id") != SYNTHETIC_MODEL_ID:
        reasons.append("stage26_0_synthetic_model_id_mismatch")
    if summary.get("synthetic_terrain_hash") in (None, ""):
        reasons.append("stage26_0_synthetic_hash_missing")
    if summary.get("source_kind") != SYNTHETIC_SOURCE_KIND:
        reasons.append("stage26_0_synthetic_source_kind_mismatch")
    if summary.get("physical_obstacle_cells_written") is not False:
        reasons.append("stage26_0_physical_obstacle_pollution")
    if summary.get("coverage_source") != COVERAGE_SOURCE:
        reasons.append("stage26_0_coverage_source_mismatch")
    if summary.get("path_cost_source") != PATH_COST_SOURCE:
        reasons.append("stage26_0_path_cost_source_mismatch")
    return reasons, summary


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is not False]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _summary_boundary_clean(summary: dict[str, Any]) -> bool:
    for key in BOUNDARY_FIELDS:
        if summary.get(key) is True:
            return False
    return float(summary.get("canary_traffic_fraction", 0.0) or 0.0) == 0.0


def _nested(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _equivalent_value(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1.0e-9, abs_tol=1.0e-9)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equivalent_value(left, right) for left, right in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_equivalent_value(a[key], b[key]) for key in a)
    return a == b


def _finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _positive_float(value: Any, name: str) -> float:
    parsed = _finite_or_none(value)
    if parsed is None or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return parsed


def _nonnegative_float(value: Any, name: str) -> float:
    parsed = _finite_or_none(value)
    if parsed is None or parsed < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite number")
    return parsed


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _render_report(summary: dict[str, Any], runtime: dict[str, Any], equivalence: dict[str, Any], worker: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.4A Parallel Stage21.1 Hybrid A* Candidate Cost Audit",
            "",
            f"- status: {summary.get('status')}",
            f"- next_required_change: {summary.get('next_required_change')}",
            f"- serial_worker_count: {summary.get('serial_worker_count')}",
            f"- parallel_worker_count: {summary.get('parallel_worker_count')}",
            f"- equivalence_mismatch_count: {equivalence.get('mismatch_count')}",
            f"- worker_audit_missing_count: {worker.get('audit_missing_count')}",
            f"- wall_clock_speedup_ratio: {runtime.get('wall_clock_speedup_ratio')}",
            "",
            "This stage only validates candidate-level parallel evaluation in the Stage21.1 collector. It does not run PPO, publish checkpoints, replace policy, connect executor, or start canary traffic.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
