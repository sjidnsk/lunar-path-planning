from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:  # pragma: no cover
    import run_xunce_stage26_8m_generalized_resumable_training_pipeline as stage26_8m
    import run_xunce_stage26_8n_aggressive_sample_update_sweep as stage26_8n
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage26_8m_generalized_resumable_training_pipeline as stage26_8m
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as stage26_8n

import xunce_artifact_io as artifact_io

STAGE_ID = "xunce-stage26-8r-pose-gate-repair"
CONFIG_SCHEMA_VERSION = "xunce-stage26-8r-pose-gate-repair-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-8r-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-8r-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-8r-manifest/v1"
RECOMMENDED_SCHEMA_VERSION = "xunce-stage26-8r-recommended-stage26-8n-config/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_8r_pose_gate_repair_v1.json"
DEFAULT_OUTPUT_ROOT = "D:/xunce/out/s26_8r"
SUMMARY_FILE = "xunce-stage26-8r-summary.json"
ROUTING_FILE = "xunce-stage26-8r-routing.json"
MANIFEST_FILE = "xunce-stage26-8r-manifest.json"
REPORT_FILE = "xunce-stage26-8r-report.md"
MATRIX_FILE = "xunce-stage26-8r-combo-results.jsonl"
RECOMMENDED_FILE = "xunce-stage26-8r-recommended-stage26-8n-config.json"

ROUTE_USE_RECOMMENDED = "use_stage26_8r_recommended_pose_gate_config"
ROUTE_REPAIR = "repair_candidate_pose_generation_or_pose_gate_budget"
ROUTE_INPUTS = "rerun_stage26_8r_required_inputs"

BOUNDARY_FIELDS = (
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)

REPAIR_POLICY = "candidate_current_bearing_sweep/v1"
LEGACY_POLICY = "candidate_viewpoint_current_step/v1"
GATE_SOURCE = "hybrid_astar_pose_reachability/v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=STAGE_ID)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_8r_pose_gate_repair(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_8r_pose_gate_repair(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root)
    output_root = _resolve_path(output_root, repo_root)
    artifact_io.make_dirs(output_root)

    boundary_rejections = _boundary_rejections(config)
    input_rejections = _input_rejections(config)
    base_8m_config = _read_json_if_exists(Path(config["stage26_8n_aggressive_config_path"]))
    combo_results: list[dict[str, Any]] = []

    if not boundary_rejections and not input_rejections:
        for combo in config["repair_combos"]:
            if combo["combo_id"] == "theta5_iter200" and combo_results:
                previous = combo_results[-1]
                if int(previous.get("no_hybrid_astar_pose_reachable_candidate_for_action_mask_count") or 0) == 0:
                    combo_results.append(_skipped_combo_result(combo, reason="pose_gate_already_repaired_before_iter200"))
                    continue
            combo_results.append(
                _run_collector_combo(
                    combo,
                    base_8m_config=base_8m_config,
                    output_root=output_root,
                    repo_root=repo_root,
                )
            )
            if _combo_passes(combo_results[-1], min_trainable_transition_count=int(config["min_trainable_transition_count"])):
                break

    recommendation = _select_recommended_combo(
        combo_results,
        min_trainable_transition_count=int(config["min_trainable_transition_count"]),
    )
    recommended_path = output_root / RECOMMENDED_FILE
    if recommendation:
        _write_json(recommended_path, _recommended_payload(config, recommendation))

    route = ROUTE_INPUTS if input_rejections else ROUTE_USE_RECOMMENDED if recommendation else ROUTE_REPAIR
    if boundary_rejections:
        route = ROUTE_REPAIR
    status = "passed" if route == ROUTE_USE_RECOMMENDED else "failed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "combo_results": combo_results,
        "recommended_combo_id": recommendation.get("combo_id") if recommendation else None,
        "recommended_config": str(recommended_path) if recommendation else None,
        "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
        "stage26_8n_root": config["stage26_8n_root"],
        "stage26_8n_aggressive_config_path": config["stage26_8n_aggressive_config_path"],
        "candidate_reachability_gate_source": GATE_SOURCE,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
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
        "summary": str(output_root / SUMMARY_FILE),
        "routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
        "combo_results": str(output_root / MATRIX_FILE),
        "recommended_config": str(recommended_path) if recommendation else None,
        "stage26_8n_aggressive_config_path": config["stage26_8n_aggressive_config_path"],
    }
    _write_jsonl(output_root / MATRIX_FILE, combo_results)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    artifact_io.write_text(output_root / REPORT_FILE, _render_report(summary))
    return summary


def _run_collector_combo(
    combo: dict[str, Any],
    *,
    base_8m_config: dict[str, Any],
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    combo_root = output_root / str(combo["combo_id"])
    artifact_io.make_dirs(combo_root)
    config_path = combo_root / "xunce-stage26-8m-config.json"
    stage26_8m_root = combo_root / "m"
    generated = _combo_stage26_8m_config(base_8m_config, combo)
    _write_json(config_path, generated)
    loaded = stage26_8m._load_config(config_path, repo_root)
    job = stage26_8m._expand_jobs(loaded, stage26_8m_root, repo_root)[0]
    collector_root = Path(str(job["collector_root"]))
    collector_row = _collector_scan_row(loaded, stage26_8m_root, repo_root, str(job["job_id"]))
    execution_blocking_reason = ""
    if collector_row.get("status") == "pending":
        execution_summary = stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
            config_path=config_path,
            output_root=stage26_8m_root,
            repo_root=repo_root,
            run_mode_override="run_phase",
            job_id_override=str(job["job_id"]),
            phase_override="collector",
            max_jobs_override=1,
        )
        execution_blocking_reason = _collector_only_execution_blocking_reason(execution_summary)
        collector_row = _collector_scan_row(loaded, stage26_8m_root, repo_root, str(job["job_id"]))
    collector_summary_path = collector_root / stage26_8m.stage26_1.SUMMARY_FILE
    collector_summary = _read_json_if_exists(collector_summary_path)
    return _combo_result(
        combo,
        stage26_8m_root,
        collector_root,
        collector_summary,
        collector_row=collector_row,
        execution_blocking_reason=execution_blocking_reason,
    )


def _combo_stage26_8m_config(base: dict[str, Any], combo: dict[str, Any]) -> dict[str, Any]:
    payload = dict(base)
    payload.update(
        {
            "schema_version": stage26_8m.CONFIG_SCHEMA_VERSION,
            "stage_id": stage26_8m.STAGE_ID,
            "run_mode": "run_next",
            "max_jobs_per_invocation": 1,
            "collector_reuse_policy": stage26_8m.COLLECTOR_REUSE_BY_HORIZON_SEED_SCENARIO_ROLLOUT,
            "update_combos": [
                {
                    "combo_id": "collector_only_probe",
                    "epochs": 1,
                    "learning_rate": 1.0e-6,
                    "policy_loss_coefficient": 1.0,
                    "value_loss_coefficient": 0.01,
                    "entropy_coefficient": 0.005,
                    "loss_scale": 0.25,
                }
            ],
            "candidate_reachability_gate_source": GATE_SOURCE,
            "candidate_reachability_theta_proposal_policy": combo["candidate_reachability_theta_proposal_policy"],
            "candidate_reachability_max_theta_proposals_per_candidate": int(
                combo["candidate_reachability_max_theta_proposals_per_candidate"]
            ),
            "hybrid_astar_max_iterations": int(combo["hybrid_astar_max_iterations"]),
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return payload


def _combo_result(
    combo: dict[str, Any],
    stage26_8m_root: Path,
    collector_root: Path,
    collector_summary: dict[str, Any],
    *,
    collector_row: dict[str, Any],
    execution_blocking_reason: str = "",
) -> dict[str, Any]:
    validation_blocking_reason = (
        execution_blocking_reason
        or str(collector_row.get("blocking_reason") or "")
        or ("collector_phase_not_complete" if collector_row.get("status") != "complete" else "")
    )
    collector_valid = collector_row.get("status") == "complete" and not validation_blocking_reason
    return {
        "schema_version": "xunce-stage26-8r-combo-result/v1",
        "combo_id": combo["combo_id"],
        "status": collector_summary.get("status", "missing") if collector_valid else "failed",
        "stage26_8m_root": str(stage26_8m_root),
        "collector_root": str(collector_root),
        "collector_validation_status": "passed" if collector_valid else "failed",
        "collector_validation_blocking_reason": validation_blocking_reason,
        "collector_phase_config_hash": collector_row.get("config_hash"),
        "collector_input_hash": collector_row.get("input_hash"),
        "candidate_reachability_theta_proposal_policy": combo["candidate_reachability_theta_proposal_policy"],
        "candidate_reachability_max_theta_proposals_per_candidate": int(
            combo["candidate_reachability_max_theta_proposals_per_candidate"]
        ),
        "hybrid_astar_max_iterations": int(combo["hybrid_astar_max_iterations"]),
        "trainable_transition_count": _collector_trainable_transition_count(collector_summary),
        "transition_count": _int_value(collector_summary.get("transition_count")),
        "no_hybrid_astar_pose_reachable_candidate_for_action_mask_count": _int_value(
            collector_summary.get("stage21_1_no_hybrid_astar_pose_reachable_candidate_for_action_mask_count")
            or collector_summary.get("rejection_no_hybrid_astar_pose_reachable_candidate_for_action_mask_count")
        ),
        "selected_pose_unreachable_terminal_count": _int_value(
            collector_summary.get("stage21_1_selected_pose_unreachable_terminal_count")
        ),
        "scenario_early_terminal_step_histogram": collector_summary.get("stage21_1_scenario_early_terminal_step_histogram"),
        "trainable_transition_count_by_scenario": collector_summary.get("stage21_1_trainable_transition_count_by_scenario"),
    }


def _collector_scan_row(config: dict[str, Any], output_root: Path, repo_root: Path, job_id: str) -> dict[str, Any]:
    rows = stage26_8m._scan_jobs(config=config, output_root=output_root, repo_root=repo_root)
    for row in rows:
        if str(row.get("job_id")) == job_id and row.get("phase") == "collector":
            return row
    return {"status": "missing", "blocking_reason": "collector_phase_state_missing"}


def _collector_only_execution_blocking_reason(summary: dict[str, Any]) -> str:
    executions = summary.get("phase_executions") or []
    if len(executions) > 1:
        return "stage26_8m_executed_multiple_phases"
    if any(str(row.get("phase") or "") != "collector" for row in executions):
        return "stage26_8m_executed_non_collector_phase"
    return ""


def _collector_trainable_transition_count(summary: dict[str, Any]) -> int:
    return _int_value(
        summary.get("trainable_transition_count")
        or summary.get("batch_row_count")
        or summary.get("transition_count")
    )


def _skipped_combo_result(combo: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage26-8r-combo-result/v1",
        "combo_id": combo["combo_id"],
        "status": "skipped",
        "skip_reason": reason,
        "candidate_reachability_theta_proposal_policy": combo["candidate_reachability_theta_proposal_policy"],
        "candidate_reachability_max_theta_proposals_per_candidate": int(
            combo["candidate_reachability_max_theta_proposals_per_candidate"]
        ),
        "hybrid_astar_max_iterations": int(combo["hybrid_astar_max_iterations"]),
    }


def _combo_passes(result: dict[str, Any], *, min_trainable_transition_count: int) -> bool:
    return (
        result.get("status") == "passed"
        and int(result.get("trainable_transition_count") or 0) >= min_trainable_transition_count
        and int(result.get("no_hybrid_astar_pose_reachable_candidate_for_action_mask_count") or 0) == 0
        and int(result.get("selected_pose_unreachable_terminal_count") or 0) == 0
    )


def _select_recommended_combo(
    results: list[dict[str, Any]],
    *,
    min_trainable_transition_count: int,
) -> dict[str, Any]:
    eligible = [row for row in results if _combo_passes(row, min_trainable_transition_count=min_trainable_transition_count)]
    if not eligible:
        return {}
    return sorted(
        eligible,
        key=lambda row: (
            int(row.get("candidate_reachability_max_theta_proposals_per_candidate") or 0),
            int(row.get("hybrid_astar_max_iterations") or 0),
            -int(row.get("trainable_transition_count") or 0),
            str(row.get("combo_id") or ""),
        ),
    )[0]


def _recommended_payload(config: dict[str, Any], recommendation: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RECOMMENDED_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "recommended_combo_id": recommendation["combo_id"],
        "candidate_reachability_gate_source": GATE_SOURCE,
        "candidate_reachability_theta_proposal_policy": recommendation["candidate_reachability_theta_proposal_policy"],
        "candidate_reachability_max_theta_proposals_per_candidate": int(
            recommendation["candidate_reachability_max_theta_proposals_per_candidate"]
        ),
        "hybrid_astar_max_iterations": int(recommendation["hybrid_astar_max_iterations"]),
        "trainable_transition_count": int(recommendation["trainable_transition_count"]),
        "min_trainable_transition_count": int(config["min_trainable_transition_count"]),
        "no_hybrid_astar_pose_reachable_candidate_for_action_mask_count": int(
            recommendation["no_hybrid_astar_pose_reachable_candidate_for_action_mask_count"]
        ),
        "selected_pose_unreachable_terminal_count": int(recommendation["selected_pose_unreachable_terminal_count"]),
        "collector_root": recommendation.get("collector_root"),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    if payload.get("stage_id") != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID}")
    defaults = {
        "stage26_8n_root": stage26_8n.DEFAULT_OUTPUT_ROOT,
        "stage26_8n_aggressive_config_path": "",
        "min_trainable_transition_count": 100,
        "repair_combos": _default_repair_combos(),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    config["stage26_8n_root"] = str(_resolve_path(Path(str(config["stage26_8n_root"])), repo_root))
    if not config.get("stage26_8n_aggressive_config_path"):
        config["stage26_8n_aggressive_config_path"] = str(Path(config["stage26_8n_root"]) / stage26_8n.AGGRESSIVE_CONFIG_FILE)
    config["stage26_8n_aggressive_config_path"] = str(
        _resolve_path(Path(str(config["stage26_8n_aggressive_config_path"])), repo_root)
    )
    config["min_trainable_transition_count"] = _positive_int(
        config["min_trainable_transition_count"],
        "min_trainable_transition_count",
    )
    config["repair_combos"] = _repair_combos(config["repair_combos"])
    for field in BOUNDARY_FIELDS:
        config[field] = bool(config.get(field, False))
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0))
    return config


def _default_repair_combos() -> list[dict[str, Any]]:
    return [
        _combo("baseline_readonly", LEGACY_POLICY, 1, 50),
        _combo("theta3_iter50", REPAIR_POLICY, 3, 50),
        _combo("theta3_iter100", REPAIR_POLICY, 3, 100),
        _combo("theta5_iter100", REPAIR_POLICY, 5, 100),
        _combo("theta5_iter200", REPAIR_POLICY, 5, 200),
    ]


def _combo(combo_id: str, policy: str, cap: int, iterations: int) -> dict[str, Any]:
    return {
        "combo_id": combo_id,
        "candidate_reachability_theta_proposal_policy": policy,
        "candidate_reachability_max_theta_proposals_per_candidate": cap,
        "hybrid_astar_max_iterations": iterations,
    }


def _repair_combos(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("repair_combos must be a non-empty array")
    combos = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("repair_combos entries must be objects")
        combo_id = str(item.get("combo_id") or "").strip()
        if not combo_id:
            raise ValueError("repair combo_id is required")
        if combo_id in seen:
            raise ValueError(f"duplicate repair combo_id: {combo_id}")
        seen.add(combo_id)
        policy = str(item.get("candidate_reachability_theta_proposal_policy") or LEGACY_POLICY)
        if policy not in {LEGACY_POLICY, REPAIR_POLICY}:
            raise ValueError("candidate_reachability_theta_proposal_policy is invalid")
        combos.append(
            _combo(
                combo_id,
                policy,
                _positive_int(item.get("candidate_reachability_max_theta_proposals_per_candidate"), "candidate_reachability_max_theta_proposals_per_candidate"),
                _positive_int(item.get("hybrid_astar_max_iterations"), "hybrid_astar_max_iterations"),
            )
        )
    return combos


def _input_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not artifact_io.path_is_file(Path(config["stage26_8n_aggressive_config_path"])):
        reasons.append("stage26_8n_aggressive_config_missing")
    else:
        base = _read_json_if_exists(Path(config["stage26_8n_aggressive_config_path"]))
        if base.get("candidate_reachability_gate_source") != GATE_SOURCE:
            reasons.append("stage26_8n_aggressive_config_not_hybrid_pose_gate")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage26.8R Pose Gate Repair",
        "",
        f"- status: `{summary['status']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- recommended_combo_id: `{summary.get('recommended_combo_id')}`",
        f"- recommended_config: `{summary.get('recommended_config')}`",
        "",
        "## Combo Results",
    ]
    for row in summary.get("combo_results", []):
        lines.append(
            "- "
            f"{row.get('combo_id')}: status=`{row.get('status')}`, "
            f"policy=`{row.get('candidate_reachability_theta_proposal_policy')}`, "
            f"cap=`{row.get('candidate_reachability_max_theta_proposals_per_candidate')}`, "
            f"max_iterations=`{row.get('hybrid_astar_max_iterations')}`, "
            f"trainable=`{row.get('trainable_transition_count')}`, "
            f"pose_gate_empty=`{row.get('no_hybrid_astar_pose_reachable_candidate_for_action_mask_count')}`"
        )
    lines.extend(
        [
            "",
            "This stage runs collector-only pose-gate repair probes. It does not run PPO update/eval, publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    return artifact_io.read_json(path)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return artifact_io.read_json(path) if artifact_io.path_is_file(path) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    artifact_io.write_json(path, payload)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    artifact_io.write_jsonl(path, rows)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _positive_int(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive") from exc
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _int_value(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if math.isfinite(float(number)) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
