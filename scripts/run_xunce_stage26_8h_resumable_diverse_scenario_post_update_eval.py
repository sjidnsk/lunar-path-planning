from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_high_fidelity_exploration_coverage_comparison as hf
    import run_xunce_stage21_5_post_update_offline_trajectory_evaluation as stage21_5
    import run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as stage26_3
    import run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as stage26_8
    import run_xunce_stage26_8g_repair_synthetic_scenario_diversity as stage26_8g
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_high_fidelity_exploration_coverage_comparison as hf
    import scripts.run_xunce_stage21_5_post_update_offline_trajectory_evaluation as stage21_5
    import scripts.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as stage26_3
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as stage26_8
    import scripts.run_xunce_stage26_8g_repair_synthetic_scenario_diversity as stage26_8g


STAGE_ID = "xunce-stage26-8h-resumable-diverse-scenario-post-update-eval"
CONFIG_SCHEMA_VERSION = "xunce-stage26-8h-resumable-diverse-scenario-post-update-eval-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-8h-summary/v1"
PHASE_STATE_SCHEMA_VERSION = "xunce-stage26-8h-phase-state/v1"
EVAL_AUDIT_SCHEMA_VERSION = "xunce-stage26-8h-eval-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-8h-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-8h-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_8h_resumable_diverse_scenario_post_update_eval_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval_v1"
)
DEFAULT_STAGE26_8G_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8g_repair_synthetic_scenario_diversity_v1"
)

SUMMARY_FILE = "xunce-stage26-8h-summary.json"
PHASE_STATE_FILE = "xunce-stage26-8h-phase-state.jsonl"
PRE_EVAL_AUDIT_FILE = "xunce-stage26-8h-pre-eval-audit.json"
POST_EVAL_AUDIT_FILE = "xunce-stage26-8h-post-eval-audit.json"
STAGE26_3_SUMMARY_FILE = "xunce-stage26-8h-stage26-3-summary.json"
RECHECK_FILE = "xunce-stage26-8h-scenario-diversity-recheck-audit.json"
ROUTING_FILE = "xunce-stage26-8h-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-8h-report.md"
MANIFEST_FILE = "xunce-stage26-8h-manifest.json"

PHASES = ("pre_eval", "post_eval", "stage26_3_aggregate")
ROUTE_INPUTS = "rerun_stage26_8h_required_inputs"
ROUTE_FIXTURE = "repair_stage26_8g_collector_scenario_diversity_binding"
ROUTE_PRE = "continue_stage26_8h_pre_eval"
ROUTE_POST = "continue_stage26_8h_post_eval"
ROUTE_AGGREGATE = "repair_stage26_8h_stage26_3_aggregate_contract"
ROUTE_EVAL_DIVERSITY = "repair_stage26_8g_eval_scenario_diversity_binding"
ROUTE_BINDING = "repair_stage26_8h_eval_binding_or_safety"
ROUTE_POLICY_SIGNAL = "repair_stage26_synthetic_policy_update_signal_strength"
ROUTE_RESUME = "resume_stage26_8d_seed_horizon_jobs_with_diverse_scenarios"
ROUTE_BOUNDARY = "resolve_stage26_8h_boundary_rejections"

BOUNDARY_FIELDS = ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resume Stage26.8G diverse-scenario Stage26.3 eval.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-mode", choices=("run_next", "aggregate_only", "run_phase"))
    parser.add_argument("--phase", choices=PHASES)
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
        run_mode_override=args.run_mode,
        phase_override=args.phase,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "next_required_change": summary["next_required_change"],
                "next_phase": summary.get("next_phase"),
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    run_mode_override: str | None = None,
    phase_override: str | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(config_path, repo_root)
    if run_mode_override is not None:
        config["run_mode"] = run_mode_override
    if phase_override is not None:
        config["phase"] = phase_override
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage26_8g_root = Path(config["stage26_8g_root"])
    boundary_rejections = _boundary_rejections(config)
    input_rejections = _input_rejections(stage26_8g_root)
    fixture_rows = _fixture_rows(stage26_8g_root)
    fixture_rejections = _fixture_rejections(fixture_rows)

    phase_execution: dict[str, Any] = {}
    phase_rows = _scan_phases(config, output_root, stage26_8g_root)
    if not boundary_rejections and not input_rejections and not fixture_rejections and config["run_mode"] != "aggregate_only":
        phase = _select_phase(phase_rows, config)
        if phase:
            try:
                phase_execution = _run_phase(phase, config=config, output_root=output_root, stage26_8g_root=stage26_8g_root, repo_root=repo_root)
            except Exception as exc:
                phase_execution = {
                    "phase": phase,
                    "started_at": _utc_now(),
                    "finished_at": _utc_now(),
                    "status": "failed",
                    "blocking_reason": f"{type(exc).__name__}:{exc}",
                }

    phase_rows = _scan_phases(config, output_root, stage26_8g_root, phase_execution=phase_execution)
    pre_audit = _eval_audit(_phase_root(phase_rows, "pre_eval"), "pre")
    post_audit = _eval_audit(_phase_root(phase_rows, "post_eval"), "post")
    stage26_3_summary = _read_json_if_exists(output_root / "s26_3_aggregate" / stage26_3.SUMMARY_FILE)
    recheck = _scenario_recheck(output_root / "s26_3_aggregate", config)
    route = _route(
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        fixture_rejections=fixture_rejections,
        phase_rows=phase_rows,
        stage26_3_summary=stage26_3_summary,
        recheck=recheck,
    )
    phase_execution_failed = phase_execution.get("status") == "failed"
    status = "passed" if route in {ROUTE_PRE, ROUTE_POST, ROUTE_POLICY_SIGNAL, ROUTE_RESUME} and not phase_execution_failed else "failed"
    next_phase = _next_phase(phase_rows)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "run_mode": config["run_mode"],
        "phase_execution": phase_execution,
        "phase_execution_failed": phase_execution_failed,
        "next_phase": next_phase,
        "stage26_8g_root": str(stage26_8g_root),
        "stage26_8g_stage26_1_passed": _stage_summary_passed(stage26_8g_root / "h16_seed260801" / "s26_1" / "xunce-stage26-1-summary.json"),
        "stage26_8g_stage26_2_passed": _stage_summary_passed(stage26_8g_root / "h16_seed260801" / "s26_2" / "xunce-stage26-2-summary.json"),
        "fixture_count": len(fixture_rows),
        "fixture_unique_content_hash_count": len({str(row.get("scenario_diversity_content_hash") or "") for row in fixture_rows}),
        "pre_eval_complete": _phase_complete(phase_rows, "pre_eval"),
        "post_eval_complete": _phase_complete(phase_rows, "post_eval"),
        "stage26_3_aggregate_complete": _phase_complete(phase_rows, "stage26_3_aggregate"),
        "stage26_3_status": stage26_3_summary.get("status"),
        "stage26_3_next_required_change": stage26_3_summary.get("next_required_change"),
        "stage26_3_root": str(output_root / "s26_3_aggregate"),
        "scenario_signature_duplicate_group_count": recheck.get("scenario_signature_duplicate_group_count"),
        "scenario_diversity_repaired": recheck.get("scenario_diversity_repaired"),
        "strong_state_join_available_count": stage26_3_summary.get("strong_state_join_available_count"),
        "selected_action_changed_count": stage26_3_summary.get("selected_action_changed_count"),
        "main_coverage_per_100m_delta": _float(stage26_3_summary.get("main_coverage_per_100m_delta") or stage26_3_summary.get("coverage_per_100m_delta")),
        "coverage_denominator_source": stage26_8.COVERAGE_DENOMINATOR_SOURCE,
        "post_update_success_metric": stage26_8.SUCCESS_METRIC,
        "coverage_source": stage26_8.COVERAGE_SOURCE,
        "path_cost_source": stage26_8.PATH_COST_SOURCE,
        "synthetic_source_kind": stage26_8.SYNTHETIC_SOURCE_KIND,
        "action_space_type": stage26_8.ACTION_SPACE_TYPE,
        "max_traversable_slope_deg": 30.0,
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
        "fixture_rejections": fixture_rejections,
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
        "fixture_rejections": fixture_rejections,
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
            "phase_state": str(output_root / PHASE_STATE_FILE),
            "pre_eval_audit": str(output_root / PRE_EVAL_AUDIT_FILE),
            "post_eval_audit": str(output_root / POST_EVAL_AUDIT_FILE),
            "stage26_3_summary": str(output_root / STAGE26_3_SUMMARY_FILE),
            "scenario_diversity_recheck": str(output_root / RECHECK_FILE),
            "routing": str(output_root / ROUTING_FILE),
            "report": str(output_root / REPORT_FILE),
        },
    }
    _write_jsonl(output_root / PHASE_STATE_FILE, phase_rows)
    _write_json(output_root / PRE_EVAL_AUDIT_FILE, pre_audit)
    _write_json(output_root / POST_EVAL_AUDIT_FILE, post_audit)
    _write_json(output_root / STAGE26_3_SUMMARY_FILE, stage26_3_summary)
    _write_json(output_root / RECHECK_FILE, recheck)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _run_phase(
    phase: str,
    *,
    config: dict[str, Any],
    output_root: Path,
    stage26_8g_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    started_at = _utc_now()
    if phase == "pre_eval":
        root = output_root / "pre_ppo_xunce"
        _run_eval_phase(config, stage26_8g_root, repo_root, root, label="pre")
    elif phase == "post_eval":
        root = output_root / "post_ppo_xunce"
        _run_eval_phase(config, stage26_8g_root, repo_root, root, label="post")
    elif phase == "stage26_3_aggregate":
        root = output_root / "s26_3_aggregate"
        _run_stage26_3_aggregate(config, stage26_8g_root, output_root, repo_root)
    else:  # pragma: no cover
        raise ValueError(f"unknown phase: {phase}")
    summary_path = _summary_path_for_phase(output_root, stage26_8g_root, phase)
    summary = _read_json_if_exists(summary_path)
    return {
        "phase": phase,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "output_root": str(root),
        "summary_path": str(summary_path),
        "status": summary.get("status"),
        "next_required_change": summary.get("next_required_change"),
    }


def _run_eval_phase(config: dict[str, Any], stage26_8g_root: Path, repo_root: Path, output_root: Path, *, label: str) -> None:
    stage21_5_config = _stage21_5_config(stage26_8g_root)
    stage21_4_summary = _read_json(Path(stage21_5_config["stage21_4_tiny_ppo_update_smoke_root"]) / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json")
    checkpoint_key = "source_xunce_candidate_checkpoint" if label == "pre" else "experimental_checkpoint_path"
    stage21_5._run_high_fidelity_eval(
        stage21_5_config,
        repo_root=repo_root,
        output_root=output_root,
        checkpoint_path=Path(stage21_4_summary[checkpoint_key]),
        label=label,
    )


def _run_stage26_3_aggregate(config: dict[str, Any], stage26_8g_root: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    stage26_3_config = _read_json(_stage26_3_config_path(stage26_8g_root))
    pre_root = _completed_eval_root(output_root, stage26_8g_root, "pre_eval")
    post_root = _completed_eval_root(output_root, stage26_8g_root, "post_eval")
    stage26_3_config.update(
        {
            "execute_high_fidelity_evaluations": False,
            "pre_ppo_evaluation_root": str(pre_root),
            "post_ppo_evaluation_root": str(post_root),
            "stage21_5_timeout_seconds": 0.0,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    config_path = output_root / "xunce-stage26-8h-stage26-3-aggregate-config.json"
    _write_json(config_path, stage26_3_config)
    return stage26_3.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke(
        config_path=config_path,
        output_root=output_root / "s26_3_aggregate",
        repo_root=repo_root,
    )


def _scan_phases(
    config: dict[str, Any],
    output_root: Path,
    stage26_8g_root: Path,
    *,
    phase_execution: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    execution_phase = (phase_execution or {}).get("phase")
    rows = []
    for phase in PHASES:
        root = _phase_root_from_outputs(output_root, stage26_8g_root, phase)
        summary_path = _summary_path_for_phase(output_root, stage26_8g_root, phase)
        summary = _read_json_if_exists(summary_path)
        complete = _phase_summary_complete(phase, root, summary)
        rows.append(
            {
                "schema_version": PHASE_STATE_SCHEMA_VERSION,
                "phase": phase,
                "status": "complete" if complete else "pending",
                "resume_decision": "reuse_completed_artifact" if complete else "run_next_pending_phase",
                "source_root": str(root),
                "output_root": str(_phase_output_root(output_root, phase)),
                "summary_path": str(summary_path),
                "blocking_reason": _phase_row_blocking_reason(phase, root, summary, phase_execution if execution_phase == phase else {}),
                "execution": phase_execution if execution_phase == phase else {},
            }
        )
    return rows


def _select_phase(phase_rows: list[dict[str, Any]], config: dict[str, Any]) -> str | None:
    if config["run_mode"] == "run_phase":
        return str(config.get("phase") or "")
    for row in phase_rows:
        if row["status"] != "complete":
            return str(row["phase"])
    return None


def _route(
    *,
    boundary_rejections: list[str],
    input_rejections: list[str],
    fixture_rejections: list[str],
    phase_rows: list[dict[str, Any]],
    stage26_3_summary: dict[str, Any],
    recheck: dict[str, Any],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections:
        return ROUTE_INPUTS
    if fixture_rejections:
        return ROUTE_FIXTURE
    if not _phase_complete(phase_rows, "pre_eval"):
        return ROUTE_PRE
    if not _phase_complete(phase_rows, "post_eval"):
        return ROUTE_POST
    if not _phase_complete(phase_rows, "stage26_3_aggregate"):
        return ROUTE_AGGREGATE
    if int(recheck.get("scenario_signature_duplicate_group_count") or 0) > 0:
        return ROUTE_EVAL_DIVERSITY
    if stage26_8._binding_or_safety_failure(stage26_3_summary) or stage26_8._stage26_3_execution_failure(stage26_3_summary) or stage26_8._lineage_mismatch(stage26_3_summary):
        return ROUTE_BINDING
    if _float(stage26_3_summary.get("main_coverage_per_100m_delta") or stage26_3_summary.get("coverage_per_100m_delta")) > 0.0:
        return ROUTE_RESUME
    return ROUTE_POLICY_SIGNAL


def _input_rejections(stage26_8g_root: Path) -> list[str]:
    reasons: list[str] = []
    if not stage26_8g_root.exists():
        return ["stage26_8g_root_missing"]
    for rel, label in (
        ("h16_seed260801/s26_1/xunce-stage26-1-summary.json", "stage26_1"),
        ("h16_seed260801/s26_2/xunce-stage26-2-summary.json", "stage26_2"),
    ):
        summary = _read_json_if_exists(stage26_8g_root / rel)
        if summary.get("status") != "passed":
            reasons.append(f"{label}_not_passed")
    for path, label in (
        (_stage26_3_config_path(stage26_8g_root), "stage26_3_config"),
        (_stage21_5_config_path(stage26_8g_root), "stage21_5_config"),
        (_high_fidelity_config_path(stage26_8g_root), "high_fidelity_config"),
    ):
        if not path.is_file():
            reasons.append(f"{label}_missing")
    return reasons


def _fixture_rejections(rows: list[dict[str, Any]]) -> list[str]:
    if len(rows) < 3:
        return ["scenario_fixture_count_below_3"]
    content_hashes = {str(row.get("scenario_diversity_content_hash") or "") for row in rows}
    signatures = {str(row.get("scenario_diversity_signature_hash") or "") for row in rows}
    starts = {_json_key(row.get("scenario_start_cell")) for row in rows}
    reasons: list[str] = []
    if "" in content_hashes or len(content_hashes) != len(rows):
        reasons.append("scenario_content_hash_not_unique")
    if "" in signatures or len(signatures) != len(rows):
        reasons.append("scenario_signature_hash_not_unique")
    if any(not _valid_start_cell(row.get("scenario_start_cell")) for row in rows):
        reasons.append("scenario_start_cell_missing_or_invalid")
    if len(starts) != len(rows):
        reasons.append("scenario_start_cell_not_unique")
    return reasons


def _fixture_rows(stage26_8g_root: Path) -> list[dict[str, Any]]:
    for path in (
        stage26_8g_root / stage26_8g.FIXTURE_FILE,
        stage26_8g_root / "h16_seed260801" / "s26_1" / "src" / "xunce-stage26-scenario-fixtures.jsonl",
    ):
        rows = _read_jsonl_if_exists(path)
        if rows:
            return rows
    return []


def _valid_start_cell(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    return all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)


def _phase_root_from_outputs(output_root: Path, stage26_8g_root: Path, phase: str) -> Path:
    current = _phase_output_root(output_root, phase)
    if _phase_summary_complete(phase, current, _read_json_if_exists(_summary_for_root(phase, current))):
        return current
    if phase in {"pre_eval", "post_eval"}:
        carryover = stage26_8g_root / "h16_seed260801" / "s26_3" / ("pre" if phase == "pre_eval" else "post")
        if _phase_summary_complete(phase, carryover, _read_json_if_exists(_summary_for_root(phase, carryover))):
            return carryover
    return current


def _phase_output_root(output_root: Path, phase: str) -> Path:
    if phase == "pre_eval":
        return output_root / "pre_ppo_xunce"
    if phase == "post_eval":
        return output_root / "post_ppo_xunce"
    return output_root / "s26_3_aggregate"


def _completed_eval_root(output_root: Path, stage26_8g_root: Path, phase: str) -> Path:
    root = _phase_root_from_outputs(output_root, stage26_8g_root, phase)
    if not _phase_summary_complete(phase, root, _read_json_if_exists(_summary_for_root(phase, root))):
        raise RuntimeError(f"{phase} is not complete")
    return root


def _phase_root(phase_rows: list[dict[str, Any]], phase: str) -> Path:
    for row in phase_rows:
        if row["phase"] == phase:
            return Path(str(row["source_root"]))
    return Path()


def _summary_path_for_phase(output_root: Path, stage26_8g_root: Path, phase: str) -> Path:
    return _summary_for_root(phase, _phase_root_from_outputs(output_root, stage26_8g_root, phase))


def _summary_for_root(phase: str, root: Path) -> Path:
    if phase in {"pre_eval", "post_eval"}:
        return root / hf.SUMMARY_FILE
    return root / stage26_3.SUMMARY_FILE


def _phase_summary_complete(phase: str, root: Path, summary: dict[str, Any]) -> bool:
    if not summary:
        return False
    if phase in {"pre_eval", "post_eval"}:
        if summary.get("status") != "passed":
            return False
        return (
            (root / hf.EPISODES_FILE).is_file()
            and (root / hf.MODEL_INFERENCE_FILE).is_file()
            and _jsonl_count(root / hf.EPISODES_FILE) > 0
            and _jsonl_count(root / hf.MODEL_INFERENCE_FILE) > 0
        )
    return summary.get("schema_version") == stage26_3.SUMMARY_SCHEMA_VERSION and summary.get("status") in {"passed", "failed"}


def _phase_complete(rows: list[dict[str, Any]], phase: str) -> bool:
    return any(row["phase"] == phase and row["status"] == "complete" for row in rows)


def _next_phase(rows: list[dict[str, Any]]) -> str | None:
    for row in rows:
        if row["status"] != "complete":
            return str(row["phase"])
    return None


def _phase_blocking_reason(phase: str, root: Path, summary: dict[str, Any]) -> str:
    if not summary:
        return "summary_missing"
    if summary.get("status") != "passed":
        return "summary_not_passed"
    if phase in {"pre_eval", "post_eval"} and not (root / hf.MODEL_INFERENCE_FILE).is_file():
        return "model_inference_missing"
    if phase in {"pre_eval", "post_eval"} and not (root / hf.EPISODES_FILE).is_file():
        return "episodes_missing"
    return ""


def _phase_row_blocking_reason(phase: str, root: Path, summary: dict[str, Any], execution: dict[str, Any]) -> str:
    if execution.get("status") == "failed":
        return str(execution.get("blocking_reason") or "phase_execution_failed")
    complete = _phase_summary_complete(phase, root, summary)
    return "" if complete else _phase_blocking_reason(phase, root, summary)


def _eval_audit(root: Path, label: str) -> dict[str, Any]:
    summary = _read_json_if_exists(root / hf.SUMMARY_FILE)
    return {
        "schema_version": EVAL_AUDIT_SCHEMA_VERSION,
        "label": label,
        "root": str(root),
        "summary_exists": bool(summary),
        "status": summary.get("status"),
        "episode_count": _jsonl_count(root / hf.EPISODES_FILE),
        "inference_row_count": _jsonl_count(root / hf.MODEL_INFERENCE_FILE),
        "steps_row_count": _jsonl_count(root / hf.STEPS_FILE),
        "complete": _phase_summary_complete(f"{label}_eval", root, summary),
    }


def _scenario_recheck(stage26_3_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not (stage26_3_root / stage26_3.SUMMARY_FILE).is_file():
        return {"schema_version": "xunce-stage26-8h-scenario-diversity-recheck/v1", "scenario_signature_duplicate_group_count": 0, "scenario_diversity_repaired": False}
    recheck_config = {
        "collector_rollout_steps": int(config["rollout_steps"]),
        "seed": int(config["seed"]),
        "scenario_duplicate_ratio_threshold": float(config["scenario_duplicate_ratio_threshold"]),
    }
    payload = stage26_8g._recheck_stage26_3(stage26_3_root, recheck_config)
    payload["schema_version"] = "xunce-stage26-8h-scenario-diversity-recheck/v1"
    return payload


def _stage21_5_config(stage26_8g_root: Path) -> dict[str, Any]:
    config = _read_json(_stage21_5_config_path(stage26_8g_root))
    config["high_fidelity_config"] = str(_high_fidelity_config_path(stage26_8g_root))
    return config


def _stage26_3_config_path(stage26_8g_root: Path) -> Path:
    return stage26_8g_root / "h16_seed260801" / "xunce-stage26-8g-stage26-3-config.json"


def _stage21_5_config_path(stage26_8g_root: Path) -> Path:
    return stage26_8g_root / "h16_seed260801" / "s26_3" / stage26_3.STAGE21_5_CONFIG_FILE


def _high_fidelity_config_path(stage26_8g_root: Path) -> Path:
    return stage26_8g_root / "h16_seed260801" / "s26_3" / stage26_3.GENERATED_HIGH_FIDELITY_CONFIG_FILE


def _stage_summary_passed(path: Path) -> bool:
    return _read_json_if_exists(path).get("status") == "passed"


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"config schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    config.setdefault("stage26_8g_root", DEFAULT_STAGE26_8G_ROOT)
    config["stage26_8g_root"] = str(_resolve_path(Path(str(config["stage26_8g_root"])), repo_root))
    config["run_mode"] = str(config.get("run_mode", "run_next"))
    if config["run_mode"] not in {"run_next", "aggregate_only", "run_phase"}:
        raise ValueError("run_mode must be run_next, aggregate_only, or run_phase")
    config["phase"] = str(config.get("phase", ""))
    if config["phase"] and config["phase"] not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    config["seed"] = int(config.get("seed", 260801))
    config["rollout_steps"] = int(config.get("rollout_steps", 16))
    config["scenario_duplicate_ratio_threshold"] = float(config.get("scenario_duplicate_ratio_threshold", 0.01))
    for field in BOUNDARY_FIELDS:
        config.setdefault(field, False)
    config.setdefault("canary_traffic_fraction", 0.0)
    return config


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.8H Resumable Diverse Scenario Eval",
            "",
            f"- status: `{summary.get('status')}`",
            f"- next_required_change: `{summary.get('next_required_change')}`",
            f"- next_phase: `{summary.get('next_phase')}`",
            f"- pre_eval_complete: `{summary.get('pre_eval_complete')}`",
            f"- post_eval_complete: `{summary.get('post_eval_complete')}`",
            f"- stage26_3_aggregate_complete: `{summary.get('stage26_3_aggregate_complete')}`",
            f"- main_coverage_per_100m_delta: `{summary.get('main_coverage_per_100m_delta')}`",
            "",
            "This stage only resumes Stage26.3 evaluation artifacts. It does not train or publish checkpoints.",
        ]
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _jsonl_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())


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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
