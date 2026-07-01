from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair as stage26_7
    import run_xunce_stage26_7b_coverable_cell_semantics_contract as stage26_7b
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair as stage26_7
    import scripts.run_xunce_stage26_7b_coverable_cell_semantics_contract as stage26_7b


STAGE_ID = "xunce-stage26-7c-main-coverable-coverage-efficiency-rerun"
CONFIG_SCHEMA_VERSION = "xunce-stage26-7c-main-coverable-coverage-efficiency-rerun-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-7c-summary/v1"
EFFICIENCY_AUDIT_SCHEMA_VERSION = "xunce-stage26-7c-main-coverable-efficiency-audit/v1"
BINDING_AUDIT_SCHEMA_VERSION = "xunce-stage26-7c-denominator-binding-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-7c-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-7c-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_7c_main_coverable_coverage_efficiency_rerun_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_7c_main_coverable_coverage_efficiency_rerun_v1"
)
DEFAULT_STAGE26_7B_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_7b_coverable_cell_semantics_contract_v1"
)

SUMMARY_FILE = "xunce-stage26-7c-summary.json"
STAGE26_7_CONFIG_FILE = "xunce-stage26-7c-stage26-7-config.json"
STAGE26_7_SUMMARY_FILE = "xunce-stage26-7c-stage26-7-summary.json"
EFFICIENCY_AUDIT_FILE = "xunce-stage26-7c-main-coverable-efficiency-audit.json"
BINDING_AUDIT_FILE = "xunce-stage26-7c-denominator-binding-audit.json"
ROUTING_FILE = "xunce-stage26-7c-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-7c-report.md"
MANIFEST_FILE = "xunce-stage26-7c-manifest.json"

ROUTE_INPUTS = "rerun_stage26_7c_required_inputs"
ROUTE_BINDING = "repair_stage26_7c_main_coverable_binding"
ROUTE_SAFETY = "repair_stage26_7c_safety_regression"
ROUTE_MARGIN = "calibrate_stage26_synthetic_discrete_margin_crossing_after_credit"
ROUTE_TARGET_SCORE = "repair_stage26_7_path_efficiency_target_score"
ROUTE_CREDIT = "repair_stage26_synthetic_credit_assignment"
ROUTE_MULTI_SEED = "run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot"
ROUTE_BOUNDARY = "resolve_stage26_7c_boundary_rejections"
CHILD_BLOCKING_ROUTES = {
    stage26_7.ROUTE_INPUTS,
    stage26_7.ROUTE_SAMPLER,
    stage26_7.ROUTE_LOGPROB,
    stage26_7.ROUTE_STABILITY,
}

BOUNDARY_FIELDS = (
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.7C main-coverable coverage efficiency rerun.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_7c_main_coverable_coverage_efficiency_rerun(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_7c_main_coverable_coverage_efficiency_rerun(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage26_7b_summary = _read_json_if_exists(Path(config["stage26_7b_root"]) / stage26_7b.SUMMARY_FILE)
    boundary_rejections = _boundary_rejections(config)
    input_rejections = _input_rejections(stage26_7b_summary)

    stage26_7_config = _build_stage26_7_config(config)
    stage26_7_config_path = output_root / STAGE26_7_CONFIG_FILE
    _write_json(stage26_7_config_path, stage26_7_config)
    # Keep the nested rerun root short; Stage26.7 -> Stage26.1 -> Stage21.1
    # writes long artifact names and Windows still commonly hits MAX_PATH.
    stage26_7_root = output_root / "r"
    stage26_7_summary: dict[str, Any] = {}
    if not boundary_rejections and not input_rejections and bool(config.get("run_stage26_7", True)):
        stage26_7_summary = stage26_7.run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair(
            config_path=stage26_7_config_path,
            output_root=stage26_7_root,
            repo_root=repo_root,
        )

    binding_audit = _denominator_binding_audit(stage26_7_config, stage26_7_root, stage26_7_summary)
    efficiency_audit = _efficiency_audit(stage26_7_summary)
    route = _route(
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        binding_audit=binding_audit,
        stage26_7_summary=stage26_7_summary,
        efficiency_audit=efficiency_audit,
    )
    status = "passed" if route == ROUTE_MULTI_SEED else "failed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "stage26_7b_root": config["stage26_7b_root"],
        "stage26_7b_status": stage26_7b_summary.get("status"),
        "stage26_7b_next_required_change": stage26_7b_summary.get("next_required_change"),
        "stage26_7_root": str(stage26_7_root),
        "stage26_7_status": stage26_7_summary.get("status"),
        "stage26_7_next_required_change": stage26_7_summary.get("next_required_change"),
        "coverage_denominator_mode": "main_coverable_cells",
        "coverage_denominator_source": "main_coverable_cells/v1",
        "main_final_coverage_delta": efficiency_audit["final_coverage_delta"],
        "main_coverage_auc_delta": efficiency_audit["coverage_auc_delta"],
        "main_coverage_per_100m_delta": efficiency_audit["coverage_per_100m_delta"],
        "hybrid_astar_path_cost_delta": efficiency_audit["hybrid_astar_path_cost_delta"],
        "hybrid_astar_path_cost_delta_is_diagnostic_only": True,
        "selected_action_changed_count": efficiency_audit["selected_action_changed_count"],
        "scenario_regression_count": efficiency_audit["scenario_regression_count"],
        "denominator_binding_missing_count": binding_audit["denominator_binding_missing_count"],
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "summary": str(output_root / SUMMARY_FILE),
        "stage26_7_config": str(stage26_7_config_path),
        "stage26_7_summary": str(output_root / STAGE26_7_SUMMARY_FILE),
        "main_coverable_efficiency_audit": str(output_root / EFFICIENCY_AUDIT_FILE),
        "denominator_binding_audit": str(output_root / BINDING_AUDIT_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / STAGE26_7_SUMMARY_FILE, stage26_7_summary)
    _write_json(output_root / EFFICIENCY_AUDIT_FILE, efficiency_audit)
    _write_json(output_root / BINDING_AUDIT_FILE, binding_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    if payload.get("stage_id") != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID}")
    config = dict(payload)
    config["stage26_7b_root"] = str(_resolve_path(Path(config.get("stage26_7b_root") or DEFAULT_STAGE26_7B_ROOT), repo_root))
    config["stage26_7_base_config"] = str(
        _resolve_path(Path(config.get("stage26_7_base_config") or stage26_7.DEFAULT_CONFIG), repo_root)
    )
    config["stage26_6_root"] = str(
        _resolve_path(Path(config.get("stage26_6_root") or stage26_7.DEFAULT_STAGE26_6_ROOT), repo_root)
    )
    config["stage26_0_root"] = str(
        _resolve_path(Path(config.get("stage26_0_root") or stage26_7.DEFAULT_STAGE26_0_ROOT), repo_root)
    )
    config["coverage_denominator_mode"] = str(config.get("coverage_denominator_mode", "main_coverable_cells"))
    config["coverage_denominator_source"] = str(config.get("coverage_denominator_source", "main_coverable_cells/v1"))
    if config["coverage_denominator_mode"] != "main_coverable_cells":
        raise ValueError("coverage_denominator_mode must be main_coverable_cells")
    if config["coverage_denominator_source"] != "main_coverable_cells/v1":
        raise ValueError("coverage_denominator_source must be main_coverable_cells/v1")
    config["hybrid_astar_candidate_eval_workers"] = int(config.get("hybrid_astar_candidate_eval_workers", 4))
    config["run_stage26_7"] = bool(config.get("run_stage26_7", True))
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0))
    for field in BOUNDARY_FIELDS:
        config[field] = bool(config.get(field, False))
    return config


def _build_stage26_7_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = _read_json(Path(config["stage26_7_base_config"]))
    cfg.update(
        {
            "stage26_6_root": config["stage26_6_root"],
            "stage26_0_root": config["stage26_0_root"],
            "coverage_denominator_mode": "main_coverable_cells",
            "coverage_denominator_source": "main_coverable_cells/v1",
            "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
            "run_stage26_chain": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    return cfg


def _input_rejections(stage26_7b_summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not stage26_7b_summary:
        return ["missing_stage26_7b_summary"]
    if stage26_7b_summary.get("stage_id") != stage26_7b.STAGE_ID:
        reasons.append("stage26_7b_wrong_stage_id")
    if stage26_7b_summary.get("status") != "passed":
        reasons.append("stage26_7b_not_passed")
    if stage26_7b_summary.get("next_required_change") != "rerun_stage26_7_path_efficiency_with_main_coverable_coverage":
        reasons.append("stage26_7b_route_mismatch")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _denominator_binding_audit(
    stage26_7_config: dict[str, Any],
    stage26_7_root: Path,
    stage26_7_summary: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    rows.append(_binding_row("stage26_7_config", stage26_7_config))
    if stage26_7_summary:
        rows.append(_binding_row("stage26_7_summary", stage26_7_summary))
    for path in sorted(stage26_7_root.glob("c*/xunce-stage26-7-stage26-3-config.json")):
        rows.append(_binding_row(str(path.relative_to(stage26_7_root)), _read_json_if_exists(path)))
    for path in sorted(stage26_7_root.glob("c*/s26_3/xunce-stage26-3-stage21-5-config.json")):
        rows.append(_binding_row(str(path.relative_to(stage26_7_root)), _read_json_if_exists(path)))
    for path in sorted(stage26_7_root.glob("c*/s26_3/xunce-stage26-3-high-fidelity-config.json")):
        rows.append(_binding_row(str(path.relative_to(stage26_7_root)), _read_json_if_exists(path)))
    missing = sum(1 for row in rows if row["coverage_denominator_bound"] is not True)
    return {
        "schema_version": BINDING_AUDIT_SCHEMA_VERSION,
        "coverage_denominator_mode": "main_coverable_cells",
        "coverage_denominator_source": "main_coverable_cells/v1",
        "binding_row_count": len(rows),
        "denominator_binding_missing_count": missing,
        "rows": rows,
    }


def _binding_row(label: str, payload: dict[str, Any]) -> dict[str, Any]:
    mode = payload.get("coverage_denominator_mode")
    source = payload.get("coverage_denominator_source")
    return {
        "label": label,
        "coverage_denominator_mode": mode,
        "coverage_denominator_source": source,
        "coverage_denominator_bound": mode == "main_coverable_cells" and source == "main_coverable_cells/v1",
    }


def _efficiency_audit(stage26_7_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": EFFICIENCY_AUDIT_SCHEMA_VERSION,
        "selected_action_changed_count": int(stage26_7_summary.get("best_combo_selected_action_changed_count") or 0),
        "final_coverage_delta": float(stage26_7_summary.get("best_combo_final_coverage_delta") or 0.0),
        "coverage_auc_delta": float(stage26_7_summary.get("best_combo_coverage_auc_delta") or 0.0),
        "coverage_per_100m_delta": float(stage26_7_summary.get("best_combo_coverage_per_100m_delta") or 0.0),
        "hybrid_astar_path_cost_delta": float(stage26_7_summary.get("best_combo_hybrid_astar_path_cost_delta") or 0.0),
        "hybrid_astar_path_cost_delta_is_diagnostic_only": True,
        "scenario_regression_count": int(stage26_7_summary.get("best_combo_scenario_regression_count") or 0),
    }


def _route(
    *,
    boundary_rejections: list[str],
    input_rejections: list[str],
    binding_audit: dict[str, Any],
    stage26_7_summary: dict[str, Any],
    efficiency_audit: dict[str, Any],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections:
        return ROUTE_INPUTS
    if int(binding_audit.get("denominator_binding_missing_count") or 0) > 0:
        return ROUTE_BINDING
    if _safety_regressed(stage26_7_summary):
        return ROUTE_SAFETY
    if int(efficiency_audit.get("scenario_regression_count") or 0) > 0:
        return ROUTE_TARGET_SCORE
    child_route = str(stage26_7_summary.get("next_required_change") or "")
    if child_route in CHILD_BLOCKING_ROUTES:
        return child_route
    if stage26_7_summary.get("status") == "passed" and stage26_7_summary.get("next_required_change") == ROUTE_MULTI_SEED:
        return ROUTE_MULTI_SEED
    if int(efficiency_audit.get("selected_action_changed_count") or 0) <= 0:
        return ROUTE_MARGIN
    if float(efficiency_audit.get("coverage_per_100m_delta") or 0.0) < 0.0:
        return ROUTE_TARGET_SCORE
    if (
        float(efficiency_audit.get("final_coverage_delta") or 0.0) <= 0.0
        or float(efficiency_audit.get("coverage_auc_delta") or 0.0) <= 0.0
    ):
        return ROUTE_CREDIT
    return str(stage26_7_summary.get("next_required_change") or ROUTE_CREDIT)


def _safety_regressed(summary: dict[str, Any]) -> bool:
    keys = (
        "hard_risk_violation_count",
        "mask_violation_count",
        "path_planning_failure_count",
        "open_grid_fallback_count",
    )
    return any(int(summary.get(key) or 0) > 0 for key in keys)


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.7C Main Coverable Coverage Efficiency Rerun",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- main_coverage_per_100m_delta: `{summary['main_coverage_per_100m_delta']}`",
            f"- hybrid_astar_path_cost_delta: `{summary['hybrid_astar_path_cost_delta']}`",
            "- note: Hybrid A* path cost delta is diagnostic only; success is judged by unit-distance main coverage.",
            "",
        ]
    )


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    try:
        return _read_json(path)
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
