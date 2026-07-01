from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_stage26_7c_main_coverable_coverage_efficiency_rerun as stage26_7c
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage26_7c_main_coverable_coverage_efficiency_rerun as stage26_7c


STAGE_ID = "xunce-stage26-7d-repair-synthetic-credit-sampler-continuous-theta-reachability"
CONFIG_SCHEMA_VERSION = "xunce-stage26-7d-repair-synthetic-credit-sampler-continuous-theta-reachability-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-7d-summary/v1"
AUDIT_SCHEMA_VERSION = "xunce-stage26-7d-collector-precheck-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-7d-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-7d-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_7d_repair_synthetic_credit_sampler_continuous_theta_reachability_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_7d_repair_synthetic_credit_sampler_continuous_theta_reachability_v1"
)
DEFAULT_STAGE26_7C_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_7c_main_coverable_coverage_efficiency_rerun_v1"
)

SUMMARY_FILE = "xunce-stage26-7d-summary.json"
COLLECTOR_AUDIT_FILE = "xunce-stage26-7d-collector-precheck-audit.json"
THETA_AUDIT_FILE = "xunce-stage26-7d-theta-reachability-proposal-audit.json"
BEHAVIOR_AUDIT_FILE = "xunce-stage26-7d-behavior-logprob-audit.json"
POST_EVAL_AUDIT_FILE = "xunce-stage26-7d-post-update-eval-audit.json"
ROUTING_FILE = "xunce-stage26-7d-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-7d-report.md"
MANIFEST_FILE = "xunce-stage26-7d-manifest.json"

ROUTE_INPUTS = "rerun_stage26_7d_required_inputs"
ROUTE_THETA = "repair_stage26_7d_theta_reachability_proposals"
ROUTE_BINDING = "repair_stage26_7d_theta_binding_contract"
ROUTE_LOGPROB = "repair_stage26_7d_behavior_theta_logprob_contract"
ROUTE_EXPAND = "expand_stage26_7d_collector_samples"
ROUTE_STABILITY = "repair_stage26_7d_credit_ppo_update_stability"
ROUTE_MARGIN = "calibrate_stage26_synthetic_discrete_margin_crossing_after_credit"
ROUTE_CREDIT = "repair_stage26_synthetic_credit_assignment"
ROUTE_MULTI_SEED = "run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot"
ROUTE_BOUNDARY = "resolve_stage26_7d_boundary_rejections"

BOUNDARY_FIELDS = (
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.7D continuous-theta reachability repair.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_7d_repair_synthetic_credit_sampler_continuous_theta_reachability(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_7d_repair_synthetic_credit_sampler_continuous_theta_reachability(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    chain_output_root = _chain_output_root(output_root, config, repo_root)

    stage26_7c_summary = _read_json_if_exists(Path(config["stage26_7c_root"]) / stage26_7c.SUMMARY_FILE)
    boundary_rejections = _boundary_rejections(config)
    input_rejections = _input_rejections(stage26_7c_summary)

    rerun_summary: dict[str, Any] = {}
    if not boundary_rejections and not input_rejections and bool(config.get("run_stage26_chain", True)):
        rerun_config = _build_stage26_7c_config(config, stage26_7c_summary)
        rerun_config_path = output_root / "xunce-stage26-7d-stage26-7c-config.json"
        _write_json(rerun_config_path, rerun_config)
        rerun_summary = stage26_7c.run_xunce_stage26_7c_main_coverable_coverage_efficiency_rerun(
            config_path=rerun_config_path,
            output_root=chain_output_root,
            repo_root=repo_root,
        )

    collector_audit = _collector_precheck_audit(chain_output_root, rerun_summary, int(config["min_trainable_transition_count"]))
    theta_audit = _theta_audit(collector_audit)
    behavior_audit = _behavior_audit(collector_audit)
    post_eval_audit = _post_eval_audit(rerun_summary)
    route = _route(
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        collector_audit=collector_audit,
        theta_audit=theta_audit,
        behavior_audit=behavior_audit,
        post_eval_audit=post_eval_audit,
        rerun_summary=rerun_summary,
    )
    status = "passed" if route == ROUTE_MULTI_SEED else "failed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "stage26_7c_root": config["stage26_7c_root"],
        "stage26_7c_status": stage26_7c_summary.get("status"),
        "stage26_7c_next_required_change": stage26_7c_summary.get("next_required_change"),
        "stage26_7c_rerun_output_root": str(chain_output_root),
        "rerun_stage26_7c_status": rerun_summary.get("status"),
        "rerun_stage26_7c_next_required_change": rerun_summary.get("next_required_change"),
        "trainable_transition_count": collector_audit["trainable_transition_count"],
        "synthetic_credit_target_selected_count": collector_audit["synthetic_credit_target_selected_count"],
        "selected_continuous_theta_hybrid_astar_unreachable_count": collector_audit[
            "selected_continuous_theta_hybrid_astar_unreachable_count"
        ],
        "behavior_theta_logprob_recomputable": behavior_audit["behavior_theta_logprob_recomputable"],
        "coverage_source": config["coverage_source"],
        "path_cost_source": config["path_cost_source"],
        "coverage_denominator_mode": config["coverage_denominator_mode"],
        "coverage_denominator_source": config["coverage_denominator_source"],
        "synthetic_source_kind": config["synthetic_source_kind"],
        "action_space_type": config["action_space_type"],
        "max_traversable_slope_deg": config["max_traversable_slope_deg"],
        "hybrid_astar_candidate_eval_workers": config["hybrid_astar_candidate_eval_workers"],
        "main_coverage_per_100m_delta": post_eval_audit["main_coverage_per_100m_delta"],
        "hybrid_astar_path_cost_delta_is_diagnostic_only": True,
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
        "collector_precheck_audit": str(output_root / COLLECTOR_AUDIT_FILE),
        "theta_reachability_proposal_audit": str(output_root / THETA_AUDIT_FILE),
        "behavior_logprob_audit": str(output_root / BEHAVIOR_AUDIT_FILE),
        "post_update_eval_audit": str(output_root / POST_EVAL_AUDIT_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
        "stage26_7c_rerun_output_root": str(chain_output_root),
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / COLLECTOR_AUDIT_FILE, collector_audit)
    _write_json(output_root / THETA_AUDIT_FILE, theta_audit)
    _write_json(output_root / BEHAVIOR_AUDIT_FILE, behavior_audit)
    _write_json(output_root / POST_EVAL_AUDIT_FILE, post_eval_audit)
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
    config["stage26_7c_root"] = str(_resolve_path(Path(config.get("stage26_7c_root") or DEFAULT_STAGE26_7C_ROOT), repo_root))
    config["run_stage26_chain"] = bool(config.get("run_stage26_chain", True))
    config["min_trainable_transition_count"] = int(config.get("min_trainable_transition_count", 12))
    config["coverage_source"] = str(config.get("coverage_source") or "endpoint_theta_slope_obstacle_los/v1")
    config["path_cost_source"] = str(config.get("path_cost_source") or "hybrid_astar_pose_path/v1")
    config["coverage_denominator_mode"] = str(config.get("coverage_denominator_mode") or "main_coverable_cells")
    config["coverage_denominator_source"] = str(config.get("coverage_denominator_source") or "main_coverable_cells/v1")
    config["synthetic_source_kind"] = str(config.get("synthetic_source_kind") or "synthetic_terrain_obstacle_proxy/v1")
    config["action_space_type"] = str(config.get("action_space_type") or "hybrid_discrete_xy_continuous_theta/v1")
    config["max_traversable_slope_deg"] = float(config.get("max_traversable_slope_deg", 30.0))
    config["hybrid_astar_candidate_eval_workers"] = int(config.get("hybrid_astar_candidate_eval_workers", 4))
    if config.get("stage26_7c_rerun_output_root"):
        config["stage26_7c_rerun_output_root"] = str(
            _resolve_path(Path(str(config["stage26_7c_rerun_output_root"])), repo_root)
        )
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0))
    for field in BOUNDARY_FIELDS:
        config[field] = bool(config.get(field, False))
    return config


def _chain_output_root(output_root: Path, config: dict[str, Any], repo_root: Path) -> Path:
    configured = config.get("stage26_7c_rerun_output_root")
    if configured:
        root = _resolve_path(Path(str(configured)), repo_root)
    else:
        root = output_root / "r"
        if _worst_case_stage21_1_summary_path_length(root) >= 240:
            root = output_root.parent / "s26_7d_chain_v1"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _worst_case_stage21_1_summary_path_length(root: Path) -> int:
    probe = root / "r" / "c0" / "s26_1" / "s21_1" / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json"
    return len(str(probe))


def _build_stage26_7c_config(config: dict[str, Any], stage26_7c_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": stage26_7c.CONFIG_SCHEMA_VERSION,
        "stage_id": stage26_7c.STAGE_ID,
        "stage26_7b_root": stage26_7c_summary.get("stage26_7b_root"),
        "stage26_6_root": _stage26_6_root_from_7c(stage26_7c_summary),
        "coverage_denominator_mode": "main_coverable_cells",
        "coverage_denominator_source": "main_coverable_cells/v1",
        "hybrid_astar_candidate_eval_workers": int(config["hybrid_astar_candidate_eval_workers"]),
        "run_stage26_7": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _stage26_6_root_from_7c(stage26_7c_summary: dict[str, Any]) -> str | None:
    stage26_7_root = Path(str(stage26_7c_summary.get("stage26_7_root") or ""))
    cfg = _read_json_if_exists(stage26_7_root.parent / stage26_7c.STAGE26_7_CONFIG_FILE)
    return cfg.get("stage26_6_root")


def _input_rejections(stage26_7c_summary: dict[str, Any]) -> list[str]:
    if not stage26_7c_summary:
        return ["missing_stage26_7c_summary"]
    reasons: list[str] = []
    if stage26_7c_summary.get("stage_id") != stage26_7c.STAGE_ID:
        reasons.append("stage26_7c_wrong_stage_id")
    if stage26_7c_summary.get("next_required_change") != "repair_stage26_7_path_efficiency_credit_sampler":
        reasons.append("stage26_7c_route_mismatch")
    return reasons


def _collector_precheck_audit(root: Path, summary: dict[str, Any], minimum: int) -> dict[str, Any]:
    stage26_7_root = Path(str(summary.get("stage26_7_root") or root / "r"))
    trainable_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    for path in sorted(stage26_7_root.glob("c*/s26_1/s21_1/xunce-stage21-1-ppo-trainable-batch.jsonl")):
        trainable_rows.extend(_read_jsonl(path))
    for path in sorted(stage26_7_root.glob("c*/s26_1/s21_1/xunce-stage21-1-rejection-report.jsonl")):
        rejection_rows.extend(_read_jsonl(path))
    selected = [row for row in trainable_rows if row.get("synthetic_credit_target_selected") is True]
    behavior_rows = [row for row in trainable_rows if row.get("behavior_policy_id") == "synthetic_credit_mixture_policy/v1"]
    unreachable = [
        row
        for row in rejection_rows
        if row.get("terminal_reason") == "selected_continuous_theta_hybrid_astar_unreachable"
    ]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "trainable_transition_count": len(trainable_rows),
        "min_trainable_transition_count": minimum,
        "synthetic_credit_target_selected_count": len(selected),
        "selected_continuous_theta_hybrid_astar_unreachable_count": len(unreachable),
        "behavior_policy_row_count": len(behavior_rows),
        "behavior_theta_logprob_mismatch_count": _behavior_theta_logprob_mismatch_count(behavior_rows),
    }


def _theta_audit(collector_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage26-7d-theta-reachability-proposal-audit/v1",
        "selected_continuous_theta_hybrid_astar_unreachable_count": collector_audit[
            "selected_continuous_theta_hybrid_astar_unreachable_count"
        ],
        "theta_reachability_contract_satisfied": collector_audit[
            "selected_continuous_theta_hybrid_astar_unreachable_count"
        ]
        == 0,
    }


def _behavior_audit(collector_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage26-7d-behavior-logprob-audit/v1",
        "behavior_policy_row_count": collector_audit["behavior_policy_row_count"],
        "behavior_theta_logprob_mismatch_count": collector_audit["behavior_theta_logprob_mismatch_count"],
        "behavior_theta_logprob_recomputable": (
            collector_audit["behavior_policy_row_count"] > 0
            and collector_audit["behavior_theta_logprob_mismatch_count"] == 0
        ),
    }


def _behavior_theta_logprob_mismatch_count(rows: list[dict[str, Any]]) -> int:
    mismatch = 0
    for row in rows:
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        merged = dict(info)
        merged.update(row)
        if merged.get("synthetic_credit_theta_policy_id") != "reachability_theta_proposal_mixture/v1":
            mismatch += 1
            continue
        proposals = merged.get("synthetic_credit_theta_proposals_deg")
        if not isinstance(proposals, list) or not proposals:
            mismatch += 1
            continue
        selected_index = _int_or_none(merged.get("synthetic_credit_theta_selected_proposal_index"))
        if selected_index is None or selected_index < 0 or selected_index >= len(proposals):
            mismatch += 1
            continue
        reachable_count = _int_or_none(merged.get("synthetic_credit_theta_reachable_proposal_count"))
        if reachable_count is None or reachable_count <= 0 or reachable_count > len(proposals):
            mismatch += 1
            continue
        old_behavior_theta = _finite(merged.get("old_behavior_theta_log_prob"))
        expected = -math.log(float(reachable_count))
        if old_behavior_theta is None or abs(float(old_behavior_theta) - expected) > 1.0e-6:
            mismatch += 1
            continue
        selected_theta = _finite(merged.get("selected_theta_deg", merged.get("synthetic_credit_target_theta_deg")))
        proposal_theta = _finite(proposals[selected_index])
        if selected_theta is None or proposal_theta is None or _angle_delta_abs_deg(selected_theta, proposal_theta) > 1.0e-6:
            mismatch += 1
    return mismatch


def _post_eval_audit(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage26-7d-post-update-eval-audit/v1",
        "main_final_coverage_delta": float(summary.get("main_final_coverage_delta") or 0.0),
        "main_coverage_auc_delta": float(summary.get("main_coverage_auc_delta") or 0.0),
        "main_coverage_per_100m_delta": float(summary.get("main_coverage_per_100m_delta") or 0.0),
        "selected_action_changed_count": int(summary.get("selected_action_changed_count") or 0),
        "scenario_regression_count": int(summary.get("scenario_regression_count") or 0),
    }


def _route(
    *,
    boundary_rejections: list[str],
    input_rejections: list[str],
    collector_audit: dict[str, Any],
    theta_audit: dict[str, Any],
    behavior_audit: dict[str, Any],
    post_eval_audit: dict[str, Any],
    rerun_summary: dict[str, Any],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections:
        return ROUTE_INPUTS
    if theta_audit["theta_reachability_contract_satisfied"] is not True:
        return ROUTE_THETA
    if collector_audit["trainable_transition_count"] < collector_audit["min_trainable_transition_count"]:
        return ROUTE_EXPAND
    if collector_audit["synthetic_credit_target_selected_count"] <= 0:
        return ROUTE_THETA
    if behavior_audit["behavior_theta_logprob_recomputable"] is not True:
        return ROUTE_LOGPROB
    if rerun_summary.get("next_required_change") in {ROUTE_STABILITY, ROUTE_MARGIN, ROUTE_CREDIT, ROUTE_MULTI_SEED}:
        return str(rerun_summary["next_required_change"])
    if post_eval_audit["scenario_regression_count"] > 0 or post_eval_audit["main_coverage_per_100m_delta"] < 0.0:
        return ROUTE_CREDIT
    return str(rerun_summary.get("next_required_change") or ROUTE_CREDIT)


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _angle_delta_abs_deg(a: float, b: float) -> float:
    return abs(((float(a) - float(b) + 180.0) % 360.0) - 180.0)


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.7D Continuous Theta Reachability Repair",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- trainable_transition_count: `{summary['trainable_transition_count']}`",
            f"- selected_continuous_theta_hybrid_astar_unreachable_count: `{summary['selected_continuous_theta_hybrid_astar_unreachable_count']}`",
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
