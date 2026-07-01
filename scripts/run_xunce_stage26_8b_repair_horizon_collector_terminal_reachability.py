from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_stage26_1_synthetic_terrain_collector_smoke as stage26_1
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage26_1_synthetic_terrain_collector_smoke as stage26_1


STAGE_ID = "xunce-stage26-8b-repair-horizon-collector-terminal-reachability"
CONFIG_SCHEMA_VERSION = "xunce-stage26-8b-repair-horizon-collector-terminal-reachability-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-8b-summary/v1"
COLLECTOR_AUDIT_SCHEMA_VERSION = "xunce-stage26-8b-h12-vs-h16-collector-audit/v1"
TERMINAL_AUDIT_SCHEMA_VERSION = "xunce-stage26-8b-terminal-reachability-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-8b-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-8b-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_8b_repair_horizon_collector_terminal_reachability_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8b_repair_horizon_collector_terminal_reachability_v1"
)
DEFAULT_STAGE26_8A_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8a_expand_seed_or_horizon_budget_v1"
)

SUMMARY_FILE = "xunce-stage26-8b-summary.json"
COLLECTOR_AUDIT_FILE = "xunce-stage26-8b-h12-vs-h16-collector-audit.json"
TERMINAL_AUDIT_FILE = "xunce-stage26-8b-terminal-reachability-audit.json"
RERUN_STAGE26_1_SUMMARY_FILE = "xunce-stage26-8b-rerun-stage26-1-summary.json"
ROUTING_FILE = "xunce-stage26-8b-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-8b-report.md"
MANIFEST_FILE = "xunce-stage26-8b-manifest.json"

ROUTE_INPUTS = "rerun_stage26_8b_required_inputs"
ROUTE_MAP_BINDING = "repair_stage26_1_collector_synthetic_map_binding"
ROUTE_TERMINAL_DIAGNOSTICS = "repair_stage26_8b_terminal_sampling_mask_diagnostics"
ROUTE_EXPAND = "expand_stage26_8b_horizon_collector_samples"
ROUTE_REWARD_BATCH = "repair_stage26_8b_reward_batch_after_terminal"
ROUTE_RESUME = "resume_stage26_8a_from_h16_h20"
ROUTE_BOUNDARY = "resolve_stage26_8b_boundary_rejections"

BOUNDARY_FIELDS = (
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair Stage26.8A H16 collector terminal reachability.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_8b_repair_horizon_collector_terminal_reachability(
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


def run_xunce_stage26_8b_repair_horizon_collector_terminal_reachability(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(config_path, repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage26_8a_root = Path(config["stage26_8a_root"])
    stage26_8a_summary = _read_json_if_exists(stage26_8a_root / "xunce-stage26-8a-summary.json")
    boundary_rejections = _boundary_rejections(config)
    input_rejections = _input_rejections(config, stage26_8a_summary)

    h12_stage26_1_root = Path(config["h12_stage26_1_root"])
    h16_prior_stage26_1_root = Path(config["h16_prior_stage26_1_root"])
    h16_stage26_1_config = Path(config["h16_stage26_1_config"])

    rerun_summary: dict[str, Any] = {}
    rerun_root = output_root / "r" / "s26_1"
    if not boundary_rejections and not input_rejections and bool(config["run_repair_chain"]):
        rerun_summary = stage26_1.run_xunce_stage26_1_synthetic_terrain_collector_smoke(
            config_path=h16_stage26_1_config,
            output_root=rerun_root,
            repo_root=repo_root,
        )

    collector_audit = _collector_audit(
        h12_stage26_1_root=h12_stage26_1_root,
        h16_prior_stage26_1_root=h16_prior_stage26_1_root,
        h16_repaired_stage26_1_root=rerun_root,
    )
    terminal_audit = _terminal_reachability_audit(
        h16_prior_stage26_1_root=h16_prior_stage26_1_root,
        h16_repaired_stage26_1_root=rerun_root,
    )
    route = _route(
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        collector_audit=collector_audit,
        terminal_audit=terminal_audit,
        rerun_summary=rerun_summary,
    )
    status = "passed" if route == ROUTE_RESUME else "failed"
    blocking = _blocking_reasons(
        route=route,
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        collector_audit=collector_audit,
        terminal_audit=terminal_audit,
        rerun_summary=rerun_summary,
    )

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "blocking_reason_codes": blocking,
        "stage26_8a_root": str(stage26_8a_root),
        "stage26_8a_status": stage26_8a_summary.get("status"),
        "stage26_8a_next_required_change": stage26_8a_summary.get("next_required_change"),
        "h12_stage26_1_root": str(h12_stage26_1_root),
        "h16_prior_stage26_1_root": str(h16_prior_stage26_1_root),
        "h16_repaired_stage26_1_root": str(rerun_root),
        "h16_repaired_status": rerun_summary.get("status"),
        "h16_repaired_next_required_change": rerun_summary.get("next_required_change"),
        "h16_repaired_transition_count": rerun_summary.get("transition_count"),
        "h16_repaired_reward_row_count": rerun_summary.get("reward_row_count"),
        "h16_repaired_batch_row_count": rerun_summary.get("batch_row_count"),
        "h16_repaired_stage21_1_terminal_count": rerun_summary.get(
            "stage21_1_no_hybrid_reachable_candidate_terminal_count"
        ),
        "synthetic_terrain_hash": rerun_summary.get("synthetic_terrain_hash")
        or collector_audit.get("h16_prior_synthetic_terrain_hash"),
        "synthetic_source_kind": rerun_summary.get("synthetic_source_kind")
        or collector_audit.get("h16_prior_synthetic_source_kind"),
        "coverage_source": rerun_summary.get("coverage_source") or "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": rerun_summary.get("path_cost_source") or "hybrid_astar_pose_path/v1",
        "max_traversable_slope_deg": rerun_summary.get("max_traversable_slope_deg") or 30.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "summary": str(output_root / SUMMARY_FILE),
    }
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "artifacts": {
            "summary": str(output_root / SUMMARY_FILE),
            "h12_vs_h16_collector_audit": str(output_root / COLLECTOR_AUDIT_FILE),
            "terminal_reachability_audit": str(output_root / TERMINAL_AUDIT_FILE),
            "rerun_stage26_1_summary": str(output_root / RERUN_STAGE26_1_SUMMARY_FILE),
            "next_stage_routing": str(output_root / ROUTING_FILE),
            "report": str(output_root / REPORT_FILE),
        },
    }

    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / COLLECTOR_AUDIT_FILE, collector_audit)
    _write_json(output_root / TERMINAL_AUDIT_FILE, terminal_audit)
    _write_json(output_root / RERUN_STAGE26_1_SUMMARY_FILE, rerun_summary)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary, collector_audit, terminal_audit), encoding="utf-8")
    return summary


def _collector_audit(
    *,
    h12_stage26_1_root: Path,
    h16_prior_stage26_1_root: Path,
    h16_repaired_stage26_1_root: Path,
) -> dict[str, Any]:
    h12 = _read_json_if_exists(h12_stage26_1_root / stage26_1.SUMMARY_FILE)
    h16_prior = _read_json_if_exists(h16_prior_stage26_1_root / stage26_1.SUMMARY_FILE)
    h16_prior_contract = _read_json_if_exists(h16_prior_stage26_1_root / stage26_1.AUDIT_FILE)
    h16_repaired = _read_json_if_exists(h16_repaired_stage26_1_root / stage26_1.SUMMARY_FILE)
    h16_repaired_contract = _read_json_if_exists(h16_repaired_stage26_1_root / stage26_1.AUDIT_FILE)
    return {
        "schema_version": COLLECTOR_AUDIT_SCHEMA_VERSION,
        "h12_status": h12.get("status"),
        "h12_transition_count": h12.get("transition_count"),
        "h12_reward_row_count": h12.get("reward_row_count"),
        "h12_batch_row_count": h12.get("batch_row_count"),
        "h16_prior_status": h16_prior.get("status"),
        "h16_prior_next_required_change": h16_prior.get("next_required_change"),
        "h16_prior_stage21_1_status": h16_prior.get("stage21_1_status"),
        "h16_prior_stage21_1_source_roi_expansion_root_match": h16_prior_contract.get(
            "stage21_1_source_roi_expansion_root_match"
        ),
        "h16_prior_synthetic_transition_contract_missing_count": h16_prior_contract.get(
            "synthetic_transition_contract_missing_count"
        ),
        "h16_prior_synthetic_terrain_hash": h16_prior.get("synthetic_terrain_hash"),
        "h16_prior_synthetic_source_kind": h16_prior.get("synthetic_source_kind"),
        "h16_repaired_status": h16_repaired.get("status"),
        "h16_repaired_next_required_change": h16_repaired.get("next_required_change"),
        "h16_repaired_transition_count": h16_repaired.get("transition_count"),
        "h16_repaired_reward_row_count": h16_repaired.get("reward_row_count"),
        "h16_repaired_batch_row_count": h16_repaired.get("batch_row_count"),
        "h16_repaired_transition_missing_reward_count": h16_repaired_contract.get(
            "transition_missing_reward_count"
        ),
        "h16_repaired_transition_missing_batch_count": h16_repaired_contract.get(
            "transition_missing_batch_count"
        ),
        "h16_repaired_synthetic_transition_contract_missing_count": h16_repaired_contract.get(
            "synthetic_transition_contract_missing_count"
        ),
        "h16_repaired_synthetic_reward_provenance_missing_count": h16_repaired_contract.get(
            "synthetic_reward_provenance_missing_count"
        ),
        "h16_repaired_synthetic_batch_contract_missing_count": h16_repaired_contract.get(
            "synthetic_batch_contract_missing_count"
        ),
    }


def _terminal_reachability_audit(
    *,
    h16_prior_stage26_1_root: Path,
    h16_repaired_stage26_1_root: Path,
) -> dict[str, Any]:
    prior_stage21_1 = _read_json_if_exists(h16_prior_stage26_1_root / "s21_1" / stage26_1.stage21_1.SUMMARY_FILE)
    repaired_stage21_1 = _read_json_if_exists(
        h16_repaired_stage26_1_root / "s21_1" / stage26_1.stage21_1.SUMMARY_FILE
    )
    prior_rejections = _read_jsonl_if_exists(h16_prior_stage26_1_root / "s21_1" / stage26_1.stage21_1.REJECTION_FILE)
    repaired_rejections = _read_jsonl_if_exists(
        h16_repaired_stage26_1_root / "s21_1" / stage26_1.stage21_1.REJECTION_FILE
    )
    return {
        "schema_version": TERMINAL_AUDIT_SCHEMA_VERSION,
        "prior_stage21_1_status": prior_stage21_1.get("status"),
        "prior_stage21_1_blocking_reason_codes": prior_stage21_1.get("blocking_reason_codes", []),
        "prior_no_hybrid_reachable_candidate_terminal_count": _count_rejections(
            prior_rejections, "no_hybrid_reachable_candidate_terminal"
        ),
        "prior_no_hard_risk_clean_action_count": _count_rejections(prior_rejections, "no_hard_risk_clean_action"),
        "repaired_stage21_1_status": repaired_stage21_1.get("status"),
        "repaired_stage21_1_next_required_change": repaired_stage21_1.get("next_required_change"),
        "repaired_stage21_1_blocking_reason_codes": repaired_stage21_1.get("blocking_reason_codes", []),
        "repaired_trainable_transition_count": repaired_stage21_1.get("trainable_transition_count"),
        "repaired_min_trainable_transition_count": repaired_stage21_1.get("min_trainable_transition_count"),
        "repaired_no_hybrid_reachable_candidate_terminal_count": repaired_stage21_1.get(
            "no_hybrid_reachable_candidate_terminal_count"
        )
        or _count_rejections(repaired_rejections, "no_hybrid_reachable_candidate_terminal"),
        "repaired_terminal_rejection_has_hybrid_mask_count": sum(
            1
            for row in repaired_rejections
            if row.get("reason") == "no_hybrid_reachable_candidate_terminal"
            and isinstance(row.get("hybrid_astar_reachable_mask"), list)
        ),
        "repaired_terminal_rejection_missing_mask_count": sum(
            1
            for row in repaired_rejections
            if row.get("reason") == "no_hybrid_reachable_candidate_terminal"
            and not isinstance(row.get("hybrid_astar_reachable_mask"), list)
        ),
    }


def _route(
    *,
    boundary_rejections: list[str],
    input_rejections: list[str],
    collector_audit: dict[str, Any],
    terminal_audit: dict[str, Any],
    rerun_summary: dict[str, Any],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections:
        return ROUTE_INPUTS
    if collector_audit.get("h16_prior_stage21_1_source_roi_expansion_root_match") is not True:
        return ROUTE_MAP_BINDING
    if int(collector_audit.get("h16_prior_synthetic_transition_contract_missing_count") or 0) > 0:
        return ROUTE_MAP_BINDING
    if not rerun_summary:
        return ROUTE_TERMINAL_DIAGNOSTICS
    if rerun_summary.get("status") == "passed":
        return ROUTE_RESUME
    if int(rerun_summary.get("transition_count") or 0) <= 0 or int(rerun_summary.get("transition_count") or 0) < int(
        rerun_summary.get("min_trainable_transition_count") or 0
    ):
        return ROUTE_EXPAND
    if int(terminal_audit.get("repaired_terminal_rejection_missing_mask_count") or 0) > 0:
        return ROUTE_TERMINAL_DIAGNOSTICS
    if int(rerun_summary.get("transition_missing_reward_count") or 0) > 0 or int(
        rerun_summary.get("transition_missing_batch_count") or 0
    ) > 0:
        return ROUTE_REWARD_BATCH
    return ROUTE_TERMINAL_DIAGNOSTICS


def _blocking_reasons(
    *,
    route: str,
    boundary_rejections: list[str],
    input_rejections: list[str],
    collector_audit: dict[str, Any],
    terminal_audit: dict[str, Any],
    rerun_summary: dict[str, Any],
) -> list[str]:
    reasons = [*boundary_rejections, *input_rejections]
    if route == ROUTE_MAP_BINDING:
        if collector_audit.get("h16_prior_stage21_1_source_roi_expansion_root_match") is not True:
            reasons.append("h16_source_roi_expansion_root_mismatch")
        if int(collector_audit.get("h16_prior_synthetic_transition_contract_missing_count") or 0) > 0:
            reasons.append("h16_synthetic_transition_contract_missing")
    if route == ROUTE_TERMINAL_DIAGNOSTICS:
        reasons.append("terminal_sampling_mask_diagnostics_incomplete")
    if route == ROUTE_EXPAND:
        reasons.append("h16_trainable_transition_count_below_minimum")
    if route == ROUTE_REWARD_BATCH:
        reasons.append("h16_reward_or_batch_missing_after_terminal_repair")
    if rerun_summary.get("blocking_reason_codes"):
        reasons.extend(str(reason) for reason in rerun_summary.get("blocking_reason_codes") or [])
    if int(terminal_audit.get("repaired_terminal_rejection_missing_mask_count") or 0) > 0:
        reasons.append("terminal_rejection_hybrid_mask_missing")
    return _unique(reasons)


def _input_rejections(config: dict[str, Any], stage26_8a_summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not stage26_8a_summary:
        reasons.append("missing_stage26_8a_summary")
    else:
        if stage26_8a_summary.get("schema_version") != "xunce-stage26-8a-summary/v1":
            reasons.append("stage26_8a_schema_version_mismatch")
        if stage26_8a_summary.get("stage_id") != "xunce-stage26-8a-expand-seed-or-horizon-budget":
            reasons.append("stage26_8a_stage_id_mismatch")
        if stage26_8a_summary.get("status") != "failed":
            reasons.append("stage26_8a_status_not_failed")
        if stage26_8a_summary.get("next_required_change") != "repair_stage26_8a_horizon_eval_binding_or_safety":
            reasons.append("stage26_8a_route_not_horizon_binding_or_safety")
    for key in ("h12_stage26_1_root", "h16_prior_stage26_1_root"):
        if not Path(config[key]).is_dir():
            reasons.append(f"{key}_missing")
    if not Path(config["h16_stage26_1_config"]).is_file():
        reasons.append("h16_stage26_1_config_missing")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is not False]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    config.setdefault("stage26_8a_root", DEFAULT_STAGE26_8A_ROOT)
    stage26_8a_root = _resolve_path(Path(config["stage26_8a_root"]), repo_root)
    config["stage26_8a_root"] = str(stage26_8a_root)
    config.setdefault("h12_stage26_1_root", str(stage26_8a_root / "h12" / "s26_8" / "s0" / "s26_1"))
    config.setdefault("h16_prior_stage26_1_root", str(stage26_8a_root / "h16" / "s26_8" / "s0" / "s26_1"))
    config.setdefault(
        "h16_stage26_1_config",
        str(stage26_8a_root / "h16" / "s26_8" / "s0" / "xunce-stage26-8-stage26-1-config.json"),
    )
    for key in ("h12_stage26_1_root", "h16_prior_stage26_1_root", "h16_stage26_1_config"):
        config[key] = str(_resolve_path(Path(config[key]), repo_root))
    config.setdefault("run_repair_chain", True)
    config["run_repair_chain"] = bool(config["run_repair_chain"])
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0))
    for field in BOUNDARY_FIELDS:
        config.setdefault(field, False)
    return config


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _count_rejections(rows: list[dict[str, Any]], reason: str) -> int:
    return sum(1 for row in rows if row.get("reason") == reason)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _render_report(summary: dict[str, Any], collector_audit: dict[str, Any], terminal_audit: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.8B Horizon Collector Terminal Reachability Repair",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- H16 repaired Stage26.1 status: `{summary.get('h16_repaired_status')}`",
            f"- H16 repaired transition/reward/batch: `{summary.get('h16_repaired_transition_count')}` / `{summary.get('h16_repaired_reward_row_count')}` / `{summary.get('h16_repaired_batch_row_count')}`",
            f"- prior source root match: `{collector_audit.get('h16_prior_stage21_1_source_roi_expansion_root_match')}`",
            f"- repaired no-hybrid-reachable terminal count: `{terminal_audit.get('repaired_no_hybrid_reachable_candidate_terminal_count')}`",
            "",
            "This stage only repairs and audits collector terminal reachability. It does not run PPO update, publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
        ]
    ) + "\n"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
