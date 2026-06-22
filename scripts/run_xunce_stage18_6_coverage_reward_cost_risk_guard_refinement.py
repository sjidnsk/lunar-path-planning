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

MODEL_EXPLORER_SRC = SCRIPT_DIR.parent / "model-explorer" / "src"
if str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))

from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
from global_99_governance_common import global_99_boundary_defaults
from model_explorer.policy.canonical_reward import load_canonical_reward_profile
from xunce_stage18_guard_thresholds import stage18_guard_thresholds
from xunce_candidate_guard_semantics import (
    CANDIDATE_GUARD_MODE,
    candidate_guard_failure_reasons,
    candidate_guard_observation,
    paired_decision_key,
    selected_baseline_candidate,
)


CONFIG_SCHEMA_VERSION = "xunce-stage18-6-coverage-reward-cost-risk-guard-refinement-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage18-6-guard-refinement-summary/v1"
SELECTED_ACTION_ROW_SCHEMA_VERSION = "xunce-stage18-6-selected-action-guard-replay-row/v1"
SCENARIO_ROW_SCHEMA_VERSION = "xunce-stage18-6-scenario-guard-replay-row/v1"
CALIBRATION_ROW_SCHEMA_VERSION = "xunce-stage18-6-calibration-grid-row/v1"
CANDIDATE_READINESS_SCHEMA_VERSION = "xunce-stage18-6-candidate-metric-readiness/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage18-6-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage18-6-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage18_6_coverage_reward_cost_risk_guard_refinement_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement_v1"
DEFAULT_CANONICAL_PROFILE = "configs/xunce_canonical_reward_guard_profile_v2.json"

STAGE18_5_SUMMARY_FILE = "xunce-stage18-5-evidence-attribution-summary.json"
STAGE18_5_GUARD_FILE = "xunce-stage18-5-guard-evaluation.json"
STAGE18_5_ROUTING_FILE = "xunce-stage18-5-next-stage-routing.json"
STAGE18_5_PAIRED_FILE = "xunce-stage18-5-paired-decision-summary.json"
STAGE18_5_REGRESSION_FILE = "xunce-stage18-5-regression-attribution.jsonl"

COVERAGE_SUMMARY_FILE = "xunce-exploration-coverage-comparison-summary.json"
COVERAGE_AGGREGATE_FILE = "xunce-exploration-coverage-comparison-aggregate.json"
COVERAGE_PAIRS_FILE = "xunce-exploration-coverage-comparison-pairs.jsonl"
COVERAGE_EPISODES_FILE = "xunce-exploration-coverage-episodes.jsonl"
PAIRED_DECISION_AUDIT_FILE = "xunce-exploration-coverage-paired-decision-audit.jsonl"
CANDIDATE_METRIC_AUDIT_FILE = "xunce-exploration-coverage-candidate-metric-audit.jsonl"

SUMMARY_FILE = "xunce-stage18-6-guard-refinement-summary.json"
SELECTED_ACTION_FILE = "xunce-stage18-6-selected-action-guard-replay.jsonl"
SCENARIO_FILE = "xunce-stage18-6-scenario-guard-replay.jsonl"
CALIBRATION_FILE = "xunce-stage18-6-calibration-grid.jsonl"
CANDIDATE_READINESS_FILE = "xunce-stage18-6-candidate-metric-readiness.json"
ROUTING_FILE = "xunce-stage18-6-next-stage-routing.json"
REPORT_FILE = "xunce-stage18-6-guard-refinement-report.md"
MANIFEST_FILE = "xunce-stage18-6-manifest.json"

BOUNDARY_NEXT_REQUIRED_CHANGE = "resolve_stage18_6_guard_refinement_boundary_rejections"
STAGE18_5_RERUN_NEXT_REQUIRED_CHANGE = "rerun_xunce_stage18_5_evidence_attribution_review"
STAGE18_4_RERUN_NEXT_REQUIRED_CHANGE = "rerun_xunce_stage18_4e_coverage_comparison_with_required_artifacts"
CANDIDATE_AUDIT_NEXT_REQUIRED_CHANGE = "rerun_stage18_4e_with_candidate_metric_audit"
CANDIDATE_EXPANSION_NEXT_REQUIRED_CHANGE = "expand_candidate_generation_roi_complexity"
GUARD_REFINEMENT_NEXT_REQUIRED_CHANGE = "refine_coverage_reward_and_cost_guard"
STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE = "prepare_stage19_evaluator_critic_preflight"

STAGE18_5_SUMMARY_SCHEMA_VERSION = "xunce-stage18-5-evidence-attribution-summary/v1"
STAGE18_5_GUARD_SCHEMA_VERSION = "xunce-stage18-5-guard-evaluation/v1"
STAGE18_5_ROUTING_SCHEMA_VERSION = "xunce-stage18-5-next-stage-routing/v1"
STAGE18_4_SUMMARY_SCHEMA_VERSION = "xunce-exploration-coverage-comparison-summary/v1"

BOUNDARY_FIELDS = tuple(global_99_boundary_defaults()) + (
    "default_policy_replacement_approved",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
    "runs_new_ppo_update",
    "real_world_release_approved",
    "real_world_performance_claimed",
)
LEGACY_V1_COMPONENT_FIELDS = {
    "path_cost_excess",
    "risk_excess",
    "risk_cost_exposure_penalty",
    "path_cost_efficiency_penalty",
    "risk_efficiency_penalty",
    "efficiency_regression_penalty",
    "controlled_regression_penalty",
    "cost_efficiency_reward_components",
}
CANDIDATE_REQUIRED_FIELDS = {
    "schema_version",
    "scenario_id",
    "split",
    "roi_group",
    "step_index",
    "current_cell",
    "covered_cells_hash",
    "candidate_set_id",
    "candidate_set_hash",
    "candidate_index",
    "candidate_cell",
    "action_mask_valid",
    "expected_new_coverage_cell_count",
    "roi_weighted_coverage_delta",
    "path_cost",
    "risk",
    "risk_source",
    "coverage_gain_per_path_cost",
    "profile_id",
    "profile_version",
    "profile_hash",
}
TOLERANCE = 1.0e-12


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Xunce Stage 18.6 offline reward/cost-risk guard refinement.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--stage18-5-attribution-root")
    parser.add_argument("--coverage-comparison-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    overrides = {
        "stage18_5_attribution_root": args.stage18_5_attribution_root,
        "coverage_comparison_root": args.coverage_comparison_root,
    }
    try:
        summary = run_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement(
            config_path=resolve_path(Path(args.config), repo_root).resolve(),
            output_root=resolve_path(Path(args.output_root), repo_root).resolve(),
            repo_root=repo_root,
            config_overrides={key: value for key, value in overrides.items() if value is not None},
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "guard_refinement_passed": summary["guard_refinement_passed"],
                "counterfactual_reselection_claimed": summary["counterfactual_reselection_claimed"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = _load_config(config_path, repo_root, config_overrides=config_overrides)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_root)

    stage18_5, coverage, required_artifact_reasons = _load_evidence(config)
    boundary_reasons = _boundary_reasons(config, stage18_5.get("summary", {}), coverage.get("summary", {}))
    lineage_reasons = _stage18_5_lineage_reasons(config, stage18_5)
    profile_reasons, profile_route = _profile_lineage_reasons(config, stage18_5, coverage)
    selected_rows, selected_summary = _selected_action_guard_replay(coverage.get("paired_decision_rows", []), config)
    scenario_rows, scenario_summary = _scenario_guard_replay(coverage.get("pair_rows", []), config)
    candidate_readiness = _candidate_metric_readiness(config, coverage)
    guarded_summary = _guarded_reselection_summary(
        coverage.get("candidate_metric_rows", []),
        coverage.get("paired_decision_rows", []),
        candidate_readiness,
        config,
    )
    calibration_rows = _calibration_grid(selected_rows, config)
    utility_bypass = any(row.get("utility_profile_win_blocked_by_canonical_guard") for row in scenario_rows)
    legacy_detected = any(_legacy_component_detected(row) for row in coverage.get("candidate_metric_rows", []))

    blocking = unique_sorted([*boundary_reasons, *required_artifact_reasons, *lineage_reasons, *profile_reasons])
    diagnostic = _diagnostic_reasons(
        selected_summary=selected_summary,
        scenario_summary=scenario_summary,
        candidate_readiness=candidate_readiness,
        utility_bypass=utility_bypass,
        legacy_detected=legacy_detected,
    )
    replay_reason_codes = _replay_reason_codes(selected_rows, scenario_rows)
    stage18_5_paired = _paired_summary(stage18_5)
    same_candidate_advantage = (
        stage18_5_paired.get("same_candidate_set_advantage_established") is True
        and selected_summary["guard_clean_coverage_win_count"] > 0
        and scenario_summary["guard_clean_coverage_win_count"] > 0
    )
    guard_refinement_passed = (
        not blocking
        and candidate_readiness["full_candidate_metric_replay_available"] is True
        and scenario_summary["false_coverage_win_cost_or_risk_regression_count"] == 0
        and selected_summary["false_coverage_win_cost_or_risk_regression_count"] == 0
        and scenario_summary["guard_clean_coverage_win_count"] > 0
    )
    route = _route(
        blocking=blocking,
        profile_route=profile_route,
        candidate_readiness=candidate_readiness,
        guarded_summary=guarded_summary,
        guard_refinement_passed=guard_refinement_passed,
        same_candidate_advantage=same_candidate_advantage,
        legacy_detected=legacy_detected,
    )
    readiness = {
        "schema_version": "xunce-stage18-6-stage19-readiness/v1",
        "readiness": "ready_for_stage19_preflight_human_review_only"
        if route == STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE
        else "not_authorized",
        "authorized": False,
        "guard_refinement_passed": guard_refinement_passed,
        "same_candidate_set_guard_clean_advantage_established": same_candidate_advantage,
        "full_candidate_metric_replay_available": candidate_readiness["full_candidate_metric_replay_available"],
    }
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "primary_route": route,
        "stage19_authorized": False,
        "stage19_readiness": readiness["readiness"],
    }
    generated_at = utc_now()
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "failed" if blocking else "passed",
        "reason_codes": unique_sorted([*blocking, *diagnostic, *replay_reason_codes]),
        "blocking_reason_codes": blocking,
        "diagnostic_reason_codes": diagnostic,
        "stage18_5_attribution_root": str(config["stage18_5_attribution_root"]),
        "coverage_comparison_root": str(config["coverage_comparison_root"]),
        "canonical_reward_profile": str(config["canonical_reward_profile"]),
        "profile_id": config["profile_id"],
        "profile_version": config["profile_version"],
        "profile_hash": config["profile_hash"],
        "guard_thresholds": config["guard_thresholds"],
        "guard_refinement_passed": guard_refinement_passed,
        "counterfactual_reselection_claimed": bool(candidate_readiness["full_candidate_metric_replay_available"]),
        "calibration_support_level": "full_candidate_metric_replay"
        if candidate_readiness["full_candidate_metric_replay_available"]
        else "selected_action_only",
        "candidate_metric_readiness": candidate_readiness,
        "selected_action_guard_replay_summary": selected_summary,
        "scenario_guard_replay_summary": scenario_summary,
        "guarded_reselection_summary": guarded_summary,
        "paired_decision_summary": {
            **stage18_5_paired,
            "same_candidate_set_guard_clean_advantage_established": same_candidate_advantage,
        },
        "utility_profile_bypass_blocked": utility_bypass,
        "legacy_v1_reward_component_detected": legacy_detected,
        "next_stage_routing": routing,
        "next_required_change": route,
        "stage19_readiness": readiness,
        "stage19_authorized": False,
        "canary_traffic_fraction": 0.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "runs_new_ppo_update": False,
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
        "default_policy_replacement_approved": False,
        "summary": str(paths["summary"]),
        "selected_action_guard_replay": str(paths["selected_action"]),
        "scenario_guard_replay": str(paths["scenario"]),
        "calibration_grid": str(paths["calibration"]),
        "candidate_metric_readiness_path": str(paths["candidate_readiness"]),
        "next_stage_routing_path": str(paths["routing"]),
        "report": str(paths["report"]),
        "manifest": str(paths["manifest"]),
        "config": str(config_path),
        "output_root": str(output_root),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "summary_status": summary["status"],
        "next_required_change": route,
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    write_jsonl(paths["selected_action"], selected_rows)
    write_jsonl(paths["scenario"], scenario_rows)
    write_jsonl(paths["calibration"], calibration_rows)
    write_json(paths["candidate_readiness"], candidate_readiness)
    write_json(paths["routing"], routing)
    write_json(paths["manifest"], manifest)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    write_json(paths["summary"], summary)
    return summary


def _load_config(path: Path, repo_root: Path, config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"expected schema_version {CONFIG_SCHEMA_VERSION}")
    merged = {**payload, **(config_overrides or {})}
    for key in ("stage18_5_attribution_root", "coverage_comparison_root"):
        value = merged.get(key)
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{key} must be a non-empty path string")
    profile_path = resolve_path(Path(str(merged.get("canonical_reward_profile", DEFAULT_CANONICAL_PROFILE))), repo_root).resolve()
    profile = load_canonical_reward_profile(profile_path)
    guard_thresholds = stage18_guard_thresholds(profile)
    return {
        **merged,
        "stage18_5_attribution_root": resolve_path(Path(str(merged["stage18_5_attribution_root"])), repo_root).resolve(),
        "coverage_comparison_root": resolve_path(Path(str(merged["coverage_comparison_root"])), repo_root).resolve(),
        "canonical_reward_profile": profile_path,
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
        "guard_thresholds": guard_thresholds.to_artifact_dict(),
        "cost_weight_grid": _number_list(merged.get("cost_weight_grid", [0.0])),
        "risk_weight_grid": _number_list(merged.get("risk_weight_grid", [0.0])),
        "risk_cost_weight_grid": _number_list(merged.get("risk_cost_weight_grid", [0.0])),
    }


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "selected_action": output_root / SELECTED_ACTION_FILE,
        "scenario": output_root / SCENARIO_FILE,
        "calibration": output_root / CALIBRATION_FILE,
        "candidate_readiness": output_root / CANDIDATE_READINESS_FILE,
        "routing": output_root / ROUTING_FILE,
        "report": output_root / REPORT_FILE,
        "manifest": output_root / MANIFEST_FILE,
    }


def _load_evidence(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    missing: list[str] = []
    stage18_5_root = Path(config["stage18_5_attribution_root"])
    coverage_root = Path(config["coverage_comparison_root"])
    stage18_5 = {
        "summary": _read_json(stage18_5_root / STAGE18_5_SUMMARY_FILE, missing, "missing_stage18_5_attribution_summary"),
        "guard": _read_json(stage18_5_root / STAGE18_5_GUARD_FILE, missing, "missing_stage18_5_guard_evaluation"),
        "routing": _read_json(stage18_5_root / STAGE18_5_ROUTING_FILE, missing, "missing_stage18_5_next_stage_routing"),
        "paired": _read_json(stage18_5_root / STAGE18_5_PAIRED_FILE, missing, "missing_stage18_5_paired_decision_summary"),
        "regression_rows": _read_jsonl(stage18_5_root / STAGE18_5_REGRESSION_FILE, missing, "missing_stage18_5_regression_attribution"),
    }
    candidate_path = coverage_root / CANDIDATE_METRIC_AUDIT_FILE
    coverage = {
        "summary": _read_json(coverage_root / COVERAGE_SUMMARY_FILE, missing, "missing_coverage_comparison_summary"),
        "aggregate": _read_json(coverage_root / COVERAGE_AGGREGATE_FILE, missing, "missing_coverage_comparison_aggregate"),
        "pair_rows": _read_jsonl(coverage_root / COVERAGE_PAIRS_FILE, missing, "missing_coverage_comparison_pairs"),
        "episode_rows": _read_jsonl(coverage_root / COVERAGE_EPISODES_FILE, missing, "missing_coverage_episodes"),
        "paired_decision_rows": _read_jsonl(coverage_root / PAIRED_DECISION_AUDIT_FILE, missing, "missing_paired_decision_audit"),
        "candidate_metric_path": candidate_path,
        "candidate_metric_rows": _read_optional_jsonl(candidate_path),
    }
    return stage18_5, coverage, missing


def _stage18_5_lineage_reasons(config: dict[str, Any], stage18_5: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    summary = stage18_5.get("summary", {})
    guard = stage18_5.get("guard") or summary.get("guard_evaluation")
    routing = stage18_5.get("routing") or summary.get("next_stage_routing")
    if summary and summary.get("schema_version") != STAGE18_5_SUMMARY_SCHEMA_VERSION:
        reasons.append("invalid_stage18_5_attribution_summary_schema")
    if guard and guard.get("schema_version") != STAGE18_5_GUARD_SCHEMA_VERSION:
        reasons.append("invalid_stage18_5_guard_summary_schema")
    if routing and routing.get("schema_version") != STAGE18_5_ROUTING_SCHEMA_VERSION:
        reasons.append("invalid_stage18_5_routing_summary_schema")
    if summary and not _same_path(summary.get("coverage_comparison_root"), Path(config["coverage_comparison_root"])):
        reasons.append("stale_stage18_5_attribution_root")
    if routing:
        if routing.get("primary_route") != GUARD_REFINEMENT_NEXT_REQUIRED_CHANGE:
            reasons.append("stage18_5_route_not_guard_refinement")
        if routing.get("stage19_authorized") is not False:
            reasons.append("stage18_5_stage19_authorized")
    readiness = summary.get("stage19_readiness", {}) if isinstance(summary, dict) else {}
    if isinstance(readiness, dict) and readiness.get("authorized") is not False:
        reasons.append("stage18_5_stage19_authorized")
    return unique_sorted(reasons)


def _profile_lineage_reasons(
    config: dict[str, Any],
    stage18_5: dict[str, Any],
    coverage: dict[str, Any],
) -> tuple[list[str], str | None]:
    missing = False
    mismatch = False
    expected = {
        "profile_id": config["profile_id"],
        "profile_version": config["profile_version"],
        "profile_hash": config["profile_hash"],
    }
    payloads = [
        ("stage18_5_summary", stage18_5.get("summary", {}), "profile_id", "profile_version", "profile_hash"),
        (
            "stage18_5_guard_thresholds",
            (stage18_5.get("guard", {}) or {}).get("thresholds", {}),
            "profile_id",
            "profile_version",
            "profile_hash",
        ),
        ("coverage_summary", coverage.get("summary", {}), "profile_id", "profile_version", "profile_hash"),
    ]
    for _, payload, id_field, version_field, hash_field in payloads:
        if not isinstance(payload, dict):
            missing = True
            continue
        for field in (id_field, version_field, hash_field):
            if not payload.get(field):
                missing = True
        if payload.get(id_field) and payload.get(id_field) != expected["profile_id"]:
            mismatch = True
        if payload.get(version_field) and payload.get(version_field) != expected["profile_version"]:
            mismatch = True
        if payload.get(hash_field) and payload.get(hash_field) != expected["profile_hash"]:
            mismatch = True
    for row in coverage.get("pair_rows", []):
        pair_identity = {
            "profile_id": row.get("canonical_guard_profile_id"),
            "profile_version": row.get("canonical_guard_profile_version"),
            "profile_hash": row.get("canonical_guard_profile_hash"),
        }
        for field, expected_value in expected.items():
            value = pair_identity[field]
            if not value:
                missing = True
            elif value != expected_value:
                mismatch = True
    for row in coverage.get("candidate_metric_rows", []):
        for field, expected_value in expected.items():
            value = row.get(field)
            if not value:
                missing = True
            elif value != expected_value:
                mismatch = True
    reasons = []
    route = None
    if missing:
        reasons.append("profile_hash_missing")
        route = STAGE18_5_RERUN_NEXT_REQUIRED_CHANGE
    if mismatch:
        reasons.append("profile_hash_mismatch")
        route = STAGE18_4_RERUN_NEXT_REQUIRED_CHANGE
    return unique_sorted(reasons), route


def _selected_action_guard_replay(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    counts = {"guard_clean_coverage_win": 0, "false_coverage_win_cost_or_risk_regression": 0, "neutral_or_loss": 0, "non_comparable": 0}
    thresholds = config["guard_thresholds"]
    for index, row in enumerate(rows):
        observed = {
            "coverage_delta": _delta(row, "xunce_selected_expected_new_coverage_cell_count", "incumbent_selected_expected_new_coverage_cell_count"),
            "roi_delta": _delta(row, "xunce_selected_roi_weighted_coverage_delta", "incumbent_selected_roi_weighted_coverage_delta"),
            "path_cost_delta": _delta(row, "xunce_selected_path_cost", "incumbent_selected_path_cost"),
            "risk_delta": _delta(row, "xunce_selected_risk", "incumbent_selected_risk"),
            "risk_cost_weighted_delta": _risk_cost_delta(row),
            "coverage_per_cost_delta": _ratio_delta(
                row.get("xunce_selected_expected_new_coverage_cell_count"),
                row.get("xunce_selected_path_cost"),
                row.get("incumbent_selected_expected_new_coverage_cell_count"),
                row.get("incumbent_selected_path_cost"),
            ),
        }
        required_keys = [
            "coverage_delta",
            "roi_delta",
            "path_cost_delta",
            "risk_cost_weighted_delta",
        ]
        if thresholds.get("risk_delta_hard_gate_enabled", True):
            required_keys.append("risk_delta")
        if any(observed[key] is None for key in required_keys):
            classification = "non_comparable"
            reasons = ["missing_selected_action_metric"]
        elif observed["coverage_delta"] < thresholds["min_coverage_delta_cells"]:
            classification = "neutral_or_loss"
            reasons = []
        elif _passes_selected_action_budget_guard(observed, thresholds):
            classification = "guard_clean_coverage_win"
            reasons = []
        else:
            classification = "false_coverage_win_cost_or_risk_regression"
            reasons = _selected_action_guard_failure_reasons(observed, thresholds)
        counts[classification] += 1
        output.append(
            {
                "schema_version": SELECTED_ACTION_ROW_SCHEMA_VERSION,
                "row_index": index,
                "scenario_id": row.get("scenario_id"),
                "step_index": row.get("step_index"),
                "candidate_set_hash": row.get("candidate_set_hash"),
                "covered_cells_hash": row.get("covered_cells_hash"),
                "classification": classification,
                "observed": observed,
                "reason_codes": unique_sorted(reasons),
            }
        )
    return output, {
        "schema_version": "xunce-stage18-6-selected-action-guard-replay-summary/v1",
        "row_count": len(output),
        "guard_clean_coverage_win_count": counts["guard_clean_coverage_win"],
        "false_coverage_win_cost_or_risk_regression_count": counts["false_coverage_win_cost_or_risk_regression"],
        "neutral_or_loss_count": counts["neutral_or_loss"],
        "non_comparable_count": counts["non_comparable"],
    }


def _scenario_guard_replay(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    counts = {"guard_clean_coverage_win": 0, "false_coverage_win_cost_or_risk_regression": 0, "neutral_or_loss": 0, "non_comparable": 0}
    thresholds = config["guard_thresholds"]
    for index, row in enumerate(rows):
        observed = {
            "coverage_delta": _finite(row.get("coverage_delta_cells")),
            "path_cost_delta": _finite(row.get("path_cost_delta_m")),
            "risk_delta": _finite(row.get("risk_delta")),
            "risk_cost_weighted_delta": _finite(row.get("risk_cost_weighted_delta")),
            "coverage_per_100m_delta": _finite(row.get("coverage_per_100m_delta")),
            "coverage_gain_per_path_cost_delta": _finite(row.get("coverage_gain_per_path_cost_delta")),
        }
        utility_win = _utility_profile_win(row.get("utility_profile_outcomes"))
        required_keys = ["coverage_delta", "path_cost_delta", "risk_cost_weighted_delta", "coverage_per_100m_delta"]
        if thresholds.get("risk_delta_hard_gate_enabled", True):
            required_keys.append("risk_delta")
        if any(observed[key] is None for key in required_keys):
            classification = "non_comparable"
            reasons = ["missing_scenario_guard_metric"]
        elif observed["coverage_delta"] < thresholds["min_coverage_delta_cells"]:
            classification = "neutral_or_loss"
            reasons = []
        elif _passes_delta_guard(observed, thresholds, use_per_cost=False):
            classification = "guard_clean_coverage_win"
            reasons = []
        else:
            classification = "false_coverage_win_cost_or_risk_regression"
            reasons = _guard_failure_reasons(observed, thresholds, use_per_cost=False)
        counts[classification] += 1
        output.append(
            {
                "schema_version": SCENARIO_ROW_SCHEMA_VERSION,
                "row_index": index,
                "scenario_id": row.get("scenario_id"),
                "classification": classification,
                "observed": observed,
                "utility_profile_win_blocked_by_canonical_guard": bool(
                    utility_win and classification == "false_coverage_win_cost_or_risk_regression"
                ),
                "reason_codes": unique_sorted(reasons),
            }
        )
    return output, {
        "schema_version": "xunce-stage18-6-scenario-guard-replay-summary/v1",
        "row_count": len(output),
        "guard_clean_coverage_win_count": counts["guard_clean_coverage_win"],
        "false_coverage_win_cost_or_risk_regression_count": counts["false_coverage_win_cost_or_risk_regression"],
        "neutral_or_loss_count": counts["neutral_or_loss"],
        "non_comparable_count": counts["non_comparable"],
    }


def _candidate_metric_readiness(config: dict[str, Any], coverage: dict[str, Any]) -> dict[str, Any]:
    path = Path(coverage["candidate_metric_path"])
    rows = coverage.get("candidate_metric_rows", [])
    blocking: list[str] = []
    diagnostic: list[str] = []
    if not path.is_file():
        blocking.append("missing_candidate_metric_audit")
    paired_keys = {
        paired_decision_key(row)
        for row in coverage.get("paired_decision_rows", [])
    }
    candidate_keys = set()
    for row in rows:
        for field in CANDIDATE_REQUIRED_FIELDS:
            if field not in row:
                blocking.append(f"missing_candidate_metric_field:{field}")
        key = paired_decision_key(row)
        candidate_keys.add(key)
        if paired_keys and key not in paired_keys:
            diagnostic.append("candidate_metric_extra_candidate_set_key")
        for field in ("profile_id", "profile_version", "profile_hash"):
            if field in row and not row.get(field):
                blocking.append(f"missing_candidate_metric_field:{field}")
        if row.get("profile_id") and row.get("profile_id") != config["profile_id"]:
            blocking.append("profile_hash_mismatch")
        if row.get("profile_version") and row.get("profile_version") != config["profile_version"]:
            blocking.append("profile_hash_mismatch")
        if row.get("profile_hash") and row.get("profile_hash") != config["profile_hash"]:
            blocking.append("profile_hash_mismatch")
    if paired_keys and candidate_keys and paired_keys - candidate_keys:
        blocking.append("candidate_metric_missing_paired_decision_key")
    full = bool(path.is_file() and rows and not blocking)
    return {
        "schema_version": CANDIDATE_READINESS_SCHEMA_VERSION,
        "candidate_metric_audit": str(path) if path.is_file() else None,
        "candidate_metric_audit_row_count": len(rows),
        "paired_decision_key_count": len(paired_keys),
        "candidate_metric_key_count": len(candidate_keys),
        "missing_paired_decision_key_count": len(paired_keys - candidate_keys) if paired_keys else 0,
        "extra_candidate_set_key_count": len(candidate_keys - paired_keys) if paired_keys else 0,
        "full_candidate_metric_replay_available": full,
        "counterfactual_reselection_claim_allowed": full,
        "reason_codes": unique_sorted([*blocking, *diagnostic]),
        "blocking_reason_codes": unique_sorted(blocking),
        "diagnostic_reason_codes": unique_sorted(diagnostic),
    }


def _guarded_reselection_summary(
    rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    readiness: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if readiness["full_candidate_metric_replay_available"] is not True:
        return {
            "schema_version": "xunce-stage18-6-guarded-reselection-summary/v1",
            "evaluated": False,
            "candidate_guard_mode": CANDIDATE_GUARD_MODE,
            "absolute_risk_proxy_is_audit_only": True,
            "safe_candidate_available_count": 0,
            "unsafe_high_coverage_candidate_rejected_count": 0,
            "reason_codes": readiness["reason_codes"],
        }
    thresholds = config["guard_thresholds"]
    rows_by_step: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_step.setdefault(paired_decision_key(row), []).append(row)
    safe = 0
    unsafe_high = 0
    evaluated = 0
    non_comparable = 0
    failure_counts: dict[str, int] = {}
    for paired in paired_rows:
        candidates = rows_by_step.get(paired_decision_key(paired), [])
        baseline = selected_baseline_candidate(paired, candidates)
        for row in candidates:
            observation = candidate_guard_observation(row, paired, baseline_candidate_row=baseline)
            if observation.get("action_mask_valid") is not True:
                continue
            evaluated += 1
            if observation.get("comparable") is not True:
                non_comparable += 1
                for reason in observation.get("reason_codes", []):
                    failure_counts[reason] = failure_counts.get(reason, 0) + 1
                continue
            failures = candidate_guard_failure_reasons(observation, thresholds)
            if not failures:
                safe += 1
            elif (_finite(observation.get("coverage_delta_cells")) or 0.0) >= thresholds["min_coverage_delta_cells"]:
                unsafe_high += 1
                for reason in failures:
                    failure_counts[reason] = failure_counts.get(reason, 0) + 1
    return {
        "schema_version": "xunce-stage18-6-guarded-reselection-summary/v1",
        "evaluated": True,
        "candidate_guard_mode": CANDIDATE_GUARD_MODE,
        "absolute_risk_proxy_is_audit_only": True,
        "evaluated_candidate_count": evaluated,
        "non_comparable_candidate_count": non_comparable,
        "safe_candidate_available_count": safe,
        "guard_clean_advantage_candidate_count": safe,
        "unsafe_high_coverage_candidate_rejected_count": unsafe_high,
        "candidate_guard_failure_reason_counts": dict(sorted(failure_counts.items())),
        "reason_codes": [],
    }


def _calibration_grid(selected_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cost_weight in config["cost_weight_grid"]:
        for risk_weight in config["risk_weight_grid"]:
            for risk_cost_weight in config["risk_cost_weight_grid"]:
                retained = 0
                false_count = 0
                clean_count = 0
                for row in selected_rows:
                    obs = row.get("observed", {})
                    if row.get("classification") == "false_coverage_win_cost_or_risk_regression":
                        false_count += 1
                    if row.get("classification") == "guard_clean_coverage_win":
                        clean_count += 1
                    roi = _finite(obs.get("roi_delta")) or 0.0
                    utility = (
                        roi
                        - cost_weight * max(0.0, _finite(obs.get("path_cost_delta")) or 0.0)
                        - risk_weight * max(0.0, _finite(obs.get("risk_delta")) or 0.0)
                        - risk_cost_weight * max(0.0, _finite(obs.get("risk_cost_weighted_delta")) or 0.0)
                    )
                    if utility > 0:
                        retained += 1
                rows.append(
                    {
                        "schema_version": CALIBRATION_ROW_SCHEMA_VERSION,
                        "cost_weight": cost_weight,
                        "risk_weight": risk_weight,
                        "risk_cost_weight": risk_cost_weight,
                        "false_coverage_win_count": false_count,
                        "guard_clean_coverage_win_count": clean_count,
                        "retained_policy_disagreement_count": retained,
                        "calibration_viable": false_count == 0 and clean_count > 0,
                    }
                )
    return rows


def _route(
    *,
    blocking: list[str],
    profile_route: str | None,
    candidate_readiness: dict[str, Any],
    guarded_summary: dict[str, Any],
    guard_refinement_passed: bool,
    same_candidate_advantage: bool,
    legacy_detected: bool,
) -> str:
    blockers = set(blocking)
    if {"boundary_violation", "canary_traffic_fraction_nonzero"} & blockers:
        return BOUNDARY_NEXT_REQUIRED_CHANGE
    if profile_route == STAGE18_5_RERUN_NEXT_REQUIRED_CHANGE or any(reason.startswith("invalid_stage18_5") or reason == "stale_stage18_5_attribution_root" for reason in blockers):
        return STAGE18_5_RERUN_NEXT_REQUIRED_CHANGE
    if profile_route == STAGE18_4_RERUN_NEXT_REQUIRED_CHANGE or any(reason.startswith("missing_coverage") or reason == "profile_hash_mismatch" for reason in blockers):
        return STAGE18_4_RERUN_NEXT_REQUIRED_CHANGE
    if blocking:
        return STAGE18_4_RERUN_NEXT_REQUIRED_CHANGE
    if candidate_readiness["full_candidate_metric_replay_available"] is not True:
        return CANDIDATE_AUDIT_NEXT_REQUIRED_CHANGE
    if legacy_detected:
        return GUARD_REFINEMENT_NEXT_REQUIRED_CHANGE
    if guard_refinement_passed and same_candidate_advantage:
        return STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE
    if guarded_summary["safe_candidate_available_count"] == 0:
        return CANDIDATE_EXPANSION_NEXT_REQUIRED_CHANGE
    return GUARD_REFINEMENT_NEXT_REQUIRED_CHANGE


def _diagnostic_reasons(
    *,
    selected_summary: dict[str, Any],
    scenario_summary: dict[str, Any],
    candidate_readiness: dict[str, Any],
    utility_bypass: bool,
    legacy_detected: bool,
) -> list[str]:
    reasons = []
    if scenario_summary["false_coverage_win_cost_or_risk_regression_count"] > 0:
        reasons.append("cost_efficiency_regression")
        reasons.append("path_cost_regression")
        reasons.append("risk_regression")
    if selected_summary["false_coverage_win_cost_or_risk_regression_count"] > 0:
        reasons.append("same_candidate_selected_action_guard_regression")
    reasons.extend(candidate_readiness["reason_codes"])
    if utility_bypass:
        reasons.append("utility_profile_win_blocked_by_canonical_guard")
    if legacy_detected:
        reasons.append("legacy_v1_reward_component_detected")
    return unique_sorted(reasons)


def _boundary_reasons(config: dict[str, Any], *payloads: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    all_payloads = [config, *payloads]
    for payload in all_payloads:
        if not isinstance(payload, dict):
            continue
        if _finite(payload.get("canary_traffic_fraction")) not in (None, 0.0):
            reasons.append("canary_traffic_fraction_nonzero")
        for field in BOUNDARY_FIELDS:
            if payload.get(field) is True:
                reasons.append("boundary_violation")
    return unique_sorted(reasons)


def _passes_delta_guard(observed: dict[str, Any], thresholds: dict[str, Any], *, use_per_cost: bool) -> bool:
    if observed["coverage_delta"] < thresholds["min_coverage_delta_cells"]:
        return False
    if observed["path_cost_delta"] > thresholds["max_path_cost_delta_m"] + TOLERANCE:
        return False
    if (
        thresholds.get("risk_delta_hard_gate_enabled", True)
        and thresholds.get("max_risk_delta") is not None
        and observed["risk_delta"] > thresholds["max_risk_delta"] + TOLERANCE
    ):
        return False
    soft_risk_budget = thresholds.get("max_soft_risk_exposure_delta", thresholds.get("max_risk_cost_weighted_delta"))
    if soft_risk_budget is not None and observed["risk_cost_weighted_delta"] > soft_risk_budget + TOLERANCE:
        return False
    if use_per_cost:
        return observed["coverage_per_cost_delta"] >= 0.0
    return observed["coverage_per_100m_delta"] >= thresholds["min_coverage_per_100m_delta"]


def _passes_selected_action_budget_guard(observed: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    if observed["coverage_delta"] < thresholds["min_coverage_delta_cells"]:
        return False
    if observed["path_cost_delta"] > thresholds["max_path_cost_delta_m"] + TOLERANCE:
        return False
    if (
        thresholds.get("risk_delta_hard_gate_enabled", True)
        and thresholds.get("max_risk_delta") is not None
        and observed["risk_delta"] > thresholds["max_risk_delta"] + TOLERANCE
    ):
        return False
    soft_risk_budget = thresholds.get("max_soft_risk_exposure_delta", thresholds.get("max_risk_cost_weighted_delta"))
    if soft_risk_budget is not None and observed["risk_cost_weighted_delta"] > soft_risk_budget + TOLERANCE:
        return False
    return True


def _selected_action_guard_failure_reasons(observed: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    reasons = []
    if observed["path_cost_delta"] > thresholds["max_path_cost_delta_m"] + TOLERANCE:
        reasons.append("path_cost_budget_exceeded")
    if (
        thresholds.get("risk_delta_hard_gate_enabled", True)
        and thresholds.get("max_risk_delta") is not None
        and observed["risk_delta"] > thresholds["max_risk_delta"] + TOLERANCE
    ):
        reasons.append("risk_budget_exceeded")
    soft_risk_budget = thresholds.get("max_soft_risk_exposure_delta", thresholds.get("max_risk_cost_weighted_delta"))
    if soft_risk_budget is not None and observed["risk_cost_weighted_delta"] > soft_risk_budget + TOLERANCE:
        reasons.append(
            "soft_risk_exposure_budget_exceeded"
            if thresholds.get("candidate_level_risk_delta_guard_is_diagnostic_only")
            else "risk_cost_weighted_budget_exceeded"
        )
    return reasons


def _guard_failure_reasons(observed: dict[str, Any], thresholds: dict[str, Any], *, use_per_cost: bool) -> list[str]:
    reasons = []
    if observed["path_cost_delta"] > thresholds["max_path_cost_delta_m"] + TOLERANCE:
        reasons.append("path_cost_budget_exceeded")
    if (
        thresholds.get("risk_delta_hard_gate_enabled", True)
        and thresholds.get("max_risk_delta") is not None
        and observed["risk_delta"] > thresholds["max_risk_delta"] + TOLERANCE
    ):
        reasons.append("risk_budget_exceeded")
    soft_risk_budget = thresholds.get("max_soft_risk_exposure_delta", thresholds.get("max_risk_cost_weighted_delta"))
    if soft_risk_budget is not None and observed["risk_cost_weighted_delta"] > soft_risk_budget + TOLERANCE:
        reasons.append(
            "soft_risk_exposure_budget_exceeded"
            if thresholds.get("candidate_level_risk_delta_guard_is_diagnostic_only")
            else "risk_cost_weighted_budget_exceeded"
        )
    if use_per_cost:
        if observed["coverage_per_cost_delta"] < 0.0:
            reasons.append("coverage_gain_per_path_cost_regression")
    elif observed["coverage_per_100m_delta"] < thresholds["min_coverage_per_100m_delta"]:
        reasons.append("coverage_per_100m_regression")
    return reasons


def _replay_reason_codes(selected_rows: list[dict[str, Any]], scenario_rows: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for row in [*selected_rows, *scenario_rows]:
        if not isinstance(row, dict):
            continue
        row_reasons = row.get("reason_codes")
        if isinstance(row_reasons, list):
            reasons.extend(str(reason) for reason in row_reasons if reason)
    return unique_sorted(reasons)


def _read_json(path: Path, missing: list[str], reason: str) -> dict[str, Any]:
    if not path.is_file():
        missing.append(reason)
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        missing.append(f"invalid_{reason.removeprefix('missing_')}")
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path, missing: list[str], reason: str) -> list[dict[str, Any]]:
    if not path.is_file():
        missing.append(reason)
        return []
    return _read_optional_jsonl(path)


def _read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _paired_summary(stage18_5: dict[str, Any]) -> dict[str, Any]:
    summary = stage18_5.get("paired") or (stage18_5.get("summary", {}) or {}).get("paired_decision_summary", {})
    return summary if isinstance(summary, dict) else {}


def _utility_profile_win(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    for value in payload.values():
        if isinstance(value, dict) and value.get("winner") == "xunce":
            return True
        if value == "xunce":
            return True
    return False


def _legacy_component_detected(row: dict[str, Any]) -> bool:
    return any(field in row for field in LEGACY_V1_COMPONENT_FIELDS)


def _same_path(value: Any, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return Path(value).resolve() == expected.resolve()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _delta(row: dict[str, Any], left: str, right: str) -> float | None:
    left_value = _finite(row.get(left))
    right_value = _finite(row.get(right))
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _risk_cost_delta(row: dict[str, Any]) -> float | None:
    left_cost = _finite(row.get("xunce_selected_path_cost"))
    right_cost = _finite(row.get("incumbent_selected_path_cost"))
    left_risk = _finite(row.get("xunce_selected_risk"))
    right_risk = _finite(row.get("incumbent_selected_risk"))
    if None in (left_cost, right_cost, left_risk, right_risk):
        return None
    return left_cost * left_risk - right_cost * right_risk


def _ratio_delta(left_num: Any, left_den: Any, right_num: Any, right_den: Any) -> float | None:
    left_num_value = _finite(left_num)
    left_den_value = _finite(left_den)
    right_num_value = _finite(right_num)
    right_den_value = _finite(right_den)
    if None in (left_num_value, left_den_value, right_num_value, right_den_value):
        return None
    if abs(left_den_value) <= TOLERANCE or abs(right_den_value) <= TOLERANCE:
        return None
    return left_num_value / left_den_value - right_num_value / right_den_value


def _number_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    numbers = [_finite(item) for item in value]
    return [float(item) for item in numbers if item is not None]


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Stage 18.6 Coverage Reward Cost-Risk Guard Refinement",
            "",
            f"- status: `{summary['status']}`",
            f"- guard_refinement_passed: `{summary['guard_refinement_passed']}`",
            f"- counterfactual_reselection_claimed: `{summary['counterfactual_reselection_claimed']}`",
            f"- full_candidate_metric_replay_available: `{summary['candidate_metric_readiness']['full_candidate_metric_replay_available']}`",
            f"- candidate_guard_mode: `{summary['guarded_reselection_summary'].get('candidate_guard_mode')}`",
            f"- absolute_risk_proxy_is_audit_only: `{summary['guarded_reselection_summary'].get('absolute_risk_proxy_is_audit_only')}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- stage19_authorized: `{summary['stage19_readiness']['authorized']}`",
            "",
            "Stage 18.6 is an offline guard-refinement review only. It does not train PPO, publish checkpoints, replace the default policy, connect a real executor, or start canary traffic.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
