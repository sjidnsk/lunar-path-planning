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

try:
    from git_provenance import git_snapshot
    from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from global_99_governance_common import global_99_boundary_defaults
    from xunce_stage18_guard_thresholds import stage18_guard_thresholds
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from scripts.global_99_governance_common import global_99_boundary_defaults
    from scripts.xunce_stage18_guard_thresholds import stage18_guard_thresholds

from model_explorer.policy.canonical_reward import load_canonical_reward_profile


CONFIG_SCHEMA_VERSION = "xunce-stage18-5-evidence-attribution-review-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage18-5-evidence-attribution-summary/v1"
DEFAULT_CONFIG = "configs/xunce_stage18_5_evidence_attribution_review_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_stage18_5_evidence_attribution_review_v1"
DEFAULT_CANONICAL_PROFILE = "configs/xunce_canonical_reward_guard_profile_v2.json"

COVERAGE_SUMMARY_FILE = "xunce-exploration-coverage-comparison-summary.json"
COVERAGE_AGGREGATE_FILE = "xunce-exploration-coverage-comparison-aggregate.json"
COVERAGE_PAIRS_FILE = "xunce-exploration-coverage-comparison-pairs.jsonl"
COVERAGE_EPISODES_FILE = "xunce-exploration-coverage-episodes.jsonl"
PAIRED_DECISION_AUDIT_FILE = "xunce-exploration-coverage-paired-decision-audit.jsonl"

SUMMARY_FILE = "xunce-stage18-5-evidence-attribution-summary.json"
ATTRIBUTION_FILE = "xunce-stage18-5-regression-attribution.jsonl"
PAIRED_SUMMARY_FILE = "xunce-stage18-5-paired-decision-summary.json"
GUARD_EVALUATION_FILE = "xunce-stage18-5-guard-evaluation.json"
NEXT_STAGE_ROUTING_FILE = "xunce-stage18-5-next-stage-routing.json"
REPORT_FILE = "xunce-stage18-5-evidence-attribution-report.md"
MANIFEST_FILE = "xunce-stage18-5-manifest.json"

MISSING_ARTIFACT_NEXT_REQUIRED_CHANGE = "rerun_xunce_stage18_4e_coverage_comparison_with_required_artifacts"
BOUNDARY_NEXT_REQUIRED_CHANGE = "resolve_stage18_5_evidence_review_boundary_rejections"
GUARD_REFINEMENT_NEXT_REQUIRED_CHANGE = "refine_coverage_reward_and_cost_guard"
PAIRED_ADVANTAGE_NEXT_REQUIRED_CHANGE = "establish_same_candidate_set_policy_selection_advantage"
STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE = "prepare_stage19_evaluator_critic_preflight"

RISK_PROXY_CAVEAT = (
    "risk_delta and risk_cost_weighted_delta are treated as offline risk-source/proxy diagnostics, "
    "not real-world safety claims."
)

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

REQUIRED_JSON_ARTIFACTS = {
    "coverage_comparison_summary": (COVERAGE_SUMMARY_FILE, "xunce-exploration-coverage-comparison-summary/v1"),
    "coverage_comparison_aggregate": (COVERAGE_AGGREGATE_FILE, "xunce-exploration-coverage-comparison-aggregate/v1"),
}

REQUIRED_JSONL_ARTIFACTS = {
    "coverage_comparison_pairs": (COVERAGE_PAIRS_FILE, "xunce-exploration-coverage-comparison-pair/v1"),
    "coverage_episodes": (COVERAGE_EPISODES_FILE, "xunce-exploration-coverage-episode/v1"),
    "paired_decision_audit": (PAIRED_DECISION_AUDIT_FILE, "xunce-exploration-coverage-paired-decision-audit/v1"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review Stage 18.5 evidence attribution from Stage 18.4E coverage artifacts.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--coverage-comparison-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    overrides = {"coverage_comparison_root": args.coverage_comparison_root} if args.coverage_comparison_root else None
    try:
        summary = run_xunce_stage18_5_evidence_attribution_review(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
            config_overrides=overrides,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "next_required_change": summary["next_required_change"],
                "blocking_reason_codes": summary["blocking_reason_codes"],
                "diagnostic_reason_codes": summary["diagnostic_reason_codes"],
                "primary_route": summary["next_stage_routing"]["primary_route"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage18_5_evidence_attribution_review(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path, repo_root, config_overrides=config_overrides)
    output_root.mkdir(parents=True, exist_ok=True)
    evidence = _load_evidence(Path(config["coverage_comparison_root"]))
    boundary_reasons = _boundary_reasons(config, evidence)
    paired_summary = _paired_decision_summary(evidence["paired_decision_audit"], evidence["coverage_comparison_summary"])
    attribution_rows = _attribution_rows(evidence)
    guard_evaluation = _guard_evaluation(config, evidence)
    candidate_diagnostics = _candidate_generation_diagnostics(evidence)
    decision = _decision(
        evidence=evidence,
        boundary_reasons=boundary_reasons,
        guard_evaluation=guard_evaluation,
        paired_summary=paired_summary,
        candidate_diagnostics=candidate_diagnostics,
    )
    routing = _next_stage_routing(decision=decision, guard_evaluation=guard_evaluation, paired_summary=paired_summary)

    paths = {
        "summary": output_root / SUMMARY_FILE,
        "regression_attribution": output_root / ATTRIBUTION_FILE,
        "paired_decision_summary": output_root / PAIRED_SUMMARY_FILE,
        "guard_evaluation": output_root / GUARD_EVALUATION_FILE,
        "next_stage_routing": output_root / NEXT_STAGE_ROUTING_FILE,
        "report": output_root / REPORT_FILE,
        "manifest": output_root / MANIFEST_FILE,
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "blocking_reason_codes": decision["blocking_reason_codes"],
        "diagnostic_reason_codes": decision["diagnostic_reason_codes"],
        "evidence_authenticity_gate_passed": decision["evidence_authenticity_gate_passed"],
        "candidate_validity_gate_passed": decision["candidate_validity_gate_passed"],
        "comparison_allowed": decision["comparison_allowed"],
        "next_required_change": routing["primary_route"],
        "coverage_comparison_root": config["coverage_comparison_root"],
        "canonical_reward_profile": config["canonical_reward_profile"],
        "profile_id": config["profile_id"],
        "profile_version": config["profile_version"],
        "profile_hash": config["profile_hash"],
        "guard_config": _guard_config_payload(config),
        "guard_evaluation": guard_evaluation,
        "paired_decision_summary": paired_summary,
        "stage19_readiness": _stage19_readiness(paired_summary, guard_evaluation, decision),
        "next_stage_routing": routing,
        "attribution_class_counts": _attribution_class_counts(attribution_rows),
        "regression_count_summary": _regression_count_summary(evidence["coverage_comparison_summary"], evidence["coverage_comparison_pairs"]),
        "candidate_generation_diagnostics": candidate_diagnostics,
        "risk_source_proxy_caveat": RISK_PROXY_CAVEAT,
        "risk_source_summary": _risk_source_summary(evidence["coverage_comparison_summary"], evidence["coverage_episodes"]),
        "artifact_validation": evidence["artifact_validation"],
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "regression_attribution": str(paths["regression_attribution"]),
        "paired_decision_summary_path": str(paths["paired_decision_summary"]),
        "guard_evaluation_path": str(paths["guard_evaluation"]),
        "next_stage_routing_path": str(paths["next_stage_routing"]),
        "report": str(paths["report"]),
        "manifest": str(paths["manifest"]),
        "git_provenance": git_snapshot(repo_root),
        "governance_boundary": _boundary_fields(),
        **_boundary_fields(),
    }
    manifest = _manifest(paths, config=config, evidence=evidence, summary=summary)

    write_json(paths["guard_evaluation"], guard_evaluation)
    write_json(paths["paired_decision_summary"], paired_summary)
    write_json(paths["next_stage_routing"], routing)
    write_jsonl(paths["regression_attribution"], attribution_rows)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _load_config(config_path: Path, repo_root: Path, config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"expected schema_version {CONFIG_SCHEMA_VERSION}")
    merged = {**payload, **(config_overrides or {})}
    root_value = merged.get("coverage_comparison_root")
    if not isinstance(root_value, str) or not root_value:
        raise ConfigError("coverage_comparison_root must be a non-empty path string")
    profile_path = resolve_path(Path(str(merged.get("canonical_reward_profile", DEFAULT_CANONICAL_PROFILE))), repo_root)
    canonical_profile = load_canonical_reward_profile(profile_path)
    guard_thresholds = stage18_guard_thresholds(canonical_profile)
    coverage_gain_per_path_cost_delta_mode = str(
        merged.get(
            "coverage_gain_per_path_cost_delta_mode",
            guard_thresholds.coverage_gain_per_path_cost_delta_mode,
        )
    )
    if coverage_gain_per_path_cost_delta_mode != "audit_only":
        raise ConfigError("coverage_gain_per_path_cost_delta_mode must be audit_only")
    config = {
        "schema_version": merged["schema_version"],
        "coverage_comparison_root": str(resolve_path(Path(root_value), repo_root).resolve()),
        "canonical_reward_profile": str(profile_path.resolve()),
        "profile_id": canonical_profile.profile_id,
        "profile_version": canonical_profile.profile_version,
        "profile_hash": canonical_profile.profile_hash,
        "max_path_cost_delta_m": _config_float(
            merged,
            "max_path_cost_delta_m",
            guard_thresholds.max_path_cost_delta_m,
        ),
        "max_risk_delta": _config_float(merged, "max_risk_delta", guard_thresholds.max_risk_delta)
        if guard_thresholds.max_risk_delta is not None
        else None,
        "max_risk_cost_weighted_delta": _config_float(
            merged,
            "max_risk_cost_weighted_delta",
            guard_thresholds.max_risk_cost_weighted_delta,
        )
        if guard_thresholds.max_risk_cost_weighted_delta is not None
        else None,
        "max_soft_risk_exposure_delta": _config_float(
            merged,
            "max_soft_risk_exposure_delta",
            guard_thresholds.max_soft_risk_exposure_delta,
        ),
        "max_hard_risk_violation_count": _config_float(
            merged,
            "max_hard_risk_violation_count",
            guard_thresholds.max_hard_risk_violation_count,
        ),
        "min_coverage_delta_cells": _config_float(
            merged,
            "min_coverage_delta_cells",
            guard_thresholds.min_coverage_delta_cells,
        ),
        "min_coverage_per_100m_delta": _config_float(
            merged,
            "min_coverage_per_100m_delta",
            guard_thresholds.min_coverage_per_100m_delta,
        ),
        "coverage_gain_per_path_cost_delta_mode": coverage_gain_per_path_cost_delta_mode,
        "risk_delta_hard_gate_enabled": guard_thresholds.risk_delta_hard_gate_enabled,
        "candidate_level_risk_delta_guard_is_diagnostic_only": guard_thresholds.candidate_level_risk_delta_guard_is_diagnostic_only,
        "canary_traffic_fraction_input": _config_float(merged, "canary_traffic_fraction", 0.0),
        "canary_traffic_fraction": 0.0,
    }
    for field in BOUNDARY_FIELDS:
        if merged.get(field) is True:
            config[field] = True
    return config


def _load_evidence(root: Path) -> dict[str, Any]:
    validation: list[dict[str, Any]] = []
    blocking_reasons: list[str] = []
    evidence: dict[str, Any] = {
        "coverage_comparison_summary": {},
        "coverage_comparison_aggregate": {},
        "coverage_comparison_pairs": [],
        "coverage_episodes": [],
        "paired_decision_audit": [],
    }
    for artifact_id, (filename, schema_version) in REQUIRED_JSON_ARTIFACTS.items():
        payload, reason = _read_json_artifact(root / filename, schema_version=schema_version)
        evidence[artifact_id] = payload
        validation.append(_validation_row(artifact_id, root / filename, payload, reason))
        if reason:
            blocking_reasons.append(reason)
    for artifact_id, (filename, schema_version) in REQUIRED_JSONL_ARTIFACTS.items():
        rows, reason = _read_jsonl_artifact(root / filename, schema_version=schema_version)
        evidence[artifact_id] = rows
        validation.append(_validation_row(artifact_id, root / filename, rows, reason))
        if reason:
            blocking_reasons.append(reason)
    evidence["artifact_validation"] = validation
    evidence["artifact_blocking_reasons"] = unique_sorted(blocking_reasons)
    return evidence


def _read_json_artifact(path: Path, *, schema_version: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        return {}, f"missing_{_artifact_reason_stem(path)}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, f"invalid_{_artifact_reason_stem(path)}"
    if not isinstance(payload, dict):
        return {}, f"invalid_{_artifact_reason_stem(path)}"
    if payload.get("schema_version") != schema_version:
        return payload, f"invalid_{_artifact_reason_stem(path)}_schema"
    return payload, ""


def _read_jsonl_artifact(path: Path, *, schema_version: str) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        return [], f"missing_{_artifact_reason_stem(path)}"
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                return [], f"invalid_{_artifact_reason_stem(path)}"
            rows.append(row)
    except json.JSONDecodeError:
        return [], f"invalid_{_artifact_reason_stem(path)}"
    if not rows:
        return [], f"invalid_{_artifact_reason_stem(path)}_empty"
    if any(row.get("schema_version") != schema_version for row in rows):
        return rows, f"invalid_{_artifact_reason_stem(path)}_schema"
    return rows, ""


def _artifact_reason_stem(path: Path) -> str:
    mapping = {
        COVERAGE_SUMMARY_FILE: "coverage_comparison_summary",
        COVERAGE_AGGREGATE_FILE: "coverage_comparison_aggregate",
        COVERAGE_PAIRS_FILE: "coverage_comparison_pairs",
        COVERAGE_EPISODES_FILE: "coverage_episodes",
        PAIRED_DECISION_AUDIT_FILE: "paired_decision_audit",
    }
    return mapping.get(path.name, path.stem.replace("-", "_"))


def _validation_row(artifact_id: str, path: Path, payload: Any, reason: str) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "path": str(path),
        "present": path.is_file(),
        "readable_schema_payload": not bool(reason),
        "row_count": len(payload) if isinstance(payload, list) else None,
        "reason_code": reason,
    }


def _boundary_reasons(config: dict[str, Any], evidence: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if _float(config.get("canary_traffic_fraction_input")) != 0.0:
        reasons.append("canary_traffic_fraction_nonzero")
    for field in BOUNDARY_FIELDS:
        if config.get(field) is True:
            reasons.append("boundary_violation")
    payloads: list[dict[str, Any]] = [
        evidence["coverage_comparison_summary"],
        evidence["coverage_comparison_aggregate"],
    ]
    payloads.extend(evidence["coverage_comparison_pairs"])
    payloads.extend(evidence["coverage_episodes"])
    payloads.extend(evidence["paired_decision_audit"])
    for payload in payloads:
        if not payload:
            continue
        if _float(payload.get("canary_traffic_fraction", 0.0)) != 0.0:
            reasons.append("canary_traffic_fraction_nonzero")
        for field in BOUNDARY_FIELDS:
            if payload.get(field) is True:
                reasons.append("boundary_violation")
    return unique_sorted(reasons)


def _paired_decision_summary(rows: list[dict[str, Any]], coverage_summary: dict[str, Any]) -> dict[str, Any]:
    source_summary = coverage_summary.get("same_candidate_set_policy_selection_summary", {})
    row_count = len(rows)
    same_set_rows = [row for row in rows if row.get("same_state_same_candidate_set") is True]
    disagreement_rows = [row for row in same_set_rows if row.get("policy_disagreement") is True]
    policy_preference_rows = [
        row
        for row in disagreement_rows
        if _paired_policy_preference(row)
    ]
    useful_disagreement_count = int(
        source_summary.get("useful_disagreement_count", len(policy_preference_rows)) or 0
    )
    policy_disagreement_count = int(
        source_summary.get("policy_disagreement_count", len(disagreement_rows)) or 0
    )
    same_candidate_advantage = (
        coverage_summary.get("same_candidate_set_policy_selection_advantage_established")
        if "same_candidate_set_policy_selection_advantage_established" in coverage_summary
        else useful_disagreement_count > 0
    )
    return {
        "schema_version": "xunce-stage18-5-paired-decision-summary/v1",
        "paired_decision_audit_row_count": int(source_summary.get("paired_decision_audit_row_count", row_count) or 0),
        "actual_paired_decision_audit_row_count": row_count,
        "same_state_same_candidate_set_count": len(same_set_rows),
        "policy_disagreement_count": policy_disagreement_count,
        "useful_disagreement_count": useful_disagreement_count,
        "policy_preference_attribution_count": len(policy_preference_rows),
        "same_candidate_set_advantage_established": bool(same_candidate_advantage),
    }


def _attribution_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    summary = evidence["coverage_comparison_summary"]
    pairs = evidence["coverage_comparison_pairs"]
    episodes = evidence["coverage_episodes"]
    paired = evidence["paired_decision_audit"]
    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        classes: list[str] = []
        reasons: list[str] = []
        if _float(pair.get("path_cost_delta_m")) > 0.0:
            classes.append("path_cost")
            reasons.append("path_cost_delta_m_positive")
        if _float(pair.get("coverage_per_100m_delta")) < 0.0:
            classes.append("path_cost")
            reasons.append("coverage_per_100m_delta_negative")
        if _float(pair.get("risk_delta")) > 0.0 or _float(pair.get("risk_cost_weighted_delta")) > 0.0:
            classes.append("risk_proxy")
            reasons.append("risk_or_weighted_risk_delta_positive")
        if classes:
            rows.append(
                {
                    "schema_version": "xunce-stage18-5-regression-attribution-row/v1",
                    "source_artifact": COVERAGE_PAIRS_FILE,
                    "row_index": index,
                    "scenario_id": pair.get("scenario_id"),
                    "attribution_classes": unique_sorted(classes),
                    "reason_codes": unique_sorted(reasons),
                    "risk_source_proxy_caveat": RISK_PROXY_CAVEAT if "risk_proxy" in classes else "",
                    "metrics": _compact_metrics(pair),
                }
            )
    if _float(summary.get("coverage_gain_per_path_cost_delta_vs_incumbent")) < 0.0:
        rows.append(
            {
                "schema_version": "xunce-stage18-5-regression-attribution-row/v1",
                "source_artifact": COVERAGE_SUMMARY_FILE,
                "row_index": None,
                "scenario_id": None,
                "attribution_classes": ["path_cost"],
                "reason_codes": ["summary_coverage_gain_per_path_cost_delta_negative"],
                "risk_source_proxy_caveat": "",
                "metrics": {
                    "coverage_gain_per_path_cost_delta_vs_incumbent": summary.get(
                        "coverage_gain_per_path_cost_delta_vs_incumbent"
                    )
                },
            }
        )
    for index, row in enumerate(paired):
        if _paired_policy_preference(row):
            rows.append(
                {
                    "schema_version": "xunce-stage18-5-regression-attribution-row/v1",
                    "source_artifact": PAIRED_DECISION_AUDIT_FILE,
                    "row_index": index,
                    "scenario_id": row.get("scenario_id"),
                    "attribution_classes": ["policy_preference"],
                    "reason_codes": ["xunce_higher_coverage_or_roi_at_higher_cost_or_risk"],
                    "risk_source_proxy_caveat": RISK_PROXY_CAVEAT if _paired_risk_delta(row) > 0.0 else "",
                    "metrics": _paired_metrics(row),
                }
            )
    candidate_generation_reasons = _candidate_generation_reason_codes(summary)
    if candidate_generation_reasons:
        rows.append(
            {
                "schema_version": "xunce-stage18-5-regression-attribution-row/v1",
                "source_artifact": COVERAGE_SUMMARY_FILE,
                "row_index": None,
                "scenario_id": None,
                "attribution_classes": ["candidate_generation"],
                "reason_codes": candidate_generation_reasons,
                "risk_source_proxy_caveat": "",
                "metrics": {
                    "dynamic_validation_attempt_count": summary.get("dynamic_validation_attempt_count"),
                    "dynamic_validation_success_count": summary.get("dynamic_validation_success_count"),
                    "coverage_frontier_candidate_count": summary.get("coverage_frontier_candidate_count"),
                    "undercovered_component_candidate_count": summary.get("undercovered_component_candidate_count"),
                    "low_cost_bridge_candidate_count": summary.get("low_cost_bridge_candidate_count"),
                    "conservative_local_candidate_count": summary.get("conservative_local_candidate_count"),
                },
            }
        )
    for index, episode in enumerate(episodes):
        if _episode_candidate_exhausted(episode):
            rows.append(
                {
                    "schema_version": "xunce-stage18-5-regression-attribution-row/v1",
                    "source_artifact": COVERAGE_EPISODES_FILE,
                    "row_index": index,
                    "scenario_id": episode.get("scenario_id"),
                    "attribution_classes": ["candidate_exhaustion"],
                    "reason_codes": ["candidate_generation_exhausted"],
                    "risk_source_proxy_caveat": "",
                    "metrics": {
                        "episode_termination_reason": episode.get("episode_termination_reason"),
                        "terminal_reason": episode.get("terminal_reason"),
                        "candidate_generation_exhausted_count": episode.get("candidate_generation_exhausted_count"),
                    },
                }
            )
    return rows


def _guard_evaluation(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    summary = evidence["coverage_comparison_summary"]
    aggregate = evidence["coverage_comparison_aggregate"]
    pairs = evidence["coverage_comparison_pairs"]
    missing_guard_metrics: list[str] = []
    coverage_values = _guard_numbers(
        "coverage_delta_cells",
        missing_guard_metrics,
        aggregate.get("coverage_delta_cells_min"),
        *[pair.get("coverage_delta_cells") for pair in pairs],
        aggregate.get("coverage_delta_cells_mean"),
        summary.get("xunce_new_covered_cell_delta_vs_incumbent"),
    )
    path_values = _guard_numbers(
        "path_cost_delta_m",
        missing_guard_metrics,
        *[pair.get("path_cost_delta_m") for pair in pairs],
        aggregate.get("path_cost_delta_m_mean"),
        summary.get("xunce_path_cost_delta_vs_incumbent"),
    )
    risk_delta_gate_enabled = bool(config.get("risk_delta_hard_gate_enabled", True))
    risk_values = _guard_numbers(
        "risk_delta",
        missing_guard_metrics if risk_delta_gate_enabled else [],
        *[pair.get("risk_delta") for pair in pairs],
        aggregate.get("risk_delta_mean"),
        summary.get("xunce_risk_delta_vs_incumbent"),
    )
    risk_cost_values = _guard_numbers(
        "risk_cost_weighted_delta",
        missing_guard_metrics,
        *[pair.get("soft_risk_exposure_delta") for pair in pairs],
        aggregate.get("soft_risk_exposure_delta_mean"),
        *[pair.get("risk_cost_weighted_delta") for pair in pairs],
        aggregate.get("risk_cost_weighted_delta_mean"),
        summary.get("risk_cost_weighted_delta_vs_incumbent"),
    )
    coverage_per_100m_values = _guard_numbers(
        "coverage_per_100m_delta",
        missing_guard_metrics,
        *[pair.get("coverage_per_100m_delta") for pair in pairs],
        aggregate.get("coverage_per_100m_delta_mean"),
        summary.get("coverage_per_100m_delta_vs_incumbent"),
    )
    observed = {
        "coverage_delta_cells": min(coverage_values) if coverage_values else None,
        "path_cost_delta_m": max(path_values) if path_values else None,
        "risk_delta": max(risk_values) if risk_values else None,
        "risk_cost_weighted_delta": max(risk_cost_values) if risk_cost_values else None,
        "coverage_per_100m_delta": min(coverage_per_100m_values) if coverage_per_100m_values else None,
        "coverage_gain_per_path_cost_delta": _finite_number(
            summary.get("coverage_gain_per_path_cost_delta_vs_incumbent")
        ),
    }
    audit_only_gain_per_cost = config.get("coverage_gain_per_path_cost_delta_mode") == "audit_only"
    risk_delta_check = (
        True
        if not risk_delta_gate_enabled
        else observed["risk_delta"] is not None
        and config["max_risk_delta"] is not None
        and observed["risk_delta"] <= config["max_risk_delta"]
    )
    soft_risk_check = (
        observed["risk_cost_weighted_delta"] is not None
        and observed["risk_cost_weighted_delta"] <= config["max_soft_risk_exposure_delta"]
    )
    checks = {
        "coverage_delta_cells": (
            observed["coverage_delta_cells"] is not None
            and observed["coverage_delta_cells"] >= config["min_coverage_delta_cells"]
        ),
        "path_cost_delta_m": (
            observed["path_cost_delta_m"] is not None
            and observed["path_cost_delta_m"] <= config["max_path_cost_delta_m"]
        ),
        "risk_delta": risk_delta_check,
        "risk_delta_audit_only": not risk_delta_gate_enabled,
        "risk_cost_weighted_delta": soft_risk_check,
        "soft_risk_exposure_delta": soft_risk_check,
        "coverage_per_100m_delta": (
            observed["coverage_per_100m_delta"] is not None
            and observed["coverage_per_100m_delta"] >= config["min_coverage_per_100m_delta"]
        ),
        "coverage_gain_per_path_cost_delta": (
            True
            if audit_only_gain_per_cost
            else observed["coverage_gain_per_path_cost_delta"] is not None
            and observed["coverage_gain_per_path_cost_delta"] >= 0.0
        ),
        "coverage_gain_per_path_cost_delta_audit_only": audit_only_gain_per_cost,
    }
    failed = []
    if missing_guard_metrics:
        failed.append("missing_guard_metric")
    if not checks["coverage_delta_cells"]:
        failed.append("coverage_gain_insufficient")
    if not checks["path_cost_delta_m"]:
        failed.append("path_cost_regression")
    if not checks["risk_delta"] or not checks["risk_cost_weighted_delta"]:
        failed.append("soft_risk_exposure_regression" if not risk_delta_gate_enabled else "risk_regression")
    if not checks["coverage_per_100m_delta"] or not checks["coverage_gain_per_path_cost_delta"]:
        failed.append("cost_efficiency_regression")
    return {
        "schema_version": "xunce-stage18-5-guard-evaluation/v1",
        "passed": not failed,
        "failed_guards": unique_sorted(failed),
        "reason_codes": unique_sorted(failed + missing_guard_metrics),
        "observed": observed,
        "thresholds": _guard_config_payload(config),
        "checks": checks,
    }


def _candidate_generation_diagnostics(evidence: dict[str, Any]) -> dict[str, Any]:
    summary = evidence["coverage_comparison_summary"]
    episodes = evidence["coverage_episodes"]
    exhausted_from_episodes = sum(1 for episode in episodes if _episode_candidate_exhausted(episode))
    exhausted_count = max(
        int(_float(summary.get("candidate_generation_exhausted_count"))),
        exhausted_from_episodes,
        sum(int(_float(episode.get("candidate_generation_exhausted_count"))) for episode in episodes),
    )
    reason_codes = _candidate_generation_reason_codes(summary)
    if exhausted_count > 0:
        reason_codes.append("candidate_generation_exhausted")
    return {
        "schema_version": "xunce-stage18-5-candidate-generation-diagnostics/v1",
        "candidate_generation_exhausted_count": exhausted_count,
        "reason_codes": unique_sorted(reason_codes),
        "diagnostic_only": True,
    }


def _decision(
    *,
    evidence: dict[str, Any],
    boundary_reasons: list[str],
    guard_evaluation: dict[str, Any],
    paired_summary: dict[str, Any],
    candidate_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    blocking = unique_sorted([*evidence["artifact_blocking_reasons"], *boundary_reasons])
    diagnostic: list[str] = []
    if not guard_evaluation["passed"]:
        diagnostic.append("guard_evaluation_failed")
    if not paired_summary["same_candidate_set_advantage_established"]:
        diagnostic.append("same_candidate_set_policy_selection_advantage_not_established")
    diagnostic.extend(candidate_diagnostics["reason_codes"])
    status = "failed" if blocking else "passed"
    authenticity_blockers = {
        "boundary_violation",
        "canary_traffic_fraction_nonzero",
        "missing_coverage_comparison_summary",
        "missing_coverage_comparison_aggregate",
        "missing_coverage_comparison_pairs",
        "missing_coverage_episodes",
        "missing_paired_decision_audit",
    }
    evidence_gate = not bool(set(blocking) & authenticity_blockers) and not any(
        reason.startswith("invalid_") for reason in blocking
    )
    candidate_gate = evidence_gate
    return {
        "status": status,
        "reason_codes": unique_sorted([*blocking, *diagnostic]),
        "blocking_reason_codes": unique_sorted(blocking),
        "diagnostic_reason_codes": unique_sorted(diagnostic),
        "evidence_authenticity_gate_passed": evidence_gate,
        "candidate_validity_gate_passed": candidate_gate,
        "comparison_allowed": status == "passed" and evidence_gate and candidate_gate,
    }


def _next_stage_routing(
    *,
    decision: dict[str, Any],
    guard_evaluation: dict[str, Any],
    paired_summary: dict[str, Any],
) -> dict[str, Any]:
    blockers = set(decision["blocking_reason_codes"])
    if {"boundary_violation", "canary_traffic_fraction_nonzero"} & blockers:
        primary_route = BOUNDARY_NEXT_REQUIRED_CHANGE
    elif any(reason.startswith("missing_") or reason.startswith("invalid_") for reason in blockers):
        primary_route = MISSING_ARTIFACT_NEXT_REQUIRED_CHANGE
    elif not guard_evaluation["passed"]:
        primary_route = GUARD_REFINEMENT_NEXT_REQUIRED_CHANGE
    elif not paired_summary["same_candidate_set_advantage_established"]:
        primary_route = PAIRED_ADVANTAGE_NEXT_REQUIRED_CHANGE
    else:
        primary_route = STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE
    return {
        "schema_version": "xunce-stage18-5-next-stage-routing/v1",
        "primary_route": primary_route,
        "stage19_authorized": False,
        "stage19_readiness": "not_authorized"
        if primary_route != STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE
        else "ready_for_stage19_preflight_human_review_only",
    }


def _stage19_readiness(
    paired_summary: dict[str, Any],
    guard_evaluation: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    ready = (
        decision["status"] == "passed"
        and guard_evaluation["passed"]
        and paired_summary["same_candidate_set_advantage_established"]
    )
    return {
        "schema_version": "xunce-stage18-5-stage19-readiness/v1",
        "readiness": "ready_for_stage19_preflight_human_review_only" if ready else "not_authorized",
        "authorized": False,
        "same_candidate_set_policy_selection_advantage_established": paired_summary[
            "same_candidate_set_advantage_established"
        ],
        "guard_passed": guard_evaluation["passed"],
        "evidence_status": decision["status"],
    }


def _candidate_generation_reason_codes(summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if int(_float(summary.get("candidate_generation_exhausted_count"))) > 0:
        reasons.append("dynamic_candidate_exhausted")
    attempts = int(_float(summary.get("dynamic_validation_attempt_count")))
    successes = int(_float(summary.get("dynamic_validation_success_count")))
    if attempts > 0 and successes < attempts:
        reasons.append("validated_candidate_shortage")
    frontier_count = int(_float(summary.get("coverage_frontier_candidate_count"))) + int(
        _float(summary.get("undercovered_component_candidate_count"))
    )
    backup_count = int(_float(summary.get("low_cost_bridge_candidate_count"))) + int(
        _float(summary.get("conservative_local_candidate_count"))
    )
    if backup_count > 0 and frontier_count <= 0:
        reasons.append("frontier_backup_family_imbalance")
    return unique_sorted(reasons)


def _episode_candidate_exhausted(row: dict[str, Any]) -> bool:
    return (
        row.get("candidate_generation_exhausted") is True
        or row.get("episode_termination_reason") == "candidate_generation_exhausted"
        or row.get("terminal_reason") == "candidate_generation_exhausted"
        or int(_float(row.get("candidate_generation_exhausted_count"))) > 0
        or "candidate_generation_exhausted" in set(row.get("reason_codes", []))
    )


def _paired_policy_preference(row: dict[str, Any]) -> bool:
    if row.get("same_state_same_candidate_set") is not True:
        return False
    if row.get("policy_disagreement") is not True and row.get("xunce_selected_action_index") == row.get(
        "incumbent_selected_action_index"
    ):
        return False
    coverage_delta = _paired_coverage_delta(row)
    cost_delta = _paired_cost_delta(row)
    risk_delta = _paired_risk_delta(row)
    return coverage_delta > 0.0 and (cost_delta > 0.0 or risk_delta > 0.0)


def _paired_coverage_delta(row: dict[str, Any]) -> float:
    return max(
        _delta(
            row,
            "xunce_selected_expected_new_coverage_cell_count",
            "incumbent_selected_expected_new_coverage_cell_count",
        ),
        _delta(row, "xunce_selected_roi_weighted_coverage_delta", "incumbent_selected_roi_weighted_coverage_delta"),
        _delta(row, "xunce_expected_coverage_rate_delta", "incumbent_expected_coverage_rate_delta"),
        _delta(row, "xunce_roi_weighted_coverage_delta", "incumbent_roi_weighted_coverage_delta"),
        _delta(row, "xunce_coverage_gain_per_path_cost", "incumbent_coverage_gain_per_path_cost"),
        _delta(row, "xunce_expected_new_coverage_area", "incumbent_expected_new_coverage_area"),
    )


def _paired_cost_delta(row: dict[str, Any]) -> float:
    return max(
        _delta(row, "xunce_selected_path_cost", "incumbent_selected_path_cost"),
        _delta(row, "xunce_path_cost", "incumbent_path_cost"),
        _delta(row, "xunce_path_cost_m", "incumbent_path_cost_m"),
    )


def _paired_risk_delta(row: dict[str, Any]) -> float:
    return max(
        _delta(row, "xunce_selected_risk", "incumbent_selected_risk"),
        _delta(row, "xunce_risk", "incumbent_risk"),
        _delta(row, "xunce_risk_cost_weighted", "incumbent_risk_cost_weighted"),
    )


def _delta(row: dict[str, Any], left: str, right: str) -> float:
    if left not in row or right not in row:
        return 0.0
    return _float(row.get(left)) - _float(row.get(right))


def _compact_metrics(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "coverage_delta_cells",
        "path_cost_delta_m",
        "risk_delta",
        "risk_cost_weighted_delta",
        "coverage_per_100m_delta",
        "roi_weighted_coverage_delta",
    )
    return {key: row.get(key) for key in keys if key in row}


def _paired_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "coverage_delta": _paired_coverage_delta(row),
        "path_cost_delta": _paired_cost_delta(row),
        "risk_delta": _paired_risk_delta(row),
        "xunce_selected_action_index": row.get("xunce_selected_action_index"),
        "incumbent_selected_action_index": row.get("incumbent_selected_action_index"),
    }


def _attribution_class_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "candidate_generation": 0,
        "policy_preference": 0,
        "path_cost": 0,
        "risk_proxy": 0,
        "candidate_exhaustion": 0,
    }
    for row in rows:
        for class_name in row.get("attribution_classes", []):
            counts[class_name] = counts.get(class_name, 0) + 1
    return dict(sorted(counts.items()))


def _regression_count_summary(summary: dict[str, Any], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage18-5-regression-count-summary/v1",
        "source_efficiency_regression_count": int(_float(summary.get("xunce_efficiency_regression_count"))),
        "source_safety_regression_count": int(_float(summary.get("xunce_safety_regression_count"))),
        "efficiency_regression_attributed_count": sum(
            1
            for pair in pairs
            if _float(pair.get("path_cost_delta_m")) > 0.0 or _float(pair.get("coverage_per_100m_delta")) < 0.0
        ),
        "safety_regression_attributed_count": sum(1 for pair in pairs if _float(pair.get("risk_delta")) > 0.0),
        "risk_proxy_attribution_count": sum(
            1
            for pair in pairs
            if _float(pair.get("risk_delta")) > 0.0 or _float(pair.get("risk_cost_weighted_delta")) > 0.0
        ),
    }


def _risk_source_summary(summary: dict[str, Any], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    episode_sources: dict[str, int] = {}
    for episode in episodes:
        source = episode.get("risk_source")
        if source:
            episode_sources[str(source)] = episode_sources.get(str(source), 0) + 1
    return {
        "risk_source_counts": summary.get("risk_source_counts", {}),
        "formal_risk_source_counts": summary.get("formal_risk_source_counts", {}),
        "selected_risk_source_counts": summary.get("selected_risk_source_counts", {}),
        "route_derived_risk_count": int(_float(summary.get("route_derived_risk_count"))),
        "episode_risk_source_counts": dict(sorted(episode_sources.items())),
        "caveat": RISK_PROXY_CAVEAT,
    }


def _manifest(
    paths: dict[str, Path],
    *,
    config: dict[str, Any],
    evidence: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "xunce-stage18-5-evidence-attribution-manifest/v1",
        "generated_at": summary["generated_at"],
        "input_root": config["coverage_comparison_root"],
        "input_artifacts": [row["path"] for row in evidence["artifact_validation"]],
        "output_artifacts": {key: str(path) for key, path in sorted(paths.items())},
        "status": summary["status"],
        "next_required_change": summary["next_required_change"],
        "governance_boundary": _boundary_fields(),
    }


def _guard_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile_id": config["profile_id"],
        "profile_version": config["profile_version"],
        "profile_hash": config["profile_hash"],
        "max_path_cost_delta_m": config["max_path_cost_delta_m"],
        "max_risk_delta": config["max_risk_delta"],
        "max_risk_cost_weighted_delta": config["max_risk_cost_weighted_delta"],
        "max_soft_risk_exposure_delta": config["max_soft_risk_exposure_delta"],
        "max_hard_risk_violation_count": config["max_hard_risk_violation_count"],
        "min_coverage_delta_cells": config["min_coverage_delta_cells"],
        "min_coverage_per_100m_delta": config["min_coverage_per_100m_delta"],
        "coverage_gain_per_path_cost_delta_mode": config["coverage_gain_per_path_cost_delta_mode"],
        "risk_delta_hard_gate_enabled": config["risk_delta_hard_gate_enabled"],
        "candidate_level_risk_delta_guard_is_diagnostic_only": config[
            "candidate_level_risk_delta_guard_is_diagnostic_only"
        ],
        "canary_traffic_fraction": 0.0,
    }


def _boundary_fields() -> dict[str, Any]:
    fields = {field: False for field in BOUNDARY_FIELDS}
    fields["canary_traffic_fraction"] = 0.0
    return fields


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Stage 18.5 Evidence Attribution Review",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- evidence_authenticity_gate_passed: `{summary['evidence_authenticity_gate_passed']}`",
            f"- guard_passed: `{summary['guard_evaluation']['passed']}`",
            f"- failed_guards: `{summary['guard_evaluation']['failed_guards']}`",
            f"- same_candidate_set_advantage_established: `{summary['paired_decision_summary']['same_candidate_set_advantage_established']}`",
            f"- stage19_readiness: `{summary['stage19_readiness']['readiness']}`",
            f"- attribution_class_counts: `{summary['attribution_class_counts']}`",
            "",
            "This review is offline evidence attribution only. It does not start PPO, publish checkpoints, replace the default policy, connect a real executor, start online canary traffic, or make real-world performance claims.",
            "",
        ]
    )


def _float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _config_float(payload: dict[str, Any], field: str, default: float) -> float:
    if field not in payload:
        return float(default)
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{field} must be a finite number")
    if not math.isfinite(float(value)):
        raise ConfigError(f"{field} must be a finite number")
    return float(value)


def _guard_numbers(metric: str, missing_guard_metrics: list[str], *values: Any) -> list[float]:
    numbers = [float(value) for value in values if not isinstance(value, bool) and isinstance(value, int | float)]
    if not numbers:
        missing_guard_metrics.append(f"missing_{metric}")
    return numbers


def _first_number(*values: Any) -> float:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return float(value)
    return 0.0


def _minimum_number(*values: Any) -> float:
    numbers = [float(value) for value in values if not isinstance(value, bool) and isinstance(value, int | float)]
    return min(numbers) if numbers else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
