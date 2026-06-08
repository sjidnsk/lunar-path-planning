from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot
from git_provenance import git_snapshots_match as _git_snapshots_match

from run_controlled_hybrid_policy_training_candidate import (
    CHECKPOINT_METADATA_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION as CANDIDATE_SUMMARY_SCHEMA_VERSION,
)
from run_fresh_holdout_policy_candidate_evaluation import (
    SUMMARY_SCHEMA_VERSION as FRESH_HOLDOUT_SCHEMA_VERSION,
)
from run_fresh_holdout_policy_candidate_evaluation import (
    _candidate_features,
    _global_features,
    _missing_indicators,
)


CONFIG_SCHEMA_VERSION = "scenario-disjoint-policy-rollout-evaluation-config/v1"
SUMMARY_SCHEMA_VERSION = "scenario-disjoint-policy-rollout-evaluation-summary/v1"
DECISION_SCHEMA_VERSION = "scenario-disjoint-policy-rollout-decision/v1"
REGRESSION_REPORT_SCHEMA_VERSION = "scenario-disjoint-policy-rollout-regression-report/v1"
NEXT_REQUIRED_CHANGE = "policy_rollout_objective_or_sample_weight_refinement_required"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an experimental policy candidate in controlled scenario-disjoint shadow rollout."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    _install_model_explorer_path(repo_root)
    source_root = _resolve_path(args.source_root, repo_root)
    candidate_root = _resolve_path(args.candidate_root, repo_root)
    batch_root = _resolve_path(args.batch_root, repo_root)
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    output_paths = _output_paths(batch_root, config)
    summary, decisions, regression_report = run_scenario_disjoint_policy_rollout_evaluation(
        source_root=source_root,
        candidate_root=candidate_root,
        batch_root=batch_root,
        config=config,
        repo_root=repo_root,
        output_paths=output_paths,
    )
    print(
        json.dumps(
            {
                "status": "config validated" if summary["status"] == "passed" else "validation failed",
                "source_root": _display_path(source_root, repo_root),
                "candidate_root": _display_path(candidate_root, repo_root),
                "batch_root": _display_path(batch_root, repo_root),
                "reason_codes": summary["reason_codes"],
                "scenario_disjoint_context_count": summary["scenario_disjoint_context_count"],
                "policy_decision_count": summary["policy_decision_count"],
                "regression_count": summary["regression_count"],
                "summary": _display_path(output_paths["summary"], repo_root),
            },
            ensure_ascii=False,
        )
    )
    if not args.validate_only:
        batch_root.mkdir(parents=True, exist_ok=True)
        _write_json(output_paths["summary"], summary)
        _write_json(output_paths["regression_report"], regression_report)
        output_paths["decisions"].write_text(
            "".join(json.dumps(decision, ensure_ascii=False) + "\n" for decision in decisions),
            encoding="utf-8",
        )
    return 1 if summary["status"] == "failed" else 0


def run_scenario_disjoint_policy_rollout_evaluation(
    *,
    source_root: Path,
    candidate_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    output_paths: dict[str, Path],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    reason_codes: list[str] = []
    paths = _input_paths(source_root, candidate_root, batch_root, config)
    source_batch = _load_summary(paths["source_batch_summary"], expected_schema=None, label="source_batch_summary", reason_codes=reason_codes)
    holdout_batch = _load_summary(paths["holdout_batch_summary"], expected_schema=None, label="holdout_batch_summary", reason_codes=reason_codes)
    fresh_holdout = _load_summary(
        paths["fresh_holdout_summary"],
        expected_schema=FRESH_HOLDOUT_SCHEMA_VERSION,
        label="fresh_holdout_summary",
        reason_codes=reason_codes,
    )
    candidate = _load_summary(
        paths["candidate_summary"],
        expected_schema=CANDIDATE_SUMMARY_SCHEMA_VERSION,
        label="candidate_summary",
        reason_codes=reason_codes,
    )
    metadata = _load_summary(
        paths["checkpoint_metadata"],
        expected_schema=CHECKPOINT_METADATA_SCHEMA_VERSION,
        label="checkpoint_metadata",
        reason_codes=reason_codes,
    )

    if source_batch.get("failed_count", 0) or _string_list(source_batch.get("reason_codes")):
        _append_reason(reason_codes, "source_batch_failed")
    if holdout_batch.get("failed_count", 0) or _string_list(holdout_batch.get("reason_codes")):
        _append_reason(reason_codes, "holdout_batch_failed")
    if fresh_holdout.get("status") != "passed" or _string_list(fresh_holdout.get("reason_codes")):
        _append_reason(reason_codes, "fresh_holdout_summary_failed")
    if candidate.get("status") != "passed" or candidate.get("candidate_training_status") != "passed":
        _append_reason(reason_codes, "candidate_summary_failed")
    if _string_list(candidate.get("reason_codes")):
        _append_reason(reason_codes, "candidate_summary_failed")
    if not metadata.get("experimental", False):
        _append_reason(reason_codes, "checkpoint_metadata_not_experimental")
    if metadata.get("publishes_checkpoint") or candidate.get("publishes_checkpoint"):
        _append_reason(reason_codes, "checkpoint_publication_detected")
    if metadata.get("replaces_default_policy") or candidate.get("replaces_default_policy"):
        _append_reason(reason_codes, "default_policy_replacement_detected")
    if metadata.get("performance_claimed") or candidate.get("performance_claimed"):
        _append_reason(reason_codes, "performance_claim_detected")
    if not paths["checkpoint"].is_file():
        _append_reason(reason_codes, "experimental_checkpoint_missing")

    current_git = _git_snapshot(repo_root)
    candidate_git_current_matches_sources = _summary_git_matches_current(candidate, current_git)
    checkpoint_metadata_git_current_matches_sources = _summary_git_matches_current(metadata, current_git)
    if config["validation"].get("require_candidate_git_current_match"):
        if not candidate_git_current_matches_sources:
            _append_reason(reason_codes, "candidate_git_current_mismatch")
        if not checkpoint_metadata_git_current_matches_sources:
            _append_reason(reason_codes, "checkpoint_metadata_git_current_mismatch")

    scenario_groups = _collect_holdout_scenarios(batch_root, repo_root)
    context_records = [candidate for group in scenario_groups for candidate in group["candidates"]]
    context_id_missing_count = sum(1 for record in context_records if not record.get("context_id"))
    if config["validation"].get("require_context_id") and context_id_missing_count:
        _append_reason(reason_codes, "context_id_missing")

    decisions: list[dict[str, Any]] = []
    if paths["checkpoint"].is_file() and not any(
        reason
        in {
            "checkpoint_metadata_not_experimental",
            "checkpoint_publication_detected",
            "default_policy_replacement_detected",
            "performance_claim_detected",
            "experimental_checkpoint_missing",
        }
        for reason in reason_codes
    ):
        try:
            decisions = _score_scenarios(
                checkpoint_path=paths["checkpoint"],
                scenario_groups=scenario_groups,
                config=config,
                repo_root=repo_root,
            )
        except Exception as exc:  # noqa: BLE001
            _append_reason(reason_codes, "policy_rollout_scoring_failed")
            decisions = []
            scoring_error = str(exc)
        else:
            scoring_error = None
    else:
        scoring_error = None

    counts = Counter(decision["decision_class"] for decision in decisions)
    regression_decisions = [
        decision for decision in decisions if decision.get("decision_class") == "regression"
    ]
    invalid_action_mask_count = sum(1 for decision in decisions if not decision.get("action_mask_valid"))
    fallback_or_open_grid_count = max(
        _fallback_or_open_grid_count(holdout_batch),
        sum(1 for decision in decisions if "fallback_or_open_grid" in decision.get("regression_reason_codes", [])),
    )
    safety_regression_count = max(
        _int_value(holdout_batch.get("safety_regression_count")),
        sum(1 for decision in decisions if "safety_regression" in decision.get("regression_reason_codes", [])),
    )
    contract_violation_count = max(
        _int_value(holdout_batch.get("candidate_contract_alignment_gap_count")),
        _int_value(holdout_batch.get("contract_violation_count")),
        sum(1 for decision in decisions if "contract_violation" in decision.get("regression_reason_codes", [])),
    )
    path_cost_regression_count = sum(
        1 for decision in decisions if "path_cost_regression" in decision.get("regression_reason_codes", [])
    )
    risk_regression_count = sum(
        1 for decision in decisions if "risk_regression" in decision.get("regression_reason_codes", [])
    )
    source_selection_regression_count = sum(
        1 for decision in decisions if "source_selection_regression" in decision.get("regression_reason_codes", [])
    )
    regression_count = len(regression_decisions)
    validation = config["validation"]
    if len(context_records) < _int_value(validation.get("min_scenario_disjoint_context_count")):
        _append_reason(reason_codes, "scenario_disjoint_context_count_zero")
    if invalid_action_mask_count > _int_value(validation.get("max_invalid_action_mask_count")):
        _append_reason(reason_codes, "invalid_action_mask")
    if fallback_or_open_grid_count > _int_value(validation.get("max_fallback_or_open_grid_count")):
        _append_reason(reason_codes, "fallback_or_open_grid")
    if safety_regression_count > _int_value(validation.get("max_safety_regression_count")):
        _append_reason(reason_codes, "safety_regression")
    if contract_violation_count > _int_value(validation.get("max_contract_violation_count")):
        _append_reason(reason_codes, "contract_violation")
    if path_cost_regression_count > _int_value(validation.get("max_path_cost_regression_count")):
        _append_reason(reason_codes, "path_cost_regression")
    if risk_regression_count > _int_value(validation.get("max_risk_regression_count")):
        _append_reason(reason_codes, "risk_regression")
    if source_selection_regression_count > _int_value(validation.get("max_source_selection_regression_count")):
        _append_reason(reason_codes, "source_selection_regression")
    if regression_count > _int_value(validation.get("max_regression_count")):
        _append_reason(reason_codes, "policy_rollout_regression")

    status = "failed" if reason_codes else "passed"
    regression_report = {
        "schema_version": REGRESSION_REPORT_SCHEMA_VERSION,
        "status": status,
        "reason_codes": reason_codes,
        "regression_count": regression_count,
        "raw_policy_regression_count": sum(
            1 for decision in decisions if decision.get("raw_policy_decision_class") == "regression"
        ),
        "regression_decisions": regression_decisions,
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "source_root": _display_path(source_root, repo_root),
        "candidate_root": _display_path(candidate_root, repo_root),
        "batch_root": _display_path(batch_root, repo_root),
        "summary": _display_path(output_paths["summary"], repo_root),
        "decisions_path": _display_path(output_paths["decisions"], repo_root),
        "regression_report_path": _display_path(output_paths["regression_report"], repo_root),
        "candidate_summary_path": _display_path(paths["candidate_summary"], repo_root),
        "checkpoint_path": _display_path(paths["checkpoint"], repo_root),
        "checkpoint_metadata_path": _display_path(paths["checkpoint_metadata"], repo_root),
        "shadow_mode": bool(config["evaluation"].get("shadow_mode", True)),
        "controlled_selection_mode": bool(config["evaluation"].get("controlled_selection_mode", False)),
        "scenario_disjoint_context_count": len(context_records),
        "policy_decision_count": len(decisions),
        "decision_changed_count": sum(1 for decision in decisions if decision.get("decision_changed")),
        "aligned_decision_count": counts.get("aligned", 0),
        "acceptable_alternative_count": counts.get("acceptable_alternative", 0),
        "regression_count": regression_count,
        "raw_policy_regression_count": regression_report["raw_policy_regression_count"],
        "context_id_missing_count": context_id_missing_count,
        "invalid_action_mask_count": invalid_action_mask_count,
        "fallback_or_open_grid_count": fallback_or_open_grid_count,
        "safety_regression_count": safety_regression_count,
        "contract_violation_count": contract_violation_count,
        "path_cost_regression_count": path_cost_regression_count,
        "risk_regression_count": risk_regression_count,
        "source_selection_regression_count": source_selection_regression_count,
        "candidate_git_current_matches_sources": candidate_git_current_matches_sources,
        "checkpoint_metadata_git_current_matches_sources": checkpoint_metadata_git_current_matches_sources,
        "scoring_error": scoring_error,
        "next_required_change": NEXT_REQUIRED_CHANGE if regression_count else None,
        "experimental_checkpoint": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": current_git, "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }
    return summary, decisions, regression_report


def _score_scenarios(
    *,
    checkpoint_path: Path,
    scenario_groups: list[dict[str, Any]],
    config: dict[str, Any],
    repo_root: Path,
) -> list[dict[str, Any]]:
    import torch

    from model_explorer.policy.architectures import build_policy_network
    from model_explorer.policy.features import (
        CANDIDATE_FEATURE_NAMES,
        GLOBAL_FEATURE_NAMES,
        MISSING_INDICATOR_NAMES,
        PolicyObservation,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training = checkpoint.get("training", {}) if isinstance(checkpoint, dict) else {}
    observation = PolicyObservation(
        candidate_feature_names=CANDIDATE_FEATURE_NAMES,
        candidate_features=(
            tuple([0.0] * len(CANDIDATE_FEATURE_NAMES)),
            tuple([0.0] * len(CANDIDATE_FEATURE_NAMES)),
        ),
        global_feature_names=GLOBAL_FEATURE_NAMES,
        global_features=tuple([0.0] * len(GLOBAL_FEATURE_NAMES)),
        action_mask=(True, True),
        candidate_cells=((0, 0), (0, 1)),
        candidate_missing_indicator_names=MISSING_INDICATOR_NAMES,
        candidate_missing_indicators=(
            tuple([0.0] * len(MISSING_INDICATOR_NAMES)),
            tuple([0.0] * len(MISSING_INDICATOR_NAMES)),
        ),
    )
    network = build_policy_network(
        None,
        observation=observation,
        hidden_size=_int_value(training.get("hidden_size") or config["evaluation"].get("hidden_size")),
    )
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    decisions: list[dict[str, Any]] = []
    for group in scenario_groups:
        candidates = group["candidates"]
        if not candidates:
            continue
        source = _source_selected_candidate(group)
        valid_mask = [bool(record.get("reachable")) and not bool(record.get("replan_required")) for record in candidates]
        if not any(valid_mask):
            decisions.append(_decision_record(group, source, source, source, [], [], action_mask_valid=False))
            continue
        candidate_features = torch.tensor(
            [[_candidate_features(record) for record in candidates]],
            dtype=torch.float32,
        )
        global_features = torch.tensor([_global_features(candidates)], dtype=torch.float32)
        action_mask = torch.tensor([valid_mask], dtype=torch.bool)
        missing = torch.tensor([[_missing_indicators(record) for record in candidates]], dtype=torch.float32)
        with torch.no_grad():
            logits = network(
                candidate_features=candidate_features,
                global_features=global_features,
                action_mask=action_mask,
                candidate_missing_indicators=missing,
            ).masked_logits[0].tolist()
        raw_index = max(range(len(candidates)), key=lambda index: logits[index])
        raw_policy = candidates[raw_index]
        for candidate, logit in zip(candidates, logits):
            candidate["policy_logit"] = float(logit)
        raw_reasons = _regression_reasons(raw_policy, source, config=config)
        controlled = raw_policy if not raw_reasons else source
        controlled_reasons = _regression_reasons(controlled, source, config=config)
        decisions.append(
            _decision_record(
                group,
                source,
                raw_policy,
                controlled,
                raw_reasons,
                controlled_reasons,
                action_mask_valid=_candidate_action_mask_valid(controlled),
            )
        )
    return decisions


def _decision_record(
    group: dict[str, Any],
    source: dict[str, Any] | None,
    raw_policy: dict[str, Any] | None,
    controlled: dict[str, Any] | None,
    raw_reasons: list[str],
    controlled_reasons: list[str],
    *,
    action_mask_valid: bool,
) -> dict[str, Any]:
    controlled = controlled or {}
    source = source or {}
    raw_policy = raw_policy or {}
    decision_changed = _action_index(controlled) != _action_index(source)
    decision_class = "regression" if controlled_reasons else ("acceptable_alternative" if decision_changed else "aligned")
    raw_policy_changed = _action_index(raw_policy) != _action_index(source)
    raw_policy_decision_class = "regression" if raw_reasons else ("acceptable_alternative" if raw_policy_changed else "aligned")
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "run_id": group.get("run_id"),
        "source_path": group.get("source_path"),
        "context_id": controlled.get("context_id") or source.get("context_id") or raw_policy.get("context_id"),
        "scenario_id": group.get("scenario_id"),
        "scenario_group": group.get("scenario_group"),
        "scenario_seed": group.get("scenario_seed"),
        "scenario_variant_id": group.get("scenario_variant_id"),
        "diagnostic_profile": group.get("diagnostic_profile"),
        "planning_backend": group.get("planning_backend"),
        "source_selected_action_index": _action_index(source),
        "raw_policy_selected_action_index": _action_index(raw_policy),
        "policy_selected_action_index": _action_index(controlled),
        "source_selected_context_id": source.get("context_id"),
        "raw_policy_selected_context_id": raw_policy.get("context_id"),
        "policy_selected_context_id": controlled.get("context_id"),
        "action_mask_valid": bool(action_mask_valid),
        "policy_selected_candidate_present": bool(controlled),
        "policy_selected_contract_safe": _contract_safe(controlled),
        "policy_selected_path_cost_delta": _delta(controlled.get("path_cost"), source.get("path_cost")),
        "policy_selected_risk_delta": _delta(controlled.get("risk"), source.get("risk")),
        "decision_changed": decision_changed,
        "decision_class": decision_class,
        "regression_reason_codes": controlled_reasons,
        "raw_policy_decision_class": raw_policy_decision_class,
        "raw_policy_regression_reason_codes": raw_reasons,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
    }


def _collect_holdout_scenarios(batch_root: Path, repo_root: Path) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for summary_path in sorted(batch_root.glob("**/path-feedback-summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        run_id = summary_path.parent.name
        for scenario in payload.get("scenarios", []):
            if not isinstance(scenario, dict):
                continue
            path_feedback = scenario.get("path_feedback") if isinstance(scenario.get("path_feedback"), dict) else {}
            candidates = path_feedback.get("candidates")
            candidates = candidates if isinstance(candidates, list) else []
            group = {
                "run_id": run_id,
                "source_path": _display_path(summary_path, repo_root),
                "scenario_id": str(scenario.get("scenario_id", "")),
                "scenario_group": str(scenario.get("scenario_group") or "unknown"),
                "scenario_seed": scenario.get("scenario_seed"),
                "scenario_variant_id": scenario.get("scenario_variant_id"),
                "diagnostic_profile": payload.get("diagnostic_profile"),
                "planning_backend": _planning_backend(payload),
                "best_by_path_cost": path_feedback.get("best_by_path_cost") if isinstance(path_feedback.get("best_by_path_cost"), dict) else None,
                "candidates": [
                    _candidate_record(candidate, scenario=scenario, summary=payload, group_path=summary_path, repo_root=repo_root)
                    for candidate in candidates
                    if isinstance(candidate, dict)
                ],
            }
            groups.append(group)
    return groups


def _candidate_record(
    candidate: dict[str, Any],
    *,
    scenario: dict[str, Any],
    summary: dict[str, Any],
    group_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    source_action_index = candidate.get("source_action_index", candidate.get("action_index"))
    generation = candidate.get("candidate_generation") if isinstance(candidate.get("candidate_generation"), dict) else {}
    anchor_projection = (
        candidate.get("platform_goal_feasibility", {}).get("anchor_projection", {})
        if isinstance(candidate.get("platform_goal_feasibility"), dict)
        else {}
    )
    if not isinstance(anchor_projection, dict):
        anchor_projection = {}
    return {
        "source_path": _display_path(group_path, repo_root),
        "scenario_id": str(scenario.get("scenario_id", "")),
        "scenario_group": str(scenario.get("scenario_group") or "unknown"),
        "scenario_seed": scenario.get("scenario_seed"),
        "scenario_variant_id": scenario.get("scenario_variant_id"),
        "diagnostic_profile": summary.get("diagnostic_profile"),
        "planning_backend": _planning_backend(summary),
        "context_id": str(candidate.get("context_id")) if candidate.get("context_id") else None,
        "context_id_schema_version": candidate.get("context_id_schema_version"),
        "context_id_source": candidate.get("context_id_source"),
        "source_action_index": _optional_int(source_action_index),
        "action_index": _optional_int(candidate.get("action_index")),
        "candidate_role": candidate.get("candidate_role"),
        "policy_target_cell": _cell(candidate.get("policy_target_cell"), candidate.get("cell")),
        "execution_goal_cell": _cell(candidate.get("execution_goal_cell"), candidate.get("cell")),
        "target_binding_mode": candidate.get("target_binding_mode") or candidate.get("candidate_role"),
        "reachable": bool(candidate.get("reachable")),
        "replan_required": bool(candidate.get("replan_required")),
        "open_grid_fallback_used": bool(candidate.get("open_grid_fallback_used") or scenario.get("open_grid_fallback_used")),
        "tracking_safety_violation_count": _int_value(
            candidate.get("tracking_safety_violation_count", scenario.get("tracking_safety_violation_count"))
        ),
        "contract_safe": _contract_safe(candidate),
        "path_cost": _float_or_none(candidate.get("path_cost")),
        "risk": _float_or_none(candidate.get("risk")),
        "utility": _float_or_none(candidate.get("utility")),
        "source_selection_status": generation.get("source_selection_status") or anchor_projection.get("source_selection_status"),
        "source_selection_quality_regression": bool(
            generation.get("source_selection_quality_regression")
            or anchor_projection.get("source_selection_quality_regression")
        ),
        "planner_validated_mining_decision": generation.get("planner_validated_mining_decision")
        or anchor_projection.get("planner_validated_mining_decision"),
    }


def _source_selected_candidate(group: dict[str, Any]) -> dict[str, Any] | None:
    candidates = group.get("candidates", [])
    for candidate in candidates:
        if candidate.get("source_selection_status") == "source_selected":
            return candidate
    for candidate in candidates:
        decision = str(candidate.get("planner_validated_mining_decision") or "")
        if decision.startswith("selected_"):
            return candidate
    best = group.get("best_by_path_cost")
    if isinstance(best, dict):
        best_context = best.get("context_id")
        for candidate in candidates:
            if best_context and candidate.get("context_id") == best_context:
                return candidate
        best_action = _optional_int(best.get("source_action_index", best.get("action_index")))
        best_cell = _cell(best.get("policy_target_cell"), best.get("cell"))
        for candidate in candidates:
            if _action_index(candidate) == best_action and candidate.get("policy_target_cell") == best_cell:
                return candidate
    valid = [candidate for candidate in candidates if _candidate_action_mask_valid(candidate)]
    if not valid:
        return candidates[0] if candidates else None
    return min(valid, key=lambda record: _float_or_none(record.get("path_cost")) or 0.0)


def _regression_reasons(candidate: dict[str, Any] | None, source: dict[str, Any] | None, *, config: dict[str, Any]) -> list[str]:
    if not candidate:
        return ["policy_selected_candidate_missing"]
    reasons: list[str] = []
    candidate_changed = not _same_candidate(candidate, source)
    if not _candidate_action_mask_valid(candidate):
        reasons.append("invalid_action_mask")
    if candidate.get("open_grid_fallback_used"):
        reasons.append("fallback_or_open_grid")
    if _int_value(candidate.get("tracking_safety_violation_count")):
        reasons.append("safety_regression")
    if candidate_changed and not _contract_safe(candidate):
        reasons.append("contract_violation")
    if source:
        path_delta = _delta(candidate.get("path_cost"), source.get("path_cost"))
        risk_delta = _delta(candidate.get("risk"), source.get("risk"))
        if path_delta > float(config["evaluation"].get("max_path_cost_regression", 0.0)):
            reasons.append("path_cost_regression")
        if risk_delta > float(config["evaluation"].get("max_risk_regression", 0.0)):
            reasons.append("risk_regression")
    if candidate_changed and candidate.get("source_selection_quality_regression"):
        reasons.append("source_selection_regression")
    return reasons


def _same_candidate(candidate: dict[str, Any] | None, source: dict[str, Any] | None) -> bool:
    if not candidate or not source:
        return False
    candidate_context = candidate.get("context_id")
    source_context = source.get("context_id")
    if candidate_context and source_context:
        return candidate_context == source_context
    return (
        _action_index(candidate) == _action_index(source)
        and candidate.get("policy_target_cell") == source.get("policy_target_cell")
        and candidate.get("execution_goal_cell") == source.get("execution_goal_cell")
        and candidate.get("target_binding_mode") == source.get("target_binding_mode")
    )


def _candidate_action_mask_valid(candidate: dict[str, Any] | None) -> bool:
    return bool(candidate and candidate.get("reachable") and not candidate.get("replan_required"))


def _action_index(candidate: dict[str, Any] | None) -> int | None:
    if not candidate:
        return None
    value = candidate.get("source_action_index")
    if value is None:
        value = candidate.get("action_index")
    return _optional_int(value)


def _contract_safe(candidate: dict[str, Any] | None) -> bool:
    if not candidate:
        return False
    if "contract_safe" in candidate:
        return candidate.get("contract_safe") is not False
    generation = candidate.get("candidate_generation")
    if isinstance(generation, dict) and "contract_safe" in generation:
        return generation.get("contract_safe") is not False
    feasibility = candidate.get("platform_goal_feasibility")
    if isinstance(feasibility, dict):
        if "contract_reachable" in feasibility:
            return feasibility.get("contract_reachable") is not False
        anchor_projection = feasibility.get("anchor_projection")
        if isinstance(anchor_projection, dict) and "contract_safe" in anchor_projection:
            return anchor_projection.get("contract_safe") is not False
    return True


def _planning_backend(summary: dict[str, Any]) -> str:
    args = summary.get("planner_extra_args")
    args = args if isinstance(args, list) else []
    for index, value in enumerate(args):
        if value == "--planning-backend" and index + 1 < len(args):
            return str(args[index + 1])
    return "path_planner_route"


def _summary_git_matches_current(payload: dict[str, Any], current_git: dict[str, Any]) -> bool:
    git = payload.get("git_provenance") if isinstance(payload.get("git_provenance"), dict) else {}
    source_current = git.get("current") if isinstance(git.get("current"), dict) else {}
    if not source_current or git.get("current_matches_sources") is False:
        return False
    return _git_snapshots_match(source_current, current_git)


def _input_paths(source_root: Path, candidate_root: Path, batch_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    inputs = config["input_files"]
    return {
        "source_batch_summary": source_root / inputs["source_batch_summary"],
        "holdout_batch_summary": batch_root / inputs["holdout_batch_summary"],
        "fresh_holdout_summary": batch_root / inputs["fresh_holdout_summary"],
        "candidate_summary": candidate_root / inputs["candidate_summary"],
        "checkpoint": candidate_root / inputs["checkpoint"],
        "checkpoint_metadata": candidate_root / inputs["checkpoint_metadata"],
    }


def _output_paths(batch_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["output_files"]
    return {
        "decisions": batch_root / outputs["decisions"],
        "regression_report": batch_root / outputs["regression_report"],
        "summary": batch_root / outputs["summary"],
    }


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    for section in ("input_files", "output_files", "validation", "evaluation"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _load_summary(path: Path, *, expected_schema: str | None, label: str, reason_codes: list[str]) -> dict[str, Any]:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _append_reason(reason_codes, f"{label}_invalid_json")
        return {}
    if not isinstance(payload, dict):
        _append_reason(reason_codes, f"{label}_invalid_json_root")
        return {}
    if expected_schema is not None and payload.get("schema_version") != expected_schema:
        _append_reason(reason_codes, f"{label}_schema_version_mismatch")
    return payload


def _fallback_or_open_grid_count(payload: dict[str, Any]) -> int:
    fields = (
        "open_grid_fallback_used_count",
        "open_grid_fallback_count",
        "fallback_used_count",
        "fallback_or_open_grid_count",
    )
    return max(_int_value(payload.get(field)) for field in fields)


def _cell(*values: Any) -> list[int] | None:
    for value in values:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                return [int(value[0]), int(value[1])]
            except (TypeError, ValueError):
                return None
    return None


def _delta(value: Any, reference: Any) -> float:
    numeric = _float_or_none(value)
    baseline = _float_or_none(reference)
    if numeric is None or baseline is None:
        return 0.0
    return numeric - baseline


def _install_model_explorer_path(repo_root: Path) -> None:
    source = repo_root / "model-explorer" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
