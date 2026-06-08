from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import isfinite, log1p
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot
from git_provenance import git_snapshots_match as _git_snapshots_match

from run_controlled_hybrid_policy_training_candidate import (
    CHECKPOINT_METADATA_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION as CANDIDATE_SUMMARY_SCHEMA_VERSION,
)
from run_hybrid_policy_training_dry_run import PAIRWISE_SAMPLE_TYPES, _pairwise_batch


CONFIG_SCHEMA_VERSION = "fresh-holdout-policy-candidate-evaluation-config/v1"
SUMMARY_SCHEMA_VERSION = "fresh-holdout-policy-candidate-evaluation-summary/v1"
NEXT_FRESH_HOLDOUT_REQUIRED = "fresh_holdout_scenario_or_candidate_generation_required"
NEXT_SCENARIO_DISJOINT_REQUIRED = "scenario_disjoint_holdout_generation_required"
NEXT_STABLE_CONTEXT_ID_REQUIRED = "stable_context_id_contract_required"
NEXT_TRAINING_OBJECTIVE_REQUIRED = "training_objective_or_sample_weight_refinement_required"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an experimental hybrid policy candidate on fresh disjoint holdout contexts."
    )
    parser.add_argument("--source-root")
    parser.add_argument("--candidate-root")
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    _install_model_explorer_path(repo_root)
    batch_root = _resolve_path(args.batch_root, repo_root)
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    source_root = _resolve_path(_root_arg_or_default(args.source_root, config, "source_root"), repo_root)
    candidate_root = _resolve_path(
        _root_arg_or_default(args.candidate_root, config, "candidate_root"),
        repo_root,
    )

    output_paths = _output_paths(batch_root, config)
    summary = run_fresh_holdout_policy_candidate_evaluation(
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
                "fresh_disjoint_context_count": summary["fresh_disjoint_context_count"],
                "identity_overlap_count": summary["identity_overlap_count"],
                "summary": _display_path(output_paths["summary"], repo_root),
            },
            ensure_ascii=False,
        )
    )
    if not args.validate_only:
        batch_root.mkdir(parents=True, exist_ok=True)
        _write_json(output_paths["summary"], summary)
        _write_json(output_paths["overlap_report"], summary["overlap_report"])
        _write_json(output_paths["score_report"], summary["candidate_score_report"])
    return 1 if summary["status"] == "failed" else 0


def run_fresh_holdout_policy_candidate_evaluation(
    *,
    source_root: Path,
    candidate_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    reason_codes: list[str] = []
    paths = _input_paths(source_root, candidate_root, batch_root, config)
    candidate = _load_summary(
        paths["candidate_summary"],
        expected_schema=CANDIDATE_SUMMARY_SCHEMA_VERSION,
        label="controlled_hybrid_policy_training_candidate_summary",
        reason_codes=reason_codes,
    )
    metadata = _load_summary(
        paths["checkpoint_metadata"],
        expected_schema=CHECKPOINT_METADATA_SCHEMA_VERSION,
        label="checkpoint_metadata",
        reason_codes=reason_codes,
    )
    batch_summary = _load_summary(
        paths["batch_summary"],
        expected_schema=None,
        label="fresh_holdout_batch_summary",
        reason_codes=reason_codes,
    )
    if candidate.get("status") != "passed" or candidate.get("candidate_training_status") != "passed":
        _append_reason(reason_codes, "controlled_hybrid_policy_training_candidate_summary_failed")
    if _string_list(candidate.get("reason_codes")):
        _append_reason(reason_codes, "controlled_hybrid_policy_training_candidate_summary_failed")
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
    if batch_summary.get("failed_count", 0):
        _append_reason(reason_codes, "fresh_holdout_batch_failed")
    if _string_list(batch_summary.get("reason_codes")):
        _append_reason(reason_codes, "fresh_holdout_batch_failed")
    current_git = _git_snapshot(repo_root)
    candidate_git_current_matches_sources = _summary_git_matches_current(candidate, current_git)
    checkpoint_metadata_git_current_matches_sources = _summary_git_matches_current(metadata, current_git)
    validation = config["validation"]
    if validation.get("require_candidate_git_current_match"):
        if not candidate_git_current_matches_sources:
            _append_reason(reason_codes, "candidate_git_current_mismatch")
        if not checkpoint_metadata_git_current_matches_sources:
            _append_reason(reason_codes, "checkpoint_metadata_git_current_mismatch")

    source_records = _collect_context_records(source_root, repo_root, source_label="source")
    source_records.extend(_collect_context_records(candidate_root, repo_root, source_label="candidate"))
    holdout_records = _collect_context_records(batch_root, repo_root, source_label="fresh_holdout")
    source_keys = {record["identity_key"] for record in source_records if record.get("identity_key")}
    source_scenarios = {record["scenario_id"] for record in source_records if record.get("scenario_id")}
    holdout_scenarios = {record["scenario_id"] for record in holdout_records if record.get("scenario_id")}
    scenario_overlap_ids = sorted(source_scenarios & holdout_scenarios)
    context_id_missing_count = sum(1 for record in holdout_records if not record.get("context_id"))
    legacy_identity_fallback_count = sum(
        1 for record in holdout_records if record.get("legacy_identity_fallback_used")
    )

    accepted_records: list[dict[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []
    identity_key_missing_count = 0
    identity_overlap_count = 0
    for record in holdout_records:
        key = record.get("identity_key")
        if not key:
            identity_key_missing_count += 1
            excluded_records.append({**record, "exclusion_reason": "identity_key_missing"})
            continue
        if key in source_keys:
            identity_overlap_count += 1
            excluded_records.append({**record, "exclusion_reason": "identity_overlap"})
            continue
        accepted_records.append(record)

    fallback_or_open_grid_count = max(
        _fallback_or_open_grid_count(batch_summary),
        sum(1 for record in accepted_records if record.get("open_grid_fallback_used")),
    )
    safety_regression_count = max(
        _int_value(batch_summary.get("safety_regression_count")),
        sum(1 for record in accepted_records if _int_value(record.get("tracking_safety_violation_count")) > 0),
    )
    contract_violation_count = max(
        _int_value(batch_summary.get("candidate_contract_alignment_gap_count")),
        _int_value(batch_summary.get("contract_violation_count")),
        sum(1 for record in accepted_records if record.get("contract_safe") is False),
    )
    path_cost_regression_count = _int_value(batch_summary.get("path_cost_regression_count"))
    risk_regression_count = _int_value(batch_summary.get("risk_regression_count"))
    source_selection_regression_count = _int_value(batch_summary.get("source_selection_regression_count"))
    action_mask_invalid_count, empty_action_mask_count, teacher_action_agreement_count = (
        _action_label_gate_counts(accepted_records)
    )

    if len(accepted_records) < _int_value(validation.get("min_fresh_disjoint_context_count")):
        _append_reason(reason_codes, "fresh_disjoint_context_count_zero")
    max_identity_overlap_count = validation.get("max_identity_overlap_count")
    if max_identity_overlap_count is not None and identity_overlap_count > _int_value(max_identity_overlap_count):
        _append_reason(reason_codes, "identity_overlap")
    if validation.get("require_context_id") and context_id_missing_count:
        _append_reason(reason_codes, "context_id_missing")
    max_legacy_identity_fallback_count = validation.get("max_legacy_identity_fallback_count")
    if (
        max_legacy_identity_fallback_count is not None
        and legacy_identity_fallback_count > _int_value(max_legacy_identity_fallback_count)
    ):
        _append_reason(reason_codes, "legacy_identity_fallback_used")
    if validation.get("require_scenario_disjoint") and scenario_overlap_ids:
        _append_reason(reason_codes, "scenario_overlap")
    max_scenario_overlap_count = validation.get("max_scenario_overlap_count")
    if max_scenario_overlap_count is not None and len(scenario_overlap_ids) > _int_value(max_scenario_overlap_count):
        _append_reason(reason_codes, "scenario_overlap")
    if fallback_or_open_grid_count > _int_value(validation.get("max_fallback_or_open_grid_count")):
        _append_reason(reason_codes, "fallback_or_open_grid")
    if safety_regression_count > _int_value(validation.get("max_safety_regression_count")):
        _append_reason(reason_codes, "safety_regression")
    if contract_violation_count > _int_value(validation.get("max_contract_violation_count")):
        _append_reason(reason_codes, "contract_violation")
    if action_mask_invalid_count > _int_value(validation.get("max_action_mask_invalid_count")):
        _append_reason(reason_codes, "action_mask_invalid")
    if empty_action_mask_count > _int_value(validation.get("max_empty_action_mask_count")):
        _append_reason(reason_codes, "empty_action_mask")
    if path_cost_regression_count:
        _append_reason(reason_codes, "path_cost_regression")
    if risk_regression_count:
        _append_reason(reason_codes, "risk_regression")
    if source_selection_regression_count:
        _append_reason(reason_codes, "source_selection_regression")

    score_report = _score_candidate_records(
        checkpoint_path=paths["checkpoint"],
        records=accepted_records,
        config=config,
        repo_root=repo_root,
        reason_codes=reason_codes,
    )
    overlap_report = {
        "schema_version": "fresh-holdout-overlap-report/v1",
        "source_identity_count": len(source_keys),
        "holdout_context_count": len(holdout_records),
        "fresh_disjoint_context_count": len(accepted_records),
        "identity_overlap_count": identity_overlap_count,
        "identity_key_missing_count": identity_key_missing_count,
        "context_id_missing_count": context_id_missing_count,
        "legacy_identity_fallback_count": legacy_identity_fallback_count,
        "accepted_identity_overlap_count": 0,
        "accepted_identity_key_missing_count": 0,
        "scenario_overlap_count": len(scenario_overlap_ids),
        "scenario_overlap_ids": scenario_overlap_ids,
        "excluded_records": _report_records(excluded_records),
        "accepted_records": _report_records(accepted_records),
    }

    quality_failed = bool(
        fallback_or_open_grid_count
        or safety_regression_count
        or contract_violation_count
        or path_cost_regression_count
        or risk_regression_count
        or source_selection_regression_count
    )
    next_required_change = None
    scenario_disjoint_required = bool(
        validation.get("require_scenario_disjoint")
        or validation.get("max_scenario_overlap_count") is not None
    )
    stable_context_id_required = bool(
        validation.get("require_context_id")
        or validation.get("max_legacy_identity_fallback_count") is not None
    )
    if len(accepted_records) <= 0:
        next_required_change = NEXT_FRESH_HOLDOUT_REQUIRED
    if scenario_disjoint_required and scenario_overlap_ids:
        next_required_change = NEXT_SCENARIO_DISJOINT_REQUIRED
    elif stable_context_id_required and (context_id_missing_count or legacy_identity_fallback_count):
        next_required_change = NEXT_STABLE_CONTEXT_ID_REQUIRED
    elif quality_failed:
        next_required_change = NEXT_TRAINING_OBJECTIVE_REQUIRED

    status = "failed" if reason_codes else "passed"
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "source_root": _display_path(source_root, repo_root),
        "candidate_root": _display_path(candidate_root, repo_root),
        "batch_root": _display_path(batch_root, repo_root),
        "summary": _display_path(output_paths["summary"], repo_root),
        "overlap_report_path": _display_path(output_paths["overlap_report"], repo_root),
        "candidate_score_report_path": _display_path(output_paths["score_report"], repo_root),
        "candidate_summary_path": _display_path(paths["candidate_summary"], repo_root),
        "checkpoint_path": _display_path(paths["checkpoint"], repo_root),
        "checkpoint_metadata_path": _display_path(paths["checkpoint_metadata"], repo_root),
        "raw_holdout_context_count": len(holdout_records),
        "fresh_disjoint_context_count": len(accepted_records),
        "require_context_id": bool(validation.get("require_context_id")),
        "require_scenario_disjoint": bool(validation.get("require_scenario_disjoint")),
        "identity_overlap_count": identity_overlap_count,
        "identity_key_missing_count": identity_key_missing_count,
        "identity_overlap_ratio": (
            identity_overlap_count / len(holdout_records)
            if holdout_records
            else 0.0
        ),
        "context_id_missing_count": context_id_missing_count,
        "context_id_coverage_rate": (
            (len(holdout_records) - context_id_missing_count) / len(holdout_records)
            if holdout_records
            else 0.0
        ),
        "legacy_identity_fallback_count": legacy_identity_fallback_count,
        "accepted_identity_overlap_count": 0,
        "accepted_identity_key_missing_count": 0,
        "scenario_overlap_count": len(scenario_overlap_ids),
        "scenario_disjoint": len(scenario_overlap_ids) == 0,
        "fallback_or_open_grid_count": fallback_or_open_grid_count,
        "safety_regression_count": safety_regression_count,
        "contract_violation_count": contract_violation_count,
        "path_cost_regression_count": path_cost_regression_count,
        "risk_regression_count": risk_regression_count,
        "source_selection_regression_count": source_selection_regression_count,
        "action_mask_invalid_count": action_mask_invalid_count,
        "empty_action_mask_count": empty_action_mask_count,
        "teacher_action_agreement_count": teacher_action_agreement_count,
        "preference_margin_satisfied_count": score_report["preference_margin_satisfied_count"],
        "preference_margin_failed_count": score_report["preference_margin_failed_count"],
        "scoreable_record_count": score_report["scoreable_record_count"],
        "checkpoint_loaded": score_report["checkpoint_loaded"],
        "next_required_change": next_required_change,
        "experimental_checkpoint": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "candidate_git_current_matches_sources": candidate_git_current_matches_sources,
        "checkpoint_metadata_git_current_matches_sources": checkpoint_metadata_git_current_matches_sources,
        "git_provenance": {"current": current_git, "current_matches_sources": True},
        "overlap_report": overlap_report,
        "candidate_score_report": score_report,
        "non_goals": list(config.get("non_goals", [])),
    }


def _collect_context_records(root: Path, repo_root: Path, *, source_label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    registry_path = root / "unified-policy-sample-registry.jsonl"
    if registry_path.is_file():
        for index, payload in enumerate(_load_jsonl(registry_path)):
            record = _record_from_registry(payload, index=index, source_label=source_label, repo_root=repo_root)
            records.append(record)
    for summary_path in sorted(root.glob("**/path-feedback-summary.json")):
        records.extend(_records_from_path_feedback_summary(summary_path, root, repo_root, source_label=source_label))
    return records


def _records_from_path_feedback_summary(
    summary_path: Path,
    root: Path,
    repo_root: Path,
    *,
    source_label: str,
) -> list[dict[str, Any]]:
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    run_id = summary_path.parent.name
    records: list[dict[str, Any]] = []
    for scenario in payload.get("scenarios", []):
        if not isinstance(scenario, dict):
            continue
        scenario_id = str(scenario.get("scenario_id", ""))
        candidates = scenario.get("path_feedback", {}).get("candidates")
        if not isinstance(candidates, list):
            best = scenario.get("path_feedback", {}).get("best_by_path_cost")
            candidates = [best] if isinstance(best, dict) else []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            record = _record_from_candidate(
                candidate,
                scenario=scenario,
                summary=payload,
                run_id=run_id,
                source_label=source_label,
                summary_path=summary_path,
                root=root,
                repo_root=repo_root,
            )
            records.append(record)
    return records


def _record_from_candidate(
    candidate: dict[str, Any],
    *,
    scenario: dict[str, Any],
    summary: dict[str, Any],
    run_id: str,
    source_label: str,
    summary_path: Path,
    root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    source_action_index = candidate.get("source_action_index")
    if source_action_index is None:
        source_action_index = candidate.get("action_index")
    policy_target_cell = _cell(candidate.get("policy_target_cell"), candidate.get("cell"))
    execution_goal_cell = _cell(candidate.get("execution_goal_cell"), candidate.get("cell"))
    target_binding_mode = str(candidate.get("target_binding_mode") or candidate.get("candidate_role") or "path_feedback_candidate")
    scenario_id = str(scenario.get("scenario_id", ""))
    context_id = candidate.get("context_id")
    record = {
        "source_label": source_label,
        "source_path": _display_path(summary_path, repo_root),
        "root_relative_path": _display_path(summary_path, root),
        "run_id": run_id,
        "scenario_id": scenario_id,
        "scenario_group": str(scenario.get("scenario_group") or "unknown"),
        "scenario_seed": scenario.get("scenario_seed"),
        "scenario_variant_id": scenario.get("scenario_variant_id"),
        "diagnostic_profile": summary.get("diagnostic_profile"),
        "planning_backend": _planning_backend_from_record(candidate, summary),
        "top_k": _optional_int(summary.get("top_k")),
        "sample_type": "path_feedback_candidate",
        "candidate_role": str(candidate.get("candidate_role") or "policy_target"),
        "context_id": str(context_id) if context_id else None,
        "context_id_schema_version": candidate.get("context_id_schema_version"),
        "context_id_source": candidate.get("context_id_source"),
        "source_action_index": _optional_int(source_action_index),
        "action_index": _optional_int(candidate.get("action_index")),
        "policy_target_cell": policy_target_cell,
        "execution_goal_cell": execution_goal_cell,
        "target_binding_mode": target_binding_mode,
        "reachable": bool(candidate.get("reachable")),
        "replan_required": bool(candidate.get("replan_required")),
        "open_grid_fallback_used": bool(
            candidate.get("open_grid_fallback_used") or scenario.get("open_grid_fallback_used")
        ),
        "tracking_safety_violation_count": _int_value(
            candidate.get("tracking_safety_violation_count", scenario.get("tracking_safety_violation_count"))
        ),
        "contract_safe": _contract_safe(candidate),
        "path_cost": _float_or_none(candidate.get("path_cost")),
        "risk": _float_or_none(candidate.get("risk")),
        "utility": _float_or_none(candidate.get("utility")),
    }
    record["identity_key"] = _identity_key(record)
    return record


def _record_from_registry(
    payload: dict[str, Any],
    *,
    index: int,
    source_label: str,
    repo_root: Path,
) -> dict[str, Any]:
    record = {
        "source_label": source_label,
        "source_path": _display_path(repo_root / "unified-policy-sample-registry.jsonl", repo_root),
        "run_id": str(payload.get("run_id", "")),
        "scenario_id": str(payload.get("scenario_id", "")),
        "scenario_group": str(payload.get("scenario_group") or "unknown"),
        "scenario_seed": payload.get("scenario_seed"),
        "scenario_variant_id": payload.get("scenario_variant_id"),
        "diagnostic_profile": payload.get("diagnostic_profile"),
        "planning_backend": payload.get("planning_backend"),
        "top_k": _optional_int(payload.get("top_k")),
        "sample_type": str(payload.get("sample_type", "")),
        "candidate_role": str(payload.get("candidate_role") or payload.get("sample_type", "")),
        "context_id": str(payload.get("context_id")) if payload.get("context_id") else None,
        "context_id_schema_version": payload.get("context_id_schema_version"),
        "context_id_source": payload.get("context_id_source"),
        "source_action_index": _optional_int(payload.get("source_action_index")),
        "action_index": _optional_int(payload.get("action_index")),
        "policy_target_cell": _cell(payload.get("policy_target_cell")),
        "execution_goal_cell": _cell(payload.get("execution_goal_cell")),
        "target_binding_mode": str(payload.get("target_binding_mode") or payload.get("sample_type", "")),
        "registry_index": index,
        "reachable": True,
        "replan_required": False,
        "open_grid_fallback_used": False,
        "tracking_safety_violation_count": 0,
        "contract_safe": payload.get("contract_safe", True) is not False,
        "path_cost": _float_or_none(payload.get("path_cost")),
        "risk": _float_or_none(payload.get("risk")),
        "utility": _float_or_none(payload.get("utility")),
        "raw_record": payload,
    }
    record["identity_key"] = _identity_key(record)
    return record


def _identity_key(record: dict[str, Any]) -> str | None:
    context_id = record.get("context_id")
    if context_id:
        record["legacy_identity_fallback_used"] = False
        return "context:" + str(context_id)
    fields = {
        "scenario_id": record.get("scenario_id"),
        "sample_type": record.get("sample_type"),
        "source_action_index": record.get("source_action_index"),
        "policy_target_cell": record.get("policy_target_cell"),
        "execution_goal_cell": record.get("execution_goal_cell"),
        "target_binding_mode": record.get("target_binding_mode"),
    }
    if any(value in (None, "", []) for value in fields.values()):
        record["legacy_identity_fallback_used"] = False
        return None
    record["legacy_identity_fallback_used"] = True
    return "legacy:" + json.dumps(fields, sort_keys=True, separators=(",", ":"))


def _planning_backend_from_record(candidate: dict[str, Any], summary: dict[str, Any]) -> str:
    args = summary.get("planner_extra_args")
    args = args if isinstance(args, list) else []
    for index, value in enumerate(args):
        if value == "--planning-backend" and index + 1 < len(args):
            return str(args[index + 1])
    backend = candidate.get("planning_backend")
    if isinstance(backend, dict):
        for key in ("backend", "name", "source"):
            value = backend.get(key)
            if value:
                return str(value)
    return "path_planner_route"


def _summary_git_matches_current(payload: dict[str, Any], current_git: dict[str, Any]) -> bool:
    git = payload.get("git_provenance") if isinstance(payload.get("git_provenance"), dict) else {}
    source_current = git.get("current") if isinstance(git.get("current"), dict) else {}
    if not source_current:
        return False
    if git.get("current_matches_sources") is False:
        return False
    return _git_snapshots_match(source_current, current_git)


def _action_label_gate_counts(records: list[dict[str, Any]]) -> tuple[int, int, int]:
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("sample_type") == "path_feedback_candidate":
            by_scenario[str(record.get("scenario_id"))].append(record)
    invalid = 0
    empty = 0
    agreement = 0
    for candidates in by_scenario.values():
        reachable = [record for record in candidates if record.get("reachable") and not record.get("replan_required")]
        if not reachable:
            empty += 1
            continue
        teacher = min(reachable, key=lambda record: _float_or_none(record.get("path_cost")) or 0.0)
        if teacher.get("source_action_index") is None:
            invalid += 1
        else:
            agreement += 1
    return invalid, empty, agreement


def _score_candidate_records(
    *,
    checkpoint_path: Path,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    repo_root: Path,
    reason_codes: list[str],
) -> dict[str, Any]:
    report = {
        "schema_version": "fresh-holdout-candidate-score-report/v1",
        "checkpoint_path": _display_path(checkpoint_path, repo_root),
        "checkpoint_loaded": False,
        "scoreable_record_count": 0,
        "action_label_score_count": 0,
        "preference_margin_satisfied_count": 0,
        "preference_margin_failed_count": 0,
        "preference_margin_threshold": float(config["evaluation"].get("preference_margin_threshold", 0.0)),
        "records": _report_records(records),
    }
    if not checkpoint_path.is_file() or not records:
        return report
    try:
        _score_with_checkpoint(checkpoint_path, records, config=config, repo_root=repo_root, report=report)
    except Exception as exc:  # noqa: BLE001
        _append_reason(reason_codes, "fresh_holdout_candidate_score_failed")
        report["score_error"] = str(exc)
    return report


def _score_with_checkpoint(
    checkpoint_path: Path,
    records: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    repo_root: Path,
    report: dict[str, Any],
) -> None:
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
    report["checkpoint_loaded"] = True

    scoreable = [record for record in records if record.get("sample_type") == "path_feedback_candidate"]
    if scoreable:
        candidate_features = torch.tensor(
            [[_candidate_features(record) for record in scoreable]],
            dtype=torch.float32,
        )
        global_features = torch.tensor([_global_features(scoreable)], dtype=torch.float32)
        action_mask = torch.tensor([[bool(record.get("reachable")) for record in scoreable]], dtype=torch.bool)
        missing = torch.tensor(
            [[_missing_indicators(record) for record in scoreable]],
            dtype=torch.float32,
        )
        with torch.no_grad():
            output = network(
                candidate_features=candidate_features,
                global_features=global_features,
                action_mask=action_mask,
                candidate_missing_indicators=missing,
            )
        logits = output.masked_logits[0].tolist()
        report["scoreable_record_count"] = len(scoreable)
        report["action_label_score_count"] = len(scoreable)
        for record, logit in zip(scoreable, logits):
            record["candidate_logit"] = float(logit)

    pairwise = [record["raw_record"] for record in records if record.get("sample_type") in PAIRWISE_SAMPLE_TYPES and isinstance(record.get("raw_record"), dict)]
    if pairwise:
        batch = _pairwise_batch(pairwise)
        with torch.no_grad():
            output = network(
                candidate_features=batch["candidate_features"],
                global_features=batch["global_features"],
                action_mask=batch["action_mask"],
                candidate_missing_indicators=batch["candidate_missing_indicators"],
            )
            margins = output.masked_logits[:, 0] - output.masked_logits[:, 1]
        threshold = float(config["evaluation"].get("preference_margin_threshold", 0.0))
        satisfied = int((margins > threshold).sum().item())
        report["preference_margin_satisfied_count"] = satisfied
        report["preference_margin_failed_count"] = int(margins.numel()) - satisfied
        report["scoreable_record_count"] += int(margins.numel())


def _candidate_features(record: dict[str, Any]) -> list[float]:
    cell = record.get("policy_target_cell") or [0, 0]
    x = _float_or_none(cell[0] if isinstance(cell, list) and cell else 0) or 0.0
    y = _float_or_none(cell[1] if isinstance(cell, list) and len(cell) > 1 else 0) or 0.0
    path_cost = _float_or_none(record.get("path_cost")) or 0.0
    risk = _float_or_none(record.get("risk")) or 0.0
    utility = _float_or_none(record.get("utility")) or 0.0
    return [
        _clip_unit(x / 100.0),
        _clip_unit(y / 100.0),
        _clip_signed_unit(x / 100.0),
        _clip_signed_unit(y / 100.0),
        _clip_unit((x * x + y * y) ** 0.5 / 150.0),
        _clip_unit(utility),
        1.0 if record.get("reachable") else 0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        _clip_unit(risk),
        _clip_unit(log1p(max(path_cost, 0.0)) / log1p(100.0)),
        0.0,
    ]


def _global_features(records: list[dict[str, Any]]) -> list[float]:
    return [0.02, 0.02, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0 if records else 0.0]


def _missing_indicators(record: dict[str, Any]) -> list[float]:
    return [
        1.0 if _float_or_none(record.get("utility")) is None else 0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0 if _float_or_none(record.get("risk")) is None else 0.0,
        1.0 if _float_or_none(record.get("path_cost")) is None else 0.0,
    ]


def _input_paths(source_root: Path, candidate_root: Path, batch_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    inputs = config["input_files"]
    return {
        "source_batch_summary": source_root / inputs["source_batch_summary"],
        "batch_summary": batch_root / inputs["batch_summary"],
        "candidate_summary": candidate_root / inputs["candidate_summary"],
        "checkpoint": candidate_root / inputs["checkpoint"],
        "checkpoint_metadata": candidate_root / inputs["checkpoint_metadata"],
    }


def _output_paths(batch_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["output_files"]
    return {
        "summary": batch_root / outputs["summary"],
        "overlap_report": batch_root / outputs["overlap_report"],
        "score_report": batch_root / outputs["score_report"],
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


def _root_arg_or_default(value: str | None, config: dict[str, Any], key: str) -> str:
    if value:
        return value
    defaults = config.get("default_roots")
    if isinstance(defaults, dict) and isinstance(defaults.get(key), str) and defaults[key]:
        return defaults[key]
    raise ConfigError(f"--{key.replace('_', '-')} is required unless default_roots.{key} is set")


def _load_summary(
    path: Path,
    *,
    expected_schema: str | None,
    label: str,
    reason_codes: list[str],
) -> dict[str, Any]:
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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    except json.JSONDecodeError:
        return []
    return rows


def _contract_safe(candidate: dict[str, Any]) -> bool:
    feasibility = candidate.get("platform_goal_feasibility")
    if isinstance(feasibility, dict) and feasibility.get("contract_reachable") is False:
        return False
    return candidate.get("contract_safe", True) is not False


def _cell(*values: Any) -> list[int] | None:
    for value in values:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                return [int(value[0]), int(value[1])]
            except (TypeError, ValueError):
                return None
    return None


def _report_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public_fields = (
        "source_label",
        "source_path",
        "run_id",
        "scenario_id",
        "scenario_group",
        "scenario_seed",
        "scenario_variant_id",
        "diagnostic_profile",
        "planning_backend",
        "top_k",
        "sample_type",
        "candidate_role",
        "context_id",
        "context_id_schema_version",
        "context_id_source",
        "legacy_identity_fallback_used",
        "source_action_index",
        "policy_target_cell",
        "execution_goal_cell",
        "target_binding_mode",
        "identity_key",
        "exclusion_reason",
        "path_cost",
        "risk",
        "utility",
        "candidate_logit",
    )
    return [{key: record.get(key) for key in public_fields if key in record} for record in records]


def _fallback_or_open_grid_count(payload: dict[str, Any]) -> int:
    fields = (
        "open_grid_fallback_used_count",
        "open_grid_fallback_count",
        "fallback_used_count",
        "fallback_or_open_grid_count",
    )
    return max(_int_value(payload.get(field)) for field in fields)


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


def _clip_unit(value: float) -> float:
    if not isfinite(value):
        return 0.0
    return min(max(float(value), 0.0), 1.0)


def _clip_signed_unit(value: float) -> float:
    if not isfinite(value):
        return 0.0
    return min(max(float(value), -1.0), 1.0)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
