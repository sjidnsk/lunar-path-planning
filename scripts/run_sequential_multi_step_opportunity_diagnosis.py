from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from git_provenance import git_snapshot as _git_snapshot

try:
    from scripts.run_scenario_disjoint_policy_rollout_evaluation import _collect_holdout_scenarios
    from scripts.run_sequential_canary_failure_mining import (
        _candidate_is_better,
        _candidate_regression_reasons,
        _delta,
        _display_path,
        _find_candidate,
        _index_groups,
        _int_value,
        _load_json,
        _load_jsonl,
        _match_group,
        _resolve_path,
        _source_selected_candidate,
        _string_list,
        _write_json,
    )
except ModuleNotFoundError:
    from run_scenario_disjoint_policy_rollout_evaluation import _collect_holdout_scenarios
    from run_sequential_canary_failure_mining import (
        _candidate_is_better,
        _candidate_regression_reasons,
        _delta,
        _display_path,
        _find_candidate,
        _index_groups,
        _int_value,
        _load_json,
        _load_jsonl,
        _match_group,
        _resolve_path,
        _source_selected_candidate,
        _string_list,
        _write_json,
    )


CONFIG_SCHEMA_VERSION = "sequential-multi-step-opportunity-diagnosis-config/v1"
SUMMARY_SCHEMA_VERSION = "sequential-multi-step-opportunity-diagnosis-summary/v1"
DIAGNOSTIC_SCHEMA_VERSION = "sequential-multi-step-opportunity-diagnostic/v1"
EXCLUSION_SCHEMA_VERSION = "sequential-multi-step-opportunity-exclusion-report/v1"

NEXT_OPPORTUNITY_GAP = "sequential_multi_step_opportunity_generation_gap"
NEXT_POLICY_ALIGNMENT = "policy_sequential_multi_step_choice_alignment_insufficient"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose safe-better alternatives across sequential canary steps."
    )
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    batch_root = _resolve_path(args.batch_root, repo_root)
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    summary, diagnostics, exclusion_report = run_sequential_multi_step_opportunity_diagnosis(
        batch_root=batch_root,
        config=config,
        repo_root=repo_root,
    )
    output_paths = _output_paths(batch_root, config)
    if not args.validate_only:
        batch_root.mkdir(parents=True, exist_ok=True)
        _write_json(output_paths["summary"], summary)
        output_paths["diagnostics"].write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in diagnostics),
            encoding="utf-8",
        )
        _write_json(output_paths["exclusion_report"], exclusion_report)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "multi_step_opportunity_episode_count": summary[
                    "multi_step_opportunity_episode_count"
                ],
                "family_with_multi_step_opportunity_count": summary[
                    "family_with_multi_step_opportunity_count"
                ],
                "safe_better_alternative_step_count": summary[
                    "safe_better_alternative_step_count"
                ],
                "summary": _display_path(output_paths["summary"], repo_root),
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def run_sequential_multi_step_opportunity_diagnosis(
    *,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    reason_codes: list[str] = []
    paths = _input_paths(batch_root, config)
    rollout_summary = _load_json(paths["summary"], "sequential_summary", reason_codes)
    steps = _load_jsonl(paths["steps"], "sequential_steps", reason_codes)
    episodes = _load_jsonl(paths["episodes"], "sequential_episodes", reason_codes)
    _load_json(paths["rejection_report"], "sequential_rejection_report", reason_codes)

    scenario_groups = _collect_holdout_scenarios(batch_root, repo_root)
    group_index = _index_groups(scenario_groups)
    diagnostics: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    for step in steps:
        group = _match_group(step, group_index)
        if group is None:
            exclusions.append(_exclusion(step, "path_feedback_group_missing"))
            continue
        source = _source_candidate_for_step(step, group)
        if source is None:
            exclusions.append(_exclusion(step, "source_candidate_missing"))
            continue
        diagnostic = _diagnose_step(step, group, source, config=config)
        diagnostics.append(diagnostic)

    step_count = len(steps)
    episode_ids = {str(step.get("episode_id")) for step in steps if step.get("episode_id")}
    episode_count = len(episode_ids) or len(episodes)
    family_set = {str(step.get("scenario_group") or "unknown") for step in steps}
    steps_with_safe_better = [
        row for row in diagnostics if int(row.get("safe_better_alternative_count", 0)) > 0
    ]
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in steps_with_safe_better:
        by_episode[str(row.get("episode_id"))].append(row)
        by_family[str(row.get("scenario_group") or "unknown")].append(row)

    multi_step_opportunity_episode_ids = sorted(
        episode_id for episode_id, rows in by_episode.items() if len(rows) >= 2
    )
    family_with_multi_step_opportunity = sorted(
        str(row.get("scenario_group") or "unknown")
        for episode_id in multi_step_opportunity_episode_ids
        for row in by_episode[episode_id][:1]
    )
    family_with_multi_step_opportunity = sorted(set(family_with_multi_step_opportunity))
    opportunity_class_counts = Counter(str(row.get("opportunity_class")) for row in diagnostics)
    funnel_totals = _funnel_totals(diagnostics)

    validation = config.get("validation", {})
    if episode_count < _int_value(validation.get("min_episode_count")):
        _append_reason(reason_codes, "episode_count_below_threshold")
    if step_count < _int_value(validation.get("min_step_count")):
        _append_reason(reason_codes, "step_count_below_threshold")
    if len(multi_step_opportunity_episode_ids) < _int_value(
        validation.get("min_multi_step_opportunity_episode_count")
    ):
        _append_reason(reason_codes, "multi_step_opportunity_episode_count_below_threshold")
    if len(family_with_multi_step_opportunity) < _int_value(
        validation.get("min_family_with_multi_step_opportunity_count")
    ):
        _append_reason(reason_codes, "family_with_multi_step_opportunity_count_below_threshold")
    if len(steps_with_safe_better) < _int_value(
        validation.get("min_safe_better_alternative_step_count")
    ):
        _append_reason(reason_codes, "safe_better_alternative_step_count_below_threshold")
    if len(exclusions) > _int_value(validation.get("max_opportunity_exclusion_count")):
        _append_reason(reason_codes, "opportunity_exclusion_count_above_threshold")

    status = "failed" if reason_codes else "passed"
    current_git = _git_snapshot(repo_root)
    next_required_change = _next_required_change(reason_codes, opportunity_class_counts)
    exclusion_report = {
        "schema_version": EXCLUSION_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "opportunity_exclusion_count": len(exclusions),
        "exclusion_reason_counts": dict(sorted(Counter(row["reason_code"] for row in exclusions).items())),
        "exclusions": exclusions,
        "non_goals": list(config.get("non_goals", [])),
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "batch_root": _display_path(batch_root, repo_root),
        "sequential_summary_status": rollout_summary.get("status"),
        "episode_count": episode_count,
        "step_count": step_count,
        "scenario_family_count": len(family_set),
        "safe_better_alternative_step_count": len(steps_with_safe_better),
        "safe_better_alternative_count": sum(
            int(row.get("safe_better_alternative_count", 0)) for row in diagnostics
        ),
        "multi_step_opportunity_episode_count": len(multi_step_opportunity_episode_ids),
        "multi_step_opportunity_episode_ids": multi_step_opportunity_episode_ids,
        "family_with_multi_step_opportunity_count": len(family_with_multi_step_opportunity),
        "family_with_multi_step_opportunity": family_with_multi_step_opportunity,
        "opportunity_missing_count": opportunity_class_counts.get("opportunity_missing", 0),
        "policy_missed_existing_opportunity_count": opportunity_class_counts.get(
            "policy_missed_existing_opportunity",
            0,
        ),
        "policy_used_existing_opportunity_count": opportunity_class_counts.get(
            "policy_used_existing_opportunity",
            0,
        ),
        "policy_rejected_existing_opportunity_count": opportunity_class_counts.get(
            "policy_rejected_existing_opportunity",
            0,
        ),
        "opportunity_class_counts": dict(sorted(opportunity_class_counts.items())),
        "opportunity_funnel_totals": funnel_totals,
        "opportunity_exclusion_count": len(exclusions),
        "exclusion_reason_counts": exclusion_report["exclusion_reason_counts"],
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": current_git, "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
        "next_required_change": next_required_change,
    }
    return summary, diagnostics, exclusion_report


def _diagnose_step(
    step: dict[str, Any],
    group: dict[str, Any],
    source: dict[str, Any],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    candidates = [candidate for candidate in group.get("candidates", []) if isinstance(candidate, dict)]
    alternative_records = []
    for candidate in candidates:
        if _same_context(candidate, source):
            continue
        regression_reasons = _candidate_regression_reasons(candidate, source, config=_sample_config(config))
        better = _candidate_is_better(candidate, source, config=_sample_config(config))
        record = {
            "context_id": candidate.get("context_id"),
            "action_index": candidate.get("action_index"),
            "source_action_index": candidate.get("source_action_index"),
            "policy_target_cell": candidate.get("policy_target_cell"),
            "execution_goal_cell": candidate.get("execution_goal_cell"),
            "action_mask_valid": "invalid_action_mask" not in regression_reasons,
            "reachable": bool(candidate.get("reachable")),
            "no_replan": not bool(candidate.get("replan_required")),
            "no_fallback_or_open_grid": "fallback_or_open_grid" not in regression_reasons,
            "contract_safe": "contract_violation" not in regression_reasons,
            "path_risk_non_regressive": not (
                {"path_cost_regression", "risk_regression"} & set(regression_reasons)
            ),
            "source_selection_non_regressive": "source_selection_regression" not in regression_reasons,
            "better": better,
            "safe_better": not regression_reasons and better,
            "regression_reason_codes": regression_reasons,
            "path_cost_delta": _delta(candidate.get("path_cost"), source.get("path_cost")),
            "risk_delta": _delta(candidate.get("risk"), source.get("risk")),
            "utility_delta": _delta(candidate.get("utility"), source.get("utility")),
        }
        alternative_records.append(record)
    safe_better_count = sum(1 for record in alternative_records if record["safe_better"])
    opportunity_class = _opportunity_class(step, safe_better_count)
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "episode_id": step.get("episode_id"),
        "step_index": step.get("step_index"),
        "scenario_id": step.get("scenario_id") or group.get("scenario_id"),
        "scenario_group": step.get("scenario_group") or group.get("scenario_group"),
        "scenario_seed": group.get("scenario_seed"),
        "scenario_variant_id": group.get("scenario_variant_id"),
        "source_path": step.get("source_path") or group.get("source_path"),
        "decision_class": step.get("decision_class"),
        "opportunity_class": opportunity_class,
        "source_context_id": source.get("context_id"),
        "source_action_index": source.get("action_index"),
        "source_execution_goal_cell": source.get("execution_goal_cell"),
        "alternative_count": len(alternative_records),
        "action_mask_valid_alternative_count": sum(1 for record in alternative_records if record["action_mask_valid"]),
        "reachable_alternative_count": sum(1 for record in alternative_records if record["reachable"]),
        "no_replan_alternative_count": sum(1 for record in alternative_records if record["no_replan"]),
        "no_fallback_or_open_grid_alternative_count": sum(
            1 for record in alternative_records if record["no_fallback_or_open_grid"]
        ),
        "contract_safe_alternative_count": sum(1 for record in alternative_records if record["contract_safe"]),
        "path_risk_non_regressive_alternative_count": sum(
            1 for record in alternative_records if record["path_risk_non_regressive"]
        ),
        "source_selection_non_regressive_alternative_count": sum(
            1 for record in alternative_records if record["source_selection_non_regressive"]
        ),
        "better_alternative_count": sum(1 for record in alternative_records if record["better"]),
        "safe_better_alternative_count": safe_better_count,
        "alternatives": alternative_records,
    }


def _source_candidate_for_step(step: dict[str, Any], group: dict[str, Any]) -> dict[str, Any] | None:
    candidate = _find_candidate(
        group,
        step.get("source_selected_context_id") or step.get("context_id"),
        action_index=step.get("source_selected_action_index"),
        execution_goal_cell=step.get("source_selected_execution_goal_cell")
        or step.get("source_execution_goal_cell"),
    )
    return candidate or _source_selected_candidate(group)


def _opportunity_class(step: dict[str, Any], safe_better_count: int) -> str:
    if safe_better_count <= 0:
        return "opportunity_missing"
    decision_class = str(step.get("decision_class") or "")
    if decision_class == "source_aligned":
        return "policy_missed_existing_opportunity"
    if decision_class == "canary_accepted_policy_choice":
        return "policy_used_existing_opportunity"
    if decision_class == "canary_rejected_policy_choice":
        return "policy_rejected_existing_opportunity"
    return "safe_better_opportunity_available"


def _sample_config(config: dict[str, Any]) -> dict[str, Any]:
    evaluation = config.get("evaluation", {})
    return {
        "sample": {
            "max_path_cost_regression": evaluation.get("max_path_cost_regression", 0.0),
            "max_risk_regression": evaluation.get("max_risk_regression", 0.0),
            "min_better_path_cost_delta": evaluation.get("min_better_path_cost_delta", 0.25),
            "min_better_risk_delta": evaluation.get("min_better_risk_delta", 0.01),
            "min_better_utility_delta": evaluation.get("min_better_utility_delta", 0.005),
        }
    }


def _same_context(candidate: dict[str, Any], source: dict[str, Any]) -> bool:
    candidate_context = candidate.get("context_id")
    source_context = source.get("context_id")
    return bool(candidate_context and source_context and candidate_context == source_context)


def _funnel_totals(diagnostics: list[dict[str, Any]]) -> dict[str, int]:
    fields = (
        "alternative_count",
        "action_mask_valid_alternative_count",
        "reachable_alternative_count",
        "no_replan_alternative_count",
        "no_fallback_or_open_grid_alternative_count",
        "contract_safe_alternative_count",
        "path_risk_non_regressive_alternative_count",
        "source_selection_non_regressive_alternative_count",
        "better_alternative_count",
        "safe_better_alternative_count",
    )
    return {
        field: sum(int(row.get(field, 0)) for row in diagnostics)
        for field in fields
    }


def _next_required_change(
    reason_codes: list[str],
    opportunity_class_counts: Counter[str],
) -> str | None:
    if not reason_codes:
        return None
    opportunity_reasons = {
        "episode_count_below_threshold",
        "step_count_below_threshold",
        "multi_step_opportunity_episode_count_below_threshold",
        "family_with_multi_step_opportunity_count_below_threshold",
        "safe_better_alternative_step_count_below_threshold",
        "opportunity_exclusion_count_above_threshold",
    }
    if set(reason_codes) & opportunity_reasons:
        return NEXT_OPPORTUNITY_GAP
    if opportunity_class_counts.get("policy_missed_existing_opportunity", 0):
        return NEXT_POLICY_ALIGNMENT
    return NEXT_OPPORTUNITY_GAP


def _input_paths(batch_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    inputs = config["input_files"]
    return {
        "summary": batch_root / inputs["summary"],
        "steps": batch_root / inputs["steps"],
        "episodes": batch_root / inputs["episodes"],
        "rejection_report": batch_root / inputs["rejection_report"],
    }


def _output_paths(batch_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["output_files"]
    return {
        "summary": batch_root / outputs["summary"],
        "diagnostics": batch_root / outputs["diagnostics"],
        "exclusion_report": batch_root / outputs["exclusion_report"],
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
    for section in ("input_files", "output_files", "evaluation", "validation"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _exclusion(step: dict[str, Any], reason_code: str) -> dict[str, Any]:
    return {
        "reason_code": reason_code,
        "episode_id": step.get("episode_id"),
        "step_index": step.get("step_index"),
        "scenario_id": step.get("scenario_id"),
        "scenario_group": step.get("scenario_group"),
        "decision_class": step.get("decision_class"),
        "source_selected_context_id": step.get("source_selected_context_id"),
        "canary_rejection_reason_codes": _string_list(step.get("canary_rejection_reason_codes")),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
