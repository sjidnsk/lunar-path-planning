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

try:
    from scripts.run_fresh_holdout_policy_candidate_evaluation import (
        _candidate_features,
        _global_features,
        _missing_indicators,
    )
    from scripts.run_scenario_disjoint_policy_rollout_evaluation import (
        _candidate_action_mask_valid,
        _collect_holdout_scenarios,
    )
except ModuleNotFoundError:
    from run_fresh_holdout_policy_candidate_evaluation import (
        _candidate_features,
        _global_features,
        _missing_indicators,
    )
    from run_scenario_disjoint_policy_rollout_evaluation import (
        _candidate_action_mask_valid,
        _collect_holdout_scenarios,
    )


CONFIG_SCHEMA_VERSION = "sequential-canary-failure-mining-config/v1"
SUMMARY_SCHEMA_VERSION = "sequential-canary-failure-mining-summary/v1"
SAMPLE_SCHEMA_VERSION = "sequential-canary-hard-negative-preference-sample/v1"
DIAGNOSTIC_SCHEMA_VERSION = "sequential-canary-failure-diagnostic/v1"
EXCLUSION_SCHEMA_VERSION = "sequential-canary-failure-exclusion-report/v1"
SAMPLE_TYPE = "raw_policy_regression_preference_pair"
SEQUENTIAL_HARD_NEGATIVE_SAMPLE_TYPE = "sequential_hard_negative_preference_pair"
SEQUENTIAL_MISSED_SAFE_CHOICE_SAMPLE_TYPE = "sequential_missed_safe_choice_preference_pair"
MISSED_SAFE_CHOICE_REASON = "missed_safe_better_choice"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mine rejected sequential canary choices into hard-negative preference samples."
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

    outputs = _output_paths(batch_root, config)
    summary, samples, diagnostics, exclusion_report = run_sequential_canary_failure_mining(
        batch_root=batch_root,
        config=config,
        repo_root=repo_root,
    )
    if not args.validate_only:
        batch_root.mkdir(parents=True, exist_ok=True)
        _write_json(outputs["summary"], summary)
        _write_json(outputs["exclusion_report"], exclusion_report)
        outputs["samples"].write_text(
            "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
            encoding="utf-8",
        )
        outputs["diagnostics"].write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in diagnostics),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "sequential_rejected_step_count": summary["sequential_rejected_step_count"],
                "sequential_hard_negative_preference_pair_count": summary[
                    "sequential_hard_negative_preference_pair_count"
                ],
                "sequential_missed_safe_choice_preference_pair_count": summary[
                    "sequential_missed_safe_choice_preference_pair_count"
                ],
                "summary": _display_path(outputs["summary"], repo_root),
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def run_sequential_canary_failure_mining(
    *,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    reason_codes: list[str] = []
    paths = _input_paths(batch_root, config)
    rollout_summary = _load_json(paths["summary"], "sequential_summary", reason_codes)
    steps = _load_jsonl(paths["steps"], "sequential_steps", reason_codes)
    _load_jsonl(paths["episodes"], "sequential_episodes", reason_codes)
    rejection_report = _load_json(paths["rejection_report"], "sequential_rejection_report", reason_codes)

    validation = config.get("validation", {})
    if validation.get("require_failed_sequential_summary", True):
        if rollout_summary.get("status") != "failed":
            _append_reason(reason_codes, "sequential_summary_not_failed_baseline")

    scenario_groups = _collect_holdout_scenarios(batch_root, repo_root)
    group_index = _index_groups(scenario_groups)
    rejected_steps = [
        step
        for step in steps
        if step.get("decision_class") == "canary_rejected_policy_choice"
        or _allowed_reason_codes(step, config)
    ]
    source_aligned_steps = [
        step
        for step in steps
        if step.get("decision_class") == "source_aligned"
        and bool(config.get("sample", {}).get("mine_source_aligned_safe_alternatives", True))
    ]
    hard_negative_samples: list[dict[str, Any]] = []
    missed_safe_choice_samples: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    source_aligned_skipped_reason_counts: Counter[str] = Counter()
    seen: set[str] = set()

    for step in rejected_steps:
        diagnostics.append(_diagnostic(step))
        key = _step_key(step)
        if key in seen:
            exclusions.append(_exclusion(step, "duplicate_rejected_step"))
            continue
        seen.add(key)
        reasons = _allowed_reason_codes(step, config)
        if not reasons:
            exclusions.append(_exclusion(step, "no_supported_regression_reason"))
            continue
        preferred, alternative, group = _preferred_and_alternative(step, group_index)
        if group is None:
            exclusions.append(_exclusion(step, "path_feedback_group_missing"))
            continue
        exclusion = _sample_exclusion_reason(preferred, alternative)
        if exclusion:
            exclusions.append(_exclusion(step, exclusion))
            continue
        hard_negative_samples.append(
            _sample(
                step,
                group,
                preferred,
                alternative,
                reasons,
                config=config,
                sequential_sample_type=SEQUENTIAL_HARD_NEGATIVE_SAMPLE_TYPE,
            )
        )

    missed_seen: set[str] = set()
    for step in source_aligned_steps:
        key = _step_key(step)
        if key in missed_seen:
            exclusions.append(_exclusion(step, "duplicate_source_aligned_step"))
            continue
        missed_seen.add(key)
        sample_or_reason = _missed_safe_choice_sample(step, group_index, config=config)
        if sample_or_reason is None:
            continue
        if isinstance(sample_or_reason, str):
            if bool(config.get("sample", {}).get("record_source_aligned_exclusions", False)):
                exclusions.append(_exclusion(step, sample_or_reason))
            else:
                source_aligned_skipped_reason_counts.update([sample_or_reason])
            continue
        missed_safe_choice_samples.append(sample_or_reason)

    samples = hard_negative_samples + missed_safe_choice_samples

    hard_positive_added_count = 0
    if hard_positive_added_count > _int_value(validation.get("max_hard_positive_added_count")):
        _append_reason(reason_codes, "hard_positive_added_count_nonzero")
    min_total_preferences = _int_value(validation.get("min_sequential_preference_pair_count"))
    missed_only_allowed = (
        not rejected_steps
        and bool(validation.get("allow_missed_safe_choice_only_when_no_rejected_steps", True))
        and len(missed_safe_choice_samples) > 0
        and len(samples) >= min_total_preferences
    )
    if (
        len(hard_negative_samples)
        < _int_value(validation.get("min_sequential_hard_negative_preference_pair_count"))
        and not missed_only_allowed
    ):
        _append_reason(reason_codes, "sequential_hard_negative_preference_pair_count_below_threshold")
    if min_total_preferences and len(samples) < min_total_preferences:
        _append_reason(reason_codes, "sequential_preference_pair_count_below_threshold")
    if len(exclusions) > _int_value(validation.get("max_exclusion_count")):
        _append_reason(reason_codes, "sequential_failure_exclusion_count_above_threshold")

    status = "failed" if reason_codes else "passed"
    current_git = _git_snapshot(repo_root)
    exclusion_report = {
        "schema_version": EXCLUSION_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "exclusion_count": len(exclusions),
        "exclusion_reason_counts": dict(sorted(Counter(item["reason_code"] for item in exclusions).items())),
        "exclusions": exclusions,
        "hard_positive_added_count": hard_positive_added_count,
        "non_goals": list(config.get("non_goals", [])),
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "batch_root": _display_path(batch_root, repo_root),
        "sequential_summary_status": rollout_summary.get("status"),
        "sequential_rejected_step_count": len(rejected_steps),
        "unique_sequential_rejected_step_count": len(seen),
        "source_aligned_step_count": len(source_aligned_steps),
        "sequential_hard_negative_preference_pair_count": len(hard_negative_samples),
        "sequential_missed_safe_choice_preference_pair_count": len(missed_safe_choice_samples),
        "sequential_preference_pair_count": len(samples),
        "sequential_failure_diagnostic_count": len(diagnostics),
        "exclusion_count": len(exclusions),
        "exclusion_reason_counts": exclusion_report["exclusion_reason_counts"],
        "source_aligned_skipped_reason_counts": dict(
            sorted(source_aligned_skipped_reason_counts.items())
        ),
        "hard_positive_added_count": hard_positive_added_count,
        "canary_rejection_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for step in rejected_steps
                    for reason in _string_list(step.get("canary_rejection_reason_codes"))
                ).items()
            )
        ),
        "source_report_failed_step_count": len(rejection_report.get("failed_steps", []))
        if isinstance(rejection_report.get("failed_steps"), list)
        else 0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": current_git, "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
        "next_required_change": "sequential_hard_negative_signal_insufficient" if status == "failed" else None,
    }
    return summary, samples, diagnostics, exclusion_report


def _preferred_and_alternative(
    step: dict[str, Any],
    group_index: dict[tuple[str | None, str | None, str | None], dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    group = _match_group(step, group_index)
    embedded_preferred = step.get("preferred_candidate")
    embedded_alternative = step.get("alternative_candidate")
    if isinstance(embedded_preferred, dict) and isinstance(embedded_alternative, dict):
        return embedded_preferred, embedded_alternative, group or _embedded_group(step)
    if group is None:
        return None, None, None
    preferred = _find_candidate(
        group,
        step.get("source_selected_context_id") or step.get("context_id"),
        action_index=step.get("source_selected_action_index"),
        execution_goal_cell=step.get("source_selected_execution_goal_cell"),
    )
    alternative = _find_candidate(
        group,
        step.get("raw_policy_selected_context_id") or step.get("policy_selected_context_id"),
        action_index=step.get("raw_policy_selected_action_index")
        if step.get("raw_policy_selected_action_index") is not None
        else step.get("policy_selected_action_index"),
        execution_goal_cell=step.get("raw_policy_selected_execution_goal_cell")
        or step.get("policy_selected_execution_goal_cell"),
    )
    return preferred, alternative, group


def _sample(
    step: dict[str, Any],
    group: dict[str, Any],
    preferred: dict[str, Any],
    alternative: dict[str, Any],
    reasons: list[str],
    *,
    config: dict[str, Any],
    sequential_sample_type: str,
) -> dict[str, Any]:
    sample_weight = _sample_weight(reasons, config)
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "sample_type": SAMPLE_TYPE,
        "sequential_sample_type": sequential_sample_type,
        "context_id": preferred.get("context_id"),
        "alternative_context_id": alternative.get("context_id"),
        "episode_id": step.get("episode_id"),
        "step_index": _int_value(step.get("step_index")),
        "input_start_cell": step.get("input_start_cell"),
        "scenario_id": step.get("scenario_id") or group.get("scenario_id"),
        "scenario_group": step.get("scenario_group") or group.get("scenario_group"),
        "scenario_seed": group.get("scenario_seed"),
        "scenario_variant_id": group.get("scenario_variant_id"),
        "diagnostic_profile": group.get("diagnostic_profile"),
        "planning_backend": group.get("planning_backend"),
        "source_path": step.get("source_path") or group.get("source_path"),
        "source_selected_action_index": step.get("source_selected_action_index"),
        "raw_policy_selected_action_index": step.get("raw_policy_selected_action_index"),
        "policy_selected_action_index": step.get("policy_selected_action_index"),
        "source_execution_goal_cell": step.get("source_execution_goal_cell"),
        "policy_execution_goal_cell": step.get("policy_execution_goal_cell"),
        "controlled_execution_goal_cell": step.get("controlled_execution_goal_cell"),
        "path_cost_delta": _delta(alternative.get("path_cost"), preferred.get("path_cost")),
        "risk_delta": _delta(alternative.get("risk"), preferred.get("risk")),
        "utility_delta": _delta(alternative.get("utility"), preferred.get("utility")),
        "raw_policy_regression_reason_codes": reasons,
        "sequential_regression_reason_codes": reasons,
        "missed_safe_choice": sequential_sample_type == SEQUENTIAL_MISSED_SAFE_CHOICE_SAMPLE_TYPE,
        "preferred": _sample_side(preferred),
        "alternative": _sample_side(alternative),
        "selected": _sample_side(preferred),
        "global_features": _global_features(group.get("candidates", [preferred, alternative])),
        "candidate_missing_indicators": [
            _missing_indicators(preferred),
            _missing_indicators(alternative),
        ],
        "sample_weight": sample_weight,
        "hard_positive_added": False,
        "hard_positive_added_count": 0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
    }


def _sample_side(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_id": candidate.get("context_id"),
        "action_index": candidate.get("action_index"),
        "source_action_index": candidate.get("source_action_index"),
        "policy_target_cell": candidate.get("policy_target_cell"),
        "execution_goal_cell": candidate.get("execution_goal_cell"),
        "target_binding_mode": candidate.get("target_binding_mode"),
        "path_cost": candidate.get("path_cost"),
        "risk": candidate.get("risk"),
        "utility": candidate.get("utility"),
        "contract_safe": candidate.get("contract_safe"),
        "reachable": candidate.get("reachable"),
        "replan_required": candidate.get("replan_required"),
        "source_selection_status": candidate.get("source_selection_status"),
        "candidate_features": _candidate_features(candidate),
    }


def _missed_safe_choice_sample(
    step: dict[str, Any],
    group_index: dict[tuple[str | None, str | None, str | None], dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any] | str | None:
    group = _match_group(step, group_index)
    if group is None:
        return "path_feedback_group_missing"
    source = _find_candidate(
        group,
        step.get("source_selected_context_id") or step.get("context_id"),
        action_index=step.get("source_selected_action_index"),
        execution_goal_cell=step.get("source_selected_execution_goal_cell")
        or step.get("source_execution_goal_cell"),
    )
    if source is None:
        source = _source_selected_candidate(group)
    if source is None:
        return "source_candidate_missing"
    exclusion = _sample_exclusion_reason(source, source)
    if exclusion:
        return exclusion.replace("preferred_", "source_")
    alternative = _best_safe_better_alternative(group, source, config=config)
    if alternative is None:
        return None
    exclusion = _sample_exclusion_reason(alternative, source)
    if exclusion:
        return exclusion
    sample = _sample(
        step,
        group,
        alternative,
        source,
        [MISSED_SAFE_CHOICE_REASON],
        config=config,
        sequential_sample_type=SEQUENTIAL_MISSED_SAFE_CHOICE_SAMPLE_TYPE,
    )
    sample["source_selected_context_id"] = source.get("context_id")
    sample["missed_safe_choice_context_id"] = alternative.get("context_id")
    sample["missed_safe_choice_execution_goal_cell"] = alternative.get("execution_goal_cell")
    sample["missed_safe_choice_reason_codes"] = [MISSED_SAFE_CHOICE_REASON]
    return sample


def _source_selected_candidate(group: dict[str, Any]) -> dict[str, Any] | None:
    for candidate in group.get("candidates", []):
        if candidate.get("source_selection_status") == "source_selected":
            return candidate
    candidates = [candidate for candidate in group.get("candidates", []) if _candidate_action_mask_valid(candidate)]
    if candidates:
        return min(candidates, key=lambda candidate: _float_or_none(candidate.get("path_cost")) or 0.0)
    return None


def _best_safe_better_alternative(
    group: dict[str, Any],
    source: dict[str, Any],
    *,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    alternatives: list[dict[str, Any]] = []
    for candidate in group.get("candidates", []):
        if _same_candidate(candidate, source):
            continue
        if _candidate_regression_reasons(candidate, source, config=config):
            continue
        if not _candidate_is_better(candidate, source, config=config):
            continue
        alternatives.append(candidate)
    if not alternatives:
        return None
    return min(
        alternatives,
        key=lambda candidate: (
            _float_or_none(_delta(candidate.get("path_cost"), source.get("path_cost"))) or 0.0,
            _float_or_none(_delta(candidate.get("risk"), source.get("risk"))) or 0.0,
            -(_float_or_none(_delta(candidate.get("utility"), source.get("utility"))) or 0.0),
        ),
    )


def _candidate_regression_reasons(
    candidate: dict[str, Any] | None,
    source: dict[str, Any] | None,
    *,
    config: dict[str, Any],
) -> list[str]:
    sample = config.get("sample", {})
    if candidate is None:
        return ["candidate_missing"]
    reasons: list[str] = []
    candidate_changed = not _same_candidate(candidate, source)
    if not _candidate_action_mask_valid(candidate):
        reasons.append("invalid_action_mask")
    if candidate.get("open_grid_fallback_used"):
        reasons.append("fallback_or_open_grid")
    if _int_value(candidate.get("tracking_safety_violation_count")):
        reasons.append("safety_regression")
    if candidate_changed and candidate.get("contract_safe") is False:
        reasons.append("contract_violation")
    if source is not None:
        path_delta = _delta(candidate.get("path_cost"), source.get("path_cost"))
        risk_delta = _delta(candidate.get("risk"), source.get("risk"))
        max_path = float(sample.get("max_path_cost_regression", 0.0))
        max_risk = float(sample.get("max_risk_regression", 0.0))
        if path_delta is None:
            reasons.append("path_cost_missing")
        elif path_delta > max_path:
            reasons.append("path_cost_regression")
        if risk_delta is None:
            reasons.append("risk_missing")
        elif risk_delta > max_risk:
            reasons.append("risk_regression")
    if candidate_changed and candidate.get("source_selection_quality_regression"):
        reasons.append("source_selection_regression")
    return reasons


def _candidate_is_better(
    candidate: dict[str, Any],
    source: dict[str, Any],
    *,
    config: dict[str, Any],
) -> bool:
    sample = config.get("sample", {})
    path_threshold = float(sample.get("min_better_path_cost_delta", 0.25))
    risk_threshold = float(sample.get("min_better_risk_delta", 0.01))
    utility_threshold = float(sample.get("min_better_utility_delta", 0.005))
    path_delta = _delta(candidate.get("path_cost"), source.get("path_cost"))
    risk_delta = _delta(candidate.get("risk"), source.get("risk"))
    utility_delta = _delta(candidate.get("utility"), source.get("utility"))
    return bool(
        (path_delta is not None and path_delta <= -path_threshold)
        or (risk_delta is not None and risk_delta <= -risk_threshold)
        or (utility_delta is not None and utility_delta >= utility_threshold)
    )


def _same_candidate(candidate: dict[str, Any] | None, source: dict[str, Any] | None) -> bool:
    if not candidate or not source:
        return False
    candidate_context = candidate.get("context_id")
    source_context = source.get("context_id")
    if candidate_context and source_context:
        return candidate_context == source_context
    return (
        _optional_int(candidate.get("source_action_index", candidate.get("action_index")))
        == _optional_int(source.get("source_action_index", source.get("action_index")))
        and _cell(candidate.get("policy_target_cell")) == _cell(source.get("policy_target_cell"))
        and _cell(candidate.get("execution_goal_cell")) == _cell(source.get("execution_goal_cell"))
        and candidate.get("target_binding_mode") == source.get("target_binding_mode")
    )


def _sample_exclusion_reason(
    preferred: dict[str, Any] | None,
    alternative: dict[str, Any] | None,
) -> str | None:
    if preferred is None:
        return "preferred_candidate_missing"
    if alternative is None:
        return "alternative_candidate_missing"
    if not preferred.get("context_id"):
        return "preferred_context_id_missing"
    if not alternative.get("context_id"):
        return "alternative_context_id_missing"
    if not _valid_cell(preferred.get("execution_goal_cell")):
        return "preferred_execution_goal_missing"
    if not _valid_cell(alternative.get("execution_goal_cell")):
        return "alternative_execution_goal_missing"
    if not _candidate_action_mask_valid(preferred):
        return "preferred_action_mask_invalid"
    if not _candidate_action_mask_valid(alternative):
        return "alternative_action_mask_invalid"
    for label, candidate in (("preferred", preferred), ("alternative", alternative)):
        if _float_or_none(candidate.get("path_cost")) is None:
            return f"{label}_path_cost_missing"
        if _float_or_none(candidate.get("risk")) is None:
            return f"{label}_risk_missing"
        if not isinstance(_candidate_features(candidate), list):
            return f"{label}_candidate_features_missing"
    return None


def _diagnostic(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "sample_type": "sequential_canary_failure_diagnostic",
        "episode_id": step.get("episode_id"),
        "step_index": step.get("step_index"),
        "scenario_id": step.get("scenario_id"),
        "scenario_group": step.get("scenario_group"),
        "input_start_cell": step.get("input_start_cell"),
        "source_selected_context_id": step.get("source_selected_context_id"),
        "raw_policy_selected_context_id": step.get("raw_policy_selected_context_id"),
        "policy_selected_context_id": step.get("policy_selected_context_id"),
        "canary_rejection_reason_codes": _string_list(step.get("canary_rejection_reason_codes")),
        "raw_policy_regression_reason_codes": _string_list(step.get("raw_policy_regression_reason_codes")),
        "controlled_regression_reason_codes": _string_list(step.get("controlled_regression_reason_codes")),
        "source_execution_goal_cell": step.get("source_execution_goal_cell"),
        "policy_execution_goal_cell": step.get("policy_execution_goal_cell"),
        "controlled_execution_goal_cell": step.get("controlled_execution_goal_cell"),
        "path_cost_delta": step.get("raw_policy_selected_path_cost_delta") or step.get("path_cost_delta"),
        "risk_delta": step.get("raw_policy_selected_risk_delta") or step.get("risk_delta"),
        "utility_delta": step.get("raw_policy_selected_utility_delta") or step.get("utility_delta"),
    }


def _allowed_reason_codes(step: dict[str, Any], config: dict[str, Any]) -> list[str]:
    allowed = set(config.get("sample", {}).get("supported_reason_codes") or [])
    if not allowed:
        allowed = {"path_cost_regression", "risk_regression"}
    reasons: list[str] = []
    for key in (
        "canary_rejection_reason_codes",
        "raw_policy_regression_reason_codes",
        "controlled_regression_reason_codes",
    ):
        for reason in _string_list(step.get(key)):
            if reason in allowed and reason not in reasons:
                reasons.append(reason)
    return reasons


def _sample_weight(reasons: list[str], config: dict[str, Any]) -> float:
    sample = config.get("sample", {})
    if MISSED_SAFE_CHOICE_REASON in reasons:
        return float(sample.get("missed_safe_choice_sample_weight", sample.get("sample_weight", 1.0)))
    weight = float(sample.get("sequential_hard_negative_loss_weight", sample.get("sample_weight", 1.0)))
    if "path_cost_regression" in reasons:
        weight *= float(sample.get("path_cost_regression_negative_weight", 1.0))
    if "risk_regression" in reasons:
        weight *= float(sample.get("risk_regression_negative_weight", 1.0))
    return weight


def _index_groups(groups: list[dict[str, Any]]) -> dict[tuple[str | None, str | None, str | None], dict[str, Any]]:
    index: dict[tuple[str | None, str | None, str | None], dict[str, Any]] = {}
    for group in groups:
        for key in (
            (group.get("source_path"), group.get("scenario_id"), group.get("run_id")),
            (group.get("source_path"), group.get("scenario_id"), None),
            (None, group.get("scenario_id"), group.get("run_id")),
            (None, group.get("scenario_id"), None),
        ):
            index.setdefault(key, group)
    return index


def _match_group(
    step: dict[str, Any],
    index: dict[tuple[str | None, str | None, str | None], dict[str, Any]],
) -> dict[str, Any] | None:
    for key in (
        (step.get("source_path"), step.get("scenario_id"), step.get("run_id")),
        (step.get("source_path"), step.get("scenario_id"), None),
        (None, step.get("scenario_id"), step.get("run_id")),
        (None, step.get("scenario_id"), None),
    ):
        group = index.get(key)
        if group is not None:
            return group
    return None


def _find_candidate(
    group: dict[str, Any],
    context_id: Any,
    *,
    action_index: Any = None,
    execution_goal_cell: Any = None,
) -> dict[str, Any] | None:
    if not context_id:
        return None
    candidates = [candidate for candidate in group.get("candidates", []) if candidate.get("context_id") == context_id]
    if not candidates:
        return None
    action = _optional_int(action_index)
    if action is not None:
        action_matches = [
            candidate
            for candidate in candidates
            if _optional_int(candidate.get("action_index")) == action
            or _optional_int(candidate.get("source_action_index")) == action
        ]
        if action_matches:
            candidates = action_matches
    goal = _cell(execution_goal_cell)
    if goal is not None:
        goal_matches = [candidate for candidate in candidates if _cell(candidate.get("execution_goal_cell")) == goal]
        if goal_matches:
            candidates = goal_matches
    return candidates[0]


def _embedded_group(step: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        candidate
        for candidate in (step.get("preferred_candidate"), step.get("alternative_candidate"))
        if isinstance(candidate, dict)
    ]
    return {
        "source_path": step.get("source_path"),
        "scenario_id": step.get("scenario_id"),
        "scenario_group": step.get("scenario_group"),
        "scenario_seed": step.get("scenario_seed"),
        "scenario_variant_id": step.get("scenario_variant_id"),
        "diagnostic_profile": step.get("diagnostic_profile"),
        "planning_backend": step.get("planning_backend"),
        "candidates": candidates,
    }


def _step_key(step: dict[str, Any]) -> str:
    return "|".join(
        str(step.get(field) or "")
        for field in (
            "episode_id",
            "step_index",
            "scenario_id",
            "source_selected_context_id",
            "raw_policy_selected_context_id",
        )
    )


def _exclusion(step: dict[str, Any], reason_code: str) -> dict[str, Any]:
    return {
        "reason_code": reason_code,
        "episode_id": step.get("episode_id"),
        "step_index": step.get("step_index"),
        "scenario_id": step.get("scenario_id"),
        "scenario_group": step.get("scenario_group"),
        "source_selected_context_id": step.get("source_selected_context_id"),
        "raw_policy_selected_context_id": step.get("raw_policy_selected_context_id"),
        "canary_rejection_reason_codes": _string_list(step.get("canary_rejection_reason_codes")),
    }


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
        "samples": batch_root / outputs["samples"],
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
    for section in ("input_files", "output_files", "sample", "validation"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _load_json(path: Path, label: str, reason_codes: list[str]) -> dict[str, Any]:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _append_reason(reason_codes, f"{label}_invalid_json")
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_jsonl(path: Path, label: str, reason_codes: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    except json.JSONDecodeError:
        _append_reason(reason_codes, f"{label}_invalid_json")
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _delta(candidate: Any, source: Any) -> float | None:
    left = _float_or_none(candidate)
    right = _float_or_none(source)
    if left is None or right is None:
        return None
    return left - right


def _cell(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return [int(value[0]), int(value[1])]
        except (TypeError, ValueError):
            return None
    return None


def _valid_cell(value: Any) -> bool:
    return _cell(value) is not None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
