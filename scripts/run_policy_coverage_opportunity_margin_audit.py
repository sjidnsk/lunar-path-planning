from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot


SUMMARY_SCHEMA_VERSION = "policy-coverage-opportunity-margin-audit-summary/v1"
ACTION_ROW_SCHEMA_VERSION = "policy-coverage-action-level-audit-row/v1"
POLICY_MARGIN_SCHEMA_VERSION = "policy-margin-audit/v1"
CANDIDATE_FEATURE_SCHEMA_VERSION = "policy-coverage-candidate-feature-audit/v1"
REJECTION_SCHEMA_VERSION = "policy-coverage-opportunity-rejection-report/v1"

COVERAGE_DRIVEN_SUMMARY_FILE = "coverage-driven-ppo-improvement-run-summary.json"
REPLAY_AUDIT_FILE = "coverage-driven-ppo-replay-audit.json"
UPDATE_SUMMARY_FILE = "coverage-driven-ppo-update-summary.json"
BATCH_EPISODES_FILE = "coverage-aware-ppo-batch/ppo-rollout-episodes.jsonl"
REWARD_REFINEMENT_SUMMARY_FILE = "coverage-aware-reward-refinement-summary.json"
COVERAGE_SIGNAL_SUMMARY_FILE = "exploration-coverage-signal-audit-summary.json"
COVERAGE_PERFORMANCE_SUMMARY_FILE = "exploration-coverage-performance-evaluation-summary.json"

SUMMARY_FILE = "policy-coverage-opportunity-margin-audit-summary.json"
ACTION_AUDIT_FILE = "policy-coverage-action-level-audit.jsonl"
POLICY_MARGIN_FILE = "policy-margin-audit.json"
CANDIDATE_FEATURE_FILE = "candidate-feature-audit.json"
REJECTION_REPORT_FILE = "policy-coverage-opportunity-rejection-report.json"
REPORT_FILE = "policy-coverage-opportunity-margin-audit-report.md"

TOLERANCE = 1.0e-9
MAX_ALLOWED_RELATIVE_COST_REGRESSION = 0.05
COVERAGE_SIGNAL_FIELDS = (
    "expected_coverage_rate_delta",
    "information_gain",
    "value",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit safe coverage opportunities and policy margins after a coverage-driven PPO run."
    )
    parser.add_argument("--coverage-driven-root", required=True)
    parser.add_argument("--reward-refinement-root", required=True)
    parser.add_argument("--coverage-signal-root", required=True)
    parser.add_argument("--coverage-performance-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--candidate-coverage-overlay")
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    summary = run_policy_coverage_opportunity_margin_audit(
        coverage_driven_root=_resolve_path(Path(args.coverage_driven_root), repo_root, repo_root),
        reward_refinement_root=_resolve_path(Path(args.reward_refinement_root), repo_root, repo_root),
        coverage_signal_root=_resolve_path(Path(args.coverage_signal_root), repo_root, repo_root),
        coverage_performance_root=_resolve_path(Path(args.coverage_performance_root), repo_root, repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root, repo_root),
        repo_root=repo_root,
        candidate_coverage_overlay_path=_resolve_optional_path(
            args.candidate_coverage_overlay,
            repo_root,
            repo_root,
        ),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "next_required_change": summary["next_required_change"],
                "performance_claimed": summary["performance_claimed"],
                "runs_new_ppo_update": summary["runs_new_ppo_update"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_policy_coverage_opportunity_margin_audit(
    *,
    coverage_driven_root: Path,
    reward_refinement_root: Path,
    coverage_signal_root: Path,
    coverage_performance_root: Path,
    output_root: Path,
    repo_root: Path,
    candidate_coverage_overlay_path: Path | None = None,
) -> dict[str, Any]:
    coverage_driven_root = Path(coverage_driven_root)
    reward_refinement_root = Path(reward_refinement_root)
    coverage_signal_root = Path(coverage_signal_root)
    coverage_performance_root = Path(coverage_performance_root)
    output_root = Path(output_root)
    repo_root = Path(repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    input_reasons: list[str] = []
    coverage_driven_summary = _read_json(
        coverage_driven_root / COVERAGE_DRIVEN_SUMMARY_FILE,
        input_reasons,
        "coverage_driven_summary_missing",
    )
    reward_summary = _read_json(
        reward_refinement_root / REWARD_REFINEMENT_SUMMARY_FILE,
        input_reasons,
        "reward_refinement_summary_missing",
    )
    coverage_signal_summary = _read_json(
        coverage_signal_root / COVERAGE_SIGNAL_SUMMARY_FILE,
        input_reasons,
        "coverage_signal_summary_missing",
    )
    coverage_performance_summary = _read_json(
        coverage_performance_root / COVERAGE_PERFORMANCE_SUMMARY_FILE,
        input_reasons,
        "coverage_performance_summary_missing",
    )

    replay_audit_path = _resolve_optional_path(
        coverage_driven_summary.get("replay_audit"),
        coverage_driven_root,
        repo_root,
    ) or (coverage_driven_root / REPLAY_AUDIT_FILE)
    update_summary_path = _resolve_optional_path(
        coverage_driven_summary.get("ppo_update_summary"),
        coverage_driven_root,
        repo_root,
    ) or (coverage_driven_root / UPDATE_SUMMARY_FILE)
    batch_episodes_path = _resolve_optional_path(
        coverage_driven_summary.get("coverage_aware_batch_episodes"),
        coverage_driven_root,
        repo_root,
    ) or (coverage_driven_root / BATCH_EPISODES_FILE)

    replay_audit = _read_json(replay_audit_path, input_reasons, "replay_audit_missing")
    update_summary = _read_json(update_summary_path, input_reasons, "ppo_update_summary_missing")
    episodes = _read_jsonl(batch_episodes_path, input_reasons, "coverage_aware_batch_episodes_missing")
    candidate_overlay_rows = (
        _read_jsonl(candidate_coverage_overlay_path, input_reasons, "candidate_coverage_overlay_missing")
        if candidate_coverage_overlay_path is not None
        else []
    )
    candidate_overlay_index = _index_candidate_coverage_overlay(candidate_overlay_rows)

    scorer_reasons: list[str] = []
    pre_scorer = _load_checkpoint_scorer(
        checkpoint_path=_pre_checkpoint_path(coverage_driven_summary, repo_root),
        metadata_path=_pre_metadata_path(coverage_driven_summary, repo_root),
        repo_root=repo_root,
        reason_codes=scorer_reasons,
        reason_label="pre_policy_checkpoint_unscorable",
    )
    post_scorer = _load_checkpoint_scorer(
        checkpoint_path=_resolve_optional_path(
            coverage_driven_summary.get("checkpoint_path"),
            coverage_driven_root,
            repo_root,
        ),
        metadata_path=_resolve_optional_path(
            coverage_driven_summary.get("checkpoint_metadata_path"),
            coverage_driven_root,
            repo_root,
        ),
        repo_root=repo_root,
        reason_codes=scorer_reasons,
        reason_label="post_policy_checkpoint_unscorable",
    )

    replay_index = _index_replay_rows(replay_audit.get("rows") if isinstance(replay_audit, dict) else [])
    context_audits: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    feature_counter = _FeatureCounter()

    for transition_index, transition in enumerate(_iter_transitions(episodes)):
        context = _context_audit(
            transition_index=transition_index,
            transition=transition,
            replay_row=_matching_replay_row(transition, replay_index),
            pre_scorer=pre_scorer,
            post_scorer=post_scorer,
            candidate_overlay_index=candidate_overlay_index,
        )
        context_audits.append(context)
        action_rows.extend(context["action_rows"])
        feature_counter.add_context(context)

    margin_audit = _policy_margin_audit(context_audits)
    candidate_feature_audit = feature_counter.audit(context_audits)
    controlled_regression_count = _controlled_regression_count(
        context_audits,
        coverage_driven_summary,
        reward_summary,
        coverage_signal_summary,
    )
    fallback_gain_contamination_count = _fallback_gain_contamination_count(
        context_audits,
        coverage_driven_summary,
        reward_summary,
        coverage_signal_summary,
    )

    safe_better_contexts = [context for context in context_audits if context["safe_better_than_teacher"]]
    safe_better_family_count = len(
        {
            context.get("scenario_family")
            for context in safe_better_contexts
            if context.get("scenario_family") is not None
        }
    )
    reason_codes = _diagnostic_reason_codes(
        input_reasons=input_reasons,
        scorer_reasons=scorer_reasons,
        safe_better_count=len(safe_better_contexts),
        candidate_feature_audit=candidate_feature_audit,
        policy_margin_audit=margin_audit,
        controlled_regression_count=controlled_regression_count,
        fallback_gain_contamination_count=fallback_gain_contamination_count,
    )
    next_required_change = _next_required_change(
        reason_codes=reason_codes,
        safe_better_count=len(safe_better_contexts),
        candidate_feature_audit=candidate_feature_audit,
        controlled_regression_count=controlled_regression_count,
        fallback_gain_contamination_count=fallback_gain_contamination_count,
    )
    status = (
        "failed"
        if input_reasons or controlled_regression_count > 0 or fallback_gain_contamination_count > 0
        else "passed"
    )

    paths = _paths(output_root)
    rejection_report = _rejection_report(
        status=status,
        reason_codes=reason_codes,
        next_required_change=next_required_change,
        safe_better_count=len(safe_better_contexts),
        candidate_feature_audit=candidate_feature_audit,
        policy_margin_audit=margin_audit,
        controlled_regression_count=controlled_regression_count,
        fallback_gain_contamination_count=fallback_gain_contamination_count,
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": next_required_change,
        "policy_coverage_opportunity_margin_audit_status": status,
        "coverage_driven_root": str(coverage_driven_root),
        "reward_refinement_root": str(reward_refinement_root),
        "coverage_signal_root": str(coverage_signal_root),
        "coverage_performance_root": str(coverage_performance_root),
        "candidate_coverage_overlay": str(candidate_coverage_overlay_path) if candidate_coverage_overlay_path else None,
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "action_level_audit": str(paths["action_audit"]),
        "policy_margin_audit": str(paths["policy_margin"]),
        "candidate_feature_audit": str(paths["candidate_feature"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "coverage_driven_status": coverage_driven_summary.get("status"),
        "coverage_driven_reason_codes": coverage_driven_summary.get("reason_codes", []),
        "coverage_return_improvement": _float_or_default(
            coverage_driven_summary.get("coverage_return_improvement"),
            0.0,
        ),
        "cumulative_coverage_rate_delta_improvement": _float_or_default(
            coverage_driven_summary.get("cumulative_coverage_rate_delta_improvement"),
            0.0,
        ),
        "valuable_area_covered_improvement": _float_or_default(
            coverage_driven_summary.get("valuable_area_covered_improvement"),
            0.0,
        ),
        "context_count": len(context_audits),
        "action_candidate_row_count": len(action_rows),
        "safe_candidate_row_count": sum(1 for row in action_rows if row["action_mask_valid"]),
        "safe_non_teacher_candidate_count": sum(
            1
            for row in action_rows
            if row["action_mask_valid"] and not row["is_teacher_action"]
        ),
        "safe_better_than_teacher_count": len(safe_better_contexts),
        "safe_better_than_teacher_family_count": safe_better_family_count,
        "policy_argmax_changed_count": margin_audit["policy_argmax_changed_count"],
        "post_update_teacher_equal_raw_count": margin_audit["post_update_teacher_equal_raw_count"],
        "post_update_teacher_equal_controlled_count": margin_audit[
            "post_update_teacher_equal_controlled_count"
        ],
        "teacher_margin_sample_count": margin_audit["teacher_margin_sample_count"],
        "missing_teacher_signal_count": margin_audit["missing_teacher_signal_count"],
        "candidate_expected_coverage_nonzero_count": candidate_feature_audit[
            "candidate_expected_coverage_nonzero_count"
        ],
        "candidate_information_gain_nonzero_count": candidate_feature_audit[
            "candidate_information_gain_nonzero_count"
        ],
        "candidate_value_nonzero_count": candidate_feature_audit["candidate_value_nonzero_count"],
        "candidate_coverage_overlay_row_count": len(candidate_overlay_rows),
        "candidate_coverage_overlay_connected_count": sum(
            1 for row in action_rows if row.get("candidate_coverage_overlay_connected")
        ),
        "candidate_coverage_overlay_missing_source_count": sum(
            1
            for row in action_rows
            if row.get("candidate_coverage_overlay_row_present")
            and not row.get("candidate_coverage_overlay_connected")
        ),
        "fallback_gain_contamination_count": fallback_gain_contamination_count,
        "controlled_regression_count": controlled_regression_count,
        "candidate_feature_decision_signal_usable": candidate_feature_audit[
            "decision_time_coverage_features_usable"
        ],
        "best_safe_non_teacher_alternative_count": sum(
            1 for context in context_audits if context.get("best_safe_non_teacher_action_index") is not None
        ),
        "coverage_performance_status": coverage_performance_summary.get("coverage_performance_status"),
        "coverage_performance_reason_codes": coverage_performance_summary.get("reason_codes", []),
        "coverage_signal_status": coverage_signal_summary.get("coverage_signal_status"),
        "reward_refinement_status": reward_summary.get("reward_refinement_status"),
        "ppo_update_optimizer_train_transition_count": update_summary.get("optimizer_train_transition_count", 0),
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_release_claimed": False,
        "formal_training_ready_claimed": False,
        "connects_real_executor": False,
        "relaxes_guard": False,
        "modifies_network_or_action_space": False,
        "modifies_default_astar": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }

    _write_jsonl(paths["action_audit"], action_rows)
    _write_json(paths["policy_margin"], margin_audit)
    _write_json(paths["candidate_feature"], candidate_feature_audit)
    _write_json(paths["rejection_report"], rejection_report)
    _write_json(paths["summary"], summary)
    paths["report"].write_text(
        _render_report(summary, candidate_feature_audit, margin_audit, rejection_report),
        encoding="utf-8",
    )
    return summary


def _context_audit(
    *,
    transition_index: int,
    transition: dict[str, Any],
    replay_row: dict[str, Any] | None,
    pre_scorer: Callable[[dict[str, Any]], dict[str, Any]] | None,
    post_scorer: Callable[[dict[str, Any]], dict[str, Any]] | None,
    candidate_overlay_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    info = transition.get("info") if isinstance(transition.get("info"), dict) else {}
    observation = transition.get("observation") if isinstance(transition.get("observation"), dict) else {}
    context_id = str(
        info.get("context_id")
        or transition.get("context_id")
        or (replay_row or {}).get("context_id")
        or f"transition-{transition_index}"
    )
    teacher_action_index = _first_int(
        info.get("teacher_action_index"),
        transition.get("teacher_action_index"),
        (replay_row or {}).get("teacher_action_index"),
    )
    pre_selected_action_index = _first_int(
        info.get("controlled_action_index"),
        transition.get("controlled_action_index"),
        transition.get("action_index"),
    )
    post_raw_action_index = _first_int((replay_row or {}).get("raw_policy_action_index"))
    post_controlled_action_index = _first_int((replay_row or {}).get("controlled_action_index"))
    if post_controlled_action_index is None:
        post_controlled_action_index = pre_selected_action_index
    if post_raw_action_index is None:
        post_raw_action_index = post_controlled_action_index

    action_mask = _bool_list(observation.get("action_mask") or transition.get("action_mask"))
    candidate_features = _list_of_lists(observation.get("candidate_features"))
    candidate_feature_names = [str(name) for name in observation.get("candidate_feature_names", [])]
    candidate_cells = _list_of_lists(observation.get("candidate_cells"))
    candidate_count = max(len(candidate_features), len(action_mask), len(candidate_cells))
    if candidate_count <= 0:
        candidate_count = max(
            _max_int(teacher_action_index, pre_selected_action_index, post_raw_action_index, post_controlled_action_index)
            + 1,
            0,
        )

    pre_scores = _score_transition(
        transition=transition,
        observation=observation,
        scorer=pre_scorer,
        logits_field="pre_policy_logits",
        probs_field="pre_policy_probs",
    )
    post_scores = _score_transition(
        transition=transition,
        observation=observation,
        scorer=post_scorer,
        logits_field="post_policy_logits",
        probs_field="post_policy_probs",
    )
    pre_argmax = _masked_argmax(pre_scores.get("logits"), action_mask)
    post_argmax = _masked_argmax(post_scores.get("logits"), action_mask)
    if post_raw_action_index is None:
        post_raw_action_index = post_argmax

    feature_index = {name: index for index, name in enumerate(candidate_feature_names)}
    episode_id = info.get("episode_id") or transition.get("episode_id") or (replay_row or {}).get("episode_id")
    step_index = _first_int(info.get("step_index"), transition.get("step_index"), (replay_row or {}).get("step_index"))
    scenario_id = info.get("scenario_id") or (replay_row or {}).get("scenario_id")
    scenario_family = info.get("scenario_family") or (replay_row or {}).get("scenario_family")
    split = info.get("split") or (replay_row or {}).get("split")
    candidate_feature_maps: list[dict[str, Any]] = []
    candidate_overlays: list[dict[str, Any] | None] = []
    for action_index in range(candidate_count):
        overlay = _matching_candidate_coverage_overlay(
            candidate_overlay_index=candidate_overlay_index,
            context_id=context_id,
            episode_id=episode_id,
            step_index=step_index,
            scenario_id=scenario_id,
            scenario_family=scenario_family,
            split=split,
            action_index=action_index,
            candidate_cell=_candidate_cell(candidate_cells, action_index),
        )
        features = _candidate_feature_map(candidate_features, feature_index, action_index)
        if overlay is not None:
            features = _merge_candidate_coverage_overlay(features, overlay)
        candidate_feature_maps.append(features)
        candidate_overlays.append(overlay)

    teacher_features = _feature_map_at(candidate_feature_maps, teacher_action_index)
    teacher_decision_score = _coverage_decision_score(teacher_features)
    teacher_risk = _float_or_none(teacher_features.get("risk"))
    teacher_path_cost = _float_or_none(teacher_features.get("path_cost"))
    teacher_energy = _float_or_none(teacher_features.get("energy_cost"))
    best_safe = _best_safe_non_teacher_candidate(
        candidate_feature_maps=candidate_feature_maps,
        action_mask=action_mask,
        teacher_action_index=teacher_action_index,
        teacher_decision_score=teacher_decision_score,
        teacher_risk=teacher_risk,
        teacher_path_cost=teacher_path_cost,
        teacher_energy=teacher_energy,
    )

    action_rows = []
    for action_index in range(candidate_count):
        features = _feature_map_at(candidate_feature_maps, action_index)
        overlay = candidate_overlays[action_index] if action_index < len(candidate_overlays) else None
        overlay_connected = bool(overlay and overlay.get("coverage_source_available"))
        overlay_missing_reasons = _string_list((overlay or {}).get("missing_reason_codes"))
        action_mask_valid = _mask_value(action_mask, action_index)
        is_teacher = action_index == teacher_action_index
        is_selected = action_index == pre_selected_action_index
        observed_executed_action = action_index == pre_selected_action_index
        decision_score = _coverage_decision_score(features)
        expected_coverage = _float_or_default(features.get("expected_coverage_rate_delta"), 0.0)
        information_gain = _float_or_default(features.get("information_gain"), 0.0)
        value_signal = _float_or_default(features.get("value"), 0.0)
        utility = _float_or_default(features.get("utility"), 0.0)
        actual_coverage_rate_delta = _float_or_none(info.get("coverage_rate_delta")) if observed_executed_action else None
        actual_valuable = _float_or_none(info.get("valuable_area_covered")) if observed_executed_action else None
        row = {
            "schema_version": ACTION_ROW_SCHEMA_VERSION,
            "audit_index": len(action_rows),
            "transition_index": transition_index,
            "context_id": context_id,
            "episode_id": episode_id,
            "step_index": step_index,
            "scenario_id": scenario_id,
            "scenario_family": scenario_family,
            "action_index": action_index,
            "candidate_cell": _candidate_cell(candidate_cells, action_index),
            "action_mask_valid": action_mask_valid,
            "teacher_action_index": teacher_action_index,
            "pre_improvement_selected_action_index": pre_selected_action_index,
            "post_update_raw_action_index": post_raw_action_index,
            "post_update_controlled_action_index": post_controlled_action_index,
            "is_teacher_action": is_teacher,
            "is_pre_improvement_selected_action": is_selected,
            "is_post_update_raw_action": action_index == post_raw_action_index,
            "is_post_update_controlled_action": action_index == post_controlled_action_index,
            "is_best_safe_non_teacher_alternative": action_index == best_safe.get("action_index"),
            "safe_better_than_teacher": action_index == best_safe.get("action_index") and best_safe.get("better_than_teacher", False),
            "safe_better_basis": best_safe.get("basis") if action_index == best_safe.get("action_index") else None,
            "candidate_expected_coverage_rate_delta": expected_coverage,
            "candidate_expected_new_coverage_area": _float_or_default(
                features.get("expected_new_coverage_area"),
                0.0,
            ),
            "candidate_information_gain": information_gain,
            "candidate_confidence_gain": _float_or_default(features.get("confidence_gain"), 0.0),
            "candidate_value": value_signal,
            "candidate_utility": utility,
            "candidate_decision_coverage_score": decision_score,
            "teacher_decision_coverage_score": teacher_decision_score,
            "candidate_valuable_coverage_proxy": expected_coverage * utility,
            "candidate_coverage_overlay_row_present": overlay is not None,
            "candidate_coverage_overlay_connected": overlay_connected,
            "candidate_coverage_source_path": (overlay or {}).get("source_path") if overlay_connected else None,
            "candidate_coverage_match_method": (overlay or {}).get("match_method") if overlay_connected else None,
            "candidate_coverage_source_confidence": _float_or_none((overlay or {}).get("source_confidence")) if overlay_connected else None,
            "candidate_coverage_missing_reason_codes": overlay_missing_reasons,
            "coverage_rate_delta": actual_coverage_rate_delta,
            "valuable_area_covered": actual_valuable,
            "new_area_covered": _float_or_none(info.get("new_area_covered")) if observed_executed_action else None,
            "information_gain_executed": _float_or_none(info.get("information_gain")) if observed_executed_action else None,
            "path_cost": _float_or_default(features.get("path_cost"), _float_or_default(info.get("path_cost"), 0.0)),
            "risk": _float_or_default(features.get("risk"), _float_or_default(info.get("risk"), 0.0)),
            "energy_cost": _float_or_default(features.get("energy_cost"), _float_or_default(info.get("energy_cost"), 0.0)),
            "pre_policy_logit": _list_value(pre_scores.get("logits"), action_index),
            "pre_policy_prob": _list_value(pre_scores.get("probs"), action_index),
            "post_policy_logit": _list_value(post_scores.get("logits"), action_index),
            "post_policy_prob": _list_value(post_scores.get("probs"), action_index),
            "pre_policy_argmax": pre_argmax,
            "post_policy_argmax": post_argmax,
            "actual_coverage_gain_source": info.get("actual_coverage_gain_source"),
            "fallback_like": bool(info.get("fallback_like") or (replay_row or {}).get("fallback_like")),
            "guard_rejected": bool((replay_row or {}).get("guard_rejected")),
            "policy_action_accepted": bool((replay_row or {}).get("policy_action_accepted", True)),
            "controlled_regression_reason_codes": _string_list(
                info.get("controlled_regression_reason_codes")
                or (replay_row or {}).get("controlled_regression_reason_codes")
            ),
            "counterfactual_coverage_observed": observed_executed_action,
        }
        action_rows.append(row)

    post_teacher_margin = _teacher_margin(post_scores.get("logits"), action_mask, teacher_action_index)
    pre_teacher_margin = _teacher_margin(pre_scores.get("logits"), action_mask, teacher_action_index)
    return {
        "context_id": context_id,
        "episode_id": episode_id,
        "step_index": step_index,
        "scenario_id": scenario_id,
        "scenario_family": scenario_family,
        "teacher_action_index": teacher_action_index,
        "pre_improvement_selected_action_index": pre_selected_action_index,
        "post_update_raw_action_index": post_raw_action_index,
        "post_update_controlled_action_index": post_controlled_action_index,
        "pre_policy_argmax": pre_argmax,
        "post_policy_argmax": post_argmax,
        "pre_teacher_margin": pre_teacher_margin,
        "post_teacher_margin": post_teacher_margin,
        "teacher_margin_available": pre_teacher_margin is not None or post_teacher_margin is not None,
        "missing_teacher_signal": teacher_action_index is None,
        "best_safe_non_teacher_action_index": best_safe.get("action_index"),
        "best_safe_non_teacher_decision_coverage_score": best_safe.get("decision_score"),
        "best_safe_non_teacher_margin_over_teacher": best_safe.get("margin_over_teacher"),
        "safe_better_than_teacher": bool(best_safe.get("better_than_teacher")),
        "safe_better_basis": best_safe.get("basis"),
        "fallback_like": bool(info.get("fallback_like") or (replay_row or {}).get("fallback_like")),
        "guard_rejected": bool((replay_row or {}).get("guard_rejected")),
        "controlled_regression_reason_codes": _string_list(
            info.get("controlled_regression_reason_codes")
            or (replay_row or {}).get("controlled_regression_reason_codes")
        ),
        "action_rows": action_rows,
    }


class _FeatureCounter:
    def __init__(self) -> None:
        self.candidate_row_count = 0
        self.nonzero_counts: Counter[str] = Counter()
        self.present_counts: Counter[str] = Counter()
        self.missing_indicator_counts: Counter[str] = Counter()
        self.feature_name_counts: Counter[str] = Counter()

    def add_context(self, context: dict[str, Any]) -> None:
        for row in context["action_rows"]:
            self.candidate_row_count += 1
            for field, row_field in (
                ("expected_coverage_rate_delta", "candidate_expected_coverage_rate_delta"),
                ("information_gain", "candidate_information_gain"),
                ("value", "candidate_value"),
                ("expected_new_coverage_area", "candidate_expected_new_coverage_area"),
                ("confidence_gain", "candidate_confidence_gain"),
            ):
                value = _float_or_default(row.get(row_field), 0.0)
                self.present_counts[field] += 1
                if abs(value) > TOLERANCE:
                    self.nonzero_counts[field] += 1
            for key in row:
                if key.startswith("candidate_"):
                    self.feature_name_counts[key] += 1

    def audit(self, contexts: list[dict[str, Any]]) -> dict[str, Any]:
        expected_nonzero = self.nonzero_counts["expected_coverage_rate_delta"]
        info_nonzero = self.nonzero_counts["information_gain"]
        value_nonzero = self.nonzero_counts["value"]
        decision_signal_nonzero = expected_nonzero + info_nonzero + value_nonzero
        return {
            "schema_version": CANDIDATE_FEATURE_SCHEMA_VERSION,
            "context_count": len(contexts),
            "action_candidate_row_count": self.candidate_row_count,
            "candidate_expected_coverage_nonzero_count": expected_nonzero,
            "candidate_information_gain_nonzero_count": info_nonzero,
            "candidate_value_nonzero_count": value_nonzero,
            "candidate_expected_new_coverage_nonzero_count": self.nonzero_counts[
                "expected_new_coverage_area"
            ],
            "candidate_confidence_gain_nonzero_count": self.nonzero_counts["confidence_gain"],
            "coverage_signal_nonzero_count": decision_signal_nonzero,
            "decision_time_coverage_features_usable": decision_signal_nonzero > 0,
            "feature_nonzero_counts": dict(sorted(self.nonzero_counts.items())),
            "feature_present_counts": dict(sorted(self.present_counts.items())),
            "feature_name_counts": dict(sorted(self.feature_name_counts.items())),
        }


def _policy_margin_audit(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    pre_margins = [context["pre_teacher_margin"] for context in contexts if context["pre_teacher_margin"] is not None]
    post_margins = [
        context["post_teacher_margin"] for context in contexts if context["post_teacher_margin"] is not None
    ]
    changed_contexts = [
        context
        for context in contexts
        if context.get("pre_policy_argmax") is not None
        and context.get("post_policy_argmax") is not None
        and context.get("pre_policy_argmax") != context.get("post_policy_argmax")
    ]
    return {
        "schema_version": POLICY_MARGIN_SCHEMA_VERSION,
        "context_count": len(contexts),
        "teacher_margin_sample_count": sum(1 for context in contexts if context["teacher_margin_available"]),
        "missing_teacher_signal_count": sum(1 for context in contexts if context["missing_teacher_signal"]),
        "policy_argmax_changed_count": len(changed_contexts),
        "post_update_teacher_equal_raw_count": sum(
            1
            for context in contexts
            if context.get("teacher_action_index") is not None
            and context.get("post_update_raw_action_index") == context.get("teacher_action_index")
        ),
        "post_update_teacher_equal_controlled_count": sum(
            1
            for context in contexts
            if context.get("teacher_action_index") is not None
            and context.get("post_update_controlled_action_index") == context.get("teacher_action_index")
        ),
        "pre_teacher_margin_stats": _stats(pre_margins),
        "post_teacher_margin_stats": _stats(post_margins),
        "pre_policy_argmax_counts": _counter_to_str(
            Counter(context.get("pre_policy_argmax") for context in contexts)
        ),
        "post_policy_argmax_counts": _counter_to_str(
            Counter(context.get("post_policy_argmax") for context in contexts)
        ),
        "post_update_raw_action_counts": _counter_to_str(
            Counter(context.get("post_update_raw_action_index") for context in contexts)
        ),
        "post_update_controlled_action_counts": _counter_to_str(
            Counter(context.get("post_update_controlled_action_index") for context in contexts)
        ),
        "changed_context_sample": [
            {
                "context_id": context["context_id"],
                "teacher_action_index": context["teacher_action_index"],
                "pre_policy_argmax": context["pre_policy_argmax"],
                "post_policy_argmax": context["post_policy_argmax"],
                "pre_teacher_margin": context["pre_teacher_margin"],
                "post_teacher_margin": context["post_teacher_margin"],
            }
            for context in changed_contexts[:50]
        ],
    }


def _best_safe_non_teacher_candidate(
    *,
    candidate_feature_maps: list[dict[str, Any]],
    action_mask: list[bool],
    teacher_action_index: int | None,
    teacher_decision_score: float,
    teacher_risk: float | None,
    teacher_path_cost: float | None,
    teacher_energy: float | None,
) -> dict[str, Any]:
    if teacher_action_index is None:
        return {}
    best: dict[str, Any] = {}
    for action_index, features in enumerate(candidate_feature_maps):
        if action_index == teacher_action_index or not _mask_value(action_mask, action_index):
            continue
        decision_score = _coverage_decision_score(features)
        risk = _float_or_none(features.get("risk"))
        path_cost = _float_or_none(features.get("path_cost"))
        energy = _float_or_none(features.get("energy_cost"))
        cost_safe = (
            _not_materially_worse(risk, teacher_risk)
            and _not_materially_worse(path_cost, teacher_path_cost)
            and _not_materially_worse(energy, teacher_energy)
        )
        better = decision_score > teacher_decision_score + TOLERANCE and cost_safe
        margin = decision_score - teacher_decision_score
        candidate = {
            "action_index": action_index,
            "decision_score": decision_score,
            "margin_over_teacher": margin,
            "better_than_teacher": better,
            "basis": "decision_time_candidate_coverage_features_and_action_mask",
        }
        if not best or (
            bool(candidate["better_than_teacher"]),
            candidate["decision_score"],
            -_float_or_default(path_cost, 0.0),
            -_float_or_default(risk, 0.0),
        ) > (
            bool(best.get("better_than_teacher")),
            _float_or_default(best.get("decision_score"), 0.0),
            -_float_or_default(best.get("path_cost"), 0.0),
            -_float_or_default(best.get("risk"), 0.0),
        ):
            candidate["path_cost"] = path_cost
            candidate["risk"] = risk
            best = candidate
    return best


def _diagnostic_reason_codes(
    *,
    input_reasons: list[str],
    scorer_reasons: list[str],
    safe_better_count: int,
    candidate_feature_audit: dict[str, Any],
    policy_margin_audit: dict[str, Any],
    controlled_regression_count: int,
    fallback_gain_contamination_count: int,
) -> list[str]:
    reasons: list[str] = []
    for reason in input_reasons:
        _add_reason(reasons, reason)
    if controlled_regression_count > 0:
        _add_reason(reasons, "controlled_regression_present")
    if fallback_gain_contamination_count > 0:
        _add_reason(reasons, "fallback_gain_contamination_present")
    if not candidate_feature_audit["decision_time_coverage_features_usable"]:
        _add_reason(reasons, "candidate_coverage_features_missing")
    if safe_better_count <= 0:
        _add_reason(reasons, "no_safe_better_than_teacher_alternatives")
    if (
        policy_margin_audit["context_count"] > 0
        and policy_margin_audit["post_update_teacher_equal_raw_count"] == policy_margin_audit["context_count"]
        and policy_margin_audit["post_update_teacher_equal_controlled_count"] == policy_margin_audit["context_count"]
    ):
        _add_reason(reasons, "post_update_policy_teacher_equivalent")
    for reason in scorer_reasons:
        _add_reason(reasons, reason)
    return reasons


def _next_required_change(
    *,
    reason_codes: list[str],
    safe_better_count: int,
    candidate_feature_audit: dict[str, Any],
    controlled_regression_count: int,
    fallback_gain_contamination_count: int,
) -> str:
    if controlled_regression_count > 0 or fallback_gain_contamination_count > 0:
        return "fix_policy_coverage_audit_inputs"
    if reason_codes and any(reason.endswith("_missing") for reason in reason_codes if reason.startswith("coverage_")):
        return "fix_policy_coverage_audit_inputs"
    if safe_better_count > 0 and candidate_feature_audit["decision_time_coverage_features_usable"]:
        return "refine_coverage_reward_or_advantage"
    return "collect_more_policy_differentiating_coverage"


def _rejection_report(
    *,
    status: str,
    reason_codes: list[str],
    next_required_change: str,
    safe_better_count: int,
    candidate_feature_audit: dict[str, Any],
    policy_margin_audit: dict[str, Any],
    controlled_regression_count: int,
    fallback_gain_contamination_count: int,
) -> dict[str, Any]:
    recommendations = (
        [
            "reduce or condition teacher_skill_retention_bonus",
            "increase coverage-return advantage or add pairwise margin loss for safe better alternatives",
            "normalize coverage, risk, path cost, and energy terms before changing policy rank",
            "keep hard guard and controlled regression penalties unchanged",
        ]
        if next_required_change == "refine_coverage_reward_or_advantage"
        else [
            "materialize counterfactual candidate coverage gain at decision time",
            "populate expected_coverage_rate_delta, information_gain, and value for each candidate",
            "collect policy-differentiating coverage rollouts with explicit safe non-teacher labels",
            "keep fallback/source gains separate from policy gains",
        ]
    )
    return {
        "schema_version": REJECTION_SCHEMA_VERSION,
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": next_required_change,
        "safe_better_than_teacher_count": safe_better_count,
        "candidate_feature_decision_signal_usable": candidate_feature_audit[
            "decision_time_coverage_features_usable"
        ],
        "post_update_teacher_equal_raw_count": policy_margin_audit[
            "post_update_teacher_equal_raw_count"
        ],
        "post_update_teacher_equal_controlled_count": policy_margin_audit[
            "post_update_teacher_equal_controlled_count"
        ],
        "controlled_regression_count": controlled_regression_count,
        "fallback_gain_contamination_count": fallback_gain_contamination_count,
        "performance_claim_rejected": True,
        "recommendations": recommendations,
    }


def _render_report(
    summary: dict[str, Any],
    candidate_feature_audit: dict[str, Any],
    margin_audit: dict[str, Any],
    rejection_report: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# Policy Coverage Opportunity / Margin Audit v1",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- contexts: `{summary['context_count']}`",
            f"- action candidates: `{summary['action_candidate_row_count']}`",
            f"- safe_better_than_teacher_count: `{summary['safe_better_than_teacher_count']}`",
            f"- safe_better_than_teacher_family_count: `{summary['safe_better_than_teacher_family_count']}`",
            f"- policy_argmax_changed_count: `{summary['policy_argmax_changed_count']}`",
            f"- post_update_teacher_equal_raw_count: `{summary['post_update_teacher_equal_raw_count']}`",
            f"- post_update_teacher_equal_controlled_count: `{summary['post_update_teacher_equal_controlled_count']}`",
            f"- candidate_expected_coverage_nonzero_count: `{candidate_feature_audit['candidate_expected_coverage_nonzero_count']}`",
            f"- candidate_information_gain_nonzero_count: `{candidate_feature_audit['candidate_information_gain_nonzero_count']}`",
            f"- candidate_value_nonzero_count: `{candidate_feature_audit['candidate_value_nonzero_count']}`",
            f"- teacher_margin_sample_count: `{margin_audit['teacher_margin_sample_count']}`",
            f"- missing_teacher_signal_count: `{margin_audit['missing_teacher_signal_count']}`",
            f"- controlled_regression_count: `{summary['controlled_regression_count']}`",
            f"- fallback_gain_contamination_count: `{summary['fallback_gain_contamination_count']}`",
            "",
            "## Interpretation",
            "",
            _interpretation(summary),
            "",
            "## Recommended Next Step",
            "",
            *[f"- {item}" for item in rejection_report["recommendations"]],
            "",
            "## Guardrails",
            "",
            "- This diagnostic did not run PPO.",
            "- It did not publish or replace any checkpoint.",
            "- It does not claim coverage performance improvement.",
        ]
    )


def _interpretation(summary: dict[str, Any]) -> str:
    if summary["next_required_change"] == "refine_coverage_reward_or_advantage":
        return (
            "Safe non-teacher alternatives with stronger decision-time coverage signal exist. "
            "The next step should change the reward, advantage, or margin objective so those "
            "alternatives can outrank teacher-equivalent actions without relaxing guards."
        )
    if summary["next_required_change"] == "collect_more_policy_differentiating_coverage":
        return (
            "The current audited decisions do not expose enough usable candidate-level coverage "
            "signal to prove a better safe non-teacher action. The next step is data/materialization, "
            "not another PPO update."
        )
    return "The audit inputs need repair before reward or data-routing decisions are reliable."


def _score_transition(
    *,
    transition: dict[str, Any],
    observation: dict[str, Any],
    scorer: Callable[[dict[str, Any]], dict[str, Any]] | None,
    logits_field: str,
    probs_field: str,
) -> dict[str, Any]:
    logits = _float_list(transition.get(logits_field))
    probs = _float_list(transition.get(probs_field))
    if logits:
        return {"logits": logits, "probs": probs or _softmax(logits)}
    if scorer is None or not observation:
        return {"logits": [], "probs": []}
    try:
        return scorer(observation)
    except Exception:
        return {"logits": [], "probs": []}


def _load_checkpoint_scorer(
    *,
    checkpoint_path: Path | None,
    metadata_path: Path | None,
    repo_root: Path,
    reason_codes: list[str],
    reason_label: str,
) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
    if checkpoint_path is None or not checkpoint_path.is_file():
        return None
    try:
        import torch

        model_src = repo_root / "model-explorer" / "src"
        if str(model_src) not in sys.path:
            sys.path.insert(0, str(model_src))
        from model_explorer.policy.architectures import build_policy_network
        from model_explorer.policy.features import PolicyObservation
        from model_explorer.policy.torch_policy import observation_to_tensors
    except Exception:
        _add_reason(reason_codes, reason_label)
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = _read_json(metadata_path, [], "metadata_missing") if metadata_path else {}
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
        if not state_dict:
            _add_reason(reason_codes, reason_label)
            return None
        architecture = checkpoint.get("architecture") or metadata.get("architecture")
        training = checkpoint.get("training") if isinstance(checkpoint.get("training"), dict) else {}
        hidden_size = _int(training.get("hidden_size") or metadata.get("hidden_size") or 64)
        architecture_config = (
            checkpoint.get("architecture_config")
            or training.get("architecture_config")
            or metadata.get("architecture_config")
        )
        network_cache: dict[tuple[int, int, int], Any] = {}

        def score(observation_dict: dict[str, Any]) -> dict[str, Any]:
            observation = _policy_observation_from_dict(observation_dict, PolicyObservation)
            signature = (
                len(observation.candidate_feature_names),
                len(observation.global_feature_names),
                len(observation.candidate_missing_indicator_names),
            )
            if signature not in network_cache:
                network = build_policy_network(
                    architecture,
                    observation=observation,
                    hidden_size=hidden_size,
                    architecture_config=architecture_config,
                )
                network.load_state_dict(state_dict)
                network.eval()
                network_cache[signature] = network
            network = network_cache[signature]
            with torch.no_grad():
                output = network(**observation_to_tensors(observation))
            logits = [float(value) for value in output.masked_logits[0].detach().cpu().tolist()]
            probs = [float(value) for value in output.action_probs[0].detach().cpu().tolist()]
            return {"logits": logits, "probs": probs}

        return score
    except Exception:
        _add_reason(reason_codes, reason_label)
        return None


def _policy_observation_from_dict(observation: dict[str, Any], policy_observation_type: Any) -> Any:
    return policy_observation_type(
        candidate_feature_names=tuple(str(name) for name in observation.get("candidate_feature_names", [])),
        candidate_features=tuple(tuple(_float_or_default(value, 0.0) for value in row) for row in _list_of_lists(observation.get("candidate_features"))),
        global_feature_names=tuple(str(name) for name in observation.get("global_feature_names", [])),
        global_features=tuple(_float_or_default(value, 0.0) for value in observation.get("global_features", [])),
        action_mask=tuple(_bool_list(observation.get("action_mask"))),
        candidate_cells=tuple(
            tuple(int(value) for value in row[:2]) if isinstance(row, list) and len(row) >= 2 else None
            for row in _list_of_lists(observation.get("candidate_cells"))
        ),
        candidate_missing_feature_names=tuple(
            tuple(str(value) for value in row)
            for row in _list_of_lists(observation.get("candidate_missing_feature_names"))
        ),
        candidate_missing_indicator_names=tuple(
            str(name) for name in observation.get("candidate_missing_indicator_names", [])
        ),
        candidate_missing_indicators=tuple(
            tuple(_float_or_default(value, 0.0) for value in row)
            for row in _list_of_lists(observation.get("candidate_missing_indicators"))
        ),
    )


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "action_audit": output_root / ACTION_AUDIT_FILE,
        "policy_margin": output_root / POLICY_MARGIN_FILE,
        "candidate_feature": output_root / CANDIDATE_FEATURE_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _pre_checkpoint_path(summary: dict[str, Any], repo_root: Path) -> Path | None:
    base_root = _resolve_optional_path(summary.get("base_candidate_root"), repo_root, repo_root)
    if base_root is None:
        return None
    return base_root / "experimental-hybrid-policy-candidate.pt"


def _pre_metadata_path(summary: dict[str, Any], repo_root: Path) -> Path | None:
    base_root = _resolve_optional_path(summary.get("base_candidate_root"), repo_root, repo_root)
    if base_root is None:
        return None
    return base_root / "experimental-hybrid-policy-candidate-metadata.json"


def _iter_transitions(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for episode in episodes:
        if isinstance(episode.get("transitions"), list):
            transitions.extend(row for row in episode["transitions"] if isinstance(row, dict))
        elif isinstance(episode, dict):
            transitions.append(episode)
    return transitions


def _index_replay_rows(rows: Any) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return index
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in _row_keys(row):
            index.setdefault(key, row)
    return index


def _matching_replay_row(transition: dict[str, Any], replay_index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    info = transition.get("info") if isinstance(transition.get("info"), dict) else {}
    for key in _row_keys(info) + _row_keys(transition):
        if key in replay_index:
            return replay_index[key]
    return None


def _index_candidate_coverage_overlay(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in _candidate_coverage_overlay_keys(row):
            index.setdefault(key, row)
    return index


def _matching_candidate_coverage_overlay(
    *,
    candidate_overlay_index: dict[str, dict[str, Any]],
    context_id: str | None,
    episode_id: Any,
    step_index: int | None,
    scenario_id: Any,
    scenario_family: Any,
    split: Any,
    action_index: int,
    candidate_cell: list[int] | None,
) -> dict[str, Any] | None:
    probe = {
        "context_id": context_id,
        "episode_id": episode_id,
        "step_index": step_index,
        "scenario_id": scenario_id,
        "scenario_family": scenario_family,
        "split": split,
        "action_index": action_index,
        "candidate_cell": candidate_cell,
    }
    for key in _candidate_coverage_overlay_keys(probe):
        if key in candidate_overlay_index:
            return candidate_overlay_index[key]
    return None


def _candidate_coverage_overlay_keys(row: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    action_index = _first_int(row.get("action_index"))
    if action_index is None:
        return keys
    context_id = row.get("context_id")
    episode_id = row.get("episode_id")
    step_index = _first_int(row.get("step_index"))
    if context_id and episode_id is not None and step_index is not None:
        keys.append(f"context-episode-step-action:{context_id}:{episode_id}:{step_index}:{action_index}")
    if context_id:
        keys.append(f"context-action:{context_id}:{action_index}")
    scenario_id = row.get("scenario_id")
    if scenario_id:
        if step_index is not None:
            keys.append(f"scenario-step-action:{scenario_id}:{step_index}:{action_index}")
        keys.append(f"scenario-action:{scenario_id}:{action_index}")
        cell_key = _cell_key(row.get("candidate_cell") or row.get("cell"))
        if cell_key is not None:
            keys.append(f"scenario-cell:{scenario_id}:{cell_key}")
    scenario_family = row.get("scenario_family") or row.get("roi_group") or row.get("scenario_group")
    split = row.get("split")
    if scenario_family and split:
        keys.append(f"family-split-action:{scenario_family}:{split}:{action_index}")
    return keys


def _merge_candidate_coverage_overlay(
    features: dict[str, Any],
    overlay: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(features)
    field_map = {
        "expected_coverage_rate_delta": ("expected_coverage_rate_delta", "candidate_expected_coverage_rate_delta"),
        "expected_new_coverage_area": ("expected_new_coverage_area", "candidate_expected_new_coverage_area"),
        "information_gain": ("information_gain", "candidate_information_gain"),
        "confidence_gain": ("confidence_gain", "candidate_confidence_gain"),
        "value": ("value", "candidate_value"),
        "utility": ("utility", "candidate_utility"),
        "path_cost": ("path_cost",),
        "risk": ("risk",),
        "energy_cost": ("energy_cost", "energy"),
    }
    for feature_name, source_names in field_map.items():
        for source_name in source_names:
            numeric = _float_or_none(overlay.get(source_name))
            if numeric is not None:
                merged[feature_name] = numeric
                break
    return merged


def _feature_map_at(feature_maps: list[dict[str, Any]], action_index: int | None) -> dict[str, Any]:
    if action_index is None or action_index < 0 or action_index >= len(feature_maps):
        return {}
    return feature_maps[action_index]


def _row_keys(row: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    context_id = row.get("context_id")
    if context_id:
        keys.append(f"context:{context_id}")
    episode_id = row.get("episode_id") or row.get("source_episode_id")
    step_index = row.get("step_index") or row.get("source_step_index")
    if episode_id is not None and step_index is not None:
        keys.append(f"episode-step:{episode_id}:{step_index}")
    scenario_id = row.get("scenario_id")
    if scenario_id is not None and step_index is not None:
        keys.append(f"scenario-step:{scenario_id}:{step_index}")
    return keys


def _candidate_feature_map(
    candidate_features: list[list[Any]],
    feature_index: dict[str, int],
    action_index: int | None,
) -> dict[str, float]:
    if action_index is None or action_index < 0 or action_index >= len(candidate_features):
        return {}
    values = candidate_features[action_index]
    result: dict[str, float] = {}
    for name, index in feature_index.items():
        if 0 <= index < len(values):
            result[name] = _float_or_default(values[index], 0.0)
    return result


def _coverage_decision_score(features: dict[str, Any]) -> float:
    return sum(_float_or_default(features.get(field), 0.0) for field in COVERAGE_SIGNAL_FIELDS)


def _not_materially_worse(value: float | None, baseline: float | None) -> bool:
    if value is None or baseline is None:
        return True
    return value <= baseline * (1.0 + MAX_ALLOWED_RELATIVE_COST_REGRESSION) + TOLERANCE


def _controlled_regression_count(
    contexts: list[dict[str, Any]],
    *summaries: dict[str, Any],
) -> int:
    context_count = sum(1 for context in contexts if context["controlled_regression_reason_codes"])
    summary_count = sum(_int(summary.get("controlled_regression_count")) for summary in summaries if isinstance(summary, dict))
    return context_count + summary_count


def _fallback_gain_contamination_count(
    contexts: list[dict[str, Any]],
    *summaries: dict[str, Any],
) -> int:
    context_count = sum(1 for context in contexts if context["fallback_like"])
    summary_count = sum(
        _int(summary.get("fallback_coverage_gain_claimed_as_policy_gain_count"))
        + _int(summary.get("fallback_policy_gain_contamination_count"))
        for summary in summaries
        if isinstance(summary, dict)
    )
    return context_count + summary_count


def _teacher_margin(
    logits: list[float] | None,
    action_mask: list[bool],
    teacher_action_index: int | None,
) -> float | None:
    if not logits or teacher_action_index is None or teacher_action_index < 0 or teacher_action_index >= len(logits):
        return None
    non_teacher = [
        value
        for index, value in enumerate(logits)
        if index != teacher_action_index and _mask_value(action_mask, index) and _finite(value)
    ]
    if not non_teacher:
        return None
    teacher_logit = logits[teacher_action_index]
    if not _finite(teacher_logit):
        return None
    return float(teacher_logit - max(non_teacher))


def _masked_argmax(logits: list[float] | None, action_mask: list[bool]) -> int | None:
    if not logits:
        return None
    best_index: int | None = None
    best_value = -math.inf
    for index, value in enumerate(logits):
        if not _mask_value(action_mask, index) or not _finite(value):
            continue
        if best_index is None or value > best_value:
            best_index = index
            best_value = value
    return best_index


def _stats(values: list[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if _finite(value)]
    if not finite:
        return {"count": 0, "min": None, "max": None, "avg": None}
    return {
        "count": len(finite),
        "min": min(finite),
        "max": max(finite),
        "avg": sum(finite) / len(finite),
    }


def _softmax(logits: list[float]) -> list[float]:
    finite_logits = [value if _finite(value) else -1.0e9 for value in logits]
    if not finite_logits:
        return []
    max_logit = max(finite_logits)
    exps = [math.exp(value - max_logit) for value in finite_logits]
    total = sum(exps)
    if total <= 0:
        return [0.0 for _ in logits]
    return [value / total for value in exps]


def _read_json(path: Path | None, reason_codes: list[str], reason: str) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        _add_reason(reason_codes, reason)
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _add_reason(reason_codes, reason)
        return {}


def _read_jsonl(path: Path | None, reason_codes: list[str], reason: str) -> list[dict[str, Any]]:
    if path is None or not Path(path).is_file():
        _add_reason(reason_codes, reason)
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
    except json.JSONDecodeError:
        _add_reason(reason_codes, reason)
        return []
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_optional_path(value: Any, base: Path, repo_root: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return _resolve_path(Path(str(value)), base, repo_root)


def _resolve_path(path: Path, base: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    base_candidate = base / path
    if base_candidate.exists():
        return base_candidate
    return repo_root / path


def _float_or_default(value: Any, default: float) -> float:
    numeric = _float_or_none(value)
    return default if numeric is None else numeric


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if _finite(numeric) else None


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    return [_float_or_default(item, 0.0) for item in value]


def _bool_list(value: Any) -> list[bool]:
    if not isinstance(value, list):
        return []
    return [bool(item) for item in value]


def _list_of_lists(value: Any) -> list[list[Any]]:
    if not isinstance(value, list):
        return []
    return [list(item) if isinstance(item, (list, tuple)) else [] for item in value]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _int(value: Any) -> int:
    return _first_int(value) or 0


def _max_int(*values: int | None) -> int:
    ints = [value for value in values if value is not None]
    return max(ints) if ints else -1


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _list_value(values: list[float] | None, index: int) -> float | None:
    if not values or index < 0 or index >= len(values):
        return None
    return values[index]


def _mask_value(action_mask: list[bool], index: int) -> bool:
    if not action_mask:
        return True
    return 0 <= index < len(action_mask) and bool(action_mask[index])


def _candidate_cell(candidate_cells: list[list[Any]], action_index: int) -> list[int] | None:
    if action_index < 0 or action_index >= len(candidate_cells):
        return None
    cell = candidate_cells[action_index]
    if len(cell) < 2:
        return None
    try:
        return [int(cell[0]), int(cell[1])]
    except (TypeError, ValueError):
        return None


def _cell_key(value: Any) -> str | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        return f"{int(value[0])},{int(value[1])}"
    except (TypeError, ValueError):
        return None


def _counter_to_str(counter: Counter) -> dict[str, int]:
    return {str(key): value for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
