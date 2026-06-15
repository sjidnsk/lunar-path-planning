from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot


SUMMARY_SCHEMA_VERSION = "exploration-coverage-performance-evaluation-summary/v1"
METRIC_ROW_SCHEMA_VERSION = "coverage-performance-metric-row/v1"
COMPARISON_AUDIT_SCHEMA_VERSION = "coverage-performance-comparison-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "coverage-performance-rejection-report/v1"

FORMAL_SUMMARY_FILE = "formal-ppo-training-run-summary.json"
FORMAL_SEED_SUMMARIES_FILE = "formal-ppo-training-run-seed-summaries.jsonl"
REPLAY_SUMMARY_FILE = "formal-ppo-post-training-stability-replay-summary.json"
SELECTED_SUMMARY_FILE = "selected-formal-ppo-candidate-promotion-preflight-summary.json"
SHADOW_SUMMARY_FILE = "multihorizon-shadow-rollout-summary.json"
SHADOW_STEPS_FILE = "multihorizon-shadow-rollout-steps.jsonl"
COVERAGE_SIGNAL_SUMMARY_FILE = "exploration-coverage-signal-audit-summary.json"
COVERAGE_DELTA_AUDIT_FILE = "coverage-delta-audit.jsonl"

SUMMARY_FILE = "exploration-coverage-performance-evaluation-summary.json"
METRIC_TABLE_FILE = "coverage-performance-metric-table.jsonl"
COMPARISON_AUDIT_FILE = "coverage-performance-comparison-audit.json"
REJECTION_REPORT_FILE = "coverage-performance-rejection-report.json"
REPORT_FILE = "exploration-coverage-performance-evaluation-report.md"

SELECTED_ACTOR = "selected_ppo_candidate"
BASELINE_ACTORS = {"teacher", "source_default", "default_policy"}
FALLBACK_ACTORS = {"source_fallback", "fallback", "teacher_fallback"}
VALID_ACTUAL_GAIN_SOURCES = {"map", "sidecar", "path_feedback"}
DISCOUNT_FACTOR = 0.99
TOLERANCE = 1e-9
MAX_ALLOWED_RELATIVE_REGRESSION = 0.05


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate exploration coverage performance.")
    parser.add_argument("--formal-training-root", required=True)
    parser.add_argument("--post-training-replay-root", required=True)
    parser.add_argument("--selected-candidate-root", required=True)
    parser.add_argument("--shadow-root", required=True)
    parser.add_argument("--coverage-signal-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    summary = run_exploration_coverage_performance_evaluation(
        formal_training_root=_resolve_path(Path(args.formal_training_root), repo_root, repo_root),
        post_training_replay_root=_resolve_path(Path(args.post_training_replay_root), repo_root, repo_root),
        selected_candidate_root=_resolve_path(Path(args.selected_candidate_root), repo_root, repo_root),
        shadow_root=_resolve_path(Path(args.shadow_root), repo_root, repo_root),
        coverage_signal_root=_resolve_path(Path(args.coverage_signal_root), repo_root, repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root, repo_root),
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "coverage_performance_status": summary["coverage_performance_status"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_exploration_coverage_performance_evaluation(
    *,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    shadow_root: Path,
    coverage_signal_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    formal_training_root = Path(formal_training_root)
    post_training_replay_root = Path(post_training_replay_root)
    selected_candidate_root = Path(selected_candidate_root)
    shadow_root = Path(shadow_root)
    coverage_signal_root = Path(coverage_signal_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary": output_root / SUMMARY_FILE,
        "metric_table": output_root / METRIC_TABLE_FILE,
        "comparison_audit": output_root / COMPARISON_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }

    input_reasons: list[str] = []
    formal_summary_path = formal_training_root / FORMAL_SUMMARY_FILE
    seed_summaries_path = formal_training_root / FORMAL_SEED_SUMMARIES_FILE
    replay_summary_path = post_training_replay_root / REPLAY_SUMMARY_FILE
    selected_summary_path = selected_candidate_root / SELECTED_SUMMARY_FILE
    shadow_summary_path = shadow_root / SHADOW_SUMMARY_FILE
    coverage_signal_summary_path = coverage_signal_root / COVERAGE_SIGNAL_SUMMARY_FILE

    formal_summary = _read_json(formal_summary_path, input_reasons, "formal_training_summary")
    seed_summaries = _read_jsonl(seed_summaries_path, input_reasons, "formal_seed_summaries")
    replay_summary = _read_json(replay_summary_path, input_reasons, "post_training_replay_summary")
    selected_summary = _read_json(selected_summary_path, input_reasons, "selected_candidate_summary")
    shadow_summary = _read_json(shadow_summary_path, input_reasons, "shadow_summary")
    coverage_signal_summary = _read_json(
        coverage_signal_summary_path,
        input_reasons,
        "coverage_signal_summary",
    )

    shadow_steps_path = _resolve_optional_path(
        shadow_summary.get("steps"),
        shadow_root,
        repo_root,
    ) or (shadow_root / SHADOW_STEPS_FILE)
    coverage_delta_path = _resolve_optional_path(
        coverage_signal_summary.get("coverage_delta_audit"),
        coverage_signal_root,
        repo_root,
    ) or (coverage_signal_root / COVERAGE_DELTA_AUDIT_FILE)

    shadow_steps = _read_jsonl(shadow_steps_path, input_reasons, "shadow_steps")
    delta_rows = _read_jsonl(coverage_delta_path, input_reasons, "coverage_delta_audit")

    reason_codes = list(input_reasons)
    _validate_upstream_artifacts(
        formal_summary=formal_summary,
        seed_summaries=seed_summaries,
        replay_summary=replay_summary,
        selected_summary=selected_summary,
        shadow_summary=shadow_summary,
        coverage_signal_summary=coverage_signal_summary,
        reason_codes=reason_codes,
    )

    performance_rows = _performance_rows(delta_rows, shadow_steps)
    actor_rows = _actor_rows_with_teacher_equivalence(performance_rows, selected_summary, shadow_summary)
    metric_rows = [_aggregate_actor_metrics(actor, rows) for actor, rows in sorted(actor_rows.items())]
    metrics_by_actor = {row["actor"]: row for row in metric_rows}
    comparison_audit = _comparison_audit(metrics_by_actor, actor_rows)
    rejection_report = _rejection_report(
        reason_codes=reason_codes,
        performance_rows=performance_rows,
        metrics_by_actor=metrics_by_actor,
        comparison_audit=comparison_audit,
        coverage_signal_summary=coverage_signal_summary,
    )
    for reason in rejection_report["reason_codes"]:
        _add_reason(reason_codes, reason)

    coverage_performance_status = "passed" if not reason_codes else "failed"
    next_required_change = _next_required_change(reason_codes)
    selected_metrics = metrics_by_actor.get(SELECTED_ACTOR, _empty_metric_row(SELECTED_ACTOR))
    best_baseline_actor = comparison_audit.get("best_baseline_actor")
    best_baseline_metrics = (
        metrics_by_actor.get(best_baseline_actor, _empty_metric_row(str(best_baseline_actor)))
        if best_baseline_actor
        else _empty_metric_row("none")
    )
    controlled_regression_count = _controlled_regression_count(
        formal_summary,
        replay_summary,
        selected_summary,
        shadow_summary,
        coverage_signal_summary,
        performance_rows,
    )
    status = "passed" if coverage_performance_status == "passed" else "failed"

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": next_required_change,
        "coverage_performance_status": coverage_performance_status,
        "formal_training_root": str(formal_training_root),
        "post_training_replay_root": str(post_training_replay_root),
        "selected_candidate_root": str(selected_candidate_root),
        "shadow_root": str(shadow_root),
        "coverage_signal_root": str(coverage_signal_root),
        "output_root": str(output_root),
        "formal_training_summary": str(formal_summary_path),
        "formal_seed_summaries": str(seed_summaries_path),
        "post_training_replay_summary": str(replay_summary_path),
        "selected_candidate_summary": str(selected_summary_path),
        "shadow_summary": str(shadow_summary_path),
        "shadow_steps": str(shadow_steps_path),
        "coverage_signal_summary": str(coverage_signal_summary_path),
        "coverage_delta_audit": str(coverage_delta_path),
        "summary": str(paths["summary"]),
        "metric_table": str(paths["metric_table"]),
        "comparison_audit": str(paths["comparison_audit"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "selected_actor": SELECTED_ACTOR,
        "selected_seed": selected_summary.get("selected_seed"),
        "selected_budget": selected_summary.get("selected_budget"),
        "selected_metrics": selected_metrics,
        "best_baseline_actor": best_baseline_actor,
        "best_baseline_metrics": best_baseline_metrics,
        "baseline_actors": comparison_audit["baseline_actors"],
        "baseline_metrics": [
            metrics_by_actor[actor] for actor in comparison_audit["baseline_actors"] if actor in metrics_by_actor
        ],
        "comparator_actor_count": len(comparison_audit["baseline_actors"]),
        "metric_actor_count": len(metric_rows),
        "audited_row_count": len(performance_rows),
        "controlled_regression_count": controlled_regression_count,
        "expected_actual_coverage_confusion_count": rejection_report[
            "expected_actual_coverage_confusion_count"
        ],
        "fallback_coverage_gain_claimed_as_policy_gain_count": rejection_report[
            "fallback_coverage_gain_claimed_as_policy_gain_count"
        ],
        "fallback_coverage_gain_total": rejection_report["fallback_coverage_gain_total"],
        "teacher_agreement_rate": selected_metrics.get("teacher_agreement_rate"),
        "accepted_policy_activation_rate": selected_metrics.get("accepted_policy_activation_rate"),
        "fallback_rate": selected_metrics.get("fallback_rate"),
        "coverage_return_improvement": comparison_audit.get("coverage_return_improvement"),
        "cumulative_coverage_rate_delta_improvement": comparison_audit.get(
            "cumulative_coverage_rate_delta_improvement"
        ),
        "valuable_area_covered_improvement": comparison_audit.get(
            "valuable_area_covered_improvement"
        ),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "runs_new_ppo_update": False,
        "performance_claimed": False,
        "formal_release_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }

    _write_jsonl(paths["metric_table"], metric_rows)
    _write_json(paths["comparison_audit"], comparison_audit)
    _write_json(paths["rejection_report"], rejection_report)
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, metric_rows, rejection_report), encoding="utf-8")
    return summary


def _validate_upstream_artifacts(
    *,
    formal_summary: dict[str, Any],
    seed_summaries: list[dict[str, Any]],
    replay_summary: dict[str, Any],
    selected_summary: dict[str, Any],
    shadow_summary: dict[str, Any],
    coverage_signal_summary: dict[str, Any],
    reason_codes: list[str],
) -> None:
    for label, summary in (
        ("formal_training", formal_summary),
        ("post_training_replay", replay_summary),
        ("selected_candidate", selected_summary),
        ("shadow_rollout", shadow_summary),
    ):
        if summary and summary.get("status") != "passed":
            _add_reason(reason_codes, f"{label}_not_passed")
        if summary and _string_list(summary.get("reason_codes")):
            _add_reason(reason_codes, f"{label}_has_reason_codes")
        if _int(summary.get("controlled_regression_count")) > 0:
            _add_reason(reason_codes, "controlled_regression_present")
    if coverage_signal_summary:
        if coverage_signal_summary.get("coverage_signal_status") != "passed":
            _add_reason(reason_codes, "coverage_signal_not_passed")
        if _int(coverage_signal_summary.get("nonzero_actual_coverage_delta_count")) <= 0:
            _add_reason(reason_codes, "insufficient_exploration_coverage_signal")
        if _int(coverage_signal_summary.get("expected_actual_coverage_confusion_count")) > 0:
            _add_reason(reason_codes, "expected_actual_coverage_confusion")
        if _int(coverage_signal_summary.get("fallback_coverage_gain_claimed_as_policy_gain_count")) > 0:
            _add_reason(reason_codes, "fallback_coverage_gain_claimed_as_policy_gain")
        if _int(coverage_signal_summary.get("controlled_regression_count")) > 0:
            _add_reason(reason_codes, "controlled_regression_present")
    if seed_summaries:
        failed_seed_count = sum(1 for row in seed_summaries if row.get("status") != "passed")
        if failed_seed_count:
            _add_reason(reason_codes, "formal_training_seed_not_passed")


def _performance_rows(delta_rows: list[dict[str, Any]], shadow_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shadow_index: dict[str, dict[str, Any]] = {}
    for step in shadow_steps:
        for key in _row_keys(step):
            shadow_index.setdefault(key, step)

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(delta_rows):
        shadow = _matching_shadow(row, shadow_index)
        merged = dict(shadow)
        merged.update(row)
        actor = _normalize_actor(
            merged.get("canonical_coverage_actor")
            or merged.get("coverage_gain_claimed_actor")
            or merged.get("controlled_choice_source")
        )
        raw_claimed_actor = str(merged.get("coverage_gain_claimed_actor") or "")
        claimed_actor = _normalize_actor(raw_claimed_actor or actor)
        delta = _float(merged.get("coverage_rate_delta"))
        expected_delta = _float(merged.get("expected_coverage_rate_delta"))
        path_cost = _first_finite_metric(merged, shadow, ("path_cost", "path_cost_delta"), "path_cost")
        risk = _first_finite_metric(merged, shadow, ("risk", "risk_delta"), "risk")
        energy = _first_finite_metric(merged, shadow, ("energy_cost", "energy_cost_delta", "energy"), "energy_cost")
        value = _first_finite_metric(merged, shadow, ("value",), "value")
        utility = _first_finite_metric(merged, shadow, ("utility",), "utility")
        information_gain = _first_finite_metric(
            merged,
            shadow,
            ("information_gain", "actual_information_gain"),
            "information_gain",
        )
        new_area = _first_finite_metric(
            merged,
            shadow,
            ("new_area_covered", "actual_new_coverage_area", "expected_new_coverage_area"),
            "expected_new_coverage_area",
        )
        if new_area is None and _finite(delta):
            new_area = max(float(delta), 0.0)
        valuable = _first_finite_metric(
            merged,
            shadow,
            ("valuable_area_covered", "actual_valuable_area_covered"),
            "value",
        )
        if valuable is None and _finite(delta):
            multiplier = value if value is not None else utility
            valuable = max(float(delta), 0.0) * (multiplier if multiplier is not None else 0.0)
        coverage_gain_source = str(merged.get("actual_coverage_gain_source") or "missing")
        choice_source = str(merged.get("controlled_choice_source") or "")
        fallback_like = actor in FALLBACK_ACTORS or "fallback" in choice_source
        rows.append(
            {
                "schema_version": "coverage-performance-evaluation-row/v1",
                "performance_index": index,
                "actor": actor,
                "claimed_actor": claimed_actor,
                "raw_claimed_actor": raw_claimed_actor,
                "episode_id": str(merged.get("episode_id") or ""),
                "source_episode_id": str(merged.get("source_episode_id") or merged.get("episode_id") or ""),
                "step_index": _int(merged.get("step_index")),
                "source_step_index": _int(merged.get("source_step_index", merged.get("step_index"))),
                "context_id": merged.get("context_id"),
                "scenario_id": merged.get("scenario_id"),
                "scenario_family": merged.get("scenario_family"),
                "split": merged.get("split"),
                "controlled_choice_source": choice_source,
                "controlled_choice_detail": merged.get("controlled_choice_detail"),
                "controlled_action_index": merged.get("controlled_action_index"),
                "teacher_action_index": merged.get("teacher_action_index"),
                "policy_takes_control": bool(merged.get("policy_takes_control")) or actor == SELECTED_ACTOR,
                "actual_coverage_gain_source": coverage_gain_source,
                "coverage_rate_delta": delta,
                "cumulative_coverage_rate_delta": _float(merged.get("cumulative_coverage_rate_delta")),
                "final_coverage_rate": _float(merged.get("final_coverage_rate")),
                "expected_coverage_rate_delta": expected_delta,
                "path_cost": path_cost,
                "risk": risk,
                "energy_cost": energy,
                "new_area_covered": new_area,
                "valuable_area_covered": valuable,
                "information_gain": information_gain,
                "fallback_like": fallback_like,
                "fallback_gain_claimed_as_policy": fallback_like
                and raw_claimed_actor in {"policy", SELECTED_ACTOR, "selected_policy"},
                "expected_actual_confused": (not _finite(delta))
                and _finite(expected_delta)
                and abs(float(expected_delta)) > TOLERANCE,
                "valid_actual_gain": _finite(delta)
                and coverage_gain_source in VALID_ACTUAL_GAIN_SOURCES,
                "controlled_regression_reason_codes": _string_list(
                    merged.get("controlled_regression_reason_codes")
                ),
            }
        )
    return rows


def _actor_rows_with_teacher_equivalence(
    performance_rows: list[dict[str, Any]],
    selected_summary: dict[str, Any],
    shadow_summary: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in performance_rows:
        grouped[row["actor"]].append(row)
    if "teacher" in grouped or SELECTED_ACTOR not in grouped:
        return grouped
    selected_rows = grouped[SELECTED_ACTOR]
    if not selected_rows:
        return grouped
    upstream_teacher_agreement = _float(
        selected_summary.get("teacher_agreement_rate", shadow_summary.get("teacher_agreement_rate"))
    )
    all_policy_teacher_aligned = all(
        row.get("controlled_choice_detail") == "policy_teacher_aligned"
        and _same_action(row.get("controlled_action_index"), row.get("teacher_action_index"))
        for row in selected_rows
    )
    if upstream_teacher_agreement == 1.0 and all_policy_teacher_aligned:
        grouped["teacher"] = [
            dict(
                row,
                actor="teacher",
                policy_takes_control=False,
                comparator_basis="teacher_action_equivalent_from_policy_teacher_aligned_shadow",
            )
            for row in selected_rows
        ]
    return grouped


def _aggregate_actor_metrics(actor: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_count = len(rows)
    valid_rows = [row for row in rows if _finite(row.get("coverage_rate_delta"))]
    total_gain = sum(float(row["coverage_rate_delta"]) for row in valid_rows)
    coverage_return = sum(
        float(row["coverage_rate_delta"]) * (DISCOUNT_FACTOR ** max(_int(row.get("step_index")), 0))
        for row in valid_rows
    )
    path_cost_total = _sum_metric(rows, "path_cost")
    risk_total = _sum_metric(rows, "risk")
    energy_total = _sum_metric(rows, "energy_cost")
    new_area_covered = _sum_metric(rows, "new_area_covered")
    valuable_area_covered = _sum_metric(rows, "valuable_area_covered")
    information_gain = _sum_metric(rows, "information_gain")
    final_rates = _last_final_rates(rows)
    teacher_action_rows = [
        row for row in rows if row.get("teacher_action_index") is not None and row.get("controlled_action_index") is not None
    ]
    teacher_agreement_rate = (
        sum(1 for row in teacher_action_rows if _same_action(row["controlled_action_index"], row["teacher_action_index"]))
        / len(teacher_action_rows)
        if teacher_action_rows
        else None
    )
    policy_activation_count = sum(1 for row in rows if row.get("policy_takes_control"))
    fallback_count = sum(1 for row in rows if row.get("fallback_like"))
    comparator_basis = sorted(
        {
            str(row.get("comparator_basis"))
            for row in rows
            if row.get("comparator_basis")
        }
    )
    return {
        "schema_version": METRIC_ROW_SCHEMA_VERSION,
        "actor": actor,
        "row_count": row_count,
        "episode_count": len({row.get("episode_id") for row in rows if row.get("episode_id")}),
        "actual_coverage_row_count": len(valid_rows),
        "coverage_return": round(coverage_return, 12),
        "cumulative_coverage_rate_delta": round(total_gain, 12),
        "final_coverage_rate": _mean(final_rates),
        "max_final_coverage_rate": max(final_rates) if final_rates else None,
        "new_area_covered": round(new_area_covered, 12),
        "valuable_area_covered": round(valuable_area_covered, 12),
        "information_gain": round(information_gain, 12),
        "path_cost": round(path_cost_total, 12),
        "risk": round(risk_total, 12),
        "energy_cost": round(energy_total, 12),
        "coverage_gain_per_path_cost": _safe_ratio(total_gain, path_cost_total),
        "coverage_gain_per_risk": _safe_ratio(total_gain, risk_total),
        "coverage_gain_per_energy": _safe_ratio(total_gain, energy_total),
        "accepted_policy_activation_rate": _safe_ratio(policy_activation_count, row_count),
        "fallback_rate": _safe_ratio(fallback_count, row_count),
        "teacher_agreement_rate": teacher_agreement_rate,
        "controlled_regression_count": sum(
            1 for row in rows if _string_list(row.get("controlled_regression_reason_codes"))
        ),
        "fallback_coverage_gain": round(
            sum(float(row["coverage_rate_delta"]) for row in valid_rows if row.get("fallback_like")),
            12,
        ),
        "comparator_basis": comparator_basis,
    }


def _comparison_audit(
    metrics_by_actor: dict[str, dict[str, Any]],
    actor_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    baseline_actors = sorted(actor for actor in metrics_by_actor if actor in BASELINE_ACTORS)
    best_baseline_actor = None
    if baseline_actors:
        best_baseline_actor = max(
            baseline_actors,
            key=lambda actor: _float_or_default(metrics_by_actor[actor].get("coverage_return"), -math.inf),
        )
    selected = metrics_by_actor.get(SELECTED_ACTOR, _empty_metric_row(SELECTED_ACTOR))
    baseline = (
        metrics_by_actor.get(best_baseline_actor, _empty_metric_row(str(best_baseline_actor)))
        if best_baseline_actor
        else _empty_metric_row("none")
    )
    return {
        "schema_version": COMPARISON_AUDIT_SCHEMA_VERSION,
        "selected_actor": SELECTED_ACTOR,
        "selected_actor_present": SELECTED_ACTOR in metrics_by_actor,
        "baseline_actors": baseline_actors,
        "baseline_actor_count": len(baseline_actors),
        "best_baseline_actor": best_baseline_actor,
        "best_baseline_selection_rule": "max_coverage_return",
        "available_actors": sorted(metrics_by_actor),
        "actor_row_counts": {actor: len(rows) for actor, rows in sorted(actor_rows.items())},
        "coverage_return_improvement": _metric_improvement(selected, baseline, "coverage_return"),
        "cumulative_coverage_rate_delta_improvement": _metric_improvement(
            selected,
            baseline,
            "cumulative_coverage_rate_delta",
        ),
        "valuable_area_covered_improvement": _metric_improvement(
            selected,
            baseline,
            "valuable_area_covered",
        ),
        "path_cost_delta_vs_baseline": _metric_delta(selected, baseline, "path_cost"),
        "risk_delta_vs_baseline": _metric_delta(selected, baseline, "risk"),
        "energy_cost_delta_vs_baseline": _metric_delta(selected, baseline, "energy_cost"),
        "selected_teacher_equivalent": "teacher" in metrics_by_actor
        and "teacher_action_equivalent_from_policy_teacher_aligned_shadow"
        in metrics_by_actor["teacher"].get("comparator_basis", []),
    }


def _rejection_report(
    *,
    reason_codes: list[str],
    performance_rows: list[dict[str, Any]],
    metrics_by_actor: dict[str, dict[str, Any]],
    comparison_audit: dict[str, Any],
    coverage_signal_summary: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    selected = metrics_by_actor.get(SELECTED_ACTOR)
    baseline_actor = comparison_audit.get("best_baseline_actor")
    baseline = metrics_by_actor.get(baseline_actor) if baseline_actor else None

    expected_actual_confusion_count = _int(
        coverage_signal_summary.get("expected_actual_coverage_confusion_count")
    ) + sum(1 for row in performance_rows if row.get("expected_actual_confused"))
    fallback_claimed_policy_count = _int(
        coverage_signal_summary.get("fallback_coverage_gain_claimed_as_policy_gain_count")
    ) + sum(1 for row in performance_rows if row.get("fallback_gain_claimed_as_policy"))
    controlled_regression_row_count = sum(
        1 for row in performance_rows if _string_list(row.get("controlled_regression_reason_codes"))
    )
    fallback_gain_total = sum(
        float(row["coverage_rate_delta"])
        for row in performance_rows
        if row.get("fallback_like") and _finite(row.get("coverage_rate_delta"))
    )

    if not selected:
        _add_reason(reasons, "selected_ppo_candidate_evidence_missing")
    if not baseline:
        _add_reason(reasons, "insufficient_comparator_evidence")
    if expected_actual_confusion_count > 0:
        _add_reason(reasons, "expected_actual_coverage_confusion")
    if fallback_claimed_policy_count > 0:
        _add_reason(reasons, "fallback_coverage_gain_claimed_as_policy_gain")
    if controlled_regression_row_count > 0:
        _add_reason(reasons, "controlled_regression_present")
    if selected and fallback_gain_total > _float_or_default(selected.get("cumulative_coverage_rate_delta"), 0.0):
        _add_reason(reasons, "fallback_dominates_gain")
    if selected and baseline:
        _validate_improvement(selected, baseline, reasons)
        _validate_cost_risk_fallback(selected, baseline, reasons)

    rejection_reason_counts = Counter(reason_codes)
    rejection_reason_counts.update(reasons)
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "reason_codes": reasons,
        "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "expected_actual_coverage_confusion_count": expected_actual_confusion_count,
        "fallback_coverage_gain_claimed_as_policy_gain_count": fallback_claimed_policy_count,
        "fallback_coverage_gain_total": round(fallback_gain_total, 12),
        "controlled_regression_row_count": controlled_regression_row_count,
        "selected_actor_present": bool(selected),
        "best_baseline_actor": baseline_actor,
    }


def _validate_improvement(
    selected: dict[str, Any],
    baseline: dict[str, Any],
    reasons: list[str],
) -> None:
    coverage_return_improved = _strictly_greater(selected, baseline, "coverage_return")
    cumulative_improved = _strictly_greater(selected, baseline, "cumulative_coverage_rate_delta")
    valuable_improved = _strictly_greater(selected, baseline, "valuable_area_covered")
    if not coverage_return_improved or not cumulative_improved:
        _add_reason(reasons, "coverage_performance_not_improved")
    if not valuable_improved:
        _add_reason(reasons, "valuable_coverage_not_improved")
    efficiency_metrics = (
        "coverage_gain_per_path_cost",
        "coverage_gain_per_risk",
        "coverage_gain_per_energy",
    )
    for metric in efficiency_metrics:
        selected_value = selected.get(metric)
        baseline_value = baseline.get(metric)
        if selected_value is None or baseline_value is None:
            continue
        if float(selected_value) + TOLERANCE < float(baseline_value) * (1.0 - MAX_ALLOWED_RELATIVE_REGRESSION):
            _add_reason(reasons, "coverage_efficiency_regression")


def _validate_cost_risk_fallback(
    selected: dict[str, Any],
    baseline: dict[str, Any],
    reasons: list[str],
) -> None:
    for metric, reason in (
        ("path_cost", "path_cost_regression"),
        ("risk", "risk_regression"),
        ("energy_cost", "energy_cost_regression"),
    ):
        selected_value = _float_or_default(selected.get(metric), 0.0)
        baseline_value = _float_or_default(baseline.get(metric), 0.0)
        if selected_value > baseline_value * (1.0 + MAX_ALLOWED_RELATIVE_REGRESSION) + TOLERANCE:
            _add_reason(reasons, reason)
    selected_fallback_rate = _float_or_default(selected.get("fallback_rate"), 0.0)
    baseline_fallback_rate = _float_or_default(baseline.get("fallback_rate"), 0.0)
    if selected_fallback_rate > max(0.05, baseline_fallback_rate + 0.05):
        _add_reason(reasons, "fallback_rate_regression")


def _controlled_regression_count(*items: Any) -> int:
    counts: list[int] = []
    for item in items:
        if isinstance(item, dict):
            counts.append(_int(item.get("controlled_regression_count")))
        elif isinstance(item, list):
            counts.append(
                sum(
                    1
                    for row in item
                    if isinstance(row, dict)
                    and _string_list(row.get("controlled_regression_reason_codes"))
                )
            )
    return max(counts) if counts else 0


def _next_required_change(reason_codes: list[str]) -> str:
    if not reason_codes:
        return "coverage_aware_reward_refinement"
    if "insufficient_comparator_evidence" in reason_codes:
        return "collect_comparator_coverage_evidence"
    if "expected_actual_coverage_confusion" in reason_codes or "fallback_coverage_gain_claimed_as_policy_gain" in reason_codes:
        return "fix_coverage_performance_evaluation_inputs"
    return "coverage_aware_reward_refinement"


def _matching_shadow(row: dict[str, Any], shadow_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for key in _row_keys(row):
        if key in shadow_index:
            return shadow_index[key]
    return {}


def _row_keys(row: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    context_id = row.get("context_id")
    if context_id:
        keys.append(f"context:{context_id}")
    for episode_field, step_field in (
        ("episode_id", "step_index"),
        ("source_episode_id", "source_step_index"),
        ("source_episode_id", "step_index"),
        ("episode_id", "source_step_index"),
    ):
        episode_id = row.get(episode_field)
        step_index = row.get(step_field)
        if episode_id is not None and step_index is not None:
            keys.append(f"episode-step:{episode_id}:{step_index}")
    return keys


def _first_finite_metric(
    row: dict[str, Any],
    shadow: dict[str, Any],
    fields: tuple[str, ...],
    feature_name: str,
) -> float | None:
    feature_value = _candidate_feature_value(row.get("observation"), row.get("controlled_action_index"), feature_name)
    if not _finite(feature_value):
        feature_value = _candidate_feature_value(
            shadow.get("observation"),
            shadow.get("controlled_action_index"),
            feature_name,
        )
    for source in (row, shadow):
        for field in fields:
            value = _float(source.get(field))
            if _finite(value):
                if (
                    feature_name in {"path_cost", "risk", "energy_cost"}
                    and abs(float(value)) <= TOLERANCE
                    and _finite(feature_value)
                    and abs(float(feature_value)) > TOLERANCE
                ):
                    return float(feature_value)
                return float(value)
    if _finite(feature_value):
        return float(feature_value)
    return None


def _candidate_feature_value(
    observation: Any,
    action_index_value: Any,
    feature_name: str,
) -> float | None:
    if not isinstance(observation, dict):
        return None
    names = observation.get("candidate_feature_names")
    features = observation.get("candidate_features")
    if not isinstance(names, list) or not isinstance(features, list) or feature_name not in names:
        return None
    action_index = _int(action_index_value)
    if action_index < 0 or action_index >= len(features):
        return None
    action_features = features[action_index]
    if not isinstance(action_features, list):
        return None
    feature_index = names.index(feature_name)
    if feature_index >= len(action_features):
        return None
    return _float(action_features[feature_index])


def _normalize_actor(value: Any) -> str:
    actor = str(value or "").strip()
    if actor in {"policy", "selected_policy", "ppo", "ppo_candidate"}:
        return SELECTED_ACTOR
    if actor in {"source", "source_default_policy", "default", "default_source"}:
        return "source_default"
    if actor in {"source_fallback", "fallback"}:
        return "source_fallback"
    if actor in {"teacher", "teacher_fallback"}:
        return "teacher" if actor == "teacher" else "teacher_fallback"
    return actor or "unknown"


def _last_final_rates(rows: list[dict[str, Any]]) -> list[float]:
    by_episode: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not _finite(row.get("final_coverage_rate")):
            continue
        episode_id = str(row.get("episode_id") or "")
        current = by_episode.get(episode_id)
        if current is None or _int(row.get("step_index")) >= _int(current.get("step_index")):
            by_episode[episode_id] = row
    return [float(row["final_coverage_rate"]) for row in by_episode.values()]


def _sum_metric(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows if _finite(row.get(field)))


def _strictly_greater(selected: dict[str, Any], baseline: dict[str, Any], field: str) -> bool:
    return _float_or_default(selected.get(field), -math.inf) > _float_or_default(baseline.get(field), math.inf) + TOLERANCE


def _metric_improvement(selected: dict[str, Any], baseline: dict[str, Any], field: str) -> float | None:
    selected_value = selected.get(field)
    baseline_value = baseline.get(field)
    if selected_value is None or baseline_value is None:
        return None
    return round(float(selected_value) - float(baseline_value), 12)


def _metric_delta(selected: dict[str, Any], baseline: dict[str, Any], field: str) -> float | None:
    return _metric_improvement(selected, baseline, field)


def _empty_metric_row(actor: str) -> dict[str, Any]:
    return {
        "schema_version": METRIC_ROW_SCHEMA_VERSION,
        "actor": actor,
        "row_count": 0,
        "episode_count": 0,
        "actual_coverage_row_count": 0,
        "coverage_return": 0.0,
        "cumulative_coverage_rate_delta": 0.0,
        "final_coverage_rate": None,
        "max_final_coverage_rate": None,
        "new_area_covered": 0.0,
        "valuable_area_covered": 0.0,
        "information_gain": 0.0,
        "path_cost": 0.0,
        "risk": 0.0,
        "energy_cost": 0.0,
        "coverage_gain_per_path_cost": None,
        "coverage_gain_per_risk": None,
        "coverage_gain_per_energy": None,
        "accepted_policy_activation_rate": None,
        "fallback_rate": None,
        "teacher_agreement_rate": None,
        "controlled_regression_count": 0,
        "fallback_coverage_gain": 0.0,
        "comparator_basis": [],
    }


def _read_json(path: Path, reason_codes: list[str], label: str) -> dict[str, Any]:
    if not path.exists():
        _add_reason(reason_codes, f"{label}_missing")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _add_reason(reason_codes, f"{label}_invalid_json")
        return {}


def _read_jsonl(path: Path, reason_codes: list[str], label: str) -> list[dict[str, Any]]:
    if not path.exists():
        _add_reason(reason_codes, f"{label}_missing")
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except json.JSONDecodeError:
        _add_reason(reason_codes, f"{label}_invalid_jsonl")
        return []
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_optional_path(value: Any, base: Path, repo_root: Path) -> Path | None:
    if not value:
        return None
    return _resolve_path(Path(str(value)), base, repo_root)


def _resolve_path(path: Path, base: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    if (base / path).exists():
        return base / path
    return repo_root / path


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _float_or_default(value: Any, default: float) -> float:
    parsed = _float(value)
    if _finite(parsed):
        return float(parsed)
    return default


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if not _finite(numerator) or not _finite(denominator):
        return None
    denominator_float = float(denominator)
    if abs(denominator_float) <= TOLERANCE:
        return None
    return round(float(numerator) / denominator_float, 12)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 12)


def _same_action(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return _int(left) == _int(right)


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _render_report(
    summary: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    rejection_report: dict[str, Any],
) -> str:
    lines = [
        "# Exploration Coverage Performance Evaluation v1",
        "",
        f"Status: `{summary['status']}`",
        f"Coverage performance status: `{summary['coverage_performance_status']}`",
        f"Reason codes: `{summary['reason_codes']}`",
        f"Next required change: `{summary['next_required_change']}`",
        "",
        "## Selected Candidate",
        "",
        f"- Selected seed: `{summary.get('selected_seed')}`",
        f"- Selected budget: `{summary.get('selected_budget')}`",
        f"- Coverage return: `{summary['selected_metrics'].get('coverage_return')}`",
        f"- Cumulative coverage delta: `{summary['selected_metrics'].get('cumulative_coverage_rate_delta')}`",
        f"- Valuable area covered: `{summary['selected_metrics'].get('valuable_area_covered')}`",
        f"- Coverage gain per path cost: `{summary['selected_metrics'].get('coverage_gain_per_path_cost')}`",
        f"- Coverage gain per risk: `{summary['selected_metrics'].get('coverage_gain_per_risk')}`",
        f"- Fallback rate: `{summary.get('fallback_rate')}`",
        "",
        "## Best Baseline",
        "",
        f"- Actor: `{summary.get('best_baseline_actor')}`",
        f"- Coverage return: `{summary['best_baseline_metrics'].get('coverage_return')}`",
        f"- Cumulative coverage delta: `{summary['best_baseline_metrics'].get('cumulative_coverage_rate_delta')}`",
        "",
        "## Actor Metrics",
        "",
    ]
    for row in metric_rows:
        lines.append(
            "- `{actor}`: coverage_return=`{coverage_return}`, cumulative_delta=`{cumulative}`, "
            "path_cost=`{path_cost}`, risk=`{risk}`, fallback_rate=`{fallback_rate}`".format(
                actor=row["actor"],
                coverage_return=row["coverage_return"],
                cumulative=row["cumulative_coverage_rate_delta"],
                path_cost=row["path_cost"],
                risk=row["risk"],
                fallback_rate=row["fallback_rate"],
            )
        )
    lines.extend(["", "## Rejection Reason Counts", ""])
    if rejection_report["rejection_reason_counts"]:
        for reason, count in rejection_report["rejection_reason_counts"].items():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This evaluation is read-only. It does not run a new PPO update, publish checkpoints, "
            "replace the default policy, connect a real executor, or make a formal release claim.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
