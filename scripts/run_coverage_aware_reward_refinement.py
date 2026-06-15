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


SUMMARY_SCHEMA_VERSION = "coverage-aware-reward-refinement-summary/v1"
COMPONENT_AUDIT_ROW_SCHEMA_VERSION = "coverage-aware-reward-component-audit-row/v1"
SOURCE_AUDIT_SCHEMA_VERSION = "coverage-aware-reward-source-field-audit/v1"
RESCORE_COMPARISON_SCHEMA_VERSION = "coverage-aware-reward-rescore-comparison/v1"
REJECTION_REPORT_SCHEMA_VERSION = "coverage-aware-reward-refinement-rejection-report/v1"

FORMAL_SUMMARY_FILE = "formal-ppo-training-run-summary.json"
FORMAL_SEED_SUMMARIES_FILE = "formal-ppo-training-run-seed-summaries.jsonl"
REPLAY_SUMMARY_FILE = "formal-ppo-post-training-stability-replay-summary.json"
SELECTED_SUMMARY_FILE = "selected-formal-ppo-candidate-promotion-preflight-summary.json"
COVERAGE_SIGNAL_SUMMARY_FILE = "exploration-coverage-signal-audit-summary.json"
COVERAGE_DELTA_AUDIT_FILE = "coverage-delta-audit.jsonl"
COVERAGE_PERFORMANCE_SUMMARY_FILE = "exploration-coverage-performance-evaluation-summary.json"
PERFORMANCE_METRIC_TABLE_FILE = "coverage-performance-metric-table.jsonl"
PERFORMANCE_COMPARISON_AUDIT_FILE = "coverage-performance-comparison-audit.json"
SHADOW_STEPS_FILE = "multihorizon-shadow-rollout-steps.jsonl"
CONNECT_SOURCE_SUMMARY_FILE = "connect-reward-component-source-fields-summary.json"
CONNECT_SOURCE_PROVENANCE_FILE = "component-provenance.jsonl"
DEFAULT_CONNECT_SOURCE_ROOT = Path("outputs/path_feedback_batch_connect_reward_component_source_fields_v1")

SUMMARY_FILE = "coverage-aware-reward-refinement-summary.json"
REWARD_COMPONENT_AUDIT_FILE = "reward-component-audit.jsonl"
SOURCE_FIELD_AUDIT_FILE = "source-field-audit.json"
RESCORE_COMPARISON_FILE = "reward-rescore-comparison.json"
REJECTION_REPORT_FILE = "reward-refinement-rejection-report.json"
REPORT_FILE = "coverage-aware-reward-refinement-report.md"

SELECTED_ACTOR = "selected_ppo_candidate"
BASELINE_ACTORS = {"teacher", "source_default", "default_policy"}
FALLBACK_ACTORS = {"source_fallback", "fallback", "teacher_fallback"}
VALID_ACTUAL_GAIN_SOURCES = {"map", "sidecar", "path_feedback"}
TOLERANCE = 1e-9

REWARD_WEIGHTS = {
    "coverage_gain_bonus": 1.0,
    "valuable_area_bonus": 1.0,
    "information_gain_bonus": 1.0,
    "teacher_skill_retention_bonus": 0.1,
    "safe_disagreement_retention_bonus": 0.05,
    "path_cost_penalty": 0.01,
    "risk_penalty": 0.05,
    "energy_penalty": 0.005,
    "fallback_penalty": 1.0,
    "controlled_regression_penalty": 2.0,
}

REWARD_COMPONENTS = (
    "coverage_gain_bonus",
    "valuable_area_bonus",
    "information_gain_bonus",
    "teacher_skill_retention_bonus",
    "path_cost_penalty",
    "risk_penalty",
    "energy_penalty",
    "fallback_penalty",
    "controlled_regression_penalty",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit and re-score coverage-aware reward contract.")
    parser.add_argument("--formal-training-root", required=True)
    parser.add_argument("--post-training-replay-root")
    parser.add_argument("--selected-candidate-root", required=True)
    parser.add_argument("--coverage-signal-root", required=True)
    parser.add_argument("--coverage-performance-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reward-component-source-root")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    default_replay_root = repo_root / "outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1"
    post_training_replay_root = (
        _resolve_path(Path(args.post_training_replay_root), repo_root, repo_root)
        if args.post_training_replay_root
        else default_replay_root
    )
    summary = run_coverage_aware_reward_refinement(
        formal_training_root=_resolve_path(Path(args.formal_training_root), repo_root, repo_root),
        post_training_replay_root=post_training_replay_root,
        selected_candidate_root=_resolve_path(Path(args.selected_candidate_root), repo_root, repo_root),
        coverage_signal_root=_resolve_path(Path(args.coverage_signal_root), repo_root, repo_root),
        coverage_performance_root=_resolve_path(Path(args.coverage_performance_root), repo_root, repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root, repo_root),
        repo_root=repo_root,
        reward_component_source_root=_resolve_path(Path(args.reward_component_source_root), repo_root, repo_root)
        if args.reward_component_source_root
        else None,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "reward_refinement_status": summary["reward_refinement_status"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_coverage_aware_reward_refinement(
    *,
    formal_training_root: Path,
    post_training_replay_root: Path,
    selected_candidate_root: Path,
    coverage_signal_root: Path,
    coverage_performance_root: Path,
    output_root: Path,
    repo_root: Path,
    reward_component_source_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    formal_training_root = Path(formal_training_root)
    post_training_replay_root = Path(post_training_replay_root)
    selected_candidate_root = Path(selected_candidate_root)
    coverage_signal_root = Path(coverage_signal_root)
    coverage_performance_root = Path(coverage_performance_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary": output_root / SUMMARY_FILE,
        "reward_component_audit": output_root / REWARD_COMPONENT_AUDIT_FILE,
        "source_field_audit": output_root / SOURCE_FIELD_AUDIT_FILE,
        "rescore_comparison": output_root / RESCORE_COMPARISON_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }

    input_reasons: list[str] = []
    formal_summary_path = formal_training_root / FORMAL_SUMMARY_FILE
    seed_summaries_path = formal_training_root / FORMAL_SEED_SUMMARIES_FILE
    replay_summary_path = post_training_replay_root / REPLAY_SUMMARY_FILE
    selected_summary_path = selected_candidate_root / SELECTED_SUMMARY_FILE
    coverage_signal_summary_path = coverage_signal_root / COVERAGE_SIGNAL_SUMMARY_FILE
    performance_summary_path = coverage_performance_root / COVERAGE_PERFORMANCE_SUMMARY_FILE

    formal_summary = _read_json(formal_summary_path, input_reasons, "formal_training_summary")
    seed_summaries = _read_jsonl(seed_summaries_path, input_reasons, "formal_seed_summaries")
    replay_summary = _read_json(replay_summary_path, input_reasons, "post_training_replay_summary")
    selected_summary = _read_json(selected_summary_path, input_reasons, "selected_candidate_summary")
    coverage_signal_summary = _read_json(
        coverage_signal_summary_path,
        input_reasons,
        "coverage_signal_summary",
    )
    performance_summary = _read_json(
        performance_summary_path,
        input_reasons,
        "coverage_performance_summary",
    )

    delta_path = _resolve_optional_path(
        coverage_signal_summary.get("coverage_delta_audit"),
        coverage_signal_root,
        repo_root,
    ) or (coverage_signal_root / COVERAGE_DELTA_AUDIT_FILE)
    shadow_steps_path = _resolve_optional_path(
        selected_summary.get("multihorizon_steps"),
        selected_candidate_root,
        repo_root,
    ) or _resolve_optional_path(
        performance_summary.get("shadow_steps"),
        coverage_performance_root,
        repo_root,
    ) or (selected_candidate_root / SHADOW_STEPS_FILE)
    metric_table_path = _resolve_optional_path(
        performance_summary.get("metric_table"),
        coverage_performance_root,
        repo_root,
    ) or (coverage_performance_root / PERFORMANCE_METRIC_TABLE_FILE)
    comparison_audit_path = _resolve_optional_path(
        performance_summary.get("comparison_audit"),
        coverage_performance_root,
        repo_root,
    ) or (coverage_performance_root / PERFORMANCE_COMPARISON_AUDIT_FILE)

    delta_rows = _read_jsonl(delta_path, input_reasons, "coverage_delta_audit")
    shadow_steps = _read_jsonl(shadow_steps_path, input_reasons, "shadow_steps")
    metric_rows = _read_jsonl(metric_table_path, input_reasons, "coverage_performance_metric_table")
    performance_comparison = _read_json(
        comparison_audit_path,
        input_reasons,
        "coverage_performance_comparison_audit",
    )
    collector_reward_audit = _collector_reward_audit(seed_summaries, repo_root, input_reasons)
    component_source_overlay = _load_component_source_overlay(
        reward_component_source_root=reward_component_source_root,
        repo_root=repo_root,
        coverage_signal_root=coverage_signal_root,
        coverage_performance_root=coverage_performance_root,
        reason_codes=input_reasons,
    )

    reason_codes = list(input_reasons)
    _validate_upstream_artifacts(
        formal_summary=formal_summary,
        replay_summary=replay_summary,
        selected_summary=selected_summary,
        coverage_signal_summary=coverage_signal_summary,
        reason_codes=reason_codes,
    )

    component_rows = _reward_component_rows(delta_rows, shadow_steps, component_source_overlay)
    source_field_audit = _source_field_audit(component_rows, collector_reward_audit)
    rescore_comparison = _rescore_comparison(
        component_rows=component_rows,
        performance_summary=performance_summary,
        performance_comparison=performance_comparison,
        metric_rows=metric_rows,
    )
    rejection_report = _rejection_report(
        reason_codes=reason_codes,
        source_field_audit=source_field_audit,
        component_rows=component_rows,
        coverage_signal_summary=coverage_signal_summary,
        rescore_comparison=rescore_comparison,
    )
    for reason in rejection_report["reason_codes"]:
        _add_reason(reason_codes, reason)

    reward_refinement_status = "passed" if not reason_codes else "failed"
    status = "passed" if reward_refinement_status == "passed" else "failed"
    next_required_change = _next_required_change(reason_codes)
    selected_actor_rescore = rescore_comparison["actor_rescores"].get(
        SELECTED_ACTOR,
        _empty_actor_rescore(SELECTED_ACTOR),
    )
    best_baseline_actor = rescore_comparison.get("best_baseline_actor")
    best_baseline_rescore = (
        rescore_comparison["actor_rescores"].get(
            best_baseline_actor,
            _empty_actor_rescore(str(best_baseline_actor)),
        )
        if best_baseline_actor
        else _empty_actor_rescore("none")
    )

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": next_required_change,
        "reward_refinement_status": reward_refinement_status,
        "formal_training_root": str(formal_training_root),
        "post_training_replay_root": str(post_training_replay_root),
        "selected_candidate_root": str(selected_candidate_root),
        "coverage_signal_root": str(coverage_signal_root),
        "coverage_performance_root": str(coverage_performance_root),
        "output_root": str(output_root),
        "reward_component_source_root": str(
            reward_component_source_root
            or (repo_root / DEFAULT_CONNECT_SOURCE_ROOT)
        ),
        "formal_training_summary": str(formal_summary_path),
        "formal_seed_summaries": str(seed_summaries_path),
        "post_training_replay_summary": str(replay_summary_path),
        "selected_candidate_summary": str(selected_summary_path),
        "coverage_signal_summary": str(coverage_signal_summary_path),
        "coverage_delta_audit": str(delta_path),
        "shadow_steps": str(shadow_steps_path),
        "coverage_performance_summary": str(performance_summary_path),
        "coverage_performance_metric_table": str(metric_table_path),
        "coverage_performance_comparison_audit": str(comparison_audit_path),
        "summary": str(paths["summary"]),
        "reward_component_audit": str(paths["reward_component_audit"]),
        "source_field_audit": str(paths["source_field_audit"]),
        "reward_rescore_comparison": str(paths["rescore_comparison"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "reward_contract": _reward_contract_payload(),
        "reward_component_source_status": source_field_audit["component_status"],
        "existing_reward_audit": collector_reward_audit,
        "selected_actor": SELECTED_ACTOR,
        "selected_seed": selected_summary.get("selected_seed"),
        "selected_budget": selected_summary.get("selected_budget"),
        "selected_actor_rescore": selected_actor_rescore,
        "best_baseline_actor": best_baseline_actor,
        "best_baseline_rescore": best_baseline_rescore,
        "coverage_aware_reward_improvement": rescore_comparison.get(
            "coverage_aware_reward_improvement"
        ),
        "selected_teacher_equivalent": rescore_comparison["selected_teacher_equivalent"],
        "selected_candidate_performance_improved": rescore_comparison[
            "selected_candidate_performance_improved"
        ],
        "previous_coverage_performance_status": performance_summary.get("coverage_performance_status"),
        "previous_coverage_performance_reason_codes": _string_list(performance_summary.get("reason_codes")),
        "expected_actual_coverage_confusion_count": rejection_report[
            "expected_actual_coverage_confusion_count"
        ],
        "fallback_coverage_gain_claimed_as_policy_gain_count": rejection_report[
            "fallback_coverage_gain_claimed_as_policy_gain_count"
        ],
        "controlled_regression_count": rejection_report["controlled_regression_count"],
        "audited_row_count": len(component_rows),
        "source_field_missing_component_count": source_field_audit[
            "missing_required_component_source_count"
        ],
        "component_source_overlay_row_count": len(component_source_overlay),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "runs_new_ppo_update": False,
        "performance_claimed": False,
        "formal_release_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }

    _write_jsonl(paths["reward_component_audit"], component_rows)
    _write_json(paths["source_field_audit"], source_field_audit)
    _write_json(paths["rescore_comparison"], rescore_comparison)
    _write_json(paths["rejection_report"], rejection_report)
    _write_json(paths["summary"], summary)
    paths["report"].write_text(
        _render_report(summary, source_field_audit, rescore_comparison, rejection_report),
        encoding="utf-8",
    )
    return summary


def _validate_upstream_artifacts(
    *,
    formal_summary: dict[str, Any],
    replay_summary: dict[str, Any],
    selected_summary: dict[str, Any],
    coverage_signal_summary: dict[str, Any],
    reason_codes: list[str],
) -> None:
    for label, summary in (
        ("formal_training", formal_summary),
        ("post_training_replay", replay_summary),
        ("selected_candidate", selected_summary),
    ):
        if summary and summary.get("status") != "passed":
            _add_reason(reason_codes, f"{label}_not_passed")
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
            _add_reason(reason_codes, "fallback_policy_gain_contamination")
        if _int(coverage_signal_summary.get("controlled_regression_count")) > 0:
            _add_reason(reason_codes, "controlled_regression_present")


def _reward_component_rows(
    delta_rows: list[dict[str, Any]],
    shadow_steps: list[dict[str, Any]],
    component_source_overlay: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    shadow_index: dict[str, dict[str, Any]] = {}
    for step in shadow_steps:
        for key in _row_keys(step):
            shadow_index.setdefault(key, step)
    overlay_index: dict[str, dict[str, Any]] = {}
    for overlay in component_source_overlay or []:
        for key in _row_keys(overlay):
            overlay_index.setdefault(key, overlay)

    rows: list[dict[str, Any]] = []
    for index, delta_row in enumerate(delta_rows):
        shadow = _matching_shadow(delta_row, shadow_index)
        overlay = _matching_shadow(delta_row, overlay_index)
        merged = dict(shadow)
        merged.update(delta_row)
        if overlay:
            _apply_component_source_overlay(merged, overlay)
        actor = _normalize_actor(
            merged.get("canonical_coverage_actor")
            or merged.get("coverage_gain_claimed_actor")
            or merged.get("controlled_choice_source")
        )
        raw_claimed_actor = str(merged.get("coverage_gain_claimed_actor") or "")
        claimed_actor = _normalize_actor(raw_claimed_actor or actor)
        choice_source = str(merged.get("controlled_choice_source") or "")
        fallback_like = actor in FALLBACK_ACTORS or "fallback" in choice_source
        fallback_contamination = fallback_like and claimed_actor == SELECTED_ACTOR
        controlled_reasons = _string_list(merged.get("controlled_regression_reason_codes"))
        actual_delta = _float(merged.get("coverage_rate_delta"))
        expected_delta = _float(merged.get("expected_coverage_rate_delta"))
        gain_source = str(merged.get("actual_coverage_gain_source") or "missing")
        expected_actual_confused = (
            not _finite(actual_delta)
            and _finite(expected_delta)
            and abs(float(expected_delta)) > TOLERANCE
        )
        valid_actual_delta = _finite(actual_delta) and gain_source in VALID_ACTUAL_GAIN_SOURCES

        components = {name: 0.0 for name in REWARD_COMPONENTS}
        source_fields: dict[str, str | None] = {name: None for name in REWARD_COMPONENTS}
        source_values: dict[str, float | None] = {name: None for name in REWARD_COMPONENTS}

        if valid_actual_delta and not fallback_contamination:
            coverage_gain = max(float(actual_delta), 0.0)
            components["coverage_gain_bonus"] = _round(
                coverage_gain * REWARD_WEIGHTS["coverage_gain_bonus"]
            )
            source_fields["coverage_gain_bonus"] = "coverage_rate_delta"
            source_values["coverage_gain_bonus"] = float(actual_delta)

        value_metric = _valuable_metric(merged, shadow, actual_delta)
        if value_metric["present"]:
            valuable_value = max(float(value_metric["value"]), 0.0)
            components["valuable_area_bonus"] = _round(
                valuable_value * REWARD_WEIGHTS["valuable_area_bonus"]
            )
            source_fields["valuable_area_bonus"] = _overlay_source_field(
                merged,
                "valuable_area_bonus",
                value_metric["source"],
            )
            source_values["valuable_area_bonus"] = valuable_value

        info_metric = _information_metric(merged, shadow)
        if info_metric["present"]:
            info_value = max(float(info_metric["value"]), 0.0)
            components["information_gain_bonus"] = _round(
                info_value * REWARD_WEIGHTS["information_gain_bonus"]
            )
            source_fields["information_gain_bonus"] = _overlay_source_field(
                merged,
                "information_gain_bonus",
                info_metric["source"],
            )
            source_values["information_gain_bonus"] = info_value

        retention = _teacher_retention_bonus(merged, controlled_reasons)
        if retention["present"]:
            components["teacher_skill_retention_bonus"] = _round(float(retention["bonus"]))
            source_fields["teacher_skill_retention_bonus"] = retention["source"]
            source_values["teacher_skill_retention_bonus"] = float(retention["bonus"])

        for field_name, component_name, weight_name in (
            ("path_cost", "path_cost_penalty", "path_cost_penalty"),
            ("risk", "risk_penalty", "risk_penalty"),
            ("energy_cost", "energy_penalty", "energy_penalty"),
        ):
            metric = _cost_metric(merged, shadow, field_name)
            if metric["present"]:
                value = max(float(metric["value"]), 0.0)
                components[component_name] = _round(-value * REWARD_WEIGHTS[weight_name])
                source_fields[component_name] = _overlay_source_field(
                    merged,
                    component_name,
                    metric["source"],
                )
                source_values[component_name] = value

        source_fields["fallback_penalty"] = "controlled_choice_source"
        source_values["fallback_penalty"] = 1.0 if fallback_like else 0.0
        if fallback_like:
            components["fallback_penalty"] = _round(-REWARD_WEIGHTS["fallback_penalty"])

        source_fields["controlled_regression_penalty"] = "controlled_regression_reason_codes"
        source_values["controlled_regression_penalty"] = float(len(controlled_reasons))
        if controlled_reasons:
            components["controlled_regression_penalty"] = _round(
                -REWARD_WEIGHTS["controlled_regression_penalty"] * len(controlled_reasons)
            )

        rows.append(
            {
                "schema_version": COMPONENT_AUDIT_ROW_SCHEMA_VERSION,
                "reward_audit_index": index,
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
                "actual_coverage_gain_source": gain_source,
                "coverage_rate_delta": actual_delta,
                "expected_coverage_rate_delta": expected_delta,
                "fallback_like": fallback_like,
                "fallback_policy_gain_contamination": fallback_contamination,
                "expected_actual_coverage_confusion": expected_actual_confused,
                "controlled_regression_reason_codes": controlled_reasons,
                "reward_components": components,
                "coverage_aware_reward": _round(sum(components.values())),
                "source_fields": source_fields,
                "source_values": source_values,
                "existing_reward": _float(merged.get("reward")),
                "existing_reward_components": merged.get("reward_components")
                if isinstance(merged.get("reward_components"), dict)
                else {},
            }
        )
    return rows


def _load_component_source_overlay(
    *,
    reward_component_source_root: Path | None,
    repo_root: Path,
    coverage_signal_root: Path,
    coverage_performance_root: Path,
    reason_codes: list[str],
) -> list[dict[str, Any]]:
    root = Path(reward_component_source_root) if reward_component_source_root else repo_root / DEFAULT_CONNECT_SOURCE_ROOT
    if not root.exists():
        return []
    summary_path = root / CONNECT_SOURCE_SUMMARY_FILE
    summary = _read_json(
        summary_path,
        reason_codes if reward_component_source_root else [],
        "reward_component_source_summary",
        required=reward_component_source_root is not None,
    )
    if not summary:
        return []
    if summary.get("status") != "passed":
        if reward_component_source_root is not None:
            _add_reason(reason_codes, "reward_component_source_overlay_not_passed")
        return []
    summary_signal_root = summary.get("coverage_signal_root")
    summary_performance_root = summary.get("coverage_performance_root")
    if not (
        _same_path(summary_signal_root, coverage_signal_root)
        and _same_path(summary_performance_root, coverage_performance_root)
    ):
        if reward_component_source_root is not None:
            _add_reason(reason_codes, "reward_component_source_overlay_root_mismatch")
        return []
    provenance_path = _resolve_optional_path(
        summary.get("component_provenance"),
        root,
        repo_root,
    ) or (root / CONNECT_SOURCE_PROVENANCE_FILE)
    return _read_jsonl(
        provenance_path,
        reason_codes if reward_component_source_root else [],
        "reward_component_source_provenance",
    )


def _apply_component_source_overlay(row: dict[str, Any], overlay: dict[str, Any]) -> None:
    for field in ("actual_valuable_area_covered", "actual_information_gain", "risk"):
        value = _float(overlay.get(field))
        if _finite(value):
            row[field] = float(value)
    source_fields = overlay.get("source_fields")
    if isinstance(source_fields, dict):
        row["_component_source_overlay_fields"] = {
            str(key): str(value)
            for key, value in source_fields.items()
            if value
        }
    source_values = overlay.get("source_values")
    if isinstance(source_values, dict):
        row["_component_source_overlay_values"] = source_values


def _overlay_source_field(row: dict[str, Any], component: str, default_source: str | None) -> str | None:
    source_fields = row.get("_component_source_overlay_fields")
    if isinstance(source_fields, dict) and source_fields.get(component):
        return str(source_fields[component])
    return default_source


def _valuable_metric(row: dict[str, Any], shadow: dict[str, Any], actual_delta: float | None) -> dict[str, Any]:
    direct = _first_finite_field(
        row,
        shadow,
        ("valuable_area_covered", "actual_valuable_area_covered"),
    )
    if direct["present"]:
        return direct
    if not _finite(actual_delta):
        return {"present": False, "value": None, "source": None}
    value_feature = _candidate_feature_metric(row, shadow, "value")
    if value_feature["present"]:
        return {
            "present": True,
            "value": max(float(actual_delta), 0.0) * float(value_feature["value"]),
            "source": f"coverage_rate_delta*{value_feature['source']}",
        }
    return {"present": False, "value": None, "source": None}


def _information_metric(row: dict[str, Any], shadow: dict[str, Any]) -> dict[str, Any]:
    direct = _first_finite_field(row, shadow, ("information_gain", "actual_information_gain"))
    if direct["present"]:
        missing = _boolish(row.get("information_gain_missing")) or _boolish(shadow.get("information_gain_missing"))
        if not missing:
            return direct
    return _candidate_feature_metric(row, shadow, "information_gain")


def _cost_metric(row: dict[str, Any], shadow: dict[str, Any], field_name: str) -> dict[str, Any]:
    field_aliases = {
        "path_cost": ("path_cost", "path_cost_delta"),
        "risk": ("risk", "risk_delta"),
        "energy_cost": ("energy_cost", "energy_cost_delta", "energy"),
    }[field_name]
    feature = _candidate_feature_metric(row, shadow, field_name)
    direct = _first_finite_field(row, shadow, field_aliases)
    if (
        direct["present"]
        and abs(float(direct["value"])) <= TOLERANCE
        and _candidate_feature_declared_missing(row, shadow, field_name)
    ):
        return {"present": False, "value": None, "source": None}
    if (
        direct["present"]
        and abs(float(direct["value"])) <= TOLERANCE
        and feature["present"]
        and abs(float(feature["value"])) > TOLERANCE
    ):
        return feature
    if direct["present"] and abs(float(direct["value"])) > TOLERANCE:
        return direct
    if feature["present"]:
        return feature
    return direct


def _candidate_feature_declared_missing(
    row: dict[str, Any],
    shadow: dict[str, Any],
    feature_name: str,
) -> bool:
    for source in (row, shadow):
        observation = source.get("observation")
        if not isinstance(observation, dict):
            continue
        action_index = _int(source.get("controlled_action_index"))
        if _candidate_feature_missing(observation, action_index, feature_name):
            return True
    return False


def _candidate_feature_metric(row: dict[str, Any], shadow: dict[str, Any], feature_name: str) -> dict[str, Any]:
    for source, label in ((row, "observation"), (shadow, "observation")):
        feature = _candidate_feature_value(
            source.get("observation"),
            source.get("controlled_action_index"),
            feature_name,
        )
        if feature["present"]:
            source_label = f"{label}.candidate_features.{feature_name}"
            return {"present": True, "value": feature["value"], "source": source_label}
    return {"present": False, "value": None, "source": None}


def _candidate_feature_value(
    observation: Any,
    action_index_value: Any,
    feature_name: str,
) -> dict[str, Any]:
    if not isinstance(observation, dict):
        return {"present": False, "value": None}
    names = observation.get("candidate_feature_names")
    features = observation.get("candidate_features")
    if not isinstance(names, list) or not isinstance(features, list) or feature_name not in names:
        return {"present": False, "value": None}
    action_index = _int(action_index_value)
    if action_index < 0 or action_index >= len(features):
        return {"present": False, "value": None}
    action_features = features[action_index]
    if not isinstance(action_features, list):
        return {"present": False, "value": None}
    feature_index = names.index(feature_name)
    if feature_index >= len(action_features):
        return {"present": False, "value": None}
    if _candidate_feature_missing(observation, action_index, feature_name):
        return {"present": False, "value": None}
    value = _float(action_features[feature_index])
    if not _finite(value):
        return {"present": False, "value": None}
    return {"present": True, "value": float(value)}


def _candidate_feature_missing(observation: dict[str, Any], action_index: int, feature_name: str) -> bool:
    missing_names = observation.get("candidate_missing_indicator_names")
    missing_indicators = observation.get("candidate_missing_indicators")
    missing_name = f"{feature_name}_missing"
    if not isinstance(missing_names, list) or missing_name not in missing_names:
        return False
    if not isinstance(missing_indicators, list) or action_index >= len(missing_indicators):
        return False
    indicators = missing_indicators[action_index]
    if not isinstance(indicators, list):
        return False
    indicator_index = missing_names.index(missing_name)
    if indicator_index >= len(indicators):
        return False
    return _boolish(indicators[indicator_index])


def _first_finite_field(
    row: dict[str, Any],
    shadow: dict[str, Any],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    for source, prefix in ((row, ""), (shadow, "shadow_step.")):
        for field in fields:
            value = _float(source.get(field))
            if _finite(value):
                return {"present": True, "value": float(value), "source": f"{prefix}{field}"}
    return {"present": False, "value": None, "source": None}


def _teacher_retention_bonus(row: dict[str, Any], controlled_reasons: list[str]) -> dict[str, Any]:
    detail = str(row.get("controlled_choice_detail") or "")
    controlled_action = row.get("controlled_action_index")
    teacher_action = row.get("teacher_action_index")
    has_actions = controlled_action is not None and teacher_action is not None
    if has_actions and _int(controlled_action) == _int(teacher_action):
        return {
            "present": True,
            "bonus": REWARD_WEIGHTS["teacher_skill_retention_bonus"],
            "source": "controlled_action_index/teacher_action_index",
        }
    if detail == "policy_safe_disagreement" and not controlled_reasons:
        return {
            "present": True,
            "bonus": REWARD_WEIGHTS["safe_disagreement_retention_bonus"],
            "source": "controlled_choice_detail",
        }
    if has_actions or detail:
        return {"present": True, "bonus": 0.0, "source": "controlled_choice_detail"}
    return {"present": False, "bonus": 0.0, "source": None}


def _source_field_audit(
    component_rows: list[dict[str, Any]],
    collector_reward_audit: dict[str, Any],
) -> dict[str, Any]:
    component_status: dict[str, dict[str, Any]] = {}
    row_count = len(component_rows)
    for component in REWARD_COMPONENTS:
        present_rows = [row for row in component_rows if row["source_fields"].get(component)]
        positive_rows = [row for row in component_rows if row["reward_components"].get(component, 0.0) > TOLERANCE]
        negative_rows = [row for row in component_rows if row["reward_components"].get(component, 0.0) < -TOLERANCE]
        zero_rows = [row for row in component_rows if abs(row["reward_components"].get(component, 0.0)) <= TOLERANCE]
        component_status[component] = {
            "component": component,
            "required": True,
            "present_count": len(present_rows),
            "missing_count": row_count - len(present_rows),
            "positive_count": len(positive_rows),
            "negative_count": len(negative_rows),
            "zero_count": len(zero_rows),
            "source_fields": sorted(
                {str(row["source_fields"][component]) for row in present_rows}
            ),
            "source_status": "passed" if row_count > 0 and len(present_rows) == row_count else "failed",
        }

    missing_required = [
        name
        for name, status in component_status.items()
        if status["source_status"] != "passed"
    ]
    return {
        "schema_version": SOURCE_AUDIT_SCHEMA_VERSION,
        "audited_row_count": row_count,
        "component_status": component_status,
        "missing_required_components": missing_required,
        "missing_required_component_source_count": len(missing_required),
        "existing_reward_audit": collector_reward_audit,
    }


def _collector_reward_audit(
    seed_summaries: list[dict[str, Any]],
    repo_root: Path,
    reason_codes: list[str],
) -> dict[str, Any]:
    transition_count = 0
    reward_component_missing_count = 0
    non_finite_reward_count = 0
    component_counter: Counter[str] = Counter()
    collector_episode_paths: list[str] = []
    for seed in seed_summaries:
        collector_root = _resolve_optional_path(seed.get("collector_root"), repo_root, repo_root)
        if collector_root is None:
            continue
        episodes_path = collector_root / "ppo-rollout-episodes.jsonl"
        if not episodes_path.exists():
            _add_reason(reason_codes, "collector_episodes_missing")
            continue
        collector_episode_paths.append(str(episodes_path))
        for episode in _read_jsonl(episodes_path, reason_codes, "collector_episodes"):
            transitions = episode.get("transitions")
            if not isinstance(transitions, list):
                continue
            for transition in transitions:
                transition_count += 1
                if not _finite(transition.get("reward")):
                    non_finite_reward_count += 1
                components = transition.get("reward_components")
                if isinstance(components, dict) and components:
                    component_counter.update(str(name) for name in components)
                else:
                    reward_component_missing_count += 1
    current_component_names = sorted(component_counter)
    existing_reward_is_teacher_following = (
        transition_count > 0
        and (
            "teacher_following_bonus" in current_component_names
            or reward_component_missing_count == transition_count
        )
        and not any(
            name
            in {
                "coverage_gain_bonus",
                "valuable_area_bonus",
                "information_gain_bonus",
            }
            for name in current_component_names
        )
    )
    return {
        "collector_episode_paths": collector_episode_paths,
        "collector_transition_count": transition_count,
        "collector_reward_component_missing_count": reward_component_missing_count,
        "collector_non_finite_reward_count": non_finite_reward_count,
        "current_reward_component_counts": dict(sorted(component_counter.items())),
        "existing_reward_is_teacher_following": existing_reward_is_teacher_following,
    }


def _rescore_comparison(
    *,
    component_rows: list[dict[str, Any]],
    performance_summary: dict[str, Any],
    performance_comparison: dict[str, Any],
    metric_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    actor_rescores = {
        actor: _aggregate_actor_rescore(actor, rows)
        for actor, rows in sorted(_group_by_actor(component_rows).items())
    }
    selected_teacher_equivalent = bool(
        performance_comparison.get("selected_teacher_equivalent")
        or _string_list(
            performance_summary.get("best_baseline_metrics", {}).get("comparator_basis")
            if isinstance(performance_summary.get("best_baseline_metrics"), dict)
            else None
        )
        == ["teacher_action_equivalent_from_policy_teacher_aligned_shadow"]
    )
    if selected_teacher_equivalent and "teacher" not in actor_rescores and SELECTED_ACTOR in actor_rescores:
        teacher_rescore = dict(actor_rescores[SELECTED_ACTOR])
        teacher_rescore["actor"] = "teacher"
        teacher_rescore["comparator_basis"] = ["teacher_action_equivalent_from_policy_teacher_aligned_shadow"]
        actor_rescores["teacher"] = teacher_rescore

    best_baseline_actor = str(
        performance_comparison.get("best_baseline_actor")
        or performance_summary.get("best_baseline_actor")
        or "teacher"
    )
    if best_baseline_actor not in actor_rescores:
        baseline_candidates = sorted(actor for actor in actor_rescores if actor in BASELINE_ACTORS)
        best_baseline_actor = baseline_candidates[0] if baseline_candidates else ""
    selected = actor_rescores.get(SELECTED_ACTOR)
    baseline = actor_rescores.get(best_baseline_actor) if best_baseline_actor else None
    improvement = None
    if selected and baseline:
        improvement = _round(selected["coverage_aware_reward"] - baseline["coverage_aware_reward"])
    performance_improvement_metrics = (
        performance_summary.get("coverage_return_improvement"),
        performance_summary.get("cumulative_coverage_rate_delta_improvement"),
        performance_summary.get("valuable_area_covered_improvement"),
    )
    selected_candidate_performance_improved = (
        not selected_teacher_equivalent
        and all(_float_or_default(value, 0.0) > TOLERANCE for value in performance_improvement_metrics)
    )
    return {
        "schema_version": RESCORE_COMPARISON_SCHEMA_VERSION,
        "actor_rescores": actor_rescores,
        "metric_actor_count": len(metric_rows),
        "selected_actor": SELECTED_ACTOR,
        "best_baseline_actor": best_baseline_actor or None,
        "coverage_aware_reward_improvement": improvement,
        "selected_teacher_equivalent": selected_teacher_equivalent,
        "selected_candidate_performance_improved": selected_candidate_performance_improved,
        "previous_coverage_performance_status": performance_summary.get("coverage_performance_status"),
        "previous_coverage_performance_reason_codes": _string_list(performance_summary.get("reason_codes")),
        "previous_coverage_return_improvement": performance_summary.get("coverage_return_improvement"),
        "previous_cumulative_coverage_rate_delta_improvement": performance_summary.get(
            "cumulative_coverage_rate_delta_improvement"
        ),
        "previous_valuable_area_covered_improvement": performance_summary.get(
            "valuable_area_covered_improvement"
        ),
    }


def _aggregate_actor_rescore(actor: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    component_totals = {
        component: _round(sum(float(row["reward_components"][component]) for row in rows))
        for component in REWARD_COMPONENTS
    }
    return {
        "actor": actor,
        "row_count": len(rows),
        "coverage_aware_reward": _round(sum(component_totals.values())),
        "reward_component_totals": component_totals,
        "actual_coverage_gain": _round(
            sum(
                float(row["coverage_rate_delta"])
                for row in rows
                if _finite(row.get("coverage_rate_delta"))
            )
        ),
        "fallback_row_count": sum(1 for row in rows if row.get("fallback_like")),
        "controlled_regression_count": sum(
            1 for row in rows if _string_list(row.get("controlled_regression_reason_codes"))
        ),
        "comparator_basis": [],
    }


def _group_by_actor(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("actor") or "unknown")].append(row)
    return grouped


def _rejection_report(
    *,
    reason_codes: list[str],
    source_field_audit: dict[str, Any],
    component_rows: list[dict[str, Any]],
    coverage_signal_summary: dict[str, Any],
    rescore_comparison: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    expected_actual_count = _int(
        coverage_signal_summary.get("expected_actual_coverage_confusion_count")
    ) + sum(1 for row in component_rows if row.get("expected_actual_coverage_confusion"))
    fallback_contamination_count = _int(
        coverage_signal_summary.get("fallback_coverage_gain_claimed_as_policy_gain_count")
    ) + sum(1 for row in component_rows if row.get("fallback_policy_gain_contamination"))
    controlled_regression_count = _int(coverage_signal_summary.get("controlled_regression_count")) + sum(
        1 for row in component_rows if _string_list(row.get("controlled_regression_reason_codes"))
    )
    if expected_actual_count > 0:
        _add_reason(reasons, "expected_actual_coverage_confusion")
    if fallback_contamination_count > 0:
        _add_reason(reasons, "fallback_policy_gain_contamination")
    if controlled_regression_count > 0:
        _add_reason(reasons, "controlled_regression_present")
    missing_components = source_field_audit["missing_required_components"]
    if missing_components:
        _add_reason(reasons, "reward_component_source_missing")
    if "coverage_gain_bonus" in missing_components:
        _add_reason(reasons, "actual_coverage_signal_missing")
    if "valuable_area_bonus" in missing_components:
        _add_reason(reasons, "valuable_coverage_signal_missing")
    if "information_gain_bonus" in missing_components:
        _add_reason(reasons, "information_gain_signal_missing")
    if "path_cost_penalty" in missing_components:
        _add_reason(reasons, "path_cost_signal_missing")
    if "risk_penalty" in missing_components:
        _add_reason(reasons, "risk_signal_missing")
    if "energy_penalty" in missing_components:
        _add_reason(reasons, "energy_signal_missing")
    if SELECTED_ACTOR not in rescore_comparison["actor_rescores"]:
        _add_reason(reasons, "selected_ppo_candidate_evidence_missing")

    rejection_reason_counts = Counter(reason_codes)
    rejection_reason_counts.update(reasons)
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "reason_codes": reasons,
        "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "expected_actual_coverage_confusion_count": expected_actual_count,
        "fallback_coverage_gain_claimed_as_policy_gain_count": fallback_contamination_count,
        "controlled_regression_count": controlled_regression_count,
        "missing_required_components": missing_components,
        "selected_candidate_performance_improved": rescore_comparison[
            "selected_candidate_performance_improved"
        ],
        "selected_teacher_equivalent": rescore_comparison["selected_teacher_equivalent"],
    }


def _next_required_change(reason_codes: list[str]) -> str:
    if not reason_codes:
        return "coverage_driven_ppo_improvement_run"
    if "controlled_regression_present" in reason_codes:
        return "fix_controlled_regression_before_reward_refinement"
    if "expected_actual_coverage_confusion" in reason_codes or "fallback_policy_gain_contamination" in reason_codes:
        return "fix_reward_refinement_inputs"
    if "reward_component_source_missing" in reason_codes:
        return "connect_reward_component_source_fields"
    return "coverage_aware_reward_refinement"


def _reward_contract_payload() -> dict[str, Any]:
    return {
        "formula": (
            "coverage_gain_bonus + valuable_area_bonus + information_gain_bonus + "
            "teacher_skill_retention_bonus - path_cost_penalty - risk_penalty - "
            "energy_penalty - fallback_penalty - controlled_regression_penalty"
        ),
        "weights": REWARD_WEIGHTS,
        "actual_coverage_sources": sorted(VALID_ACTUAL_GAIN_SOURCES),
        "fallback_policy_gain_disallowed": True,
        "expected_coverage_as_actual_gain_disallowed": True,
    }


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
        ("shadow_episode_id", "shadow_step_index"),
    ):
        episode_id = row.get(episode_field)
        step_index = row.get(step_field)
        if episode_id is not None and step_index is not None:
            keys.append(f"episode-step:{episode_id}:{step_index}")
    return keys


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


def _empty_actor_rescore(actor: str) -> dict[str, Any]:
    return {
        "actor": actor,
        "row_count": 0,
        "coverage_aware_reward": 0.0,
        "reward_component_totals": {component: 0.0 for component in REWARD_COMPONENTS},
        "actual_coverage_gain": 0.0,
        "fallback_row_count": 0,
        "controlled_regression_count": 0,
        "comparator_basis": [],
    }


def _read_json(
    path: Path,
    reason_codes: list[str],
    label: str,
    *,
    required: bool = True,
) -> dict[str, Any]:
    if not path.exists():
        if required:
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


def _same_path(value: Any, expected: Path) -> bool:
    if not value:
        return False
    try:
        return Path(str(value)).resolve() == Path(expected).resolve()
    except OSError:
        return str(value) == str(expected)


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


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return abs(float(value)) > TOLERANCE
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _round(value: float) -> float:
    return round(float(value), 12)


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _render_report(
    summary: dict[str, Any],
    source_field_audit: dict[str, Any],
    rescore_comparison: dict[str, Any],
    rejection_report: dict[str, Any],
) -> str:
    lines = [
        "# Coverage-Aware Reward Refinement v1",
        "",
        f"Status: `{summary['status']}`",
        f"Reward refinement status: `{summary['reward_refinement_status']}`",
        f"Reason codes: `{summary['reason_codes']}`",
        f"Next required change: `{summary['next_required_change']}`",
        "",
        "## Reward Contract",
        "",
        f"Formula: `{summary['reward_contract']['formula']}`",
        "",
        "## Source Field Audit",
        "",
    ]
    for component, status in source_field_audit["component_status"].items():
        lines.append(
            "- `{component}`: status=`{source_status}`, present=`{present}`, missing=`{missing}`, "
            "positive=`{positive}`, negative=`{negative}`, sources=`{sources}`".format(
                component=component,
                source_status=status["source_status"],
                present=status["present_count"],
                missing=status["missing_count"],
                positive=status["positive_count"],
                negative=status["negative_count"],
                sources=status["source_fields"],
            )
        )
    lines.extend(
        [
            "",
            "## Re-score",
            "",
            f"- Selected actor: `{summary['selected_actor']}`",
            f"- Best baseline actor: `{summary.get('best_baseline_actor')}`",
            f"- Coverage-aware reward improvement: `{summary.get('coverage_aware_reward_improvement')}`",
            f"- Selected teacher equivalent: `{summary.get('selected_teacher_equivalent')}`",
            f"- Selected candidate performance improved: `{summary.get('selected_candidate_performance_improved')}`",
            "",
            "## Rejection Reason Counts",
            "",
        ]
    )
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
            "This stage is read-only. It does not run a new PPO update, publish checkpoints, "
            "replace the default policy, relax guards, connect a real executor, or make a formal performance claim.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
