from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
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


SUMMARY_SCHEMA_VERSION = "connect-reward-component-source-fields-summary/v1"
PROVENANCE_ROW_SCHEMA_VERSION = "connect-reward-component-provenance-row/v1"
SOURCE_FIELD_WIRING_SCHEMA_VERSION = "connect-reward-source-field-wiring-audit/v1"
REPLAY_VALIDATION_SCHEMA_VERSION = "connect-reward-source-field-replay-validation/v1"
REJECTION_REPORT_SCHEMA_VERSION = "connect-reward-source-field-rejection-report/v1"

COVERAGE_SIGNAL_SUMMARY_FILE = "exploration-coverage-signal-audit-summary.json"
COVERAGE_DELTA_AUDIT_FILE = "coverage-delta-audit.jsonl"
COVERAGE_PERFORMANCE_SUMMARY_FILE = "exploration-coverage-performance-evaluation-summary.json"
REWARD_REFINEMENT_SUMMARY_FILE = "coverage-aware-reward-refinement-summary.json"
SOURCE_FIELD_AUDIT_FILE = "source-field-audit.json"

SUMMARY_FILE = "connect-reward-component-source-fields-summary.json"
SOURCE_FIELD_WIRING_FILE = "source-field-wiring-audit.json"
COMPONENT_PROVENANCE_FILE = "component-provenance.jsonl"
REPLAY_VALIDATION_FILE = "replay-validation.json"
REJECTION_REPORT_FILE = "connect-reward-component-source-fields-rejection-report.json"
REPORT_FILE = "connect-reward-component-source-fields-report.md"

DEFAULT_PATH_FEEDBACK_SUMMARIES = (
    Path(
        "outputs/path_feedback_batch_quasi_real_teacher_distillation_dataset_v1/"
        "quasi-real-teacher-distillation-path-feedback-summary.json"
    ),
    Path(
        "outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1/"
        "quasi-real-map-path-feedback-summary.json"
    ),
)

REQUIRED_COMPONENTS = (
    "valuable_area_bonus",
    "information_gain_bonus",
    "risk_penalty",
)
FALLBACK_MARKERS = {"fallback", "source_fallback", "teacher_fallback"}
TOLERANCE = 1e-9


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Connect real source fields for coverage-aware reward components."
    )
    parser.add_argument("--coverage-signal-root", required=True)
    parser.add_argument("--coverage-performance-root", required=True)
    parser.add_argument("--reward-refinement-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--path-feedback-root",
        action="append",
        default=[],
        help="Path-feedback summary file or directory. May be repeated.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    summary = run_connect_reward_component_source_fields(
        coverage_signal_root=_resolve_path(Path(args.coverage_signal_root), repo_root, repo_root),
        coverage_performance_root=_resolve_path(Path(args.coverage_performance_root), repo_root, repo_root),
        reward_refinement_root=_resolve_path(Path(args.reward_refinement_root), repo_root, repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root, repo_root),
        repo_root=repo_root,
        path_feedback_roots=[
            _resolve_path(Path(value), repo_root, repo_root)
            for value in args.path_feedback_root
        ],
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "reward_component_source_field_status": summary[
                    "reward_component_source_field_status"
                ],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_connect_reward_component_source_fields(
    *,
    coverage_signal_root: Path,
    coverage_performance_root: Path,
    reward_refinement_root: Path,
    output_root: Path,
    repo_root: Path,
    path_feedback_roots: list[Path] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    coverage_signal_root = Path(coverage_signal_root)
    coverage_performance_root = Path(coverage_performance_root)
    reward_refinement_root = Path(reward_refinement_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary": output_root / SUMMARY_FILE,
        "source_field_wiring_audit": output_root / SOURCE_FIELD_WIRING_FILE,
        "component_provenance": output_root / COMPONENT_PROVENANCE_FILE,
        "replay_validation": output_root / REPLAY_VALIDATION_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }

    input_reasons: list[str] = []
    coverage_signal_summary_path = coverage_signal_root / COVERAGE_SIGNAL_SUMMARY_FILE
    coverage_performance_summary_path = coverage_performance_root / COVERAGE_PERFORMANCE_SUMMARY_FILE
    reward_refinement_summary_path = reward_refinement_root / REWARD_REFINEMENT_SUMMARY_FILE
    reward_source_audit_path = reward_refinement_root / SOURCE_FIELD_AUDIT_FILE

    coverage_signal_summary = _read_json(
        coverage_signal_summary_path,
        input_reasons,
        "coverage_signal_summary",
    )
    coverage_performance_summary = _read_json(
        coverage_performance_summary_path,
        input_reasons,
        "coverage_performance_summary",
    )
    reward_refinement_summary = _read_json(
        reward_refinement_summary_path,
        input_reasons,
        "reward_refinement_summary",
    )
    reward_source_audit = _read_json(
        _resolve_optional_path(
            reward_refinement_summary.get("source_field_audit"),
            reward_refinement_root,
            repo_root,
        )
        or reward_source_audit_path,
        input_reasons,
        "reward_source_field_audit",
    )

    delta_path = _resolve_optional_path(
        coverage_signal_summary.get("coverage_delta_audit"),
        coverage_signal_root,
        repo_root,
    ) or (coverage_signal_root / COVERAGE_DELTA_AUDIT_FILE)
    shadow_steps_path = _resolve_optional_path(
        coverage_performance_summary.get("shadow_steps"),
        coverage_performance_root,
        repo_root,
    )
    if shadow_steps_path is None:
        _add_reason(input_reasons, "shadow_steps_missing")
        shadow_steps_path = coverage_performance_root / "shadow-steps.jsonl"

    delta_rows = _read_jsonl(delta_path, input_reasons, "coverage_delta_audit")
    shadow_steps = _read_jsonl(shadow_steps_path, input_reasons, "shadow_steps")

    path_feedback_paths = _path_feedback_summary_paths(
        path_feedback_roots,
        repo_root,
        input_reasons,
    )
    candidate_index = _path_feedback_candidate_index(path_feedback_paths, repo_root, input_reasons)
    shadow_index = _shadow_index(shadow_steps)

    provenance_rows = [
        _provenance_row(
            delta_row=row,
            row_index=index,
            shadow=_matching_shadow(row, shadow_index),
            candidate_index=candidate_index,
        )
        for index, row in enumerate(delta_rows)
    ]
    source_field_wiring_audit = _source_field_wiring_audit(
        provenance_rows,
        reward_source_audit,
        path_feedback_paths,
    )
    replay_validation = _replay_validation(
        provenance_rows,
        coverage_signal_summary,
        coverage_performance_summary,
    )
    rejection_report = _rejection_report(
        input_reasons=input_reasons,
        provenance_rows=provenance_rows,
        source_field_wiring_audit=source_field_wiring_audit,
        replay_validation=replay_validation,
        coverage_signal_summary=coverage_signal_summary,
    )
    reason_codes = list(input_reasons)
    for reason in rejection_report["reason_codes"]:
        _add_reason(reason_codes, reason)

    reward_component_source_field_status = "passed" if not reason_codes else "failed"
    status = "passed" if reward_component_source_field_status == "passed" else "failed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": _next_required_change(reason_codes),
        "reward_component_source_field_status": reward_component_source_field_status,
        "coverage_signal_root": str(coverage_signal_root),
        "coverage_performance_root": str(coverage_performance_root),
        "reward_refinement_root": str(reward_refinement_root),
        "output_root": str(output_root),
        "coverage_signal_summary": str(coverage_signal_summary_path),
        "coverage_performance_summary": str(coverage_performance_summary_path),
        "reward_refinement_summary": str(reward_refinement_summary_path),
        "coverage_delta_audit": str(delta_path),
        "shadow_steps": str(shadow_steps_path),
        "path_feedback_summaries": [str(path) for path in path_feedback_paths],
        "summary": str(paths["summary"]),
        "source_field_wiring_audit": str(paths["source_field_wiring_audit"]),
        "component_provenance": str(paths["component_provenance"]),
        "replay_validation": str(paths["replay_validation"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "audited_row_count": len(provenance_rows),
        "connected_row_count": source_field_wiring_audit["all_required_source_present_count"],
        "expected_actual_coverage_confusion_count": replay_validation[
            "expected_actual_coverage_confusion_count"
        ],
        "fallback_coverage_gain_claimed_as_policy_gain_count": replay_validation[
            "fallback_coverage_gain_claimed_as_policy_gain_count"
        ],
        "controlled_regression_count": replay_validation["controlled_regression_count"],
        "component_status": source_field_wiring_audit["component_status"],
        "missing_required_components": source_field_wiring_audit[
            "missing_required_components"
        ],
        "path_feedback_match_method_counts": source_field_wiring_audit[
            "path_feedback_match_method_counts"
        ],
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_release_claimed": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }

    _write_json(paths["summary"], summary)
    _write_json(paths["source_field_wiring_audit"], source_field_wiring_audit)
    _write_jsonl(paths["component_provenance"], provenance_rows)
    _write_json(paths["replay_validation"], replay_validation)
    _write_json(paths["rejection_report"], rejection_report)
    _write_report(paths["report"], summary, source_field_wiring_audit, replay_validation)
    return summary


def _provenance_row(
    *,
    delta_row: dict[str, Any],
    row_index: int,
    shadow: dict[str, Any],
    candidate_index: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(shadow)
    merged.update(delta_row)
    action_index = _first_int(
        merged.get("controlled_action_index"),
        merged.get("action_index"),
        merged.get("raw_policy_action_index"),
    )
    actual_delta = _float(merged.get("coverage_rate_delta"))
    expected_delta = _float(merged.get("expected_coverage_rate_delta"))
    candidate, match_method = _match_path_feedback_candidate(
        merged,
        action_index,
        candidate_index,
    )
    row_reasons: list[str] = []
    expected_actual_confusion = (
        _finite(expected_delta)
        and abs(float(expected_delta)) > TOLERANCE
        and (not _finite(actual_delta) or abs(float(actual_delta)) <= TOLERANCE)
    )
    if expected_actual_confusion:
        row_reasons.append("expected_actual_coverage_confusion")

    choice_source = str(merged.get("controlled_choice_source") or "")
    claimed_actor = str(merged.get("coverage_gain_claimed_actor") or "")
    fallback_like = _is_fallback_like(choice_source)
    fallback_policy_gain_contamination = fallback_like and _normalize_actor(claimed_actor) == "selected_ppo_candidate"
    if fallback_policy_gain_contamination:
        row_reasons.append("fallback_policy_gain_contamination")

    controlled_reasons = _string_list(
        merged.get("controlled_regression_reason_codes")
        or shadow.get("controlled_regression_reason_codes")
    )
    if controlled_reasons:
        row_reasons.append("controlled_regression_present")

    utility = _float(candidate.get("utility")) if candidate else None
    risk = _float(candidate.get("risk")) if candidate else None
    actual_valuable = None
    actual_information = None
    source_fields = {
        "valuable_area_bonus": None,
        "information_gain_bonus": None,
        "risk_penalty": None,
    }
    source_values = {
        "valuable_area_bonus": None,
        "information_gain_bonus": None,
        "risk_penalty": None,
    }
    if candidate is None:
        row_reasons.append("path_feedback_candidate_source_missing")
    if not _finite(actual_delta):
        row_reasons.append("actual_coverage_signal_missing")
    elif not expected_actual_confusion:
        actual_information = max(float(actual_delta), 0.0)
        source_fields["information_gain_bonus"] = "path_feedback.coverage_rate_delta"
        source_values["information_gain_bonus"] = actual_information
    if candidate is not None and not _finite(utility):
        row_reasons.append("valuable_coverage_signal_missing")
    if _finite(actual_delta) and _finite(utility) and not expected_actual_confusion:
        actual_valuable = max(float(actual_delta), 0.0) * float(utility)
        source_fields["valuable_area_bonus"] = "path_feedback.coverage_rate_delta*path_feedback.candidates.utility"
        source_values["valuable_area_bonus"] = actual_valuable
    if candidate is not None and not _finite(risk):
        row_reasons.append("risk_signal_missing")
    if _finite(risk):
        source_fields["risk_penalty"] = "path_feedback.candidates.risk"
        source_values["risk_penalty"] = float(risk)

    return {
        "schema_version": PROVENANCE_ROW_SCHEMA_VERSION,
        "provenance_index": row_index,
        "audit_index": delta_row.get("audit_index"),
        "context_id": merged.get("context_id"),
        "scenario_id": merged.get("scenario_id"),
        "scenario_family": merged.get("scenario_family"),
        "split": merged.get("split"),
        "episode_id": merged.get("episode_id"),
        "step_index": merged.get("step_index"),
        "source_episode_id": merged.get("source_episode_id"),
        "source_step_index": merged.get("source_step_index"),
        "controlled_action_index": action_index,
        "controlled_choice_source": choice_source,
        "coverage_gain_claimed_actor": claimed_actor,
        "coverage_rate_delta": actual_delta,
        "expected_coverage_rate_delta": expected_delta,
        "actual_coverage_gain_source": merged.get("actual_coverage_gain_source"),
        "actual_valuable_area_covered": _round_optional(actual_valuable),
        "actual_information_gain": _round_optional(actual_information),
        "risk": _round_optional(risk),
        "path_feedback_candidate_context_id": candidate.get("context_id") if candidate else None,
        "path_feedback_candidate_scenario_id": candidate.get("scenario_id") if candidate else None,
        "path_feedback_candidate_action_index": candidate.get("action_index") if candidate else None,
        "path_feedback_source_artifact": candidate.get("_source_artifact") if candidate else None,
        "path_feedback_match_method": match_method,
        "source_fields": source_fields,
        "source_values": {key: _round_optional(value) for key, value in source_values.items()},
        "row_reason_codes": _unique(row_reasons),
        "expected_actual_coverage_confusion": expected_actual_confusion,
        "fallback_policy_gain_contamination": fallback_policy_gain_contamination,
        "controlled_regression_reason_codes": controlled_reasons,
    }


def _match_path_feedback_candidate(
    row: dict[str, Any],
    action_index: int | None,
    candidate_index: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    context_id = row.get("context_id")
    if context_id and context_id in candidate_index["by_context"]:
        return candidate_index["by_context"][context_id][0], "context"
    if action_index is None:
        return None, "missing_action"
    scenario_id = row.get("scenario_id")
    for index_name, method in (
        ("by_container_scenario_action", "scenario_action"),
        ("by_source_scenario_action", "source_scenario_action"),
    ):
        candidate_key = (str(scenario_id), int(action_index))
        matches = candidate_index[index_name].get(candidate_key)
        if matches:
            return matches[0], method
    return None, "missing"


def _path_feedback_candidate_index(
    paths: list[Path],
    repo_root: Path,
    reason_codes: list[str],
) -> dict[str, Any]:
    by_context: dict[str, list[dict[str, Any]]] = {}
    by_container_scenario_action: dict[tuple[str, int], list[dict[str, Any]]] = {}
    by_source_scenario_action: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for path in paths:
        summary = _read_json(path, reason_codes, "path_feedback_summary")
        for scenario in _list_of_dicts(summary.get("scenarios")):
            container_scenario_id = str(scenario.get("scenario_id") or "")
            candidates = _path_feedback_candidates(scenario)
            for candidate in candidates:
                enriched = dict(candidate)
                enriched["_source_artifact"] = str(path)
                enriched["_container_scenario_id"] = container_scenario_id
                context_id = enriched.get("context_id")
                if context_id:
                    by_context.setdefault(str(context_id), []).append(enriched)
                action_index = _first_int(enriched.get("action_index"))
                if action_index is not None:
                    if container_scenario_id:
                        by_container_scenario_action.setdefault(
                            (container_scenario_id, action_index),
                            [],
                        ).append(enriched)
                    source_scenario_id = str(enriched.get("scenario_id") or "")
                    if source_scenario_id:
                        by_source_scenario_action.setdefault(
                            (source_scenario_id, action_index),
                            [],
                        ).append(enriched)
    if not (by_context or by_container_scenario_action or by_source_scenario_action):
        _add_reason(reason_codes, "path_feedback_candidate_index_empty")
    return {
        "by_context": by_context,
        "by_container_scenario_action": by_container_scenario_action,
        "by_source_scenario_action": by_source_scenario_action,
        "path_count": len(paths),
        "candidate_count": sum(len(rows) for rows in by_context.values()),
    }


def _path_feedback_candidates(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    path_feedback = scenario.get("path_feedback")
    if isinstance(path_feedback, dict):
        candidates.extend(_list_of_dicts(path_feedback.get("candidates")))
        for key in ("best_by_path_cost", "selected_candidate", "best_candidate"):
            value = path_feedback.get(key)
            if isinstance(value, dict):
                candidates.append(value)
    for key in (
        "candidate_audit",
        "channel_aware_astar_candidate_audit",
        "sampled_region_path_candidate_audit",
    ):
        candidates.extend(_list_of_dicts(scenario.get(key)))
    return candidates


def _path_feedback_summary_paths(
    roots: list[Path] | None,
    repo_root: Path,
    reason_codes: list[str],
) -> list[Path]:
    requested = list(roots or [])
    if not requested:
        requested = [repo_root / path for path in DEFAULT_PATH_FEEDBACK_SUMMARIES]
    paths: list[Path] = []
    for root in requested:
        root = _resolve_path(Path(root), repo_root, repo_root)
        if root.is_file():
            paths.append(root)
            continue
        if not root.exists():
            _add_reason(reason_codes, "path_feedback_root_missing")
            continue
        for name in (
            "path-feedback-summary.json",
            "quasi-real-map-path-feedback-summary.json",
            "quasi-real-teacher-distillation-path-feedback-summary.json",
        ):
            candidate = root / name
            if candidate.is_file():
                paths.append(candidate)
        if not any(path.parent == root for path in paths):
            paths.extend(sorted(root.glob("*path-feedback-summary.json")))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        _add_reason(reason_codes, "path_feedback_summary_missing")
    return unique


def _source_field_wiring_audit(
    provenance_rows: list[dict[str, Any]],
    reward_source_audit: dict[str, Any],
    path_feedback_paths: list[Path],
) -> dict[str, Any]:
    row_count = len(provenance_rows)
    component_status: dict[str, dict[str, Any]] = {}
    for component in REQUIRED_COMPONENTS:
        present_rows = [row for row in provenance_rows if row["source_fields"].get(component)]
        positive_rows = [
            row
            for row in provenance_rows
            if _float(row["source_values"].get(component)) is not None
            and float(row["source_values"][component]) > TOLERANCE
        ]
        component_status[component] = {
            "component": component,
            "required": True,
            "present_count": len(present_rows),
            "missing_count": row_count - len(present_rows),
            "positive_count": len(positive_rows),
            "source_fields": sorted(
                {
                    str(row["source_fields"][component])
                    for row in present_rows
                    if row["source_fields"].get(component)
                }
            ),
            "source_status": "passed" if row_count > 0 and len(present_rows) == row_count else "failed",
        }
    missing_required = [
        name for name, status in component_status.items() if status["source_status"] != "passed"
    ]
    match_counts = Counter(str(row.get("path_feedback_match_method") or "missing") for row in provenance_rows)
    return {
        "schema_version": SOURCE_FIELD_WIRING_SCHEMA_VERSION,
        "audited_row_count": row_count,
        "path_feedback_summary_paths": [str(path) for path in path_feedback_paths],
        "component_status": component_status,
        "missing_required_components": missing_required,
        "missing_required_component_source_count": len(missing_required),
        "all_required_source_present_count": sum(
            1
            for row in provenance_rows
            if all(row["source_fields"].get(component) for component in REQUIRED_COMPONENTS)
        ),
        "path_feedback_match_method_counts": dict(sorted(match_counts.items())),
        "previous_reward_source_audit_missing_required_components": _string_list(
            reward_source_audit.get("missing_required_components")
        ),
    }


def _replay_validation(
    provenance_rows: list[dict[str, Any]],
    coverage_signal_summary: dict[str, Any],
    coverage_performance_summary: dict[str, Any],
) -> dict[str, Any]:
    expected_actual_count = max(
        _int(coverage_signal_summary.get("expected_actual_coverage_confusion_count")),
        sum(1 for row in provenance_rows if row.get("expected_actual_coverage_confusion")),
    )
    fallback_count = max(
        _int(coverage_signal_summary.get("fallback_coverage_gain_claimed_as_policy_gain_count")),
        sum(1 for row in provenance_rows if row.get("fallback_policy_gain_contamination")),
    )
    controlled_regression_count = max(
        _int(coverage_signal_summary.get("controlled_regression_count")),
        sum(1 for row in provenance_rows if _string_list(row.get("controlled_regression_reason_codes"))),
    )
    return {
        "schema_version": REPLAY_VALIDATION_SCHEMA_VERSION,
        "audited_row_count": len(provenance_rows),
        "coverage_signal_status": coverage_signal_summary.get("coverage_signal_status"),
        "coverage_performance_status": coverage_performance_summary.get(
            "coverage_performance_status"
        ),
        "expected_actual_coverage_confusion_count": expected_actual_count,
        "fallback_coverage_gain_claimed_as_policy_gain_count": fallback_count,
        "controlled_regression_count": controlled_regression_count,
        "actual_valuable_area_covered_positive_count": sum(
            1
            for row in provenance_rows
            if _float(row.get("actual_valuable_area_covered")) is not None
            and float(row["actual_valuable_area_covered"]) > TOLERANCE
        ),
        "actual_information_gain_positive_count": sum(
            1
            for row in provenance_rows
            if _float(row.get("actual_information_gain")) is not None
            and float(row["actual_information_gain"]) > TOLERANCE
        ),
        "risk_source_present_count": sum(
            1 for row in provenance_rows if row["source_fields"].get("risk_penalty")
        ),
    }


def _rejection_report(
    *,
    input_reasons: list[str],
    provenance_rows: list[dict[str, Any]],
    source_field_wiring_audit: dict[str, Any],
    replay_validation: dict[str, Any],
    coverage_signal_summary: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if replay_validation["expected_actual_coverage_confusion_count"] > 0:
        _add_reason(reasons, "expected_actual_coverage_confusion")
    if replay_validation["fallback_coverage_gain_claimed_as_policy_gain_count"] > 0:
        _add_reason(reasons, "fallback_policy_gain_contamination")
    if replay_validation["controlled_regression_count"] > 0:
        _add_reason(reasons, "controlled_regression_present")
    missing_components = source_field_wiring_audit["missing_required_components"]
    if missing_components:
        _add_reason(reasons, "reward_component_source_missing")
    if "valuable_area_bonus" in missing_components:
        _add_reason(reasons, "valuable_coverage_signal_missing")
    if "information_gain_bonus" in missing_components:
        _add_reason(reasons, "information_gain_signal_missing")
    if "risk_penalty" in missing_components:
        _add_reason(reasons, "risk_signal_missing")
    row_reason_counts: Counter[str] = Counter()
    for row in provenance_rows:
        row_reason_counts.update(_string_list(row.get("row_reason_codes")))
    if row_reason_counts.get("path_feedback_candidate_source_missing"):
        _add_reason(reasons, "path_feedback_candidate_source_missing")
    if _int(coverage_signal_summary.get("nonzero_actual_coverage_delta_count")) <= 0:
        _add_reason(reasons, "actual_coverage_signal_missing")

    rejection_reason_counts = Counter(input_reasons)
    rejection_reason_counts.update(reasons)
    rejection_reason_counts.update(row_reason_counts)
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "reason_codes": reasons,
        "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "row_reason_counts": dict(sorted(row_reason_counts.items())),
        "missing_required_components": missing_components,
        "expected_actual_coverage_confusion_count": replay_validation[
            "expected_actual_coverage_confusion_count"
        ],
        "fallback_coverage_gain_claimed_as_policy_gain_count": replay_validation[
            "fallback_coverage_gain_claimed_as_policy_gain_count"
        ],
        "controlled_regression_count": replay_validation["controlled_regression_count"],
    }


def _shadow_index(shadow_steps: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for step in shadow_steps:
        for key in _row_keys(step):
            index.setdefault(key, step)
    return index


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


def _next_required_change(reason_codes: list[str]) -> str:
    if not reason_codes:
        return "rerun_coverage_aware_reward_refinement"
    if (
        "expected_actual_coverage_confusion" in reason_codes
        or "fallback_policy_gain_contamination" in reason_codes
        or "controlled_regression_present" in reason_codes
    ):
        return "fix_reward_component_source_inputs"
    return "connect_reward_component_source_fields"


def _write_report(
    path: Path,
    summary: dict[str, Any],
    source_field_wiring_audit: dict[str, Any],
    replay_validation: dict[str, Any],
) -> None:
    lines = [
        "# Connect Reward Component Source Fields Report",
        "",
        f"- Status: `{summary['status']}`",
        f"- Reward component source field status: `{summary['reward_component_source_field_status']}`",
        f"- Reason codes: `{summary['reason_codes']}`",
        f"- Audited rows: `{summary['audited_row_count']}`",
        f"- Connected rows: `{summary['connected_row_count']}`",
        f"- Expected/actual confusion: `{replay_validation['expected_actual_coverage_confusion_count']}`",
        f"- Fallback policy-gain contamination: `{replay_validation['fallback_coverage_gain_claimed_as_policy_gain_count']}`",
        f"- Controlled regression: `{replay_validation['controlled_regression_count']}`",
        "",
        "## Component Sources",
    ]
    for component, status in source_field_wiring_audit["component_status"].items():
        lines.append(
            "- `{component}`: status=`{source_status}`, present=`{present}`, missing=`{missing}`, sources=`{sources}`".format(
                component=component,
                source_status=status["source_status"],
                present=status["present_count"],
                missing=status["missing_count"],
                sources=status["source_fields"],
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This stage is read-only: it does not run PPO, publish a checkpoint, replace the default policy, or claim performance.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            _add_reason(reason_codes, f"{label}_invalid_jsonl")
            continue
        if isinstance(payload, dict):
            rows.append(payload)
        else:
            _add_reason(reason_codes, f"{label}_row_{line_number}_not_object")
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_optional_path(value: Any, base: Path, repo_root: Path) -> Path | None:
    if not value:
        return None
    return _resolve_path(Path(str(value)), base, repo_root)


def _resolve_path(path: Path, base: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    return repo_root / path


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason and reason not in reason_codes:
        reason_codes.append(reason)


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite(value: Any) -> bool:
    return _float(value) is not None


def _round_optional(value: Any) -> float | None:
    parsed = _float(value)
    if parsed is None:
        return None
    return round(float(parsed), 12)


def _is_fallback_like(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in FALLBACK_MARKERS or "fallback" in normalized


def _normalize_actor(value: Any) -> str:
    actor = str(value or "").strip()
    if actor in {"policy", "selected_policy", "ppo", "ppo_candidate", "selected_ppo_candidate"}:
        return "selected_ppo_candidate"
    if actor in {"source", "source_default_policy", "default", "default_source"}:
        return "source_default"
    if actor in {"source_fallback", "fallback"}:
        return "source_fallback"
    if actor in {"teacher", "teacher_fallback"}:
        return "teacher" if actor == "teacher" else "teacher_fallback"
    return actor or "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
