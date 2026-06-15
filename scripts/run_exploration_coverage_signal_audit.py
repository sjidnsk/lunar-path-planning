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


SUMMARY_SCHEMA_VERSION = "exploration-coverage-signal-audit-summary/v1"
FORMAL_SUMMARY_FILE = "formal-ppo-training-run-summary.json"
FORMAL_SEED_SUMMARIES_FILE = "formal-ppo-training-run-seed-summaries.jsonl"
PROMOTION_MANIFEST_FILE = "promotion-candidate-manifest.json"
SHADOW_SUMMARY_FILE = "multihorizon-shadow-rollout-summary.json"
SHADOW_STEPS_FILE = "multihorizon-shadow-rollout-steps.jsonl"

SUMMARY_FILE = "exploration-coverage-signal-audit-summary.json"
FIELD_AUDIT_FILE = "coverage-field-presence-audit.json"
DELTA_AUDIT_FILE = "coverage-delta-audit.jsonl"
ATTRIBUTION_AUDIT_FILE = "coverage-attribution-audit.json"
STATE_AUDIT_FILE = "coverage-state-transition-audit.json"
REJECTION_REPORT_FILE = "coverage-signal-rejection-report.json"
REPORT_FILE = "exploration-coverage-signal-audit-report.md"

VALID_GAIN_SOURCES = {"map", "sidecar", "path_feedback"}
TOLERANCE = 1e-9
PATH_FEEDBACK_COVERAGE_SUMMARIES = (
    Path("outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1/quasi-real-map-path-feedback-summary.json"),
    Path(
        "outputs/path_feedback_batch_quasi_real_teacher_distillation_dataset_v1/"
        "quasi-real-teacher-distillation-path-feedback-summary.json"
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit exploration coverage telemetry signal.")
    parser.add_argument("--formal-training-root", required=True)
    parser.add_argument("--selected-candidate-root", required=True)
    parser.add_argument("--shadow-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    summary = run_exploration_coverage_signal_audit(
        formal_training_root=_resolve_path(Path(args.formal_training_root), repo_root, repo_root),
        selected_candidate_root=_resolve_path(Path(args.selected_candidate_root), repo_root, repo_root),
        shadow_root=_resolve_path(Path(args.shadow_root), repo_root, repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root, repo_root),
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "coverage_signal_status": summary["coverage_signal_status"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_exploration_coverage_signal_audit(
    *,
    formal_training_root: Path,
    selected_candidate_root: Path,
    shadow_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    formal_training_root = Path(formal_training_root)
    selected_candidate_root = Path(selected_candidate_root)
    shadow_root = Path(shadow_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary": output_root / SUMMARY_FILE,
        "field_audit": output_root / FIELD_AUDIT_FILE,
        "delta_audit": output_root / DELTA_AUDIT_FILE,
        "attribution_audit": output_root / ATTRIBUTION_AUDIT_FILE,
        "state_audit": output_root / STATE_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }
    input_reasons: list[str] = []
    formal_summary_path = formal_training_root / FORMAL_SUMMARY_FILE
    seed_summaries_path = formal_training_root / FORMAL_SEED_SUMMARIES_FILE
    promotion_manifest_path = selected_candidate_root / PROMOTION_MANIFEST_FILE
    shadow_summary_path = shadow_root / SHADOW_SUMMARY_FILE

    formal_summary = _read_json(formal_summary_path, input_reasons, "formal_training_summary")
    seed_summaries = _read_jsonl(seed_summaries_path, input_reasons, "formal_seed_summaries")
    promotion_manifest = _read_json(
        promotion_manifest_path,
        input_reasons,
        "promotion_candidate_manifest",
        required=False,
    )
    shadow_summary = _read_json(shadow_summary_path, input_reasons, "shadow_summary", required=False)

    shadow_steps_path = _resolve_optional_path(
        promotion_manifest.get("multihorizon_steps"),
        selected_candidate_root,
        repo_root,
    ) or _resolve_optional_path(shadow_summary.get("steps"), shadow_root, repo_root) or (shadow_root / SHADOW_STEPS_FILE)
    shadow_steps = _read_jsonl(shadow_steps_path, input_reasons, "shadow_steps")

    upstream = _load_upstream_collector_transitions(seed_summaries, repo_root, input_reasons)
    audit_rows = _primary_audit_rows(shadow_steps, upstream["transition_rows"])
    coverage_index = _path_feedback_coverage_index(repo_root)
    audit_rows = _apply_path_feedback_coverage_backfill(audit_rows, coverage_index)
    audit_rows = _apply_backfilled_coverage_rollup(audit_rows)

    delta_rows = [_coverage_delta_row(row, index) for index, row in enumerate(audit_rows)]
    field_audit = _field_presence_audit(delta_rows)
    attribution_audit = _attribution_audit(delta_rows)
    state_audit = _state_transition_audit(delta_rows)
    rejection_report = _rejection_report(delta_rows, field_audit, attribution_audit, state_audit)

    reason_codes = list(input_reasons)
    _validate_inputs(formal_summary, seed_summaries, promotion_manifest, shadow_summary, reason_codes)
    _validate_coverage_signal(field_audit, attribution_audit, state_audit, rejection_report, reason_codes)

    nonzero_actual_coverage_delta_count = rejection_report["nonzero_actual_coverage_delta_count"]
    coverage_delta_default_zero_count = rejection_report["coverage_delta_default_zero_count"]
    expected_actual_coverage_confusion_count = rejection_report["expected_actual_coverage_confusion_count"]
    controlled_regression_count = sum(1 for row in audit_rows if _string_list(row.get("controlled_regression_reason_codes")))
    actual_coverage_gain_source = _actual_coverage_gain_source(delta_rows)
    coverage_field_presence_status = "passed" if field_audit["missing_required_field_count"] == 0 and field_audit["non_finite_field_count"] == 0 else "failed"
    attribution_status = attribution_audit["policy_source_fallback_coverage_attribution_status"]
    coverage_signal_status = (
        "passed"
        if (
            nonzero_actual_coverage_delta_count > 0
            and coverage_delta_default_zero_count == 0
            and expected_actual_coverage_confusion_count == 0
            and actual_coverage_gain_source in VALID_GAIN_SOURCES
            and coverage_field_presence_status == "passed"
            and state_audit["multi_step_state_update_verified"] is True
            and attribution_status == "passed"
            and controlled_regression_count == 0
            and not input_reasons
        )
        else "failed"
    )
    if coverage_signal_status == "failed" and nonzero_actual_coverage_delta_count == 0:
        _add_reason(reason_codes, "insufficient_exploration_coverage_signal")
    if controlled_regression_count:
        _add_reason(reason_codes, "controlled_regression_present")

    status = "passed" if not reason_codes and coverage_signal_status == "passed" else "failed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": "run_exploration_coverage_performance_evaluation"
        if status == "passed"
        else "connect_real_exploration_coverage_signal",
        "formal_training_root": str(formal_training_root),
        "selected_candidate_root": str(selected_candidate_root),
        "shadow_root": str(shadow_root),
        "output_root": str(output_root),
        "formal_training_summary": str(formal_summary_path),
        "formal_seed_summaries": str(seed_summaries_path),
        "promotion_candidate_manifest": str(promotion_manifest_path),
        "shadow_summary": str(shadow_summary_path),
        "shadow_steps": str(shadow_steps_path),
        "summary": str(paths["summary"]),
        "coverage_field_presence_audit": str(paths["field_audit"]),
        "coverage_delta_audit": str(paths["delta_audit"]),
        "coverage_attribution_audit": str(paths["attribution_audit"]),
        "coverage_state_transition_audit": str(paths["state_audit"]),
        "coverage_signal_rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "coverage_signal_status": coverage_signal_status,
        "coverage_field_presence_status": coverage_field_presence_status,
        "actual_coverage_gain_source": actual_coverage_gain_source,
        "nonzero_actual_coverage_delta_count": nonzero_actual_coverage_delta_count,
        "coverage_delta_default_zero_count": coverage_delta_default_zero_count,
        "expected_actual_coverage_confusion_count": expected_actual_coverage_confusion_count,
        "multi_step_state_update_verified": state_audit["multi_step_state_update_verified"],
        "policy_source_fallback_coverage_attribution_status": attribution_status,
        "fallback_coverage_gain_claimed_as_policy_gain_count": attribution_audit[
            "fallback_coverage_gain_claimed_as_policy_gain_count"
        ],
        "controlled_regression_count": controlled_regression_count,
        "audited_shadow_step_count": len(shadow_steps),
        "audited_upstream_transition_count": len(upstream["transition_rows"]),
        "audited_row_count": len(delta_rows),
        "input_artifact_paths": {
            "upstream_collector_episodes": upstream["episode_paths"],
        },
        "field_missing_counts": field_audit["missing_counts"],
        "rejection_reason_counts": rejection_report["rejection_reason_counts"],
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "runs_new_ppo_update": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }

    _write_json(paths["field_audit"], field_audit)
    _write_jsonl(paths["delta_audit"], delta_rows)
    _write_json(paths["attribution_audit"], attribution_audit)
    _write_json(paths["state_audit"], state_audit)
    _write_json(paths["rejection_report"], rejection_report)
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, rejection_report), encoding="utf-8")
    return summary


def _load_upstream_collector_transitions(
    seed_summaries: list[dict[str, Any]],
    repo_root: Path,
    reason_codes: list[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    episode_paths: list[str] = []
    for seed in seed_summaries:
        collector_root = _resolve_optional_path(seed.get("collector_root"), repo_root, repo_root)
        if collector_root is None:
            continue
        episodes_path = collector_root / "ppo-rollout-episodes.jsonl"
        if not episodes_path.exists():
            _add_reason(reason_codes, "upstream_collector_episodes_missing")
            continue
        episode_paths.append(str(episodes_path))
        for episode_index, episode in enumerate(_read_jsonl(episodes_path, reason_codes, "upstream_collector_episodes")):
            transitions = episode.get("transitions")
            if not isinstance(transitions, list):
                continue
            metrics = episode.get("metrics") if isinstance(episode.get("metrics"), dict) else {}
            for transition_index, transition in enumerate(transitions):
                info = transition.get("info") if isinstance(transition.get("info"), dict) else {}
                row = dict(info)
                if "observation" not in row and isinstance(transition.get("observation"), dict):
                    row["observation"] = transition["observation"]
                if "controlled_action_index" not in row and "action_index" in transition:
                    row["controlled_action_index"] = transition.get("action_index")
                row.setdefault("episode_id", episode.get("episode_id") or info.get("episode_id"))
                row.setdefault("step_index", info.get("step_index", transition_index))
                row["_upstream_episode_index"] = episode_index
                row["_upstream_transition_index"] = transition_index
                row["_upstream_metrics"] = metrics
                row["_coverage_artifact_source"] = "path_feedback"
                row["_source_artifact"] = str(episodes_path)
                rows.append(row)
    return {"transition_rows": rows, "episode_paths": episode_paths}


def _primary_audit_rows(shadow_steps: list[dict[str, Any]], upstream_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not shadow_steps:
        return [dict(row, _primary_source="upstream_collector_episode") for row in upstream_rows]
    upstream_by_key: dict[str, dict[str, Any]] = {}
    for upstream in upstream_rows:
        for key in _row_keys(upstream):
            upstream_by_key.setdefault(key, upstream)
    rows: list[dict[str, Any]] = []
    merge_keys = {
        "initial_coverage_rate",
        "final_coverage_rate",
        "coverage_rate_delta",
        "cumulative_coverage_rate_delta",
        "expected_coverage_rate_delta",
        "actual_coverage_gain_source",
        "coverage_gain_source",
        "coverage_signal_source",
        "coverage_gain_claimed_actor",
        "coverage_gain_actor",
        "actual_coverage_gain_actor",
    }
    for shadow in shadow_steps:
        merged = dict(shadow)
        for key in _row_keys(shadow):
            upstream = upstream_by_key.get(key)
            if not upstream:
                continue
            for field in merge_keys:
                if field not in merged and field in upstream:
                    merged[field] = upstream[field]
            if "observation" not in merged and isinstance(upstream.get("observation"), dict):
                merged["observation"] = upstream["observation"]
            merged["_upstream_source_artifact"] = upstream.get("_source_artifact")
            merged["_coverage_artifact_source"] = upstream.get("_coverage_artifact_source")
            break
        merged["_primary_source"] = "shadow_step"
        rows.append(merged)
    return rows


def _path_feedback_coverage_index(repo_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    index: dict[str, dict[str, dict[str, Any]]] = {"scenario": {}, "context": {}}
    cache: dict[str, dict[str, Any]] = {}
    for relative_path in PATH_FEEDBACK_COVERAGE_SUMMARIES:
        _index_path_feedback_summary(_resolve_path(relative_path, repo_root, repo_root), repo_root, index, cache)
    return index


def _index_path_feedback_summary(
    summary_path: Path,
    repo_root: Path,
    index: dict[str, dict[str, dict[str, Any]]],
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    key = str(summary_path)
    if key in cache:
        return cache[key]
    summary = _read_json_silent(summary_path)
    cache[key] = summary
    scenarios = summary.get("scenarios") if isinstance(summary.get("scenarios"), list) else []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        scenario_id = str(scenario.get("scenario_id") or "")
        signal = _coverage_signal_from_path_feedback_payload(scenario, summary_path)
        if not signal:
            signal = _coverage_signal_from_preferred_candidate_source(scenario, summary_path, repo_root, index, cache)
        if not signal:
            continue
        if scenario_id:
            index["scenario"].setdefault(scenario_id, signal)
        path_feedback = scenario.get("path_feedback") if isinstance(scenario.get("path_feedback"), dict) else {}
        candidates = path_feedback.get("candidates") if isinstance(path_feedback.get("candidates"), list) else []
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("context_id"):
                index["context"].setdefault(str(candidate["context_id"]), signal)
    return summary


def _coverage_signal_from_preferred_candidate_source(
    scenario: dict[str, Any],
    summary_path: Path,
    repo_root: Path,
    index: dict[str, dict[str, dict[str, Any]]],
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    path_feedback = scenario.get("path_feedback") if isinstance(scenario.get("path_feedback"), dict) else {}
    candidates = path_feedback.get("candidates") if isinstance(path_feedback.get("candidates"), list) else []
    preferred = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    source_path = preferred.get("source_path")
    source_scenario_id = preferred.get("scenario_id") or preferred.get("source_scenario_id")
    if not source_path or not source_scenario_id:
        return {}
    source_summary_path = _resolve_path(Path(str(source_path)), summary_path.parent, repo_root)
    source_summary = _index_path_feedback_summary(source_summary_path, repo_root, index, cache)
    source_scenarios = (
        source_summary.get("scenarios") if isinstance(source_summary.get("scenarios"), list) else []
    )
    for source_scenario in source_scenarios:
        if not isinstance(source_scenario, dict):
            continue
        if str(source_scenario.get("scenario_id")) != str(source_scenario_id):
            continue
        return _coverage_signal_from_path_feedback_payload(source_scenario, source_summary_path)
    return {}


def _coverage_signal_from_path_feedback_payload(payload: dict[str, Any], source_artifact: Path) -> dict[str, Any]:
    delta = _first_float(
        payload,
        (
            ("coverage_rate_delta",),
            ("actual_coverage_rate_delta",),
            ("coverage_gain",),
            ("actual_coverage_gain",),
            ("baseline_vs_feedback", "coverage_rate_delta"),
            ("path_feedback", "coverage_rate_delta"),
        ),
    )
    if delta is None:
        return {}
    signal: dict[str, Any] = {
        "coverage_rate_delta": delta,
        "coverage_gain_source": "path_feedback",
        "actual_coverage_gain_source": "path_feedback",
        "coverage_signal_source": "path_feedback",
        "coverage_gain_claimed_actor": "policy",
        "coverage_source_artifact": str(source_artifact),
        "coverage_source_scenario_id": payload.get("scenario_id"),
    }
    for field, candidates in {
        "initial_coverage_rate": (("initial_coverage_rate",), ("coverage_rate_before",)),
        "final_coverage_rate": (("final_coverage_rate",), ("coverage_rate_after",)),
        "cumulative_coverage_rate_delta": (("cumulative_coverage_rate_delta",),),
    }.items():
        value = _first_float(payload, candidates)
        if value is not None:
            signal[field] = value
    return signal


def _first_float(payload: dict[str, Any], candidates: tuple[tuple[str, ...], ...]) -> float | None:
    for path in candidates:
        value: Any = payload
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        parsed = _float(value)
        if parsed is not None and math.isfinite(parsed):
            return parsed
    return None


def _apply_path_feedback_coverage_backfill(
    rows: list[dict[str, Any]],
    coverage_index: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [_apply_path_feedback_coverage_to_row(row, coverage_index) for row in rows]


def _apply_path_feedback_coverage_to_row(
    row: dict[str, Any],
    coverage_index: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    signal = _coverage_signal_for_row(row, coverage_index)
    if not signal:
        return row
    delta, _ = _coverage_value(row, "coverage_rate_delta")
    final, _ = _coverage_value(row, "final_coverage_rate")
    cumulative, _ = _coverage_value(row, "cumulative_coverage_rate_delta")
    should_backfill = delta is None or (
        abs(delta) <= TOLERANCE and (final is None or cumulative is None) and abs(signal["coverage_rate_delta"]) > TOLERANCE
    )
    if not should_backfill:
        return row
    enriched = dict(row)
    for field, value in signal.items():
        enriched[field] = value
    enriched["_coverage_backfilled_from_path_feedback"] = True
    enriched["_coverage_artifact_source"] = "path_feedback"
    return enriched


def _coverage_signal_for_row(
    row: dict[str, Any],
    coverage_index: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    for field, namespace in (("scenario_id", "scenario"), ("context_id", "context"), ("source_context_id", "context")):
        value = row.get(field)
        if value is None:
            continue
        signal = coverage_index[namespace].get(str(value))
        if signal:
            return signal
    return {}


def _apply_backfilled_coverage_rollup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("_coverage_backfilled_from_path_feedback") is not True:
            continue
        episode_id = str(row.get("shadow_episode_id") or row.get("episode_id") or "__coverage_backfill__")
        grouped[episode_id].append((index, row))
    enriched_rows = [dict(row) for row in rows]
    for entries in grouped.values():
        entries.sort(key=lambda item: _int(item[1].get("shadow_step_index", item[1].get("step_index", item[0]))))
        running: float | None = None
        cumulative = 0.0
        for index, row in entries:
            delta = _float(row.get("coverage_rate_delta"))
            if delta is None or not math.isfinite(delta):
                continue
            if running is None:
                initial = _float(row.get("initial_coverage_rate"))
                running = initial if initial is not None and math.isfinite(initial) else 0.0
            initial = running
            final = round(initial + delta, 12)
            cumulative = round(cumulative + delta, 12)
            running = final
            enriched_rows[index].update(
                {
                    "initial_coverage_rate": round(initial, 12),
                    "final_coverage_rate": final,
                    "coverage_rate_delta": delta,
                    "cumulative_coverage_rate_delta": cumulative,
                    "coverage_gain_source": row.get("coverage_gain_source") or "path_feedback",
                    "actual_coverage_gain_source": row.get("actual_coverage_gain_source") or "path_feedback",
                    "coverage_signal_source": row.get("coverage_signal_source") or "path_feedback",
                    "coverage_gain_claimed_actor": row.get("coverage_gain_claimed_actor") or _canonical_actor(
                        str(row.get("controlled_choice_source") or "")
                    ),
                    "coverage_state_rollup_source": "path_feedback_actual_delta",
                }
            )
    return enriched_rows


def _coverage_delta_row(row: dict[str, Any], audit_index: int) -> dict[str, Any]:
    initial, initial_source = _coverage_value(row, "initial_coverage_rate")
    final, final_source = _coverage_value(row, "final_coverage_rate")
    delta, delta_source = _coverage_value(row, "coverage_rate_delta")
    cumulative, cumulative_source = _coverage_value(row, "cumulative_coverage_rate_delta")
    expected, expected_source = _coverage_value(row, "expected_coverage_rate_delta")
    choice_source = str(row.get("controlled_choice_source") or row.get("choice_source") or "unknown")
    canonical_actor = _canonical_actor(choice_source)
    claimed_actor = str(
        row.get("coverage_gain_claimed_actor")
        or row.get("coverage_gain_actor")
        or row.get("actual_coverage_gain_actor")
        or canonical_actor
    )
    gain_source = str(
        row.get("actual_coverage_gain_source")
        or row.get("coverage_gain_source")
        or row.get("coverage_signal_source")
        or row.get("_coverage_artifact_source")
        or ""
    )
    if not gain_source and _finite(delta) and abs(float(delta)) > TOLERANCE:
        gain_source = "missing"
    if not gain_source and _finite(delta) and abs(float(delta)) <= TOLERANCE:
        gain_source = "default_zero"

    return {
        "schema_version": "exploration-coverage-delta-audit-row/v1",
        "audit_index": audit_index,
        "episode_id": str(row.get("shadow_episode_id") or row.get("episode_id") or ""),
        "source_episode_id": str(row.get("episode_id") or ""),
        "step_index": _int(row.get("shadow_step_index", row.get("step_index", audit_index))),
        "source_step_index": _int(row.get("step_index", audit_index)),
        "context_id": row.get("context_id"),
        "scenario_id": row.get("scenario_id"),
        "scenario_family": row.get("scenario_family") or row.get("scenario_group"),
        "split": row.get("split"),
        "primary_source": row.get("_primary_source"),
        "source_artifact": row.get("_upstream_source_artifact") or row.get("_source_artifact"),
        "controlled_choice_source": choice_source,
        "controlled_choice_detail": row.get("controlled_choice_detail"),
        "canonical_coverage_actor": canonical_actor,
        "coverage_gain_claimed_actor": claimed_actor,
        "ppo_trainable": row.get("ppo_trainable") is True or row.get("shadow_trainable") is True,
        "initial_coverage_rate": initial,
        "initial_coverage_rate_source": initial_source,
        "final_coverage_rate": final,
        "final_coverage_rate_source": final_source,
        "coverage_rate_delta": delta,
        "coverage_rate_delta_source": delta_source,
        "cumulative_coverage_rate_delta": cumulative,
        "cumulative_coverage_rate_delta_source": cumulative_source,
        "expected_coverage_rate_delta": expected,
        "expected_coverage_rate_delta_source": expected_source,
        "actual_coverage_gain_source": gain_source or "missing",
        "actual_delta_present": delta is not None,
        "actual_delta_finite": _finite(delta),
        "actual_delta_nonzero": _finite(delta) and abs(float(delta)) > TOLERANCE,
        "actual_delta_default_zero": _finite(delta) and abs(float(delta)) <= TOLERANCE,
        "expected_delta_nonzero": _finite(expected) and abs(float(expected)) > TOLERANCE,
        "controlled_regression_reason_codes": _string_list(row.get("controlled_regression_reason_codes")),
    }


def _field_presence_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "initial_coverage_rate",
        "final_coverage_rate",
        "coverage_rate_delta",
        "cumulative_coverage_rate_delta",
        "expected_coverage_rate_delta",
    ]
    present_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    non_finite_counts: Counter[str] = Counter()
    for row in rows:
        for field in fields:
            value = row.get(field)
            if value is None:
                missing_counts[field] += 1
            else:
                present_counts[field] += 1
                if not _finite(value):
                    non_finite_counts[field] += 1
    return {
        "schema_version": "coverage-field-presence-audit/v1",
        "row_count": len(rows),
        "fields": fields,
        "present_counts": dict(sorted(present_counts.items())),
        "missing_counts": {field: int(missing_counts.get(field, 0)) for field in fields},
        "non_finite_counts": {field: int(non_finite_counts.get(field, 0)) for field in fields},
        "missing_required_field_count": sum(missing_counts.values()),
        "non_finite_field_count": sum(non_finite_counts.values()),
    }


def _attribution_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actor_counts: Counter[str] = Counter()
    gain_by_actor: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    fallback_claimed_policy = 0
    unknown_actor_count = 0
    for row in rows:
        actor = str(row.get("canonical_coverage_actor") or "unknown")
        claimed_actor = str(row.get("coverage_gain_claimed_actor") or actor)
        choice_source = str(row.get("controlled_choice_source") or "")
        source_counts[str(row.get("actual_coverage_gain_source") or "missing")] += 1
        actor_counts[actor] += 1
        if row.get("actual_delta_nonzero"):
            gain_by_actor[actor] += 1
        if actor == "unknown":
            unknown_actor_count += 1
        if choice_source in {"source_fallback", "fallback", "teacher_fallback"} and claimed_actor in {
            "policy",
            "selected_ppo_candidate",
            "ppo_policy",
        }:
            fallback_claimed_policy += 1
    status = "passed" if fallback_claimed_policy == 0 and unknown_actor_count == 0 else "failed"
    return {
        "schema_version": "coverage-attribution-audit/v1",
        "policy_source_fallback_coverage_attribution_status": status,
        "actor_counts": dict(sorted(actor_counts.items())),
        "nonzero_gain_counts_by_actor": dict(sorted(gain_by_actor.items())),
        "actual_coverage_gain_source_counts": dict(sorted(source_counts.items())),
        "fallback_coverage_gain_claimed_as_policy_gain_count": fallback_claimed_policy,
        "unknown_coverage_actor_count": unknown_actor_count,
    }


def _state_transition_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[str(row.get("episode_id") or "")].append(row)
    episode_reports: list[dict[str, Any]] = []
    cumulative_mismatch_count = 0
    state_link_mismatch_count = 0
    static_state_episode_count = 0
    multistep_episode_count = 0
    for episode_id, episode_rows in by_episode.items():
        ordered = sorted(episode_rows, key=lambda item: _int(item.get("step_index")))
        if len(ordered) > 1:
            multistep_episode_count += 1
        running = 0.0
        episode_cumulative_mismatch = 0
        episode_state_mismatch = 0
        coverage_states: list[tuple[Any, Any]] = []
        for index, row in enumerate(ordered):
            delta = row.get("coverage_rate_delta")
            if _finite(delta):
                running += float(delta)
            cumulative = row.get("cumulative_coverage_rate_delta")
            if _finite(cumulative) and abs(float(cumulative) - running) > TOLERANCE:
                cumulative_mismatch_count += 1
                episode_cumulative_mismatch += 1
            initial = row.get("initial_coverage_rate")
            final = row.get("final_coverage_rate")
            coverage_states.append((initial, final))
            if index > 0:
                previous_final = ordered[index - 1].get("final_coverage_rate")
                if _finite(previous_final) and _finite(initial) and abs(float(previous_final) - float(initial)) > TOLERANCE:
                    state_link_mismatch_count += 1
                    episode_state_mismatch += 1
        if len(ordered) > 1 and len(set(coverage_states)) == 1:
            static_state_episode_count += 1
        episode_reports.append(
            {
                "episode_id": episode_id,
                "step_count": len(ordered),
                "cumulative_delta_sum": running,
                "cumulative_mismatch_count": episode_cumulative_mismatch,
                "state_link_mismatch_count": episode_state_mismatch,
                "static_coverage_state": len(ordered) > 1 and len(set(coverage_states)) == 1,
            }
        )
    verified = (
        bool(rows)
        and multistep_episode_count > 0
        and cumulative_mismatch_count == 0
        and state_link_mismatch_count == 0
        and static_state_episode_count == 0
        and all(row.get("initial_coverage_rate") is not None and row.get("final_coverage_rate") is not None for row in rows)
    )
    return {
        "schema_version": "coverage-state-transition-audit/v1",
        "episode_count": len(by_episode),
        "multistep_episode_count": multistep_episode_count,
        "cumulative_coverage_delta_mismatch_count": cumulative_mismatch_count,
        "state_link_mismatch_count": state_link_mismatch_count,
        "static_coverage_state_episode_count": static_state_episode_count,
        "multi_step_state_update_verified": verified,
        "episodes": episode_reports,
    }


def _rejection_report(
    rows: list[dict[str, Any]],
    field_audit: dict[str, Any],
    attribution_audit: dict[str, Any],
    state_audit: dict[str, Any],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    nonzero = sum(1 for row in rows if row.get("actual_delta_nonzero"))
    zero_default = sum(1 for row in rows if row.get("ppo_trainable") and row.get("actual_delta_default_zero"))
    missing_actual = sum(1 for row in rows if row.get("coverage_rate_delta") is None)
    confusion = sum(
        1
        for row in rows
        if row.get("expected_delta_nonzero") and (row.get("coverage_rate_delta") is None or row.get("actual_delta_default_zero"))
    )
    if not rows:
        counts["coverage_audit_rows_missing"] += 1
    if nonzero == 0:
        if missing_actual:
            counts["actual_coverage_delta_missing"] += missing_actual
        else:
            counts["actual_coverage_delta_all_zero"] += len(rows)
    if zero_default:
        counts["coverage_delta_default_zero"] += zero_default
    if confusion:
        counts["expected_actual_coverage_confusion"] += confusion
    if field_audit["missing_required_field_count"]:
        counts["coverage_required_fields_missing"] += field_audit["missing_required_field_count"]
    if field_audit["non_finite_field_count"]:
        counts["coverage_non_finite_fields"] += field_audit["non_finite_field_count"]
    if attribution_audit["fallback_coverage_gain_claimed_as_policy_gain_count"]:
        counts["fallback_coverage_gain_claimed_as_policy_gain"] += attribution_audit[
            "fallback_coverage_gain_claimed_as_policy_gain_count"
        ]
    if state_audit["cumulative_coverage_delta_mismatch_count"]:
        counts["cumulative_coverage_delta_mismatch"] += state_audit["cumulative_coverage_delta_mismatch_count"]
    if not state_audit["multi_step_state_update_verified"]:
        counts["multi_step_state_not_updated"] += 1
    invalid_source_count = sum(
        1
        for row in rows
        if row.get("actual_delta_nonzero") and row.get("actual_coverage_gain_source") not in VALID_GAIN_SOURCES
    )
    if invalid_source_count:
        counts["actual_coverage_gain_source_invalid"] += invalid_source_count
    return {
        "schema_version": "coverage-signal-rejection-report/v1",
        "status": "passed" if not counts else "failed",
        "rejection_reason_counts": dict(sorted(counts.items())),
        "nonzero_actual_coverage_delta_count": nonzero,
        "coverage_delta_default_zero_count": zero_default,
        "actual_coverage_delta_missing_count": missing_actual,
        "expected_actual_coverage_confusion_count": confusion,
        "sample_rejected_rows": [
            row
            for row in rows
            if row.get("coverage_rate_delta") is None
            or row.get("actual_delta_default_zero")
            or row.get("expected_delta_nonzero") and (row.get("coverage_rate_delta") is None or row.get("actual_delta_default_zero"))
        ][:20],
    }


def _validate_inputs(
    formal_summary: dict[str, Any],
    seed_summaries: list[dict[str, Any]],
    promotion_manifest: dict[str, Any],
    shadow_summary: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if formal_summary.get("status") != "passed":
        _add_reason(reason_codes, "formal_training_summary_not_passed")
    if _string_list(formal_summary.get("reason_codes")):
        _add_reason(reason_codes, "formal_training_summary_has_reason_codes")
    if not seed_summaries:
        _add_reason(reason_codes, "formal_seed_summaries_missing")
    if promotion_manifest and promotion_manifest.get("experimental_candidate_only") is not True:
        _add_reason(reason_codes, "selected_candidate_not_marked_experimental_only")
    if shadow_summary and shadow_summary.get("status") not in {None, "passed"}:
        _add_reason(reason_codes, "shadow_summary_not_passed")
    for source in (formal_summary, promotion_manifest, shadow_summary):
        if not source:
            continue
        for field, reason in (
            ("publishes_checkpoint", "unexpected_checkpoint_publication_claimed"),
            ("replaces_default_policy", "unexpected_default_policy_replacement_claimed"),
            ("performance_claimed", "unexpected_performance_claimed"),
            ("formal_training_ready_claimed", "unexpected_formal_ready_claimed"),
        ):
            if source.get(field) is True:
                _add_reason(reason_codes, reason)


def _validate_coverage_signal(
    field_audit: dict[str, Any],
    attribution_audit: dict[str, Any],
    state_audit: dict[str, Any],
    rejection_report: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if field_audit["missing_required_field_count"] or field_audit["non_finite_field_count"]:
        _add_reason(reason_codes, "coverage_field_presence_audit_failed")
    if rejection_report["coverage_delta_default_zero_count"]:
        _add_reason(reason_codes, "coverage_delta_default_zero_present")
    if rejection_report["expected_actual_coverage_confusion_count"]:
        _add_reason(reason_codes, "expected_actual_coverage_confusion")
    if attribution_audit["fallback_coverage_gain_claimed_as_policy_gain_count"]:
        _add_reason(reason_codes, "fallback_coverage_gain_claimed_as_policy_gain")
    if attribution_audit["policy_source_fallback_coverage_attribution_status"] != "passed":
        _add_reason(reason_codes, "coverage_attribution_audit_failed")
    if state_audit["cumulative_coverage_delta_mismatch_count"]:
        _add_reason(reason_codes, "cumulative_coverage_delta_mismatch")
    if not state_audit["multi_step_state_update_verified"]:
        _add_reason(reason_codes, "multi_step_state_not_updated")
    if rejection_report["rejection_reason_counts"].get("actual_coverage_gain_source_invalid"):
        _add_reason(reason_codes, "actual_coverage_gain_source_invalid")


def _coverage_value(row: dict[str, Any], field: str) -> tuple[float | None, str | None]:
    for candidate in _field_candidates(field):
        value = _nested_get(row, candidate)
        if value is not None:
            return (_float(value), ".".join(candidate))
    if field == "initial_coverage_rate":
        value = _feature_value(row.get("observation"), "global_feature_names", "global_features", "coverage_rate")
        if value is not None:
            return (_float(value), "observation.global_features.coverage_rate")
    if field == "expected_coverage_rate_delta":
        value = _selected_candidate_feature(row, "expected_coverage_rate_delta")
        if value is not None:
            return (_float(value), "observation.candidate_features.expected_coverage_rate_delta")
    return (None, None)


def _field_candidates(field: str) -> list[tuple[str, ...]]:
    return {
        "initial_coverage_rate": [
            ("initial_coverage_rate",),
            ("info", "initial_coverage_rate"),
            ("metrics", "initial_coverage_rate"),
        ],
        "final_coverage_rate": [
            ("final_coverage_rate",),
            ("info", "final_coverage_rate"),
            ("metrics", "final_coverage_rate"),
            ("_upstream_metrics", "final_coverage_rate"),
        ],
        "coverage_rate_delta": [
            ("coverage_rate_delta",),
            ("actual_coverage_rate_delta",),
            ("coverage_gain",),
            ("actual_coverage_gain",),
            ("info", "coverage_rate_delta"),
        ],
        "cumulative_coverage_rate_delta": [
            ("cumulative_coverage_rate_delta",),
            ("info", "cumulative_coverage_rate_delta"),
            ("metrics", "cumulative_coverage_rate_delta"),
            ("_upstream_metrics", "cumulative_coverage_rate_delta"),
        ],
        "expected_coverage_rate_delta": [
            ("expected_coverage_rate_delta",),
            ("info", "expected_coverage_rate_delta"),
        ],
    }[field]


def _selected_candidate_feature(row: dict[str, Any], name: str) -> float | None:
    observation = row.get("observation")
    if not isinstance(observation, dict):
        return None
    names = observation.get("candidate_feature_names")
    features = observation.get("candidate_features")
    if not isinstance(names, list) or not isinstance(features, list) or name not in names:
        return None
    action_index = _int(row.get("controlled_action_index", row.get("action_index", 0)))
    if action_index < 0 or action_index >= len(features) or not isinstance(features[action_index], list):
        return None
    feature_index = names.index(name)
    if feature_index >= len(features[action_index]):
        return None
    return _float(features[action_index][feature_index])


def _feature_value(
    observation: Any,
    names_field: str,
    values_field: str,
    name: str,
) -> float | None:
    if not isinstance(observation, dict):
        return None
    names = observation.get(names_field)
    values = observation.get(values_field)
    if not isinstance(names, list) or not isinstance(values, list) or name not in names:
        return None
    index = names.index(name)
    if index >= len(values):
        return None
    return _float(values[index])


def _actual_coverage_gain_source(rows: list[dict[str, Any]]) -> str:
    sources = sorted(
        {
            str(row.get("actual_coverage_gain_source"))
            for row in rows
            if row.get("actual_delta_nonzero") and row.get("actual_coverage_gain_source") in VALID_GAIN_SOURCES
        }
    )
    if len(sources) == 1:
        return sources[0]
    if len(sources) > 1:
        return "mixed"
    if any(row.get("actual_delta_default_zero") for row in rows):
        return "default_zero"
    return "none"


def _canonical_actor(choice_source: str) -> str:
    if choice_source == "policy":
        return "selected_ppo_candidate"
    if choice_source in {"source", "source_default", "default_policy"}:
        return "source_default_policy"
    if choice_source in {"source_fallback", "fallback"}:
        return "source_fallback"
    if choice_source in {"teacher", "teacher_fallback"}:
        return "teacher"
    return "unknown"


def _row_keys(row: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    context_id = row.get("context_id")
    if context_id:
        keys.append(f"context:{context_id}")
    episode_id = row.get("episode_id") or row.get("source_episode_id")
    step_index = row.get("step_index")
    if episode_id is not None and step_index is not None:
        keys.append(f"episode-step:{episode_id}:{step_index}")
    return keys


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _read_json(path: Path, reason_codes: list[str], label: str, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            _add_reason(reason_codes, f"{label}_missing")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _add_reason(reason_codes, f"{label}_invalid_json")
        return {}


def _read_json_silent(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
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


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _render_report(summary: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    lines = [
        "# Exploration Coverage Signal Audit v1",
        "",
        f"Status: `{summary['status']}`",
        f"Coverage signal status: `{summary['coverage_signal_status']}`",
        f"Reason codes: `{summary['reason_codes']}`",
        "",
        "## Key Counts",
        "",
        f"- Audited rows: {summary['audited_row_count']}",
        f"- Nonzero actual coverage deltas: {summary['nonzero_actual_coverage_delta_count']}",
        f"- Default-zero trainable coverage deltas: {summary['coverage_delta_default_zero_count']}",
        f"- Expected/actual confusion count: {summary['expected_actual_coverage_confusion_count']}",
        f"- Multi-step state update verified: {summary['multi_step_state_update_verified']}",
        f"- Actual coverage gain source: `{summary['actual_coverage_gain_source']}`",
        "",
        "## Rejection Reason Counts",
        "",
    ]
    for reason, count in rejection_report["rejection_reason_counts"].items():
        lines.append(f"- `{reason}`: {count}")
    if not rejection_report["rejection_reason_counts"]:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This audit does not run PPO, publish checkpoints, replace the default policy, "
            "or claim policy performance.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
