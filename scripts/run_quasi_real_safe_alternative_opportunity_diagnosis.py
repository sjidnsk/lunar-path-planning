from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot

from run_scenario_disjoint_policy_rollout_evaluation import _display_path


CONFIG_SCHEMA_VERSION = "quasi-real-safe-alternative-opportunity-diagnosis-config/v1"
SUMMARY_SCHEMA_VERSION = "quasi-real-safe-alternative-opportunity-summary/v1"
DIAGNOSTIC_SCHEMA_VERSION = "quasi-real-safe-alternative-opportunity-diagnostic/v1"
EXCLUSION_SCHEMA_VERSION = "quasi-real-safe-alternative-opportunity-exclusion-report/v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_quasi_real_safe_alternative_opportunity_diagnosis_v1"


class ConfigError(ValueError):
    pass


def run_quasi_real_safe_alternative_opportunity_diagnosis(
    *,
    quasi_real_root: str | Path,
    guarded_pilot_root: str | Path,
    alignment_root: str | Path,
    output_root: str | Path,
    config: dict[str, Any],
    repo_root: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    quasi = Path(quasi_real_root).resolve()
    guarded = Path(guarded_pilot_root).resolve()
    alignment = Path(alignment_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)

    reason_codes: list[str] = []
    paths = _input_paths(quasi, guarded, alignment, config)
    quasi_summary = _load_json(paths["quasi_real_path_feedback_summary"], reason_codes, "quasi_real_path_feedback_summary")
    domain_gap = _load_json(paths["quasi_real_domain_gap_summary"], reason_codes, "quasi_real_domain_gap_summary")
    alignment_summary = _load_json(paths["alignment_summary"], reason_codes, "alignment_summary")
    guarded_summary = _load_json(paths["guarded_pilot_summary"], reason_codes, "guarded_pilot_summary")
    slices = _load_jsonl(paths["quasi_real_slices"], reason_codes, "quasi_real_slices")
    guarded_decisions = _load_jsonl(paths["guarded_decisions"], reason_codes, "guarded_decisions")

    _validate_inputs(domain_gap, alignment_summary, guarded_summary, reason_codes)
    slice_by_scenario = {str(item.get("scenario_id")): item for item in slices}
    decision_by_scenario = {str(item.get("scenario_id")): item for item in guarded_decisions}

    diagnostics: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for scenario in quasi_summary.get("scenarios", []) if isinstance(quasi_summary.get("scenarios"), list) else []:
        if not isinstance(scenario, dict):
            continue
        diagnostic, exclusion = _diagnose_scenario(
            scenario=scenario,
            slice_record=slice_by_scenario.get(str(scenario.get("scenario_id")), {}),
            guarded_decision=decision_by_scenario.get(str(scenario.get("scenario_id")), {}),
            config=config,
        )
        if exclusion:
            exclusions.append(exclusion)
        if diagnostic:
            diagnostics.append(diagnostic)

    summary = _summary(
        quasi_real_root=quasi,
        guarded_pilot_root=guarded,
        alignment_root=alignment,
        output_root=output,
        diagnostics=diagnostics,
        exclusions=exclusions,
        reason_codes=reason_codes,
        config=config,
        repo_root=repo,
    )
    exclusion_report = {
        "schema_version": EXCLUSION_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "status": "passed" if not exclusions else "failed",
        "opportunity_exclusion_count": len(exclusions),
        "exclusion_reason_counts": dict(sorted(Counter(row["reason_code"] for row in exclusions).items())),
        "exclusions": exclusions,
    }
    output_paths = _output_paths(output, config)
    output_paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    output_paths["diagnostics"].write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in diagnostics),
        encoding="utf-8",
    )
    output_paths["exclusion_report"].write_text(
        json.dumps(exclusion_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_paths["report"].write_text(_report(summary), encoding="utf-8")
    summary["summary_output"] = str(output_paths["summary"])
    return summary


def _diagnose_scenario(
    *,
    scenario: dict[str, Any],
    slice_record: dict[str, Any],
    guarded_decision: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    scenario_id = str(scenario.get("scenario_id") or "")
    path_feedback = scenario.get("path_feedback") if isinstance(scenario.get("path_feedback"), dict) else {}
    candidates = path_feedback.get("candidates") if isinstance(path_feedback.get("candidates"), list) else []
    if not scenario_id or not candidates:
        return None, _exclusion(scenario_id, "bridge_or_feedback_gap", "scenario_or_candidates_missing")
    if any(not isinstance(candidate, dict) or not candidate.get("context_id") for candidate in candidates):
        return None, _exclusion(scenario_id, "bridge_or_feedback_gap", "candidate_context_id_missing")

    source = _source_candidate(candidates, path_feedback, guarded_decision)
    if source is None:
        return None, _exclusion(scenario_id, "bridge_or_feedback_gap", "source_candidate_missing")

    thresholds = config.get("thresholds", {})
    source_action = _int(source.get("action_index"), -1)
    raw_policy_action = _int(guarded_decision.get("raw_policy_action_index"), source_action)
    alternatives = [
        candidate for candidate in candidates if _int(candidate.get("action_index"), -9999) != source_action
    ]
    candidate_diagnostics = [
        _candidate_funnel(candidate, source=source, thresholds=thresholds)
        for candidate in alternatives
    ]
    safe = [row for row in candidate_diagnostics if row["safe_alternative"]]
    safe_better = [row for row in candidate_diagnostics if row["safe_better_alternative"]]
    selected_safe_better = any(
        row["action_index"] == raw_policy_action for row in safe_better
    )
    if safe_better:
        opportunity_class = (
            "safe_better_opportunity_policy_selected"
            if selected_safe_better
            else "safe_better_opportunity_exists_policy_source_aligned"
        )
    elif safe:
        opportunity_class = "safe_alternative_exists_but_not_better"
    else:
        opportunity_class = "opportunity_missing"

    funnel_reason_counts = Counter(
        reason for row in candidate_diagnostics for reason in row["gate_reason_codes"]
    )
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "context_id": source.get("context_id"),
        "roi_group": slice_record.get("roi_group") or slice_record.get("roi_name") or scenario.get("scenario_group"),
        "roi_name": slice_record.get("roi_name") or scenario.get("scenario_group"),
        "split": slice_record.get("split"),
        "map_id": slice_record.get("map_id"),
        "slice_id": slice_record.get("slice_id") or scenario_id,
        "start_cell": _normal_cell(slice_record.get("start_cell")),
        "source_action_index": source_action,
        "raw_policy_action_index": raw_policy_action,
        "candidate_count": len(candidates),
        "alternative_count": len(alternatives),
        "safe_alternative_count": len(safe),
        "safe_better_alternative_count": len(safe_better),
        "policy_selected_safe_better": selected_safe_better,
        "opportunity_class": opportunity_class,
        "candidate_funnel": candidate_diagnostics,
        "invalid_action_mask_count": 0,
        "fallback_or_open_grid_count": 0,
        "safety_regression_count": 0,
        "contract_violation_count": 0,
        "path_cost_regression_count": 0,
        "risk_regression_count": 0,
        "source_selection_regression_count": 0,
        "funnel_invalid_action_mask_count": funnel_reason_counts.get("invalid_action_mask", 0),
        "funnel_unreachable_count": funnel_reason_counts.get("unreachable", 0),
        "funnel_replan_required_count": funnel_reason_counts.get("replan_required", 0),
        "funnel_fallback_or_open_grid_count": funnel_reason_counts.get("fallback_or_open_grid", 0),
        "funnel_contract_violation_count": funnel_reason_counts.get("contract_violation", 0),
        "funnel_path_cost_regression_count": funnel_reason_counts.get("path_cost_regression", 0),
        "funnel_risk_regression_count": funnel_reason_counts.get("risk_regression", 0),
        "funnel_source_selection_regression_count": funnel_reason_counts.get("source_selection_regression", 0),
    }, None


def _candidate_funnel(candidate: dict[str, Any], *, source: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    action_index = _int(candidate.get("action_index"), -1)
    path_delta = _float(candidate.get("path_cost")) - _float(source.get("path_cost"))
    risk_delta = _float(candidate.get("risk")) - _float(source.get("risk"))
    utility_delta = _float(candidate.get("utility")) - _float(source.get("utility"))
    gate_reasons: list[str] = []
    if not _action_mask_valid(candidate):
        gate_reasons.append("invalid_action_mask")
    if not bool(candidate.get("reachable", True)):
        gate_reasons.append("unreachable")
    if bool(candidate.get("replan_required")):
        gate_reasons.append("replan_required")
    if bool(candidate.get("open_grid_fallback_used")) or bool(candidate.get("fallback_or_open_grid_used")):
        gate_reasons.append("fallback_or_open_grid")
    if not _contract_safe(candidate):
        gate_reasons.append("contract_violation")
    if path_delta > _float(thresholds.get("max_path_cost_regression"), 0.0):
        gate_reasons.append("path_cost_regression")
    if risk_delta > _float(thresholds.get("max_risk_regression"), 0.0):
        gate_reasons.append("risk_regression")
    if _source_selection_regression(candidate):
        gate_reasons.append("source_selection_regression")
    safe = not gate_reasons
    better = (
        path_delta <= _float(thresholds.get("better_path_cost_delta"), -0.25)
        or risk_delta <= _float(thresholds.get("better_risk_delta"), -0.01)
        or utility_delta >= _float(thresholds.get("better_utility_delta"), 0.005)
    )
    return {
        "action_index": action_index,
        "context_id": candidate.get("context_id"),
        "path_cost_delta": path_delta,
        "risk_delta": risk_delta,
        "utility_delta": utility_delta,
        "candidate_present": True,
        "action_mask_valid": "invalid_action_mask" not in gate_reasons,
        "reachable": bool(candidate.get("reachable", True)),
        "no_replan": not bool(candidate.get("replan_required")),
        "no_fallback_or_open_grid": "fallback_or_open_grid" not in gate_reasons,
        "contract_safe": "contract_violation" not in gate_reasons,
        "path_cost_no_regression": "path_cost_regression" not in gate_reasons,
        "risk_no_regression": "risk_regression" not in gate_reasons,
        "source_selection_no_regression": "source_selection_regression" not in gate_reasons,
        "safe_alternative": safe,
        "safe_better_alternative": safe and better,
        "gate_reason_codes": gate_reasons,
    }


def _summary(
    *,
    quasi_real_root: Path,
    guarded_pilot_root: Path,
    alignment_root: Path,
    output_root: Path,
    diagnostics: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    reason_codes: list[str],
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    validation = config.get("validation", {})
    counts = Counter(row["opportunity_class"] for row in diagnostics)
    roi_groups = sorted({str(row.get("roi_group")) for row in diagnostics if row.get("roi_group")})
    context_id_missing_count = len(exclusions)
    if len(diagnostics) < _int(validation.get("min_quasi_real_context_count")):
        _append_reason(reason_codes, "quasi_real_context_count_below_threshold")
    if len(roi_groups) < _int(validation.get("min_roi_group_count")):
        _append_reason(reason_codes, "quasi_real_roi_group_count_below_threshold")
    if len(exclusions) > _int(validation.get("max_opportunity_exclusion_count"), 0):
        _append_reason(reason_codes, "quasi_real_opportunity_exclusion_count_above_threshold")

    metric_counts = _metric_counts(diagnostics)
    if context_id_missing_count:
        _append_reason(reason_codes, "quasi_real_safe_alternative_context_id_missing")
    for field, reason in (
        ("invalid_action_mask_count", "real_map_action_mask_contract_gap"),
        ("fallback_or_open_grid_count", "real_map_bridge_or_feedback_gap"),
        ("safety_regression_count", "real_map_bridge_or_feedback_gap"),
        ("contract_violation_count", "real_map_action_mask_contract_gap"),
        ("path_cost_regression_count", "real_map_bridge_or_feedback_gap"),
        ("risk_regression_count", "real_map_bridge_or_feedback_gap"),
        ("source_selection_regression_count", "real_map_bridge_or_feedback_gap"),
    ):
        if metric_counts[field] and field in {"invalid_action_mask_count", "contract_violation_count"}:
            _append_reason(reason_codes, reason)

    verdict = _opportunity_verdict(reason_codes, counts)
    status = "failed" if _hard_failure(reason_codes) else "passed"
    funnel_metric_counts = _funnel_metric_counts(diagnostics)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "status": status,
        "reason_codes": _unique(reason_codes),
        "opportunity_verdict": verdict,
        "next_required_change": _next_required_change(verdict, counts),
        "quasi_real_root": _display_path(quasi_real_root, repo_root),
        "guarded_pilot_root": _display_path(guarded_pilot_root, repo_root),
        "alignment_root": _display_path(alignment_root, repo_root),
        "output_root": _display_path(output_root, repo_root),
        "quasi_real_context_count": len(diagnostics),
        "policy_decision_count": len(diagnostics),
        "roi_group_count": len(roi_groups),
        "roi_groups": roi_groups,
        "safe_alternative_context_count": sum(1 for row in diagnostics if row["safe_alternative_count"] > 0),
        "safe_better_opportunity_context_count": sum(1 for row in diagnostics if row["safe_better_alternative_count"] > 0),
        "safe_better_alternative_count": sum(int(row["safe_better_alternative_count"]) for row in diagnostics),
        "roi_group_with_safe_better_opportunity_count": sum(
            1
            for row in _roi_group_summary(diagnostics).values()
            if int(row.get("safe_better_opportunity_context_count", 0)) > 0
        ),
        "policy_missed_safe_better_opportunity_count": counts.get(
            "safe_better_opportunity_exists_policy_source_aligned", 0
        ),
        "policy_selected_safe_better_opportunity_count": counts.get(
            "safe_better_opportunity_policy_selected", 0
        ),
        "opportunity_missing_count": counts.get("opportunity_missing", 0),
        "safe_alternative_exists_but_not_better_count": counts.get(
            "safe_alternative_exists_but_not_better", 0
        ),
        "opportunity_class_counts": dict(sorted(counts.items())),
        "roi_group_opportunity_summary": _roi_group_summary(diagnostics),
        "context_id_missing_count": context_id_missing_count,
        "opportunity_exclusion_count": len(exclusions),
        **metric_counts,
        **funnel_metric_counts,
        "runs_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "git_provenance": {"current": _git_snapshot(repo_root), "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }


def _input_paths(quasi: Path, guarded: Path, alignment: Path, config: dict[str, Any]) -> dict[str, Path]:
    inputs = config["input_files"]
    return {
        "quasi_real_path_feedback_summary": quasi / inputs["quasi_real_path_feedback_summary"],
        "quasi_real_slices": quasi / inputs["quasi_real_slices"],
        "quasi_real_domain_gap_summary": quasi / inputs["quasi_real_domain_gap_summary"],
        "guarded_pilot_summary": guarded / inputs["guarded_pilot_summary"],
        "guarded_decisions": guarded / inputs["guarded_decisions"],
        "alignment_summary": alignment / inputs["alignment_summary"],
    }


def _output_paths(output: Path, config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["output_files"]
    return {
        "summary": output / outputs["summary"],
        "diagnostics": output / outputs["diagnostics"],
        "exclusion_report": output / outputs["exclusion_report"],
        "report": output / outputs["report"],
    }


def _source_candidate(candidates: list[Any], path_feedback: dict[str, Any], guarded_decision: dict[str, Any]) -> dict[str, Any] | None:
    source_action = guarded_decision.get("source_action_index")
    if source_action is None:
        source_action = path_feedback.get("source_selected_action_index")
    if source_action is not None:
        for candidate in candidates:
            if isinstance(candidate, dict) and _int(candidate.get("action_index"), -9999) == _int(source_action):
                return candidate
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        generation = candidate.get("candidate_generation") if isinstance(candidate.get("candidate_generation"), dict) else {}
        if generation.get("source_selection_status") == "source_selected":
            return candidate
    return candidates[0] if candidates and isinstance(candidates[0], dict) else None


def _action_mask_valid(candidate: dict[str, Any]) -> bool:
    if candidate.get("action_mask_valid") is False:
        return False
    if candidate.get("ppo_consumable_action") is False:
        return False
    return candidate.get("action_index") is not None


def _contract_safe(candidate: dict[str, Any]) -> bool:
    if candidate.get("contract_safe") is False:
        return False
    feasibility = candidate.get("platform_goal_feasibility") if isinstance(candidate.get("platform_goal_feasibility"), dict) else {}
    if feasibility.get("contract_reachable") is False:
        return False
    if feasibility.get("classification") in {"blocked_by_platform_footprint", "blocked"}:
        return False
    return True


def _source_selection_regression(candidate: dict[str, Any]) -> bool:
    generation = candidate.get("candidate_generation") if isinstance(candidate.get("candidate_generation"), dict) else {}
    reason_codes = generation.get("reason_codes") if isinstance(generation.get("reason_codes"), list) else []
    return "source_selection_quality_regression" in reason_codes


def _metric_counts(diagnostics: list[dict[str, Any]]) -> dict[str, int]:
    fields = (
        "invalid_action_mask_count",
        "fallback_or_open_grid_count",
        "safety_regression_count",
        "contract_violation_count",
        "path_cost_regression_count",
        "risk_regression_count",
        "source_selection_regression_count",
    )
    return {field: sum(int(row.get(field, 0)) for row in diagnostics) for field in fields}


def _funnel_metric_counts(diagnostics: list[dict[str, Any]]) -> dict[str, int]:
    fields = (
        "funnel_invalid_action_mask_count",
        "funnel_unreachable_count",
        "funnel_replan_required_count",
        "funnel_fallback_or_open_grid_count",
        "funnel_contract_violation_count",
        "funnel_path_cost_regression_count",
        "funnel_risk_regression_count",
        "funnel_source_selection_regression_count",
    )
    return {field: sum(int(row.get(field, 0)) for row in diagnostics) for field in fields}


def _roi_group_summary(diagnostics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "context_count": 0,
        "safe_alternative_context_count": 0,
        "safe_better_opportunity_context_count": 0,
        "policy_missed_safe_better_opportunity_count": 0,
        "policy_selected_safe_better_opportunity_count": 0,
        "opportunity_missing_count": 0,
        "start_cells": [],
        "safe_better_start_cells": [],
    })
    for row in diagnostics:
        group = str(row.get("roi_group") or "unknown")
        bucket = summary[group]
        bucket["context_count"] += 1
        start_cell = _normal_cell(row.get("start_cell"))
        if start_cell is not None and start_cell not in bucket["start_cells"]:
            bucket["start_cells"].append(start_cell)
        if int(row.get("safe_alternative_count", 0)) > 0:
            bucket["safe_alternative_context_count"] += 1
        if int(row.get("safe_better_alternative_count", 0)) > 0:
            bucket["safe_better_opportunity_context_count"] += 1
            if start_cell is not None and start_cell not in bucket["safe_better_start_cells"]:
                bucket["safe_better_start_cells"].append(start_cell)
        cls = row.get("opportunity_class")
        if cls == "safe_better_opportunity_exists_policy_source_aligned":
            bucket["policy_missed_safe_better_opportunity_count"] += 1
        elif cls == "safe_better_opportunity_policy_selected":
            bucket["policy_selected_safe_better_opportunity_count"] += 1
        elif cls == "opportunity_missing":
            bucket["opportunity_missing_count"] += 1
    return {key: dict(value) for key, value in sorted(summary.items())}


def _opportunity_verdict(reason_codes: list[str], counts: Counter[str]) -> str:
    reasons = set(reason_codes)
    if "real_map_action_mask_contract_gap" in reasons:
        return "real_map_action_mask_contract_gap"
    if any(reason in reasons for reason in ("real_map_bridge_or_feedback_gap", "quasi_real_safe_alternative_context_id_missing")):
        return "real_map_bridge_or_feedback_gap"
    safe_better_contexts = counts.get("safe_better_opportunity_exists_policy_source_aligned", 0) + counts.get(
        "safe_better_opportunity_policy_selected", 0
    )
    if safe_better_contexts <= 0:
        return "quasi_real_safe_alternative_opportunity_gap"
    return "acceptable_for_quasi_real_safe_choice_calibration"


def _next_required_change(verdict: str, counts: Counter[str]) -> str | None:
    if verdict == "acceptable_for_quasi_real_safe_choice_calibration":
        if counts.get("safe_better_opportunity_exists_policy_source_aligned", 0) > 0:
            return "quasi_real_policy_safe_choice_alignment_insufficient"
        return None
    return verdict


def _hard_failure(reason_codes: list[str]) -> bool:
    hard = {
        "quasi_real_opportunity_exclusion_count_above_threshold",
        "quasi_real_safe_alternative_context_id_missing",
        "real_map_action_mask_contract_gap",
        "real_map_bridge_or_feedback_gap",
    }
    return any(reason in hard for reason in reason_codes)


def _validate_inputs(domain_gap: dict[str, Any], alignment: dict[str, Any], guarded: dict[str, Any], reason_codes: list[str]) -> None:
    if domain_gap.get("status") != "passed" or domain_gap.get("domain_gap_verdict") != "acceptable_for_next_pilot":
        _append_reason(reason_codes, "real_map_bridge_or_feedback_gap")
    if alignment.get("status") != "passed" or alignment.get("alignment_verdict") != "acceptable_for_quasi_real_shadow_audit":
        _append_reason(reason_codes, "real_map_bridge_or_feedback_gap")
    if guarded and guarded.get("guarded_pilot_verdict") not in {
        "policy_over_conservative_on_quasi_real",
        "acceptable_for_quasi_real_collector_dry_run",
    }:
        _append_reason(reason_codes, "real_map_bridge_or_feedback_gap")


def _exclusion(scenario_id: str, category: str, reason_code: str) -> dict[str, str]:
    return {
        "scenario_id": scenario_id,
        "category": category,
        "reason_code": reason_code,
    }


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Quasi-Real Safe-Alternative Opportunity Diagnosis",
        "",
        f"- status: {summary['status']}",
        f"- opportunity_verdict: {summary['opportunity_verdict']}",
        f"- quasi_real_context_count: {summary['quasi_real_context_count']}",
        f"- safe_better_opportunity_context_count: {summary['safe_better_opportunity_context_count']}",
        f"- policy_missed_safe_better_opportunity_count: {summary['policy_missed_safe_better_opportunity_count']}",
        "",
        "| roi_group | contexts | safe-better contexts | safe-better starts | missed safe-better |",
        "|---|---:|---:|---|---:|",
    ]
    for group, row in summary["roi_group_opportunity_summary"].items():
        lines.append(
            f"| {group} | {row['context_count']} | {row['safe_better_opportunity_context_count']} | "
            f"{row.get('safe_better_start_cells', [])} | "
            f"{row['policy_missed_safe_better_opportunity_count']} |"
        )
    lines.append("")
    lines.append("Diagnostic only: no PPO update, no PPO transitions, and no policy release.")
    return "\n".join(lines)


def _load_json(path: Path, reason_codes: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        _append_reason(reason_codes, "real_map_bridge_or_feedback_gap")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _append_reason(reason_codes, "real_map_bridge_or_feedback_gap")
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_jsonl(path: Path, reason_codes: list[str], label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        _append_reason(reason_codes, "real_map_bridge_or_feedback_gap")
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            _append_reason(reason_codes, "real_map_bridge_or_feedback_gap")
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    for section in ("input_files", "output_files", "thresholds", "validation"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _append_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normal_cell(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return [int(value[0]), int(value[1])]
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve(path: str, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose safe alternatives in quasi-real guarded contexts.")
    parser.add_argument("--quasi-real-root", required=True)
    parser.add_argument("--guarded-pilot-root", required=True)
    parser.add_argument("--alignment-root", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    config = _load_config(_resolve(args.config, repo_root))
    summary = run_quasi_real_safe_alternative_opportunity_diagnosis(
        quasi_real_root=_resolve(args.quasi_real_root, repo_root),
        guarded_pilot_root=_resolve(args.guarded_pilot_root, repo_root),
        alignment_root=_resolve(args.alignment_root, repo_root),
        output_root=_resolve(args.output_root, repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "opportunity_verdict": summary["opportunity_verdict"],
                "safe_better_opportunity_context_count": summary[
                    "safe_better_opportunity_context_count"
                ],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary_output"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
