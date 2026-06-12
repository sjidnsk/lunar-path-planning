from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot

from run_quasi_real_shadow_policy_behavior_audit import (
    ScoreDecisions,
    run_quasi_real_shadow_policy_behavior_audit,
)
from run_scenario_disjoint_policy_rollout_evaluation import _display_path


CONFIG_SCHEMA_VERSION = "quasi-real-guarded-policy-pilot-config/v1"
SUMMARY_SCHEMA_VERSION = "quasi-real-guarded-policy-pilot-summary/v1"
DECISION_SCHEMA_VERSION = "quasi-real-guarded-policy-decision/v1"
REJECTION_REPORT_SCHEMA_VERSION = "quasi-real-guarded-policy-rejection-report/v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_quasi_real_guarded_policy_pilot_v1"


class ConfigError(ValueError):
    pass


def run_quasi_real_guarded_policy_pilot(
    *,
    source_root: str | Path,
    candidate_root: str | Path,
    quasi_real_root: str | Path,
    alignment_summary: str | Path | None,
    output_root: str | Path,
    config: dict[str, Any],
    repo_root: str | Path,
    score_decisions: ScoreDecisions | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    source = Path(source_root).resolve()
    candidate = Path(candidate_root).resolve()
    quasi = Path(quasi_real_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)

    output_paths = _output_paths(output, config)
    shadow_output = output / "_shadow_policy_behavior_audit"
    reason_codes: list[str] = []

    alignment_path = (
        Path(alignment_summary).resolve()
        if alignment_summary is not None
        else candidate / config["input_files"].get(
            "alignment_summary",
            "quasi-real-shadow-alignment-summary.json",
        )
    )
    alignment = _load_json(alignment_path, reason_codes, "quasi_real_shadow_alignment_summary")
    _validate_alignment_summary(alignment, reason_codes)

    shadow_summary = run_quasi_real_shadow_policy_behavior_audit(
        source_root=source,
        candidate_root=candidate,
        quasi_real_root=quasi,
        output_root=shadow_output,
        config=_shadow_config(config),
        repo_root=repo,
        score_decisions=score_decisions,
    )
    shadow_decisions = _load_jsonl(shadow_output / "quasi-real-shadow-policy-decisions.jsonl")
    decisions = [_guarded_decision_record(decision) for decision in shadow_decisions]

    summary = _summary(
        source_root=source,
        candidate_root=candidate,
        quasi_real_root=quasi,
        alignment_summary_path=alignment_path,
        output_root=output,
        output_paths=output_paths,
        shadow_summary=shadow_summary,
        alignment=alignment,
        decisions=decisions,
        reason_codes=reason_codes,
        config=config,
        repo_root=repo,
    )
    rejection_report = _rejection_report(decisions, summary)
    group_report = _group_report(summary)

    output_paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    output_paths["decisions"].write_text(
        "".join(json.dumps(decision, ensure_ascii=False) + "\n" for decision in decisions),
        encoding="utf-8",
    )
    output_paths["rejection_report"].write_text(
        json.dumps(rejection_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_paths["group_report"].write_text(group_report, encoding="utf-8")
    summary["summary_output"] = str(output_paths["summary"])
    return summary


def _guarded_decision_record(shadow: dict[str, Any]) -> dict[str, Any]:
    decision_class = str(shadow.get("decision_class") or "not_scored")
    if decision_class == "policy_changed_gate_passed":
        controlled_choice_source = "policy"
        controlled_action_index = shadow.get("raw_policy_action_index")
    elif decision_class == "policy_changed_gate_rejected":
        controlled_choice_source = "source_fallback"
        controlled_action_index = shadow.get("source_action_index")
    elif decision_class == "source_aligned":
        controlled_choice_source = "source"
        controlled_action_index = shadow.get("source_action_index")
    else:
        controlled_choice_source = "none"
        controlled_action_index = None
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "context_id": shadow.get("context_id"),
        "scenario_id": shadow.get("scenario_id"),
        "roi_group": shadow.get("roi_group"),
        "roi_name": shadow.get("roi_name"),
        "split": shadow.get("split"),
        "map_id": shadow.get("map_id"),
        "slice_id": shadow.get("slice_id"),
        "source_action_index": shadow.get("source_action_index"),
        "raw_policy_action_index": shadow.get("raw_policy_action_index"),
        "controlled_action_index": controlled_action_index,
        "controlled_choice_source": controlled_choice_source,
        "logit_margin": shadow.get("logit_margin"),
        "action_mask_valid": shadow.get("action_mask_valid"),
        "path_cost_delta": shadow.get("path_cost_delta"),
        "risk_delta": shadow.get("risk_delta"),
        "gate_reason_codes": list(shadow.get("gate_reason_codes", [])),
        "decision_class": decision_class,
        "policy_takes_control": controlled_choice_source == "policy",
        "ppo_trainable": False,
        "runs_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
    }


def _summary(
    *,
    source_root: Path,
    candidate_root: Path,
    quasi_real_root: Path,
    alignment_summary_path: Path,
    output_root: Path,
    output_paths: dict[str, Path],
    shadow_summary: dict[str, Any],
    alignment: dict[str, Any],
    decisions: list[dict[str, Any]],
    reason_codes: list[str],
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    validation = config.get("validation", {})
    counts = Counter(decision.get("decision_class") for decision in decisions)
    roi_groups = sorted({str(decision.get("roi_group")) for decision in decisions if decision.get("roi_group")})
    roi_group_guarded_coverage = {
        roi_group: sum(1 for decision in decisions if decision.get("roi_group") == roi_group)
        for roi_group in roi_groups
    }
    gate_reason_counts = Counter(
        reason for decision in decisions for reason in decision.get("gate_reason_codes", [])
    )
    metric_counts = {
        "invalid_action_mask_count": gate_reason_counts.get("invalid_action_mask", 0),
        "fallback_or_open_grid_count": gate_reason_counts.get("fallback_or_open_grid", 0),
        "open_grid_fallback_count": gate_reason_counts.get("fallback_or_open_grid", 0),
        "safety_regression_count": gate_reason_counts.get("safety_regression", 0),
        "contract_violation_count": gate_reason_counts.get("contract_violation", 0),
        "contract_regression_count": gate_reason_counts.get("contract_violation", 0),
        "path_cost_regression_count": gate_reason_counts.get("path_cost_regression", 0),
        "risk_regression_count": gate_reason_counts.get("risk_regression", 0),
        "source_selection_regression_count": gate_reason_counts.get("source_selection_regression", 0),
    }
    quasi_real_context_count = _int(shadow_summary.get("shadow_context_count"))
    policy_decision_count = len(decisions)
    context_id_missing_count = _int(shadow_summary.get("context_id_missing_count"))
    changed_passed = counts.get("policy_changed_gate_passed", 0)
    changed_rejected = counts.get("policy_changed_gate_rejected", 0)

    if shadow_summary.get("status") != "passed":
        _map_shadow_reasons(shadow_summary, reason_codes)
    if quasi_real_context_count < _int(validation.get("min_quasi_real_context_count")):
        _append_reason(reason_codes, "quasi_real_guarded_context_count_below_threshold")
    if len(roi_groups) < _int(validation.get("min_roi_group_count")):
        _append_reason(reason_codes, "quasi_real_guarded_roi_group_count_below_threshold")
    if policy_decision_count != quasi_real_context_count:
        _append_reason(reason_codes, "quasi_real_guarded_policy_scoring_failed")
    if context_id_missing_count:
        _append_reason(reason_codes, "quasi_real_guarded_context_id_missing")
    for field, reason in (
        ("invalid_action_mask_count", "quasi_real_guarded_action_mask_contract_gap"),
        ("fallback_or_open_grid_count", "quasi_real_guarded_gate_regression"),
        ("safety_regression_count", "quasi_real_guarded_gate_regression"),
        ("contract_violation_count", "quasi_real_guarded_gate_regression"),
        ("path_cost_regression_count", "quasi_real_guarded_gate_regression"),
        ("risk_regression_count", "quasi_real_guarded_gate_regression"),
        ("source_selection_regression_count", "quasi_real_guarded_gate_regression"),
    ):
        if metric_counts[field] > _int(validation.get(f"max_{field}", 0)):
            _append_reason(reason_codes, reason)
    if changed_rejected > _int(validation.get("max_policy_changed_gate_rejected_count", 0)):
        _append_reason(reason_codes, "quasi_real_guarded_gate_regression")
    min_changed_passed = _int(validation.get("min_policy_changed_gate_passed_count"))
    over_conservative = changed_passed < min_changed_passed
    if over_conservative:
        _append_reason(reason_codes, "quasi_real_guarded_policy_over_conservative")

    unique_reasons = _unique(reason_codes)
    verdict = _guarded_verdict(unique_reasons)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "status": "passed" if not unique_reasons else "failed",
        "reason_codes": unique_reasons,
        "guarded_pilot_verdict": verdict,
        "next_required_change": None if not unique_reasons else verdict,
        "source_root": _display_path(source_root, repo_root),
        "candidate_root": _display_path(candidate_root, repo_root),
        "quasi_real_root": _display_path(quasi_real_root, repo_root),
        "alignment_summary": _display_path(alignment_summary_path, repo_root),
        "output_root": _display_path(output_root, repo_root),
        "decisions_path": _display_path(output_paths["decisions"], repo_root),
        "shadow_behavior_status": shadow_summary.get("status"),
        "shadow_behavior_verdict": shadow_summary.get("behavior_verdict"),
        "alignment_status": alignment.get("status"),
        "alignment_verdict": alignment.get("alignment_verdict"),
        "quasi_real_context_count": quasi_real_context_count,
        "shadow_context_count": quasi_real_context_count,
        "policy_decision_count": policy_decision_count,
        "source_aligned_count": counts.get("source_aligned", 0),
        "policy_source_aligned_count": counts.get("source_aligned", 0),
        "policy_changed_decision_count": changed_passed + changed_rejected,
        "policy_changed_gate_passed_count": changed_passed,
        "policy_changed_gate_rejected_count": changed_rejected,
        "source_fallback_count": changed_rejected,
        "not_scored_count": counts.get("not_scored", 0),
        "roi_group_count": len(roi_groups),
        "roi_groups": roi_groups,
        "roi_group_guarded_coverage": roi_group_guarded_coverage,
        "context_id_missing_count": context_id_missing_count,
        "over_conservative_policy_detected": over_conservative,
        **metric_counts,
        "safe_better_opportunity_count": _int(alignment.get("safe_better_opportunity_count")),
        "policy_takes_control": changed_passed > 0,
        "runs_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "git_provenance": {"current": _git_snapshot(repo_root), "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }


def _map_shadow_reasons(shadow_summary: dict[str, Any], reason_codes: list[str]) -> None:
    for reason in _string_list(shadow_summary.get("reason_codes")):
        if reason == "quasi_real_shadow_action_mask_contract_gap":
            _append_reason(reason_codes, "quasi_real_guarded_action_mask_contract_gap")
        elif reason == "quasi_real_shadow_context_id_missing":
            _append_reason(reason_codes, "quasi_real_guarded_context_id_missing")
        elif reason == "quasi_real_shadow_gate_regression":
            _append_reason(reason_codes, "quasi_real_guarded_gate_regression")
        elif reason in {
            "quasi_real_shadow_policy_scoring_failed",
            "quasi_real_domain_gap_not_acceptable",
            "quasi_real_path_feedback_summary_failed",
        }:
            _append_reason(reason_codes, "quasi_real_guarded_policy_scoring_failed")


def _validate_alignment_summary(alignment: dict[str, Any], reason_codes: list[str]) -> None:
    if not alignment:
        _append_reason(reason_codes, "quasi_real_guarded_pilot_contract_invalid")
        return
    if alignment.get("status") != "passed" or _string_list(alignment.get("reason_codes")):
        _append_reason(reason_codes, "quasi_real_guarded_pilot_contract_invalid")
    if alignment.get("alignment_verdict") != "acceptable_for_quasi_real_shadow_audit":
        _append_reason(reason_codes, "quasi_real_guarded_pilot_contract_invalid")
    if _int(alignment.get("hard_positive_added_count")):
        _append_reason(reason_codes, "quasi_real_guarded_pilot_contract_invalid")
    if _int(alignment.get("ppo_transition_added_count")):
        _append_reason(reason_codes, "quasi_real_guarded_pilot_contract_invalid")
    if alignment.get("publishes_checkpoint") or alignment.get("replaces_default_policy") or alignment.get("performance_claimed"):
        _append_reason(reason_codes, "quasi_real_guarded_pilot_contract_invalid")


def _rejection_report(decisions: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    rejections = [
        decision
        for decision in decisions
        if decision.get("decision_class") == "policy_changed_gate_rejected"
    ]
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "status": "passed" if not rejections else "failed",
        "summary_status": summary["status"],
        "rejected_count": len(rejections),
        "source_fallback_count": summary["source_fallback_count"],
        "rejections": rejections,
    }


def _group_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Quasi-Real Guarded Policy Pilot",
        "",
        f"- status: {summary['status']}",
        f"- guarded_pilot_verdict: {summary['guarded_pilot_verdict']}",
        f"- quasi_real_context_count: {summary['quasi_real_context_count']}",
        f"- policy_changed_gate_passed_count: {summary['policy_changed_gate_passed_count']}",
        f"- policy_changed_gate_rejected_count: {summary['policy_changed_gate_rejected_count']}",
        "",
        "| roi_group | guarded decisions |",
        "|---|---:|",
    ]
    for roi_group, count in sorted(summary["roi_group_guarded_coverage"].items()):
        lines.append(f"| {roi_group} | {count} |")
    lines.append("")
    lines.append("Guarded pilot only: policy may take a gate-approved controlled choice, but no PPO update is run.")
    return "\n".join(lines)


def _guarded_verdict(reason_codes: list[str]) -> str:
    reasons = set(reason_codes)
    if not reasons:
        return "acceptable_for_quasi_real_collector_dry_run"
    if "quasi_real_guarded_action_mask_contract_gap" in reasons:
        return "real_map_action_mask_contract_gap"
    if "quasi_real_guarded_policy_scoring_failed" in reasons or "quasi_real_guarded_pilot_contract_invalid" in reasons:
        return "real_map_bridge_or_feedback_gap"
    if "quasi_real_guarded_context_count_below_threshold" in reasons or "quasi_real_guarded_roi_group_count_below_threshold" in reasons:
        return "scenario_expansion_required"
    if "quasi_real_guarded_gate_regression" in reasons:
        return "policy_real_map_alignment_refinement_required"
    if "quasi_real_guarded_policy_over_conservative" in reasons:
        return "policy_over_conservative_on_quasi_real"
    return "policy_real_map_alignment_refinement_required"


def _shadow_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_files": dict(config.get("input_files", {})),
        "output_files": {
            "decisions": "quasi-real-shadow-policy-decisions.jsonl",
            "summary": "quasi-real-shadow-policy-behavior-summary.json",
            "rejection_report": "quasi-real-shadow-policy-rejection-report.json",
            "group_report": "quasi-real-shadow-policy-group-report.md",
        },
        "validation": {
            "min_shadow_context_count": config.get("validation", {}).get("min_quasi_real_context_count", 0),
            "min_roi_group_count": config.get("validation", {}).get("min_roi_group_count", 0),
            "max_invalid_action_mask_count": config.get("validation", {}).get("max_invalid_action_mask_count", 0),
            "max_fallback_or_open_grid_count": config.get("validation", {}).get("max_fallback_or_open_grid_count", 0),
            "max_safety_regression_count": config.get("validation", {}).get("max_safety_regression_count", 0),
            "max_contract_violation_count": config.get("validation", {}).get("max_contract_violation_count", 0),
            "max_path_cost_regression_count": config.get("validation", {}).get("max_path_cost_regression_count", 0),
            "max_risk_regression_count": config.get("validation", {}).get("max_risk_regression_count", 0),
            "max_source_selection_regression_count": config.get("validation", {}).get("max_source_selection_regression_count", 0),
        },
        "evaluation": dict(config.get("evaluation", {})),
        "non_goals": list(config.get("non_goals", [])),
    }


def _output_paths(output_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["output_files"]
    return {
        "decisions": output_root / outputs["decisions"],
        "summary": output_root / outputs["summary"],
        "rejection_report": output_root / outputs["rejection_report"],
        "group_report": output_root / outputs["group_report"],
    }


def _load_json(path: Path, reason_codes: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _append_reason(reason_codes, f"{label}_invalid_json")
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    for section in ("input_files", "output_files", "validation", "evaluation"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _append_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve(path: str, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run quasi-real guarded policy pilot.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--quasi-real-root", required=True)
    parser.add_argument("--alignment-summary")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    config = _load_config(_resolve(args.config, repo_root))
    summary = run_quasi_real_guarded_policy_pilot(
        source_root=_resolve(args.source_root, repo_root),
        candidate_root=_resolve(args.candidate_root, repo_root),
        quasi_real_root=_resolve(args.quasi_real_root, repo_root),
        alignment_summary=_resolve(args.alignment_summary, repo_root) if args.alignment_summary else None,
        output_root=_resolve(args.output_root, repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "guarded_pilot_verdict": summary["guarded_pilot_verdict"],
                "quasi_real_context_count": summary["quasi_real_context_count"],
                "policy_changed_gate_passed_count": summary["policy_changed_gate_passed_count"],
                "summary": summary["summary_output"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
