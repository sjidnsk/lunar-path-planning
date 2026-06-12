from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from git_provenance import git_snapshot as _git_snapshot
from run_quasi_real_shadow_policy_behavior_audit import (
    run_quasi_real_shadow_policy_behavior_audit,
)
from run_scenario_disjoint_policy_rollout_evaluation import _display_path


CONFIG_SCHEMA_VERSION = "quasi-real-teacher-equivalent-validation-config/v1"
SUMMARY_SCHEMA_VERSION = "quasi-real-teacher-equivalent-summary/v1"
DECISION_SCHEMA_VERSION = "quasi-real-teacher-equivalent-decision/v1"
DISAGREEMENT_REPORT_SCHEMA_VERSION = "quasi-real-teacher-equivalent-disagreement-report/v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_quasi_real_teacher_equivalent_validation_v1"


ScoreDecisions = Callable[..., list[dict[str, Any]]]


class ConfigError(ValueError):
    pass


def run_quasi_real_teacher_equivalent_validation(
    *,
    source_root: str | Path,
    candidate_root: str | Path,
    quasi_real_root: str | Path,
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
    shadow_root = output / "_shadow_policy_behavior_audit"
    shadow_summary = run_quasi_real_shadow_policy_behavior_audit(
        source_root=source,
        candidate_root=candidate,
        quasi_real_root=quasi,
        output_root=shadow_root,
        config=_shadow_config(config),
        repo_root=repo,
        score_decisions=score_decisions,
    )
    shadow_decisions = _read_jsonl(shadow_root / "quasi-real-shadow-policy-decisions.jsonl")
    decisions = [_teacher_decision_record(decision) for decision in shadow_decisions]
    summary = _summary(
        source_root=source,
        candidate_root=candidate,
        quasi_real_root=quasi,
        output_root=output,
        output_paths=output_paths,
        shadow_summary=shadow_summary,
        decisions=decisions,
        config=config,
        repo_root=repo,
    )
    disagreement_report = _disagreement_report(decisions, summary)
    group_report = _group_report(summary)

    output_paths["decisions"].write_text(
        "".join(json.dumps(decision, ensure_ascii=False) + "\n" for decision in decisions),
        encoding="utf-8",
    )
    output_paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    output_paths["disagreement_report"].write_text(
        json.dumps(disagreement_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_paths["group_report"].write_text(group_report, encoding="utf-8")
    summary["summary_output"] = str(output_paths["summary"])
    return summary


def _teacher_decision_record(decision: dict[str, Any]) -> dict[str, Any]:
    decision_class = str(decision.get("decision_class", "not_scored"))
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "context_id": decision.get("context_id"),
        "scenario_id": decision.get("scenario_id"),
        "roi_group": decision.get("roi_group"),
        "roi_name": decision.get("roi_name"),
        "split": decision.get("split"),
        "map_id": decision.get("map_id"),
        "slice_id": decision.get("slice_id"),
        "source_action_index": decision.get("source_action_index"),
        "teacher_action_index": decision.get("source_action_index"),
        "raw_policy_action_index": decision.get("raw_policy_action_index"),
        "policy_action_index": decision.get("raw_policy_action_index"),
        "logit_margin": decision.get("logit_margin"),
        "action_mask_valid": decision.get("action_mask_valid"),
        "path_cost_delta": decision.get("path_cost_delta"),
        "risk_delta": decision.get("risk_delta"),
        "gate_reason_codes": list(decision.get("gate_reason_codes", [])),
        "decision_class": decision_class,
        "teacher_aligned": decision_class == "source_aligned",
        "safe_disagreement": decision_class == "policy_changed_gate_passed",
        "unsafe_disagreement": decision_class == "policy_changed_gate_rejected",
        "policy_takes_control": False,
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
    output_root: Path,
    output_paths: dict[str, Path],
    shadow_summary: dict[str, Any],
    decisions: list[dict[str, Any]],
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    validation = config.get("validation", {})
    counts = Counter(str(decision.get("decision_class", "not_scored")) for decision in decisions)
    roi_groups = sorted({str(decision.get("roi_group")) for decision in decisions if decision.get("roi_group")})
    context_count = _int_value(shadow_summary.get("shadow_context_count"))
    policy_decision_count = len(decisions)
    teacher_aligned_count = counts.get("source_aligned", 0)
    safe_disagreement_count = counts.get("policy_changed_gate_passed", 0)
    unsafe_disagreement_count = counts.get("policy_changed_gate_rejected", 0)
    agreement_rate = teacher_aligned_count / policy_decision_count if policy_decision_count else 0.0
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
    context_id_missing_count = sum(1 for decision in decisions if not decision.get("context_id"))
    roi_group_teacher_agreement_summary = _roi_group_summary(decisions)

    reason_codes: list[str] = []
    if shadow_summary.get("policy_decision_count") != shadow_summary.get("shadow_context_count"):
        _append_reason(reason_codes, "quasi_real_teacher_equivalent_scoring_failed")
    if shadow_summary.get("scoring_error"):
        _append_reason(reason_codes, "quasi_real_teacher_equivalent_scoring_failed")
    if context_count < _int_value(validation.get("min_teacher_equivalent_context_count")):
        _append_reason(reason_codes, "quasi_real_teacher_equivalent_context_coverage_insufficient")
    if policy_decision_count != context_count:
        _append_reason(reason_codes, "quasi_real_teacher_equivalent_scoring_failed")
    if len(roi_groups) < _int_value(validation.get("min_roi_group_count")):
        _append_reason(reason_codes, "quasi_real_teacher_equivalent_context_coverage_insufficient")
    if context_id_missing_count:
        _append_reason(reason_codes, "quasi_real_teacher_equivalent_context_id_missing")
    if agreement_rate < float(validation.get("min_teacher_agreement_rate", 0.9)):
        _append_reason(reason_codes, "quasi_real_teacher_equivalent_context_coverage_insufficient")
    if unsafe_disagreement_count:
        _append_reason(reason_codes, "quasi_real_teacher_equivalent_unsafe_disagreement")
    if counts.get("not_scored", 0):
        _append_reason(reason_codes, "quasi_real_teacher_equivalent_scoring_failed")
    for field, reason in (
        ("invalid_action_mask_count", "quasi_real_teacher_equivalent_gate_regression"),
        ("fallback_or_open_grid_count", "quasi_real_teacher_equivalent_gate_regression"),
        ("safety_regression_count", "quasi_real_teacher_equivalent_gate_regression"),
        ("contract_violation_count", "quasi_real_teacher_equivalent_gate_regression"),
        ("path_cost_regression_count", "quasi_real_teacher_equivalent_gate_regression"),
        ("risk_regression_count", "quasi_real_teacher_equivalent_gate_regression"),
        ("source_selection_regression_count", "quasi_real_teacher_equivalent_gate_regression"),
    ):
        if metric_counts[field]:
            _append_reason(reason_codes, reason)

    unique_reasons = _unique(reason_codes)
    verdict = _teacher_equivalent_verdict(unique_reasons)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "status": "passed" if not unique_reasons else "failed",
        "reason_codes": unique_reasons,
        "teacher_equivalent_verdict": verdict,
        "next_required_change": None if not unique_reasons else verdict,
        "source_root": _display_path(source_root, repo_root),
        "candidate_root": _display_path(candidate_root, repo_root),
        "quasi_real_root": _display_path(quasi_real_root, repo_root),
        "output_root": _display_path(output_root, repo_root),
        "shadow_summary_path": shadow_summary.get("summary_output"),
        "decisions_path": _display_path(output_paths["decisions"], repo_root),
        "teacher_equivalent_context_count": context_count,
        "policy_decision_count": policy_decision_count,
        "teacher_aligned_count": teacher_aligned_count,
        "teacher_agreement_rate": agreement_rate,
        "safe_disagreement_count": safe_disagreement_count,
        "unsafe_disagreement_count": unsafe_disagreement_count,
        "policy_changed_gate_passed_count": safe_disagreement_count,
        "policy_changed_gate_rejected_count": unsafe_disagreement_count,
        "not_scored_count": counts.get("not_scored", 0),
        "roi_group_count": len(roi_groups),
        "roi_groups": roi_groups,
        "roi_group_teacher_agreement_summary": roi_group_teacher_agreement_summary,
        "context_id_missing_count": context_id_missing_count,
        **metric_counts,
        "runs_ppo_update": False,
        "policy_takes_control": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "git_provenance": {"current": _git_snapshot(repo_root), "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }


def _roi_group_summary(decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for decision in decisions:
        roi_group = str(decision.get("roi_group") or "unknown")
        grouped.setdefault(roi_group, []).append(decision)
    summary: dict[str, dict[str, Any]] = {}
    for roi_group, records in sorted(grouped.items()):
        aligned = sum(1 for record in records if record.get("teacher_aligned"))
        safe = sum(1 for record in records if record.get("safe_disagreement"))
        unsafe = sum(1 for record in records if record.get("unsafe_disagreement"))
        summary[roi_group] = {
            "context_count": len(records),
            "teacher_aligned_count": aligned,
            "teacher_agreement_rate": aligned / len(records) if records else 0.0,
            "safe_disagreement_count": safe,
            "unsafe_disagreement_count": unsafe,
        }
    return summary


def _disagreement_report(decisions: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    disagreements = [
        decision
        for decision in decisions
        if decision.get("safe_disagreement") or decision.get("unsafe_disagreement")
    ]
    return {
        "schema_version": DISAGREEMENT_REPORT_SCHEMA_VERSION,
        "status": summary["status"],
        "teacher_equivalent_verdict": summary["teacher_equivalent_verdict"],
        "disagreement_count": len(disagreements),
        "unsafe_disagreement_count": summary["unsafe_disagreement_count"],
        "disagreements": disagreements,
    }


def _group_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Quasi-Real Teacher-Equivalent Validation",
        "",
        f"- status: {summary['status']}",
        f"- teacher_equivalent_verdict: {summary['teacher_equivalent_verdict']}",
        f"- teacher_equivalent_context_count: {summary['teacher_equivalent_context_count']}",
        f"- teacher_agreement_rate: {summary['teacher_agreement_rate']:.4f}",
        f"- unsafe_disagreement_count: {summary['unsafe_disagreement_count']}",
        "",
        "| roi_group | contexts | teacher aligned | agreement rate | safe disagreement | unsafe disagreement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for roi_group, record in sorted(summary["roi_group_teacher_agreement_summary"].items()):
        lines.append(
            f"| {roi_group} | {record['context_count']} | {record['teacher_aligned_count']} | "
            f"{record['teacher_agreement_rate']:.4f} | {record['safe_disagreement_count']} | "
            f"{record['unsafe_disagreement_count']} |"
        )
    lines.append("")
    lines.append("Teacher-equivalent validation only: source-aligned behavior is valid, policy does not take control, and no PPO update is run.")
    return "\n".join(lines)


def _shadow_config(config: dict[str, Any]) -> dict[str, Any]:
    outputs = config["output_files"]
    validation = config.get("validation", {})
    return {
        "schema_version": "quasi-real-shadow-policy-behavior-audit-config/v1",
        "input_files": dict(config["input_files"]),
        "output_files": {
            "decisions": "quasi-real-shadow-policy-decisions.jsonl",
            "summary": "quasi-real-shadow-policy-behavior-summary.json",
            "rejection_report": "quasi-real-shadow-policy-rejection-report.json",
            "group_report": "quasi-real-shadow-policy-group-report.md",
        },
        "evaluation": dict(config.get("evaluation", {})),
        "validation": {
            "min_shadow_context_count": validation.get("min_teacher_equivalent_context_count", 48),
            "min_roi_group_count": validation.get("min_roi_group_count", 4),
            "max_invalid_action_mask_count": 0,
            "max_fallback_or_open_grid_count": 0,
            "max_safety_regression_count": 0,
            "max_contract_violation_count": 0,
            "max_path_cost_regression_count": 0,
            "max_risk_regression_count": 0,
            "max_source_selection_regression_count": 0,
        },
        "non_goals": list(config.get("non_goals", [])),
        "_teacher_equivalent_outputs": outputs,
    }


def _output_paths(output_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["output_files"]
    return {
        "decisions": output_root / outputs["decisions"],
        "summary": output_root / outputs["summary"],
        "disagreement_report": output_root / outputs["disagreement_report"],
        "group_report": output_root / outputs["group_report"],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _teacher_equivalent_verdict(reason_codes: list[str]) -> str:
    reasons = set(reason_codes)
    if not reasons:
        return "teacher_equivalent_validated"
    if "quasi_real_teacher_equivalent_unsafe_disagreement" in reasons:
        return "quasi_real_teacher_equivalent_unsafe_disagreement"
    if "quasi_real_teacher_equivalent_gate_regression" in reasons:
        return "quasi_real_teacher_equivalent_gate_regression"
    if "quasi_real_teacher_equivalent_context_id_missing" in reasons:
        return "quasi_real_teacher_equivalent_context_id_missing"
    if "quasi_real_teacher_equivalent_context_coverage_insufficient" in reasons:
        return "quasi_real_teacher_equivalent_context_coverage_insufficient"
    return "quasi_real_teacher_equivalent_scoring_failed"


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    for section in ("input_files", "output_files", "validation", "evaluation"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    for output in ("decisions", "summary", "disagreement_report", "group_report"):
        if not payload["output_files"].get(output):
            raise ConfigError(f"output_files.{output} must be set")
    return payload


def _append_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (repo_root / candidate).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run quasi-real teacher-equivalent validation.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--quasi-real-root", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    config = _load_config(_resolve(args.config, repo_root))
    summary = run_quasi_real_teacher_equivalent_validation(
        source_root=_resolve(args.source_root, repo_root),
        candidate_root=_resolve(args.candidate_root, repo_root),
        quasi_real_root=_resolve(args.quasi_real_root, repo_root),
        output_root=_resolve(args.output_root, repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "teacher_equivalent_verdict": summary["teacher_equivalent_verdict"],
                "teacher_equivalent_context_count": summary["teacher_equivalent_context_count"],
                "teacher_agreement_rate": summary["teacher_agreement_rate"],
                "summary": summary["summary_output"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
