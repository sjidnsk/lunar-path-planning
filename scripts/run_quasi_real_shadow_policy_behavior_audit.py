from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from git_provenance import git_snapshot as _git_snapshot

from run_controlled_hybrid_policy_training_candidate import CHECKPOINT_METADATA_SCHEMA_VERSION
from run_scenario_disjoint_policy_rollout_evaluation import (
    _candidate_record,
    _display_path,
    _int_value,
    _load_summary,
    _planning_backend,
    _score_scenarios,
    _string_list,
)


CONFIG_SCHEMA_VERSION = "quasi-real-shadow-policy-behavior-audit-config/v1"
SUMMARY_SCHEMA_VERSION = "quasi-real-shadow-policy-behavior-summary/v1"
DECISION_SCHEMA_VERSION = "quasi-real-shadow-policy-decision/v1"
REJECTION_REPORT_SCHEMA_VERSION = "quasi-real-shadow-policy-rejection-report/v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_quasi_real_shadow_policy_behavior_v1"


ScoreDecisions = Callable[..., list[dict[str, Any]]]


class ConfigError(ValueError):
    pass


def run_quasi_real_shadow_policy_behavior_audit(
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

    paths = _input_paths(source, candidate, quasi, config)
    reason_codes: list[str] = []
    source_batch = _load_summary(
        paths["source_batch_summary"], expected_schema=None, label="source_batch_summary", reason_codes=reason_codes
    )
    domain_gap = _load_summary(
        paths["quasi_real_domain_gap_summary"],
        expected_schema="quasi-real-map-domain-gap-summary/v1",
        label="quasi_real_domain_gap_summary",
        reason_codes=reason_codes,
    )
    quasi_summary = _load_summary(
        paths["quasi_real_path_feedback_summary"],
        expected_schema=None,
        label="quasi_real_path_feedback_summary",
        reason_codes=reason_codes,
    )
    candidate_summary = _load_summary(
        paths["candidate_summary"],
        expected_schema="raw-policy-generalization-candidate-summary/v1",
        label="candidate_summary",
        reason_codes=reason_codes,
    )
    metadata = _load_summary(
        paths["checkpoint_metadata"],
        expected_schema=CHECKPOINT_METADATA_SCHEMA_VERSION,
        label="checkpoint_metadata",
        reason_codes=reason_codes,
    )
    slices = _load_slices(paths["quasi_real_slices"], reason_codes)

    if source_batch.get("failed_count", 0) or _string_list(source_batch.get("reason_codes")):
        _append_reason(reason_codes, "source_batch_failed")
    if domain_gap.get("status") != "passed" or domain_gap.get("domain_gap_verdict") != "acceptable_for_next_pilot":
        _append_reason(reason_codes, "quasi_real_domain_gap_not_acceptable")
    if quasi_summary.get("status") not in {"completed", "passed", None}:
        _append_reason(reason_codes, "quasi_real_path_feedback_summary_failed")
    if candidate_summary.get("status") != "passed":
        _append_reason(reason_codes, "candidate_summary_failed")
    if metadata and metadata.get("experimental") is not True:
        _append_reason(reason_codes, "checkpoint_metadata_not_experimental")
    if candidate_summary.get("publishes_checkpoint") or metadata.get("publishes_checkpoint"):
        _append_reason(reason_codes, "checkpoint_publication_detected")
    if candidate_summary.get("replaces_default_policy") or metadata.get("replaces_default_policy"):
        _append_reason(reason_codes, "default_policy_replacement_detected")
    if candidate_summary.get("performance_claimed") or metadata.get("performance_claimed"):
        _append_reason(reason_codes, "performance_claim_detected")
    if not paths["checkpoint"].is_file():
        _append_reason(reason_codes, "experimental_checkpoint_missing")

    scenario_groups = _quasi_real_scenario_groups(
        quasi_summary=quasi_summary,
        slices_by_scenario_id={str(item.get("scenario_id")): item for item in slices},
        summary_path=paths["quasi_real_path_feedback_summary"],
        repo_root=repo,
    )
    context_id_missing_count = sum(
        1 for group in scenario_groups for record in group.get("candidates", []) if not record.get("context_id")
    )
    if context_id_missing_count:
        _append_reason(reason_codes, "quasi_real_shadow_context_id_missing")

    raw_decisions: list[dict[str, Any]] = []
    scoring_error = None
    if score_decisions is not None:
        raw_decisions = score_decisions(
            checkpoint_path=paths["checkpoint"],
            scenario_groups=scenario_groups,
            config=_scoring_config(config),
            repo_root=repo,
        )
    elif not any(reason in reason_codes for reason in ("experimental_checkpoint_missing", "checkpoint_metadata_not_experimental")):
        try:
            _ensure_model_explorer_src(repo)
            raw_decisions = _score_scenarios(
                checkpoint_path=paths["checkpoint"],
                scenario_groups=scenario_groups,
                config=_scoring_config(config),
                repo_root=repo,
            )
        except Exception as exc:  # noqa: BLE001
            scoring_error = str(exc)
            _append_reason(reason_codes, "quasi_real_shadow_policy_scoring_failed")

    decisions = [
        _shadow_decision_record(decision, slices_by_scenario_id={str(item.get("scenario_id")): item for item in slices})
        for decision in raw_decisions
    ]
    summary = _summary(
        source_root=source,
        candidate_root=candidate,
        quasi_real_root=quasi,
        output_root=output,
        output_paths=_output_paths(output, config),
        reason_codes=reason_codes,
        scenario_groups=scenario_groups,
        decisions=decisions,
        scoring_error=scoring_error,
        config=config,
        repo_root=repo,
    )
    rejection_report = _rejection_report(decisions, summary)
    group_report = _group_report(summary)
    output_paths = _output_paths(output, config)
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


def _quasi_real_scenario_groups(
    *,
    quasi_summary: dict[str, Any],
    slices_by_scenario_id: dict[str, dict[str, Any]],
    summary_path: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    scenarios = quasi_summary.get("scenarios")
    if not isinstance(scenarios, list):
        return groups
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        scenario_id = str(scenario.get("scenario_id", ""))
        path_feedback = scenario.get("path_feedback") if isinstance(scenario.get("path_feedback"), dict) else {}
        candidates = path_feedback.get("candidates")
        candidates = candidates if isinstance(candidates, list) else []
        slice_record = slices_by_scenario_id.get(scenario_id, {})
        group = {
            "run_id": "quasi_real_map_domain_gap",
            "source_path": _display_path(summary_path, repo_root),
            "scenario_id": scenario_id,
            "scenario_group": str(scenario.get("scenario_group") or slice_record.get("roi_name") or "unknown"),
            "scenario_seed": scenario.get("scenario_seed"),
            "scenario_variant_id": scenario.get("scenario_variant_id"),
            "diagnostic_profile": quasi_summary.get("diagnostic_profile"),
            "planning_backend": _planning_backend(quasi_summary),
            "best_by_path_cost": path_feedback.get("best_by_path_cost") if isinstance(path_feedback.get("best_by_path_cost"), dict) else None,
            "quasi_real_slice": slice_record,
            "candidates": [
                _candidate_record(candidate, scenario=scenario, summary=quasi_summary, group_path=summary_path, repo_root=repo_root)
                for candidate in candidates
                if isinstance(candidate, dict)
            ],
        }
        groups.append(group)
    return groups


def _shadow_decision_record(
    decision: dict[str, Any],
    *,
    slices_by_scenario_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    scenario_id = str(decision.get("scenario_id", ""))
    slice_record = slices_by_scenario_id.get(scenario_id, {})
    gate_reasons = list(decision.get("raw_policy_regression_reason_codes", []))
    raw_changed = _raw_policy_changed(decision)
    if not decision:
        decision_class = "not_scored"
    elif not raw_changed:
        decision_class = "source_aligned"
    elif gate_reasons:
        decision_class = "policy_changed_gate_rejected"
    else:
        decision_class = "policy_changed_gate_passed"
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "context_id": decision.get("context_id"),
        "scenario_id": scenario_id,
        "roi_group": slice_record.get("roi_name") or decision.get("scenario_group"),
        "roi_name": slice_record.get("roi_name") or decision.get("scenario_group"),
        "split": slice_record.get("split"),
        "map_id": slice_record.get("map_id"),
        "slice_id": slice_record.get("slice_id") or scenario_id,
        "source_action_index": decision.get("source_selected_action_index"),
        "raw_policy_action_index": decision.get("raw_policy_selected_action_index"),
        "logit_margin": decision.get("raw_policy_logit_margin_vs_source"),
        "action_mask_valid": bool(decision.get("action_mask_valid")),
        "path_cost_delta": decision.get("raw_policy_selected_path_cost_delta"),
        "risk_delta": decision.get("raw_policy_selected_risk_delta"),
        "gate_reason_codes": gate_reasons,
        "decision_class": decision_class,
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
    reason_codes: list[str],
    scenario_groups: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    scoring_error: str | None,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    validation = config.get("validation", {})
    counts = Counter(decision.get("decision_class") for decision in decisions)
    roi_groups = sorted({str(decision.get("roi_group")) for decision in decisions if decision.get("roi_group")})
    roi_group_shadow_coverage = {
        roi_group: sum(1 for decision in decisions if decision.get("roi_group") == roi_group)
        for roi_group in roi_groups
    }
    context_id_missing_count = sum(
        1 for group in scenario_groups for candidate in group.get("candidates", []) if not candidate.get("context_id")
    )
    gate_reason_counts = Counter(
        reason for decision in decisions for reason in decision.get("gate_reason_codes", [])
    )
    metric_counts = {
        "invalid_action_mask_count": gate_reason_counts.get("invalid_action_mask", 0),
        "fallback_or_open_grid_count": gate_reason_counts.get("fallback_or_open_grid", 0),
        "safety_regression_count": gate_reason_counts.get("safety_regression", 0),
        "contract_violation_count": gate_reason_counts.get("contract_violation", 0),
        "contract_regression_count": gate_reason_counts.get("contract_violation", 0),
        "path_cost_regression_count": gate_reason_counts.get("path_cost_regression", 0),
        "risk_regression_count": gate_reason_counts.get("risk_regression", 0),
        "source_selection_regression_count": gate_reason_counts.get("source_selection_regression", 0),
    }
    shadow_context_count = sum(len(group.get("candidates", [])) > 0 for group in scenario_groups)
    policy_decision_count = len(decisions)
    if shadow_context_count < _int_value(validation.get("min_shadow_context_count")):
        _append_reason(reason_codes, "quasi_real_shadow_context_count_below_threshold")
    if len(roi_groups) < _int_value(validation.get("min_roi_group_count")):
        _append_reason(reason_codes, "quasi_real_shadow_roi_group_count_below_threshold")
    if policy_decision_count != shadow_context_count:
        _append_reason(reason_codes, "quasi_real_shadow_policy_scoring_failed")
    if context_id_missing_count:
        _append_reason(reason_codes, "quasi_real_shadow_context_id_missing")
    for field, reason in (
        ("invalid_action_mask_count", "quasi_real_shadow_action_mask_contract_gap"),
        ("fallback_or_open_grid_count", "quasi_real_shadow_gate_regression"),
        ("safety_regression_count", "quasi_real_shadow_gate_regression"),
        ("contract_violation_count", "quasi_real_shadow_gate_regression"),
        ("path_cost_regression_count", "quasi_real_shadow_gate_regression"),
        ("risk_regression_count", "quasi_real_shadow_gate_regression"),
        ("source_selection_regression_count", "quasi_real_shadow_gate_regression"),
    ):
        if metric_counts[field] > _int_value(validation.get(f"max_{field}", 0)):
            _append_reason(reason_codes, reason)
    rejected_count = counts.get("policy_changed_gate_rejected", 0)
    if rejected_count and "quasi_real_shadow_gate_regression" not in reason_codes:
        _append_reason(reason_codes, "quasi_real_shadow_policy_alignment_refinement_required")
    behavior_verdict = _behavior_verdict(reason_codes)
    unique_reasons = _unique(reason_codes)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "status": "passed" if not unique_reasons else "failed",
        "reason_codes": unique_reasons,
        "behavior_verdict": behavior_verdict,
        "next_required_change": None if not unique_reasons else behavior_verdict,
        "source_root": _display_path(source_root, repo_root),
        "candidate_root": _display_path(candidate_root, repo_root),
        "quasi_real_root": _display_path(quasi_real_root, repo_root),
        "output_root": _display_path(output_root, repo_root),
        "decisions_path": _display_path(output_paths["decisions"], repo_root),
        "shadow_context_count": shadow_context_count,
        "policy_decision_count": policy_decision_count,
        "policy_source_aligned_count": counts.get("source_aligned", 0),
        "policy_changed_decision_count": counts.get("policy_changed_gate_passed", 0)
        + counts.get("policy_changed_gate_rejected", 0),
        "policy_changed_gate_passed_count": counts.get("policy_changed_gate_passed", 0),
        "policy_changed_gate_rejected_count": counts.get("policy_changed_gate_rejected", 0),
        "not_scored_count": counts.get("not_scored", 0),
        "roi_group_count": len(roi_groups),
        "roi_groups": roi_groups,
        "roi_group_shadow_coverage": roi_group_shadow_coverage,
        "context_id_missing_count": context_id_missing_count,
        "open_grid_fallback_count": metric_counts["fallback_or_open_grid_count"],
        **metric_counts,
        "scoring_error": scoring_error,
        "policy_takes_control": False,
        "runs_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "git_provenance": {"current": _git_snapshot(repo_root), "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }


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
        "rejections": rejections,
    }


def _group_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Quasi-Real Shadow Policy Behavior Audit",
        "",
        f"- status: {summary['status']}",
        f"- behavior_verdict: {summary['behavior_verdict']}",
        f"- shadow_context_count: {summary['shadow_context_count']}",
        f"- policy_changed_gate_passed_count: {summary['policy_changed_gate_passed_count']}",
        "",
        "| roi_group | shadow decisions |",
        "|---|---:|",
    ]
    for roi_group, count in sorted(summary["roi_group_shadow_coverage"].items()):
        lines.append(f"| {roi_group} | {count} |")
    lines.append("")
    lines.append("Shadow audit only: policy does not take control and no PPO update is run.")
    return "\n".join(lines)


def _input_paths(source_root: Path, candidate_root: Path, quasi_real_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    inputs = config["input_files"]
    return {
        "source_batch_summary": source_root / inputs["source_batch_summary"],
        "quasi_real_path_feedback_summary": quasi_real_root / inputs["quasi_real_path_feedback_summary"],
        "quasi_real_slices": quasi_real_root / inputs["quasi_real_slices"],
        "quasi_real_domain_gap_summary": quasi_real_root / inputs["quasi_real_domain_gap_summary"],
        "candidate_summary": candidate_root / inputs["candidate_summary"],
        "checkpoint": candidate_root / inputs["checkpoint"],
        "checkpoint_metadata": candidate_root / inputs["checkpoint_metadata"],
    }


def _output_paths(output_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["output_files"]
    return {
        "decisions": output_root / outputs["decisions"],
        "summary": output_root / outputs["summary"],
        "rejection_report": output_root / outputs["rejection_report"],
        "group_report": output_root / outputs["group_report"],
    }


def _scoring_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "evaluation": dict(config.get("evaluation", {})),
        "validation": dict(config.get("validation", {})),
    }


def _ensure_model_explorer_src(repo_root: Path) -> None:
    model_explorer_src = str(repo_root / "model-explorer" / "src")
    if model_explorer_src not in sys.path:
        sys.path.insert(0, model_explorer_src)


def _load_slices(path: Path, reason_codes: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        _append_reason(reason_codes, "quasi_real_slices_missing")
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            _append_reason(reason_codes, "quasi_real_slices_invalid_json")
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _raw_policy_changed(decision: dict[str, Any]) -> bool:
    source_context = decision.get("source_selected_context_id")
    raw_context = decision.get("raw_policy_selected_context_id")
    if source_context and raw_context:
        return source_context != raw_context
    return decision.get("source_selected_action_index") != decision.get("raw_policy_selected_action_index")


def _behavior_verdict(reason_codes: list[str]) -> str:
    reasons = set(reason_codes)
    if not reasons:
        return "acceptable_for_quasi_real_guarded_pilot"
    if "quasi_real_shadow_action_mask_contract_gap" in reasons:
        return "real_map_action_mask_contract_gap"
    if "quasi_real_shadow_policy_scoring_failed" in reasons or "quasi_real_domain_gap_not_acceptable" in reasons:
        return "real_map_bridge_or_feedback_gap"
    if "quasi_real_shadow_context_count_below_threshold" in reasons or "quasi_real_shadow_roi_group_count_below_threshold" in reasons:
        return "scenario_expansion_required"
    return "policy_real_map_alignment_refinement_required"


def _append_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run quasi-real shadow policy behavior audit.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--quasi-real-root", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    config = _load_config((repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    summary = run_quasi_real_shadow_policy_behavior_audit(
        source_root=(repo_root / args.source_root).resolve() if not Path(args.source_root).is_absolute() else Path(args.source_root),
        candidate_root=(repo_root / args.candidate_root).resolve() if not Path(args.candidate_root).is_absolute() else Path(args.candidate_root),
        quasi_real_root=(repo_root / args.quasi_real_root).resolve() if not Path(args.quasi_real_root).is_absolute() else Path(args.quasi_real_root),
        output_root=(repo_root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root),
        config=config,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "behavior_verdict": summary["behavior_verdict"],
                "shadow_context_count": summary["shadow_context_count"],
                "policy_decision_count": summary["policy_decision_count"],
                "summary": summary["summary_output"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
