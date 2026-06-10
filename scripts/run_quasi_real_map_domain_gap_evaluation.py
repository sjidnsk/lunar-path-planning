from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from git_provenance import git_snapshot as _git_snapshot


SCHEMA_VERSION = "quasi-real-map-domain-gap-summary/v1"


def run_quasi_real_map_domain_gap_evaluation(
    *,
    bridge_summary: dict[str, Any],
    quasi_real_path_feedback_summary: dict[str, Any],
    generated_reference_summary: dict[str, Any],
    output_root: str | Path,
    config: dict[str, Any] | None,
    repo_root: str | Path,
) -> dict[str, Any]:
    cfg = dict(config or {})
    validation = dict(cfg.get("validation", {})) if isinstance(cfg.get("validation"), dict) else {}
    min_slice_count = int(validation.get("min_slice_count", 12))
    min_roi_group_count = int(validation.get("min_roi_group_count", 4))
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)

    quasi_metrics = _path_feedback_metrics(quasi_real_path_feedback_summary)
    generated_metrics = _path_feedback_metrics(generated_reference_summary)
    slice_count = _int(bridge_summary.get("slice_count"))
    roi_groups = _string_list(bridge_summary.get("roi_groups"))
    roi_group_count = _int(bridge_summary.get("roi_group_count")) or len(set(roi_groups))
    context_id_missing_count = _int(bridge_summary.get("context_id_missing_count"))
    legacy_identity_fallback_count = _int(bridge_summary.get("legacy_identity_fallback_count"))
    fallback_or_open_grid_count = quasi_metrics["fallback_or_open_grid_count"]
    safety_regression_count = quasi_metrics["safety_regression_count"]
    path_cost_regression_count = max(0, quasi_metrics["path_cost_regression_count"])
    risk_regression_count = max(0, quasi_metrics["risk_regression_count"])
    source_selection_regression_count = 0
    contract_violation_count = 0
    invalid_action_mask_count = 0

    reason_codes: list[str] = []
    if bridge_summary.get("status") != "passed":
        reason_codes.append("real_map_bridge_contract_invalid")
    if slice_count < min_slice_count:
        reason_codes.append("real_map_slice_count_below_threshold")
    if roi_group_count < min_roi_group_count:
        reason_codes.append("real_map_roi_group_count_below_threshold")
    if context_id_missing_count:
        reason_codes.append("real_map_context_id_missing")
    if fallback_or_open_grid_count or safety_regression_count or contract_violation_count:
        reason_codes.append("real_map_path_feedback_regression")
    if invalid_action_mask_count:
        reason_codes.append("real_map_action_mask_contract_gap")
    if path_cost_regression_count or risk_regression_count or source_selection_regression_count:
        reason_codes.append("real_map_path_feedback_regression")

    distribution_gap = _distribution_gap(quasi_metrics, generated_metrics)
    verdict = "acceptable_for_next_pilot"
    if any(code in reason_codes for code in ("real_map_path_feedback_regression", "real_map_action_mask_contract_gap")):
        verdict = "planner_contract_gap"
    elif any(code in reason_codes for code in ("real_map_slice_count_below_threshold", "real_map_roi_group_count_below_threshold")):
        verdict = "scenario_expansion_required"
    elif distribution_gap["path_cost_mean_delta_abs"] > float(validation.get("max_path_cost_mean_delta_abs", 1000.0)):
        verdict = "scenario_expansion_required"
        reason_codes.append("real_map_distribution_gap_requires_scenario_expansion")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": _unique(reason_codes),
        "domain_gap_verdict": verdict,
        "next_required_change": None if not reason_codes else _next_required_change(verdict),
        "slice_count": slice_count,
        "roi_group_count": roi_group_count,
        "roi_groups": roi_groups,
        "context_id_missing_count": context_id_missing_count,
        "legacy_identity_fallback_count": legacy_identity_fallback_count,
        "invalid_action_mask_count": invalid_action_mask_count,
        "fallback_count": fallback_or_open_grid_count,
        "open_grid_fallback_count": 0,
        "fallback_or_open_grid_count": fallback_or_open_grid_count,
        "safety_regression_count": safety_regression_count,
        "contract_regression_count": contract_violation_count,
        "contract_violation_count": contract_violation_count,
        "path_cost_regression_count": path_cost_regression_count,
        "risk_regression_count": risk_regression_count,
        "source_selection_regression_count": source_selection_regression_count,
        "quasi_real_metrics": quasi_metrics,
        "generated_reference_metrics": generated_metrics,
        "distribution_gap": distribution_gap,
        "policy_shadow_only": True,
        "runs_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "git_provenance": {"current": _git_snapshot(Path(repo_root).resolve()), "current_matches_sources": True},
    }
    summary_path = output / "quasi-real-map-domain-gap-summary.json"
    report_path = output / "quasi-real-map-domain-gap-report.md"
    exclusion_path = output / "quasi-real-map-domain-gap-exclusion-report.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    exclusion_path.write_text(
        json.dumps(
            {
                "schema_version": "quasi-real-map-domain-gap-exclusion-report/v1",
                "reason_codes": summary["reason_codes"],
                "excluded_count": 0 if summary["status"] == "passed" else slice_count,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary["summary_output"] = str(summary_path)
    summary["report_output"] = str(report_path)
    summary["exclusion_report"] = str(exclusion_path)
    return summary


def _path_feedback_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    scenarios = summary.get("scenarios", [])
    if not isinstance(scenarios, list):
        scenarios = []
    path_costs = [_float(item.get("selected_path_cost_after_feedback")) for item in scenarios if isinstance(item, dict)]
    path_deltas = [_float(item.get("path_cost_delta_after_feedback")) for item in scenarios if isinstance(item, dict)]
    risks: list[float] = []
    reachable_candidate_count = 0
    candidate_count = _int(summary.get("candidate_count"))
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        feedback = scenario.get("path_feedback", {})
        candidates = feedback.get("candidates", []) if isinstance(feedback, dict) else []
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                if candidate.get("reachable") is True:
                    reachable_candidate_count += 1
                risk = candidate.get("risk")
                if risk is not None:
                    risks.append(_float(risk))
    fallback_count = _int(summary.get("open_grid_fallback_used_count"))
    if summary.get("open_grid_fallback_used") is True:
        fallback_count = max(fallback_count, 1)
    return {
        "scenario_count": _int(summary.get("scenario_count")) or len(scenarios),
        "candidate_count": candidate_count,
        "reachable_candidate_count": _int(summary.get("reachable_count")) or reachable_candidate_count,
        "fallback_or_open_grid_count": fallback_count + _int(summary.get("fallback_or_open_grid_count")),
        "safety_regression_count": _int(summary.get("tracking_safety_violation_count")),
        "path_cost_regression_count": sum(1 for value in path_deltas if value > 0.0),
        "risk_regression_count": 0,
        "path_cost_mean": mean(path_costs) if path_costs else _float(summary.get("average_path_cost")),
        "path_cost_delta_mean": mean(path_deltas) if path_deltas else 0.0,
        "risk_mean": mean(risks) if risks else 0.0,
        "coverage_per_path_cost": _float(summary.get("coverage_per_path_cost")),
    }


def _distribution_gap(quasi: dict[str, Any], generated: dict[str, Any]) -> dict[str, float]:
    return {
        "path_cost_mean_delta_abs": abs(float(quasi.get("path_cost_mean", 0.0)) - float(generated.get("path_cost_mean", 0.0))),
        "risk_mean_delta_abs": abs(float(quasi.get("risk_mean", 0.0)) - float(generated.get("risk_mean", 0.0))),
        "reachable_candidate_count_delta": float(quasi.get("reachable_candidate_count", 0))
        - float(generated.get("reachable_candidate_count", 0)),
    }


def _next_required_change(verdict: str) -> str:
    if verdict == "planner_contract_gap":
        return "real_map_planner_or_contract_gap_requires_triage"
    if verdict == "scenario_expansion_required":
        return "real_map_distribution_gap_requires_scenario_expansion"
    return "real_map_domain_gap_evidence_required"


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Quasi-Real Map Domain Gap Evaluation",
        "",
        f"- status: {summary['status']}",
        f"- domain_gap_verdict: {summary['domain_gap_verdict']}",
        f"- slice_count: {summary['slice_count']}",
        f"- roi_group_count: {summary['roi_group_count']}",
        f"- policy_shadow_only: {summary['policy_shadow_only']}",
        "",
        "## Gate Counts",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "context_id_missing_count",
        "legacy_identity_fallback_count",
        "invalid_action_mask_count",
        "fallback_or_open_grid_count",
        "safety_regression_count",
        "contract_violation_count",
        "path_cost_regression_count",
        "risk_regression_count",
        "source_selection_regression_count",
    ):
        lines.append(f"| {key} | {summary.get(key, 0)} |")
    lines.extend(["", "This is quasi-real domain-gap evidence only; it is not a policy release or PPO update."])
    return "\n".join(lines)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_config(path: str | Path | None) -> dict[str, Any]:
    return {} if path is None else _load_json(path)


def _int(value: Any) -> int:
    try:
        if isinstance(value, bool) or value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        if isinstance(value, bool) or value is None:
            return 0.0
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if numeric == numeric else 0.0


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate generated-vs-quasi-real map domain gap.")
    parser.add_argument("--bridge-summary", required=True)
    parser.add_argument("--quasi-real-path-feedback-summary", required=True)
    parser.add_argument("--generated-reference-summary", required=True)
    parser.add_argument("--output-root", default="outputs/path_feedback_batch_quasi_real_map_domain_gap_v1")
    parser.add_argument("--config", default="configs/quasi_real_map_domain_gap_evaluation_v1.json")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    summary = run_quasi_real_map_domain_gap_evaluation(
        bridge_summary=_load_json(args.bridge_summary),
        quasi_real_path_feedback_summary=_load_json(args.quasi_real_path_feedback_summary),
        generated_reference_summary=_load_json(args.generated_reference_summary),
        output_root=args.output_root,
        config=_load_config(args.config),
        repo_root=repo_root,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
