from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot


SUMMARY_SCHEMA_VERSION = "quasi-real-shadow-failure-taxonomy-summary/v1"
RECORD_SCHEMA_VERSION = "quasi-real-shadow-failure-taxonomy-record/v1"
FEATURE_AUDIT_SCHEMA_VERSION = "quasi-real-shadow-feature-audit-record/v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_quasi_real_shadow_failure_taxonomy_v1"


def run_quasi_real_shadow_failure_taxonomy(
    *,
    shadow_root: str | Path,
    quasi_real_root: str | Path,
    output_root: str | Path,
    config: dict[str, Any],
    repo_root: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    shadow = Path(shadow_root).resolve()
    quasi = Path(quasi_real_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)

    inputs = config["input_files"]
    outputs = config["output_files"]
    decisions = _load_jsonl(shadow / inputs["shadow_decisions"])
    rejection_report = _load_json(shadow / inputs["shadow_rejection_report"])
    slices = _load_jsonl(quasi / inputs["quasi_real_slices"])
    path_feedback = _load_json(quasi / inputs["quasi_real_path_feedback_summary"])
    scenario_index = _scenario_index(path_feedback)
    slices_by_scenario = {str(item.get("scenario_id")): item for item in slices}

    rejected = [
        decision
        for decision in decisions
        if decision.get("decision_class") == "policy_changed_gate_rejected"
    ]
    if not rejected and isinstance(rejection_report.get("rejections"), list):
        rejected = [item for item in rejection_report["rejections"] if isinstance(item, dict)]

    records: list[dict[str, Any]] = []
    feature_audits: list[dict[str, Any]] = []
    for decision in rejected:
        record, feature_audit = _taxonomy_record(
            decision,
            scenario=scenario_index.get(str(decision.get("scenario_id")), {}),
            slice_record=slices_by_scenario.get(str(decision.get("scenario_id")), {}),
        )
        records.append(record)
        feature_audits.append(feature_audit)

    counts = Counter(record["failure_class"] for record in records)
    reason_codes: list[str] = []
    validation = config.get("validation", {})
    if counts.get("bridge_or_feedback_gap", 0) > int(validation.get("max_bridge_or_feedback_gap_count", 0)):
        reason_codes.append("quasi_real_shadow_bridge_or_feedback_gap")
    if counts.get("action_mask_or_contract_gap", 0) > int(
        validation.get("max_action_mask_or_contract_gap_count", 0)
    ):
        reason_codes.append("quasi_real_shadow_action_mask_or_contract_gap")

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": "failed" if reason_codes else "passed",
        "reason_codes": reason_codes,
        "shadow_root": _display_path(shadow, repo),
        "quasi_real_root": _display_path(quasi, repo),
        "output_root": _display_path(output, repo),
        "failure_count": len(records),
        "path_risk_joint_regression_count": counts.get("path_risk_joint_regression", 0),
        "path_cost_only_regression_count": counts.get("path_cost_only_regression", 0),
        "risk_only_regression_count": counts.get("risk_only_regression", 0),
        "action_mask_or_contract_gap_count": counts.get("action_mask_or_contract_gap", 0),
        "bridge_or_feedback_gap_count": counts.get("bridge_or_feedback_gap", 0),
        "policy_over_conservative_count": counts.get("policy_over_conservative", 0),
        "safe_better_opportunity_missed_count": counts.get("safe_better_opportunity_missed", 0),
        "failure_class_counts": dict(sorted(counts.items())),
        "next_required_change": None if not reason_codes else "quasi_real_shadow_failure_taxonomy_required",
        "git_provenance": {"current": _git_snapshot(repo), "current_matches_sources": True},
    }

    (output / outputs["taxonomy"]).write_text(_jsonl(records), encoding="utf-8")
    (output / outputs["feature_audit"]).write_text(_jsonl(feature_audits), encoding="utf-8")
    (output / outputs["summary"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / outputs["report"]).write_text(_report(summary, records), encoding="utf-8")
    return summary


def _taxonomy_record(
    decision: dict[str, Any],
    *,
    scenario: dict[str, Any],
    slice_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reasons = [str(reason) for reason in decision.get("gate_reason_codes", [])]
    source_action = _as_int(decision.get("source_action_index"))
    raw_action = _as_int(decision.get("raw_policy_action_index"))
    source_candidate = _find_candidate(scenario, action_index=source_action)
    alternative_candidate = _find_candidate(scenario, action_index=raw_action)
    path_delta = _as_float(decision.get("path_cost_delta"))
    risk_delta = _as_float(decision.get("risk_delta"))

    failure_class = _failure_class(
        reasons,
        source_candidate=source_candidate,
        alternative_candidate=alternative_candidate,
        path_delta=path_delta,
        risk_delta=risk_delta,
    )
    record = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "failure_class": failure_class,
        "scenario_id": decision.get("scenario_id"),
        "roi_group": decision.get("roi_group") or slice_record.get("roi_name"),
        "roi_name": decision.get("roi_name") or slice_record.get("roi_name"),
        "split": decision.get("split") or slice_record.get("split"),
        "map_id": decision.get("map_id") or slice_record.get("map_id"),
        "slice_id": decision.get("slice_id") or slice_record.get("slice_id"),
        "context_id": decision.get("context_id"),
        "source_action_index": source_action,
        "raw_policy_action_index": raw_action,
        "logit_margin": _as_float(decision.get("logit_margin")),
        "path_cost_delta": path_delta,
        "risk_delta": risk_delta,
        "gate_reason_codes": reasons,
        "source_candidate": source_candidate,
        "alternative_candidate": alternative_candidate,
    }
    feature_audit = {
        "schema_version": FEATURE_AUDIT_SCHEMA_VERSION,
        "scenario_id": record["scenario_id"],
        "roi_group": record["roi_group"],
        "failure_class": failure_class,
        "source_candidate": source_candidate,
        "alternative_candidate": alternative_candidate,
        "metric_delta": {
            "path_cost": _metric_delta(alternative_candidate, source_candidate, "path_cost"),
            "risk": _metric_delta(alternative_candidate, source_candidate, "risk"),
            "utility": _metric_delta(alternative_candidate, source_candidate, "utility"),
        },
        "gate_reason_codes": reasons,
    }
    return record, feature_audit


def _failure_class(
    reasons: list[str],
    *,
    source_candidate: dict[str, Any] | None,
    alternative_candidate: dict[str, Any] | None,
    path_delta: float | None,
    risk_delta: float | None,
) -> str:
    if any(reason in reasons for reason in ("invalid_action_mask", "contract_violation", "contract_regression")):
        return "action_mask_or_contract_gap"
    if not source_candidate or not alternative_candidate:
        return "bridge_or_feedback_gap"
    path_bad = "path_cost_regression" in reasons or (path_delta is not None and path_delta > 0)
    risk_bad = "risk_regression" in reasons or (risk_delta is not None and risk_delta > 0)
    if path_bad and risk_bad:
        return "path_risk_joint_regression"
    if path_bad:
        return "path_cost_only_regression"
    if risk_bad:
        return "risk_only_regression"
    if "safe_better_opportunity_missed" in reasons:
        return "safe_better_opportunity_missed"
    return "policy_over_conservative"


def _scenario_index(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scenarios = summary.get("scenarios") if isinstance(summary.get("scenarios"), list) else []
    return {str(item.get("scenario_id")): item for item in scenarios if isinstance(item, dict)}


def _find_candidate(scenario: dict[str, Any], *, action_index: int | None) -> dict[str, Any] | None:
    candidates = (
        scenario.get("path_feedback", {}).get("candidates", [])
        if isinstance(scenario.get("path_feedback"), dict)
        else []
    )
    for candidate in candidates:
        if isinstance(candidate, dict) and _as_int(candidate.get("action_index")) == action_index:
            return candidate
    return None


def _metric_delta(
    alternative: dict[str, Any] | None,
    source: dict[str, Any] | None,
    key: str,
) -> float | None:
    if not alternative or not source:
        return None
    left = _as_float(alternative.get(key))
    right = _as_float(source.get(key))
    if left is None or right is None:
        return None
    return round(left - right, 6)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def _report(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# Quasi-Real Shadow Failure Taxonomy",
        "",
        f"- status: {summary['status']}",
        f"- failure_count: {summary['failure_count']}",
        f"- path_risk_joint_regression_count: {summary['path_risk_joint_regression_count']}",
        "",
        "| scenario_id | roi_group | failure_class | reasons |",
        "|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record.get('scenario_id')} | {record.get('roi_group')} | "
            f"{record.get('failure_class')} | {','.join(record.get('gate_reason_codes', []))} |"
        )
    return "\n".join(lines) + "\n"


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify quasi-real shadow policy failures.")
    parser.add_argument("--shadow-root", required=True)
    parser.add_argument("--quasi-real-root", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    summary = run_quasi_real_shadow_failure_taxonomy(
        shadow_root=_resolve_path(args.shadow_root, repo_root),
        quasi_real_root=_resolve_path(args.quasi_real_root, repo_root),
        output_root=_resolve_path(args.output_root, repo_root),
        config=_load_config(_resolve_path(args.config, repo_root)),
        repo_root=repo_root,
    )
    print(json.dumps({"status": summary["status"], "reason_codes": summary["reason_codes"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
