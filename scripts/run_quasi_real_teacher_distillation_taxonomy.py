from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot
from run_quasi_real_shadow_policy_behavior_audit import _quasi_real_scenario_groups
from run_scenario_disjoint_policy_rollout_evaluation import (
    _action_index,
    _regression_reasons,
    _same_candidate,
    _source_selected_candidate,
)


SUMMARY_SCHEMA_VERSION = "quasi-real-teacher-distillation-taxonomy-summary/v1"
RECORD_SCHEMA_VERSION = "quasi-real-teacher-distillation-taxonomy-record/v1"
FEATURE_AUDIT_SCHEMA_VERSION = "quasi-real-teacher-distillation-feature-audit-record/v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_quasi_real_teacher_distillation_taxonomy_v1"


def run_quasi_real_teacher_distillation_taxonomy(
    *,
    teacher_root: str | Path,
    quasi_real_root: str | Path,
    output_root: str | Path,
    config: dict[str, Any],
    repo_root: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    teacher = Path(teacher_root).resolve()
    quasi = Path(quasi_real_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)

    inputs = config["input_files"]
    outputs = config["output_files"]
    teacher_summary = _load_json(teacher / inputs["teacher_equivalent_summary"])
    decisions = _load_jsonl(teacher / inputs["teacher_equivalent_decisions"])
    path_feedback = _load_json(quasi / inputs["quasi_real_path_feedback_summary"])
    slices = _load_jsonl(quasi / inputs["quasi_real_slices"])
    scenarios = _scenario_index(path_feedback)
    slices_by_scenario_id = {str(item.get("scenario_id")): item for item in slices}
    scenario_groups = _quasi_real_scenario_groups(
        quasi_summary=path_feedback,
        slices_by_scenario_id=slices_by_scenario_id,
        summary_path=quasi / inputs["quasi_real_path_feedback_summary"],
        repo_root=repo,
    )

    unsafe_decisions = [
        decision
        for decision in decisions
        if decision.get("unsafe_disagreement")
        or decision.get("decision_class") == "policy_changed_gate_rejected"
    ]
    raw_records: list[dict[str, Any]] = []
    feature_audits_by_key: dict[str, dict[str, Any]] = {}
    for decision in unsafe_decisions:
        scenario_id = str(decision.get("scenario_id"))
        record, audit = _taxonomy_record(
            decision,
            scenario=scenarios.get(scenario_id, {}),
            slice_record=slices_by_scenario_id.get(scenario_id, {}),
        )
        record["taxonomy_record_id"] = _record_key(record)
        record["raw_policy_selected_failure"] = True
        record["distillation_record_kind"] = "raw_policy_unsafe_disagreement"
        raw_records.append(record)
        feature_audits_by_key[_record_key(record)] = audit

    distillation_records = _distillation_pair_records(
        scenario_groups=scenario_groups,
        slices_by_scenario_id=slices_by_scenario_id,
        config=config,
        raw_records=raw_records,
    )
    for record in distillation_records:
        _, audit = _audit_from_record(record)
        feature_audits_by_key[_record_key(record)] = audit

    raw_counts = Counter(record["failure_class"] for record in raw_records)
    distillation_counts = Counter(record["failure_class"] for record in distillation_records)
    reason_codes: list[str] = []
    validation = config.get("validation", {})
    if len(unsafe_decisions) < int(validation.get("min_unsafe_disagreement_count", 0)):
        reason_codes.append("quasi_real_teacher_distillation_signal_insufficient")
    if len(raw_records) != len(unsafe_decisions):
        reason_codes.append("quasi_real_teacher_distillation_signal_insufficient")
    if raw_counts.get("bridge_or_feedback_gap", 0) > int(validation.get("max_bridge_or_feedback_gap_count", 0)):
        reason_codes.append("quasi_real_teacher_distillation_feature_schema_refinement_required")
    if raw_counts.get("action_mask_or_contract_gap", 0) > int(
        validation.get("max_action_mask_or_contract_gap_count", 0)
    ):
        reason_codes.append("quasi_real_teacher_distillation_feature_schema_refinement_required")
    if not distillation_records:
        reason_codes.append("quasi_real_teacher_distillation_signal_insufficient")

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": "failed" if reason_codes else "passed",
        "reason_codes": sorted(set(reason_codes)),
        "teacher_root": _display_path(teacher, repo),
        "quasi_real_root": _display_path(quasi, repo),
        "output_root": _display_path(output, repo),
        "teacher_equivalent_status": teacher_summary.get("status"),
        "unsafe_disagreement_count": len(unsafe_decisions),
        "classified_disagreement_count": len(raw_records),
        "path_cost_only_regression_count": raw_counts.get("path_cost_only_regression", 0),
        "path_risk_joint_regression_count": raw_counts.get("path_risk_joint_regression", 0),
        "risk_only_regression_count": raw_counts.get("risk_only_regression", 0),
        "action_mask_or_contract_gap_count": raw_counts.get("action_mask_or_contract_gap", 0),
        "bridge_or_feedback_gap_count": raw_counts.get("bridge_or_feedback_gap", 0),
        "failure_class_counts": dict(sorted(raw_counts.items())),
        "distillation_candidate_pair_count": len(distillation_records),
        "distillation_failure_class_counts": dict(sorted(distillation_counts.items())),
        "next_required_change": None if not reason_codes else "quasi_real_teacher_distillation_signal_insufficient",
        "git_provenance": {"current": _git_snapshot(repo), "current_matches_sources": True},
    }
    (output / outputs["taxonomy"]).write_text(_jsonl(distillation_records), encoding="utf-8")
    (output / outputs["feature_audit"]).write_text(
        _jsonl(list(feature_audits_by_key.values())),
        encoding="utf-8",
    )
    (output / outputs["summary"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / outputs["report"]).write_text(_report(summary, raw_records, distillation_records), encoding="utf-8")
    return summary


def _distillation_pair_records(
    *,
    scenario_groups: list[dict[str, Any]],
    slices_by_scenario_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
    raw_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_keys = {
        (
            str(record.get("scenario_id")),
            str((record.get("source_candidate") or {}).get("context_id")),
            str((record.get("alternative_candidate") or {}).get("context_id")),
        )
        for record in raw_records
    }
    records: list[dict[str, Any]] = []
    for group in scenario_groups:
        source = _source_selected_candidate(group)
        if not source:
            continue
        slice_record = slices_by_scenario_id.get(str(group.get("scenario_id")), {})
        for candidate in group.get("candidates", []):
            if _same_candidate(candidate, source):
                continue
            reasons = _regression_reasons(candidate, source, config={"evaluation": dict(config.get("evaluation", {}))})
            if "path_cost_regression" not in reasons and "risk_regression" not in reasons:
                continue
            path_delta = _metric_delta(candidate, source, "path_cost")
            risk_delta = _metric_delta(candidate, source, "risk")
            failure_class = _failure_class(
                reasons,
                source_candidate=source,
                alternative_candidate=candidate,
                path_delta=path_delta,
                risk_delta=risk_delta,
            )
            key = (str(group.get("scenario_id")), str(source.get("context_id")), str(candidate.get("context_id")))
            record = {
                "schema_version": RECORD_SCHEMA_VERSION,
                "failure_class": failure_class,
                "scenario_id": group.get("scenario_id"),
                "roi_group": slice_record.get("roi_name") or group.get("scenario_group"),
                "roi_name": slice_record.get("roi_name") or group.get("scenario_group"),
                "split": slice_record.get("split"),
                "map_id": slice_record.get("map_id"),
                "slice_id": slice_record.get("slice_id") or group.get("scenario_id"),
                "context_id": source.get("context_id"),
                "source_action_index": _action_index(source),
                "raw_policy_action_index": _action_index(candidate),
                "logit_margin": None,
                "path_cost_delta": path_delta,
                "risk_delta": risk_delta,
                "gate_reason_codes": reasons,
                "source_candidate": source,
                "alternative_candidate": candidate,
                "raw_policy_selected_failure": key in raw_keys,
                "distillation_record_kind": "teacher_vs_regressive_alternative",
            }
            record["taxonomy_record_id"] = _record_key(record)
            records.append(record)
    return records


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
        "logit_margin": path_delta if decision.get("logit_margin") is None else _as_float(decision.get("logit_margin")),
        "path_cost_delta": path_delta,
        "risk_delta": risk_delta,
        "gate_reason_codes": reasons,
        "source_candidate": source_candidate,
        "alternative_candidate": alternative_candidate,
    }
    audit = {
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
    return record, audit


def _audit_from_record(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_candidate = record.get("source_candidate") if isinstance(record.get("source_candidate"), dict) else None
    alternative_candidate = (
        record.get("alternative_candidate")
        if isinstance(record.get("alternative_candidate"), dict)
        else None
    )
    audit = {
        "schema_version": FEATURE_AUDIT_SCHEMA_VERSION,
        "scenario_id": record.get("scenario_id"),
        "roi_group": record.get("roi_group"),
        "failure_class": record.get("failure_class"),
        "source_candidate": source_candidate,
        "alternative_candidate": alternative_candidate,
        "metric_delta": {
            "path_cost": _metric_delta(alternative_candidate, source_candidate, "path_cost"),
            "risk": _metric_delta(alternative_candidate, source_candidate, "risk"),
            "utility": _metric_delta(alternative_candidate, source_candidate, "utility"),
        },
        "gate_reason_codes": record.get("gate_reason_codes", []),
    }
    return record, audit


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
    return "bridge_or_feedback_gap"


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


def _metric_delta(alternative: dict[str, Any] | None, source: dict[str, Any] | None, key: str) -> float | None:
    if not alternative or not source:
        return None
    left = _as_float(alternative.get(key))
    right = _as_float(source.get(key))
    if left is None or right is None:
        return None
    return left - right


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def _report(
    summary: dict[str, Any],
    raw_records: list[dict[str, Any]],
    distillation_records: list[dict[str, Any]],
) -> str:
    lines = [
        "# Quasi-Real Teacher Distillation Taxonomy",
        "",
        f"- status: {summary['status']}",
        f"- unsafe_disagreement_count: {summary['unsafe_disagreement_count']}",
        f"- classified_disagreement_count: {summary['classified_disagreement_count']}",
        f"- distillation_candidate_pair_count: {summary['distillation_candidate_pair_count']}",
        "",
        "## Raw unsafe disagreements",
        "",
        "| scenario_id | roi_group | split | failure_class | reasons |",
        "|---|---|---|---|---|",
    ]
    for record in raw_records:
        lines.append(
            f"| {record.get('scenario_id')} | {record.get('roi_group')} | {record.get('split')} | "
            f"{record.get('failure_class')} | {','.join(record.get('gate_reason_codes', []))} |"
        )
    lines.extend(
        [
            "",
            "## Distillation candidate-pair counts",
            "",
            "| failure_class | count |",
            "|---|---:|",
        ]
    )
    for failure_class, count in sorted(Counter(record.get("failure_class") for record in distillation_records).items()):
        lines.append(f"| {failure_class} | {count} |")
    return "\n".join(lines) + "\n"


def _record_key(record: dict[str, Any]) -> str:
    source = record.get("source_candidate") if isinstance(record.get("source_candidate"), dict) else {}
    alternative = (
        record.get("alternative_candidate")
        if isinstance(record.get("alternative_candidate"), dict)
        else {}
    )
    return "::".join(
        [
            str(record.get("scenario_id")),
            str(source.get("context_id")),
            str(alternative.get("context_id")),
        ]
    )


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


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify quasi-real teacher-equivalent unsafe disagreements.")
    parser.add_argument("--teacher-root", required=True)
    parser.add_argument("--quasi-real-root", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    config = json.loads(_resolve_path(args.config, repo_root).read_text(encoding="utf-8"))
    summary = run_quasi_real_teacher_distillation_taxonomy(
        teacher_root=_resolve_path(args.teacher_root, repo_root),
        quasi_real_root=_resolve_path(args.quasi_real_root, repo_root),
        output_root=_resolve_path(args.output_root, repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(json.dumps({"status": summary["status"], "reason_codes": summary["reason_codes"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
