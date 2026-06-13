from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot


CONFIG_SCHEMA_VERSION = "generated-sequential-gate-metric-accounting-audit-config/v1"
SUMMARY_SCHEMA_VERSION = "generated-sequential-gate-metric-accounting-audit-summary/v1"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit generated sequential gate metric/accounting origins."
    )
    parser.add_argument("--diagnosis-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    try:
        config = _load_config(_resolve_path(args.config, repo_root))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.validate_only:
        print(json.dumps({"status": "config validated", "config": args.config}, ensure_ascii=False))
        return 0

    summary = run_generated_sequential_gate_metric_accounting_audit(
        diagnosis_root=_resolve_path(args.diagnosis_root, repo_root),
        output_root=_resolve_path(args.output_root, repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "legacy_mismatch_count": summary["legacy_mismatch_count"],
                "diagnosis_verdict_after_origin_split": summary["diagnosis_verdict_after_origin_split"],
                "recommended_next_action": summary["recommended_next_action"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def run_generated_sequential_gate_metric_accounting_audit(
    *,
    diagnosis_root: Path,
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root, config)
    inputs = _input_paths(diagnosis_root, config)
    reason_codes: list[str] = []

    diagnosis = _load_json_required(inputs["diagnosis_summary"], reason_codes, "diagnosis_summary")
    comparison_rows = _read_jsonl_required(inputs["failed_step_comparison"], reason_codes, "failed_step_comparison")
    base_report = _load_json_required(inputs["base_rejection_report"], reason_codes, "base_rejection_report")
    updated_report = _load_json_required(inputs["updated_rejection_report"], reason_codes, "updated_rejection_report")
    base_steps = _failed_steps(base_report)
    updated_steps = _failed_steps(updated_report)

    evaluation = config.get("evaluation", {})
    max_path = float(evaluation.get("max_path_cost_regression", 0.0))
    max_risk = float(evaluation.get("max_risk_regression", 0.0))

    accounting_rows = [
        _origin_accounting_row(step, max_path=max_path, max_risk=max_risk)
        for step in updated_steps
    ]
    legacy_mismatch_rows = [
        row
        for row in accounting_rows
        if row["raw_policy_probe_regression"] and not row["controlled_rollout_regression"]
    ]

    raw_counts = _origin_reason_counts(accounting_rows, "raw_policy_probe")
    controlled_counts = _origin_reason_counts(accounting_rows, "controlled_rollout")
    canary_counts = _origin_reason_counts(accounting_rows, "canary_gate")
    corrected = _corrected_shadow_summary(
        updated_report=updated_report,
        updated_steps=updated_steps,
        diagnosis=diagnosis,
        raw_counts=raw_counts,
        controlled_counts=controlled_counts,
    )
    verdict = _origin_split_verdict(
        diagnosis=diagnosis,
        base_steps=base_steps,
        updated_steps=updated_steps,
        controlled_mismatch_count=sum(1 for row in accounting_rows if row["controlled_metric_mismatch"]),
    )
    recommended_next_action = _recommended_next_action(verdict)

    expected_mismatches = config.get("validation", {}).get("expected_legacy_mismatch_count")
    if expected_mismatches is not None and len(legacy_mismatch_rows) != int(expected_mismatches):
        _append_reason(reason_codes, "legacy_mismatch_count_unexpected")
    if diagnosis.get("status") != "passed" or _string_list(diagnosis.get("reason_codes")):
        _append_reason(reason_codes, "compatibility_diagnosis_not_passed")
    if not updated_steps:
        _append_reason(reason_codes, "updated_failed_steps_missing")

    _write_jsonl(paths["legacy_mismatch_rows"], legacy_mismatch_rows)
    _write_jsonl(paths["origin_aware_failed_step_accounting"], accounting_rows)
    _write_json(paths["corrected_shadow_summary"], corrected)

    family_counts = Counter(row["scenario_group"] for row in accounting_rows)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": "failed" if reason_codes else "passed",
        "reason_codes": reason_codes,
        "diagnosis_root": _display_path(diagnosis_root, repo_root),
        "output_root": _display_path(output_root, repo_root),
        "diagnosis_summary": _display_path(inputs["diagnosis_summary"], repo_root),
        "failed_step_comparison": _display_path(inputs["failed_step_comparison"], repo_root),
        "base_rejection_report": _display_path(inputs["base_rejection_report"], repo_root),
        "updated_rejection_report": _display_path(inputs["updated_rejection_report"], repo_root),
        "failed_step_count": len(updated_steps),
        "base_failed_step_count": len(base_steps),
        "updated_failed_step_count": len(updated_steps),
        "legacy_mismatch_count": len(legacy_mismatch_rows),
        "raw_policy_path_cost_regression_count": raw_counts.get("path_cost_regression", 0),
        "raw_policy_risk_regression_count": raw_counts.get("risk_regression", 0),
        "controlled_path_cost_regression_count": controlled_counts.get("path_cost_regression", 0),
        "controlled_risk_regression_count": controlled_counts.get("risk_regression", 0),
        "canary_rejection_reason_counts": dict(sorted(canary_counts.items())),
        "raw_policy_regression_reason_counts": dict(sorted(raw_counts.items())),
        "controlled_regression_reason_counts": dict(sorted(controlled_counts.items())),
        "failed_family_counts": dict(sorted(family_counts.items())),
        "diagnosis_verdict_before_origin_split": diagnosis.get("diagnosis_verdict"),
        "diagnosis_verdict_after_origin_split": verdict,
        "recommended_next_action": recommended_next_action,
        "corrected_shadow_summary": _display_path(paths["corrected_shadow_summary"], repo_root),
        "legacy_mismatch_rows": _display_path(paths["legacy_mismatch_rows"], repo_root),
        "origin_aware_failed_step_accounting": _display_path(
            paths["origin_aware_failed_step_accounting"],
            repo_root,
        ),
        "report": _display_path(paths["report"], repo_root),
        "summary": _display_path(paths["summary"], repo_root),
        "comparison_row_count": len(comparison_rows),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "runs_ppo_update": False,
        "git_provenance": {"current": _git_snapshot(repo_root), "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }
    _write_report(paths["report"], summary, accounting_rows, legacy_mismatch_rows)
    _write_json(paths["summary"], summary)
    return summary


def _origin_accounting_row(step: dict[str, Any], *, max_path: float, max_risk: float) -> dict[str, Any]:
    canary_reasons = set(_string_list(step.get("canary_rejection_reason_codes")))
    raw_reasons = set(_string_list(step.get("raw_policy_regression_reason_codes"))) or set(canary_reasons)
    controlled_reasons = set(_string_list(step.get("controlled_regression_reason_codes")))
    raw_path_delta = _float_value(step.get("raw_policy_selected_path_cost_delta"))
    raw_risk_delta = _float_value(step.get("raw_policy_selected_risk_delta"))
    controlled_path_delta = _float_value(step.get("policy_selected_path_cost_delta"))
    controlled_risk_delta = _float_value(step.get("policy_selected_risk_delta"))

    raw_path_regression = "path_cost_regression" in raw_reasons and raw_path_delta > max_path
    raw_risk_regression = "risk_regression" in raw_reasons and raw_risk_delta > max_risk
    controlled_path_regression = "path_cost_regression" in controlled_reasons and controlled_path_delta > max_path
    controlled_risk_regression = "risk_regression" in controlled_reasons and controlled_risk_delta > max_risk
    controlled_metric_mismatch = (
        ("path_cost_regression" in controlled_reasons and controlled_path_delta <= max_path)
        or ("risk_regression" in controlled_reasons and controlled_risk_delta <= max_risk)
    )

    if controlled_path_regression or controlled_risk_regression:
        origin = "controlled_rollout"
    elif raw_path_regression or raw_risk_regression:
        origin = "raw_policy_probe"
    elif canary_reasons:
        origin = "canary_gate"
    else:
        origin = "none"

    return {
        "episode_id": str(step.get("episode_id") or ""),
        "step_index": _int_value(step.get("step_index")),
        "scenario_id": step.get("scenario_id"),
        "scenario_group": str(step.get("scenario_group") or "unknown"),
        "decision_class": step.get("decision_class"),
        "controlled_choice_source": step.get("controlled_choice_source"),
        "reason_origin": origin,
        "canary_rejection_reason_codes": sorted(canary_reasons),
        "raw_policy_regression_reason_codes": sorted(raw_reasons),
        "controlled_regression_reason_codes": sorted(controlled_reasons),
        "raw_policy_probe_regression": bool(raw_path_regression or raw_risk_regression),
        "controlled_rollout_regression": bool(controlled_path_regression or controlled_risk_regression),
        "controlled_metric_mismatch": bool(controlled_metric_mismatch),
        "raw_policy_selected_path_cost_delta": raw_path_delta,
        "raw_policy_selected_risk_delta": raw_risk_delta,
        "policy_selected_path_cost_delta": controlled_path_delta,
        "policy_selected_risk_delta": controlled_risk_delta,
        "reason_origin_counts": {
            "canary_gate": _reason_count_map(canary_reasons),
            "raw_policy_probe": _reason_count_map(raw_reasons),
            "controlled_rollout": _reason_count_map(controlled_reasons),
        },
    }


def _corrected_shadow_summary(
    *,
    updated_report: dict[str, Any],
    updated_steps: list[dict[str, Any]],
    diagnosis: dict[str, Any],
    raw_counts: Counter[str],
    controlled_counts: Counter[str],
) -> dict[str, Any]:
    reason_codes = list(
        diagnosis.get("updated_generated_sequential_reason_codes")
        or updated_report.get("reason_codes")
        or []
    )
    if not controlled_counts.get("path_cost_regression"):
        reason_codes = [
            reason
            for reason in reason_codes
            if reason != "cumulative_path_cost_regression_count_above_threshold"
        ]
    if not controlled_counts.get("risk_regression"):
        reason_codes = [
            reason
            for reason in reason_codes
            if reason != "cumulative_risk_regression_count_above_threshold"
        ]
    if raw_counts and "canary_rejected_policy_choice_count_above_threshold" not in reason_codes:
        reason_codes.append("canary_rejected_policy_choice_count_above_threshold")
    return {
        "schema_version": "generated-sequential-corrected-accounting-shadow-summary/v1",
        "status": "failed" if reason_codes else "passed",
        "reason_codes": reason_codes,
        "canary_rejected_policy_choice_count": sum(
            1 for step in updated_steps if step.get("decision_class") == "canary_rejected_policy_choice"
        ),
        "raw_policy_path_cost_regression_count": raw_counts.get("path_cost_regression", 0),
        "raw_policy_risk_regression_count": raw_counts.get("risk_regression", 0),
        "controlled_path_cost_regression_count": controlled_counts.get("path_cost_regression", 0),
        "controlled_risk_regression_count": controlled_counts.get("risk_regression", 0),
        "cumulative_path_cost_regression_count": controlled_counts.get("path_cost_regression", 0),
        "cumulative_risk_regression_count": controlled_counts.get("risk_regression", 0),
    }


def _origin_split_verdict(
    *,
    diagnosis: dict[str, Any],
    base_steps: list[dict[str, Any]],
    updated_steps: list[dict[str, Any]],
    controlled_mismatch_count: int,
) -> str:
    if controlled_mismatch_count:
        return "gate_accounting_or_metric_mismatch"
    base_status = str(diagnosis.get("base_generated_sequential_status") or "")
    updated_status = str(diagnosis.get("updated_generated_sequential_status") or "")
    if base_status == "passed" and updated_status != "passed":
        return "ppo_update_induced_generated_regression"
    base_keys = {_step_key(step) for step in base_steps}
    updated_keys = {_step_key(step) for step in updated_steps}
    if base_status != "passed" and updated_status != "passed" and base_keys and base_keys == updated_keys:
        return "pre_existing_generated_sequential_contract_mismatch"
    return "diagnosis_inconclusive"


def _recommended_next_action(verdict: str) -> str:
    return {
        "ppo_update_induced_generated_regression": "update_objective_or_learning_rate_guard_required",
        "pre_existing_generated_sequential_contract_mismatch": "generated_sequential_contract_alignment_required",
        "gate_accounting_or_metric_mismatch": "generated_sequential_gate_metric_audit_required",
        "diagnosis_inconclusive": "manual_contract_compatibility_review_required",
    }.get(verdict, "manual_contract_compatibility_review_required")


def _write_report(
    path: Path,
    summary: dict[str, Any],
    accounting_rows: list[dict[str, Any]],
    legacy_mismatch_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Generated Sequential Gate Metric / Accounting Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- verdict_after_origin_split: `{summary['diagnosis_verdict_after_origin_split']}`",
        f"- recommended_next_action: `{summary['recommended_next_action']}`",
        f"- legacy_mismatch_count: `{summary['legacy_mismatch_count']}`",
        f"- raw_policy_path_cost_regression_count: `{summary['raw_policy_path_cost_regression_count']}`",
        f"- raw_policy_risk_regression_count: `{summary['raw_policy_risk_regression_count']}`",
        f"- controlled_path_cost_regression_count: `{summary['controlled_path_cost_regression_count']}`",
        f"- controlled_risk_regression_count: `{summary['controlled_risk_regression_count']}`",
        "",
        "## Failed Families",
    ]
    for family, count in summary["failed_family_counts"].items():
        lines.append(f"- `{family}`: {count}")
    lines.extend(["", "## Origin-Aware Failed Steps"])
    for row in accounting_rows:
        lines.append(
            f"- `{row['episode_id']}` step `{row['step_index']}` family "
            f"`{row['scenario_group']}` origin `{row['reason_origin']}`"
        )
    if legacy_mismatch_rows:
        lines.extend(["", "## Legacy Mismatch Rows"])
        for row in legacy_mismatch_rows:
            lines.append(
                f"- `{row['episode_id']}` step `{row['step_index']}`: raw probe rejected, "
                "controlled fallback has zero cumulative regression"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _paths(output_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["output_files"]
    return {
        "summary": output_root / outputs.get("summary", "generated-sequential-gate-metric-accounting-audit-summary.json"),
        "legacy_mismatch_rows": output_root / outputs.get("legacy_mismatch_rows", "legacy-mismatch-rows.jsonl"),
        "origin_aware_failed_step_accounting": output_root / outputs.get("origin_aware_failed_step_accounting", "origin-aware-failed-step-accounting.jsonl"),
        "corrected_shadow_summary": output_root / outputs.get("corrected_shadow_summary", "corrected-accounting-shadow-summary.json"),
        "report": output_root / outputs.get("report", "generated-sequential-gate-metric-accounting-report.md"),
    }


def _input_paths(diagnosis_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    inputs = config["input_files"]
    return {
        "diagnosis_summary": diagnosis_root / inputs.get("diagnosis_summary", "quasi-real-generated-sequential-contract-compatibility-summary.json"),
        "failed_step_comparison": diagnosis_root / inputs.get("failed_step_comparison", "failed-step-comparison.jsonl"),
        "base_rejection_report": diagnosis_root / inputs.get("base_rejection_report", "base_generated_sequential/policy-gated-sequential-canary-rejection-report.json"),
        "updated_rejection_report": diagnosis_root / inputs.get("updated_rejection_report", "updated_generated_sequential_replay/policy-gated-sequential-canary-rejection-report.json"),
    }


def _load_config(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    for section in ("input_files", "output_files", "evaluation", "validation"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _failed_steps(report: dict[str, Any]) -> list[dict[str, Any]]:
    failed = report.get("failed_steps")
    return [step for step in failed if isinstance(step, dict)] if isinstance(failed, list) else []


def _origin_reason_counts(rows: list[dict[str, Any]], origin: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row["reason_origin_counts"].get(origin, {}))
    return counts


def _reason_count_map(reasons: set[str]) -> dict[str, int]:
    return dict(sorted((reason, 1) for reason in reasons if reason))


def _load_json_required(path: Path, reason_codes: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return {}
    payload = _load_json(path)
    if not payload:
        _append_reason(reason_codes, f"{label}_invalid")
    return payload


def _read_jsonl_required(path: Path, reason_codes: list[str], label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return []
    rows = _read_jsonl(path)
    if not rows:
        _append_reason(reason_codes, f"{label}_empty")
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    rows = []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _step_key(step: dict[str, Any]) -> tuple[str, int]:
    return str(step.get("episode_id") or ""), _int_value(step.get("step_index"))


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
