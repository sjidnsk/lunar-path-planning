from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot


CONFIG_SCHEMA_VERSION = (
    "generated-sequential-long-horizon-teacher-skill-contract-alignment-config/v1"
)
SUMMARY_SCHEMA_VERSION = (
    "generated-sequential-long-horizon-teacher-skill-contract-summary/v1"
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Align generated sequential gates to long-horizon teacher-skill evidence."
    )
    parser.add_argument("--diagnosis-root", required=True)
    parser.add_argument("--accounting-audit-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--quasi-real-teacher-following-summary")
    parser.add_argument("--quasi-real-collector-summary")
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

    summary = run_generated_sequential_long_horizon_teacher_skill_contract_alignment(
        diagnosis_root=_resolve_path(args.diagnosis_root, repo_root),
        accounting_audit_root=_resolve_path(args.accounting_audit_root, repo_root),
        output_root=_resolve_path(args.output_root, repo_root),
        config=config,
        repo_root=repo_root,
        quasi_real_teacher_following_summary_path=(
            _resolve_path(args.quasi_real_teacher_following_summary, repo_root)
            if args.quasi_real_teacher_following_summary
            else None
        ),
        quasi_real_collector_summary_path=(
            _resolve_path(args.quasi_real_collector_summary, repo_root)
            if args.quasi_real_collector_summary
            else None
        ),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "verdict": summary["verdict"],
                "teacher_equivalent_episode_count": summary["teacher_equivalent_episode_count"],
                "beyond_teacher_episode_count": summary["beyond_teacher_episode_count"],
                "dominated_raw_choice_count": summary["dominated_raw_choice_count"],
                "controlled_regression_episode_count": summary["controlled_regression_episode_count"],
                "recommended_next_action": summary["recommended_next_action"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def run_generated_sequential_long_horizon_teacher_skill_contract_alignment(
    *,
    diagnosis_root: Path,
    accounting_audit_root: Path,
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    quasi_real_teacher_following_summary_path: Path | None = None,
    quasi_real_collector_summary_path: Path | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root, config)
    inputs = _input_paths(
        diagnosis_root=diagnosis_root,
        accounting_audit_root=accounting_audit_root,
        config=config,
        quasi_real_teacher_following_summary_path=quasi_real_teacher_following_summary_path,
        quasi_real_collector_summary_path=quasi_real_collector_summary_path,
    )
    reason_codes: list[str] = []

    compatibility = _load_json_required(
        inputs["compatibility_summary"],
        reason_codes,
        "compatibility_diagnosis_summary",
    )
    accounting = _load_json_required(
        inputs["accounting_audit_summary"],
        reason_codes,
        "accounting_audit_summary",
    )
    updated_steps = _read_jsonl_required(
        inputs["updated_steps"],
        reason_codes,
        "updated_generated_sequential_steps",
    )
    base_steps = _read_jsonl(inputs["base_steps"])
    updated_rejection_report = _load_json(inputs["updated_rejection_report"])
    base_rejection_report = _load_json(inputs["base_rejection_report"])
    quasi_teacher = _load_json_required(
        inputs["quasi_real_teacher_following_summary"],
        reason_codes,
        "quasi_real_teacher_following_summary",
    )
    quasi_collector = _load_json_required(
        inputs["quasi_real_collector_summary"],
        reason_codes,
        "quasi_real_collector_summary",
    )

    if compatibility and (
        compatibility.get("status") != "passed"
        or _string_list(compatibility.get("reason_codes"))
    ):
        _append_reason(reason_codes, "compatibility_diagnosis_not_passed")
    if compatibility and compatibility.get("diagnosis_verdict") not in {
        "pre_existing_generated_sequential_contract_mismatch",
        "gate_accounting_or_metric_mismatch",
    }:
        _append_reason(reason_codes, "compatibility_diagnosis_verdict_unexpected")
    if accounting and (
        accounting.get("status") != "passed"
        or _string_list(accounting.get("reason_codes"))
    ):
        _append_reason(reason_codes, "accounting_audit_not_passed")
    if accounting and accounting.get("diagnosis_verdict_after_origin_split") != (
        "pre_existing_generated_sequential_contract_mismatch"
    ):
        _append_reason(reason_codes, "accounting_audit_verdict_unresolved")
    if quasi_teacher and quasi_teacher.get("status") != "passed":
        _append_reason(reason_codes, "quasi_real_teacher_following_not_passed")
    if quasi_collector and quasi_collector.get("status") != "passed":
        _append_reason(reason_codes, "quasi_real_collector_not_passed")
    for label, payload in (
        ("compatibility_diagnosis", compatibility),
        ("accounting_audit", accounting),
        ("quasi_real_teacher_following", quasi_teacher),
        ("quasi_real_collector", quasi_collector),
    ):
        if payload and _git_current_matches(payload) is False:
            _append_reason(reason_codes, f"{label}_git_current_mismatch")

    comparison_rows, dominated_rows = _episode_rows(updated_steps, config)
    _write_jsonl(paths["return_comparison"], comparison_rows)
    _write_jsonl(paths["dominated_raw_choice_diagnostics"], dominated_rows)

    teacher_equivalent_rows = [row for row in comparison_rows if row["teacher_equivalent_episode"]]
    beyond_rows = [row for row in comparison_rows if row["beyond_teacher_episode"]]
    controlled_regression_rows = [
        row for row in comparison_rows if row["controlled_regression_episode"]
    ]
    if controlled_regression_rows:
        _append_reason(reason_codes, "long_horizon_controlled_regression_detected")
    if not comparison_rows and not reason_codes:
        _append_reason(reason_codes, "long_horizon_episode_coverage_missing")

    verdict = _verdict(reason_codes, comparison_rows, teacher_equivalent_rows)
    status = "failed" if reason_codes else "passed"
    recommended_next_action = _recommended_next_action(verdict)
    family_counts = Counter(row["scenario_group"] for row in comparison_rows)

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "verdict": verdict,
        "recommended_next_action": recommended_next_action,
        "diagnosis_root": _display_path(diagnosis_root, repo_root),
        "accounting_audit_root": _display_path(accounting_audit_root, repo_root),
        "output_root": _display_path(output_root, repo_root),
        "compatibility_diagnosis_summary": _display_path(
            inputs["compatibility_summary"],
            repo_root,
        ),
        "accounting_audit_summary": _display_path(
            inputs["accounting_audit_summary"],
            repo_root,
        ),
        "updated_generated_sequential_steps": _display_path(inputs["updated_steps"], repo_root),
        "updated_generated_sequential_rejection_report": _display_path(
            inputs["updated_rejection_report"],
            repo_root,
        ),
        "base_generated_sequential_steps": _display_path(inputs["base_steps"], repo_root),
        "base_generated_sequential_rejection_report": _display_path(
            inputs["base_rejection_report"],
            repo_root,
        ),
        "quasi_real_teacher_following_summary": _display_path(
            inputs["quasi_real_teacher_following_summary"],
            repo_root,
        ),
        "quasi_real_collector_summary": _display_path(
            inputs["quasi_real_collector_summary"],
            repo_root,
        ),
        "episode_count": len(comparison_rows),
        "step_count": len(updated_steps),
        "base_step_count": len(base_steps),
        "base_rejection_step_count": len(_failed_steps(base_rejection_report)),
        "updated_rejection_step_count": len(_failed_steps(updated_rejection_report)),
        "teacher_equivalent_episode_count": len(teacher_equivalent_rows),
        "beyond_teacher_episode_count": len(beyond_rows),
        "controlled_regression_episode_count": len(controlled_regression_rows),
        "dominated_raw_choice_count": len(dominated_rows),
        "teacher_aligned_active_choice_count": sum(
            row["teacher_aligned_active_choice_count"] for row in comparison_rows
        ),
        "controlled_path_cost_regression_count": sum(
            row["controlled_path_cost_regression_count"] for row in comparison_rows
        ),
        "controlled_risk_regression_count": sum(
            row["controlled_risk_regression_count"] for row in comparison_rows
        ),
        "raw_policy_path_cost_regression_count": sum(
            row["raw_policy_path_cost_regression_count"] for row in comparison_rows
        ),
        "raw_policy_risk_regression_count": sum(
            row["raw_policy_risk_regression_count"] for row in comparison_rows
        ),
        "failed_family_counts": dict(sorted(family_counts.items())),
        "return_comparison": _display_path(paths["return_comparison"], repo_root),
        "teacher_equivalent_report": _display_path(paths["teacher_equivalent_report"], repo_root),
        "beyond_teacher_report": _display_path(paths["beyond_teacher_report"], repo_root),
        "dominated_raw_choice_diagnostics": _display_path(
            paths["dominated_raw_choice_diagnostics"],
            repo_root,
        ),
        "summary": _display_path(paths["summary"], repo_root),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "runs_ppo_update": False,
        "runs_iterative_ppo": False,
        "git_provenance": {
            "current": _git_snapshot(repo_root),
            "current_matches_sources": True,
        },
        "non_goals": list(config.get("non_goals", [])),
    }
    _write_report(paths["teacher_equivalent_report"], summary, teacher_equivalent_rows, "Teacher-Equivalent Episodes")
    _write_report(paths["beyond_teacher_report"], summary, beyond_rows, "Beyond-Teacher Episodes")
    _write_json(paths["summary"], summary)
    return summary


def _episode_rows(
    steps: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation = config.get("evaluation", {})
    horizon_steps = _int_value(evaluation.get("horizon_steps"), default=0)
    tolerance = float(evaluation.get("teacher_equivalence_tolerance", 0.0))
    beyond_margin = float(evaluation.get("beyond_teacher_margin", 0.0))
    max_path = float(evaluation.get("max_path_cost_regression", 0.0))
    max_risk = float(evaluation.get("max_risk_regression", 0.0))
    weights = evaluation.get("return_weights", {})

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        grouped[str(step.get("episode_id") or "")].append(step)

    rows: list[dict[str, Any]] = []
    dominated_rows: list[dict[str, Any]] = []
    for episode_id in sorted(grouped):
        episode_steps = sorted(grouped[episode_id], key=lambda item: _int_value(item.get("step_index")))
        if horizon_steps > 0:
            episode_steps = episode_steps[:horizon_steps]
        teacher_return = 0.0
        controlled_return = 0.0
        raw_return = 0.0
        controlled_path_regressions = 0
        controlled_risk_regressions = 0
        raw_path_regressions = 0
        raw_risk_regressions = 0
        teacher_aligned_active = 0
        dominated_in_episode = 0
        scenario_groups = []

        for step in episode_steps:
            scenario_groups.append(str(step.get("scenario_group") or "unknown"))
            controlled_return += _step_return(step, "policy_selected", weights)
            raw_return += _step_return(step, "raw_policy_selected", weights)
            controlled_reasons = set(_string_list(step.get("controlled_regression_reason_codes")))
            raw_reasons = set(_string_list(step.get("raw_policy_regression_reason_codes"))) or set(
                _string_list(step.get("canary_rejection_reason_codes"))
            )
            policy_path_delta = _float_value(step.get("policy_selected_path_cost_delta"))
            policy_risk_delta = _float_value(step.get("policy_selected_risk_delta"))
            raw_path_delta = _float_value(step.get("raw_policy_selected_path_cost_delta"))
            raw_risk_delta = _float_value(step.get("raw_policy_selected_risk_delta"))
            controlled_path_regressed = (
                "path_cost_regression" in controlled_reasons or policy_path_delta > max_path
            )
            controlled_risk_regressed = (
                "risk_regression" in controlled_reasons or policy_risk_delta > max_risk
            )
            raw_path_regressed = "path_cost_regression" in raw_reasons or raw_path_delta > max_path
            raw_risk_regressed = "risk_regression" in raw_reasons or raw_risk_delta > max_risk
            controlled_path_regressions += int(controlled_path_regressed)
            controlled_risk_regressions += int(controlled_risk_regressed)
            raw_path_regressions += int(raw_path_regressed)
            raw_risk_regressions += int(raw_risk_regressed)
            if _active_same_as_teacher(step):
                teacher_aligned_active += 1
            if raw_path_regressed or raw_risk_regressed or _step_return(step, "raw_policy_selected", weights) < -tolerance:
                dominated_in_episode += 1
                dominated_rows.append(
                    {
                        "episode_id": episode_id,
                        "step_index": _int_value(step.get("step_index")),
                        "scenario_id": step.get("scenario_id"),
                        "scenario_group": str(step.get("scenario_group") or "unknown"),
                        "raw_policy_return_delta_vs_teacher": _step_return(
                            step,
                            "raw_policy_selected",
                            weights,
                        ),
                        "raw_policy_selected_path_cost_delta": raw_path_delta,
                        "raw_policy_selected_risk_delta": raw_risk_delta,
                        "raw_policy_regression_reason_codes": sorted(raw_reasons),
                        "controlled_choice_source": step.get("controlled_choice_source"),
                    }
                )

        controlled_regression = bool(controlled_path_regressions or controlled_risk_regressions)
        teacher_equivalent = (
            not controlled_regression and controlled_return >= teacher_return - tolerance
        )
        beyond_teacher = (
            not controlled_regression and controlled_return > teacher_return + beyond_margin
        )
        rows.append(
            {
                "episode_id": episode_id,
                "scenario_group": _majority_or_first(scenario_groups),
                "step_count": len(episode_steps),
                "teacher_cumulative_return": teacher_return,
                "controlled_policy_cumulative_return": controlled_return,
                "raw_policy_diagnostic_return": raw_return,
                "controlled_return_delta_vs_teacher": controlled_return - teacher_return,
                "raw_return_delta_vs_teacher": raw_return - teacher_return,
                "teacher_equivalent_episode": teacher_equivalent,
                "beyond_teacher_episode": beyond_teacher,
                "controlled_regression_episode": controlled_regression,
                "teacher_aligned_active_choice_count": teacher_aligned_active,
                "dominated_raw_choice_count": dominated_in_episode,
                "controlled_path_cost_regression_count": controlled_path_regressions,
                "controlled_risk_regression_count": controlled_risk_regressions,
                "raw_policy_path_cost_regression_count": raw_path_regressions,
                "raw_policy_risk_regression_count": raw_risk_regressions,
            }
        )
    return rows, dominated_rows


def _step_return(step: dict[str, Any], prefix: str, weights: dict[str, Any]) -> float:
    path_weight = float(weights.get("path_cost", 1.0))
    risk_weight = float(weights.get("risk", 1.0))
    utility_weight = float(weights.get("utility", 0.0))
    progress_weight = float(weights.get("progress", 0.0))
    terminal_weight = float(weights.get("terminal", 0.0))
    safety_penalty = float(weights.get("safety_penalty", 0.0))
    contract_penalty = float(weights.get("contract_penalty", 0.0))
    source_penalty = float(weights.get("source_selection_penalty", 0.0))
    value = 0.0
    value -= path_weight * _float_value(step.get(f"{prefix}_path_cost_delta"))
    value -= risk_weight * _float_value(step.get(f"{prefix}_risk_delta"))
    value -= utility_weight * _float_value(step.get(f"{prefix}_utility_delta"))
    value += progress_weight * (
        _float_value(step.get(f"{prefix}_progress_delta"))
        + _float_value(step.get(f"{prefix}_terminal_progress_delta"))
    )
    value += terminal_weight * _float_value(step.get(f"{prefix}_terminal_delta"))
    reasons = _string_list(step.get("controlled_regression_reason_codes" if prefix == "policy_selected" else "raw_policy_regression_reason_codes"))
    if any("safety" in reason for reason in reasons):
        value -= safety_penalty
    if any("contract" in reason for reason in reasons):
        value -= contract_penalty
    if any("source_selection" in reason for reason in reasons):
        value -= source_penalty
    return value


def _active_same_as_teacher(step: dict[str, Any]) -> bool:
    source = step.get("source_selected_action_index")
    return (
        source is not None
        and step.get("raw_policy_selected_action_index") == source
        and step.get("policy_selected_action_index") == source
        and str(step.get("controlled_choice_source") or "") != "source_fallback"
    )


def _verdict(
    reason_codes: list[str],
    comparison_rows: list[dict[str, Any]],
    teacher_equivalent_rows: list[dict[str, Any]],
) -> str:
    if any(reason.endswith("_missing") or "git_current_mismatch" in reason for reason in reason_codes):
        return "missing_or_stale_input_evidence"
    if reason_codes:
        return "long_horizon_contract_still_blocked"
    if comparison_rows and len(teacher_equivalent_rows) == len(comparison_rows):
        return "long_horizon_teacher_skill_contract_aligned"
    return "return_accounting_inconclusive"


def _recommended_next_action(verdict: str) -> str:
    return {
        "long_horizon_teacher_skill_contract_aligned": "limited_quasi_real_ppo_update_smoke_evaluated",
        "long_horizon_contract_still_blocked": "generated_sequential_contract_alignment_required",
        "return_accounting_inconclusive": "manual_long_horizon_return_accounting_review_required",
        "missing_or_stale_input_evidence": "refresh_long_horizon_contract_alignment_inputs",
    }.get(verdict, "manual_long_horizon_return_accounting_review_required")


def _write_report(
    path: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    title: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"- status: `{summary['status']}`",
        f"- verdict: `{summary['verdict']}`",
        f"- recommended_next_action: `{summary['recommended_next_action']}`",
        f"- episode_count: `{summary['episode_count']}`",
        f"- teacher_equivalent_episode_count: `{summary['teacher_equivalent_episode_count']}`",
        f"- beyond_teacher_episode_count: `{summary['beyond_teacher_episode_count']}`",
        f"- dominated_raw_choice_count: `{summary['dominated_raw_choice_count']}`",
        "",
        "## Episodes",
    ]
    if rows:
        for row in rows:
            lines.append(
                f"- `{row['episode_id']}` family `{row['scenario_group']}` "
                f"return_delta `{row['controlled_return_delta_vs_teacher']:.6f}`"
            )
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _paths(output_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["output_files"]
    return {
        "summary": output_root / outputs.get("summary", "long-horizon-teacher-skill-contract-summary.json"),
        "return_comparison": output_root / outputs.get("return_comparison", "teacher-vs-policy-return-comparison.jsonl"),
        "teacher_equivalent_report": output_root / outputs.get("teacher_equivalent_report", "teacher-equivalent-episode-report.md"),
        "beyond_teacher_report": output_root / outputs.get("beyond_teacher_report", "beyond-teacher-opportunity-report.md"),
        "dominated_raw_choice_diagnostics": output_root / outputs.get("dominated_raw_choice_diagnostics", "dominated-raw-choice-diagnostics.jsonl"),
    }


def _input_paths(
    *,
    diagnosis_root: Path,
    accounting_audit_root: Path,
    config: dict[str, Any],
    quasi_real_teacher_following_summary_path: Path | None,
    quasi_real_collector_summary_path: Path | None,
) -> dict[str, Path]:
    inputs = config["input_files"]
    return {
        "compatibility_summary": diagnosis_root / inputs.get("compatibility_summary", "quasi-real-generated-sequential-contract-compatibility-summary.json"),
        "accounting_audit_summary": accounting_audit_root / inputs.get("accounting_audit_summary", "generated-sequential-gate-metric-accounting-audit-summary.json"),
        "updated_steps": diagnosis_root / inputs.get("updated_steps", "updated_generated_sequential_replay/policy-gated-sequential-canary-steps.jsonl"),
        "updated_rejection_report": diagnosis_root / inputs.get("updated_rejection_report", "updated_generated_sequential_replay/policy-gated-sequential-canary-rejection-report.json"),
        "base_steps": diagnosis_root / inputs.get("base_steps", "base_generated_sequential/policy-gated-sequential-canary-steps.jsonl"),
        "base_rejection_report": diagnosis_root / inputs.get("base_rejection_report", "base_generated_sequential/policy-gated-sequential-canary-rejection-report.json"),
        "quasi_real_teacher_following_summary": (
            quasi_real_teacher_following_summary_path
            or diagnosis_root / inputs.get("quasi_real_teacher_following_summary", "post_update_quasi_real_teacher_following/quasi-real-guarded-teacher-following-pilot-summary.json")
        ),
        "quasi_real_collector_summary": (
            quasi_real_collector_summary_path
            or diagnosis_root / inputs.get("quasi_real_collector_summary", "post_update_quasi_real_collector/ppo-rollout-collector-summary.json")
        ),
    }


def _load_config(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    for section in ("input_files", "output_files", "evaluation"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


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


def _failed_steps(report: dict[str, Any]) -> list[dict[str, Any]]:
    failed = report.get("failed_steps")
    return [step for step in failed if isinstance(step, dict)] if isinstance(failed, list) else []


def _git_current_matches(summary: dict[str, Any]) -> bool | None:
    provenance = summary.get("git_provenance")
    if not isinstance(provenance, dict):
        return None
    value = provenance.get("current_matches_sources")
    return value if isinstance(value, bool) else None


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


def _majority_or_first(values: list[str]) -> str:
    if not values:
        return "unknown"
    return Counter(values).most_common(1)[0][0]


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
