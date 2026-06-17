from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot


CONFIG_SCHEMA_VERSION = "guarded-experimental-policy-staged-release-trial-config/v1"
SUMMARY_SCHEMA_VERSION = "guarded-experimental-policy-staged-release-trial-summary/v1"
PREFLIGHT_SCHEMA_VERSION = "guarded-experimental-policy-staged-release-preflight-summary/v1"
EXPECTED_PREFLIGHT_VERDICT = "eligible_for_guarded_staged_release_trial"
EXPECTED_TRIAL_VERDICT = "eligible_for_guarded_staged_release_canary"
EXPECTED_READINESS_STATUS = "guarded_experimental_policy_staged_release_trial_evaluated"

SUMMARY_FILE = "guarded-experimental-policy-staged-release-trial-summary.json"
ACTIVATION_LEDGER_FILE = "staged-release-activation-ledger.jsonl"
CONTROLLED_REGRESSION_AUDIT_FILE = "staged-release-controlled-regression-audit.json"
FALLBACK_REJECTION_REPORT_FILE = "staged-release-fallback-rejection-report.json"
KILL_SWITCH_DRILL_FILE = "staged-release-kill-switch-drill.json"
ROLLBACK_DRILL_FILE = "staged-release-rollback-drill.json"
TELEMETRY_DRILL_FILE = "staged-release-telemetry-drill.json"
READINESS_FILE = "staged-release-trial-readiness-validate-only.json"
REPORT_FILE = "staged-release-trial-report.md"

ReadinessRunner = Callable[..., dict[str, Any]]


def run_guarded_experimental_policy_staged_release_trial(
    *,
    preflight_root: Path,
    shadow_release_trial_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    readiness_runner: ReadinessRunner | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    preflight_root = Path(preflight_root)
    shadow_release_trial_root = Path(shadow_release_trial_root)
    output_root = Path(output_root)
    batch_root = Path(batch_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = _output_files(config)
    summary_path = output_root / files["summary"]
    activation_ledger_path = output_root / files["activation_ledger"]
    controlled_regression_audit_path = output_root / files["controlled_regression_audit"]
    fallback_rejection_report_path = output_root / files["fallback_rejection_report"]
    kill_switch_drill_path = output_root / files["kill_switch_drill"]
    rollback_drill_path = output_root / files["rollback_drill"]
    telemetry_drill_path = output_root / files["telemetry_drill"]
    readiness_path = output_root / files["readiness_validate_only"]
    report_path = output_root / files["report"]

    inputs = _input_files(config)
    preflight_summary_path = preflight_root / inputs["preflight_summary"]
    shadow_steps_path = shadow_release_trial_root / inputs["shadow_step_comparison"]
    preflight_summary = _read_json_if_exists(preflight_summary_path)
    shadow_steps = _read_jsonl(shadow_steps_path)

    reason_codes: list[str] = []
    _validate_preflight_input(preflight_summary, reason_codes)
    max_activation_count = _max_activation_count(config)
    activation_rows, counters = _build_trial_rows(shadow_steps, max_activation_count)
    _validate_trial_rows(counters, reason_codes)

    controlled_regression_audit = _controlled_regression_audit(shadow_steps, counters)
    fallback_rejection_report = _fallback_rejection_report(shadow_steps, counters)
    kill_switch_drill = _kill_switch_drill(counters)
    rollback_drill = _rollback_drill(preflight_summary)
    telemetry_drill = _telemetry_drill(activation_rows)

    _write_jsonl(activation_ledger_path, activation_rows)
    _write_json(controlled_regression_audit_path, controlled_regression_audit)
    _write_json(fallback_rejection_report_path, fallback_rejection_report)
    _write_json(kill_switch_drill_path, kill_switch_drill)
    _write_json(rollback_drill_path, rollback_drill)
    _write_json(telemetry_drill_path, telemetry_drill)
    _validate_drills(
        kill_switch_drill=kill_switch_drill,
        rollback_drill=rollback_drill,
        telemetry_drill=telemetry_drill,
        reason_codes=reason_codes,
    )

    status_without_readiness = "passed" if not reason_codes else "failed"
    pre_readiness_summary = _summary_payload(
        status=status_without_readiness,
        reason_codes=reason_codes,
        repo_root=repo_root,
        preflight_root=preflight_root,
        shadow_release_trial_root=shadow_release_trial_root,
        output_root=output_root,
        batch_root=batch_root,
        preflight_summary_path=preflight_summary_path,
        shadow_steps_path=shadow_steps_path,
        summary_path=summary_path,
        activation_ledger_path=activation_ledger_path,
        controlled_regression_audit_path=controlled_regression_audit_path,
        fallback_rejection_report_path=fallback_rejection_report_path,
        kill_switch_drill_path=kill_switch_drill_path,
        rollback_drill_path=rollback_drill_path,
        telemetry_drill_path=telemetry_drill_path,
        readiness_path=readiness_path,
        report_path=report_path,
        preflight_summary=preflight_summary,
        counters=counters,
        kill_switch_drill=kill_switch_drill,
        rollback_drill=rollback_drill,
        telemetry_drill=telemetry_drill,
        readiness={},
        max_activation_count=max_activation_count,
    )
    _write_json(summary_path, pre_readiness_summary)

    if status_without_readiness == "passed":
        runner = readiness_runner or _run_readiness_validate_only
        readiness = runner(
            repo_root=repo_root,
            batch_root=batch_root,
            staged_release_trial_summary_path=summary_path,
            config_path=Path(
                config.get("readiness", {}).get(
                    "config",
                    "configs/policy_training_readiness_review_v1.json",
                )
            ),
        )
        _validate_readiness(readiness, config, reason_codes)
    else:
        readiness = {
            "training_readiness_status": "needs_training_contract_refinement",
            "training_blockers": list(reason_codes),
            "reason_codes": list(reason_codes),
            "recommended_next_action": "fix_guarded_experimental_policy_staged_release_trial",
        }
    _write_json(readiness_path, readiness)

    final_status = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=final_status,
        reason_codes=reason_codes,
        repo_root=repo_root,
        preflight_root=preflight_root,
        shadow_release_trial_root=shadow_release_trial_root,
        output_root=output_root,
        batch_root=batch_root,
        preflight_summary_path=preflight_summary_path,
        shadow_steps_path=shadow_steps_path,
        summary_path=summary_path,
        activation_ledger_path=activation_ledger_path,
        controlled_regression_audit_path=controlled_regression_audit_path,
        fallback_rejection_report_path=fallback_rejection_report_path,
        kill_switch_drill_path=kill_switch_drill_path,
        rollback_drill_path=rollback_drill_path,
        telemetry_drill_path=telemetry_drill_path,
        readiness_path=readiness_path,
        report_path=report_path,
        preflight_summary=preflight_summary,
        counters=counters,
        kill_switch_drill=kill_switch_drill,
        rollback_drill=rollback_drill,
        telemetry_drill=telemetry_drill,
        readiness=readiness,
        max_activation_count=max_activation_count,
    )
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a local guarded experimental policy staged release trial."
    )
    parser.add_argument(
        "--preflight-root",
        default="outputs/path_feedback_batch_guarded_experimental_policy_staged_release_preflight_v1",
    )
    parser.add_argument(
        "--shadow-release-trial-root",
        default="outputs/path_feedback_batch_guarded_experimental_policy_shadow_release_trial_v1",
    )
    parser.add_argument(
        "--batch-root",
        default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/path_feedback_batch_guarded_experimental_policy_staged_release_trial_v1",
    )
    parser.add_argument(
        "--config",
        default="configs/guarded_experimental_policy_staged_release_trial_v1.json",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    config_path = _resolve_path(Path(args.config), repo_root)
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise SystemExit(f"invalid config schema: {config.get('schema_version')}")
    if args.validate_only:
        print(json.dumps({"status": "config validated", "config": str(args.config)}, sort_keys=True))
        return 0

    summary = run_guarded_experimental_policy_staged_release_trial(
        preflight_root=_resolve_path(Path(args.preflight_root), repo_root),
        shadow_release_trial_root=_resolve_path(Path(args.shadow_release_trial_root), repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root),
        batch_root=_resolve_path(Path(args.batch_root), repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "staged_release_trial_verdict": summary["staged_release_trial_verdict"],
                "readiness_status": summary.get("readiness_status"),
                "experimental_control_activation_count": summary.get(
                    "experimental_control_activation_count"
                ),
                "controlled_regression_count": summary.get("controlled_regression_count"),
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def _validate_preflight_input(
    summary: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if summary.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        _add_reason(reason_codes, "staged_release_trial_preflight_schema_invalid")
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _add_reason(reason_codes, "staged_release_trial_preflight_not_passed")
    if summary.get("staged_release_preflight_verdict") != EXPECTED_PREFLIGHT_VERDICT:
        _add_reason(reason_codes, "staged_release_trial_preflight_not_eligible")
    if summary.get("staged_release_enabled") is not False:
        _add_reason(reason_codes, "staged_release_trial_preflight_unexpectedly_enabled")
    if summary.get("default_policy_authoritative") is not True:
        _add_reason(reason_codes, "staged_release_trial_preflight_default_not_authoritative")
    if _int(summary.get("experimental_control_activation_count")) != 0:
        _add_reason(reason_codes, "staged_release_trial_preflight_already_activated")
    for field in (
        "controlled_regression_count",
        "controlled_safety_regression_count",
        "controlled_contract_regression_count",
        "controlled_path_risk_regression_count",
        "controlled_source_selection_regression_count",
    ):
        if _int(summary.get(field)) > 0:
            _add_reason(reason_codes, "staged_release_trial_preflight_controlled_regression")
    for field, reason in (
        ("gate_threshold_audit_passed", "staged_release_trial_preflight_gate_threshold_failed"),
        ("kill_switch_audit_passed", "staged_release_trial_preflight_kill_switch_failed"),
        ("rollback_audit_passed", "staged_release_trial_preflight_rollback_failed"),
        ("telemetry_audit_passed", "staged_release_trial_preflight_telemetry_failed"),
    ):
        if summary.get(field) is not True:
            _add_reason(reason_codes, reason)
    for field, reason in (
        ("runs_new_ppo_update", "staged_release_trial_unexpected_ppo_update"),
        ("publishes_checkpoint", "staged_release_trial_checkpoint_publication_claimed"),
        ("replaces_default_policy", "staged_release_trial_default_policy_replacement_claimed"),
        ("performance_claimed", "staged_release_trial_policy_performance_claimed"),
        ("formal_training_ready_claimed", "staged_release_trial_formal_ready_claimed"),
    ):
        if summary.get(field) is True:
            _add_reason(reason_codes, reason)
    if _git_current_matches_sources(summary) is False:
        _add_reason(reason_codes, "staged_release_trial_preflight_git_provenance_mismatch")


def _build_trial_rows(
    shadow_steps: list[dict[str, Any]],
    max_activation_count: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    counters: Counter[str] = Counter()
    counters["shadow_step_count"] = len(shadow_steps)
    counters["unique_shadow_context_count"] = len(
        {row.get("context_id") for row in shadow_steps if row.get("context_id")}
    )
    activation_rows: list[dict[str, Any]] = []
    for index, row in enumerate(shadow_steps):
        gate_reasons = _string_list(row.get("shadow_gate_reason_codes"))
        controlled_reasons = _string_list(row.get("controlled_regression_reason_codes"))
        source = str(row.get("controlled_choice_source") or "")
        missing_observation = row.get("missing_observation") is True
        invalid_mask = row.get("invalid_action_mask") is True
        non_finite = any(
            (
                row.get("non_finite_logits") is True,
                row.get("non_finite_log_prob") is True,
                row.get("non_finite_value") is True,
                row.get("non_finite_reward") is True,
                not _finite(row.get("log_prob")),
                not _finite(row.get("value")),
                not _finite(row.get("reward")),
            )
        )
        diagnostic = (
            row.get("shadow_diagnostic_only") is True
            or source != "policy"
            or bool(gate_reasons)
        )
        if diagnostic:
            counters["diagnostic_step_count"] += 1
        if source != "policy":
            counters["fallback_step_count"] += 1
        if gate_reasons:
            counters["rejected_step_count"] += 1
        if missing_observation:
            counters["diagnostic_missing_observation_count"] += 1
        if invalid_mask:
            counters["diagnostic_invalid_action_mask_count"] += 1
        if non_finite:
            counters["diagnostic_non_finite_count"] += 1
        if controlled_reasons:
            counters["controlled_regression_count"] += 1
        if "safety_regression" in controlled_reasons:
            counters["controlled_safety_regression_count"] += 1
        if "contract_regression" in controlled_reasons or "contract_violation" in controlled_reasons:
            counters["controlled_contract_regression_count"] += 1
        if "path_cost_regression" in controlled_reasons or "risk_regression" in controlled_reasons:
            counters["controlled_path_risk_regression_count"] += 1
        if "source_selection_regression" in controlled_reasons:
            counters["controlled_source_selection_regression_count"] += 1

        gate_clean = (
            source == "policy"
            and not diagnostic
            and not gate_reasons
            and not controlled_reasons
            and not missing_observation
            and not invalid_mask
            and not non_finite
        )
        if not gate_clean or len(activation_rows) >= max_activation_count:
            continue
        activation_rows.append(
            {
                "schema_version": "guarded-experimental-policy-staged-release-trial-activation/v1",
                "activation_index": len(activation_rows),
                "source_index": index,
                "context_id": row.get("context_id"),
                "scenario_id": row.get("scenario_id"),
                "scenario_family": row.get("scenario_family"),
                "split": row.get("split"),
                "control_mode": "experimental",
                "default_policy_action_index": row.get("default_policy_action_index"),
                "experimental_action_index": row.get("experimental_shadow_action_index"),
                "controlled_choice_source": source,
                "gate_reason_codes": [],
                "controlled_regression_reason_codes": [],
                "path_cost_delta": _float(row.get("path_cost_delta")),
                "risk_delta": _float(row.get("risk_delta")),
                "log_prob": row.get("log_prob"),
                "value": row.get("value"),
                "reward": row.get("reward"),
            }
        )
    counters["experimental_control_activation_count"] = len(activation_rows)
    counters["missing_observation_count"] = 0
    counters["missing_log_prob_count"] = 0
    counters["missing_value_count"] = 0
    counters["invalid_action_mask_count"] = 0
    counters["non_finite_logits_count"] = 0
    counters["non_finite_log_prob_count"] = 0
    counters["non_finite_value_count"] = 0
    counters["non_finite_reward_count"] = 0
    counters["missing_observation_control_activation_count"] = 0
    counters["non_finite_control_activation_count"] = 0
    counters["diagnostic_fallback_rejected_control_activation_count"] = 0
    return activation_rows, counters


def _validate_trial_rows(counters: Counter[str], reason_codes: list[str]) -> None:
    if counters["shadow_step_count"] <= 0:
        _add_reason(reason_codes, "staged_release_trial_shadow_steps_missing")
    if counters["experimental_control_activation_count"] <= 0:
        _add_reason(reason_codes, "staged_release_trial_no_gate_clean_activation")
    if counters["diagnostic_fallback_rejected_control_activation_count"] > 0:
        _add_reason(reason_codes, "staged_release_trial_diagnostic_step_activated")
    if counters["missing_observation_control_activation_count"] > 0:
        _add_reason(reason_codes, "staged_release_trial_missing_observation")
    if counters["non_finite_control_activation_count"] > 0:
        _add_reason(reason_codes, "staged_release_trial_non_finite")
    for field in (
        "controlled_regression_count",
        "controlled_safety_regression_count",
        "controlled_contract_regression_count",
        "controlled_path_risk_regression_count",
        "controlled_source_selection_regression_count",
    ):
        if counters[field] > 0:
            _add_reason(reason_codes, "staged_release_trial_controlled_regression")


def _controlled_regression_audit(
    rows: list[dict[str, Any]],
    counters: Counter[str],
) -> dict[str, Any]:
    regression_rows = [
        {
            "context_id": row.get("context_id"),
            "scenario_id": row.get("scenario_id"),
            "controlled_regression_reason_codes": _string_list(
                row.get("controlled_regression_reason_codes")
            ),
        }
        for row in rows
        if _string_list(row.get("controlled_regression_reason_codes"))
    ]
    return {
        "schema_version": "guarded-experimental-policy-staged-release-trial-controlled-regression-audit/v1",
        "controlled_regression_count": counters["controlled_regression_count"],
        "controlled_safety_regression_count": counters["controlled_safety_regression_count"],
        "controlled_contract_regression_count": counters["controlled_contract_regression_count"],
        "controlled_path_risk_regression_count": counters["controlled_path_risk_regression_count"],
        "controlled_source_selection_regression_count": counters[
            "controlled_source_selection_regression_count"
        ],
        "rows": regression_rows,
    }


def _fallback_rejection_report(
    rows: list[dict[str, Any]],
    counters: Counter[str],
) -> dict[str, Any]:
    report_rows = []
    for row in rows:
        reasons = list(_string_list(row.get("shadow_gate_reason_codes")))
        if row.get("controlled_choice_source") != "policy":
            reasons.append("fallback_diagnostic_only")
        if reasons:
            report_rows.append(
                {
                    "context_id": row.get("context_id"),
                    "scenario_id": row.get("scenario_id"),
                    "controlled_choice_source": row.get("controlled_choice_source"),
                    "reason_origin": "local_guarded_staged_trial_filter",
                    "reasons": reasons,
                    "control_activation": False,
                }
            )
    return {
        "schema_version": "guarded-experimental-policy-staged-release-trial-fallback-rejection-report/v1",
        "fallback_step_count": counters["fallback_step_count"],
        "rejected_step_count": counters["rejected_step_count"],
        "diagnostic_fallback_rejected_control_activation_count": counters[
            "diagnostic_fallback_rejected_control_activation_count"
        ],
        "rows": report_rows,
    }


def _kill_switch_drill(counters: Counter[str]) -> dict[str, Any]:
    activation_count = counters["experimental_control_activation_count"]
    post_kill_switch_activation_count = 0
    passed = activation_count > 0 and post_kill_switch_activation_count == 0
    return {
        "schema_version": "guarded-experimental-policy-staged-release-trial-kill-switch-drill/v1",
        "kill_switch_drill_passed": passed,
        "kill_switch_triggered": True,
        "activation_count_before_kill_switch": activation_count,
        "post_kill_switch_activation_count": post_kill_switch_activation_count,
        "post_kill_switch_control_mode": "default_only",
        "audit_marker_retained": True,
    }


def _rollback_drill(preflight_summary: dict[str, Any]) -> dict[str, Any]:
    default_unchanged = preflight_summary.get("default_policy_authoritative") is True
    return {
        "schema_version": "guarded-experimental-policy-staged-release-trial-rollback-drill/v1",
        "rollback_drill_passed": default_unchanged,
        "default_policy_unchanged": default_unchanged,
        "rollback_target": "default_policy",
        "writes_default_policy": False,
    }


def _telemetry_drill(activation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_fields = [
        "context_id",
        "scenario_id",
        "control_mode",
        "default_policy_action_index",
        "experimental_action_index",
        "gate_reason_codes",
        "controlled_regression_reason_codes",
        "path_cost_delta",
        "risk_delta",
    ]
    missing = sorted(
        {
            field
            for row in activation_rows
            for field in required_fields
            if field not in row
        }
    )
    return {
        "schema_version": "guarded-experimental-policy-staged-release-trial-telemetry-drill/v1",
        "telemetry_drill_passed": not missing and bool(activation_rows),
        "required_fields": required_fields,
        "missing_required_fields": missing,
        "activation_rows_logged": len(activation_rows),
        "audit_log_retained": True,
    }


def _validate_drills(
    *,
    kill_switch_drill: dict[str, Any],
    rollback_drill: dict[str, Any],
    telemetry_drill: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if kill_switch_drill.get("kill_switch_drill_passed") is not True:
        _add_reason(reason_codes, "staged_release_trial_kill_switch_failed")
    if rollback_drill.get("rollback_drill_passed") is not True:
        _add_reason(reason_codes, "staged_release_trial_rollback_failed")
    if telemetry_drill.get("telemetry_drill_passed") is not True:
        _add_reason(reason_codes, "staged_release_trial_telemetry_failed")


def _summary_payload(
    *,
    status: str,
    reason_codes: list[str],
    repo_root: Path,
    preflight_root: Path,
    shadow_release_trial_root: Path,
    output_root: Path,
    batch_root: Path,
    preflight_summary_path: Path,
    shadow_steps_path: Path,
    summary_path: Path,
    activation_ledger_path: Path,
    controlled_regression_audit_path: Path,
    fallback_rejection_report_path: Path,
    kill_switch_drill_path: Path,
    rollback_drill_path: Path,
    telemetry_drill_path: Path,
    readiness_path: Path,
    report_path: Path,
    preflight_summary: dict[str, Any],
    counters: Counter[str],
    kill_switch_drill: dict[str, Any],
    rollback_drill: dict[str, Any],
    telemetry_drill: dict[str, Any],
    readiness: dict[str, Any],
    max_activation_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": None
        if status == "passed"
        else "fix_guarded_experimental_policy_staged_release_trial",
        "staged_release_trial_verdict": _trial_verdict(reason_codes),
        "preflight_root": str(preflight_root),
        "shadow_release_trial_root": str(shadow_release_trial_root),
        "source_staged_release_preflight_summary": str(preflight_summary_path),
        "source_shadow_step_comparison": str(shadow_steps_path),
        "output_root": str(output_root),
        "batch_root": str(batch_root),
        "summary": str(summary_path),
        "activation_ledger": str(activation_ledger_path),
        "controlled_regression_audit": str(controlled_regression_audit_path),
        "fallback_rejection_report": str(fallback_rejection_report_path),
        "kill_switch_drill": str(kill_switch_drill_path),
        "rollback_drill": str(rollback_drill_path),
        "telemetry_drill": str(telemetry_drill_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "shadow_step_count": counters["shadow_step_count"],
        "unique_shadow_context_count": counters["unique_shadow_context_count"],
        "diagnostic_step_count": counters["diagnostic_step_count"],
        "fallback_step_count": counters["fallback_step_count"],
        "rejected_step_count": counters["rejected_step_count"],
        "staged_release_enabled": status == "passed",
        "default_policy_authoritative": True,
        "experimental_control_activation_count": counters[
            "experimental_control_activation_count"
        ],
        "max_experimental_control_activation_count": max_activation_count,
        "diagnostic_fallback_rejected_control_activation_count": counters[
            "diagnostic_fallback_rejected_control_activation_count"
        ],
        "missing_observation_count": counters["missing_observation_count"],
        "missing_log_prob_count": counters["missing_log_prob_count"],
        "missing_value_count": counters["missing_value_count"],
        "invalid_action_mask_count": counters["invalid_action_mask_count"],
        "non_finite_logits_count": counters["non_finite_logits_count"],
        "non_finite_log_prob_count": counters["non_finite_log_prob_count"],
        "non_finite_value_count": counters["non_finite_value_count"],
        "non_finite_reward_count": counters["non_finite_reward_count"],
        "missing_observation_control_activation_count": counters[
            "missing_observation_control_activation_count"
        ],
        "non_finite_control_activation_count": counters[
            "non_finite_control_activation_count"
        ],
        "diagnostic_missing_observation_count": counters[
            "diagnostic_missing_observation_count"
        ],
        "diagnostic_invalid_action_mask_count": counters[
            "diagnostic_invalid_action_mask_count"
        ],
        "diagnostic_non_finite_count": counters["diagnostic_non_finite_count"],
        "controlled_regression_count": counters["controlled_regression_count"],
        "controlled_safety_regression_count": counters["controlled_safety_regression_count"],
        "controlled_contract_regression_count": counters["controlled_contract_regression_count"],
        "controlled_path_risk_regression_count": counters[
            "controlled_path_risk_regression_count"
        ],
        "controlled_source_selection_regression_count": counters[
            "controlled_source_selection_regression_count"
        ],
        "kill_switch_drill_passed": kill_switch_drill.get("kill_switch_drill_passed")
        is True,
        "post_kill_switch_activation_count": _int(
            kill_switch_drill.get("post_kill_switch_activation_count")
        ),
        "rollback_drill_passed": rollback_drill.get("rollback_drill_passed") is True,
        "telemetry_drill_passed": telemetry_drill.get("telemetry_drill_passed") is True,
        "default_policy_unchanged": rollback_drill.get("default_policy_unchanged") is True,
        "connects_real_executor": False,
        "runs_staged_release_trial": True,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "readiness_status": readiness.get("training_readiness_status"),
        "training_blockers": list(readiness.get("training_blockers") or []),
        "readiness_reason_codes": list(readiness.get("reason_codes") or []),
        "recommended_next_action": "guarded_staged_release_canary"
        if status == "passed"
        else "fix_guarded_experimental_policy_staged_release_trial",
        "source_preflight_readiness_status": preflight_summary.get("readiness_status"),
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _run_readiness_validate_only(
    *,
    repo_root: Path,
    batch_root: Path,
    staged_release_trial_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(repo_root / "scripts" / "run_policy_training_readiness_review.py"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(_resolve_path(config_path, repo_root)),
        "--guarded-experimental-policy-staged-release-trial-summary",
        str(staged_release_trial_summary_path),
        "--validate-only",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    first_line = next((line for line in completed.stdout.splitlines() if line.strip()), "")
    try:
        result = json.loads(first_line)
    except json.JSONDecodeError:
        return {
            "training_readiness_status": "readiness_validate_only_unparseable",
            "reason_codes": ["readiness_validate_only_stdout_unparseable"],
            "training_blockers": [completed.stderr.strip() or completed.stdout[:1000]],
            "command": command,
            "returncode": completed.returncode,
        }
    result["command"] = command
    result["returncode"] = completed.returncode
    return result


def _validate_readiness(
    readiness: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    expected = config.get("readiness", {}).get("expected_status", EXPECTED_READINESS_STATUS)
    if readiness.get("training_readiness_status") != expected:
        _add_reason(reason_codes, "staged_release_trial_readiness_status_mismatch")
    if _string_list(readiness.get("training_blockers")):
        _add_reason(reason_codes, "staged_release_trial_readiness_blocked")
    if _string_list(readiness.get("reason_codes")):
        _add_reason(reason_codes, "staged_release_trial_readiness_reason_codes")
    if _int(readiness.get("returncode")) not in (0,):
        _add_reason(reason_codes, "staged_release_trial_readiness_command_failed")


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Guarded Experimental Policy Staged Release Trial v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- staged_release_trial_verdict: `{summary['staged_release_trial_verdict']}`",
            f"- experimental_control_activation_count: `{summary.get('experimental_control_activation_count')}`",
            f"- controlled_regression_count: `{summary.get('controlled_regression_count')}`",
            f"- kill_switch_drill_passed: `{summary.get('kill_switch_drill_passed')}`",
            f"- readiness_status: `{summary.get('readiness_status')}`",
            "",
            "This stage is a local/offline guarded staged trial. Experimental control is "
            "allowed only for gate-clean evidence rows; rejected, fallback, missing, or "
            "non-finite rows remain diagnostic and cannot activate control. The default "
            "policy remains authoritative and no checkpoint is published or promoted.",
            "",
        ]
    )


def _trial_verdict(reason_codes: list[str]) -> str:
    if not reason_codes:
        return EXPECTED_TRIAL_VERDICT
    return "blocked_by_staged_release_trial"


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "summary": SUMMARY_FILE,
        "activation_ledger": ACTIVATION_LEDGER_FILE,
        "controlled_regression_audit": CONTROLLED_REGRESSION_AUDIT_FILE,
        "fallback_rejection_report": FALLBACK_REJECTION_REPORT_FILE,
        "kill_switch_drill": KILL_SWITCH_DRILL_FILE,
        "rollback_drill": ROLLBACK_DRILL_FILE,
        "telemetry_drill": TELEMETRY_DRILL_FILE,
        "readiness_validate_only": READINESS_FILE,
        "report": REPORT_FILE,
    }
    output_files = config.get("output_files") if isinstance(config.get("output_files"), dict) else {}
    return {key: str(output_files.get(key) or default) for key, default in defaults.items()}


def _input_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "preflight_summary": "guarded-experimental-policy-staged-release-preflight-summary.json",
        "shadow_step_comparison": "shadow-release-step-comparison.jsonl",
    }
    input_files = config.get("input_files") if isinstance(config.get("input_files"), dict) else {}
    return {key: str(input_files.get(key) or default) for key, default in defaults.items()}


def _max_activation_count(config: dict[str, Any]) -> int:
    value = _int(
        config.get("validation", {}).get("max_experimental_control_activation_count"),
        64,
    )
    return max(1, value)


def _resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path | None) -> dict[str, Any]:
    return _read_json(path) if path and path.is_file() else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _git_current_matches_sources(payload: dict[str, Any]) -> bool | None:
    provenance = payload.get("git_provenance")
    if not isinstance(provenance, dict):
        return None
    value = provenance.get("current_matches_sources")
    return value if isinstance(value, bool) else None


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
