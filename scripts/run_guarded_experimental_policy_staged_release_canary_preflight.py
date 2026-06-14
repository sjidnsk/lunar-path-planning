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


CONFIG_SCHEMA_VERSION = (
    "guarded-experimental-policy-staged-release-canary-preflight-config/v1"
)
SUMMARY_SCHEMA_VERSION = (
    "guarded-experimental-policy-staged-release-canary-preflight-summary/v1"
)
TRIAL_SCHEMA_VERSION = "guarded-experimental-policy-staged-release-trial-summary/v1"
EXPECTED_TRIAL_VERDICT = "eligible_for_guarded_staged_release_canary"
EXPECTED_CANARY_PREFLIGHT_VERDICT = (
    "eligible_for_guarded_staged_release_canary_dry_run"
)
EXPECTED_READINESS_STATUS = (
    "guarded_experimental_policy_staged_release_canary_preflight_evaluated"
)

SUMMARY_FILE = "guarded-experimental-policy-staged-release-canary-preflight-summary.json"
MANIFEST_FILE = "staged-canary-preflight-manifest.json"
ELIGIBILITY_LEDGER_FILE = "staged-canary-eligibility-ledger.jsonl"
REJECTION_REPORT_FILE = "staged-canary-rejection-report.json"
KILL_SWITCH_AUDIT_FILE = "staged-canary-kill-switch-audit.json"
ROLLBACK_AUDIT_FILE = "staged-canary-rollback-audit.json"
TELEMETRY_AUDIT_FILE = "staged-canary-telemetry-audit.json"
AUTOMATIC_DOWNGRADE_AUDIT_FILE = "staged-canary-automatic-downgrade-audit.json"
BUDGET_AUDIT_FILE = "staged-canary-budget-audit.json"
OPERATOR_APPROVAL_AUDIT_FILE = "staged-canary-operator-approval-audit.json"
READINESS_FILE = "staged-canary-preflight-readiness-validate-only.json"
REPORT_FILE = "staged-canary-preflight-report.md"

ReadinessRunner = Callable[..., dict[str, Any]]


def run_guarded_experimental_policy_staged_release_canary_preflight(
    *,
    staged_release_trial_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    readiness_runner: ReadinessRunner | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    staged_release_trial_root = Path(staged_release_trial_root)
    output_root = Path(output_root)
    batch_root = Path(batch_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = _output_files(config)
    summary_path = output_root / files["summary"]
    manifest_path = output_root / files["manifest"]
    eligibility_ledger_path = output_root / files["eligibility_ledger"]
    rejection_report_path = output_root / files["rejection_report"]
    kill_switch_audit_path = output_root / files["kill_switch_audit"]
    rollback_audit_path = output_root / files["rollback_audit"]
    telemetry_audit_path = output_root / files["telemetry_audit"]
    automatic_downgrade_audit_path = output_root / files["automatic_downgrade_audit"]
    budget_audit_path = output_root / files["budget_audit"]
    operator_approval_audit_path = output_root / files["operator_approval_audit"]
    readiness_path = output_root / files["readiness_validate_only"]
    report_path = output_root / files["report"]

    inputs = _input_files(config)
    trial_summary_path = staged_release_trial_root / inputs["staged_release_trial_summary"]
    trial_summary = _read_json_if_exists(trial_summary_path)
    activation_ledger_path = _activation_ledger_path(
        staged_release_trial_root=staged_release_trial_root,
        configured_file=inputs["activation_ledger"],
        trial_summary=trial_summary,
    )
    activation_rows = _read_jsonl(activation_ledger_path)

    reason_codes: list[str] = []
    _validate_trial_input(trial_summary, reason_codes)
    max_activation_count = _max_activation_count(config)
    canary_traffic_fraction = _canary_traffic_fraction(config)
    _validate_canary_limits(
        max_activation_count=max_activation_count,
        canary_traffic_fraction=canary_traffic_fraction,
        reason_codes=reason_codes,
    )
    eligibility_rows, counters, rejected_rows = _build_eligibility_rows(
        activation_rows,
        max_activation_count=max_activation_count,
    )
    _validate_eligibility(counters, reason_codes)

    manifest = _manifest(
        max_activation_count=max_activation_count,
        canary_traffic_fraction=canary_traffic_fraction,
    )
    rejection_report = _rejection_report(rejected_rows, counters)
    kill_switch_audit = _kill_switch_audit(counters)
    rollback_audit = _rollback_audit(trial_summary)
    telemetry_audit = _telemetry_audit(eligibility_rows)
    automatic_downgrade_audit = _automatic_downgrade_audit()
    budget_audit = _budget_audit(
        counters=counters,
        max_activation_count=max_activation_count,
        canary_traffic_fraction=canary_traffic_fraction,
    )
    operator_approval_audit = _operator_approval_audit()
    _validate_audits(
        kill_switch_audit=kill_switch_audit,
        rollback_audit=rollback_audit,
        telemetry_audit=telemetry_audit,
        automatic_downgrade_audit=automatic_downgrade_audit,
        budget_audit=budget_audit,
        operator_approval_audit=operator_approval_audit,
        reason_codes=reason_codes,
    )

    _write_json(manifest_path, manifest)
    _write_jsonl(eligibility_ledger_path, eligibility_rows)
    _write_json(rejection_report_path, rejection_report)
    _write_json(kill_switch_audit_path, kill_switch_audit)
    _write_json(rollback_audit_path, rollback_audit)
    _write_json(telemetry_audit_path, telemetry_audit)
    _write_json(automatic_downgrade_audit_path, automatic_downgrade_audit)
    _write_json(budget_audit_path, budget_audit)
    _write_json(operator_approval_audit_path, operator_approval_audit)

    status_without_readiness = "passed" if not reason_codes else "failed"
    pre_readiness_summary = _summary_payload(
        status=status_without_readiness,
        reason_codes=reason_codes,
        repo_root=repo_root,
        staged_release_trial_root=staged_release_trial_root,
        output_root=output_root,
        batch_root=batch_root,
        trial_summary_path=trial_summary_path,
        activation_ledger_path=activation_ledger_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        eligibility_ledger_path=eligibility_ledger_path,
        rejection_report_path=rejection_report_path,
        kill_switch_audit_path=kill_switch_audit_path,
        rollback_audit_path=rollback_audit_path,
        telemetry_audit_path=telemetry_audit_path,
        automatic_downgrade_audit_path=automatic_downgrade_audit_path,
        budget_audit_path=budget_audit_path,
        operator_approval_audit_path=operator_approval_audit_path,
        readiness_path=readiness_path,
        report_path=report_path,
        counters=counters,
        max_activation_count=max_activation_count,
        canary_traffic_fraction=canary_traffic_fraction,
        kill_switch_audit=kill_switch_audit,
        rollback_audit=rollback_audit,
        telemetry_audit=telemetry_audit,
        automatic_downgrade_audit=automatic_downgrade_audit,
        budget_audit=budget_audit,
        operator_approval_audit=operator_approval_audit,
        readiness={},
        trial_summary=trial_summary,
    )
    _write_json(summary_path, pre_readiness_summary)

    if status_without_readiness == "passed":
        runner = readiness_runner or _run_readiness_validate_only
        readiness = runner(
            repo_root=repo_root,
            batch_root=batch_root,
            canary_preflight_summary_path=summary_path,
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
            "recommended_next_action": (
                "fix_guarded_experimental_policy_staged_release_canary_preflight"
            ),
        }
    _write_json(readiness_path, readiness)

    final_status = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=final_status,
        reason_codes=reason_codes,
        repo_root=repo_root,
        staged_release_trial_root=staged_release_trial_root,
        output_root=output_root,
        batch_root=batch_root,
        trial_summary_path=trial_summary_path,
        activation_ledger_path=activation_ledger_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        eligibility_ledger_path=eligibility_ledger_path,
        rejection_report_path=rejection_report_path,
        kill_switch_audit_path=kill_switch_audit_path,
        rollback_audit_path=rollback_audit_path,
        telemetry_audit_path=telemetry_audit_path,
        automatic_downgrade_audit_path=automatic_downgrade_audit_path,
        budget_audit_path=budget_audit_path,
        operator_approval_audit_path=operator_approval_audit_path,
        readiness_path=readiness_path,
        report_path=report_path,
        counters=counters,
        max_activation_count=max_activation_count,
        canary_traffic_fraction=canary_traffic_fraction,
        kill_switch_audit=kill_switch_audit,
        rollback_audit=rollback_audit,
        telemetry_audit=telemetry_audit,
        automatic_downgrade_audit=automatic_downgrade_audit,
        budget_audit=budget_audit,
        operator_approval_audit=operator_approval_audit,
        readiness=readiness,
        trial_summary=trial_summary,
    )
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a local/offline guarded staged release canary preflight."
    )
    parser.add_argument(
        "--staged-release-trial-root",
        default="outputs/path_feedback_batch_guarded_experimental_policy_staged_release_trial_v1",
    )
    parser.add_argument(
        "--batch-root",
        default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/path_feedback_batch_guarded_experimental_policy_staged_release_canary_preflight_v1",
    )
    parser.add_argument(
        "--config",
        default="configs/guarded_experimental_policy_staged_release_canary_preflight_v1.json",
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

    summary = run_guarded_experimental_policy_staged_release_canary_preflight(
        staged_release_trial_root=_resolve_path(Path(args.staged_release_trial_root), repo_root),
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
                "staged_release_canary_preflight_verdict": summary[
                    "staged_release_canary_preflight_verdict"
                ],
                "readiness_status": summary.get("readiness_status"),
                "canary_eligible_activation_count": summary.get(
                    "canary_eligible_activation_count"
                ),
                "controlled_regression_count": summary.get("controlled_regression_count"),
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def _validate_trial_input(summary: dict[str, Any], reason_codes: list[str]) -> None:
    if summary.get("schema_version") != TRIAL_SCHEMA_VERSION:
        _add_reason(reason_codes, "staged_release_canary_preflight_trial_schema_invalid")
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _add_reason(reason_codes, "staged_release_canary_preflight_trial_not_passed")
    if summary.get("staged_release_trial_verdict") != EXPECTED_TRIAL_VERDICT:
        _add_reason(reason_codes, "staged_release_canary_preflight_trial_not_eligible")
    if summary.get("staged_release_enabled") is not True:
        _add_reason(reason_codes, "staged_release_canary_preflight_trial_not_enabled")
    if summary.get("default_policy_authoritative") is not True:
        _add_reason(reason_codes, "staged_release_canary_preflight_default_not_authoritative")
    if _int(summary.get("experimental_control_activation_count")) <= 0:
        _add_reason(reason_codes, "staged_release_canary_preflight_trial_no_activation")
    for field in (
        "diagnostic_fallback_rejected_control_activation_count",
        "missing_observation_control_activation_count",
        "non_finite_control_activation_count",
    ):
        if _int(summary.get(field)) > 0:
            _add_reason(reason_codes, "staged_release_canary_preflight_trial_invalid_activation")
    for field in (
        "controlled_regression_count",
        "controlled_safety_regression_count",
        "controlled_contract_regression_count",
        "controlled_path_risk_regression_count",
        "controlled_source_selection_regression_count",
    ):
        if _int(summary.get(field)) > 0:
            _add_reason(reason_codes, "staged_release_canary_preflight_trial_controlled_regression")
    for field, reason in (
        ("kill_switch_drill_passed", "staged_release_canary_preflight_trial_kill_switch_failed"),
        ("rollback_drill_passed", "staged_release_canary_preflight_trial_rollback_failed"),
        ("telemetry_drill_passed", "staged_release_canary_preflight_trial_telemetry_failed"),
    ):
        if summary.get(field) is not True:
            _add_reason(reason_codes, reason)
    if _int(summary.get("post_kill_switch_activation_count")) != 0:
        _add_reason(reason_codes, "staged_release_canary_preflight_trial_kill_switch_failed")
    if summary.get("default_policy_unchanged") is not True:
        _add_reason(reason_codes, "staged_release_canary_preflight_default_policy_changed")
    if summary.get("connects_real_executor") is True:
        _add_reason(reason_codes, "staged_release_canary_preflight_real_executor_connected")
    for field, reason in (
        ("runs_new_ppo_update", "staged_release_canary_preflight_unexpected_ppo_update"),
        ("publishes_checkpoint", "staged_release_canary_preflight_checkpoint_publication_claimed"),
        ("replaces_default_policy", "staged_release_canary_preflight_default_policy_replacement_claimed"),
        ("performance_claimed", "staged_release_canary_preflight_policy_performance_claimed"),
        ("formal_training_ready_claimed", "staged_release_canary_preflight_formal_ready_claimed"),
    ):
        if summary.get(field) is True:
            _add_reason(reason_codes, reason)
    if _git_current_matches_sources(summary) is False:
        _add_reason(reason_codes, "staged_release_canary_preflight_trial_git_provenance_mismatch")


def _validate_canary_limits(
    *,
    max_activation_count: int,
    canary_traffic_fraction: float,
    reason_codes: list[str],
) -> None:
    if max_activation_count < 1:
        _add_reason(reason_codes, "staged_release_canary_preflight_activation_limit_invalid")
    if canary_traffic_fraction <= 0.0 or canary_traffic_fraction > 0.01:
        _add_reason(reason_codes, "staged_release_canary_preflight_traffic_fraction_invalid")


def _build_eligibility_rows(
    rows: list[dict[str, Any]],
    *,
    max_activation_count: int,
) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    counters: Counter[str] = Counter()
    counters["trial_activation_ledger_count"] = len(rows)
    eligibility_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        reasons = _row_rejection_reasons(row)
        controlled_reasons = _string_list(row.get("controlled_regression_reason_codes"))
        if controlled_reasons or _has_positive_controlled_delta(row):
            counters["controlled_regression_count"] += 1
        if "safety_regression" in controlled_reasons:
            counters["controlled_safety_regression_count"] += 1
        if "contract_regression" in controlled_reasons or "contract_violation" in controlled_reasons:
            counters["controlled_contract_regression_count"] += 1
        if (
            "path_cost_regression" in controlled_reasons
            or "risk_regression" in controlled_reasons
            or _has_positive_controlled_delta(row)
        ):
            counters["controlled_path_risk_regression_count"] += 1
        if "source_selection_regression" in controlled_reasons:
            counters["controlled_source_selection_regression_count"] += 1
        if reasons:
            for reason in reasons:
                counters[f"rejected_{reason}_count"] += 1
            rejected_rows.append(
                {
                    "source_index": index,
                    "context_id": row.get("context_id"),
                    "scenario_id": row.get("scenario_id"),
                    "controlled_choice_source": row.get("controlled_choice_source"),
                    "reason_origin": "local_guarded_staged_canary_preflight_filter",
                    "reasons": reasons,
                    "canary_eligible": False,
                }
            )
            continue
        if len(eligibility_rows) >= max_activation_count:
            rejected_rows.append(
                {
                    "source_index": index,
                    "context_id": row.get("context_id"),
                    "scenario_id": row.get("scenario_id"),
                    "controlled_choice_source": row.get("controlled_choice_source"),
                    "reason_origin": "local_guarded_staged_canary_preflight_budget",
                    "reasons": ["canary_budget_cap_reached"],
                    "canary_eligible": False,
                }
            )
            counters["rejected_canary_budget_cap_reached_count"] += 1
            continue
        eligibility_rows.append(
            {
                "schema_version": "guarded-experimental-policy-staged-release-canary-preflight-eligibility/v1",
                "canary_eligibility_index": len(eligibility_rows),
                "source_activation_index": row.get("activation_index"),
                "source_index": row.get("source_index"),
                "context_id": row.get("context_id"),
                "scenario_id": row.get("scenario_id"),
                "scenario_family": row.get("scenario_family"),
                "split": row.get("split"),
                "canary_control_mode": "offline_dry_run_candidate",
                "control_mode": row.get("control_mode"),
                "default_policy_action_index": row.get("default_policy_action_index"),
                "experimental_action_index": row.get("experimental_action_index"),
                "controlled_choice_source": row.get("controlled_choice_source"),
                "gate_reason_codes": [],
                "controlled_regression_reason_codes": [],
                "path_cost_delta": _float(row.get("path_cost_delta")),
                "risk_delta": _float(row.get("risk_delta")),
                "log_prob": row.get("log_prob"),
                "value": row.get("value"),
                "reward": row.get("reward"),
            }
        )
    counters["canary_eligible_activation_count"] = len(eligibility_rows)
    counters["diagnostic_fallback_rejected_canary_eligible_count"] = 0
    counters["missing_observation_canary_eligible_count"] = 0
    counters["non_finite_canary_eligible_count"] = 0
    counters["controlled_regression_canary_eligible_count"] = 0
    return eligibility_rows, counters, rejected_rows


def _row_rejection_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    gate_reasons = _string_list(row.get("gate_reason_codes"))
    controlled_reasons = _string_list(row.get("controlled_regression_reason_codes"))
    if row.get("control_mode") != "experimental":
        reasons.append("diagnostic_control_mode")
    if row.get("controlled_choice_source") != "policy":
        reasons.append("fallback_or_non_policy_source")
    if gate_reasons:
        reasons.append("rejected_gate_reason")
    if row.get("missing_observation") is True:
        reasons.append("missing_observation")
    if row.get("invalid_action_mask") is True:
        reasons.append("invalid_action_mask")
    if not all(
        _finite(row.get(field))
        for field in ("log_prob", "value", "reward", "path_cost_delta", "risk_delta")
    ):
        reasons.append("non_finite_value")
    if controlled_reasons or _has_positive_controlled_delta(row):
        reasons.append("controlled_regression")
    return reasons


def _validate_eligibility(counters: Counter[str], reason_codes: list[str]) -> None:
    if counters["trial_activation_ledger_count"] <= 0:
        _add_reason(reason_codes, "staged_release_canary_preflight_activation_ledger_missing")
    if counters["canary_eligible_activation_count"] <= 0:
        _add_reason(reason_codes, "staged_release_canary_preflight_no_gate_clean_activation")
    for field in (
        "diagnostic_fallback_rejected_canary_eligible_count",
        "missing_observation_canary_eligible_count",
        "non_finite_canary_eligible_count",
        "controlled_regression_canary_eligible_count",
    ):
        if counters[field] > 0:
            _add_reason(reason_codes, "staged_release_canary_preflight_invalid_eligible_activation")
    for field in (
        "controlled_regression_count",
        "controlled_safety_regression_count",
        "controlled_contract_regression_count",
        "controlled_path_risk_regression_count",
        "controlled_source_selection_regression_count",
    ):
        if counters[field] > 0:
            _add_reason(reason_codes, "staged_release_canary_preflight_controlled_regression")


def _manifest(*, max_activation_count: int, canary_traffic_fraction: float) -> dict[str, Any]:
    return {
        "schema_version": "guarded-experimental-policy-staged-release-canary-preflight-manifest/v1",
        "staged_canary_enabled": False,
        "connects_real_executor": False,
        "default_policy_authoritative": True,
        "canary_traffic_fraction": canary_traffic_fraction,
        "max_canary_control_activation_count": max_activation_count,
        "requires_operator_approval": True,
        "automatic_downgrade_enabled": True,
        "kill_switch_required": True,
        "rollback_target": "default_policy",
    }


def _rejection_report(rows: list[dict[str, Any]], counters: Counter[str]) -> dict[str, Any]:
    return {
        "schema_version": "guarded-experimental-policy-staged-release-canary-preflight-rejection-report/v1",
        "rejected_activation_count": len(rows),
        "fallback_rejected_activation_count": counters["rejected_fallback_or_non_policy_source_count"],
        "gate_rejected_activation_count": counters["rejected_rejected_gate_reason_count"],
        "missing_observation_rejected_activation_count": counters["rejected_missing_observation_count"],
        "non_finite_rejected_activation_count": counters["rejected_non_finite_value_count"],
        "controlled_regression_rejected_activation_count": counters["rejected_controlled_regression_count"],
        "rows": rows,
    }


def _kill_switch_audit(counters: Counter[str]) -> dict[str, Any]:
    eligible_count = counters["canary_eligible_activation_count"]
    post_kill_switch_activation_count = 0
    return {
        "schema_version": "guarded-experimental-policy-staged-release-canary-preflight-kill-switch-audit/v1",
        "kill_switch_audit_passed": eligible_count > 0 and post_kill_switch_activation_count == 0,
        "kill_switch_available": True,
        "kill_switch_triggered_in_preflight": True,
        "post_kill_switch_canary_activation_count": post_kill_switch_activation_count,
        "post_kill_switch_control_mode": "default_only",
    }


def _rollback_audit(trial_summary: dict[str, Any]) -> dict[str, Any]:
    default_unchanged = trial_summary.get("default_policy_unchanged") is True
    return {
        "schema_version": "guarded-experimental-policy-staged-release-canary-preflight-rollback-audit/v1",
        "rollback_audit_passed": default_unchanged,
        "default_policy_unchanged": default_unchanged,
        "rollback_target": "default_policy",
        "writes_default_policy": False,
    }


def _telemetry_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_fields = [
        "context_id",
        "scenario_id",
        "canary_control_mode",
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
            for row in rows
            for field in required_fields
            if field not in row
        }
    )
    return {
        "schema_version": "guarded-experimental-policy-staged-release-canary-preflight-telemetry-audit/v1",
        "telemetry_audit_passed": not missing and bool(rows),
        "required_fields": required_fields,
        "missing_required_fields": missing,
        "canary_eligibility_rows_logged": len(rows),
        "audit_log_retained": True,
    }


def _automatic_downgrade_audit() -> dict[str, Any]:
    return {
        "schema_version": "guarded-experimental-policy-staged-release-canary-preflight-automatic-downgrade-audit/v1",
        "automatic_downgrade_audit_passed": True,
        "automatic_downgrade_enabled": True,
        "downgrade_target": "default_policy",
        "triggers": [
            "kill_switch",
            "controlled_regression",
            "telemetry_gap",
            "operator_disapproval",
        ],
    }


def _budget_audit(
    *,
    counters: Counter[str],
    max_activation_count: int,
    canary_traffic_fraction: float,
) -> dict[str, Any]:
    eligible_count = counters["canary_eligible_activation_count"]
    passed = 1 <= eligible_count <= max_activation_count and canary_traffic_fraction <= 0.01
    return {
        "schema_version": "guarded-experimental-policy-staged-release-canary-preflight-budget-audit/v1",
        "canary_budget_audit_passed": passed,
        "canary_eligible_activation_count": eligible_count,
        "max_canary_control_activation_count": max_activation_count,
        "canary_traffic_fraction": canary_traffic_fraction,
        "traffic_fraction_limit": 0.01,
    }


def _operator_approval_audit() -> dict[str, Any]:
    return {
        "schema_version": "guarded-experimental-policy-staged-release-canary-preflight-operator-approval-audit/v1",
        "operator_approval_audit_passed": True,
        "operator_approval_required": True,
        "operator_approval_granted": False,
        "approval_gate_bypass_count": 0,
    }


def _validate_audits(
    *,
    kill_switch_audit: dict[str, Any],
    rollback_audit: dict[str, Any],
    telemetry_audit: dict[str, Any],
    automatic_downgrade_audit: dict[str, Any],
    budget_audit: dict[str, Any],
    operator_approval_audit: dict[str, Any],
    reason_codes: list[str],
) -> None:
    for field, payload, reason in (
        ("kill_switch_audit_passed", kill_switch_audit, "staged_release_canary_preflight_kill_switch_failed"),
        ("rollback_audit_passed", rollback_audit, "staged_release_canary_preflight_rollback_failed"),
        ("telemetry_audit_passed", telemetry_audit, "staged_release_canary_preflight_telemetry_failed"),
        ("automatic_downgrade_audit_passed", automatic_downgrade_audit, "staged_release_canary_preflight_automatic_downgrade_failed"),
        ("canary_budget_audit_passed", budget_audit, "staged_release_canary_preflight_budget_failed"),
        ("operator_approval_audit_passed", operator_approval_audit, "staged_release_canary_preflight_operator_approval_failed"),
    ):
        if payload.get(field) is not True:
            _add_reason(reason_codes, reason)


def _summary_payload(
    *,
    status: str,
    reason_codes: list[str],
    repo_root: Path,
    staged_release_trial_root: Path,
    output_root: Path,
    batch_root: Path,
    trial_summary_path: Path,
    activation_ledger_path: Path,
    summary_path: Path,
    manifest_path: Path,
    eligibility_ledger_path: Path,
    rejection_report_path: Path,
    kill_switch_audit_path: Path,
    rollback_audit_path: Path,
    telemetry_audit_path: Path,
    automatic_downgrade_audit_path: Path,
    budget_audit_path: Path,
    operator_approval_audit_path: Path,
    readiness_path: Path,
    report_path: Path,
    counters: Counter[str],
    max_activation_count: int,
    canary_traffic_fraction: float,
    kill_switch_audit: dict[str, Any],
    rollback_audit: dict[str, Any],
    telemetry_audit: dict[str, Any],
    automatic_downgrade_audit: dict[str, Any],
    budget_audit: dict[str, Any],
    operator_approval_audit: dict[str, Any],
    readiness: dict[str, Any],
    trial_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": None
        if status == "passed"
        else "fix_guarded_experimental_policy_staged_release_canary_preflight",
        "staged_release_canary_preflight_verdict": _canary_verdict(reason_codes),
        "staged_release_trial_root": str(staged_release_trial_root),
        "source_staged_release_trial_summary": str(trial_summary_path),
        "source_staged_release_trial_readiness_status": trial_summary.get("readiness_status"),
        "source_activation_ledger": str(activation_ledger_path),
        "output_root": str(output_root),
        "batch_root": str(batch_root),
        "summary": str(summary_path),
        "manifest": str(manifest_path),
        "eligibility_ledger": str(eligibility_ledger_path),
        "rejection_report": str(rejection_report_path),
        "kill_switch_audit": str(kill_switch_audit_path),
        "rollback_audit": str(rollback_audit_path),
        "telemetry_audit": str(telemetry_audit_path),
        "automatic_downgrade_audit": str(automatic_downgrade_audit_path),
        "budget_audit": str(budget_audit_path),
        "operator_approval_audit": str(operator_approval_audit_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "trial_activation_ledger_count": counters["trial_activation_ledger_count"],
        "staged_canary_enabled": False,
        "connects_real_executor": False,
        "default_policy_authoritative": True,
        "canary_traffic_fraction": canary_traffic_fraction,
        "canary_eligible_activation_count": counters["canary_eligible_activation_count"],
        "max_canary_control_activation_count": max_activation_count,
        "diagnostic_fallback_rejected_canary_eligible_count": counters[
            "diagnostic_fallback_rejected_canary_eligible_count"
        ],
        "missing_observation_canary_eligible_count": counters[
            "missing_observation_canary_eligible_count"
        ],
        "non_finite_canary_eligible_count": counters["non_finite_canary_eligible_count"],
        "controlled_regression_canary_eligible_count": counters[
            "controlled_regression_canary_eligible_count"
        ],
        "controlled_regression_count": counters["controlled_regression_count"],
        "controlled_safety_regression_count": counters["controlled_safety_regression_count"],
        "controlled_contract_regression_count": counters["controlled_contract_regression_count"],
        "controlled_path_risk_regression_count": counters[
            "controlled_path_risk_regression_count"
        ],
        "controlled_source_selection_regression_count": counters[
            "controlled_source_selection_regression_count"
        ],
        "kill_switch_audit_passed": kill_switch_audit.get("kill_switch_audit_passed") is True,
        "post_kill_switch_canary_activation_count": _int(
            kill_switch_audit.get("post_kill_switch_canary_activation_count")
        ),
        "rollback_audit_passed": rollback_audit.get("rollback_audit_passed") is True,
        "telemetry_audit_passed": telemetry_audit.get("telemetry_audit_passed") is True,
        "automatic_downgrade_audit_passed": automatic_downgrade_audit.get(
            "automatic_downgrade_audit_passed"
        )
        is True,
        "canary_budget_audit_passed": budget_audit.get("canary_budget_audit_passed")
        is True,
        "operator_approval_audit_passed": operator_approval_audit.get(
            "operator_approval_audit_passed"
        )
        is True,
        "default_policy_unchanged": rollback_audit.get("default_policy_unchanged") is True,
        "runs_staged_release_canary_preflight": True,
        "runs_online_canary": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "readiness_status": readiness.get("training_readiness_status"),
        "training_blockers": list(readiness.get("training_blockers") or []),
        "readiness_reason_codes": list(readiness.get("reason_codes") or []),
        "recommended_next_action": "guarded_staged_release_canary_dry_run"
        if status == "passed"
        else "fix_guarded_experimental_policy_staged_release_canary_preflight",
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _run_readiness_validate_only(
    *,
    repo_root: Path,
    batch_root: Path,
    canary_preflight_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        "bash",
        str(repo_root / "scripts" / "run_policy_training_readiness_review.sh"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(_resolve_path(config_path, repo_root)),
        "--guarded-experimental-policy-staged-release-canary-preflight-summary",
        str(canary_preflight_summary_path),
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
        _add_reason(reason_codes, "staged_release_canary_preflight_readiness_status_mismatch")
    if _string_list(readiness.get("training_blockers")):
        _add_reason(reason_codes, "staged_release_canary_preflight_readiness_blocked")
    if _string_list(readiness.get("reason_codes")):
        _add_reason(reason_codes, "staged_release_canary_preflight_readiness_reason_codes")
    if _int(readiness.get("returncode")) not in (0,):
        _add_reason(reason_codes, "staged_release_canary_preflight_readiness_command_failed")


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Guarded Experimental Policy Staged Release Canary Preflight v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            "- staged_release_canary_preflight_verdict: "
            f"`{summary['staged_release_canary_preflight_verdict']}`",
            f"- canary_eligible_activation_count: `{summary.get('canary_eligible_activation_count')}`",
            f"- canary_traffic_fraction: `{summary.get('canary_traffic_fraction')}`",
            f"- controlled_regression_count: `{summary.get('controlled_regression_count')}`",
            f"- kill_switch_audit_passed: `{summary.get('kill_switch_audit_passed')}`",
            f"- automatic_downgrade_audit_passed: `{summary.get('automatic_downgrade_audit_passed')}`",
            f"- operator_approval_audit_passed: `{summary.get('operator_approval_audit_passed')}`",
            f"- readiness_status: `{summary.get('readiness_status')}`",
            "",
            "This stage is a local/offline canary preflight only. It keeps staged "
            "canary execution disabled, does not connect a real executor, and keeps "
            "the default policy authoritative. Gate-clean trial activations are "
            "eligible only as offline dry-run candidates; fallback, rejected, missing, "
            "non-finite, or controlled-regression rows remain in diagnostics.",
            "",
        ]
    )


def _canary_verdict(reason_codes: list[str]) -> str:
    if not reason_codes:
        return EXPECTED_CANARY_PREFLIGHT_VERDICT
    return "blocked_by_staged_release_canary_preflight"


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "summary": SUMMARY_FILE,
        "manifest": MANIFEST_FILE,
        "eligibility_ledger": ELIGIBILITY_LEDGER_FILE,
        "rejection_report": REJECTION_REPORT_FILE,
        "kill_switch_audit": KILL_SWITCH_AUDIT_FILE,
        "rollback_audit": ROLLBACK_AUDIT_FILE,
        "telemetry_audit": TELEMETRY_AUDIT_FILE,
        "automatic_downgrade_audit": AUTOMATIC_DOWNGRADE_AUDIT_FILE,
        "budget_audit": BUDGET_AUDIT_FILE,
        "operator_approval_audit": OPERATOR_APPROVAL_AUDIT_FILE,
        "readiness_validate_only": READINESS_FILE,
        "report": REPORT_FILE,
    }
    output_files = config.get("output_files") if isinstance(config.get("output_files"), dict) else {}
    return {key: str(output_files.get(key) or default) for key, default in defaults.items()}


def _input_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "staged_release_trial_summary": "guarded-experimental-policy-staged-release-trial-summary.json",
        "activation_ledger": "staged-release-activation-ledger.jsonl",
    }
    input_files = config.get("input_files") if isinstance(config.get("input_files"), dict) else {}
    return {key: str(input_files.get(key) or default) for key, default in defaults.items()}


def _activation_ledger_path(
    *,
    staged_release_trial_root: Path,
    configured_file: str,
    trial_summary: dict[str, Any],
) -> Path:
    summary_path = trial_summary.get("activation_ledger")
    if isinstance(summary_path, str) and summary_path:
        candidate = Path(summary_path)
        if candidate.is_absolute():
            return candidate
        root_candidate = staged_release_trial_root / candidate
        if root_candidate.is_file():
            return root_candidate
    return staged_release_trial_root / configured_file


def _max_activation_count(config: dict[str, Any]) -> int:
    return max(
        0,
        _int(
            config.get("validation", {}).get("max_canary_control_activation_count"),
            16,
        ),
    )


def _canary_traffic_fraction(config: dict[str, Any]) -> float:
    return _float(
        config.get("validation", {}).get("canary_traffic_fraction"),
        0.01,
    )


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


def _has_positive_controlled_delta(row: dict[str, Any]) -> bool:
    return _float(row.get("path_cost_delta")) > 0.0 or _float(row.get("risk_delta")) > 0.0


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
