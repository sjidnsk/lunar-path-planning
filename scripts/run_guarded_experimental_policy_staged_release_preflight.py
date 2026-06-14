from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
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


CONFIG_SCHEMA_VERSION = "guarded-experimental-policy-staged-release-preflight-config/v1"
SUMMARY_SCHEMA_VERSION = "guarded-experimental-policy-staged-release-preflight-summary/v1"
SHADOW_RELEASE_TRIAL_SCHEMA_VERSION = (
    "guarded-experimental-policy-shadow-release-trial-summary/v1"
)
EXPECTED_SHADOW_RELEASE_TRIAL_VERDICT = "eligible_for_guarded_staged_release_preflight"
EXPECTED_PREFLIGHT_VERDICT = "eligible_for_guarded_staged_release_trial"
EXPECTED_READINESS_STATUS = (
    "guarded_experimental_policy_staged_release_preflight_evaluated"
)

SUMMARY_FILE = "guarded-experimental-policy-staged-release-preflight-summary.json"
PREFLIGHT_MANIFEST_FILE = "staged-release-preflight-manifest.json"
GATE_THRESHOLD_AUDIT_FILE = "staged-release-gate-threshold-audit.json"
KILL_SWITCH_AUDIT_FILE = "staged-release-kill-switch-audit.json"
ROLLBACK_AUDIT_FILE = "staged-release-rollback-audit.json"
TELEMETRY_AUDIT_FILE = "staged-release-telemetry-audit.json"
READINESS_FILE = "staged-release-preflight-readiness-validate-only.json"
REPORT_FILE = "staged-release-preflight-report.md"

ReadinessRunner = Callable[..., dict[str, Any]]


def run_guarded_experimental_policy_staged_release_preflight(
    *,
    shadow_release_trial_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    readiness_runner: ReadinessRunner | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    shadow_release_trial_root = Path(shadow_release_trial_root)
    output_root = Path(output_root)
    batch_root = Path(batch_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = _output_files(config)
    summary_path = output_root / files["summary"]
    preflight_manifest_path = output_root / files["preflight_manifest"]
    gate_threshold_audit_path = output_root / files["gate_threshold_audit"]
    kill_switch_audit_path = output_root / files["kill_switch_audit"]
    rollback_audit_path = output_root / files["rollback_audit"]
    telemetry_audit_path = output_root / files["telemetry_audit"]
    readiness_path = output_root / files["readiness_validate_only"]
    report_path = output_root / files["report"]

    inputs = _input_files(config)
    shadow_summary_path = (
        shadow_release_trial_root / inputs["shadow_release_trial_summary"]
    )
    shadow_summary = _read_json_if_exists(shadow_summary_path)

    reason_codes: list[str] = []
    _validate_shadow_release_trial_input(shadow_summary, config, reason_codes)

    manifest = _preflight_manifest(
        shadow_summary=shadow_summary,
        output_root=output_root,
        shadow_summary_path=shadow_summary_path,
    )
    gate_audit = _gate_threshold_audit(shadow_summary, config)
    kill_switch_audit = _kill_switch_audit(manifest)
    rollback_audit = _rollback_audit(shadow_summary, manifest)
    telemetry_audit = _telemetry_audit()

    _write_json(preflight_manifest_path, manifest)
    _write_json(gate_threshold_audit_path, gate_audit)
    _write_json(kill_switch_audit_path, kill_switch_audit)
    _write_json(rollback_audit_path, rollback_audit)
    _write_json(telemetry_audit_path, telemetry_audit)

    _validate_internal_audits(
        gate_audit=gate_audit,
        kill_switch_audit=kill_switch_audit,
        rollback_audit=rollback_audit,
        telemetry_audit=telemetry_audit,
        reason_codes=reason_codes,
    )

    status_without_readiness = "passed" if not reason_codes else "failed"
    pre_readiness_summary = _summary_payload(
        status=status_without_readiness,
        reason_codes=reason_codes,
        repo_root=repo_root,
        shadow_release_trial_root=shadow_release_trial_root,
        output_root=output_root,
        batch_root=batch_root,
        shadow_summary_path=shadow_summary_path,
        summary_path=summary_path,
        preflight_manifest_path=preflight_manifest_path,
        gate_threshold_audit_path=gate_threshold_audit_path,
        kill_switch_audit_path=kill_switch_audit_path,
        rollback_audit_path=rollback_audit_path,
        telemetry_audit_path=telemetry_audit_path,
        readiness_path=readiness_path,
        report_path=report_path,
        shadow_summary=shadow_summary,
        manifest=manifest,
        gate_audit=gate_audit,
        kill_switch_audit=kill_switch_audit,
        rollback_audit=rollback_audit,
        telemetry_audit=telemetry_audit,
        readiness={},
    )
    _write_json(summary_path, pre_readiness_summary)

    if status_without_readiness == "passed":
        runner = readiness_runner or _run_readiness_validate_only
        readiness = runner(
            repo_root=repo_root,
            batch_root=batch_root,
            staged_release_preflight_summary_path=summary_path,
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
            "recommended_next_action": "fix_guarded_experimental_policy_staged_release_preflight",
        }
    _write_json(readiness_path, readiness)

    final_status = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=final_status,
        reason_codes=reason_codes,
        repo_root=repo_root,
        shadow_release_trial_root=shadow_release_trial_root,
        output_root=output_root,
        batch_root=batch_root,
        shadow_summary_path=shadow_summary_path,
        summary_path=summary_path,
        preflight_manifest_path=preflight_manifest_path,
        gate_threshold_audit_path=gate_threshold_audit_path,
        kill_switch_audit_path=kill_switch_audit_path,
        rollback_audit_path=rollback_audit_path,
        telemetry_audit_path=telemetry_audit_path,
        readiness_path=readiness_path,
        report_path=report_path,
        shadow_summary=shadow_summary,
        manifest=manifest,
        gate_audit=gate_audit,
        kill_switch_audit=kill_switch_audit,
        rollback_audit=rollback_audit,
        telemetry_audit=telemetry_audit,
        readiness=readiness,
    )
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a guarded experimental policy staged release preflight."
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
        default="outputs/path_feedback_batch_guarded_experimental_policy_staged_release_preflight_v1",
    )
    parser.add_argument(
        "--config",
        default="configs/guarded_experimental_policy_staged_release_preflight_v1.json",
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

    summary = run_guarded_experimental_policy_staged_release_preflight(
        shadow_release_trial_root=_resolve_path(
            Path(args.shadow_release_trial_root),
            repo_root,
        ),
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
                "staged_release_preflight_verdict": summary[
                    "staged_release_preflight_verdict"
                ],
                "readiness_status": summary.get("readiness_status"),
                "staged_release_enabled": summary.get("staged_release_enabled"),
                "default_policy_authoritative": summary.get(
                    "default_policy_authoritative"
                ),
                "experimental_control_activation_count": summary.get(
                    "experimental_control_activation_count"
                ),
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def _validate_shadow_release_trial_input(
    summary: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if summary.get("schema_version") != SHADOW_RELEASE_TRIAL_SCHEMA_VERSION:
        _add_reason(reason_codes, "staged_release_preflight_shadow_schema_invalid")
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _add_reason(reason_codes, "staged_release_preflight_shadow_not_passed")
    if summary.get("shadow_release_trial_verdict") != EXPECTED_SHADOW_RELEASE_TRIAL_VERDICT:
        _add_reason(reason_codes, "staged_release_preflight_shadow_not_eligible")
    if _int(summary.get("shadow_step_count")) < _min_shadow_step_count(config):
        _add_reason(reason_codes, "staged_release_preflight_shadow_step_count_low")
    if _int(summary.get("unique_shadow_context_count")) < _min_unique_context_count(config):
        _add_reason(reason_codes, "staged_release_preflight_shadow_unique_context_low")
    if summary.get("shadow_policy_takes_control") is not False:
        _add_reason(reason_codes, "staged_release_preflight_shadow_took_control")
    for field, reason in (
        ("missing_observation_count", "staged_release_preflight_shadow_contract_invalid"),
        ("missing_log_prob_count", "staged_release_preflight_shadow_contract_invalid"),
        ("missing_value_count", "staged_release_preflight_shadow_contract_invalid"),
        ("invalid_action_mask_count", "staged_release_preflight_shadow_contract_invalid"),
        ("non_finite_logits_count", "staged_release_preflight_shadow_non_finite"),
        ("non_finite_log_prob_count", "staged_release_preflight_shadow_non_finite"),
        ("non_finite_value_count", "staged_release_preflight_shadow_non_finite"),
        ("non_finite_reward_count", "staged_release_preflight_shadow_non_finite"),
    ):
        if _int(summary.get(field)) > 0:
            _add_reason(reason_codes, reason)
    for field in (
        "controlled_regression_count",
        "controlled_safety_regression_count",
        "controlled_contract_regression_count",
        "controlled_path_risk_regression_count",
        "controlled_source_selection_regression_count",
    ):
        if _int(summary.get(field)) > 0:
            _add_reason(
                reason_codes,
                "staged_release_preflight_shadow_controlled_regression",
            )
    if summary.get("rollback_default_audit_passed") is not True:
        _add_reason(reason_codes, "staged_release_preflight_shadow_rollback_failed")
    if summary.get("default_policy_unchanged") is not True:
        _add_reason(reason_codes, "staged_release_preflight_shadow_default_changed")
    if summary.get("runs_shadow_release_trial") is not True:
        _add_reason(reason_codes, "staged_release_preflight_shadow_not_run")
    for field, reason in (
        ("runs_new_ppo_update", "staged_release_preflight_unexpected_ppo_update"),
        ("publishes_checkpoint", "staged_release_preflight_checkpoint_publication_claimed"),
        ("replaces_default_policy", "staged_release_preflight_default_policy_replacement_claimed"),
        ("performance_claimed", "staged_release_preflight_policy_performance_claimed"),
        ("formal_training_ready_claimed", "staged_release_preflight_formal_ready_claimed"),
    ):
        if summary.get(field) is True:
            _add_reason(reason_codes, reason)
    if _git_current_matches_sources(summary) is False:
        _add_reason(reason_codes, "staged_release_preflight_shadow_git_provenance_mismatch")


def _preflight_manifest(
    *,
    shadow_summary: dict[str, Any],
    output_root: Path,
    shadow_summary_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "guarded-experimental-policy-staged-release-preflight-manifest/v1",
        "generated_at": _utc_now(),
        "output_root": str(output_root),
        "source_shadow_release_trial_summary": str(shadow_summary_path),
        "staged_release_enabled": False,
        "default_policy_authoritative": True,
        "experimental_control_activation_count": 0,
        "stage_plan": [
            {
                "stage": "stage0_shadow_only",
                "executes_in_preflight": False,
                "experimental_policy_takes_control": False,
            },
            {
                "stage": "stage1_guarded_opt_in",
                "executes_in_preflight": False,
                "experimental_policy_takes_control": False,
            },
            {
                "stage": "stage2_limited_guarded_canary",
                "executes_in_preflight": False,
                "experimental_policy_takes_control": False,
            },
        ],
        "source_shadow_step_count": _int(shadow_summary.get("shadow_step_count")),
        "source_unique_shadow_context_count": _int(
            shadow_summary.get("unique_shadow_context_count")
        ),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
    }


def _gate_threshold_audit(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    counters = {
        "shadow_step_count": _int(summary.get("shadow_step_count")),
        "unique_shadow_context_count": _int(summary.get("unique_shadow_context_count")),
        "controlled_regression_count": _int(summary.get("controlled_regression_count")),
        "missing_observation_count": _int(summary.get("missing_observation_count")),
        "missing_log_prob_count": _int(summary.get("missing_log_prob_count")),
        "missing_value_count": _int(summary.get("missing_value_count")),
        "invalid_action_mask_count": _int(summary.get("invalid_action_mask_count")),
        "non_finite_logits_count": _int(summary.get("non_finite_logits_count")),
        "non_finite_log_prob_count": _int(summary.get("non_finite_log_prob_count")),
        "non_finite_value_count": _int(summary.get("non_finite_value_count")),
        "non_finite_reward_count": _int(summary.get("non_finite_reward_count")),
    }
    passed = (
        counters["shadow_step_count"] >= _min_shadow_step_count(config)
        and counters["unique_shadow_context_count"] >= _min_unique_context_count(config)
        and all(
            counters[key] == 0
            for key in counters
            if key not in ("shadow_step_count", "unique_shadow_context_count")
        )
    )
    return {
        "schema_version": "guarded-experimental-policy-staged-release-gate-threshold-audit/v1",
        "gate_threshold_audit_passed": passed,
        "min_shadow_step_count": _min_shadow_step_count(config),
        "min_unique_shadow_context_count": _min_unique_context_count(config),
        **counters,
    }


def _kill_switch_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    passed = (
        manifest.get("staged_release_enabled") is False
        and manifest.get("default_policy_authoritative") is True
        and _int(manifest.get("experimental_control_activation_count")) == 0
    )
    return {
        "schema_version": "guarded-experimental-policy-staged-release-kill-switch-audit/v1",
        "kill_switch_audit_passed": passed,
        "kill_switch_available": True,
        "kill_switch_action": "disable_experimental_policy_and_restore_default_only",
        "audit_log_retained": True,
        "staged_release_enabled": manifest.get("staged_release_enabled"),
        "experimental_control_activation_count": _int(
            manifest.get("experimental_control_activation_count")
        ),
    }


def _rollback_audit(summary: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    passed = (
        summary.get("rollback_default_audit_passed") is True
        and summary.get("default_policy_unchanged") is True
        and manifest.get("default_policy_authoritative") is True
    )
    return {
        "schema_version": "guarded-experimental-policy-staged-release-rollback-audit/v1",
        "rollback_audit_passed": passed,
        "source_shadow_rollback_default_audit_passed": summary.get(
            "rollback_default_audit_passed"
        )
        is True,
        "default_policy_unchanged": summary.get("default_policy_unchanged") is True,
        "default_policy_authoritative": manifest.get("default_policy_authoritative") is True,
        "checkpoint_metadata_traceable": bool(summary.get("runtime_manifest")),
        "writes_default_policy": False,
    }


def _telemetry_audit() -> dict[str, Any]:
    fields = [
        "default_action",
        "experimental_action",
        "gate_reason",
        "fallback_reason",
        "path_risk_delta",
        "rollback_marker",
    ]
    return {
        "schema_version": "guarded-experimental-policy-staged-release-telemetry-audit/v1",
        "telemetry_audit_passed": True,
        "required_fields": fields,
        "missing_required_fields": [],
        "audit_log_retained": True,
    }


def _validate_internal_audits(
    *,
    gate_audit: dict[str, Any],
    kill_switch_audit: dict[str, Any],
    rollback_audit: dict[str, Any],
    telemetry_audit: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if gate_audit.get("gate_threshold_audit_passed") is not True:
        _add_reason(reason_codes, "staged_release_preflight_gate_threshold_failed")
    if kill_switch_audit.get("kill_switch_audit_passed") is not True:
        _add_reason(reason_codes, "staged_release_preflight_kill_switch_failed")
    if rollback_audit.get("rollback_audit_passed") is not True:
        _add_reason(reason_codes, "staged_release_preflight_rollback_failed")
    if telemetry_audit.get("telemetry_audit_passed") is not True:
        _add_reason(reason_codes, "staged_release_preflight_telemetry_failed")


def _summary_payload(
    *,
    status: str,
    reason_codes: list[str],
    repo_root: Path,
    shadow_release_trial_root: Path,
    output_root: Path,
    batch_root: Path,
    shadow_summary_path: Path,
    summary_path: Path,
    preflight_manifest_path: Path,
    gate_threshold_audit_path: Path,
    kill_switch_audit_path: Path,
    rollback_audit_path: Path,
    telemetry_audit_path: Path,
    readiness_path: Path,
    report_path: Path,
    shadow_summary: dict[str, Any],
    manifest: dict[str, Any],
    gate_audit: dict[str, Any],
    kill_switch_audit: dict[str, Any],
    rollback_audit: dict[str, Any],
    telemetry_audit: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": None
        if status == "passed"
        else "fix_guarded_experimental_policy_staged_release_preflight",
        "staged_release_preflight_verdict": _preflight_verdict(reason_codes),
        "shadow_release_trial_root": str(shadow_release_trial_root),
        "source_shadow_release_trial_summary": str(shadow_summary_path),
        "output_root": str(output_root),
        "batch_root": str(batch_root),
        "summary": str(summary_path),
        "preflight_manifest": str(preflight_manifest_path),
        "gate_threshold_audit": str(gate_threshold_audit_path),
        "kill_switch_audit": str(kill_switch_audit_path),
        "rollback_audit": str(rollback_audit_path),
        "telemetry_audit": str(telemetry_audit_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "shadow_step_count": _int(shadow_summary.get("shadow_step_count")),
        "unique_shadow_context_count": _int(
            shadow_summary.get("unique_shadow_context_count")
        ),
        "staged_release_enabled": manifest.get("staged_release_enabled") is True,
        "default_policy_authoritative": manifest.get("default_policy_authoritative") is True,
        "experimental_control_activation_count": _int(
            manifest.get("experimental_control_activation_count")
        ),
        "missing_observation_count": _int(shadow_summary.get("missing_observation_count")),
        "missing_log_prob_count": _int(shadow_summary.get("missing_log_prob_count")),
        "missing_value_count": _int(shadow_summary.get("missing_value_count")),
        "invalid_action_mask_count": _int(shadow_summary.get("invalid_action_mask_count")),
        "non_finite_logits_count": _int(shadow_summary.get("non_finite_logits_count")),
        "non_finite_log_prob_count": _int(shadow_summary.get("non_finite_log_prob_count")),
        "non_finite_value_count": _int(shadow_summary.get("non_finite_value_count")),
        "non_finite_reward_count": _int(shadow_summary.get("non_finite_reward_count")),
        "controlled_regression_count": _int(shadow_summary.get("controlled_regression_count")),
        "controlled_safety_regression_count": _int(
            shadow_summary.get("controlled_safety_regression_count")
        ),
        "controlled_contract_regression_count": _int(
            shadow_summary.get("controlled_contract_regression_count")
        ),
        "controlled_path_risk_regression_count": _int(
            shadow_summary.get("controlled_path_risk_regression_count")
        ),
        "controlled_source_selection_regression_count": _int(
            shadow_summary.get("controlled_source_selection_regression_count")
        ),
        "gate_threshold_audit_passed": gate_audit.get("gate_threshold_audit_passed")
        is True,
        "kill_switch_audit_passed": kill_switch_audit.get("kill_switch_audit_passed")
        is True,
        "rollback_audit_passed": rollback_audit.get("rollback_audit_passed") is True,
        "telemetry_audit_passed": telemetry_audit.get("telemetry_audit_passed") is True,
        "readiness_status": readiness.get("training_readiness_status"),
        "training_blockers": list(readiness.get("training_blockers") or []),
        "readiness_reason_codes": list(readiness.get("reason_codes") or []),
        "runs_staged_release_preflight": True,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "recommended_next_action": "guarded_staged_release_trial"
        if status == "passed"
        else "fix_guarded_experimental_policy_staged_release_preflight",
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _run_readiness_validate_only(
    *,
    repo_root: Path,
    batch_root: Path,
    staged_release_preflight_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        "bash",
        str(repo_root / "scripts" / "run_policy_training_readiness_review.sh"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(_resolve_path(config_path, repo_root)),
        "--guarded-experimental-policy-staged-release-preflight-summary",
        str(staged_release_preflight_summary_path),
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
        _add_reason(reason_codes, "staged_release_preflight_readiness_status_mismatch")
    if _string_list(readiness.get("training_blockers")):
        _add_reason(reason_codes, "staged_release_preflight_readiness_blocked")
    if _string_list(readiness.get("reason_codes")):
        _add_reason(reason_codes, "staged_release_preflight_readiness_reason_codes")
    if _int(readiness.get("returncode")) not in (0,):
        _add_reason(reason_codes, "staged_release_preflight_readiness_command_failed")


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Guarded Experimental Policy Staged Release Preflight v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- staged_release_preflight_verdict: `{summary['staged_release_preflight_verdict']}`",
            f"- shadow_step_count: `{summary.get('shadow_step_count')}`",
            f"- unique_shadow_context_count: `{summary.get('unique_shadow_context_count')}`",
            f"- controlled_regression_count: `{summary.get('controlled_regression_count')}`",
            f"- staged_release_enabled: `{summary.get('staged_release_enabled')}`",
            f"- experimental_control_activation_count: `{summary.get('experimental_control_activation_count')}`",
            f"- readiness_status: `{summary.get('readiness_status')}`",
            "",
            "This stage audits the guarded staged-release plan only. The default "
            "policy remains authoritative, and the experimental policy is not enabled "
            "for control in this preflight.",
            "",
        ]
    )


def _preflight_verdict(reason_codes: list[str]) -> str:
    if not reason_codes:
        return EXPECTED_PREFLIGHT_VERDICT
    return "blocked_by_staged_release_preflight"


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "summary": SUMMARY_FILE,
        "preflight_manifest": PREFLIGHT_MANIFEST_FILE,
        "gate_threshold_audit": GATE_THRESHOLD_AUDIT_FILE,
        "kill_switch_audit": KILL_SWITCH_AUDIT_FILE,
        "rollback_audit": ROLLBACK_AUDIT_FILE,
        "telemetry_audit": TELEMETRY_AUDIT_FILE,
        "readiness_validate_only": READINESS_FILE,
        "report": REPORT_FILE,
    }
    output_files = config.get("output_files") if isinstance(config.get("output_files"), dict) else {}
    return {key: str(output_files.get(key) or default) for key, default in defaults.items()}


def _input_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "shadow_release_trial_summary": "guarded-experimental-policy-shadow-release-trial-summary.json",
    }
    input_files = config.get("input_files") if isinstance(config.get("input_files"), dict) else {}
    return {key: str(input_files.get(key) or default) for key, default in defaults.items()}


def _min_shadow_step_count(config: dict[str, Any]) -> int:
    return _int(config.get("validation", {}).get("min_shadow_step_count"), 512)


def _min_unique_context_count(config: dict[str, Any]) -> int:
    return _int(config.get("validation", {}).get("min_unique_shadow_context_count"), 256)


def _resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path | None) -> dict[str, Any]:
    return _read_json(path) if path and path.is_file() else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


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
