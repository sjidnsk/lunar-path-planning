from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot


DEFAULT_STAGE15_ROOT = "outputs/path_feedback_batch_checkpoint_publication_sandbox_consumer_replay_canary_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_default_policy_candidate_authorization_preflight_v1"

STAGE15_SUMMARY_FILE = "checkpoint-publication-sandbox-consumer-replay-canary-summary.json"
STAGE15_TELEMETRY_AUDIT_FILE = "checkpoint-publication-sandbox-consumer-telemetry-audit.json"
STAGE15_ROLLBACK_AUDIT_FILE = "checkpoint-publication-sandbox-consumer-rollback-audit.json"
STAGE15_RELEASE_AUDIT_FILE = "checkpoint-publication-sandbox-consumer-release-boundary-audit.json"

SUMMARY_FILE = "default-policy-candidate-authorization-preflight-summary.json"
MANIFEST_FILE = "default-policy-candidate-manifest.json"
KILL_SWITCH_AUDIT_FILE = "default-policy-candidate-kill-switch-audit.json"
ROLLBACK_AUDIT_FILE = "default-policy-candidate-rollback-audit.json"
DEFAULT_POLICY_AUDIT_FILE = "default-policy-candidate-default-policy-boundary-audit.json"
PATH_PLANNER_AUDIT_FILE = "default-policy-candidate-path-planner-isolation-audit.json"
RELEASE_AUDIT_FILE = "default-policy-candidate-release-boundary-audit.json"
AUTHORIZATION_MATRIX_FILE = "default-policy-candidate-authorization-matrix.json"
REJECTION_REPORT_FILE = "default-policy-candidate-rejection-report.json"
REPORT_FILE = "default-policy-candidate-authorization-preflight-report.md"

PASS_VERDICT = "eligible_for_default_policy_candidate_sandbox_install_preflight"
NEXT_REQUIRED_CHANGE = "default_policy_candidate_sandbox_install_preflight"
EXPECTED_STAGE15_VERDICT = "eligible_for_default_policy_candidate_authorization_preflight"

RELEASE_BOUNDARY_FIELDS = (
    "checkpoint_publication_approved",
    "default_policy_replacement_approved",
    "real_executor_connection_approved",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "real_world_release_approved",
    "real_world_performance_claimed",
    "final_release_approved",
    "relaxes_guard",
    "guard_relaxed",
    "modifies_network_or_action_space",
    "modifies_network",
    "modifies_action_space",
    "modifies_default_astar",
    "ackermann_feasible_trajectory_claimed",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run default policy candidate authorization preflight.")
    parser.add_argument("--stage15-root", default=DEFAULT_STAGE15_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    summary = run_default_policy_candidate_authorization_preflight(
        stage15_root=_resolve_path(Path(args.stage15_root), repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root),
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "authorization_verdict": summary["authorization_verdict"],
                "default_policy_candidate_authorization_preflight_passed": summary[
                    "default_policy_candidate_authorization_preflight_passed"
                ],
                "default_policy_candidate_sandbox_install_preflight_approved": summary[
                    "default_policy_candidate_sandbox_install_preflight_approved"
                ],
                "default_policy_replacement_approved": summary["default_policy_replacement_approved"],
                "real_executor_connection_approved": summary["real_executor_connection_approved"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_default_policy_candidate_authorization_preflight(
    *,
    stage15_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)
    read_reasons: list[str] = []
    stage15_summary = _read_json(stage15_root / STAGE15_SUMMARY_FILE, read_reasons, "stage15_summary")
    telemetry = _read_json(stage15_root / STAGE15_TELEMETRY_AUDIT_FILE, read_reasons, "stage15_telemetry_audit")
    rollback_source = _read_json(stage15_root / STAGE15_ROLLBACK_AUDIT_FILE, read_reasons, "stage15_rollback_audit")
    release_source = _read_json(stage15_root / STAGE15_RELEASE_AUDIT_FILE, read_reasons, "stage15_release_audit")

    stage15_gate = _stage15_gate(stage15_summary)
    evidence = _consumer_evidence_audit(stage15_summary, telemetry, rollback_source, release_source, read_reasons)
    kill_switch = _kill_switch_audit(stage15_summary)
    rollback = _rollback_audit(stage15_summary, rollback_source)
    default_policy = _default_policy_boundary_audit(stage15_summary)
    path_planner = _path_planner_isolation_audit(stage15_summary)
    release = _release_boundary_audit({"stage15_summary": stage15_summary, "stage15_release": release_source})
    docs = _docs_audit(repo_root)

    reason_codes: list[str] = []
    for audit in (stage15_gate, evidence, kill_switch, rollback, default_policy, path_planner, release):
        for reason in audit.get("reason_codes", []):
            _add_reason(reason_codes, reason)
    if docs.get("docs_audit_passed") is not True:
        _add_reason(reason_codes, "docs_not_updated")
    approved = not reason_codes

    matrix = {
        "schema_version": "default-policy-candidate-authorization-matrix/v1",
        "generated_at": _utc_now(),
        "stage15_gate_passed": stage15_gate["stage15_gate_passed"],
        "consumer_evidence_stable": evidence["consumer_evidence_stable"],
        "kill_switch_audit_passed": kill_switch["kill_switch_audit_passed"],
        "rollback_audit_passed": rollback["rollback_audit_passed"],
        "default_policy_boundary_audit_passed": default_policy["default_policy_boundary_audit_passed"],
        "path_planner_isolation_audit_passed": path_planner["path_planner_isolation_audit_passed"],
        "release_boundary_audit_passed": release["release_boundary_audit_passed"],
    }
    manifest = {
        "schema_version": "default-policy-candidate-manifest/v1",
        "generated_at": _utc_now(),
        "stage15_summary": str(stage15_root / STAGE15_SUMMARY_FILE),
        "sandbox_consumer_checkpoint_sha256": stage15_summary.get("sandbox_consumer_checkpoint_sha256"),
        "sandbox_consumer_checkpoint_size_bytes": stage15_summary.get("sandbox_consumer_checkpoint_size_bytes"),
        "next_required_change": NEXT_REQUIRED_CHANGE if approved else "resolve_default_policy_candidate_authorization_rejections",
        **_closed_boundaries(),
    }
    summary = {
        "schema_version": "default-policy-candidate-authorization-preflight-summary/v1",
        "generated_at": _utc_now(),
        "status": "passed" if approved else "failed",
        "reason_codes": reason_codes,
        "authorization_verdict": PASS_VERDICT if approved else "resolve_default_policy_candidate_authorization_rejections",
        "default_policy_candidate_authorization_preflight_passed": approved,
        "default_policy_candidate_sandbox_install_preflight_approved": approved,
        "kill_switch_audit_passed": kill_switch["kill_switch_audit_passed"],
        "rollback_audit_passed": rollback["rollback_audit_passed"],
        "default_policy_boundary_audit_passed": default_policy["default_policy_boundary_audit_passed"],
        "path_planner_isolation_audit_passed": path_planner["path_planner_isolation_audit_passed"],
        "release_boundary_audit_passed": release["release_boundary_audit_passed"],
        "next_required_change": NEXT_REQUIRED_CHANGE if approved else "resolve_default_policy_candidate_authorization_rejections",
        **_closed_boundaries(),
        "summary": str(paths["summary"]),
        "candidate_manifest": str(paths["manifest"]),
        "kill_switch_audit": str(paths["kill_switch"]),
        "rollback_audit": str(paths["rollback"]),
        "default_policy_boundary_audit": str(paths["default_policy"]),
        "path_planner_isolation_audit": str(paths["path_planner"]),
        "release_boundary_audit": str(paths["release"]),
        "authorization_matrix": str(paths["matrix"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }

    for path, payload in (
        (paths["manifest"], manifest),
        (paths["kill_switch"], kill_switch),
        (paths["rollback"], rollback),
        (paths["default_policy"], default_policy),
        (paths["path_planner"], path_planner),
        (paths["release"], release),
        (paths["matrix"], matrix),
        (paths["rejection_report"], _rejection_report(reason_codes)),
        (paths["summary"], summary),
    ):
        _write_json(path, payload)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _paths(root: Path) -> dict[str, Path]:
    return {
        "summary": root / SUMMARY_FILE,
        "manifest": root / MANIFEST_FILE,
        "kill_switch": root / KILL_SWITCH_AUDIT_FILE,
        "rollback": root / ROLLBACK_AUDIT_FILE,
        "default_policy": root / DEFAULT_POLICY_AUDIT_FILE,
        "path_planner": root / PATH_PLANNER_AUDIT_FILE,
        "release": root / RELEASE_AUDIT_FILE,
        "matrix": root / AUTHORIZATION_MATRIX_FILE,
        "rejection_report": root / REJECTION_REPORT_FILE,
        "report": root / REPORT_FILE,
    }


def _stage15_gate(summary: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    if summary.get("status") != "passed" or summary.get("reason_codes") not in ([], None):
        reasons.append("stage15_not_passed")
    if (
        summary.get("consumer_replay_canary_verdict") != EXPECTED_STAGE15_VERDICT
        or summary.get("checkpoint_publication_sandbox_consumer_replay_canary_passed") is not True
        or summary.get("default_policy_candidate_authorization_preflight_approved") is not True
        or summary.get("next_required_change") != "default_policy_candidate_authorization_preflight"
    ):
        reasons.append("consumer_evidence_unstable")
    return {"stage15_gate_passed": not reasons, "reason_codes": reasons}


def _consumer_evidence_audit(summary: dict[str, Any], telemetry: dict[str, Any], rollback: dict[str, Any], release: dict[str, Any], read_reasons: list[str]) -> dict[str, Any]:
    stable = (
        not read_reasons
        and int(summary.get("consumer_step_count", 0)) >= 64
        and float(summary.get("fallback_rate", 1.0)) < 0.5
        and int(summary.get("controlled_regression_count", 1)) == 0
        and summary.get("telemetry_audit_passed") is True
        and summary.get("rollback_audit_passed") is True
        and summary.get("release_boundary_audit_passed") is True
        and telemetry.get("telemetry_audit_passed") is True
        and rollback.get("rollback_audit_passed") is True
        and release.get("release_boundary_audit_passed") is True
    )
    return {"consumer_evidence_stable": stable, "reason_codes": [] if stable else ["consumer_evidence_unstable"]}


def _kill_switch_audit(summary: dict[str, Any]) -> dict[str, Any]:
    passed = summary.get("default_policy_candidate_authorization_preflight_approved") is True
    return {"kill_switch_audit_passed": passed, "reason_codes": [] if passed else ["kill_switch_missing"]}


def _rollback_audit(summary: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    passed = summary.get("rollback_audit_passed") is True and source.get("rollback_audit_passed") is True
    return {"rollback_audit_passed": passed, "reason_codes": [] if passed else ["rollback_boundary_invalid"]}


def _default_policy_boundary_audit(summary: dict[str, Any]) -> dict[str, Any]:
    passed = summary.get("default_policy_replacement_approved") is not True and summary.get("replaces_default_policy") is not True
    return {"default_policy_boundary_audit_passed": passed, "reason_codes": [] if passed else ["default_policy_boundary_violation"]}


def _path_planner_isolation_audit(summary: dict[str, Any]) -> dict[str, Any]:
    passed = summary.get("real_executor_connection_approved") is not True and summary.get("connects_real_executor") is not True
    return {"path_planner_isolation_audit_passed": passed, "reason_codes": [] if passed else ["path_planner_isolation_violation"]}


def _release_boundary_audit(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    violations = []
    for name, payload in payloads.items():
        for field in RELEASE_BOUNDARY_FIELDS:
            if payload.get(field) is True:
                violations.append(f"{name}.{field}")
    return {"release_boundary_audit_passed": not violations, "violations": violations, "reason_codes": [] if not violations else ["release_boundary_violation"], **_closed_boundaries()}


def _docs_audit(repo_root: Path) -> dict[str, Any]:
    docs = [
        repo_root / "README.md",
        repo_root / "docs" / "算法设计与系统架构报告.md",
        repo_root / "docs" / "superpowers" / "specs" / "2026-06-16-default-policy-candidate-authorization-preflight.md",
    ]
    missing = [str(path) for path in docs if not path.is_file()]
    return {"docs_audit_passed": not missing, "missing_docs": missing}


def _closed_boundaries() -> dict[str, bool]:
    return {
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
    }


def _read_json(path: Path, reasons: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        reasons.append(f"{label}_missing")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        reasons.append(f"{label}_invalid")
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rejection_report(reason_codes: list[str]) -> dict[str, Any]:
    return {"status": "passed" if not reason_codes else "failed", "reason_codes": list(reason_codes), "rejected": bool(reason_codes)}


def _render_report(summary: dict[str, Any]) -> str:
    return (
        "# Default Policy Candidate Authorization Preflight v1\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Reason codes: `{summary['reason_codes']}`\n"
        f"- Verdict: `{summary['authorization_verdict']}`\n"
    )


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
