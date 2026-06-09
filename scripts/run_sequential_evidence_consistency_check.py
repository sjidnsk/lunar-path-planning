from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot

try:
    from scripts.run_scenario_disjoint_policy_rollout_evaluation import _display_path
except ModuleNotFoundError:
    from run_scenario_disjoint_policy_rollout_evaluation import _display_path


SUMMARY_SCHEMA_VERSION = "sequential-evidence-consistency-summary/v1"
CONFIG_SCHEMA_VERSION = "sequential-evidence-consistency-config/v1"
NEXT_REQUIRED_CHANGE = "sequential_evidence_consistency_required"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check sequential canary evidence consistency.")
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--readiness-summary", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    batch_root = _resolve_path(args.batch_root, repo_root)
    readiness_path = _resolve_path(args.readiness_summary, repo_root)
    config_path = _resolve_path(args.config, repo_root)
    config = _load_json(config_path)
    summary = run_sequential_evidence_consistency_check(
        batch_root=batch_root,
        readiness_summary_path=readiness_path,
        config=config,
        repo_root=repo_root,
    )
    output_path = (
        _resolve_path(args.output, repo_root)
        if args.output
        else batch_root / "sequential-evidence-consistency-summary.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "summary": _display_path(output_path, repo_root),
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_sequential_evidence_consistency_check(
    *,
    batch_root: Path,
    readiness_summary_path: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    rollout_path = batch_root / config.get("input_files", {}).get(
        "rollout_summary", "policy-gated-sequential-canary-rollout-summary.json"
    )
    diagnosis_path = batch_root / config.get("input_files", {}).get(
        "diagnosis_summary", "sequential-multi-step-opportunity-diagnosis-summary.json"
    )
    rollout = _load_json_if_exists(rollout_path)
    diagnosis = _load_json_if_exists(diagnosis_path)
    readiness = _load_json_if_exists(readiness_summary_path)

    if not rollout:
        _append_reason(reason_codes, "sequential_rollout_summary_missing")
    elif rollout.get("status") != "passed" or rollout.get("reason_codes"):
        _append_reason(reason_codes, "sequential_rollout_not_passed")

    required_status = config.get("validation", {}).get(
        "require_readiness_status", "policy_gated_sequential_multi_step_opportunity_evaluated"
    )
    if not readiness:
        _append_reason(reason_codes, "readiness_summary_missing")
    elif readiness.get("status") != "passed" or readiness.get("training_readiness_status") != required_status:
        _append_reason(reason_codes, "readiness_status_mismatch")
    elif readiness.get("training_blockers"):
        _append_reason(reason_codes, "readiness_training_blockers_present")

    diagnosis_role = str(config.get("diagnosis_role", "final_gate"))
    diagnosis_preflight_only = diagnosis_role == "preflight_only" or bool(
        diagnosis.get("preflight_only") if diagnosis else False
    )
    diagnosis_conflict = bool(
        rollout
        and rollout.get("status") == "passed"
        and diagnosis
        and diagnosis.get("status") != "passed"
        and not diagnosis_preflight_only
    )
    if diagnosis_conflict:
        _append_reason(reason_codes, "sequential_diagnosis_final_gate_failed")
    if config.get("validation", {}).get("require_diagnosis_summary", True) and not diagnosis:
        _append_reason(reason_codes, "sequential_diagnosis_summary_missing")

    current_git = _git_snapshot(repo_root)
    status = "failed" if reason_codes else "passed"
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason_codes": reason_codes,
        "batch_root": _display_path(batch_root, repo_root),
        "rollout_summary": {
            "path": _display_path(rollout_path, repo_root),
            "exists": rollout_path.is_file(),
            "status": rollout.get("status"),
        },
        "diagnosis_summary": {
            "path": _display_path(diagnosis_path, repo_root),
            "exists": diagnosis_path.is_file(),
            "status": diagnosis.get("status"),
            "diagnosis_role": diagnosis_role,
            "preflight_only": diagnosis_preflight_only,
        },
        "readiness_summary": {
            "path": _display_path(readiness_summary_path, repo_root),
            "exists": readiness_summary_path.is_file(),
            "status": readiness.get("status"),
            "training_readiness_status": readiness.get("training_readiness_status"),
        },
        "next_required_change": None if status == "passed" else NEXT_REQUIRED_CHANGE,
        "git_provenance": {"current": current_git, "current_matches_sources": True},
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


if __name__ == "__main__":
    raise SystemExit(main())
