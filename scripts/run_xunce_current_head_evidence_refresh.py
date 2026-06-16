from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
    from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json
    from global_99_governance_common import global_99_boundary_defaults
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json
    from scripts.global_99_governance_common import global_99_boundary_defaults


CONFIG_SCHEMA_VERSION = "xunce-current-head-evidence-refresh-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-current-head-evidence-refresh-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-current-head-evidence-refresh-manifest/v1"
SOURCE_STATUS_AUDIT_SCHEMA_VERSION = "xunce-current-head-source-status-audit/v1"
GIT_PROVENANCE_AUDIT_SCHEMA_VERSION = "xunce-current-head-git-provenance-audit/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "xunce-current-head-boundary-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "xunce-current-head-evidence-refresh-rejection-report/v1"

DEFAULT_CONFIG = "configs/xunce_current_head_evidence_refresh_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_current_head_evidence_refresh_v1"

SUMMARY_FILE = "xunce-current-head-evidence-refresh-summary.json"
MANIFEST_FILE = "xunce-current-head-evidence-refresh-manifest.json"
SOURCE_STATUS_AUDIT_FILE = "xunce-current-head-source-status-audit.json"
GIT_PROVENANCE_AUDIT_FILE = "xunce-current-head-git-provenance-audit.json"
BOUNDARY_AUDIT_FILE = "xunce-current-head-boundary-audit.json"
REJECTION_REPORT_FILE = "xunce-current-head-evidence-refresh-rejection-report.json"
REPORT_FILE = "xunce-current-head-evidence-refresh-report.md"

PASS_NEXT_REQUIRED_CHANGE = "network_literature_bottleneck_review"
FIX_STAGE0_NEXT_REQUIRED_CHANGE = "fix_xunce_design_freeze"
FIX_REFRESH_NEXT_REQUIRED_CHANGE = "fix_current_head_evidence_refresh"

SOURCE_DEFINITIONS = {
    "xunce_design_freeze": {
        "root_key": "source_xunce_design_freeze_root",
        "file": "xunce-design-freeze-summary.json",
    },
    "controlled_installation": {
        "root_key": "source_controlled_installation_root",
        "file": "global-99-controlled-default-policy-candidate-installation-preflight-summary.json",
    },
    "real_map_shadow_replay": {
        "root_key": "source_real_map_shadow_replay_root",
        "file": "global-99-real-map-shadow-replay-summary.json",
    },
    "real_map_multi_roi": {
        "root_key": "source_real_map_multi_roi_root",
        "file": "global-99-real-map-multi-roi-generalization-summary.json",
    },
    "network_readiness": {
        "root_key": "source_network_readiness_root",
        "file": "network-architecture-upgrade-readiness-summary.json",
    },
}

BOUNDARY_FIELDS = tuple(global_99_boundary_defaults()) + (
    "real_world_release_approved",
    "real_world_performance_claimed",
    "default_policy_replacement_approved",
    "real_executor_connection_approved",
    "starts_online_canary",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Xunce Current-HEAD Evidence Refresh v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_current_head_evidence_refresh(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "current_head_evidence_refresh_passed": summary["current_head_evidence_refresh_passed"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_current_head_evidence_refresh(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    current_git: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = _load_config(Path(config_path), repo_root)
    current_git = current_git or git_snapshot(repo_root)
    source_paths = _source_paths(config, repo_root)
    sources = _load_sources(source_paths)
    paths = _artifact_paths(output_root)
    source_status_audit = _source_status_audit(config, sources)
    git_provenance_audit = _git_provenance_audit(config, sources, current_git)
    boundary_audit = _boundary_audit(config, sources)
    decision = _decision(source_status_audit, git_provenance_audit, boundary_audit)
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=Path(config_path),
        output_root=output_root,
        paths=paths,
        source_paths=source_paths,
        current_git=current_git,
        source_status_audit=source_status_audit,
        git_provenance_audit=git_provenance_audit,
        boundary_audit=boundary_audit,
        decision=decision,
    )
    manifest = _manifest(generated_at, Path(config_path), output_root, paths, summary)
    rejection_report = _rejection_report(generated_at, decision, source_status_audit, git_provenance_audit, boundary_audit)

    write_json(paths["source_status_audit"], source_status_audit)
    write_json(paths["git_provenance_audit"], git_provenance_audit)
    write_json(paths["boundary_audit"], boundary_audit)
    write_json(paths["rejection_report"], rejection_report)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, rejection_report), encoding="utf-8")
    return summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    normalized = dict(payload)
    for spec in SOURCE_DEFINITIONS.values():
        key = spec["root_key"]
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(payload[key]), repo_root))
    for key in (
        "require_current_worktree_clean",
        "require_stage0_passed",
        "allow_legacy_missing_git_provenance",
        "require_closed_boundaries",
    ):
        if not isinstance(payload.get(key), bool):
            raise ConfigError(f"{key} must be a boolean")
    return normalized


def _source_paths(config: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    paths = {}
    for label, spec in SOURCE_DEFINITIONS.items():
        root = resolve_path(Path(config[spec["root_key"]]), repo_root)
        paths[label] = root / spec["file"]
    return paths


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "source_status_audit": output_root / SOURCE_STATUS_AUDIT_FILE,
        "git_provenance_audit": output_root / GIT_PROVENANCE_AUDIT_FILE,
        "boundary_audit": output_root / BOUNDARY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _load_sources(source_paths: dict[str, Path]) -> dict[str, dict[str, Any] | None]:
    return {label: _load_json(path) for label, path in source_paths.items()}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _source_status_audit(config: dict[str, Any], sources: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    rows = []
    reasons: list[str] = []
    for label, payload in sources.items():
        if payload is None:
            rows.append({"source": label, "exists": False, "status": None, "next_required_change": None})
            reasons.append(f"missing_{label}_summary")
            continue
        status = payload.get("status")
        next_required_change = payload.get("next_required_change")
        rows.append(
            {
                "source": label,
                "exists": True,
                "status": status,
                "next_required_change": next_required_change,
            }
        )
        if status != "passed":
            reasons.append(f"{label}_not_passed")
    stage0 = sources.get("xunce_design_freeze") or {}
    if config["require_stage0_passed"]:
        if stage0.get("status") != "passed":
            reasons.append("stage0_not_passed")
        if stage0.get("next_required_change") != "current_head_evidence_refresh":
            reasons.append("stage0_next_required_change_invalid")
    return {
        "schema_version": SOURCE_STATUS_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not reasons else "failed",
        "reason_codes": unique_sorted(reasons),
        "sources": rows,
        "source_count": len(rows),
        "source_status_passed_count": sum(1 for row in rows if row["status"] == "passed"),
    }


def _git_provenance_audit(
    config: dict[str, Any],
    sources: dict[str, dict[str, Any] | None],
    current_git: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    reasons: list[str] = []
    current_dirty = _snapshot_dirty(current_git)
    if config["require_current_worktree_clean"] and current_dirty:
        reasons.append("current_worktree_dirty")
    for label, payload in sources.items():
        source_current = _source_current_git(payload)
        if not source_current:
            rows.append({"source": label, "has_git_provenance": False, "matches_current": None, "source_dirty": None})
            if not config["allow_legacy_missing_git_provenance"]:
                reasons.append("source_git_provenance_missing")
            continue
        source_dirty = _snapshot_dirty(source_current)
        matches = _snapshots_match_current(source_current, current_git)
        rows.append(
            {
                "source": label,
                "has_git_provenance": True,
                "source_sha": _parent_sha(source_current),
                "current_sha": _parent_sha(current_git),
                "matches_current": matches,
                "source_dirty": source_dirty,
            }
        )
        if source_dirty:
            reasons.append("source_git_provenance_dirty")
        if not matches:
            reasons.append("source_git_provenance_mismatch")
    return {
        "schema_version": GIT_PROVENANCE_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not reasons else "failed",
        "reason_codes": unique_sorted(reasons),
        "current_git_dirty": current_dirty,
        "current_git_parent_sha": _parent_sha(current_git),
        "sources": rows,
        "source_current_git_match_count": sum(1 for row in rows if row["matches_current"] is True),
        "legacy_missing_git_provenance_count": sum(1 for row in rows if row["has_git_provenance"] is False),
        "source_dirty_count": sum(1 for row in rows if row["source_dirty"] is True),
    }


def _boundary_audit(config: dict[str, Any], sources: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    violations = []
    if config["require_closed_boundaries"]:
        for label, payload in sources.items():
            if not isinstance(payload, dict):
                continue
            for field in BOUNDARY_FIELDS:
                if payload.get(field) is True:
                    violations.append({"source": label, "field": field, "value": True})
    return {
        "schema_version": BOUNDARY_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not violations else "failed",
        "reason_codes": [] if not violations else ["source_boundary_violation"],
        "boundary_fields": list(dict.fromkeys(BOUNDARY_FIELDS)),
        "violations": violations,
    }


def _decision(
    source_status_audit: dict[str, Any],
    git_provenance_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
) -> dict[str, Any]:
    reasons = unique_sorted(
        list(source_status_audit.get("reason_codes", []))
        + list(git_provenance_audit.get("reason_codes", []))
        + list(boundary_audit.get("reason_codes", []))
    )
    status = "passed" if not reasons else "failed"
    if status == "passed":
        next_required_change = PASS_NEXT_REQUIRED_CHANGE
    elif "stage0_not_passed" in reasons or "stage0_next_required_change_invalid" in reasons or "missing_xunce_design_freeze_summary" in reasons:
        next_required_change = FIX_STAGE0_NEXT_REQUIRED_CHANGE
    else:
        next_required_change = FIX_REFRESH_NEXT_REQUIRED_CHANGE
    return {"status": status, "reason_codes": reasons, "next_required_change": next_required_change}


def _summary(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    source_paths: dict[str, Path],
    current_git: dict[str, Any],
    source_status_audit: dict[str, Any],
    git_provenance_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "summary": str(paths["summary"]),
        "config_path": str(config_path),
        "output_root": str(output_root),
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "current_head_evidence_refresh_passed": decision["status"] == "passed",
        "source_status_audit_passed": source_status_audit["status"] == "passed",
        "git_provenance_audit_passed": git_provenance_audit["status"] == "passed",
        "boundary_audit_passed": boundary_audit["status"] == "passed",
        "source_count": source_status_audit["source_count"],
        "source_status_passed_count": source_status_audit["source_status_passed_count"],
        "source_current_git_match_count": git_provenance_audit["source_current_git_match_count"],
        "legacy_missing_git_provenance_count": git_provenance_audit["legacy_missing_git_provenance_count"],
        "source_dirty_count": git_provenance_audit["source_dirty_count"],
        "current_git_dirty": git_provenance_audit["current_git_dirty"],
        "current_git_parent_sha": git_provenance_audit["current_git_parent_sha"],
        "source_paths": {key: str(value) for key, value in source_paths.items()},
        "next_required_change": decision["next_required_change"],
        **global_99_boundary_defaults(),
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "git_provenance": {"current": current_git},
    }


def _manifest(generated_at: str, config_path: Path, output_root: Path, paths: dict[str, Path], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config_path": str(config_path),
        "output_root": str(output_root),
        "status": summary["status"],
        "next_required_change": summary["next_required_change"],
        "artifacts": {key: str(value) for key, value in paths.items()},
    }


def _rejection_report(
    generated_at: str,
    decision: dict[str, Any],
    source_status_audit: dict[str, Any],
    git_provenance_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "passed" if decision["status"] == "passed" else "failed",
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
        "source_status_rejections": source_status_audit.get("reason_codes", []),
        "git_provenance_rejections": git_provenance_audit.get("reason_codes", []),
        "boundary_rejections": boundary_audit.get("violations", []),
    }


def _render_report(summary: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Current-HEAD Evidence Refresh v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- current_git_parent_sha: `{summary['current_git_parent_sha']}`",
            f"- current_git_dirty: `{summary['current_git_dirty']}`",
            f"- source_current_git_match_count: `{summary['source_current_git_match_count']}`",
            f"- legacy_missing_git_provenance_count: `{summary['legacy_missing_git_provenance_count']}`",
            "",
            "## Rejections",
            "",
            json.dumps(rejection_report, ensure_ascii=False, indent=2),
            "",
        ]
    )


def _source_current_git(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    git = payload.get("git_provenance")
    if not isinstance(git, dict):
        return None
    current = git.get("current")
    return current if isinstance(current, dict) else None


def _snapshot_dirty(snapshot: dict[str, Any]) -> bool:
    if snapshot.get("dirty") is True:
        return True
    parent = snapshot.get("parent")
    if isinstance(parent, dict) and parent.get("dirty") is True:
        return True
    modules = snapshot.get("submodules")
    if isinstance(modules, dict):
        return any(isinstance(item, dict) and item.get("dirty") is True for item in modules.values())
    return False


def _snapshots_match_current(source: dict[str, Any], current: dict[str, Any]) -> bool:
    if _parent_sha(source) != _parent_sha(current):
        return False
    source_modules = source.get("submodules") if isinstance(source.get("submodules"), dict) else {}
    current_modules = current.get("submodules") if isinstance(current.get("submodules"), dict) else {}
    for name, source_state in source_modules.items():
        if not isinstance(source_state, dict):
            continue
        current_state = current_modules.get(name)
        if isinstance(current_state, dict) and source_state.get("sha") != current_state.get("sha"):
            return False
    return True


def _parent_sha(snapshot: dict[str, Any]) -> str | None:
    parent = snapshot.get("parent")
    if not isinstance(parent, dict):
        return None
    sha = parent.get("sha")
    return str(sha) if sha is not None else None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
