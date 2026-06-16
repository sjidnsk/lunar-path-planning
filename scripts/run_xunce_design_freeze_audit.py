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


CONFIG_SCHEMA_VERSION = "xunce-design-freeze-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-design-freeze-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-design-freeze-manifest/v1"
DOCUMENT_AUDIT_SCHEMA_VERSION = "xunce-design-document-audit/v1"
EVIDENCE_AUDIT_SCHEMA_VERSION = "xunce-design-evidence-audit/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "xunce-design-boundary-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "xunce-design-freeze-rejection-report/v1"

DEFAULT_CONFIG = "configs/xunce_design_freeze_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_design_freeze_v1"

CONTROLLED_SUMMARY_FILE = "global-99-controlled-default-policy-candidate-installation-preflight-summary.json"
NETWORK_READINESS_SUMMARY_FILE = "network-architecture-upgrade-readiness-summary.json"
REAL_MAP_MULTI_ROI_SUMMARY_FILE = "global-99-real-map-multi-roi-generalization-summary.json"

SUMMARY_FILE = "xunce-design-freeze-summary.json"
MANIFEST_FILE = "xunce-design-freeze-manifest.json"
DOCUMENT_AUDIT_FILE = "xunce-design-document-audit.json"
EVIDENCE_AUDIT_FILE = "xunce-design-evidence-audit.json"
BOUNDARY_AUDIT_FILE = "xunce-design-boundary-audit.json"
REJECTION_REPORT_FILE = "xunce-design-freeze-rejection-report.json"
REPORT_FILE = "xunce-design-freeze-report.md"

PASS_NEXT_REQUIRED_CHANGE = "current_head_evidence_refresh"
FIX_DESIGN_NEXT_REQUIRED_CHANGE = "fix_xunce_design_freeze"
FIX_EVIDENCE_NEXT_REQUIRED_CHANGE = "fix_global_99_evidence_chain"

REQUIRED_STAGE_MARKERS = (
    "0 冻结巡策设计",
    "1 Current-HEAD Evidence Refresh",
    "2 文献与项目瓶颈审计",
    "3 Topology Observation Contract",
    "4 Topology Feature Extraction Audit",
    "5 小型巡策原型",
    "6 原型机制验证",
    "7 架构对比评测",
    "8 完整巡策网络 v1",
    "9 完整网络静态合同验证",
    "10 完整网络消融实验",
    "11 完整网络压力评测",
    "12 Guarded Training Candidate Preflight",
    "13 受控训练候选",
    "14 训练后离线评测",
    "15 Shadow / Replay 验证",
    "16 Sandbox Candidate Preflight",
    "17 发布治理门禁",
)

REQUIRED_TERMS = (
    "巡策",
    "Topology-Aware Coverage Policy Network Design",
)

PLAN_FIRST_MARKERS = (
    "每一个阶段都必须先设计详细计划",
    "计划先行",
    "plan-first",
)

NON_GOAL_MARKERS = (
    "不连接真实 executor",
    "不启动 online canary",
    "不直接替换 default policy",
    "不发布 checkpoint",
    "不修改 action space/default A*",
)

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
    parser = argparse.ArgumentParser(description="Run Xunce Design Freeze Audit v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_design_freeze_audit(
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
                "design_freeze_passed": summary["design_freeze_passed"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_design_freeze_audit(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config_path = Path(config_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = _load_config(config_path, repo_root)
    paths = _artifact_paths(output_root)
    source_paths = _source_paths(config, repo_root)
    sources = _load_sources(source_paths)
    document_audit = _document_audit(config, repo_root)
    evidence_audit = _evidence_audit(sources)
    boundary_audit = _boundary_audit(sources)
    decision = _decision(document_audit, evidence_audit, boundary_audit)
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=config_path,
        output_root=output_root,
        paths=paths,
        source_paths=source_paths,
        document_audit=document_audit,
        evidence_audit=evidence_audit,
        boundary_audit=boundary_audit,
        decision=decision,
        repo_root=repo_root,
    )
    manifest = _manifest(generated_at, config_path, output_root, paths, summary)
    rejection_report = _rejection_report(generated_at, decision, document_audit, evidence_audit, boundary_audit)

    write_json(paths["document_audit"], document_audit)
    write_json(paths["evidence_audit"], evidence_audit)
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
    documents = payload.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ConfigError("documents must be a non-empty list")
    normalized = dict(payload)
    normalized["documents"] = [str(item) for item in documents]
    for key in ("source_controlled_installation_root", "source_network_readiness_root", "source_real_map_multi_roi_root"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(payload[key]), repo_root))
    for key in ("require_complete_stage_chain", "require_plan_first_rule", "require_closed_boundaries"):
        if not isinstance(payload.get(key), bool):
            raise ConfigError(f"{key} must be a boolean")
    return normalized


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "document_audit": output_root / DOCUMENT_AUDIT_FILE,
        "evidence_audit": output_root / EVIDENCE_AUDIT_FILE,
        "boundary_audit": output_root / BOUNDARY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _source_paths(config: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    controlled_root = resolve_path(Path(config["source_controlled_installation_root"]), repo_root)
    network_root = resolve_path(Path(config["source_network_readiness_root"]), repo_root)
    real_map_root = resolve_path(Path(config["source_real_map_multi_roi_root"]), repo_root)
    return {
        "controlled_installation_summary": controlled_root / CONTROLLED_SUMMARY_FILE,
        "network_readiness_summary": network_root / NETWORK_READINESS_SUMMARY_FILE,
        "real_map_multi_roi_summary": real_map_root / REAL_MAP_MULTI_ROI_SUMMARY_FILE,
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


def _document_audit(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    documents = []
    all_text_parts = []
    missing_documents: list[str] = []
    design_spec_text = ""
    design_spec_path = None
    for rel in config["documents"]:
        path = resolve_path(Path(rel), repo_root)
        exists = path.is_file()
        text = path.read_text(encoding="utf-8") if exists else ""
        if "topology-aware-coverage-policy-network-design" in rel:
            design_spec_text = text
            design_spec_path = rel
        if not exists:
            missing_documents.append(rel)
        documents.append(
            {
                "path": rel,
                "exists": exists,
                "contains_xunce": "巡策" in text,
                "contains_topology_aware_name": "Topology-Aware Coverage Policy Network Design" in text,
            }
        )
        all_text_parts.append(text)
    all_text = "\n".join(all_text_parts)
    missing_terms = [term for term in REQUIRED_TERMS if term not in all_text]
    missing_stages = [stage for stage in REQUIRED_STAGE_MARKERS if stage not in design_spec_text]
    plan_first_declared = any(marker in all_text for marker in PLAN_FIRST_MARKERS)
    missing_non_goals = [marker for marker in NON_GOAL_MARKERS if marker not in all_text]
    reason_codes: list[str] = []
    if missing_documents:
        reason_codes.append("xunce_design_documents_missing")
    if missing_terms:
        reason_codes.append("xunce_required_terms_missing")
    if config["require_complete_stage_chain"] and missing_stages:
        reason_codes.append("xunce_stage_chain_incomplete")
    if config["require_plan_first_rule"] and not plan_first_declared:
        reason_codes.append("xunce_plan_first_rule_missing")
    if missing_non_goals:
        reason_codes.append("xunce_non_goals_incomplete")
    passed = not reason_codes
    return {
        "schema_version": DOCUMENT_AUDIT_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "reason_codes": reason_codes,
        "document_count": len(documents),
        "documents": documents,
        "design_spec_path": design_spec_path,
        "missing_documents": missing_documents,
        "missing_terms": missing_terms,
        "complete_stage_chain_count": len(REQUIRED_STAGE_MARKERS) - len(missing_stages),
        "missing_stage_markers": missing_stages,
        "plan_first_rule_declared": plan_first_declared,
        "missing_non_goal_markers": missing_non_goals,
    }


def _evidence_audit(sources: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    reason_codes: list[str] = []
    source_statuses: dict[str, str | None] = {}
    source_next: dict[str, str | None] = {}
    for label, payload in sources.items():
        if payload is None:
            reason_codes.append(f"missing_{label}")
            source_statuses[label] = None
            source_next[label] = None
            continue
        source_statuses[label] = _string_or_none(payload.get("status"))
        source_next[label] = _string_or_none(payload.get("next_required_change"))
        if payload.get("status") != "passed":
            reason_codes.append(f"{label}_not_passed")
    controlled = sources.get("controlled_installation_summary") or {}
    if controlled and controlled.get("next_required_change") != "eligible_for_controlled_default_policy_candidate_installation_review":
        reason_codes.append("controlled_installation_next_required_change_unexpected")
    return {
        "schema_version": EVIDENCE_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": unique_sorted(reason_codes),
        "source_statuses": source_statuses,
        "source_next_required_changes": source_next,
    }


def _boundary_audit(sources: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
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
    document_audit: dict[str, Any],
    evidence_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
) -> dict[str, Any]:
    reason_codes = []
    for audit in (document_audit, evidence_audit, boundary_audit):
        reason_codes.extend(audit.get("reason_codes", []))
    reason_codes = unique_sorted(reason_codes)
    status = "passed" if not reason_codes else "failed"
    if status == "passed":
        next_required_change = PASS_NEXT_REQUIRED_CHANGE
    elif evidence_audit.get("status") != "passed" or boundary_audit.get("status") != "passed":
        next_required_change = FIX_EVIDENCE_NEXT_REQUIRED_CHANGE
    else:
        next_required_change = FIX_DESIGN_NEXT_REQUIRED_CHANGE
    return {
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": next_required_change,
    }


def _summary(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    source_paths: dict[str, Path],
    document_audit: dict[str, Any],
    evidence_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    controlled_status = evidence_audit["source_statuses"].get("controlled_installation_summary")
    controlled_next = evidence_audit["source_next_required_changes"].get("controlled_installation_summary")
    network_status = evidence_audit["source_statuses"].get("network_readiness_summary")
    real_map_status = evidence_audit["source_statuses"].get("real_map_multi_roi_summary")
    boundary_defaults = global_99_boundary_defaults()
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "summary": str(paths["summary"]),
        "config_path": str(config_path),
        "output_root": str(output_root),
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "design_freeze_passed": decision["status"] == "passed",
        "document_audit_passed": document_audit["status"] == "passed",
        "evidence_audit_passed": evidence_audit["status"] == "passed",
        "boundary_audit_passed": boundary_audit["status"] == "passed",
        "document_count": document_audit["document_count"],
        "complete_stage_chain_count": document_audit["complete_stage_chain_count"],
        "plan_first_rule_declared": document_audit["plan_first_rule_declared"],
        "source_controlled_installation_status": controlled_status,
        "source_controlled_installation_next_required_change": controlled_next,
        "source_network_readiness_status": network_status,
        "source_real_map_multi_roi_status": real_map_status,
        "source_paths": {key: str(value) for key, value in source_paths.items()},
        "next_required_change": decision["next_required_change"],
        **boundary_defaults,
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "git_provenance": {"current": git_snapshot(repo_root)},
    }


def _manifest(
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    summary: dict[str, Any],
) -> dict[str, Any]:
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
    document_audit: dict[str, Any],
    evidence_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "passed" if decision["status"] == "passed" else "failed",
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
        "document_rejections": document_audit.get("reason_codes", []),
        "evidence_rejections": evidence_audit.get("reason_codes", []),
        "boundary_rejections": boundary_audit.get("violations", []),
    }


def _render_report(summary: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Design Freeze Audit v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- document_count: `{summary['document_count']}`",
            f"- complete_stage_chain_count: `{summary['complete_stage_chain_count']}`",
            f"- source_controlled_installation_status: `{summary['source_controlled_installation_status']}`",
            f"- boundary_audit_passed: `{summary['boundary_audit_passed']}`",
            "",
            "## Rejections",
            "",
            json.dumps(rejection_report, ensure_ascii=False, indent=2),
            "",
        ]
    )


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
