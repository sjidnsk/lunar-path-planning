from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STAGE_ID = "xunce-stage26-5b-documentation-boundary-consolidation"
CONFIG_SCHEMA_VERSION = "xunce-stage26-5b-documentation-boundary-consolidation-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-5b-summary/v1"
AUDIT_SCHEMA_VERSION = "xunce-stage26-5b-doc-boundary-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-5b-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-5b-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_5b_documentation_boundary_consolidation_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_5b_documentation_boundary_consolidation_v1"
)

SUMMARY_FILE = "xunce-stage26-5b-summary.json"
AUDIT_FILE = "xunce-stage26-5b-doc-boundary-audit.json"
ROUTING_FILE = "xunce-stage26-5b-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-5b-report.md"
MANIFEST_FILE = "xunce-stage26-5b-manifest.json"

ROUTE_AGENTS = "repair_agents_md_boundary"
ROUTE_README = "repair_readme_documentation_map"
ROUTE_LONGFORM = "repair_longform_docs_boundary"
ROUTE_REGISTRY = "repair_stage_registry_integrity"
ROUTE_NEXT = "continue_stage26_6_synthetic_exploration_credit_assignment"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Stage26.5B documentation boundary consolidation.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    summary = run_xunce_stage26_5b_documentation_boundary_consolidation(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_5b_documentation_boundary_consolidation(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(config_path, repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    audit = _build_audit(repo_root, config)
    status, route, reasons = _route(audit)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "blocking_reason_codes": reasons,
        "agents_md_line_count": audit["agents_md"]["line_count"],
        "agents_stage_section_count": audit["agents_md"]["stage_section_count"],
        "readme_has_documentation_map": audit["readme"]["has_documentation_map"],
        "documentation_index_exists": audit["documentation_index"]["exists"],
        "longform_docs_boundary_passed": audit["longform_docs"]["passed"],
        "stage_registry_integrity_passed": audit["stage_registry"]["passed"],
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "blocking_reason_codes": reasons,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "summary": str(output_root / SUMMARY_FILE),
        "doc_boundary_audit": str(output_root / AUDIT_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }

    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / AUDIT_FILE, audit)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    _write_report(output_root / REPORT_FILE, summary, audit)
    return summary


def _build_audit(repo_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    agents = _read_text(repo_root / "AGENTS.md")
    readme = _read_text(repo_root / "README.md")
    index = _read_text(repo_root / "docs/xunce-stage-documentation-index.md")
    architecture = _read_text(repo_root / "docs/算法设计与系统架构报告.md")
    spec = _read_text(repo_root / "docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md")
    registry_ok, registry_error, stages = _load_registry(repo_root / "configs/stage_registry.json")

    required_contracts = list(config["required_contract_strings"])
    required_index_terms = list(config["required_documentation_index_terms"])
    agents_stage_sections = sum(1 for line in agents.splitlines() if line.startswith("## Stage"))

    agents_missing = [term for term in required_contracts if term not in agents]
    index_missing = [term for term in required_index_terms if term not in index]
    readme_missing = [
        term
        for term in ("## Documentation Map", "docs/xunce-stage-documentation-index.md", "Stage26.5", ROUTE_NEXT.replace("continue_stage26_6_", "repair_stage26_"))
        if term not in readme
    ]
    longform_missing = [
        term
        for term in ("docs/xunce-stage-documentation-index.md", "coverage_source=endpoint_theta_slope_obstacle_los/v1")
        if term not in architecture or term not in spec
    ]

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "agents_md": {
            "line_count": len(agents.splitlines()),
            "max_allowed_line_count": int(config["agents_md_max_lines"]),
            "stage_section_count": agents_stage_sections,
            "max_allowed_stage_section_count": int(config["agents_md_max_stage_sections"]),
            "missing_required_contracts": agents_missing,
            "passed": (
                len(agents.splitlines()) <= int(config["agents_md_max_lines"])
                and agents_stage_sections <= int(config["agents_md_max_stage_sections"])
                and not agents_missing
            ),
        },
        "readme": {
            "has_documentation_map": "## Documentation Map" in readme,
            "missing_required_terms": readme_missing,
            "passed": not readme_missing,
        },
        "documentation_index": {
            "exists": bool(index),
            "missing_required_terms": index_missing,
            "passed": bool(index) and not index_missing,
        },
        "longform_docs": {
            "missing_required_terms": longform_missing,
            "passed": not longform_missing,
        },
        "stage_registry": {
            "valid_json": registry_ok,
            "error": registry_error,
            "has_stage26_5": "xunce-stage26-5-synthetic-discrete-margin-crossing-calibration" in stages,
            "has_stage26_5b": STAGE_ID in stages,
            "passed": registry_ok
            and "xunce-stage26-5-synthetic-discrete-margin-crossing-calibration" in stages
            and STAGE_ID in stages,
        },
    }


def _route(audit: dict[str, Any]) -> tuple[str, str, list[str]]:
    reasons: list[str] = []
    if not audit["agents_md"]["passed"]:
        reasons.append("agents_md_boundary_failed")
        return "failed", ROUTE_AGENTS, reasons
    if not audit["readme"]["passed"] or not audit["documentation_index"]["passed"]:
        reasons.append("readme_or_index_boundary_failed")
        return "failed", ROUTE_README, reasons
    if not audit["longform_docs"]["passed"]:
        reasons.append("longform_docs_boundary_failed")
        return "failed", ROUTE_LONGFORM, reasons
    if not audit["stage_registry"]["passed"]:
        reasons.append("stage_registry_integrity_failed")
        return "failed", ROUTE_REGISTRY, reasons
    return "passed", ROUTE_NEXT, []


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    resolved = _resolve_path(path, repo_root)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    if payload.get("stage_id") != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID}")
    return payload


def _load_registry(path: Path) -> tuple[bool, str | None, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive diagnostic path.
        return False, str(exc), {}
    stages = payload.get("stages")
    if payload.get("schema_version") != "lunar-stage-registry/v1" or not isinstance(stages, dict):
        return False, "invalid registry schema", {}
    return True, None, stages


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _write_report(path: Path, summary: dict[str, Any], audit: dict[str, Any]) -> None:
    lines = [
        "# Stage26.5B Documentation Boundary Consolidation Report",
        "",
        f"- status: `{summary['status']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- AGENTS.md lines: `{summary['agents_md_line_count']}`",
        f"- AGENTS.md stage sections: `{summary['agents_stage_section_count']}`",
        f"- README Documentation Map: `{summary['readme_has_documentation_map']}`",
        f"- stage registry passed: `{summary['stage_registry_integrity_passed']}`",
        "",
        "This stage is a read-only documentation governance audit. It does not modify PPO, reward, Hybrid A*, candidate generation, checkpoints, policies, executors, or canary state.",
        "",
        "## Boundary Audit",
        "",
        "```json",
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
