#!/usr/bin/env python3
"""Generate a read-only code/docs/evidence consistency audit."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "project_code_docs_audit_v1"

BOUNDARY_FIELDS = [
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
    "runs_new_ppo_update",
    "modifies_network",
    "modifies_action_space",
    "modifies_default_astar",
]

CHAIN = [
    (
        "global_99_coverage_benchmark",
        "outputs/path_feedback_batch_global_99_coverage_benchmark_v1/global-99-coverage-benchmark-summary.json",
        "frontier_coverage_planner_baseline",
    ),
    (
        "frontier_coverage_planner_baseline",
        "outputs/path_feedback_batch_frontier_coverage_planner_baseline_v1/frontier-coverage-planner-baseline-summary.json",
        "coverage_memory_replanning_loop",
    ),
    (
        "coverage_memory_replanning_loop",
        "outputs/path_feedback_batch_coverage_memory_replanning_loop_v1/coverage-memory-replanning-loop-summary.json",
        "policy_guided_global_coverage",
    ),
    (
        "policy_guided_global_coverage",
        "outputs/path_feedback_batch_policy_guided_global_coverage_v1/policy-guided-global-coverage-summary.json",
        "global_99_multi_map_generalization",
    ),
    (
        "global_99_multi_map_generalization",
        "outputs/path_feedback_batch_global_99_multi_map_generalization_v1/global-99-multi-map-generalization-summary.json",
        "network_architecture_upgrade_readiness_review",
    ),
    (
        "network_architecture_upgrade_readiness_review",
        "outputs/path_feedback_batch_network_architecture_upgrade_readiness_review_v1/network-architecture-upgrade-readiness-summary.json",
        "global_99_release_governance_preflight",
    ),
    (
        "global_99_release_governance_preflight",
        "outputs/path_feedback_batch_global_99_release_governance_preflight_v1/global-99-release-governance-preflight-summary.json",
        "global_99_shadow_canary_preflight",
    ),
    (
        "global_99_shadow_canary_preflight",
        "outputs/path_feedback_batch_global_99_shadow_canary_preflight_v1/global-99-shadow-canary-preflight-summary.json",
        "global_99_shadow_canary_replay",
    ),
    (
        "global_99_shadow_canary_replay",
        "outputs/path_feedback_batch_global_99_shadow_canary_replay_v1/global-99-shadow-canary-replay-summary.json",
        "global_99_real_map_preflight",
    ),
    (
        "global_99_real_map_preflight",
        "outputs/path_feedback_batch_global_99_real_map_preflight_v1/global-99-real-map-preflight-summary.json",
        "global_99_real_map_shadow_replay",
    ),
    (
        "global_99_real_map_shadow_replay",
        "outputs/path_feedback_batch_global_99_real_map_shadow_replay_v1/global-99-real-map-shadow-replay-summary.json",
        "global_99_real_map_release_governance_preflight",
    ),
    (
        "global_99_real_map_release_governance_preflight",
        "outputs/path_feedback_batch_global_99_real_map_release_governance_preflight_v1/global-99-real-map-release-governance-preflight-summary.json",
        "global_99_real_map_shadow_canary_preflight",
    ),
    (
        "global_99_real_map_shadow_canary_preflight",
        "outputs/path_feedback_batch_global_99_real_map_shadow_canary_preflight_v1/global-99-real-map-shadow-canary-preflight-summary.json",
        "global_99_real_map_shadow_canary_replay",
    ),
    (
        "global_99_real_map_shadow_canary_replay",
        "outputs/path_feedback_batch_global_99_real_map_shadow_canary_replay_v1/global-99-real-map-shadow-canary-replay-summary.json",
        "global_99_real_map_evidence_refresh_drift_audit",
    ),
    (
        "global_99_real_map_evidence_refresh_drift_audit",
        "outputs/path_feedback_batch_global_99_real_map_evidence_refresh_drift_audit_v1/global-99-real-map-evidence-refresh-drift-audit-summary.json",
        "global_99_real_map_multi_roi_generalization",
    ),
    (
        "global_99_real_map_multi_roi_generalization",
        "outputs/path_feedback_batch_global_99_real_map_multi_roi_generalization_v1/global-99-real-map-multi-roi-generalization-summary.json",
        "default_policy_candidate_authorization_preflight",
    ),
    (
        "default_policy_candidate_authorization_preflight",
        "outputs/path_feedback_batch_global_99_default_policy_candidate_authorization_preflight_v1/global-99-default-policy-candidate-authorization-preflight-summary.json",
        "sandbox_candidate_installation_dry_run",
    ),
    (
        "sandbox_candidate_installation_dry_run",
        "outputs/path_feedback_batch_global_99_sandbox_candidate_installation_dry_run_v1/global-99-sandbox-candidate-installation-dry-run-summary.json",
        "sandbox_consumer_replay_canary",
    ),
    (
        "sandbox_consumer_replay_canary",
        "outputs/path_feedback_batch_global_99_sandbox_consumer_replay_canary_v1/global-99-sandbox-consumer-replay-canary-summary.json",
        "controlled_default_policy_candidate_installation_preflight",
    ),
    (
        "controlled_default_policy_candidate_installation_preflight",
        "outputs/path_feedback_batch_global_99_controlled_default_policy_candidate_installation_preflight_v1/global-99-controlled-default-policy-candidate-installation-preflight-summary.json",
        "eligible_for_controlled_default_policy_candidate_installation_review",
    ),
]


@dataclass(frozen=True)
class Finding:
    finding_id: str
    severity: str
    category: str
    title: str
    evidence_paths: list[str]
    documentation_claim: str
    actual_behavior: str
    risk: str
    solution: str
    verification: str

    def to_json(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "severity": self.severity,
            "category": self.category,
            "title": self.title,
            "evidence_paths": self.evidence_paths,
            "documentation_claim": self.documentation_claim,
            "actual_behavior": self.actual_behavior,
            "risk": self.risk,
            "solution": self.solution,
            "verification": self.verification,
        }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else completed.stderr.strip()


def _collect_baseline() -> dict[str, Any]:
    return {
        "branch": _git(["branch", "--show-current"]),
        "head": _git(["rev-parse", "HEAD"]),
        "git_status_short": _git(["status", "--short"]).splitlines(),
        "submodule_status": _git(["submodule", "status", "--recursive"]).splitlines(),
        "project_roots": [
            "dev-platform-constraints",
            "model-explorer",
            "path-planner",
        ],
        "legacy_reference_roots": [
            p.as_posix()
            for p in sorted(ROOT.glob("a_gcs_ws*"))
            if p.is_dir()
        ],
    }


def _collect_evidence_lineage() -> dict[str, Any]:
    rows = []
    for stage, rel_path, expected_next in CHAIN:
        path = ROOT / rel_path
        row: dict[str, Any] = {
            "stage": stage,
            "summary_path": rel_path,
            "expected_next_required_change": expected_next,
            "exists": path.exists(),
        }
        if path.exists():
            summary = _read_json(path)
            row.update(
                {
                    "status": summary.get("status"),
                    "reason_codes": summary.get("reason_codes", []),
                    "next_required_change": summary.get("next_required_change"),
                    "matches_expected_next": summary.get("next_required_change") == expected_next,
                    "passed": summary.get("status") == "passed",
                }
            )
        else:
            row.update({"status": "missing", "reason_codes": ["missing_summary"]})
        rows.append(row)
    return {
        "schema_version": "project-code-docs-evidence-lineage-audit/v1",
        "chain": rows,
        "all_summaries_present": all(row["exists"] for row in rows),
        "all_stages_passed": all(row.get("passed") for row in rows),
        "all_next_required_changes_match": all(row.get("matches_expected_next") for row in rows),
    }


def _collect_boundary_audit() -> dict[str, Any]:
    rows = []
    for stage, rel_path, _expected_next in CHAIN:
        path = ROOT / rel_path
        if not path.exists():
            rows.append(
                {
                    "stage": stage,
                    "summary_path": rel_path,
                    "exists": False,
                    "missing_boundary_fields": BOUNDARY_FIELDS,
                    "boundary_violations": {},
                }
            )
            continue
        summary = _read_json(path)
        missing = [field for field in BOUNDARY_FIELDS if field not in summary]
        violations = {
            field: summary.get(field)
            for field in BOUNDARY_FIELDS
            if field in summary and summary.get(field) not in (False, 0, 0.0)
        }
        rows.append(
            {
                "stage": stage,
                "summary_path": rel_path,
                "exists": True,
                "missing_boundary_fields": missing,
                "boundary_violations": violations,
                "boundary_complete": not missing,
                "boundary_closed": not violations,
            }
        )
    return {
        "schema_version": "project-code-docs-boundary-audit/v1",
        "boundary_fields": BOUNDARY_FIELDS,
        "stage_boundary_rows": rows,
        "all_stage_boundaries_complete": all(not row["missing_boundary_fields"] for row in rows),
        "all_stage_boundaries_closed": all(not row["boundary_violations"] for row in rows),
    }


def _scan_relative_path_planner_tests() -> dict[str, Any]:
    paths = [
        ROOT / "path-planner" / "tests" / "test_cli_diagnostics.py",
        ROOT / "path-planner" / "tests" / "test_drake_backend.py",
    ]
    hits = []
    for path in paths:
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "examples/demo_map_corridor.json" in line:
                hits.append({"path": _rel(path), "line": lineno, "text": line.strip()})
    return {
        "root_example_exists": (ROOT / "examples" / "demo_map_corridor.json").exists(),
        "subproject_example_exists": (ROOT / "path-planner" / "examples" / "demo_map_corridor.json").exists(),
        "relative_example_path_hits": hits,
    }


def _scan_runner_duplication() -> dict[str, Any]:
    runners = sorted((ROOT / "scripts").glob("run_global_99_*.py"))
    line_counts = []
    boundary_mentions = 0
    high_risk_runners = {
        "scripts/run_global_99_coverage_benchmark.py",
        "scripts/run_global_99_multi_map_generalization.py",
        "scripts/run_global_99_release_governance_preflight.py",
    }
    helper_path = ROOT / "scripts" / "global_99_governance_common.py"
    adopted_high_risk_runners = []
    for path in runners:
        text = path.read_text(encoding="utf-8")
        rel_path = _rel(path)
        line_counts.append({"path": rel_path, "line_count": len(text.splitlines())})
        boundary_mentions += sum(text.count(field) for field in BOUNDARY_FIELDS)
        if rel_path in high_risk_runners and "global_99_governance_common" in text:
            adopted_high_risk_runners.append(rel_path)
    return {
        "runner_count": len(runners),
        "total_line_count": sum(row["line_count"] for row in line_counts),
        "largest_runners": sorted(line_counts, key=lambda row: row["line_count"], reverse=True)[:8],
        "boundary_field_mentions": boundary_mentions,
        "shared_helper_present": helper_path.exists(),
        "high_risk_runners": sorted(high_risk_runners),
        "adopted_high_risk_runners": sorted(adopted_high_risk_runners),
        "high_risk_runner_helper_adoption_complete": sorted(adopted_high_risk_runners)
        == sorted(high_risk_runners),
    }


def _scan_docs_portability() -> dict[str, Any]:
    docs = [
        ROOT / "README.md",
        ROOT / "dev-platform-constraints" / "README.md",
        ROOT / "path-planner" / "README.md",
        ROOT / "docs" / "算法设计与系统架构报告.md",
    ]
    patterns = ["D:\\", "D:/conda_envs", "D:\\conda_envs"]
    hits = []
    for path in docs:
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(pattern in line for pattern in patterns):
                hits.append({"path": _rel(path), "line": lineno, "text": line.strip()})
    return {"windows_absolute_path_hits": hits}


def _build_findings(
    evidence: dict[str, Any],
    boundary: dict[str, Any],
    path_tests: dict[str, Any],
    duplication: dict[str, Any],
    docs_portability: dict[str, Any],
) -> list[Finding]:
    findings: list[Finding] = []
    if not evidence["all_summaries_present"] or not evidence["all_stages_passed"]:
        findings.append(
            Finding(
                finding_id="P0-G99-LINEAGE-001",
                severity="P0 correctness/safety",
                category="evidence-lineage",
                title="Global 99 evidence chain is incomplete or not fully passed",
                evidence_paths=["outputs/path_feedback_batch_global_99*/**/*summary.json"],
                documentation_claim="The Global 99 line advances stage-by-stage through explicit next_required_change gates.",
                actual_behavior="At least one expected summary is missing or not passed.",
                risk="A downstream governance decision could rely on an incomplete evidence chain.",
                solution="Regenerate or fix the failing stage before any release/governance review.",
                verification="Run scripts for the failing stage, then rerun scripts/run_project_code_docs_audit.py.",
            )
        )
    missing_rows = [
        row
        for row in boundary["stage_boundary_rows"]
        if row["stage"].startswith("global_99")
        and row["missing_boundary_fields"]
        and row["stage"]
        in {
            "global_99_coverage_benchmark",
            "global_99_multi_map_generalization",
            "global_99_release_governance_preflight",
        }
    ]
    if missing_rows:
        findings.append(
            Finding(
                finding_id="P1-G99-BOUNDARY-001",
                severity="P1 release-boundary",
                category="release-boundary",
                title="Some passed Global 99 primary summaries omit the online-canary boundary field",
                evidence_paths=[row["summary_path"] for row in missing_rows],
                documentation_claim="The Global 99 line keeps checkpoint/default-policy/executor/online-canary/PPO/network/action-space/default-A* boundaries explicit.",
                actual_behavior="Several passed primary summaries omit starts_online_canary, even though later governance stages audit that boundary explicitly.",
                risk="Generic release-boundary auditors can distinguish false from missing only if all primary stage summaries share a normalized boundary vocabulary.",
                solution="Add a shared Global 99 boundary field normalizer and backfill starts_online_canary=false into primary summaries/tests/docs where the stage is offline.",
                verification="Rerun all affected run_global_99_* scripts and confirm project-code-docs-boundary-audit.json reports all_stage_boundaries_complete=true.",
            )
        )
    if path_tests["relative_example_path_hits"] and not path_tests["root_example_exists"] and path_tests["subproject_example_exists"]:
        findings.append(
            Finding(
                finding_id="P2-PATH-TEST-CWD-001",
                severity="P2 maintainability",
                category="validation-command",
                title="path-planner tests are not root-cwd invariant",
                evidence_paths=[
                    "path-planner/tests/test_cli_diagnostics.py",
                    "path-planner/tests/test_drake_backend.py",
                    "path-planner/examples/demo_map_corridor.json",
                ],
                documentation_claim="The project-level audit command can run path-planner tests from the root checkout with PYTHONPATH=path-planner/src.",
                actual_behavior="CLI subprocess tests pass from path-planner/ but fail from the root because they pass examples/demo_map_corridor.json as a root-relative path.",
                risk="A root-level CI job or audit run reports false failures for path-planner even though the subproject itself is green.",
                solution="Use absolute fixture paths in subprocess tests, or document and enforce cd path-planner before running path-planner tests from root-level scripts.",
                verification="Run both PYTHONPATH=path-planner/src pytest path-planner/tests -q from root and PYTHONPATH=src pytest tests -q inside path-planner; both should pass.",
            )
        )
    if (
        duplication["runner_count"] >= 12
        and duplication["boundary_field_mentions"] >= 100
        and not duplication["high_risk_runner_helper_adoption_complete"]
    ):
        findings.append(
            Finding(
                finding_id="P2-G99-RUNNER-DRIFT-001",
                severity="P2 maintainability",
                category="runner-maintainability",
                title="Global 99 runner chain duplicates boundary and audit wiring across many scripts",
                evidence_paths=["scripts/run_global_99_*.py", "scripts/git_provenance.py"],
                documentation_claim="Each Global 99 stage preserves consistent boundary, provenance, rejection, manifest, and next_required_change contracts.",
                actual_behavior=f"{duplication['runner_count']} Global 99 runners contain {duplication['total_line_count']} total lines and {duplication['boundary_field_mentions']} boundary-field mentions.",
                risk="Future stages can drift by omitting a boundary field, reason-code route, provenance check, or doc-validation hook.",
                solution="Extract shared helpers for boundary defaults, source summary loading, rejection reports, manifest writing, and next_required_change decision records.",
                verification="Add unit tests for the shared helper and rerun tests/test_global_99_*.py.",
            )
        )
    if docs_portability["windows_absolute_path_hits"]:
        findings.append(
            Finding(
                finding_id="P3-DOCS-PORTABILITY-001",
                severity="P3 documentation drift",
                category="documentation",
                title="Some README examples retain machine-specific Windows paths",
                evidence_paths=sorted({hit["path"] for hit in docs_portability["windows_absolute_path_hits"]}),
                documentation_claim="Development commands should be reproducible in the current Linux/Conda workspace and across supported platforms.",
                actual_behavior="Several documentation lines include D:/ or D:\\ absolute paths while current validation uses /home/kai/anaconda3/envs/lunar-explorer.",
                risk="New audit runs can follow the wrong path conventions or misread stale Windows examples as current Linux commands.",
                solution="Replace absolute local paths with repo-relative paths and keep separate Windows/Linux command blocks where needed.",
                verification="rg -n 'D:/|D:\\\\' README.md dev-platform-constraints/README.md path-planner/README.md docs/算法设计与系统架构报告.md returns only intentional historical notes.",
            )
        )
    return findings


def _highest_severity(findings: list[Finding]) -> str:
    order = {
        "P0 correctness/safety": 0,
        "P1 release-boundary": 1,
        "P2 maintainability": 2,
        "P3 documentation drift": 3,
    }
    if not findings:
        return "none"
    return min((finding.severity for finding in findings), key=lambda severity: order[severity])


def _next_required_change(highest: str) -> str:
    if highest.startswith("P0"):
        return "fix_p0_project_correctness_safety_findings"
    if highest.startswith("P1"):
        return "fix_p1_release_boundary_contract_findings"
    if highest.startswith("P2"):
        return "fix_p2_validation_and_maintainability_findings"
    if highest.startswith("P3"):
        return "fix_p3_documentation_drift_findings"
    return "project_code_docs_audit_clean"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_findings(path: Path, findings: list[Finding]) -> None:
    path.write_text(
        "".join(json.dumps(finding.to_json(), sort_keys=True) + "\n" for finding in findings),
        encoding="utf-8",
    )


def _write_markdown(path: Path, summary: dict[str, Any], findings: list[Finding]) -> None:
    lines = [
        "# Project Code / Docs Audit v1",
        "",
        f"- status: `{summary['status']}`",
        f"- highest_severity: `{summary['highest_severity']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- finding_count: `{summary['finding_count']}`",
        "",
        "## Findings",
        "",
    ]
    if not findings:
        lines.append("No findings were detected by the current audit checks.")
    for finding in findings:
        lines.extend(
            [
                f"### {finding.finding_id}: {finding.title}",
                "",
                f"- severity: `{finding.severity}`",
                f"- category: `{finding.category}`",
                f"- evidence: {', '.join(f'`{p}`' for p in finding.evidence_paths)}",
                f"- documentation claim: {finding.documentation_claim}",
                f"- actual behavior: {finding.actual_behavior}",
                f"- risk: {finding.risk}",
                f"- solution: {finding.solution}",
                f"- verification: `{finding.verification}`",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_solution_plan(path: Path, findings: list[Finding]) -> None:
    grouped: dict[str, list[Finding]] = {}
    for finding in findings:
        grouped.setdefault(finding.severity, []).append(finding)
    lines = [
        "# Project Code / Docs Remediation Route",
        "",
        "This route is ordered by release risk. It does not approve checkpoint publication, default-policy replacement, real executor connection, online canary traffic, PPO updates, or network/action-space/default-A* changes.",
        "",
    ]
    for severity in ["P0 correctness/safety", "P1 release-boundary", "P2 maintainability", "P3 documentation drift"]:
        rows = grouped.get(severity, [])
        if not rows:
            continue
        lines.extend([f"## {severity}", ""])
        for finding in rows:
            lines.extend(
                [
                    f"- `{finding.finding_id}`: {finding.solution}",
                    f"  Verification: `{finding.verification}`",
                ]
            )
        lines.append("")
    if not findings:
        lines.append("No remediation route is required by the current audit.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    baseline = _collect_baseline()
    evidence = _collect_evidence_lineage()
    boundary = _collect_boundary_audit()
    path_tests = _scan_relative_path_planner_tests()
    duplication = _scan_runner_duplication()
    docs_portability = _scan_docs_portability()
    findings = _build_findings(evidence, boundary, path_tests, duplication, docs_portability)
    counts = Counter(finding.severity for finding in findings)
    highest = _highest_severity(findings)
    summary = {
        "schema_version": "project-code-docs-audit-summary/v1",
        "status": "completed_with_findings" if findings else "passed",
        "reason_codes": [finding.finding_id for finding in findings],
        "highest_severity": highest,
        "finding_count": len(findings),
        "finding_counts_by_severity": dict(sorted(counts.items())),
        "next_required_change": _next_required_change(highest),
        "scope": [
            "root scripts/configs/tests/docs/outputs",
            "dev-platform-constraints",
            "model-explorer",
            "path-planner",
        ],
        "legacy_reference_only": ["a_gcs_ws*"],
        "global_99_lineage_passed": evidence["all_stages_passed"],
        "global_99_next_required_changes_match": evidence["all_next_required_changes_match"],
        "global_99_boundaries_closed": boundary["all_stage_boundaries_closed"],
        "global_99_boundaries_complete": boundary["all_stage_boundaries_complete"],
        "baseline": baseline,
        "non_goals": {
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
            "claims_real_world_performance": False,
        },
    }
    _write_json(output_root / "project-code-docs-audit-summary.json", summary)
    _write_findings(output_root / "project-code-docs-findings.jsonl", findings)
    _write_json(output_root / "project-code-docs-boundary-audit.json", boundary)
    _write_json(output_root / "project-code-docs-evidence-lineage-audit.json", evidence)
    _write_json(
        output_root / "project-code-docs-scan-details.json",
        {
            "path_planner_relative_fixture_scan": path_tests,
            "global_99_runner_duplication_scan": duplication,
            "docs_portability_scan": docs_portability,
        },
    )
    _write_markdown(output_root / "project-code-docs-audit-report.md", summary, findings)
    _write_solution_plan(output_root / "project-code-docs-solution-plan.md", findings)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Project Code / Docs Audit v1.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    summary = run(args.output_root)
    print(json.dumps({"status": summary["status"], "finding_count": summary["finding_count"], "next_required_change": summary["next_required_change"], "summary": str(args.output_root / "project-code-docs-audit-summary.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
