from __future__ import annotations

import json
from pathlib import Path

from scripts import run_project_code_docs_audit as audit


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_project_code_docs_audit_writes_required_artifacts(tmp_path: Path) -> None:
    summary = audit.run(tmp_path)

    assert summary["schema_version"] == "project-code-docs-audit-summary/v1"
    assert summary["scope"] == [
        "root scripts/configs/tests/docs/outputs",
        "dev-platform-constraints",
        "model-explorer",
        "path-planner",
    ]
    assert (tmp_path / "project-code-docs-audit-summary.json").is_file()
    assert (tmp_path / "project-code-docs-findings.jsonl").is_file()
    assert (tmp_path / "project-code-docs-boundary-audit.json").is_file()
    assert (tmp_path / "project-code-docs-evidence-lineage-audit.json").is_file()
    assert (tmp_path / "project-code-docs-solution-plan.md").is_file()
    assert (tmp_path / "project-code-docs-audit-report.md").is_file()

    persisted = _read_json(tmp_path / "project-code-docs-audit-summary.json")
    assert persisted["next_required_change"] == summary["next_required_change"]
    assert persisted["status"] == "passed"
    assert persisted["finding_count"] == 0
    assert persisted["reason_codes"] == []
    assert persisted["global_99_boundaries_complete"] is True


def test_project_code_docs_audit_reports_clean_path_planner_root_cwd_contract(tmp_path: Path) -> None:
    audit.run(tmp_path)
    findings = [
        json.loads(line)
        for line in (tmp_path / "project-code-docs-findings.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert not any(finding["finding_id"] == "P2-PATH-TEST-CWD-001" for finding in findings)
    scan = _read_json(tmp_path / "project-code-docs-scan-details.json")
    assert scan["path_planner_relative_fixture_scan"]["subproject_example_exists"] is True
    assert scan["path_planner_relative_fixture_scan"]["root_example_exists"] is False
    assert scan["path_planner_relative_fixture_scan"]["relative_example_path_hits"] == []


def test_project_code_docs_audit_reports_global_99_runner_helper_adoption(tmp_path: Path) -> None:
    audit.run(tmp_path)
    findings = [
        json.loads(line)
        for line in (tmp_path / "project-code-docs-findings.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    scan = _read_json(tmp_path / "project-code-docs-scan-details.json")

    assert scan["global_99_runner_duplication_scan"]["shared_helper_present"] is True
    assert scan["global_99_runner_duplication_scan"]["high_risk_runner_helper_adoption_complete"] is True
    assert not any(finding["finding_id"] == "P2-G99-RUNNER-DRIFT-001" for finding in findings)
