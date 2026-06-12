from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_quasi_real_teacher_equivalent_validation import (  # noqa: E402
    SUMMARY_SCHEMA_VERSION,
    run_quasi_real_teacher_equivalent_validation,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _source_root(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    _write_json(
        root / "batch-evaluation-summary.json",
        {"schema_version": "batch-path-feedback-summary/v1", "failed_count": 0, "reason_codes": []},
    )
    return root


def _candidate_root(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    _write_json(
        root / "raw-policy-generalization-candidate-summary.json",
        {
            "schema_version": "raw-policy-generalization-candidate-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "experimental_checkpoint": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
        },
    )
    _write_json(
        root / "experimental-hybrid-policy-candidate-metadata.json",
        {
            "schema_version": "controlled-hybrid-policy-checkpoint-metadata/v1",
            "experimental": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
        },
    )
    (root / "experimental-hybrid-policy-candidate.pt").write_bytes(b"checkpoint")
    return root


def _quasi_real_root(tmp_path: Path, scenarios: list[dict]) -> Path:
    root = tmp_path / "quasi"
    slices = [
        {
            "scenario_id": scenario["scenario_id"],
            "roi_name": scenario.get("roi_group", "mixed_risk"),
            "split": scenario.get("split", "test"),
            "map_id": f"map-{scenario['scenario_id']}",
            "slice_id": f"slice-{scenario['scenario_id']}",
        }
        for scenario in scenarios
    ]
    _write_jsonl(root / "quasi-real-map-slices.jsonl", slices)
    _write_json(
        root / "quasi-real-map-domain-gap-summary.json",
        {
            "schema_version": "quasi-real-map-domain-gap-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "domain_gap_verdict": "acceptable_for_next_pilot",
        },
    )
    _write_json(
        root / "quasi-real-map-path-feedback-summary.json",
        {
            "schema_version": "quasi-real-map-path-feedback-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "diagnostic_profile": "quasi-real-test",
            "scenarios": scenarios,
        },
    )
    return root


def _scenario(scenario_id: str, roi_group: str = "mixed_risk") -> dict:
    return {
        "scenario_id": scenario_id,
        "scenario_group": roi_group,
        "roi_group": roi_group,
        "path_feedback": {
            "candidates": [
                {
                    "context_id": f"ctx-{scenario_id}",
                    "action_index": 1,
                    "source_selected": True,
                    "action_mask_valid": True,
                }
            ]
        },
    }


def _shadow_decision(
    scenario_id: str,
    *,
    source_action: int = 1,
    policy_action: int = 1,
    reasons: list[str] | None = None,
) -> dict:
    return {
        "scenario_id": scenario_id,
        "context_id": f"ctx-{scenario_id}",
        "scenario_group": "mixed_risk",
        "source_selected_action_index": source_action,
        "raw_policy_selected_action_index": policy_action,
        "source_selected_context_id": f"ctx-{scenario_id}-source",
        "raw_policy_selected_context_id": (
            f"ctx-{scenario_id}-source" if policy_action == source_action else f"ctx-{scenario_id}-policy"
        ),
        "raw_policy_regression_reason_codes": reasons or [],
        "action_mask_valid": True,
        "raw_policy_logit_margin_vs_source": 0.1,
        "raw_policy_selected_path_cost_delta": 0.0,
        "raw_policy_selected_risk_delta": 0.0,
    }


def _config(*, min_contexts: int = 1, min_groups: int = 1, min_agreement: float = 0.0) -> dict:
    return {
        "schema_version": "quasi-real-teacher-equivalent-validation-config/v1",
        "input_files": {
            "source_batch_summary": "batch-evaluation-summary.json",
            "quasi_real_path_feedback_summary": "quasi-real-map-path-feedback-summary.json",
            "quasi_real_slices": "quasi-real-map-slices.jsonl",
            "quasi_real_domain_gap_summary": "quasi-real-map-domain-gap-summary.json",
            "candidate_summary": "raw-policy-generalization-candidate-summary.json",
            "checkpoint": "experimental-hybrid-policy-candidate.pt",
            "checkpoint_metadata": "experimental-hybrid-policy-candidate-metadata.json",
        },
        "output_files": {
            "decisions": "quasi-real-teacher-equivalent-decisions.jsonl",
            "summary": "quasi-real-teacher-equivalent-summary.json",
            "disagreement_report": "quasi-real-teacher-equivalent-disagreement-report.json",
            "group_report": "quasi-real-teacher-equivalent-group-report.md",
        },
        "evaluation": {},
        "validation": {
            "min_teacher_equivalent_context_count": min_contexts,
            "min_roi_group_count": min_groups,
            "min_teacher_agreement_rate": min_agreement,
        },
        "non_goals": ["no_ppo_update", "no_policy_takeover"],
    }


def _run(tmp_path: Path, scenarios: list[dict], decisions: list[dict], config: dict | None = None) -> dict:
    return run_quasi_real_teacher_equivalent_validation(
        source_root=_source_root(tmp_path),
        candidate_root=_candidate_root(tmp_path),
        quasi_real_root=_quasi_real_root(tmp_path, scenarios),
        output_root=tmp_path / "out",
        config=config or _config(),
        repo_root=REPO_ROOT,
        score_decisions=lambda **_: decisions,
    )


def test_all_source_aligned_passes_as_teacher_equivalent(tmp_path: Path) -> None:
    scenarios = [_scenario("a"), _scenario("b")]
    summary = _run(
        tmp_path,
        scenarios,
        [_shadow_decision("a"), _shadow_decision("b")],
        _config(min_contexts=2, min_agreement=0.9),
    )

    assert summary["status"] == "passed"
    assert summary["schema_version"] == SUMMARY_SCHEMA_VERSION
    assert summary["teacher_equivalent_context_count"] == 2
    assert summary["teacher_aligned_count"] == 2
    assert summary["teacher_agreement_rate"] == 1.0
    assert summary["unsafe_disagreement_count"] == 0
    assert summary["teacher_equivalent_verdict"] == "teacher_equivalent_validated"


def test_safe_disagreement_passes_without_requiring_changed_choice(tmp_path: Path) -> None:
    scenarios = [_scenario("a"), _scenario("b")]
    summary = _run(
        tmp_path,
        scenarios,
        [_shadow_decision("a"), _shadow_decision("b", policy_action=2)],
        _config(min_contexts=2, min_agreement=0.5),
    )

    assert summary["status"] == "passed"
    assert summary["teacher_aligned_count"] == 1
    assert summary["safe_disagreement_count"] == 1
    assert summary["unsafe_disagreement_count"] == 0
    assert summary["policy_changed_gate_rejected_count"] == 0


def test_rejected_changed_choice_is_unsafe_disagreement(tmp_path: Path) -> None:
    scenarios = [_scenario("a")]
    summary = _run(
        tmp_path,
        scenarios,
        [_shadow_decision("a", policy_action=2, reasons=["path_cost_regression"])],
    )

    assert summary["status"] == "failed"
    assert summary["unsafe_disagreement_count"] == 1
    assert summary["policy_changed_gate_rejected_count"] == 1
    assert "quasi_real_teacher_equivalent_unsafe_disagreement" in summary["reason_codes"]
    assert summary["next_required_change"] == "quasi_real_teacher_equivalent_unsafe_disagreement"


def test_readiness_accepts_teacher_equivalent_summary() -> None:
    from run_policy_training_readiness_review import (
        _quasi_real_teacher_equivalent_validation_readiness,
    )

    readiness = _quasi_real_teacher_equivalent_validation_readiness(
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "passed",
            "reason_codes": [],
            "teacher_equivalent_verdict": "teacher_equivalent_validated",
            "teacher_equivalent_context_count": 48,
            "policy_decision_count": 48,
            "teacher_agreement_rate": 1.0,
            "roi_group_count": 4,
            "context_id_missing_count": 0,
            "unsafe_disagreement_count": 0,
            "policy_changed_gate_rejected_count": 0,
            "invalid_action_mask_count": 0,
            "fallback_or_open_grid_count": 0,
            "open_grid_fallback_count": 0,
            "safety_regression_count": 0,
            "contract_violation_count": 0,
            "contract_regression_count": 0,
            "path_cost_regression_count": 0,
            "risk_regression_count": 0,
            "source_selection_regression_count": 0,
            "runs_ppo_update": False,
            "policy_takes_control": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "git_provenance": {"current_matches_sources": True},
        }
    )

    assert readiness["present"] is True
    assert readiness["completed"] is True
    assert readiness["training_blockers"] == []
