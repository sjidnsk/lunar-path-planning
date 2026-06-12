from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_quasi_real_teacher_distillation_dataset import (  # noqa: E402
    run_quasi_real_teacher_distillation_dataset,
)
from run_quasi_real_teacher_distillation_preference_mining import (  # noqa: E402
    run_quasi_real_teacher_distillation_preference_mining,
)
from run_quasi_real_teacher_distillation_taxonomy import (  # noqa: E402
    run_quasi_real_teacher_distillation_taxonomy,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _teacher_root(tmp_path: Path) -> Path:
    root = tmp_path / "teacher_validation"
    decisions = [
        _decision("ctx-a", "scenario-a", "mixed_risk", ["path_cost_regression"], 1.2, -0.1),
        _decision("ctx-b", "scenario-b", "rim_or_steep_slope", ["path_cost_regression", "risk_regression"], 2.0, 0.2),
    ]
    _write_jsonl(root / "quasi-real-teacher-equivalent-decisions.jsonl", decisions)
    _write_json(
        root / "quasi-real-teacher-equivalent-summary.json",
        {
            "schema_version": "quasi-real-teacher-equivalent-summary/v1",
            "status": "failed",
            "reason_codes": ["quasi_real_teacher_equivalent_unsafe_disagreement"],
            "teacher_equivalent_context_count": 2,
            "unsafe_disagreement_count": 2,
            "path_cost_regression_count": 2,
            "risk_regression_count": 1,
        },
    )
    return root


def _quasi_real_root(tmp_path: Path) -> Path:
    root = tmp_path / "quasi_real"
    scenarios = []
    slices = []
    for scenario_id, roi_group in (("scenario-a", "mixed_risk"), ("scenario-b", "rim_or_steep_slope")):
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "scenario_group": roi_group,
                "path_feedback": {
                    "candidates": [
                        _candidate(f"source-{scenario_id}", 1, source_selected=True, path_cost=10.0, risk=0.2),
                        _candidate(f"alt-{scenario_id}", 2, source_selected=False, path_cost=11.5, risk=0.3),
                    ]
                },
            }
        )
        slices.append(
            {
                "scenario_id": scenario_id,
                "roi_name": roi_group,
                "split": "test",
                "map_id": f"map-{scenario_id}",
                "slice_id": f"slice-{scenario_id}",
            }
        )
    _write_json(
        root / "quasi-real-map-path-feedback-summary.json",
        {
            "schema_version": "quasi-real-map-path-feedback-summary/v1",
            "status": "passed",
            "scenarios": scenarios,
        },
    )
    _write_jsonl(root / "quasi-real-map-slices.jsonl", slices)
    return root


def _decision(
    context_id: str,
    scenario_id: str,
    roi_group: str,
    reasons: list[str],
    path_delta: float,
    risk_delta: float,
) -> dict:
    return {
        "schema_version": "quasi-real-teacher-equivalent-decision/v1",
        "context_id": context_id,
        "scenario_id": scenario_id,
        "roi_group": roi_group,
        "roi_name": roi_group,
        "split": "test",
        "map_id": f"map-{scenario_id}",
        "slice_id": f"slice-{scenario_id}",
        "source_action_index": 1,
        "raw_policy_action_index": 2,
        "logit_margin": 0.4,
        "path_cost_delta": path_delta,
        "risk_delta": risk_delta,
        "gate_reason_codes": reasons,
        "decision_class": "policy_changed_gate_rejected",
        "unsafe_disagreement": True,
    }


def _candidate(
    context_id: str,
    action_index: int,
    *,
    source_selected: bool,
    path_cost: float,
    risk: float,
) -> dict:
    return {
        "context_id": context_id,
        "action_index": action_index,
        "source_selected": source_selected,
        "action_mask_valid": True,
        "reachable": True,
        "replan_required": False,
        "path_cost": path_cost,
        "risk": risk,
        "utility": 0.0,
        "candidate_role": "source" if source_selected else "alternative",
    }


def _taxonomy_config() -> dict:
    return {
        "schema_version": "quasi-real-teacher-distillation-taxonomy-config/v1",
        "input_files": {
            "teacher_equivalent_summary": "quasi-real-teacher-equivalent-summary.json",
            "teacher_equivalent_decisions": "quasi-real-teacher-equivalent-decisions.jsonl",
            "quasi_real_path_feedback_summary": "quasi-real-map-path-feedback-summary.json",
            "quasi_real_slices": "quasi-real-map-slices.jsonl",
        },
        "output_files": {
            "summary": "quasi-real-teacher-distillation-taxonomy-summary.json",
            "taxonomy": "quasi-real-teacher-distillation-taxonomy.jsonl",
            "feature_audit": "quasi-real-teacher-distillation-feature-audit.jsonl",
            "report": "quasi-real-teacher-distillation-taxonomy-report.md",
        },
        "validation": {
            "min_unsafe_disagreement_count": 1,
            "max_bridge_or_feedback_gap_count": 0,
            "max_action_mask_or_contract_gap_count": 0,
        },
    }


def _dataset_config() -> dict:
    return {
        "schema_version": "quasi-real-teacher-distillation-dataset-config/v1",
        "input_files": {
            "taxonomy_summary": "quasi-real-teacher-distillation-taxonomy-summary.json",
            "taxonomy": "quasi-real-teacher-distillation-taxonomy.jsonl",
        },
        "output_files": {
            "slices": "quasi-real-teacher-distillation-slices.jsonl",
            "path_feedback_summary": "quasi-real-teacher-distillation-path-feedback-summary.json",
            "split_summary": "quasi-real-teacher-distillation-split-summary.json",
        },
        "variants": {"train": 2, "validation": 1, "holdout": 1},
    }


def _preference_config() -> dict:
    return {
        "schema_version": "quasi-real-teacher-distillation-preference-config/v1",
        "input_files": {
            "taxonomy_summary": "quasi-real-teacher-distillation-taxonomy-summary.json",
            "taxonomy": "quasi-real-teacher-distillation-taxonomy.jsonl",
            "distillation_slices": "quasi-real-teacher-distillation-slices.jsonl",
            "split_summary": "quasi-real-teacher-distillation-split-summary.json",
        },
        "output_files": {
            "samples": "quasi-real-teacher-distillation-preference-samples.jsonl",
            "summary": "quasi-real-teacher-distillation-preference-summary.json",
            "exclusion_report": "quasi-real-teacher-distillation-exclusion-report.json",
        },
        "validation": {
            "min_teacher_distillation_preference_count": 1,
            "max_hard_positive_added_count": 0,
            "max_ppo_transition_added_count": 0,
        },
        "weights": {
            "path_cost_only_regression": 1.5,
            "path_risk_joint_regression": 2.5,
            "risk_only_regression": 2.0,
        },
    }


def test_teacher_distillation_taxonomy_classifies_unsafe_disagreements(tmp_path: Path) -> None:
    summary = run_quasi_real_teacher_distillation_taxonomy(
        teacher_root=_teacher_root(tmp_path),
        quasi_real_root=_quasi_real_root(tmp_path),
        output_root=tmp_path / "taxonomy",
        config=_taxonomy_config(),
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["unsafe_disagreement_count"] == 2
    assert summary["classified_disagreement_count"] == 2
    assert summary["path_cost_only_regression_count"] == 1
    assert summary["path_risk_joint_regression_count"] == 1
    assert summary["bridge_or_feedback_gap_count"] == 0
    assert summary["action_mask_or_contract_gap_count"] == 0
    records = (tmp_path / "taxonomy" / "quasi-real-teacher-distillation-taxonomy.jsonl").read_text(
        encoding="utf-8"
    )
    assert "logit_margin" in records
    assert "source_candidate" in records


def test_teacher_distillation_dataset_and_preference_are_leakage_free(tmp_path: Path) -> None:
    taxonomy_root = tmp_path / "taxonomy"
    dataset_root = tmp_path / "dataset"
    preference_root = tmp_path / "preference"
    run_quasi_real_teacher_distillation_taxonomy(
        teacher_root=_teacher_root(tmp_path),
        quasi_real_root=_quasi_real_root(tmp_path),
        output_root=taxonomy_root,
        config=_taxonomy_config(),
        repo_root=REPO_ROOT,
    )

    split_summary = run_quasi_real_teacher_distillation_dataset(
        taxonomy_root=taxonomy_root,
        output_root=dataset_root,
        config=_dataset_config(),
        repo_root=REPO_ROOT,
    )
    preference_summary = run_quasi_real_teacher_distillation_preference_mining(
        taxonomy_root=taxonomy_root,
        dataset_root=dataset_root,
        output_root=preference_root,
        config=_preference_config(),
        repo_root=REPO_ROOT,
    )

    assert split_summary["status"] == "passed"
    assert split_summary["train_slice_count"] > 0
    assert split_summary["validation_slice_count"] > 0
    assert split_summary["holdout_slice_count"] > 0
    assert split_summary["context_id_overlap_count"] == 0
    assert split_summary["scenario_id_overlap_count"] == 0
    assert split_summary["slice_id_overlap_count"] == 0

    assert preference_summary["status"] == "passed"
    assert preference_summary["teacher_distillation_preference_count"] >= 4
    assert preference_summary["hard_positive_added_count"] == 0
    assert preference_summary["ppo_transition_added_count"] == 0
    assert preference_summary["holdout_leakage_count"] == 0
    samples = [
        json.loads(line)
        for line in (preference_root / "quasi-real-teacher-distillation-preference-samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert {sample["split"] for sample in samples} == {"train"}
    assert {sample["training_signal_type"] for sample in samples} == {"pairwise_preference"}


def test_readiness_accepts_teacher_distillation_summary() -> None:
    from run_policy_training_readiness_review import (
        _quasi_real_teacher_distillation_readiness,
    )

    readiness = _quasi_real_teacher_distillation_readiness(
        {
            "schema_version": "quasi-real-teacher-distillation-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "teacher_distillation_verdict": "teacher_distillation_robustness_validated",
            "taxonomy_unsafe_disagreement_count": 18,
            "classified_disagreement_count": 18,
            "teacher_distillation_preference_count": 18,
            "hard_positive_added_count": 0,
            "ppo_transition_added_count": 0,
            "holdout_leakage_count": 0,
            "context_id_overlap_count": 0,
            "scenario_id_overlap_count": 0,
            "slice_id_overlap_count": 0,
            "teacher_equivalent_context_count": 108,
            "policy_decision_count": 108,
            "roi_group_count": 4,
            "teacher_agreement_rate": 0.95,
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
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "git_provenance": {"current_matches_sources": True},
        }
    )

    assert readiness["present"] is True
    assert readiness["completed"] is True
    assert readiness["training_blockers"] == []
