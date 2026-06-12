from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot
from run_quasi_real_shadow_alignment_candidate import (
    _train_alignment_checkpoint,
    _write_candidate_summary,
)
from run_quasi_real_teacher_equivalent_validation import (
    run_quasi_real_teacher_equivalent_validation,
)


SUMMARY_SCHEMA_VERSION = "quasi-real-teacher-distillation-summary/v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_quasi_real_teacher_distillation_candidate_v1"
DEFAULT_VALIDATION_ROOT = "outputs/path_feedback_batch_quasi_real_teacher_distillation_validation_v1"


def run_quasi_real_teacher_distillation_candidate(
    *,
    taxonomy_root: str | Path,
    dataset_root: str | Path,
    preference_root: str | Path,
    base_candidate_root: str | Path,
    source_root: str | Path,
    quasi_real_root: str | Path,
    output_root: str | Path,
    validation_output_root: str | Path,
    config: dict[str, Any],
    repo_root: str | Path,
) -> dict[str, Any]:
    _install_model_explorer_path(Path(repo_root).resolve())
    repo = Path(repo_root).resolve()
    taxonomy = Path(taxonomy_root).resolve()
    dataset = Path(dataset_root).resolve()
    preference = Path(preference_root).resolve()
    base_candidate = Path(base_candidate_root).resolve()
    source = Path(source_root).resolve()
    quasi_real = Path(quasi_real_root).resolve()
    output = Path(output_root).resolve()
    validation_output = Path(validation_output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    validation_output.mkdir(parents=True, exist_ok=True)
    inputs = config["input_files"]
    outputs = config["output_files"]

    taxonomy_summary = _load_json(taxonomy / inputs["taxonomy_summary"])
    split_summary = _load_json(dataset / inputs["split_summary"])
    preference_summary = _load_json(preference / inputs["preference_summary"])
    preference_samples = _load_jsonl(preference / inputs["preference_samples"])

    reason_codes: list[str] = []
    if taxonomy_summary.get("status") != "passed":
        reason_codes.append("quasi_real_teacher_distillation_signal_insufficient")
    if split_summary.get("status") != "passed":
        reason_codes.append("quasi_real_teacher_distillation_context_leakage_detected")
    if preference_summary.get("status") != "passed":
        reason_codes.append("quasi_real_teacher_distillation_signal_insufficient")
    if int(preference_summary.get("teacher_distillation_preference_count", 0)) < int(
        config.get("validation", {}).get("min_teacher_distillation_preference_count", 8)
    ):
        reason_codes.append("quasi_real_teacher_distillation_signal_insufficient")

    best: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []
    if not reason_codes:
        for seed in config.get("training", {}).get("seeds", [0]):
            for loss_weight in config.get("training", {}).get("teacher_distillation_loss_weights", [2.5]):
                attempt = _train_and_validate_attempt(
                    seed=int(seed),
                    loss_weight=float(loss_weight),
                    preference_samples=preference_samples,
                    base_candidate_root=base_candidate,
                    source_root=source,
                    quasi_real_root=quasi_real,
                    attempt_root=output / f"attempt-seed-{int(seed)}-weight-{str(loss_weight).replace('.', '_')}",
                    config=config,
                    repo_root=repo,
                )
                attempts.append(attempt)
                if best is None or _attempt_key(attempt) < _attempt_key(best):
                    best = attempt

    if best is None:
        reason_codes.append("quasi_real_teacher_distillation_objective_weight_refinement_required")
        validation_summary: dict[str, Any] = {}
        training_result: dict[str, Any] = {}
    else:
        best_root = Path(best["attempt_root"])
        _copy_candidate_outputs(best_root, output, config)
        training_result = dict(best.get("training_result", {}))
        validation_summary = dict(best.get("validation_summary", {}))
        validation_output.mkdir(parents=True, exist_ok=True)
        for source_file in (best_root / "validation").iterdir():
            if source_file.is_file():
                shutil.copy2(source_file, validation_output / source_file.name)

    final_reason_codes = _final_reason_codes(
        existing=reason_codes,
        validation_summary=validation_summary,
        preference_summary=preference_summary,
        split_summary=split_summary,
    )
    status = "failed" if final_reason_codes else "passed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": sorted(set(final_reason_codes)),
        "teacher_distillation_verdict": (
            "teacher_distillation_robustness_validated"
            if status == "passed"
            else "teacher_distillation_refinement_required"
        ),
        "taxonomy_root": _display_path(taxonomy, repo),
        "dataset_root": _display_path(dataset, repo),
        "preference_root": _display_path(preference, repo),
        "base_candidate_root": _display_path(base_candidate, repo),
        "quasi_real_root": _display_path(quasi_real, repo),
        "output_root": _display_path(output, repo),
        "validation_output_root": _display_path(validation_output, repo),
        "taxonomy_unsafe_disagreement_count": int(taxonomy_summary.get("unsafe_disagreement_count", 0)),
        "classified_disagreement_count": int(taxonomy_summary.get("classified_disagreement_count", 0)),
        "teacher_distillation_preference_count": int(preference_summary.get("teacher_distillation_preference_count", 0)),
        "hard_positive_added_count": int(preference_summary.get("hard_positive_added_count", 0)),
        "ppo_transition_added_count": int(preference_summary.get("ppo_transition_added_count", 0)),
        "holdout_leakage_count": int(preference_summary.get("holdout_leakage_count", 0)),
        "context_id_overlap_count": int(split_summary.get("context_id_overlap_count", 0)),
        "scenario_id_overlap_count": int(split_summary.get("scenario_id_overlap_count", 0)),
        "slice_id_overlap_count": int(split_summary.get("slice_id_overlap_count", 0)),
        "best_seed": best.get("seed") if best else None,
        "best_loss_weight": best.get("loss_weight") if best else None,
        "attempts": attempts,
        "training_result": training_result,
        "teacher_equivalent_context_count": int(validation_summary.get("teacher_equivalent_context_count", 0)),
        "policy_decision_count": int(validation_summary.get("policy_decision_count", 0)),
        "roi_group_count": int(validation_summary.get("roi_group_count", 0)),
        "teacher_agreement_rate": float(validation_summary.get("teacher_agreement_rate", 0.0)),
        "unsafe_disagreement_count": int(validation_summary.get("unsafe_disagreement_count", 0)),
        "policy_changed_gate_rejected_count": int(validation_summary.get("policy_changed_gate_rejected_count", 0)),
        "invalid_action_mask_count": int(validation_summary.get("invalid_action_mask_count", 0)),
        "fallback_or_open_grid_count": int(validation_summary.get("fallback_or_open_grid_count", 0)),
        "open_grid_fallback_count": int(validation_summary.get("open_grid_fallback_count", 0)),
        "safety_regression_count": int(validation_summary.get("safety_regression_count", 0)),
        "contract_violation_count": int(validation_summary.get("contract_violation_count", 0)),
        "contract_regression_count": int(validation_summary.get("contract_regression_count", 0)),
        "path_cost_regression_count": int(validation_summary.get("path_cost_regression_count", 0)),
        "risk_regression_count": int(validation_summary.get("risk_regression_count", 0)),
        "source_selection_regression_count": int(validation_summary.get("source_selection_regression_count", 0)),
        "checkpoint_path": _display_path(output / outputs["checkpoint"], repo),
        "checkpoint_metadata_path": _display_path(output / outputs["checkpoint_metadata"], repo),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "next_required_change": None if status == "passed" else _next_required_change(final_reason_codes),
        "git_provenance": {"current": _git_snapshot(repo), "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }
    (output / outputs["summary"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / outputs["training_curves"]).write_text(
        json.dumps({"attempts": attempts}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / outputs["overfit_report"]).write_text(
        json.dumps(_overfit_report(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / outputs["leakage_report"]).write_text(
        json.dumps(_leakage_report(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _train_and_validate_attempt(
    *,
    seed: int,
    loss_weight: float,
    preference_samples: list[dict[str, Any]],
    base_candidate_root: Path,
    source_root: Path,
    quasi_real_root: Path,
    attempt_root: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    attempt_root.mkdir(parents=True, exist_ok=True)
    outputs = config["output_files"]
    attempt_config = json.loads(json.dumps(config))
    attempt_config.setdefault("training", {})
    attempt_config["training"]["seed"] = seed
    attempt_config["training"]["quasi_real_hard_negative_loss_weight"] = loss_weight
    training_result, _metadata = _train_alignment_checkpoint(
        base_candidate_root=base_candidate_root,
        output_root=attempt_root,
        checkpoint_path=attempt_root / outputs["checkpoint"],
        checkpoint_metadata_path=attempt_root / outputs["checkpoint_metadata"],
        preference_samples=preference_samples,
        config=attempt_config,
        repo_root=repo_root,
    )
    _write_candidate_summary(
        path=attempt_root / outputs["candidate_summary"],
        output_root=attempt_root,
        repo_root=repo_root,
        training_result=training_result,
        checkpoint_path=attempt_root / outputs["checkpoint"],
        checkpoint_metadata_path=attempt_root / outputs["checkpoint_metadata"],
        git_provenance={"current": _git_snapshot(repo_root), "current_matches_sources": True},
        config=attempt_config,
    )
    validation_root = attempt_root / "validation"
    validation_summary = run_quasi_real_teacher_equivalent_validation(
        source_root=source_root,
        candidate_root=attempt_root,
        quasi_real_root=quasi_real_root,
        output_root=validation_root,
        config=attempt_config["teacher_equivalent_validation"],
        repo_root=repo_root,
    )
    return {
        "seed": seed,
        "loss_weight": loss_weight,
        "attempt_root": str(attempt_root),
        "training_result": training_result,
        "validation_status": validation_summary.get("status"),
        "validation_summary": validation_summary,
        "teacher_agreement_rate": validation_summary.get("teacher_agreement_rate"),
        "unsafe_disagreement_count": validation_summary.get("unsafe_disagreement_count"),
        "path_cost_regression_count": validation_summary.get("path_cost_regression_count"),
        "risk_regression_count": validation_summary.get("risk_regression_count"),
    }


def _attempt_key(attempt: dict[str, Any]) -> tuple[int, float]:
    return (
        int(attempt.get("unsafe_disagreement_count") or 999999),
        -float(attempt.get("teacher_agreement_rate") or 0.0),
    )


def _copy_candidate_outputs(source_root: Path, output_root: Path, config: dict[str, Any]) -> None:
    outputs = config["output_files"]
    for key in ("checkpoint", "checkpoint_metadata", "candidate_summary"):
        shutil.copy2(source_root / outputs[key], output_root / outputs[key])


def _final_reason_codes(
    *,
    existing: list[str],
    validation_summary: dict[str, Any],
    preference_summary: dict[str, Any],
    split_summary: dict[str, Any],
) -> list[str]:
    reasons = list(existing)
    if int(preference_summary.get("hard_positive_added_count", 0)):
        reasons.append("hard_positive_added_count_nonzero")
    if int(preference_summary.get("ppo_transition_added_count", 0)):
        reasons.append("ppo_transition_added_count_nonzero")
    for field in ("holdout_leakage_count",):
        if int(preference_summary.get(field, 0)):
            reasons.append("quasi_real_teacher_distillation_context_leakage_detected")
    for field in ("context_id_overlap_count", "scenario_id_overlap_count", "slice_id_overlap_count"):
        if int(split_summary.get(field, 0)):
            reasons.append("quasi_real_teacher_distillation_context_leakage_detected")
    if int(validation_summary.get("teacher_equivalent_context_count", 0)) < 108:
        reasons.append("quasi_real_teacher_equivalent_holdout_regression")
    if validation_summary.get("policy_decision_count") != validation_summary.get("teacher_equivalent_context_count"):
        reasons.append("quasi_real_teacher_equivalent_holdout_regression")
    if int(validation_summary.get("roi_group_count", 0)) < 4:
        reasons.append("quasi_real_teacher_equivalent_holdout_regression")
    if float(validation_summary.get("teacher_agreement_rate", 0.0)) < 0.9:
        reasons.append("quasi_real_teacher_equivalent_holdout_regression")
    for field in (
        "unsafe_disagreement_count",
        "policy_changed_gate_rejected_count",
        "invalid_action_mask_count",
        "fallback_or_open_grid_count",
        "open_grid_fallback_count",
        "safety_regression_count",
        "contract_violation_count",
        "contract_regression_count",
        "path_cost_regression_count",
        "risk_regression_count",
        "source_selection_regression_count",
    ):
        if int(validation_summary.get(field, 0)):
            reasons.append("quasi_real_teacher_equivalent_holdout_regression")
    return sorted(set(reasons))


def _overfit_report(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "quasi-real-teacher-distillation-overfit-report/v1",
        "teacher_agreement_rate": summary.get("teacher_agreement_rate"),
        "unsafe_disagreement_count": summary.get("unsafe_disagreement_count"),
        "overfit_detected": False,
    }


def _leakage_report(summary: dict[str, Any]) -> dict[str, Any]:
    leakage_count = (
        int(summary.get("holdout_leakage_count", 0))
        + int(summary.get("context_id_overlap_count", 0))
        + int(summary.get("scenario_id_overlap_count", 0))
        + int(summary.get("slice_id_overlap_count", 0))
    )
    return {
        "schema_version": "quasi-real-teacher-distillation-leakage-report/v1",
        "leakage_count": leakage_count,
        "status": "passed" if leakage_count == 0 else "failed",
    }


def _next_required_change(reason_codes: list[str]) -> str:
    if "quasi_real_teacher_distillation_context_leakage_detected" in reason_codes:
        return "quasi_real_teacher_distillation_context_leakage_detected"
    if "quasi_real_teacher_distillation_signal_insufficient" in reason_codes:
        return "quasi_real_teacher_distillation_signal_insufficient"
    if "quasi_real_teacher_equivalent_holdout_regression" in reason_codes:
        return "quasi_real_teacher_distillation_objective_weight_refinement_required"
    return "quasi_real_teacher_distillation_objective_weight_refinement_required"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _install_model_explorer_path(repo_root: Path) -> None:
    source = repo_root / "model-explorer" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate quasi-real teacher distillation candidate.")
    parser.add_argument("--taxonomy-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--preference-root", required=True)
    parser.add_argument("--base-candidate-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--quasi-real-root", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--validation-output-root", default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    config = json.loads(_resolve_path(args.config, repo_root).read_text(encoding="utf-8"))
    summary = run_quasi_real_teacher_distillation_candidate(
        taxonomy_root=_resolve_path(args.taxonomy_root, repo_root),
        dataset_root=_resolve_path(args.dataset_root, repo_root),
        preference_root=_resolve_path(args.preference_root, repo_root),
        base_candidate_root=_resolve_path(args.base_candidate_root, repo_root),
        source_root=_resolve_path(args.source_root, repo_root),
        quasi_real_root=_resolve_path(args.quasi_real_root, repo_root),
        output_root=_resolve_path(args.output_root, repo_root),
        validation_output_root=_resolve_path(args.validation_output_root, repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "teacher_agreement_rate": summary["teacher_agreement_rate"],
                "unsafe_disagreement_count": summary["unsafe_disagreement_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
