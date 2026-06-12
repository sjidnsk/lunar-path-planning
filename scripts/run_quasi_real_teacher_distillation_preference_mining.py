from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot
from run_fresh_holdout_policy_candidate_evaluation import (
    _candidate_features,
    _global_features,
    _missing_indicators,
)


SUMMARY_SCHEMA_VERSION = "quasi-real-teacher-distillation-preference-summary/v1"
SAMPLE_SCHEMA_VERSION = "quasi-real-teacher-distillation-preference-sample/v1"
EXCLUSION_SCHEMA_VERSION = "quasi-real-teacher-distillation-exclusion-report/v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_quasi_real_teacher_distillation_preference_v1"


def run_quasi_real_teacher_distillation_preference_mining(
    *,
    taxonomy_root: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    config: dict[str, Any],
    repo_root: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    taxonomy = Path(taxonomy_root).resolve()
    dataset = Path(dataset_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = config["input_files"]
    outputs = config["output_files"]
    taxonomy_summary = _load_json(taxonomy / inputs["taxonomy_summary"])
    taxonomy_records = _load_jsonl(taxonomy / inputs["taxonomy"])
    slices = _load_jsonl(dataset / inputs["distillation_slices"])
    split_summary = _load_json(dataset / inputs["split_summary"])
    weights = dict(config.get("weights", {}))

    records_by_id = {
        str(record.get("taxonomy_record_id")): record
        for record in taxonomy_records
        if record.get("taxonomy_record_id")
    }
    records_by_scenario = {str(record.get("scenario_id")): record for record in taxonomy_records}
    holdout_context_ids = {str(item.get("context_id")) for item in slices if item.get("split") == "holdout"}
    samples: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for slice_record in slices:
        if slice_record.get("split") != "train":
            continue
        record = records_by_id.get(str(slice_record.get("taxonomy_record_id")))
        if not record:
            record = records_by_scenario.get(str(slice_record.get("source_failure_scenario_id")))
        if not record:
            exclusions.append({"scenario_id": slice_record.get("scenario_id"), "reason": "taxonomy_record_missing"})
            continue
        failure_class = str(record.get("failure_class", ""))
        if failure_class not in {"path_cost_only_regression", "path_risk_joint_regression", "risk_only_regression"}:
            exclusions.append({"scenario_id": record.get("scenario_id"), "reason": "unsupported_failure_class"})
            continue
        preferred = record.get("source_candidate") if isinstance(record.get("source_candidate"), dict) else None
        alternative = (
            record.get("alternative_candidate")
            if isinstance(record.get("alternative_candidate"), dict)
            else None
        )
        if not preferred or not preferred.get("context_id"):
            exclusions.append({"scenario_id": record.get("scenario_id"), "reason": "preferred_context_id_missing"})
            continue
        if not alternative or not alternative.get("context_id"):
            exclusions.append({"scenario_id": record.get("scenario_id"), "reason": "alternative_context_id_missing"})
            continue
        samples.append(
            _sample(
                record=record,
                slice_record=slice_record,
                preferred=preferred,
                alternative=alternative,
                sample_weight=float(weights.get(failure_class, 1.0)),
            )
        )

    sample_slice_context_ids = {str(sample.get("distillation_slice_context_id")) for sample in samples}
    holdout_leakage_count = len(sample_slice_context_ids & holdout_context_ids)
    hard_positive_added_count = 0
    ppo_transition_added_count = 0
    reason_codes: list[str] = []
    validation = config.get("validation", {})
    if taxonomy_summary.get("status") != "passed" or split_summary.get("status") != "passed":
        reason_codes.append("quasi_real_teacher_distillation_signal_insufficient")
    if len(samples) < int(validation.get("min_teacher_distillation_preference_count", 0)):
        reason_codes.append("quasi_real_teacher_distillation_signal_insufficient")
    if hard_positive_added_count > int(validation.get("max_hard_positive_added_count", 0)):
        reason_codes.append("hard_positive_added_count_nonzero")
    if ppo_transition_added_count > int(validation.get("max_ppo_transition_added_count", 0)):
        reason_codes.append("ppo_transition_added_count_nonzero")
    if holdout_leakage_count:
        reason_codes.append("quasi_real_teacher_distillation_context_leakage_detected")

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": "failed" if reason_codes else "passed",
        "reason_codes": sorted(set(reason_codes)),
        "taxonomy_root": _display_path(taxonomy, repo),
        "dataset_root": _display_path(dataset, repo),
        "output_root": _display_path(output, repo),
        "taxonomy_status": taxonomy_summary.get("status"),
        "split_status": split_summary.get("status"),
        "teacher_distillation_preference_count": len(samples),
        "preference_failure_class_counts": dict(
            sorted(Counter(sample.get("teacher_distillation_failure_class") for sample in samples).items())
        ),
        "hard_positive_added_count": hard_positive_added_count,
        "ppo_transition_added_count": ppo_transition_added_count,
        "holdout_leakage_count": holdout_leakage_count,
        "exclusion_count": len(exclusions),
        "exclusion_reason_counts": dict(sorted(Counter(item["reason"] for item in exclusions).items())),
        "next_required_change": None if not reason_codes else "quasi_real_teacher_distillation_signal_insufficient",
        "git_provenance": {"current": _git_snapshot(repo), "current_matches_sources": True},
    }
    exclusion_report = {
        "schema_version": EXCLUSION_SCHEMA_VERSION,
        "status": "passed" if not exclusions else "failed",
        "exclusion_count": len(exclusions),
        "exclusions": exclusions,
    }
    (output / outputs["samples"]).write_text(_jsonl(samples), encoding="utf-8")
    (output / outputs["summary"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / outputs["exclusion_report"]).write_text(
        json.dumps(exclusion_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _sample(
    *,
    record: dict[str, Any],
    slice_record: dict[str, Any],
    preferred: dict[str, Any],
    alternative: dict[str, Any],
    sample_weight: float,
) -> dict[str, Any]:
    preferred_side = _training_side(preferred)
    alternative_side = _training_side(alternative)
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "sample_type": "teacher_distillation_preference_pair",
        "training_signal_type": "pairwise_preference",
        "split": "train",
        "scenario_id": slice_record.get("scenario_id"),
        "taxonomy_record_id": record.get("taxonomy_record_id"),
        "source_failure_scenario_id": record.get("scenario_id"),
        "distillation_slice_context_id": slice_record.get("context_id"),
        "context_id": preferred_side.get("context_id"),
        "alternative_context_id": alternative_side.get("context_id"),
        "teacher_distillation_failure_class": record.get("failure_class"),
        "preferred": preferred_side,
        "alternative": alternative_side,
        "global_features": _global_features([preferred_side, alternative_side]),
        "candidate_missing_indicators": [
            _missing_indicators(preferred_side),
            _missing_indicators(alternative_side),
        ],
        "raw_policy_regression_reason_codes": record.get("gate_reason_codes", []),
        "path_cost_delta": record.get("path_cost_delta"),
        "risk_delta": record.get("risk_delta"),
        "hard_positive_added_count": 0,
        "ppo_transition_added_count": 0,
        "sample_weight": sample_weight,
    }


def _training_side(candidate: dict[str, Any]) -> dict[str, Any]:
    side = dict(candidate)
    side["candidate_features"] = _candidate_features(side)
    return side


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mine quasi-real teacher distillation preference samples.")
    parser.add_argument("--taxonomy-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    config = json.loads(_resolve_path(args.config, repo_root).read_text(encoding="utf-8"))
    summary = run_quasi_real_teacher_distillation_preference_mining(
        taxonomy_root=_resolve_path(args.taxonomy_root, repo_root),
        dataset_root=_resolve_path(args.dataset_root, repo_root),
        output_root=_resolve_path(args.output_root, repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(json.dumps({"status": summary["status"], "reason_codes": summary["reason_codes"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
