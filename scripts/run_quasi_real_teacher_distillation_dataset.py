from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot


SUMMARY_SCHEMA_VERSION = "quasi-real-teacher-distillation-split-summary/v1"
SLICE_SCHEMA_VERSION = "quasi-real-teacher-distillation-slice/v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_quasi_real_teacher_distillation_dataset_v1"


def run_quasi_real_teacher_distillation_dataset(
    *,
    taxonomy_root: str | Path,
    output_root: str | Path,
    config: dict[str, Any],
    repo_root: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    taxonomy = Path(taxonomy_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = config["input_files"]
    outputs = config["output_files"]
    taxonomy_summary = _load_json(taxonomy / inputs["taxonomy_summary"])
    records = [
        record
        for record in _load_jsonl(taxonomy / inputs["taxonomy"])
        if record.get("failure_class") in {"path_cost_only_regression", "path_risk_joint_regression", "risk_only_regression"}
    ]
    variants = config.get("variants", {"train": 2, "validation": 1, "holdout": 1})
    slices: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        for split, count in variants.items():
            for variant_index in range(int(count)):
                slices.append(_variant_slice(record, index, str(split), variant_index))
    overlap = _split_overlap_summary(slices)
    reason_codes: list[str] = []
    if taxonomy_summary.get("status") != "passed" or not records:
        reason_codes.append("quasi_real_teacher_distillation_signal_insufficient")
    for split in ("train", "validation", "holdout"):
        if sum(1 for item in slices if item.get("split") == split) <= 0:
            reason_codes.append("quasi_real_teacher_distillation_signal_insufficient")
    if any(overlap.values()):
        reason_codes.append("quasi_real_teacher_distillation_context_leakage_detected")

    path_feedback = {
        "schema_version": "quasi-real-teacher-distillation-path-feedback-summary/v1",
        "status": "passed" if not reason_codes else "failed",
        "scenario_count": len(slices),
        "scenarios": [_scenario_from_slice(item, records_by_scenario=records) for item in slices],
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": "failed" if reason_codes else "passed",
        "reason_codes": sorted(set(reason_codes)),
        "taxonomy_root": _display_path(taxonomy, repo),
        "output_root": _display_path(output, repo),
        "source_taxonomy_status": taxonomy_summary.get("status"),
        "slice_count": len(slices),
        "train_slice_count": sum(1 for item in slices if item.get("split") == "train"),
        "validation_slice_count": sum(1 for item in slices if item.get("split") == "validation"),
        "holdout_slice_count": sum(1 for item in slices if item.get("split") == "holdout"),
        **overlap,
        "next_required_change": None if not reason_codes else "quasi_real_teacher_distillation_context_leakage_detected",
        "git_provenance": {"current": _git_snapshot(repo), "current_matches_sources": True},
    }
    (output / outputs["slices"]).write_text(_jsonl(slices), encoding="utf-8")
    (output / outputs["path_feedback_summary"]).write_text(
        json.dumps(path_feedback, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / outputs["split_summary"]).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _variant_slice(record: dict[str, Any], record_index: int, split: str, variant_index: int) -> dict[str, Any]:
    taxonomy_record_id = str(record.get("taxonomy_record_id") or f"record-{record_index}")
    base = f"{taxonomy_record_id}:{split}:{variant_index}"
    scenario_id = f"qreal_teacher_distill_{record_index:03d}_{split}_{variant_index:02d}"
    return {
        "schema_version": SLICE_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "scenario_group": record.get("roi_group", "unknown"),
        "scenario_seed": 20260612 + record_index * 100 + variant_index,
        "scenario_variant_id": f"{scenario_id}-variant",
        "dataset_id": "lunar_south_pole_lro_lola_gdr_875s_20m",
        "data_class": "quasi_real_teacher_distillation",
        "map_id": record.get("map_id"),
        "slice_id": f"{scenario_id}-slice",
        "roi_name": record.get("roi_name") or record.get("roi_group"),
        "split": split,
        "context_id": _stable_id("context", base),
        "taxonomy_record_id": taxonomy_record_id,
        "source_failure_scenario_id": record.get("scenario_id"),
        "source_failure_context_id": record.get("context_id"),
        "source_failure_slice_id": record.get("slice_id"),
        "failure_class": record.get("failure_class"),
        "path_cost_delta": record.get("path_cost_delta"),
        "risk_delta": record.get("risk_delta"),
    }


def _scenario_from_slice(slice_record: dict[str, Any], *, records_by_scenario: list[dict[str, Any]]) -> dict[str, Any]:
    record = next(
        (
            item
            for item in records_by_scenario
            if item.get("taxonomy_record_id") == slice_record.get("taxonomy_record_id")
        ),
        {},
    )
    if not record:
        record = next(
            (
                item
                for item in records_by_scenario
                if item.get("scenario_id") == slice_record.get("source_failure_scenario_id")
            ),
            {},
        )
    source = dict(record.get("source_candidate") or {})
    alternative = dict(record.get("alternative_candidate") or {})
    return {
        "scenario_id": slice_record["scenario_id"],
        "scenario_group": slice_record["scenario_group"],
        "scenario_seed": slice_record["scenario_seed"],
        "scenario_variant_id": slice_record["scenario_variant_id"],
        "path_feedback": {"candidates": [source, alternative]},
    }


def _split_overlap_summary(slices: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for key in ("context_id", "scenario_id", "slice_id"):
        by_split = {
            split: {str(item.get(key)) for item in slices if item.get("split") == split}
            for split in ("train", "validation", "holdout")
        }
        overlap = (
            (by_split["train"] & by_split["validation"])
            | (by_split["train"] & by_split["holdout"])
            | (by_split["validation"] & by_split["holdout"])
        )
        result[f"{key}_overlap_count"] = len(overlap)
    return result


def _stable_id(prefix: str, value: str) -> str:
    return hashlib.sha256(f"{prefix}:{value}".encode("utf-8")).hexdigest()


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
    parser = argparse.ArgumentParser(description="Generate quasi-real teacher distillation train/validation/holdout slices.")
    parser.add_argument("--taxonomy-root", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    config = json.loads(_resolve_path(args.config, repo_root).read_text(encoding="utf-8"))
    summary = run_quasi_real_teacher_distillation_dataset(
        taxonomy_root=_resolve_path(args.taxonomy_root, repo_root),
        output_root=_resolve_path(args.output_root, repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(json.dumps({"status": summary["status"], "reason_codes": summary["reason_codes"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
