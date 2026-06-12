from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot
from run_scenario_disjoint_policy_rollout_evaluation import _display_path


CONFIG_SCHEMA_VERSION = "quasi-real-safe-better-opportunity-expansion-config/v1"
SUMMARY_SCHEMA_VERSION = "quasi-real-safe-better-opportunity-expansion-summary/v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1"


class ConfigError(ValueError):
    pass


def run_quasi_real_safe_better_opportunity_expansion(
    *,
    matrix_manifest_path: str | Path,
    output_root: str | Path,
    config: dict[str, Any],
    repo_root: str | Path,
    diagnosis_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    _ensure_model_explorer_path(repo)
    from model_explorer.data.evaluation_matrix import load_quasi_real_evaluation_manifest

    source_manifest_path = _resolve(matrix_manifest_path, repo)
    source_manifest = load_quasi_real_evaluation_manifest(source_manifest_path)
    output = _resolve(output_root, repo)
    output.mkdir(parents=True, exist_ok=True)
    output_paths = _output_paths(output, config, repo_root=repo)

    expansion_rois = _expanded_rois(source_manifest.rois, config=config)
    matrix_payload = _matrix_payload(
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        expansion_rois=expansion_rois,
        config=config,
    )
    output_paths["matrix_manifest"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["matrix_manifest"].write_text(
        json.dumps(matrix_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    diagnosis = _load_json(diagnosis_summary_path) if diagnosis_summary_path else {}
    summary = _summary(
        source_manifest_path=source_manifest_path,
        matrix_manifest_path=output_paths["matrix_manifest"],
        output_root=output,
        expansion_rois=expansion_rois,
        diagnosis=diagnosis,
        config=config,
        repo_root=repo,
    )
    output_paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    output_paths["report"].write_text(_report(summary), encoding="utf-8")
    summary["summary_output"] = str(output_paths["summary"])
    return summary


def _expanded_rois(rois: tuple[Any, ...], *, config: dict[str, Any]) -> list[dict[str, Any]]:
    offsets = [_cell_tuple(item) for item in config.get("roi_offsets", [[0, 0]])]
    starts = [_cell_tuple(item) for item in config.get("start_cells", [[0, 0]])]
    starts = [item for item in starts if item is not None]
    offsets = [item for item in offsets if item is not None]
    if not starts:
        raise ConfigError("start_cells must contain at least one [x, y] cell")
    if not offsets:
        raise ConfigError("roi_offsets must contain at least one [x, y] offset")

    candidate_count = int(config.get("candidate_count", 6))
    expanded: list[dict[str, Any]] = []
    for roi_index, roi in enumerate(rois):
        width = int(roi.roi_width)
        height = int(roi.roi_height)
        valid_starts = [
            start
            for start in starts
            if 0 <= start[0] < width and 0 <= start[1] < height
        ]
        if not valid_starts:
            valid_starts = [(0, 0)]
        for start_index, start in enumerate(valid_starts):
            offset = offsets[start_index % len(offsets)]
            roi_x = max(0, int(roi.roi_x) + offset[0])
            roi_y = max(0, int(roi.roi_y) + offset[1])
            split = str(roi.split)
            scenario_id = (
                f"qreal_safe_better_{roi.name}_{split}_"
                f"r{roi_index:03d}_s{start_index:02d}"
            )
            seed = int(roi.seed) + start_index * 101
            expanded.append(
                {
                    "name": str(roi.name),
                    "split": split,
                    "roi_x": roi_x,
                    "roi_y": roi_y,
                    "roi_width": width,
                    "roi_height": height,
                    "candidate_count": candidate_count,
                    "episode_count": 1,
                    "seed": seed,
                    "start_cell": [start[0], start[1]],
                    "scenario_id": scenario_id,
                    "scenario_variant_id": f"{scenario_id}-seed-{seed}-start-{start[0]}-{start[1]}",
                    "context_id": _context_id(
                        roi_name=str(roi.name),
                        split=split,
                        roi_x=roi_x,
                        roi_y=roi_y,
                        roi_width=width,
                        roi_height=height,
                        seed=seed,
                        start_cell=start,
                    ),
                }
            )
    return expanded


def _matrix_payload(
    *,
    source_manifest_path: Path,
    source_manifest: Any,
    expansion_rois: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "model-explorer-quasi-real-evaluation/v1",
        "name": str(config.get("matrix_name", "qreal-safe-better-opportunity-v1")),
        "run_id": str(config.get("run_id", "safe-better-opportunity-v1")),
        "dataset_manifest": str(source_manifest.dataset_manifest),
        "output_root": str(config.get("processed_output_root", "../processed/qreal_safe_better_opportunity_v1")),
        "candidate_count": int(config.get("candidate_count", 6)),
        "episode_count": 1,
        "seed": int(config.get("seed", 20260612)),
        "source_matrix_manifest": str(source_manifest_path),
        "rois": [
            {
                key: value
                for key, value in roi.items()
                if key not in {"context_id"}
            }
            for roi in expansion_rois
        ],
        "mask_stress": {"enabled": False},
        "selection": {"enabled": False},
        "dataset_validation": dict(config.get("dataset_validation", {})),
        "train": dict(config.get("train", {})),
    }


def _summary(
    *,
    source_manifest_path: Path,
    matrix_manifest_path: Path,
    output_root: Path,
    expansion_rois: list[dict[str, Any]],
    diagnosis: dict[str, Any],
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    validation = config.get("validation", {})
    reason_codes: list[str] = []
    groups = sorted({str(roi["name"]) for roi in expansion_rois})
    split_context_ids: dict[str, set[str]] = defaultdict(set)
    scenario_ids: set[str] = set()
    duplicated_context_ids = _duplicate_count([str(roi["context_id"]) for roi in expansion_rois])
    duplicated_scenario_ids = _duplicate_count([str(roi["scenario_id"]) for roi in expansion_rois])
    start_cell_missing_count = sum(1 for roi in expansion_rois if not roi.get("start_cell"))
    context_id_missing_count = sum(1 for roi in expansion_rois if not roi.get("context_id"))
    for roi in expansion_rois:
        split_context_ids[str(roi["split"])].add(str(roi["context_id"]))
        scenario_ids.add(str(roi["scenario_id"]))
    split_overlap_count = _split_overlap_count(split_context_ids)

    if len(expansion_rois) < _int(validation.get("min_candidate_context_count"), 0):
        _append(reason_codes, "quasi_real_safe_better_candidate_context_count_below_threshold")
    if len(groups) < _int(validation.get("min_roi_group_count"), 0):
        _append(reason_codes, "quasi_real_safe_better_roi_group_count_below_threshold")
    if start_cell_missing_count > _int(validation.get("max_start_cell_missing_count"), 0):
        _append(reason_codes, "quasi_real_safe_better_start_cell_missing")
    if context_id_missing_count > _int(validation.get("max_context_id_missing_count"), 0):
        _append(reason_codes, "quasi_real_safe_better_context_id_missing")
    if duplicated_context_ids or split_overlap_count:
        _append(reason_codes, "quasi_real_safe_better_context_id_overlap")
    if duplicated_scenario_ids:
        _append(reason_codes, "quasi_real_safe_better_scenario_id_overlap")

    diagnosis_counts = _diagnosis_counts(diagnosis)
    diagnosis_group_summary = diagnosis.get("roi_group_opportunity_summary")
    diagnosis_group_summary = diagnosis_group_summary if isinstance(diagnosis_group_summary, dict) else {}
    if diagnosis:
        if diagnosis.get("status") != "passed" or diagnosis.get("reason_codes"):
            _append(reason_codes, "quasi_real_safe_better_diagnosis_failed")
        if diagnosis_counts["safe_alternative_context_count"] < _int(validation.get("min_safe_alternative_context_count"), 0):
            _append(reason_codes, "quasi_real_safe_better_safe_alternative_count_below_threshold")
        if diagnosis_counts["safe_better_opportunity_context_count"] < _int(
            validation.get("min_safe_better_opportunity_context_count"),
            0,
        ):
            _append(reason_codes, "quasi_real_safe_better_opportunity_count_below_threshold")
        if diagnosis_counts["roi_group_with_safe_better_opportunity_count"] < _int(
            validation.get("min_roi_group_with_safe_better_opportunity_count"),
            0,
        ):
            _append(reason_codes, "quasi_real_safe_better_roi_group_opportunity_count_below_threshold")

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _now(),
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": reason_codes,
        "source_matrix_manifest": _display_path(source_manifest_path, repo_root),
        "expansion_matrix_manifest": _display_path(matrix_manifest_path, repo_root),
        "output_root": _display_path(output_root, repo_root),
        "candidate_context_count": len(expansion_rois),
        "roi_group_count": len(groups),
        "roi_groups": groups,
        "start_cell_missing_count": start_cell_missing_count,
        "context_id_missing_count": context_id_missing_count,
        "context_id_overlap_count": duplicated_context_ids + split_overlap_count,
        "scenario_id_overlap_count": duplicated_scenario_ids,
        "split_context_counts": {split: len(ids) for split, ids in sorted(split_context_ids.items())},
        "roi_group_context_summary": _roi_group_context_summary(expansion_rois),
        "opportunity_verdict": diagnosis.get("opportunity_verdict") if diagnosis else None,
        "roi_group_opportunity_summary": diagnosis_group_summary,
        **diagnosis_counts,
        "next_required_change": None if not reason_codes else "quasi_real_roi_start_target_expansion_required",
        "runs_ppo_update": False,
        "policy_takes_control": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "git_provenance": {"current": _git_snapshot(repo_root), "current_matches_sources": True},
    }
    if diagnosis and not reason_codes:
        summary["next_required_change"] = None
    elif diagnosis and diagnosis_counts["safe_better_opportunity_context_count"] <= 0:
        summary["next_required_change"] = "quasi_real_roi_start_target_expansion_required"
    return summary


def _diagnosis_counts(diagnosis: dict[str, Any]) -> dict[str, int]:
    group_summary = diagnosis.get("roi_group_opportunity_summary")
    group_summary = group_summary if isinstance(group_summary, dict) else {}
    return {
        "safe_alternative_context_count": _int(diagnosis.get("safe_alternative_context_count")),
        "safe_better_opportunity_context_count": _int(diagnosis.get("safe_better_opportunity_context_count")),
        "safe_better_alternative_count": _int(diagnosis.get("safe_better_alternative_count")),
        "roi_group_with_safe_better_opportunity_count": sum(
            1
            for row in group_summary.values()
            if isinstance(row, dict) and _int(row.get("safe_better_opportunity_context_count")) > 0
        ),
    }


def _roi_group_context_summary(expansion_rois: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"context_count": 0, "start_cells": []})
    for roi in expansion_rois:
        bucket = buckets[str(roi["name"])]
        bucket["context_count"] += 1
        start = roi.get("start_cell")
        if start not in bucket["start_cells"]:
            bucket["start_cells"].append(start)
    return {key: value for key, value in sorted(buckets.items())}


def _output_paths(output: Path, config: dict[str, Any], *, repo_root: Path) -> dict[str, Path]:
    files = config.get("output_files", {})
    matrix_value = files.get(
        "matrix_manifest",
        "model-explorer/data/manifests/lunar_south_pole_lro_lola_safe_better_opportunity_matrix_v1.json",
    )
    matrix_path = _resolve(matrix_value, repo_root)
    return {
        "matrix_manifest": matrix_path,
        "summary": output / str(files.get("summary", "quasi-real-safe-better-opportunity-expansion-summary.json")),
        "report": output / str(files.get("report", "quasi-real-safe-better-opportunity-expansion-report.md")),
    }


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Quasi-Real Safe-Better Opportunity Expansion",
        "",
        f"- status: {summary['status']}",
        f"- candidate_context_count: {summary['candidate_context_count']}",
        f"- safe_better_opportunity_context_count: {summary['safe_better_opportunity_context_count']}",
        f"- next_required_change: {summary['next_required_change']}",
        "",
        "| roi_group | contexts | start cells |",
        "|---|---:|---|",
    ]
    for group, row in summary["roi_group_context_summary"].items():
        lines.append(f"| {group} | {row['context_count']} | {row['start_cells']} |")
    if summary.get("roi_group_opportunity_summary"):
        lines.extend([
            "",
            "| roi_group | safe alternatives | safe-better contexts | safe-better starts | opportunity missing |",
            "|---|---:|---:|---|---:|",
        ])
        for group, row in summary["roi_group_opportunity_summary"].items():
            lines.append(
                f"| {group} | {row.get('safe_alternative_context_count', 0)} | "
                f"{row.get('safe_better_opportunity_context_count', 0)} | "
                f"{row.get('safe_better_start_cells', [])} | "
                f"{row.get('opportunity_missing_count', 0)} |"
            )
    lines.extend(["", "No PPO update, no policy takeover, and no checkpoint publication."])
    return "\n".join(lines)


def _context_id(
    *,
    roi_name: str,
    split: str,
    roi_x: int,
    roi_y: int,
    roi_width: int,
    roi_height: int,
    seed: int,
    start_cell: tuple[int, int],
) -> str:
    payload = {
        "schema_version": "quasi-real-safe-better-opportunity-context/v1",
        "roi_name": roi_name,
        "split": split,
        "roi_x": int(roi_x),
        "roi_y": int(roi_y),
        "roi_width": int(roi_width),
        "roi_height": int(roi_height),
        "seed": int(seed),
        "start_cell": [int(start_cell[0]), int(start_cell[1])],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _cell_tuple(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return (int(value[0]), int(value[1]))


def _duplicate_count(values: list[str]) -> int:
    counts = Counter(values)
    return sum(count - 1 for count in counts.values() if count > 1)


def _split_overlap_count(split_context_ids: dict[str, set[str]]) -> int:
    seen: set[str] = set()
    overlap = 0
    for ids in split_context_ids.values():
        overlap += len(seen.intersection(ids))
        seen.update(ids)
    return overlap


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("config must be a JSON object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    return payload


def _ensure_model_explorer_path(repo_root: Path) -> None:
    import sys

    src = repo_root / "model-explorer" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _resolve(path: str | Path, repo_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _append(values: list[str], reason: str) -> None:
    if reason not in values:
        values.append(reason)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate quasi-real safe-better opportunity matrix variants.")
    parser.add_argument("--matrix-manifest", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", required=True)
    parser.add_argument("--diagnosis-summary")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    config = _load_config(_resolve(args.config, repo_root))
    summary = run_quasi_real_safe_better_opportunity_expansion(
        matrix_manifest_path=args.matrix_manifest,
        output_root=args.output_root,
        config=config,
        repo_root=repo_root,
        diagnosis_summary_path=args.diagnosis_summary,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "candidate_context_count": summary["candidate_context_count"],
                "safe_better_opportunity_context_count": summary[
                    "safe_better_opportunity_context_count"
                ],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary_output"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
