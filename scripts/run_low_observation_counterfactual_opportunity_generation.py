from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot


SUMMARY_SCHEMA_VERSION = "low-observation-counterfactual-opportunity-summary/v1"
GAP_SCHEMA_VERSION = "low-observation-counterfactual-opportunity-family-gap-report/v1"
DEFAULT_CONFIG = "configs/quasi_real_low_observation_counterfactual_opportunity_v1.json"
DEFAULT_STAGE5A2_ROOT = "outputs/path_feedback_batch_policy_differentiating_counterfactual_coverage_rollouts_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_low_observation_counterfactual_opportunity_v1"
STAGE5A2_OVERLAY_FILE = "candidate-level-coverage-overlay.jsonl"
STAGE5A2_COUNTERFACTUAL_FILE = "counterfactual-coverage-rollouts.jsonl"
SUMMARY_FILE = "low-observation-counterfactual-opportunity-summary.json"
SUPPLEMENTAL_OVERLAY_FILE = "low-observation-supplemental-overlay.jsonl"
SUPPLEMENTAL_COUNTERFACTUAL_FILE = "low-observation-supplemental-counterfactual-rollouts.jsonl"
FAMILY_GAP_FILE = "low-observation-family-gap-report.json"
REPORT_FILE = "low-observation-counterfactual-opportunity-report.md"
TARGET_FAMILY = "low_observation_count"
MIN_TRAINABLE_SAFE_BETTER_PAIR_COUNT = 8
TOLERANCE = 1.0e-9


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate supplemental low-observation counterfactual coverage opportunities."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage5a2-root", default=DEFAULT_STAGE5A2_ROOT)
    parser.add_argument("--source-overlay", action="append", default=[])
    parser.add_argument("--source-counterfactual-rollouts", action="append", default=[])
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    stage5a2_root = _resolve_path(Path(args.stage5a2_root), repo_root)
    source_overlay_paths = [
        _resolve_path(Path(path), repo_root) for path in args.source_overlay
    ] or [stage5a2_root / STAGE5A2_OVERLAY_FILE]
    source_counterfactual_paths = [
        _resolve_path(Path(path), repo_root) for path in args.source_counterfactual_rollouts
    ] or [stage5a2_root / STAGE5A2_COUNTERFACTUAL_FILE]
    summary = run_low_observation_counterfactual_opportunity_generation(
        config_path=_resolve_path(Path(args.config), repo_root),
        source_overlay_paths=source_overlay_paths,
        source_counterfactual_paths=source_counterfactual_paths,
        output_root=_resolve_path(Path(args.output_root), repo_root),
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "low_observation_trainable_safe_better_pair_count": summary[
                    "low_observation_trainable_safe_better_pair_count"
                ],
                "missing_counterfactual_source_count": summary["missing_counterfactual_source_count"],
                "fallback_gain_contamination_count": summary["fallback_gain_contamination_count"],
                "performance_claimed": summary["performance_claimed"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_low_observation_counterfactual_opportunity_generation(
    *,
    config_path: Path,
    source_overlay_paths: list[Path],
    source_counterfactual_paths: list[Path],
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)
    input_reasons: list[str] = []
    config = _read_json(config_path, input_reasons, "low_observation_config_missing")
    target_family = str(config.get("target_family") or TARGET_FAMILY)
    min_trainable = _int(
        config.get("minimum_trainable_safe_better_pair_count"),
        default=MIN_TRAINABLE_SAFE_BETTER_PAIR_COUNT,
    )
    source_overlay_paths = [_resolve_path(Path(path), repo_root) for path in source_overlay_paths]
    source_counterfactual_paths = [
        _resolve_path(Path(path), repo_root) for path in source_counterfactual_paths
    ]
    source_overlay_rows = _read_jsonl_many(
        source_overlay_paths,
        input_reasons,
        "source_candidate_coverage_overlay_missing",
    )
    source_counterfactual_rows = _read_jsonl_many(
        source_counterfactual_paths,
        input_reasons,
        "source_counterfactual_coverage_rollouts_missing",
    )

    audit = _audit_low_observation_candidates(
        overlay_rows=source_overlay_rows,
        counterfactual_rows=source_counterfactual_rows,
        target_family=target_family,
    )
    reason_codes = _reason_codes(
        input_reasons=input_reasons,
        audit=audit,
        min_trainable=min_trainable,
    )
    status = "passed" if not reason_codes else "failed"
    next_required_change = (
        "rerun_safe_better_pair_expansion_across_families_with_supplemental_overlay"
        if status == "passed"
        else "generate_more_low_observation_counterfactual_coverage_candidates"
    )
    family_gap_report = _family_gap_report(audit=audit, target_family=target_family, min_trainable=min_trainable)
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": next_required_change,
        "target_family": target_family,
        "config_path": str(config_path),
        "output_root": str(output_root),
        "input_artifact_paths": {
            "source_candidate_coverage_overlays": [str(path) for path in source_overlay_paths],
            "source_counterfactual_coverage_rollouts": [
                str(path) for path in source_counterfactual_paths
            ],
        },
        "produced_artifacts": {
            "supplemental_overlay": str(paths["supplemental_overlay"]),
            "supplemental_counterfactual_rollouts": str(paths["supplemental_counterfactual"]),
            "family_gap_report": str(paths["family_gap"]),
            "report": str(paths["report"]),
        },
        "source_overlay_row_count": len(source_overlay_rows),
        "source_counterfactual_row_count": len(source_counterfactual_rows),
        "low_observation_candidate_count": audit["candidate_count"],
        "low_observation_source_available_count": audit["source_available_count"],
        "low_observation_trainable_safe_better_pair_count": audit["trainable_pair_count"],
        "low_observation_diagnostic_safe_better_pair_count": audit["diagnostic_pair_count"],
        "minimum_trainable_safe_better_pair_count": min_trainable,
        "missing_counterfactual_source_count": audit["missing_counterfactual_source_count"],
        "fallback_gain_contamination_count": audit["fallback_gain_contamination_count"],
        "controlled_regression_count": audit["controlled_regression_count"],
        "guard_rejected_candidate_count": audit["guard_rejected_candidate_count"],
        "non_positive_coverage_advantage_count": audit["non_positive_coverage_advantage_count"],
        "split_safe_better_counts": audit["split_counts"],
        "counterfactual_match_method_counts": audit["match_method_counts"],
        "uses_existing_source_backed_counterfactual_rollouts": True,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "modifies_network_or_action_space": False,
        "modifies_default_astar": False,
        "relaxes_guard": False,
        "performance_claimed": False,
        "formal_release_claimed": False,
        "git_provenance": git_snapshot(repo_root),
        "summary": _summary_sentence(status=status, audit=audit, reason_codes=reason_codes),
    }

    _write_jsonl(paths["supplemental_overlay"], audit["supplemental_overlay_rows"])
    _write_jsonl(paths["supplemental_counterfactual"], audit["supplemental_counterfactual_rows"])
    _write_json(paths["family_gap"], family_gap_report)
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, family_gap_report), encoding="utf-8")
    return summary


def _audit_low_observation_candidates(
    *,
    overlay_rows: list[dict[str, Any]],
    counterfactual_rows: list[dict[str, Any]],
    target_family: str,
) -> dict[str, Any]:
    decisions: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in overlay_rows:
        if row.get("scenario_family") == target_family:
            decisions[_decision_key(row)].append(row)
    counterfactual_index = {
        (*_decision_key(row), row.get("action_index")): row
        for row in counterfactual_rows
        if row.get("scenario_family") == target_family and row.get("action_index") is not None
    }

    candidate_count = 0
    source_available_count = 0
    missing_counterfactual_source_count = 0
    fallback_gain_contamination_count = 0
    controlled_regression_count = 0
    guard_rejected_candidate_count = 0
    non_positive_coverage_advantage_count = 0
    split_counts: Counter[str] = Counter()
    match_method_counts: Counter[str] = Counter()
    supplemental_overlay_rows: list[dict[str, Any]] = []
    supplemental_counterfactual_rows: list[dict[str, Any]] = []

    for key, rows in sorted(decisions.items(), key=lambda item: tuple(str(part) for part in item[0])):
        teacher = next((row for row in rows if bool(row.get("is_teacher_action"))), None)
        if teacher is None:
            continue
        for candidate in rows:
            if bool(candidate.get("is_teacher_action")):
                continue
            candidate_count += 1
            counterfactual = counterfactual_index.get((*key, candidate.get("action_index")))
            source_available = bool(candidate.get("coverage_source_available")) or bool(
                (counterfactual or {}).get("coverage_source_available")
            )
            if source_available:
                source_available_count += 1
            coverage_advantage = _coverage_value(candidate) - _coverage_value(teacher)
            if coverage_advantage <= TOLERANCE:
                non_positive_coverage_advantage_count += 1
                continue
            if _is_fallback_like(candidate):
                fallback_gain_contamination_count += 1
                continue
            if bool(candidate.get("guard_rejected")) or not bool(candidate.get("action_mask_valid", True)):
                guard_rejected_candidate_count += 1
                continue
            if _controlled_regression_reasons(candidate):
                controlled_regression_count += 1
                continue
            if not source_available:
                missing_counterfactual_source_count += 1
                continue

            split = str(candidate.get("split") or teacher.get("split") or "unknown")
            split_counts[split] += 1
            match_method = str(candidate.get("match_method") or (counterfactual or {}).get("match_method"))
            match_method_counts[match_method] += 1
            supplemental_overlay_rows.extend([dict(teacher), _supplemental_candidate_row(candidate, teacher)])
            supplemental_counterfactual_rows.append(
                _supplemental_counterfactual_row(
                    candidate=candidate,
                    teacher=teacher,
                    counterfactual=counterfactual,
                    coverage_advantage=coverage_advantage,
                )
            )

    supplemental_overlay_rows, _ = _dedupe_rows(
        supplemental_overlay_rows,
        key_fields=("context_id", "episode_id", "step_index", "scenario_family", "split", "action_index"),
    )
    supplemental_counterfactual_rows, _ = _dedupe_rows(
        supplemental_counterfactual_rows,
        key_fields=("context_id", "episode_id", "step_index", "scenario_family", "split", "action_index"),
    )
    trainable_pair_count = int(split_counts.get("train", 0))
    diagnostic_pair_count = sum(count for split, count in split_counts.items() if split != "train")
    return {
        "candidate_count": candidate_count,
        "source_available_count": source_available_count,
        "trainable_pair_count": trainable_pair_count,
        "diagnostic_pair_count": diagnostic_pair_count,
        "missing_counterfactual_source_count": missing_counterfactual_source_count,
        "fallback_gain_contamination_count": fallback_gain_contamination_count,
        "controlled_regression_count": controlled_regression_count,
        "guard_rejected_candidate_count": guard_rejected_candidate_count,
        "non_positive_coverage_advantage_count": non_positive_coverage_advantage_count,
        "split_counts": dict(sorted(split_counts.items())),
        "match_method_counts": dict(sorted(match_method_counts.items())),
        "supplemental_overlay_rows": supplemental_overlay_rows,
        "supplemental_counterfactual_rows": supplemental_counterfactual_rows,
    }


def _supplemental_candidate_row(candidate: dict[str, Any], teacher: dict[str, Any]) -> dict[str, Any]:
    row = dict(candidate)
    row.update(
        {
            "safe_better_than_teacher_candidate": True,
            "safe_better_basis": "low_observation_counterfactual_coverage_advantage_positive",
            "teacher_action_index": teacher.get("action_index", candidate.get("teacher_action_index")),
            "coverage_advantage": _coverage_value(candidate) - _coverage_value(teacher),
            "teacher_expected_coverage_rate_delta": _coverage_value(teacher),
            "supplemental_opportunity_source": "low_observation_counterfactual_opportunity_generation_v1",
        }
    )
    return row


def _supplemental_counterfactual_row(
    *,
    candidate: dict[str, Any],
    teacher: dict[str, Any],
    counterfactual: dict[str, Any] | None,
    coverage_advantage: float,
) -> dict[str, Any]:
    row = dict(counterfactual or candidate)
    row.update(
        {
            "schema_version": "low-observation-supplemental-counterfactual-coverage-row/v1",
            "context_id": candidate.get("context_id"),
            "episode_id": candidate.get("episode_id"),
            "step_index": candidate.get("step_index"),
            "scenario_id": candidate.get("scenario_id") or teacher.get("scenario_id"),
            "scenario_family": candidate.get("scenario_family"),
            "split": candidate.get("split") or teacher.get("split"),
            "action_index": candidate.get("action_index"),
            "teacher_action_index": teacher.get("action_index", candidate.get("teacher_action_index")),
            "coverage_source_available": True,
            "safe_better_than_teacher_candidate": True,
            "coverage_advantage": float(coverage_advantage),
            "teacher_expected_coverage_rate_delta": _coverage_value(teacher),
            "teacher_source_path": teacher.get("source_path"),
            "source_path": candidate.get("source_path") or (counterfactual or {}).get("source_path"),
            "match_method": candidate.get("match_method") or (counterfactual or {}).get("match_method"),
            "supplemental_opportunity_source": "low_observation_counterfactual_opportunity_generation_v1",
        }
    )
    return row


def _reason_codes(
    *,
    input_reasons: list[str],
    audit: dict[str, Any],
    min_trainable: int,
) -> list[str]:
    reasons: list[str] = []
    for reason in input_reasons:
        _add_reason(reasons, reason)
    if audit["candidate_count"] <= 0:
        _add_reason(reasons, "low_observation_candidate_missing")
    if audit["trainable_pair_count"] < min_trainable:
        _add_reason(reasons, "low_observation_safe_better_pair_count_below_threshold")
    if audit["trainable_pair_count"] + audit["diagnostic_pair_count"] <= 0:
        _add_reason(reasons, "supplemental_overlay_no_positive_coverage_advantage")
    if audit["missing_counterfactual_source_count"] > 0:
        _add_reason(reasons, "low_observation_counterfactual_source_missing")
    if audit["fallback_gain_contamination_count"] > 0:
        _add_reason(reasons, "fallback_gain_contamination")
    if audit["controlled_regression_count"] > 0:
        _add_reason(reasons, "controlled_regression")
    return reasons


def _family_gap_report(
    *,
    audit: dict[str, Any],
    target_family: str,
    min_trainable: int,
) -> dict[str, Any]:
    trainable_count = int(audit["trainable_pair_count"])
    return {
        "schema_version": GAP_SCHEMA_VERSION,
        "target_family": target_family,
        "minimum_trainable_safe_better_pair_count": min_trainable,
        "rows": [
            {
                "scenario_family": target_family,
                "trainable_safe_better_pair_count": trainable_count,
                "diagnostic_safe_better_pair_count": int(audit["diagnostic_pair_count"]),
                "family_gate_passed": trainable_count >= min_trainable,
                "reason_codes": []
                if trainable_count >= min_trainable
                else ["low_observation_safe_better_pair_count_below_threshold"],
            }
        ],
    }


def _render_report(summary: dict[str, Any], family_gap_report: dict[str, Any]) -> str:
    row = family_gap_report["rows"][0]
    return "\n".join(
        [
            "# Low-Observation Counterfactual Opportunity Generation v1",
            "",
            "## Summary",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- low_observation_candidate_count: `{summary['low_observation_candidate_count']}`",
            "- low_observation_trainable_safe_better_pair_count: "
            f"`{summary['low_observation_trainable_safe_better_pair_count']}`",
            f"- missing_counterfactual_source_count: `{summary['missing_counterfactual_source_count']}`",
            f"- fallback_gain_contamination_count: `{summary['fallback_gain_contamination_count']}`",
            f"- controlled_regression_count: `{summary['controlled_regression_count']}`",
            f"- performance_claimed: `{summary['performance_claimed']}`",
            "",
            "## Family Gate",
            "",
            "| Family | Trainable safe-better | Diagnostic safe-better | Gate |",
            "| --- | ---: | ---: | --- |",
            "| {family} | {trainable} | {diagnostic} | {gate} |".format(
                family=row["scenario_family"],
                trainable=row["trainable_safe_better_pair_count"],
                diagnostic=row["diagnostic_safe_better_pair_count"],
                gate="passed" if row["family_gate_passed"] else "failed",
            ),
            "",
            "## Non-Goals",
            "",
            "- no PPO update",
            "- no checkpoint publication",
            "- no default policy replacement",
            "- no real executor connection",
            "- no network/action-space/default-A* change",
            "- no guard relaxation",
            "- no performance claim",
            "",
        ]
    )


def _summary_sentence(*, status: str, audit: dict[str, Any], reason_codes: list[str]) -> str:
    if status == "passed":
        return (
            "Low-observation opportunity generation passed with "
            f"{audit['trainable_pair_count']} trainable safe-better pairs."
        )
    return (
        "Low-observation opportunity generation failed because "
        f"{reason_codes}; current trainable low-observation pairs={audit['trainable_pair_count']}."
    )


def _decision_key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (row.get("context_id"), row.get("episode_id"), row.get("step_index"))


def _coverage_value(row: dict[str, Any]) -> float:
    return _float(row.get("expected_coverage_rate_delta"))


def _is_fallback_like(row: dict[str, Any]) -> bool:
    return bool(row.get("fallback_like")) or str(row.get("source_execution_type") or "").startswith("fallback")


def _controlled_regression_reasons(row: dict[str, Any]) -> list[Any]:
    reasons = row.get("controlled_regression_reason_codes")
    return reasons if isinstance(reasons, list) else []


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "supplemental_overlay": output_root / SUPPLEMENTAL_OVERLAY_FILE,
        "supplemental_counterfactual": output_root / SUPPLEMENTAL_COUNTERFACTUAL_FILE,
        "family_gap": output_root / FAMILY_GAP_FILE,
        "report": output_root / REPORT_FILE,
    }


def _read_json(path: Path, reasons: list[str], missing_reason: str) -> dict[str, Any]:
    if not path.is_file():
        _add_reason(reasons, missing_reason)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _add_reason(reasons, f"{missing_reason}_invalid_json")
        return {}


def _read_jsonl_many(
    paths: list[Path],
    reasons: list[str],
    missing_reason: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_jsonl(path, reasons, missing_reason))
    return rows


def _read_jsonl(path: Path, reasons: list[str], missing_reason: str) -> list[dict[str, Any]]:
    if not path.is_file():
        _add_reason(reasons, missing_reason)
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                _add_reason(reasons, f"{missing_reason}_invalid_jsonl")
                break
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _dedupe_rows(
    rows: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
) -> tuple[list[dict[str, Any]], int]:
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        key = tuple(row.get(field) for field in key_fields)
        if all(value is None for value in key):
            key = ("__row_index__", index)
        indexed[key] = row
    return list(indexed.values()), len(rows) - len(indexed)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _float(value: Any, *, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, *, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
