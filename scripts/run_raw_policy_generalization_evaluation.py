from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot

from run_scenario_disjoint_policy_rollout_evaluation import (
    CONFIG_SCHEMA_VERSION as ROLLOUT_CONFIG_SCHEMA_VERSION,
    _collect_holdout_scenarios,
    run_scenario_disjoint_policy_rollout_evaluation,
)


CONFIG_SCHEMA_VERSION = "raw-policy-generalization-evaluation-config/v1"
SUMMARY_SCHEMA_VERSION = "raw-policy-generalization-evaluation-summary/v1"
NEXT_DATA_REQUIRED = "data_volume_insufficient"
NEXT_OBJECTIVE_REQUIRED = "objective_or_sample_weight_refinement_required"
NEXT_FEATURE_REQUIRED = "feature_schema_refinement_required"
NEXT_COVERAGE_REQUIRED = "scenario_distribution_gap_requires_more_holdout_coverage"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline vs aligned raw-policy generalization on DEV/VAL/TEST."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--dev-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--baseline-candidate-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    _install_model_explorer_path(repo_root)
    source_root = _resolve_path(args.source_root, repo_root)
    dev_root = _resolve_path(args.dev_root, repo_root)
    val_root = _resolve_path(args.val_root, repo_root)
    test_root = _resolve_path(args.test_root, repo_root)
    baseline_root = _resolve_path(args.baseline_candidate_root, repo_root)
    candidate_root = _resolve_path(args.candidate_root, repo_root)
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    summary, overfit_report = run_raw_policy_generalization_evaluation(
        source_root=source_root,
        dev_root=dev_root,
        val_root=val_root,
        test_root=test_root,
        baseline_candidate_root=baseline_root,
        candidate_root=candidate_root,
        config=config,
        repo_root=repo_root,
    )
    summary_path = candidate_root / config["output_files"]["summary"]
    overfit_path = candidate_root / config["output_files"]["overfit_report"]
    if not args.validate_only:
        _write_json(summary_path, summary)
        _write_json(overfit_path, overfit_report)
    print(
        json.dumps(
            {
                "status": "config validated" if summary["status"] == "passed" else "validation failed",
                "reason_codes": summary["reason_codes"],
                "test_baseline_raw_policy_regression_count": summary[
                    "test_baseline_raw_policy_regression_count"
                ],
                "test_raw_policy_regression_count": summary["test_raw_policy_regression_count"],
                "test_raw_policy_regression_reduction_rate": summary[
                    "test_raw_policy_regression_reduction_rate"
                ],
                "overfit_gap": summary["overfit_gap"],
                "summary": _display_path(summary_path, repo_root),
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def run_raw_policy_generalization_evaluation(
    *,
    source_root: Path,
    dev_root: Path,
    val_root: Path,
    test_root: Path,
    baseline_candidate_root: Path,
    candidate_root: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reason_codes: list[str] = []
    split_roots = {"dev": dev_root, "val": val_root, "test": test_root}
    context_overlap_count, context_overlap_ids = _context_overlap(split_roots, repo_root)
    if context_overlap_count > _int_value(config["validation"].get("max_context_overlap_count")):
        _append_reason(reason_codes, "train_eval_context_leakage_detected")
    candidate_summary = _load_json(
        candidate_root / "raw-policy-generalization-candidate-summary.json",
        label="raw_policy_generalization_candidate_summary",
        reason_codes=reason_codes,
    )
    if _int_value(candidate_summary.get("leaked_context_id_count")) > _int_value(
        config["validation"].get("max_candidate_leaked_context_id_count")
    ):
        _append_reason(reason_codes, "candidate_leaked_context_id_count_nonzero")

    results: dict[str, dict[str, Any]] = {}
    for split, split_root in split_roots.items():
        baseline = _evaluate(
            source_root=source_root,
            candidate_root=baseline_candidate_root,
            batch_root=split_root,
            config=config,
            repo_root=repo_root,
            candidate_kind="baseline",
            split=split,
            reason_codes=reason_codes,
        )
        aligned = _evaluate(
            source_root=source_root,
            candidate_root=candidate_root,
            batch_root=split_root,
            config=config,
            repo_root=repo_root,
            candidate_kind="aligned",
            split=split,
            reason_codes=reason_codes,
        )
        results[split] = {
            "baseline": baseline,
            "aligned": aligned,
            "raw_policy_regression_reduction_rate": _reduction_rate(
                _int_value(baseline.get("raw_policy_regression_count")),
                _int_value(aligned.get("raw_policy_regression_count")),
            ),
        }

    val_reduction = results["val"]["raw_policy_regression_reduction_rate"]
    test_reduction = results["test"]["raw_policy_regression_reduction_rate"]
    overfit_gap = max(0.0, val_reduction - test_reduction)
    test_aligned = results["test"]["aligned"]
    validation = config["validation"]
    if test_reduction < float(validation.get("min_test_raw_policy_regression_reduction_rate", 0.5)):
        _append_reason(reason_codes, "test_raw_policy_regression_reduction_below_threshold")
    if overfit_gap > float(validation.get("max_overfit_gap", 0.15)):
        _append_reason(reason_codes, "overfit_gap_above_threshold")
    _check_test_gates(test_aligned, validation=validation, reason_codes=reason_codes)

    status = "failed" if reason_codes else "passed"
    next_required_change = _next_required_change(reason_codes)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "source_root": _display_path(source_root, repo_root),
        "dev_root": _display_path(dev_root, repo_root),
        "val_root": _display_path(val_root, repo_root),
        "test_root": _display_path(test_root, repo_root),
        "baseline_candidate_root": _display_path(baseline_candidate_root, repo_root),
        "candidate_root": _display_path(candidate_root, repo_root),
        "context_overlap_count": context_overlap_count,
        "context_overlap_ids": context_overlap_ids[:25],
        "candidate_leaked_context_id_count": _int_value(candidate_summary.get("leaked_context_id_count")),
        "dev_baseline_raw_policy_regression_count": _int_value(
            results["dev"]["baseline"].get("raw_policy_regression_count")
        ),
        "dev_raw_policy_regression_count": _int_value(
            results["dev"]["aligned"].get("raw_policy_regression_count")
        ),
        "val_baseline_raw_policy_regression_count": _int_value(
            results["val"]["baseline"].get("raw_policy_regression_count")
        ),
        "val_raw_policy_regression_count": _int_value(
            results["val"]["aligned"].get("raw_policy_regression_count")
        ),
        "test_baseline_raw_policy_regression_count": _int_value(
            results["test"]["baseline"].get("raw_policy_regression_count")
        ),
        "test_raw_policy_regression_count": _int_value(
            test_aligned.get("raw_policy_regression_count")
        ),
        "test_raw_policy_regression_reduction_rate": test_reduction,
        "test_generalization_passed": status == "passed",
        "overfit_gap": overfit_gap,
        "test_regression_count": _int_value(test_aligned.get("regression_count")),
        "test_invalid_action_mask_count": _int_value(test_aligned.get("invalid_action_mask_count")),
        "test_fallback_or_open_grid_count": _int_value(test_aligned.get("fallback_or_open_grid_count")),
        "test_safety_regression_count": _int_value(test_aligned.get("safety_regression_count")),
        "test_contract_violation_count": _int_value(test_aligned.get("contract_violation_count")),
        "test_path_cost_regression_count": _int_value(test_aligned.get("path_cost_regression_count")),
        "test_risk_regression_count": _int_value(test_aligned.get("risk_regression_count")),
        "test_source_selection_regression_count": _int_value(
            test_aligned.get("source_selection_regression_count")
        ),
        "split_results": results,
        "next_required_change": next_required_change,
        "experimental_checkpoint": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": _git_snapshot(repo_root), "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }
    overfit_report = {
        "schema_version": "raw-policy-generalization-evaluation-overfit-report/v1",
        "status": status,
        "reason_codes": reason_codes,
        "dev_reduction_rate": results["dev"]["raw_policy_regression_reduction_rate"],
        "val_reduction_rate": val_reduction,
        "test_reduction_rate": test_reduction,
        "overfit_gap": overfit_gap,
        "context_overlap_count": context_overlap_count,
    }
    return summary, overfit_report


def _evaluate(
    *,
    source_root: Path,
    candidate_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    candidate_kind: str,
    split: str,
    reason_codes: list[str],
) -> dict[str, Any]:
    output_paths = {
        "decisions": batch_root / f"raw-policy-generalization-{candidate_kind}-{split}-decisions.jsonl",
        "regression_report": batch_root
        / f"raw-policy-generalization-{candidate_kind}-{split}-regression-report.json",
        "summary": batch_root / f"raw-policy-generalization-{candidate_kind}-{split}-summary.json",
    }
    candidate_summary = (
        "controlled-hybrid-policy-training-candidate-summary.json"
        if candidate_kind == "baseline"
        else "raw-policy-generalization-candidate-summary.json"
    )
    schema_versions = (
        config["validation"]["baseline_candidate_summary_schema_versions"]
        if candidate_kind == "baseline"
        else config["validation"]["aligned_candidate_summary_schema_versions"]
    )
    rollout_config = {
        "schema_version": ROLLOUT_CONFIG_SCHEMA_VERSION,
        "description": config.get("description"),
        "input_files": {
            "source_batch_summary": "batch-evaluation-summary.json",
            "holdout_batch_summary": "batch-evaluation-summary.json",
            "fresh_holdout_summary": "fresh-holdout-policy-candidate-evaluation-summary.json",
            "candidate_summary": candidate_summary,
            "checkpoint": "experimental-hybrid-policy-candidate.pt",
            "checkpoint_metadata": "experimental-hybrid-policy-candidate-metadata.json",
        },
        "output_files": {
            "decisions": output_paths["decisions"].name,
            "regression_report": output_paths["regression_report"].name,
            "summary": output_paths["summary"].name,
        },
        "evaluation": dict(config["evaluation"]),
        "validation": {
            "candidate_summary_schema_versions": schema_versions,
            "min_scenario_disjoint_context_count": config["validation"][
                "min_scenario_disjoint_context_count"
            ],
            "require_context_id": config["validation"].get("require_context_id", True),
            "require_candidate_git_current_match": config["validation"].get(
                "require_candidate_git_current_match",
                True,
            ),
            "allow_dirty_current_git_match": config["validation"].get(
                "allow_dirty_current_git_match",
                False,
            ),
            "max_invalid_action_mask_count": config["validation"]["max_invalid_action_mask_count"],
            "max_fallback_or_open_grid_count": config["validation"][
                "max_fallback_or_open_grid_count"
            ],
            "max_safety_regression_count": config["validation"]["max_safety_regression_count"],
            "max_contract_violation_count": config["validation"]["max_contract_violation_count"],
            "max_path_cost_regression_count": config["validation"][
                "max_path_cost_regression_count"
            ],
            "max_risk_regression_count": config["validation"]["max_risk_regression_count"],
            "max_source_selection_regression_count": config["validation"][
                "max_source_selection_regression_count"
            ],
            "max_regression_count": config["validation"]["max_regression_count"],
        },
        "non_goals": list(config.get("non_goals", [])),
    }
    summary, decisions, regression_report = run_scenario_disjoint_policy_rollout_evaluation(
        source_root=source_root,
        candidate_root=candidate_root,
        batch_root=batch_root,
        config=rollout_config,
        repo_root=repo_root,
        output_paths=output_paths,
    )
    _write_json(output_paths["summary"], summary)
    _write_json(output_paths["regression_report"], regression_report)
    output_paths["decisions"].write_text(
        "".join(json.dumps(decision, ensure_ascii=False) + "\n" for decision in decisions),
        encoding="utf-8",
    )
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(reason_codes, f"{candidate_kind}_{split}_rollout_failed")
    return summary


def _context_overlap(split_roots: dict[str, Path], repo_root: Path) -> tuple[int, list[str]]:
    seen: dict[str, str] = {}
    overlaps: set[str] = set()
    for split, root in split_roots.items():
        for group in _collect_holdout_scenarios(root, repo_root):
            for candidate in group.get("candidates", []):
                context_id = candidate.get("context_id")
                if not context_id:
                    continue
                context_id = str(context_id)
                owner = seen.setdefault(context_id, split)
                if owner != split:
                    overlaps.add(context_id)
    return len(overlaps), sorted(overlaps)


def _check_test_gates(
    summary: dict[str, Any],
    *,
    validation: dict[str, Any],
    reason_codes: list[str],
) -> None:
    checks = (
        ("regression_count", "test_controlled_regression"),
        ("invalid_action_mask_count", "test_invalid_action_mask"),
        ("fallback_or_open_grid_count", "test_fallback_or_open_grid"),
        ("safety_regression_count", "test_safety_regression"),
        ("contract_violation_count", "test_contract_violation"),
        ("path_cost_regression_count", "test_path_cost_regression"),
        ("risk_regression_count", "test_risk_regression"),
        ("source_selection_regression_count", "test_source_selection_regression"),
    )
    for field, reason in checks:
        max_field = "max_" + field
        if _int_value(summary.get(field)) > _int_value(validation.get(max_field)):
            _append_reason(reason_codes, reason)


def _reduction_rate(baseline: int, aligned: int) -> float:
    if baseline <= 0:
        return 1.0 if aligned <= 0 else 0.0
    return max(0.0, (baseline - aligned) / baseline)


def _next_required_change(reason_codes: list[str]) -> str | None:
    if not reason_codes:
        return None
    if any("context" in reason or "data" in reason for reason in reason_codes):
        return NEXT_DATA_REQUIRED
    if any("feature" in reason or "action_mask" in reason for reason in reason_codes):
        return NEXT_FEATURE_REQUIRED
    if any("reduction" in reason or "rollout" in reason for reason in reason_codes):
        return NEXT_OBJECTIVE_REQUIRED
    return NEXT_COVERAGE_REQUIRED


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    for section in ("output_files", "evaluation", "validation"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _load_json(path: Path, *, label: str, reason_codes: list[str]) -> dict[str, Any]:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _append_reason(reason_codes, f"{label}_invalid_json")
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _install_model_explorer_path(repo_root: Path) -> None:
    source = repo_root / "model-explorer" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


if __name__ == "__main__":
    raise SystemExit(main())
