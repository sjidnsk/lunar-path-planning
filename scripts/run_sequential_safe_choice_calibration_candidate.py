from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot

try:
    from scripts.run_raw_policy_generalization_candidate import (
        CONFIG_SCHEMA_VERSION as RAW_GENERALIZATION_CONFIG_SCHEMA_VERSION,
        run_raw_policy_generalization_candidate,
    )
except ModuleNotFoundError:
    from run_raw_policy_generalization_candidate import (
        CONFIG_SCHEMA_VERSION as RAW_GENERALIZATION_CONFIG_SCHEMA_VERSION,
        run_raw_policy_generalization_candidate,
    )


CONFIG_SCHEMA_VERSION = "sequential-safe-choice-calibration-candidate-config/v1"
SUMMARY_SCHEMA_VERSION = "sequential-safe-choice-calibration-candidate-summary/v1"
SEQUENTIAL_MINING_SCHEMA_VERSION = "sequential-canary-failure-mining-summary/v1"
RAW_MINING_SCHEMA_VERSION = "raw-policy-regression-mining-summary/v1"
NEXT_REQUIRED_CHANGE = "sequence_objective_weight_refinement_required"
SEQUENTIAL_HARD_NEGATIVE_SAMPLE_TYPE = "sequential_hard_negative_preference_pair"
SEQUENTIAL_MISSED_SAFE_CHOICE_SAMPLE_TYPE = "sequential_missed_safe_choice_preference_pair"
SEQUENTIAL_SAMPLE_TYPES = {
    SEQUENTIAL_HARD_NEGATIVE_SAMPLE_TYPE,
    SEQUENTIAL_MISSED_SAFE_CHOICE_SAMPLE_TYPE,
}


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train an experimental sequential safe-choice calibration candidate."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--sequential-mining-root", action="append", required=True)
    parser.add_argument("--train-mining-root", action="append", default=[])
    parser.add_argument("--dev-mining-root", action="append", default=[])
    parser.add_argument("--val-diagnostic-root")
    parser.add_argument("--test-diagnostic-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    _install_model_explorer_path(repo_root)
    source_root = _resolve_path(args.source_root, repo_root)
    sequential_roots = [_resolve_path(value, repo_root) for value in args.sequential_mining_root]
    train_roots = [_resolve_path(value, repo_root) for value in args.train_mining_root]
    dev_roots = [_resolve_path(value, repo_root) for value in args.dev_mining_root]
    val_root = _resolve_path(args.val_diagnostic_root, repo_root) if args.val_diagnostic_root else None
    test_root = _resolve_path(args.test_diagnostic_root, repo_root) if args.test_diagnostic_root else None
    output_root = _resolve_path(args.output_root, repo_root)
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    summary = run_sequential_safe_choice_calibration_candidate(
        source_root=source_root,
        sequential_roots=sequential_roots,
        train_roots=train_roots,
        dev_roots=dev_roots,
        val_root=val_root,
        test_root=test_root,
        output_root=output_root,
        config=config,
        repo_root=repo_root,
        validate_only=args.validate_only,
    )
    summary_path = output_root / config["output_files"]["summary"]
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "sequential_hard_negative_preference_pair_count": summary[
                    "sequential_hard_negative_preference_pair_count"
                ],
                "train_pair_count": summary["train_pair_count"],
                "summary": _display_path(summary_path, repo_root),
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def run_sequential_safe_choice_calibration_candidate(
    *,
    source_root: Path,
    sequential_roots: list[Path],
    train_roots: list[Path],
    dev_roots: list[Path],
    val_root: Path | None,
    test_root: Path | None,
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    validate_only: bool,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    current_git = _git_snapshot(repo_root)
    sequential_samples, sequential_summaries = _collect_sequential_samples(
        sequential_roots,
        config=config,
        reason_codes=reason_codes,
    )
    sequential_hard_negative_input_count = sum(
        1
        for sample in sequential_samples
        if sample.get("sequential_sample_type") == SEQUENTIAL_HARD_NEGATIVE_SAMPLE_TYPE
    )
    raw_samples, raw_summaries = _collect_raw_samples(train_roots + dev_roots, reason_codes=reason_codes)
    sequential_context_ids = _sample_context_ids(sequential_samples)
    eval_context_ids = set()
    if val_root is not None:
        eval_context_ids.update(_diagnostic_context_ids(val_root, label="val", reason_codes=reason_codes))
    if test_root is not None:
        eval_context_ids.update(_diagnostic_context_ids(test_root, label="test", reason_codes=reason_codes))
    leaked_context_ids = sorted(sequential_context_ids & eval_context_ids)
    if leaked_context_ids:
        _append_reason(reason_codes, "train_eval_context_leakage_detected")
    validation = config["validation"]
    if len(leaked_context_ids) > _int_value(validation.get("max_leaked_context_id_count")):
        _append_reason(reason_codes, "leaked_context_id_count_above_threshold")
    if sequential_hard_negative_input_count < _int_value(
        validation.get("min_sequential_hard_negative_preference_pair_count")
    ):
        _append_reason(reason_codes, "sequential_hard_negative_signal_insufficient")
    if any(summary.get("status") != "passed" for summary in sequential_summaries):
        _append_reason(reason_codes, "sequential_failure_mining_summary_failed")
    hard_positive_added_count = sum(_int_value(sample.get("hard_positive_added_count")) for sample in sequential_samples)
    if hard_positive_added_count > _int_value(validation.get("max_hard_positive_added_count")):
        _append_reason(reason_codes, "hard_positive_added_count_nonzero")

    weighted_sequential = [_weighted_sequential_sample(sample, config=config) for sample in sequential_samples]
    hard_negative_count = sum(
        1
        for sample in weighted_sequential
        if sample.get("sequential_sample_type") == SEQUENTIAL_HARD_NEGATIVE_SAMPLE_TYPE
    )
    missed_safe_choice_count = sum(
        1
        for sample in weighted_sequential
        if sample.get("sequential_sample_type") == SEQUENTIAL_MISSED_SAFE_CHOICE_SAMPLE_TYPE
    )
    combined_samples = raw_samples + weighted_sequential
    combined_root = output_root / config["output_files"]["combined_mining_root"]
    _write_combined_raw_mining(
        combined_root=combined_root,
        samples=combined_samples,
        raw_summaries=raw_summaries,
        sequential_summaries=sequential_summaries,
        source_root=source_root,
        output_root=output_root,
        repo_root=repo_root,
        current_git=current_git,
        config=config,
    )

    raw_summary: dict[str, Any] | None = None
    training_error: str | None = None
    if not reason_codes and not validate_only:
        if val_root is None or test_root is None:
            _append_reason(reason_codes, "val_test_diagnostic_roots_required")
        else:
            try:
                raw_summary = run_raw_policy_generalization_candidate(
                    source_root=source_root,
                    train_roots=[combined_root],
                    dev_roots=[],
                    val_root=val_root,
                    test_root=test_root,
                    output_root=output_root,
                    config=_raw_generalization_config(config),
                    repo_root=repo_root,
                    validate_only=False,
                )
                raw_summary_path = output_root / "raw-policy-generalization-candidate-summary.json"
                raw_summary_path.write_text(
                    json.dumps(raw_summary, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                if raw_summary.get("status") != "passed":
                    _append_reason(reason_codes, "raw_policy_generalization_candidate_failed")
            except Exception as exc:  # noqa: BLE001
                _append_reason(reason_codes, "sequential_safe_choice_candidate_training_failed")
                training_error = str(exc)

    status = "failed" if reason_codes else "passed"
    training = config["training"]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "candidate_training_status": "passed" if status == "passed" else "failed",
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "source_root": _display_path(source_root, repo_root),
        "sequential_mining_roots": [_display_path(path, repo_root) for path in sequential_roots],
        "train_mining_roots": [_display_path(path, repo_root) for path in train_roots],
        "dev_mining_roots": [_display_path(path, repo_root) for path in dev_roots],
        "output_root": _display_path(output_root, repo_root),
        "combined_mining_root": _display_path(combined_root, repo_root),
        "train_pair_count": len(combined_samples),
        "existing_raw_policy_regression_preference_pair_count": len(raw_samples),
        "sequential_hard_negative_preference_pair_count": hard_negative_count,
        "sequential_missed_safe_choice_preference_pair_count": missed_safe_choice_count,
        "sequential_preference_pair_count": len(weighted_sequential),
        "sequential_hard_negative_context_id_count": len(sequential_context_ids),
        "leaked_context_id_count": len(leaked_context_ids),
        "leaked_context_ids": leaked_context_ids[:25],
        "hard_positive_added_count": hard_positive_added_count,
        "sequential_hard_negative_loss_weight": float(training.get("sequential_hard_negative_loss_weight", 1.0)),
        "missed_safe_choice_sample_weight": float(training.get("missed_safe_choice_sample_weight", 1.0)),
        "path_cost_regression_negative_weight": float(training.get("path_cost_regression_negative_weight", 1.0)),
        "risk_regression_negative_weight": float(training.get("risk_regression_negative_weight", 1.0)),
        "best_seed": (raw_summary or {}).get("best_seed"),
        "best_epoch": (raw_summary or {}).get("best_epoch"),
        "overfit_gap": (raw_summary or {}).get("overfit_gap"),
        "raw_policy_generalization_candidate_summary": raw_summary,
        "training_error": training_error,
        "experimental_checkpoint": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "next_required_change": NEXT_REQUIRED_CHANGE if status == "failed" else None,
        "git_provenance": {"current": current_git, "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }


def _collect_sequential_samples(
    roots: list[Path],
    *,
    config: dict[str, Any],
    reason_codes: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for root in roots:
        summary = _load_json(
            root / config["input_files"]["sequential_mining_summary"],
            label="sequential_mining_summary",
            reason_codes=reason_codes,
        )
        if summary:
            summaries.append(summary)
            if summary.get("schema_version") != SEQUENTIAL_MINING_SCHEMA_VERSION:
                _append_reason(reason_codes, "sequential_mining_summary_schema_mismatch")
        for sample in _load_jsonl(
            root / config["input_files"]["sequential_hard_negative_samples"],
            label="sequential_hard_negative_samples",
            reason_codes=reason_codes,
        ):
            if sample.get("sequential_sample_type") in SEQUENTIAL_SAMPLE_TYPES:
                samples.append(sample)
    return samples, summaries


def _collect_raw_samples(
    roots: list[Path],
    *,
    reason_codes: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for root in roots:
        summary = _load_json(
            root / "raw-policy-regression-mining-summary.json",
            label="raw_policy_regression_mining_summary",
            reason_codes=reason_codes,
        )
        if summary:
            summaries.append(summary)
            if summary.get("schema_version") != RAW_MINING_SCHEMA_VERSION:
                _append_reason(reason_codes, "raw_policy_regression_mining_summary_schema_mismatch")
        for sample in _load_jsonl(
            root / "raw-policy-regression-preference-samples.jsonl",
            label="raw_policy_regression_preference_samples",
            reason_codes=reason_codes,
        ):
            if sample.get("sample_type") == "raw_policy_regression_preference_pair":
                samples.append(sample)
    return samples, summaries


def _weighted_sequential_sample(sample: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(sample)
    training = config["training"]
    sample_type = result.get("sequential_sample_type")
    if sample_type == SEQUENTIAL_MISSED_SAFE_CHOICE_SAMPLE_TYPE:
        weight = float(training.get("missed_safe_choice_sample_weight", 1.0))
    else:
        weight = float(training.get("sequential_hard_negative_loss_weight", 1.0))
        reasons = _string_list(result.get("raw_policy_regression_reason_codes"))
        if "path_cost_regression" in reasons:
            weight *= float(training.get("path_cost_regression_negative_weight", 1.0))
        if "risk_regression" in reasons:
            weight *= float(training.get("risk_regression_negative_weight", 1.0))
    result["sample_weight"] = weight
    result["calibration_stage"] = "sequential_safe_choice_calibration"
    return result


def _write_combined_raw_mining(
    *,
    combined_root: Path,
    samples: list[dict[str, Any]],
    raw_summaries: list[dict[str, Any]],
    sequential_summaries: list[dict[str, Any]],
    source_root: Path,
    output_root: Path,
    repo_root: Path,
    current_git: dict[str, Any],
    config: dict[str, Any],
) -> None:
    combined_root.mkdir(parents=True, exist_ok=True)
    samples_path = combined_root / "raw-policy-regression-preference-samples.jsonl"
    samples_path.write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8",
    )
    summary = {
        "schema_version": RAW_MINING_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": "passed",
        "reason_codes": [],
        "source_root": _display_path(source_root, repo_root),
        "output_root": _display_path(output_root, repo_root),
        "raw_policy_regression_input_count": sum(
            _int_value(item.get("raw_policy_regression_input_count")) for item in raw_summaries
        )
        + sum(_int_value(item.get("sequential_rejected_step_count")) for item in sequential_summaries),
        "raw_policy_regression_preference_pair_count": len(samples),
        "sequential_hard_negative_preference_pair_count": sum(
            _int_value(item.get("sequential_hard_negative_preference_pair_count"))
            for item in sequential_summaries
        ),
        "sequential_missed_safe_choice_preference_pair_count": sum(
            _int_value(item.get("sequential_missed_safe_choice_preference_pair_count"))
            for item in sequential_summaries
        ),
        "sequential_preference_pair_count": sum(
            _int_value(item.get("sequential_preference_pair_count"))
            for item in sequential_summaries
        ),
        "hard_positive_added_count": 0,
        "samples_path": _display_path(samples_path, repo_root),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": current_git, "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }
    _write_json(combined_root / "raw-policy-regression-mining-summary.json", summary)


def _raw_generalization_config(config: dict[str, Any]) -> dict[str, Any]:
    raw_config = deepcopy(config["raw_policy_generalization_config"])
    if raw_config.get("schema_version") != RAW_GENERALIZATION_CONFIG_SCHEMA_VERSION:
        raise ConfigError("raw_policy_generalization_config schema_version mismatch")
    raw_config["training"].update(config["training"])
    raw_config["training"].pop("sequential_hard_negative_loss_weight", None)
    raw_config["training"].pop("path_cost_regression_negative_weight", None)
    raw_config["training"].pop("risk_regression_negative_weight", None)
    return raw_config


def _diagnostic_context_ids(root: Path, *, label: str, reason_codes: list[str]) -> set[str]:
    ids: set[str] = set()
    for filename in (
        "raw-policy-regression-diagnostics.jsonl",
        "raw-policy-regression-preference-samples.jsonl",
        "sequential-canary-hard-negative-preference-samples.jsonl",
    ):
        path = root / filename
        if not path.exists():
            continue
        for record in _load_jsonl(path, label=f"{label}_{filename}", reason_codes=reason_codes):
            for key in ("context_id", "alternative_context_id"):
                if record.get(key):
                    ids.add(str(record[key]))
    return ids


def _sample_context_ids(samples: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for sample in samples:
        for key in ("context_id", "alternative_context_id"):
            if sample.get(key):
                ids.add(str(sample[key]))
    return ids


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
    for section in (
        "input_files",
        "output_files",
        "validation",
        "training",
        "raw_policy_generalization_config",
    ):
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


def _load_jsonl(path: Path, *, label: str, reason_codes: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    except json.JSONDecodeError:
        _append_reason(reason_codes, f"{label}_invalid_json")
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


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
