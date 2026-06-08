from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot

from run_controlled_hybrid_policy_training_candidate import (
    CHECKPOINT_METADATA_SCHEMA_VERSION,
    CONFIG_SCHEMA_VERSION as CONTROLLED_CONFIG_SCHEMA_VERSION,
    run_controlled_hybrid_policy_training_candidate,
)
from run_scenario_disjoint_policy_rollout_evaluation import (
    CONFIG_SCHEMA_VERSION as ROLLOUT_CONFIG_SCHEMA_VERSION,
    run_scenario_disjoint_policy_rollout_evaluation,
)


CONFIG_SCHEMA_VERSION = "raw-policy-generalization-candidate-config/v1"
SUMMARY_SCHEMA_VERSION = "raw-policy-generalization-candidate-summary/v1"
COMBINED_MINING_SCHEMA_VERSION = "raw-policy-regression-mining-summary/v1"
NEXT_REQUIRED_CHANGE = "objective_or_sample_weight_refinement_required"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train an anti-overfit raw-policy generalization candidate."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--train-mining-root", action="append", required=True)
    parser.add_argument("--dev-mining-root", action="append", default=[])
    parser.add_argument("--val-diagnostic-root", required=True)
    parser.add_argument("--test-diagnostic-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    _install_model_explorer_path(repo_root)
    source_root = _resolve_path(args.source_root, repo_root)
    train_roots = [_resolve_path(value, repo_root) for value in args.train_mining_root]
    dev_roots = [_resolve_path(value, repo_root) for value in args.dev_mining_root]
    val_root = _resolve_path(args.val_diagnostic_root, repo_root)
    test_root = _resolve_path(args.test_diagnostic_root, repo_root)
    output_root = _resolve_path(args.output_root, repo_root)
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    output_root.mkdir(parents=True, exist_ok=True)
    summary = run_raw_policy_generalization_candidate(
        source_root=source_root,
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
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "config validated" if summary["status"] == "passed" else "validation failed",
                "source_root": _display_path(source_root, repo_root),
                "output_root": _display_path(output_root, repo_root),
                "reason_codes": summary["reason_codes"],
                "train_pair_count": summary["train_pair_count"],
                "leaked_context_id_count": summary["leaked_context_id_count"],
                "best_seed": summary.get("best_seed"),
                "best_epoch": summary.get("best_epoch"),
                "summary": _display_path(summary_path, repo_root),
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def run_raw_policy_generalization_candidate(
    *,
    source_root: Path,
    train_roots: list[Path],
    dev_roots: list[Path],
    val_root: Path,
    test_root: Path,
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    validate_only: bool,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    current_git = _git_snapshot(repo_root)
    train_samples, train_summaries = _collect_training_samples(
        train_roots + dev_roots,
        reason_codes=reason_codes,
    )
    train_context_ids = _sample_context_ids(train_samples)
    val_context_ids = _diagnostic_context_ids(val_root, reason_codes=reason_codes, label="val")
    test_context_ids = _diagnostic_context_ids(test_root, reason_codes=reason_codes, label="test")
    leaked_context_ids = sorted(train_context_ids & (val_context_ids | test_context_ids))
    if leaked_context_ids:
        _append_reason(reason_codes, "train_eval_context_leakage_detected")
    if len(leaked_context_ids) > _int_value(config["validation"].get("max_leaked_context_id_count")):
        _append_reason(reason_codes, "leaked_context_id_count_above_threshold")
    if not train_samples:
        _append_reason(reason_codes, "raw_policy_generalization_train_samples_missing")
    if any(summary.get("status") != "passed" for summary in train_summaries):
        _append_reason(reason_codes, "raw_policy_train_mining_summary_failed")
    hard_positive_added_count = 0
    if hard_positive_added_count > _int_value(config["validation"].get("max_hard_positive_added_count")):
        _append_reason(reason_codes, "hard_positive_added_count_nonzero")

    output_files = config["output_files"]
    combined_summary_path = output_root / output_files["training_mining_summary"]
    combined_samples_path = output_root / output_files["training_samples"]
    overfit_report_path = output_root / output_files["overfit_report"]
    training_curves_path = output_root / output_files["training_curves"]
    if not reason_codes:
        _write_combined_training_inputs(
            samples=train_samples,
            summaries=train_summaries,
            summary_path=combined_summary_path,
            samples_path=combined_samples_path,
            source_root=source_root,
            output_root=output_root,
            repo_root=repo_root,
            current_git=current_git,
            config=config,
        )

    seed_results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    training_error: str | None = None
    if not reason_codes and not validate_only:
        for seed in _seeds(config):
            seed_root = output_root / f"seed-{seed}"
            controlled_config = _controlled_config(
                config,
                seed=seed,
                combined_summary_path=combined_summary_path,
                combined_samples_path=combined_samples_path,
            )
            seed_summary_path = seed_root / "controlled-hybrid-policy-training-candidate-summary.json"
            seed_summary = run_controlled_hybrid_policy_training_candidate(
                source_root=source_root,
                output_root=seed_root,
                config=controlled_config,
                repo_root=repo_root,
                summary_path=seed_summary_path,
                validate_only=False,
            )
            seed_summary_path.write_text(
                json.dumps(seed_summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            val_summary = _evaluate_seed_on_val(
                source_root=source_root,
                candidate_root=seed_root,
                val_root=val_root,
                output_root=seed_root,
                config=config,
                repo_root=repo_root,
            )
            result = {
                "seed": seed,
                "epoch": _int_value(config["training"].get("max_epochs")),
                "candidate_status": seed_summary.get("status"),
                "candidate_reason_codes": _string_list(seed_summary.get("reason_codes")),
                "val_status": val_summary.get("status"),
                "val_reason_codes": _string_list(val_summary.get("reason_codes")),
                "val_raw_policy_regression_count": _int_value(
                    val_summary.get("raw_policy_regression_count")
                ),
                "val_regression_count": _int_value(val_summary.get("regression_count")),
                "val_invalid_action_mask_count": _int_value(
                    val_summary.get("invalid_action_mask_count")
                ),
                "checkpoint": _display_path(seed_root / "experimental-hybrid-policy-candidate.pt", repo_root),
                "checkpoint_metadata": _display_path(
                    seed_root / "experimental-hybrid-policy-candidate-metadata.json",
                    repo_root,
                ),
            }
            seed_results.append(result)
            if result["candidate_status"] != "passed" or result["val_status"] != "passed":
                continue
            if best_result is None or _candidate_rank(result) < _candidate_rank(best_result):
                best_result = result
        if best_result is None:
            _append_reason(reason_codes, "no_val_passing_candidate")
        else:
            try:
                _promote_best_checkpoint(
                    best_result,
                    output_root=output_root,
                    config=config,
                    repo_root=repo_root,
                )
            except Exception as exc:  # noqa: BLE001
                _append_reason(reason_codes, "best_checkpoint_promotion_failed")
                training_error = str(exc)

    overfit_gap = _overfit_gap(seed_results, best_result)
    status = "failed" if reason_codes else "passed"
    overfit_report = {
        "schema_version": "raw-policy-generalization-overfit-report/v1",
        "status": status,
        "reason_codes": reason_codes,
        "seed_results": seed_results,
        "best_seed": best_result.get("seed") if best_result else None,
        "best_epoch": best_result.get("epoch") if best_result else None,
        "overfit_gap": overfit_gap,
        "leaked_context_ids": leaked_context_ids,
    }
    training_curves = {
        "schema_version": "raw-policy-generalization-training-curves/v1",
        "status": status,
        "seed_results": seed_results,
    }
    _write_json(overfit_report_path, overfit_report)
    _write_json(training_curves_path, training_curves)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "candidate_training_status": "passed" if status == "passed" else "failed",
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "source_root": _display_path(source_root, repo_root),
        "train_mining_roots": [_display_path(path, repo_root) for path in train_roots],
        "dev_mining_roots": [_display_path(path, repo_root) for path in dev_roots],
        "val_diagnostic_root": _display_path(val_root, repo_root),
        "test_diagnostic_root": _display_path(test_root, repo_root),
        "output_root": _display_path(output_root, repo_root),
        "summary": _display_path(output_root / output_files["summary"], repo_root),
        "training_samples_path": _display_path(combined_samples_path, repo_root),
        "training_mining_summary_path": _display_path(combined_summary_path, repo_root),
        "overfit_report_path": _display_path(overfit_report_path, repo_root),
        "training_curves_path": _display_path(training_curves_path, repo_root),
        "train_pair_count": len(train_samples),
        "train_context_id_count": len(train_context_ids),
        "val_diagnostic_context_id_count": len(val_context_ids),
        "test_diagnostic_context_id_count": len(test_context_ids),
        "leaked_context_id_count": len(leaked_context_ids),
        "leaked_context_ids": leaked_context_ids[:25],
        "hard_positive_added_count": hard_positive_added_count,
        "seed_results": seed_results,
        "best_seed": best_result.get("seed") if best_result else None,
        "best_epoch": best_result.get("epoch") if best_result else None,
        "val_raw_policy_regression_count": (
            best_result.get("val_raw_policy_regression_count") if best_result else None
        ),
        "overfit_gap": overfit_gap,
        "training_error": training_error,
        "checkpoint_path": _display_path(output_root / output_files["checkpoint"], repo_root),
        "checkpoint_metadata_path": _display_path(
            output_root / output_files["checkpoint_metadata"],
            repo_root,
        ),
        "experimental_checkpoint": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "next_required_change": NEXT_REQUIRED_CHANGE if status == "failed" else None,
        "git_provenance": {"current": current_git, "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }


def _collect_training_samples(
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
        for sample in _load_jsonl(
            root / "raw-policy-regression-preference-samples.jsonl",
            label="raw_policy_regression_preference_samples",
            reason_codes=reason_codes,
        ):
            if sample.get("sample_type") == "raw_policy_regression_preference_pair":
                samples.append(sample)
    return samples, summaries


def _diagnostic_context_ids(root: Path, *, reason_codes: list[str], label: str) -> set[str]:
    context_ids: set[str] = set()
    for filename in (
        "raw-policy-regression-diagnostics.jsonl",
        "raw-policy-regression-preference-samples.jsonl",
    ):
        path = root / filename
        if not path.exists():
            continue
        for record in _load_jsonl(path, label=f"{label}_{filename}", reason_codes=reason_codes):
            for key in ("context_id", "alternative_context_id"):
                if record.get(key):
                    context_ids.add(str(record[key]))
    return context_ids


def _sample_context_ids(samples: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for sample in samples:
        for key in ("context_id", "alternative_context_id"):
            if sample.get(key):
                ids.add(str(sample[key]))
    return ids


def _write_combined_training_inputs(
    *,
    samples: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    summary_path: Path,
    samples_path: Path,
    source_root: Path,
    output_root: Path,
    repo_root: Path,
    current_git: dict[str, Any],
    config: dict[str, Any],
) -> None:
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8",
    )
    summary = {
        "schema_version": COMBINED_MINING_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": "passed",
        "reason_codes": [],
        "source_root": _display_path(source_root, repo_root),
        "output_root": _display_path(output_root, repo_root),
        "raw_policy_regression_input_count": sum(
            _int_value(item.get("raw_policy_regression_input_count")) for item in summaries
        ),
        "raw_policy_regression_preference_pair_count": len(samples),
        "raw_policy_regression_excluded_count": sum(
            _int_value(item.get("raw_policy_regression_excluded_count")) for item in summaries
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
    _write_json(summary_path, summary)


def _controlled_config(
    config: dict[str, Any],
    *,
    seed: int,
    combined_summary_path: Path,
    combined_samples_path: Path,
) -> dict[str, Any]:
    training = dict(config["training"])
    training["seed"] = seed
    training["epochs"] = _int_value(training.get("max_epochs"))
    training.pop("seeds", None)
    training.pop("max_epochs", None)
    training.pop("early_stopping_patience", None)
    training.pop("val_metric", None)
    inputs = dict(config["input_files"])
    inputs["raw_policy_regression_mining_summary"] = str(combined_summary_path)
    inputs["raw_policy_regression_preference_samples"] = str(combined_samples_path)
    validation = dict(config["validation"])
    validation.pop("max_leaked_context_id_count", None)
    validation.pop("max_hard_positive_added_count", None)
    return {
        "schema_version": CONTROLLED_CONFIG_SCHEMA_VERSION,
        "description": config.get("description"),
        "input_files": inputs,
        "output_files": {
            "summary": "controlled-hybrid-policy-training-candidate-summary.json",
            "checkpoint": "experimental-hybrid-policy-candidate.pt",
            "checkpoint_metadata": "experimental-hybrid-policy-candidate-metadata.json",
        },
        "source_preconditions": dict(config["source_preconditions"]),
        "validation": validation,
        "training": training,
        "checkpoint": dict(config["checkpoint"]),
        "non_goals": list(config.get("non_goals", [])),
    }


def _evaluate_seed_on_val(
    *,
    source_root: Path,
    candidate_root: Path,
    val_root: Path,
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    output_paths = {
        "decisions": output_root / "val-raw-policy-rollout-decisions.jsonl",
        "regression_report": output_root / "val-raw-policy-rollout-regression-report.json",
        "summary": output_root / "val-raw-policy-rollout-summary.json",
    }
    rollout_config = {
        "schema_version": ROLLOUT_CONFIG_SCHEMA_VERSION,
        "description": config.get("description"),
        "input_files": {
            "source_batch_summary": "batch-evaluation-summary.json",
            "holdout_batch_summary": "batch-evaluation-summary.json",
            "fresh_holdout_summary": "fresh-holdout-policy-candidate-evaluation-summary.json",
            "candidate_summary": "controlled-hybrid-policy-training-candidate-summary.json",
            "checkpoint": "experimental-hybrid-policy-candidate.pt",
            "checkpoint_metadata": "experimental-hybrid-policy-candidate-metadata.json",
        },
        "output_files": {
            "decisions": output_paths["decisions"].name,
            "regression_report": output_paths["regression_report"].name,
            "summary": output_paths["summary"].name,
        },
        "evaluation": dict(config["evaluation"]),
        "validation": dict(config["val_evaluation"]),
        "non_goals": list(config.get("non_goals", [])),
    }
    summary, decisions, regression_report = run_scenario_disjoint_policy_rollout_evaluation(
        source_root=source_root,
        candidate_root=candidate_root,
        batch_root=val_root,
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
    return summary


def _candidate_rank(result: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        _int_value(result.get("val_raw_policy_regression_count")),
        _int_value(result.get("val_regression_count")),
        _int_value(result.get("val_invalid_action_mask_count")),
        _int_value(result.get("seed")),
    )


def _promote_best_checkpoint(
    best_result: dict[str, Any],
    *,
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> None:
    seed_root = output_root / f"seed-{best_result['seed']}"
    checkpoint_src = seed_root / "experimental-hybrid-policy-candidate.pt"
    metadata_src = seed_root / "experimental-hybrid-policy-candidate-metadata.json"
    checkpoint_dst = output_root / config["output_files"]["checkpoint"]
    metadata_dst = output_root / config["output_files"]["checkpoint_metadata"]
    shutil.copy2(checkpoint_src, checkpoint_dst)
    metadata = _load_json(metadata_src, label="checkpoint_metadata", reason_codes=[])
    metadata["schema_version"] = CHECKPOINT_METADATA_SCHEMA_VERSION
    metadata["checkpoint_path"] = _display_path(checkpoint_dst, repo_root)
    metadata["selected_by"] = "raw_policy_generalization_val_metric"
    metadata["best_seed"] = best_result["seed"]
    metadata["best_epoch"] = best_result["epoch"]
    metadata["publishes_checkpoint"] = False
    metadata["replaces_default_policy"] = False
    metadata["performance_claimed"] = False
    _write_json(metadata_dst, metadata)


def _overfit_gap(seed_results: list[dict[str, Any]], best_result: dict[str, Any] | None) -> float | None:
    if not seed_results or best_result is None:
        return None
    vals = [_int_value(item.get("val_raw_policy_regression_count")) for item in seed_results]
    return float(max(vals) - _int_value(best_result.get("val_raw_policy_regression_count")))


def _seeds(config: dict[str, Any]) -> list[int]:
    seeds = config["training"].get("seeds")
    if isinstance(seeds, list) and seeds:
        return [_int_value(seed) for seed in seeds]
    return [_int_value(config["training"].get("seed"))]


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
        "source_preconditions",
        "validation",
        "training",
        "checkpoint",
        "evaluation",
        "val_evaluation",
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
