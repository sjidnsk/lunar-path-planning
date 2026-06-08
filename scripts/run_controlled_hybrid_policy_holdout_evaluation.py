from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot

from run_controlled_hybrid_policy_training_candidate import (
    CHECKPOINT_METADATA_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION as CANDIDATE_SUMMARY_SCHEMA_VERSION,
)
from run_hybrid_policy_training_dry_run import (
    PAIRWISE_SAMPLE_TYPES,
    _pairwise_batch,
)


CONFIG_SCHEMA_VERSION = "controlled-hybrid-policy-holdout-evaluation-config/v1"
SUMMARY_SCHEMA_VERSION = "controlled-hybrid-policy-holdout-evaluation-summary/v1"
NEXT_REQUIRED_CHANGE = "training_objective_or_sample_weight_refinement_required"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a controlled hybrid policy candidate against offline holdout gates."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    _install_model_explorer_path(repo_root)
    source_root = _resolve_path(args.source_root, repo_root)
    candidate_root = _resolve_path(args.candidate_root, repo_root)
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    summary_path = candidate_root / config["output_files"]["summary"]
    summary = run_controlled_hybrid_policy_holdout_evaluation(
        source_root=source_root,
        candidate_root=candidate_root,
        config=config,
        repo_root=repo_root,
        summary_path=summary_path,
    )
    print(
        json.dumps(
            {
                "status": "config validated" if summary["status"] == "passed" else "validation failed",
                "source_root": _display_path(source_root, repo_root),
                "candidate_root": _display_path(candidate_root, repo_root),
                "reason_codes": summary["reason_codes"],
                "action_mask_invalid_count": summary["action_mask_invalid_count"],
                "fallback_or_open_grid_count": summary["fallback_or_open_grid_count"],
                "safety_regression_count": summary["safety_regression_count"],
                "contract_violation_count": summary["contract_violation_count"],
                "summary": _display_path(summary_path, repo_root),
            },
            ensure_ascii=False,
        )
    )
    if not args.validate_only:
        candidate_root.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 1 if summary["status"] == "failed" else 0


def run_controlled_hybrid_policy_holdout_evaluation(
    *,
    source_root: Path,
    candidate_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    summary_path: Path,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    paths = _input_paths(source_root, candidate_root, config)
    candidate = _load_summary(
        paths["candidate_summary"],
        expected_schema=CANDIDATE_SUMMARY_SCHEMA_VERSION,
        label="controlled_hybrid_policy_training_candidate_summary",
        reason_codes=reason_codes,
    )
    metadata = _load_summary(
        paths["checkpoint_metadata"],
        expected_schema=CHECKPOINT_METADATA_SCHEMA_VERSION,
        label="checkpoint_metadata",
        reason_codes=reason_codes,
    )
    source_payloads = {
        "batch_summary": _load_summary(paths["batch_summary"], expected_schema=None, label="batch_summary", reason_codes=reason_codes),
        "anchor_candidate_summary": _load_summary(paths["anchor_candidate_summary"], expected_schema=None, label="anchor_candidate_summary", reason_codes=reason_codes),
        "planner_validated_mining_summary": _load_summary(paths["planner_validated_mining_summary"], expected_schema=None, label="planner_validated_mining_summary", reason_codes=reason_codes),
    }
    registry = _load_jsonl(paths["unified_policy_sample_registry"], "unified_policy_sample_registry", reason_codes)
    pairwise_samples = [
        record for record in registry if record.get("sample_type") in PAIRWISE_SAMPLE_TYPES
    ]

    if candidate.get("status") != "passed" or candidate.get("candidate_training_status") != "passed":
        _append_reason(reason_codes, "controlled_hybrid_policy_training_candidate_summary_failed")
    if _string_list(candidate.get("reason_codes")):
        _append_reason(reason_codes, "controlled_hybrid_policy_training_candidate_summary_failed")
    if not metadata.get("experimental", False):
        _append_reason(reason_codes, "checkpoint_metadata_not_experimental")
    if metadata.get("publishes_checkpoint") or candidate.get("publishes_checkpoint"):
        _append_reason(reason_codes, "checkpoint_publication_detected")
    if metadata.get("replaces_default_policy") or candidate.get("replaces_default_policy"):
        _append_reason(reason_codes, "default_policy_replacement_detected")
    if candidate.get("performance_claimed") or metadata.get("performance_claimed"):
        _append_reason(reason_codes, "performance_claim_detected")
    if not paths["checkpoint"].is_file():
        _append_reason(reason_codes, "experimental_checkpoint_missing")

    action_mask_invalid_count = _int_value(candidate.get("invalid_action_mask_count"))
    empty_action_mask_count = _int_value(candidate.get("empty_action_mask_count"))
    fallback_or_open_grid_count = _fallback_or_open_grid_count(source_payloads)
    safety_regression_count = _max_count(source_payloads, "safety_regression_count")
    contract_violation_count = max(
        _max_count(source_payloads, "candidate_contract_alignment_gap_count"),
        _max_count(source_payloads, "contract_violation_count"),
    )
    path_cost_regression_count = _max_count(source_payloads, "path_cost_regression_count")
    risk_regression_count = _max_count(source_payloads, "risk_regression_count")
    source_selection_regression_count = _max_count(source_payloads, "source_selection_regression_count")

    validation = config["validation"]
    if action_mask_invalid_count > _int_value(validation.get("max_action_mask_invalid_count")):
        _append_reason(reason_codes, "action_mask_invalid")
    if empty_action_mask_count > _int_value(validation.get("max_empty_action_mask_count")):
        _append_reason(reason_codes, "empty_action_mask")
    if fallback_or_open_grid_count > _int_value(validation.get("max_fallback_or_open_grid_count")):
        _append_reason(reason_codes, "fallback_or_open_grid")
    if safety_regression_count > _int_value(validation.get("max_safety_regression_count")):
        _append_reason(reason_codes, "safety_regression")
    if contract_violation_count > _int_value(validation.get("max_contract_violation_count")):
        _append_reason(reason_codes, "contract_violation")

    preference_margin_improved_count = 0
    preference_margin_evaluated_count = 0
    preference_margin_error: str | None = None
    if paths["checkpoint"].is_file() and pairwise_samples:
        try:
            preference_margin_improved_count, preference_margin_evaluated_count = _evaluate_pairwise_margins(
                checkpoint_path=paths["checkpoint"],
                pairwise_samples=pairwise_samples,
                config=config,
                repo_root=repo_root,
            )
        except Exception as exc:  # noqa: BLE001
            _append_reason(reason_codes, "preference_margin_evaluation_failed")
            preference_margin_error = str(exc)

    next_required_change = (
        NEXT_REQUIRED_CHANGE
        if path_cost_regression_count or risk_regression_count or source_selection_regression_count
        else None
    )
    status = "failed" if reason_codes else "passed"
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "source_root": _display_path(source_root, repo_root),
        "candidate_root": _display_path(candidate_root, repo_root),
        "source_summaries": _source_summaries(paths, repo_root),
        "summary": _display_path(summary_path, repo_root),
        "candidate_summary_path": _display_path(paths["candidate_summary"], repo_root),
        "checkpoint_path": _display_path(paths["checkpoint"], repo_root),
        "checkpoint_metadata_path": _display_path(paths["checkpoint_metadata"], repo_root),
        "action_mask_invalid_count": action_mask_invalid_count,
        "empty_action_mask_count": empty_action_mask_count,
        "fallback_or_open_grid_count": fallback_or_open_grid_count,
        "safety_regression_count": safety_regression_count,
        "contract_violation_count": contract_violation_count,
        "path_cost_regression_count": path_cost_regression_count,
        "risk_regression_count": risk_regression_count,
        "source_selection_regression_count": source_selection_regression_count,
        "source_selection_agreement_count": _int_value(candidate.get("action_label_positive_count")),
        "preference_margin_improved_count": preference_margin_improved_count,
        "preference_margin_evaluated_count": preference_margin_evaluated_count,
        "preference_margin_error": preference_margin_error,
        "next_required_change": next_required_change,
        "experimental_checkpoint": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": _git_snapshot(repo_root), "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }


def _evaluate_pairwise_margins(
    *,
    checkpoint_path: Path,
    pairwise_samples: list[dict[str, Any]],
    config: dict[str, Any],
    repo_root: Path,
) -> tuple[int, int]:
    import torch

    from model_explorer.policy.architectures import build_policy_network
    from model_explorer.policy.features import (
        CANDIDATE_FEATURE_NAMES,
        GLOBAL_FEATURE_NAMES,
        MISSING_INDICATOR_NAMES,
        PolicyObservation,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training = checkpoint.get("training", {}) if isinstance(checkpoint, dict) else {}
    observation = PolicyObservation(
        candidate_feature_names=CANDIDATE_FEATURE_NAMES,
        candidate_features=(
            tuple([0.0] * len(CANDIDATE_FEATURE_NAMES)),
            tuple([0.0] * len(CANDIDATE_FEATURE_NAMES)),
        ),
        global_feature_names=GLOBAL_FEATURE_NAMES,
        global_features=tuple([0.0] * len(GLOBAL_FEATURE_NAMES)),
        action_mask=(True, True),
        candidate_cells=((0, 0), (0, 1)),
        candidate_missing_indicator_names=MISSING_INDICATOR_NAMES,
        candidate_missing_indicators=(
            tuple([0.0] * len(MISSING_INDICATOR_NAMES)),
            tuple([0.0] * len(MISSING_INDICATOR_NAMES)),
        ),
    )
    network = build_policy_network(
        None,
        observation=observation,
        hidden_size=_int_value(training.get("hidden_size") or config["evaluation"].get("hidden_size")),
    )
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    batch = _pairwise_batch(pairwise_samples)
    with torch.no_grad():
        output = network(
            candidate_features=batch["candidate_features"],
            global_features=batch["global_features"],
            action_mask=batch["action_mask"],
            candidate_missing_indicators=batch["candidate_missing_indicators"],
        )
        margins = output.masked_logits[:, 0] - output.masked_logits[:, 1]
    threshold = float(config["evaluation"].get("preference_margin_threshold", 0.0))
    improved = int((margins > threshold).sum().item())
    return improved, int(margins.numel())


def _input_paths(source_root: Path, candidate_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    inputs = config["input_files"]
    outputs = config["output_files"]
    return {
        "batch_summary": source_root / inputs["batch_summary"],
        "anchor_candidate_summary": source_root / inputs["anchor_candidate_summary"],
        "planner_validated_mining_summary": source_root / inputs["planner_validated_mining_summary"],
        "unified_policy_sample_registry": source_root / inputs["unified_policy_sample_registry"],
        "candidate_summary": candidate_root / inputs["candidate_summary"],
        "checkpoint": candidate_root / inputs["checkpoint"],
        "checkpoint_metadata": candidate_root / inputs["checkpoint_metadata"],
        "summary": candidate_root / outputs["summary"],
    }


def _load_summary(
    path: Path,
    *,
    expected_schema: str | None,
    label: str,
    reason_codes: list[str],
) -> dict[str, Any]:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _append_reason(reason_codes, f"{label}_invalid_json")
        return {}
    if not isinstance(payload, dict):
        _append_reason(reason_codes, f"{label}_invalid_json_root")
        return {}
    if expected_schema is not None and payload.get("schema_version") != expected_schema:
        _append_reason(reason_codes, f"{label}_schema_version_mismatch")
    return payload


def _load_jsonl(path: Path, label: str, reason_codes: list[str]) -> list[dict[str, Any]]:
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
    for section in ("input_files", "output_files", "validation", "evaluation"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _source_summaries(paths: dict[str, Path], repo_root: Path) -> dict[str, dict[str, Any]]:
    labels = (
        "batch_summary",
        "anchor_candidate_summary",
        "planner_validated_mining_summary",
        "unified_policy_sample_registry",
        "candidate_summary",
        "checkpoint",
        "checkpoint_metadata",
    )
    return {label: {"path": _display_path(paths[label], repo_root), "exists": paths[label].is_file()} for label in labels}


def _fallback_or_open_grid_count(payloads: dict[str, dict[str, Any]]) -> int:
    fields = (
        "open_grid_fallback_used_count",
        "open_grid_fallback_count",
        "fallback_used_count",
        "fallback_or_open_grid_count",
    )
    return max(_int_value(payload.get(field)) for payload in payloads.values() for field in fields)


def _max_count(payloads: dict[str, dict[str, Any]], field: str) -> int:
    return max((_int_value(payload.get(field)) for payload in payloads.values()), default=0)


def _install_model_explorer_path(repo_root: Path) -> None:
    source = repo_root / "model-explorer" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


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


if __name__ == "__main__":
    raise SystemExit(main())
