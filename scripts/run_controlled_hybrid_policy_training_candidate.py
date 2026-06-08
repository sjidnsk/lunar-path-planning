from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot

from run_hybrid_policy_training_dry_run import (
    PAIRWISE_SAMPLE_TYPES,
    _action_label_batch,
    _pairwise_batch,
    _pairwise_loss,
)


CONFIG_SCHEMA_VERSION = "controlled-hybrid-policy-training-candidate-config/v1"
SUMMARY_SCHEMA_VERSION = "controlled-hybrid-policy-training-candidate-summary/v1"
CHECKPOINT_SCHEMA_VERSION = "controlled-hybrid-policy-candidate-checkpoint/v1"
CHECKPOINT_METADATA_SCHEMA_VERSION = "controlled-hybrid-policy-candidate-checkpoint-metadata/v1"
MATERIALIZATION_SCHEMA_VERSION = "planner-validated-training-input-materialization-summary/v1"
REGISTRY_SCHEMA_VERSION = "unified-policy-sample-registry-summary/v1"
HYBRID_SCHEMA_VERSION = "hybrid-policy-training-dry-run-summary/v1"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a controlled local experimental hybrid policy candidate."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    _install_model_explorer_path(repo_root)
    source_root = _resolve_path(args.source_root, repo_root)
    output_root = _resolve_path(args.output_root, repo_root)
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    summary_path = output_root / config["output_files"]["summary"]
    summary = run_controlled_hybrid_policy_training_candidate(
        source_root=source_root,
        output_root=output_root,
        config=config,
        repo_root=repo_root,
        summary_path=summary_path,
        validate_only=args.validate_only,
    )
    print(
        json.dumps(
            {
                "status": "config validated" if summary["status"] == "passed" else "validation failed",
                "source_root": _display_path(source_root, repo_root),
                "output_root": _display_path(output_root, repo_root),
                "reason_codes": summary["reason_codes"],
                "action_label_positive_count": summary["action_label_positive_count"],
                "pairwise_preference_signal_count": summary["pairwise_preference_signal_count"],
                "hybrid_train_signal_count": summary["hybrid_train_signal_count"],
                "candidate_training_status": summary["candidate_training_status"],
                "summary": _display_path(summary_path, repo_root),
            },
            ensure_ascii=False,
        )
    )
    if not args.validate_only:
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 1 if summary["status"] == "failed" else 0


def run_controlled_hybrid_policy_training_candidate(
    *,
    source_root: Path,
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    summary_path: Path,
    validate_only: bool,
) -> dict[str, Any]:
    from model_explorer.policy.dataset import validate_rollout_dataset
    from model_explorer.policy.rollout_io import read_rollout_episodes

    reason_codes: list[str] = []
    paths = _input_paths(source_root, output_root, config)
    source_payloads = _load_source_payloads(paths, reason_codes)

    materialization = _load_summary(
        paths["materialization_summary"],
        expected_schema=MATERIALIZATION_SCHEMA_VERSION,
        label="materialization_summary",
        reason_codes=reason_codes,
    )
    registry_summary = _load_summary(
        paths["unified_policy_sample_registry_summary"],
        expected_schema=REGISTRY_SCHEMA_VERSION,
        label="unified_policy_sample_registry_summary",
        reason_codes=reason_codes,
    )
    hybrid_summary = _load_summary(
        paths["hybrid_policy_training_dry_run_summary"],
        expected_schema=HYBRID_SCHEMA_VERSION,
        label="hybrid_policy_training_dry_run_summary",
        reason_codes=reason_codes,
    )
    for label, payload in (
        ("materialization_summary", materialization),
        ("unified_policy_sample_registry_summary", registry_summary),
        ("hybrid_policy_training_dry_run_summary", hybrid_summary),
    ):
        if payload.get("status") != "passed" or _string_list(payload.get("reason_codes")):
            _append_reason(reason_codes, f"{label}_failed")

    _check_source_preconditions(source_payloads, config=config, reason_codes=reason_codes)
    registry = _load_jsonl(paths["unified_policy_sample_registry"], "unified_policy_sample_registry", reason_codes)
    pairwise_samples = [
        record for record in registry if record.get("sample_type") in PAIRWISE_SAMPLE_TYPES
    ]
    episodes = ()
    dataset_summary: dict[str, Any] | None = None
    training_error: str | None = None
    try:
        episodes = read_rollout_episodes(paths["rollout_episodes"])
    except Exception as exc:  # noqa: BLE001
        _append_reason(reason_codes, "rollout_episodes_unreadable")
        training_error = str(exc)
    if episodes:
        try:
            dataset_summary = validate_rollout_dataset(
                episodes,
                gates={
                    "max_invalid_action_mask_count": config["validation"].get(
                        "max_invalid_action_mask_count"
                    ),
                    "max_empty_action_mask_count": config["validation"].get(
                        "max_empty_action_mask_count"
                    ),
                    "min_trainable_transition_count": config["validation"].get(
                        "expected_action_label_positive_count"
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001
            _append_reason(reason_codes, "rollout_dataset_validation_failed")
            training_error = str(exc)

    action_label_positive_count = _int_value(materialization.get("input_positive_count"))
    existing_preference_pair_count = sum(
        1 for record in pairwise_samples if record.get("sample_type") == "counterfactual_preference_pair"
    )
    residual_preference_pair_count = sum(
        1 for record in pairwise_samples if record.get("sample_type") != "counterfactual_preference_pair"
    )
    pairwise_preference_signal_count = existing_preference_pair_count + residual_preference_pair_count
    hybrid_train_signal_count = action_label_positive_count + pairwise_preference_signal_count
    hard_positive_added_count = _int_value(registry_summary.get("hard_positive_added_count"))
    invalid_action_mask_count = max(
        _int_value(materialization.get("invalid_action_mask_count")),
        _int_value((dataset_summary or {}).get("invalid_action_mask_count")),
    )
    empty_action_mask_count = max(
        _int_value(materialization.get("empty_action_mask_count")),
        _int_value((dataset_summary or {}).get("empty_action_mask_count")),
    )

    _check_expected_count(
        reason_codes,
        config["validation"],
        "expected_action_label_positive_count",
        action_label_positive_count,
        "action_label_positive_count_mismatch",
    )
    _check_expected_count(
        reason_codes,
        config["validation"],
        "expected_pairwise_preference_signal_count",
        pairwise_preference_signal_count,
        "pairwise_preference_signal_count_mismatch",
    )
    _check_expected_count(
        reason_codes,
        config["validation"],
        "expected_hybrid_train_signal_count",
        hybrid_train_signal_count,
        "hybrid_train_signal_count_mismatch",
    )
    if hard_positive_added_count != 0:
        _append_reason(reason_codes, "hard_positive_added_count_nonzero")
    if invalid_action_mask_count > _int_value(config["validation"].get("max_invalid_action_mask_count")):
        _append_reason(reason_codes, "invalid_action_mask")
    if empty_action_mask_count > _int_value(config["validation"].get("max_empty_action_mask_count")):
        _append_reason(reason_codes, "empty_action_mask")
    if not config["checkpoint"].get("experimental", False):
        _append_reason(reason_codes, "checkpoint_not_marked_experimental")
    if config["checkpoint"].get("publishes_checkpoint", False):
        _append_reason(reason_codes, "checkpoint_publication_requested")
    if config["checkpoint"].get("replaces_default_policy", False):
        _append_reason(reason_codes, "default_policy_replacement_requested")

    training_result: dict[str, Any] | None = None
    checkpoint_metadata: dict[str, Any] | None = None
    checkpoint_path = paths["checkpoint"]
    checkpoint_metadata_path = paths["checkpoint_metadata"]
    if not reason_codes and not validate_only:
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            training_result, checkpoint_metadata = _train_candidate_checkpoint(
                episodes,
                pairwise_samples,
                config=config,
                checkpoint_path=checkpoint_path,
                checkpoint_metadata_path=checkpoint_metadata_path,
                source_root=source_root,
                output_root=output_root,
                repo_root=repo_root,
            )
        except Exception as exc:  # noqa: BLE001
            _append_reason(reason_codes, "candidate_training_failed")
            training_error = str(exc)

    status = "failed" if reason_codes else "passed"
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "candidate_training_status": "passed" if status == "passed" else "failed",
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "source_root": _display_path(source_root, repo_root),
        "output_root": _display_path(output_root, repo_root),
        "source_summaries": _source_summaries(paths, repo_root),
        "summary": _display_path(summary_path, repo_root),
        "action_label_positive_count": action_label_positive_count,
        "existing_preference_pair_count": existing_preference_pair_count,
        "residual_preference_pair_count": residual_preference_pair_count,
        "pairwise_preference_signal_count": pairwise_preference_signal_count,
        "hybrid_train_signal_count": hybrid_train_signal_count,
        "hard_positive_added_count": hard_positive_added_count,
        "invalid_action_mask_count": invalid_action_mask_count,
        "empty_action_mask_count": empty_action_mask_count,
        "dataset_summary": dataset_summary,
        "training_result": training_result,
        "training_error": training_error,
        "checkpoint_path": _display_path(checkpoint_path, repo_root),
        "checkpoint_metadata_path": _display_path(checkpoint_metadata_path, repo_root),
        "checkpoint_metadata": checkpoint_metadata,
        "experimental_checkpoint": bool(config["checkpoint"].get("experimental", False)),
        "writes_experimental_checkpoint": bool(training_result),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "runs_large_scale_training": False,
        "formal_training_ready_claimed": False,
        "does_not_modify_default_astar": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "does_not_relax_default_distance_contract": True,
        "no_ackermann_feasible_trajectory_claim": True,
        "git_provenance": {"current": _git_snapshot(repo_root), "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }


def _train_candidate_checkpoint(
    episodes,
    pairwise_samples: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_metadata_path: Path,
    source_root: Path,
    output_root: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from torch.nn import functional as F

    from model_explorer.policy.architectures import build_policy_network
    from model_explorer.policy.ppo import compute_masked_ppo_loss

    transitions = tuple(
        transition
        for episode in episodes
        for transition in episode.transitions
        if transition.action_index >= 0
    )
    if not transitions:
        raise ValueError("rollout episodes must contain trainable transitions")
    if not pairwise_samples:
        raise ValueError("registry must contain pairwise preference samples")

    training = config["training"]
    torch.manual_seed(_int_value(training.get("seed")))
    network = build_policy_network(
        None,
        observation=transitions[0].observation,
        hidden_size=_int_value(training.get("hidden_size")),
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=float(training.get("learning_rate", 1.0e-3)))
    action_batch = _action_label_batch(transitions)
    pairwise_batch = _pairwise_batch(pairwise_samples)
    action_weight = float(training.get("action_label_loss_weight", 1.0))
    preference_weight = float(training.get("preference_loss_weight", 1.0))
    residual_weight = float(training.get("residual_negative_loss_weight", 1.0))
    margin = float(training.get("margin", 0.1))
    epoch_losses: list[dict[str, Any]] = []
    for epoch in range(_int_value(training.get("epochs"))):
        optimizer.zero_grad()
        action_losses = compute_masked_ppo_loss(network, **action_batch)
        pairwise_loss = _pairwise_loss(
            network,
            pairwise_batch,
            margin=margin,
            preference_weight=preference_weight,
            residual_weight=residual_weight,
            torch=torch,
            F=F,
        )
        total_loss = action_weight * action_losses.total_loss + pairwise_loss
        total_loss.backward()
        optimizer.step()
        epoch_losses.append(
            {
                "epoch": epoch + 1,
                "total_loss": float(total_loss.detach()),
                "action_label_loss": float(action_losses.total_loss.detach()),
                "pairwise_preference_loss": float(pairwise_loss.detach()),
            }
        )
    if not epoch_losses:
        raise ValueError("epochs must be at least 1")

    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "experimental": True,
        "architecture": network.architecture_name,
        "model_state_dict": network.state_dict(),
        "training": dict(training),
        "source_root": _display_path(source_root, repo_root),
        "output_root": _display_path(output_root, repo_root),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    metadata = {
        "schema_version": CHECKPOINT_METADATA_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "experimental": True,
        "checkpoint_path": _display_path(checkpoint_path, repo_root),
        "architecture": network.architecture_name,
        "sample_count": len(transitions) + len(pairwise_samples),
        "action_label_sample_count": len(transitions),
        "pairwise_preference_sample_count": len(pairwise_samples),
        "epochs": _int_value(training.get("epochs")),
        "seed": _int_value(training.get("seed")),
        "hidden_size": _int_value(training.get("hidden_size")),
        "learning_rate": float(training.get("learning_rate", 1.0e-3)),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
    }
    checkpoint_metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    result = {
        "architecture": network.architecture_name,
        "sample_count": len(transitions) + len(pairwise_samples),
        "action_label_sample_count": len(transitions),
        "pairwise_preference_sample_count": len(pairwise_samples),
        "epochs": _int_value(training.get("epochs")),
        "seed": _int_value(training.get("seed")),
        "hidden_size": _int_value(training.get("hidden_size")),
        "learning_rate": float(training.get("learning_rate", 1.0e-3)),
        "margin": margin,
        "action_label_loss_weight": action_weight,
        "preference_loss_weight": preference_weight,
        "residual_negative_loss_weight": residual_weight,
        "epoch_losses": epoch_losses,
        "final_total_loss": epoch_losses[-1]["total_loss"],
        "action_label_loss": epoch_losses[-1]["action_label_loss"],
        "pairwise_preference_loss": epoch_losses[-1]["pairwise_preference_loss"],
    }
    return result, metadata


def _input_paths(source_root: Path, output_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    inputs = config["input_files"]
    outputs = config["output_files"]
    return {
        "batch_summary": source_root / inputs["batch_summary"],
        "anchor_candidate_summary": source_root / inputs["anchor_candidate_summary"],
        "planner_validated_mining_summary": source_root / inputs["planner_validated_mining_summary"],
        "rollout_episodes": source_root / inputs["rollout_episodes"],
        "materialization_summary": source_root / inputs["materialization_summary"],
        "unified_policy_sample_registry": source_root / inputs["unified_policy_sample_registry"],
        "unified_policy_sample_registry_summary": source_root / inputs["unified_policy_sample_registry_summary"],
        "hybrid_policy_training_dry_run_summary": source_root / inputs["hybrid_policy_training_dry_run_summary"],
        "summary": output_root / outputs["summary"],
        "checkpoint": output_root / outputs["checkpoint"],
        "checkpoint_metadata": output_root / outputs["checkpoint_metadata"],
    }


def _load_source_payloads(paths: dict[str, Path], reason_codes: list[str]) -> dict[str, dict[str, Any]]:
    return {
        "batch_summary": _load_summary(paths["batch_summary"], expected_schema=None, label="batch_summary", reason_codes=reason_codes),
        "anchor_candidate_summary": _load_summary(paths["anchor_candidate_summary"], expected_schema=None, label="anchor_candidate_summary", reason_codes=reason_codes),
        "planner_validated_mining_summary": _load_summary(paths["planner_validated_mining_summary"], expected_schema=None, label="planner_validated_mining_summary", reason_codes=reason_codes),
    }


def _check_source_preconditions(
    payloads: dict[str, dict[str, Any]],
    *,
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    validation = config["source_preconditions"]
    for label, payload in payloads.items():
        if payload.get("status") == "failed" or _string_list(payload.get("reason_codes")):
            _append_reason(reason_codes, f"{label}_failed")
    batch = payloads.get("batch_summary", {})
    if _int_value(batch.get("failed_count")) > _int_value(validation.get("max_failed_count")):
        _append_reason(reason_codes, "source_batch_failed")
    if _fallback_or_open_grid_count(payloads) > _int_value(validation.get("max_fallback_or_open_grid_count")):
        _append_reason(reason_codes, "source_fallback_or_open_grid")
    if _max_count(payloads, "safety_regression_count") > _int_value(validation.get("max_safety_regression_count")):
        _append_reason(reason_codes, "source_safety_regression")
    if _max_count(payloads, "current_git_provenance_mismatch_count") > _int_value(
        validation.get("max_current_git_provenance_mismatch_count")
    ):
        _append_reason(reason_codes, "source_current_git_provenance_mismatch")
    if _max_count(payloads, "git_provenance_mismatch_count") > _int_value(
        validation.get("max_git_provenance_mismatch_count")
    ):
        _append_reason(reason_codes, "source_git_provenance_mismatch")


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
    for section in ("input_files", "output_files", "source_preconditions", "validation", "training", "checkpoint"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _check_expected_count(
    reason_codes: list[str],
    validation: dict[str, Any],
    field: str,
    actual: int,
    reason: str,
) -> None:
    if field in validation and actual != _int_value(validation.get(field)):
        _append_reason(reason_codes, reason)


def _source_summaries(paths: dict[str, Path], repo_root: Path) -> dict[str, dict[str, Any]]:
    labels = (
        "batch_summary",
        "anchor_candidate_summary",
        "planner_validated_mining_summary",
        "materialization_summary",
        "unified_policy_sample_registry_summary",
        "hybrid_policy_training_dry_run_summary",
        "rollout_episodes",
        "unified_policy_sample_registry",
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
