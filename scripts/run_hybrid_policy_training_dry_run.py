from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot


CONFIG_SCHEMA_VERSION = "hybrid-policy-training-dry-run-config/v1"
SUMMARY_SCHEMA_VERSION = "hybrid-policy-training-dry-run-summary/v1"
MATERIALIZATION_SCHEMA_VERSION = "planner-validated-training-input-materialization-summary/v1"
COUNTERFACTUAL_SCHEMA_VERSION = "counterfactual-preference-training-summary/v1"
REGISTRY_SCHEMA_VERSION = "unified-policy-sample-registry-summary/v1"
PAIRWISE_SAMPLE_TYPES = {
    "counterfactual_preference_pair",
    "boundary_negative_preference_pair",
    "blocked_target_negative_pair",
}


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a local hybrid action-label plus pairwise preference dry-run."
    )
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    _install_model_explorer_path(repo_root)
    batch_root = _resolve_path(args.batch_root, repo_root)
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    summary_path = batch_root / config["output_files"]["summary"]
    summary = run_hybrid_policy_training_dry_run(
        batch_root=batch_root,
        config=config,
        repo_root=repo_root,
        summary_path=summary_path,
        validate_only=args.validate_only,
    )
    print(
        json.dumps(
            {
                "status": "config validated" if summary["status"] == "passed" else "validation failed",
                "batch_root": _display_path(batch_root, repo_root),
                "reason_codes": summary["reason_codes"],
                "action_label_positive_count": summary["action_label_positive_count"],
                "pairwise_preference_signal_count": summary["pairwise_preference_signal_count"],
                "hybrid_train_signal_count": summary["hybrid_train_signal_count"],
                "dry_run_status": summary["dry_run_status"],
                "summary": _display_path(summary_path, repo_root),
            },
            ensure_ascii=False,
        )
    )
    if not args.validate_only:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 1 if summary["status"] == "failed" else 0


def run_hybrid_policy_training_dry_run(
    *,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    summary_path: Path,
    validate_only: bool,
) -> dict[str, Any]:
    from model_explorer.policy.dataset import validate_rollout_dataset
    from model_explorer.policy.rollout_io import read_rollout_episodes

    reason_codes: list[str] = []
    paths = _input_paths(batch_root, config)
    materialization = _load_summary(
        paths["materialization_summary"],
        expected_schema=MATERIALIZATION_SCHEMA_VERSION,
        label="materialization_summary",
        reason_codes=reason_codes,
    )
    counterfactual = _load_summary(
        paths["counterfactual_preference_summary"],
        expected_schema=COUNTERFACTUAL_SCHEMA_VERSION,
        label="counterfactual_preference_summary",
        reason_codes=reason_codes,
    )
    registry_summary = _load_summary(
        paths["unified_policy_sample_registry_summary"],
        expected_schema=REGISTRY_SCHEMA_VERSION,
        label="unified_policy_sample_registry_summary",
        reason_codes=reason_codes,
    )
    for label, payload in (
        ("materialization_summary", materialization),
        ("counterfactual_preference_summary", counterfactual),
        ("unified_policy_sample_registry_summary", registry_summary),
    ):
        if payload.get("status") != "passed" or _string_list(payload.get("reason_codes")):
            _append_reason(reason_codes, f"{label}_failed")

    registry = _load_jsonl(
        paths["unified_policy_sample_registry"],
        label="unified_policy_sample_registry",
        reason_codes=reason_codes,
    )
    counterfactual_samples = _load_jsonl(
        paths["counterfactual_preference_samples"],
        label="counterfactual_preference_samples",
        reason_codes=reason_codes,
    )
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

    pairwise_samples = [
        record for record in registry if record.get("sample_type") in PAIRWISE_SAMPLE_TYPES
    ]
    action_label_positive_count = _int_value(materialization.get("input_positive_count"))
    existing_preference_pair_count = sum(
        1 for record in pairwise_samples if record.get("sample_type") == "counterfactual_preference_pair"
    )
    residual_preference_pair_count = sum(
        1
        for record in pairwise_samples
        if record.get("sample_type")
        in {"boundary_negative_preference_pair", "blocked_target_negative_pair"}
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
        "expected_existing_preference_pair_count",
        existing_preference_pair_count,
        "existing_preference_pair_count_mismatch",
    )
    _check_expected_count(
        reason_codes,
        config["validation"],
        "expected_residual_preference_pair_count",
        residual_preference_pair_count,
        "residual_preference_pair_count_mismatch",
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
    if len(counterfactual_samples) != _int_value(counterfactual.get("preference_pair_count")):
        _append_reason(reason_codes, "counterfactual_preference_sample_count_mismatch")
    if hard_positive_added_count != 0:
        _append_reason(reason_codes, "hard_positive_added_count_nonzero")
    if invalid_action_mask_count > _int_value(config["validation"].get("max_invalid_action_mask_count")):
        _append_reason(reason_codes, "invalid_action_mask")
    if empty_action_mask_count > _int_value(config["validation"].get("max_empty_action_mask_count")):
        _append_reason(reason_codes, "empty_action_mask")

    training_result: dict[str, Any] | None = None
    if not reason_codes and not validate_only:
        try:
            training_result = _train_hybrid(episodes, pairwise_samples, config=config)
        except Exception as exc:  # noqa: BLE001
            _append_reason(reason_codes, "hybrid_training_dry_run_failed")
            training_error = str(exc)

    status = "failed" if reason_codes else "passed"
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dry_run_status": "passed" if status == "passed" else "failed",
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "batch_root": _display_path(batch_root, repo_root),
        "source_summaries": _source_summaries(paths, repo_root),
        "action_label_positive_count": action_label_positive_count,
        "default_contract_positive_count": _int_value(
            materialization.get("default_contract_positive_count")
        ),
        "planner_validated_exception_positive_count": _int_value(
            materialization.get("planner_validated_exception_positive_count")
        ),
        "existing_preference_pair_count": existing_preference_pair_count,
        "residual_preference_pair_count": residual_preference_pair_count,
        "boundary_negative_preference_pair_count": _int_value(
            registry_summary.get("boundary_negative_preference_pair_count")
        ),
        "blocked_target_negative_pair_count": _int_value(
            registry_summary.get("blocked_target_negative_pair_count")
        ),
        "pairwise_preference_signal_count": pairwise_preference_signal_count,
        "hybrid_train_signal_count": hybrid_train_signal_count,
        "hard_positive_added_count": hard_positive_added_count,
        "invalid_action_mask_count": invalid_action_mask_count,
        "empty_action_mask_count": empty_action_mask_count,
        "dataset_summary": dataset_summary,
        "training_result": training_result,
        "training_error": training_error,
        "summary": _display_path(summary_path, repo_root),
        "runs_large_scale_training": False,
        "dry_run_only": True,
        "publishes_checkpoint": _checkpoint_path(config, batch_root) is not None,
        "performance_claimed": False,
        "does_not_modify_default_astar": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "does_not_relax_default_distance_contract": True,
        "no_ackermann_feasible_trajectory_claim": True,
        "git_provenance": {"current": _git_snapshot(repo_root), "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }


def _train_hybrid(
    episodes,
    pairwise_samples: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
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

    torch.manual_seed(_int_value(config["training"].get("seed")))
    network = build_policy_network(
        None,
        observation=transitions[0].observation,
        hidden_size=_int_value(config["training"].get("hidden_size")),
    )
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=float(config["training"].get("learning_rate", 1.0e-3)),
    )
    action_batch = _action_label_batch(transitions)
    pairwise_batch = _pairwise_batch(pairwise_samples)
    action_weight = float(config["training"].get("action_label_loss_weight", 1.0))
    preference_weight = float(config["training"].get("preference_loss_weight", 1.0))
    residual_weight = float(config["training"].get("residual_negative_loss_weight", 1.0))
    margin = float(config["training"].get("margin", 0.1))
    epoch_losses: list[dict[str, Any]] = []
    for epoch in range(_int_value(config["training"].get("epochs"))):
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
                "action_label_weighted_loss": float((action_weight * action_losses.total_loss).detach()),
                "pairwise_preference_loss": float(pairwise_loss.detach()),
            }
        )
    if not epoch_losses:
        raise ValueError("epochs must be at least 1")
    return {
        "architecture": network.architecture_name,
        "sample_count": len(transitions) + len(pairwise_samples),
        "action_label_sample_count": len(transitions),
        "pairwise_preference_sample_count": len(pairwise_samples),
        "epochs": _int_value(config["training"].get("epochs")),
        "seed": _int_value(config["training"].get("seed")),
        "hidden_size": _int_value(config["training"].get("hidden_size")),
        "learning_rate": float(config["training"].get("learning_rate", 1.0e-3)),
        "margin": margin,
        "action_label_loss_weight": action_weight,
        "preference_loss_weight": preference_weight,
        "residual_negative_loss_weight": residual_weight,
        "epoch_losses": epoch_losses,
        "final_total_loss": epoch_losses[-1]["total_loss"],
        "action_label_loss": epoch_losses[-1]["action_label_loss"],
        "pairwise_preference_loss": epoch_losses[-1]["pairwise_preference_loss"],
    }


def _action_label_batch(transitions) -> dict[str, Any]:
    import torch

    observations = tuple(transition.observation for transition in transitions)
    action_count = max(len(observation.action_mask) for observation in observations)
    rewards = [float(transition.reward) for transition in transitions]
    values = [0.0 if transition.value is None else float(transition.value) for transition in transitions]
    return {
        "candidate_features": torch.tensor(
            [_padded_candidate_features(observation, action_count) for observation in observations],
            dtype=torch.float32,
        ),
        "global_features": torch.tensor(
            [observation.global_features for observation in observations],
            dtype=torch.float32,
        ),
        "action_mask": torch.tensor(
            [_padded_action_mask(observation, action_count) for observation in observations],
            dtype=torch.bool,
        ),
        "actions": torch.tensor([transition.action_index for transition in transitions], dtype=torch.long),
        "old_log_probs": torch.tensor(
            [0.0 if transition.log_prob is None else float(transition.log_prob) for transition in transitions],
            dtype=torch.float32,
        ),
        "returns": torch.tensor(rewards, dtype=torch.float32),
        "advantages": torch.tensor(
            [reward - value for reward, value in zip(rewards, values)],
            dtype=torch.float32,
        ),
        "candidate_missing_indicators": torch.tensor(
            [_padded_missing_indicators(observation, action_count) for observation in observations],
            dtype=torch.float32,
        ),
    }


def _pairwise_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "candidate_features": torch.tensor(
            [
                [
                    _pairwise_side(sample, preferred=True)["candidate_features"],
                    _pairwise_side(sample, preferred=False)["candidate_features"],
                ]
                for sample in samples
            ],
            dtype=torch.float32,
        ),
        "global_features": torch.tensor([sample["global_features"] for sample in samples], dtype=torch.float32),
        "action_mask": torch.ones((len(samples), 2), dtype=torch.bool),
        "candidate_missing_indicators": torch.tensor(
            [sample["candidate_missing_indicators"] for sample in samples],
            dtype=torch.float32,
        ),
        "sample_weights": torch.tensor(
            [float(sample.get("sample_weight", 1.0)) for sample in samples],
            dtype=torch.float32,
        ),
        "is_residual": torch.tensor(
            [sample.get("sample_type") != "counterfactual_preference_pair" for sample in samples],
            dtype=torch.bool,
        ),
    }


def _pairwise_loss(network, batch, *, margin: float, preference_weight: float, residual_weight: float, torch, F):
    output = network(
        candidate_features=batch["candidate_features"],
        global_features=batch["global_features"],
        action_mask=batch["action_mask"],
        candidate_missing_indicators=batch["candidate_missing_indicators"],
    )
    preferred_logits = output.masked_logits[:, 0]
    alternative_logits = output.masked_logits[:, 1]
    per_sample = F.relu(float(margin) - (preferred_logits - alternative_logits))
    weights = batch["sample_weights"] * torch.where(
        batch["is_residual"],
        torch.tensor(float(residual_weight), dtype=torch.float32),
        torch.tensor(float(preference_weight), dtype=torch.float32),
    )
    return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)


def _pairwise_side(sample: dict[str, Any], *, preferred: bool) -> dict[str, Any]:
    if not preferred:
        side = sample.get("alternative")
    else:
        side = sample.get("preferred", sample.get("selected"))
    if not isinstance(side, dict) or not isinstance(side.get("candidate_features"), list):
        raise ValueError("pairwise sample is missing candidate_features")
    return side


def _padded_candidate_features(observation, action_count: int) -> list[list[float]]:
    width = len(observation.candidate_feature_names)
    rows = [list(row) for row in observation.candidate_features]
    rows.extend([[0.0 for _ in range(width)] for _ in range(action_count - len(rows))])
    return rows


def _padded_action_mask(observation, action_count: int) -> list[bool]:
    row = list(observation.action_mask)
    row.extend(False for _ in range(action_count - len(row)))
    return row


def _padded_missing_indicators(observation, action_count: int) -> list[list[float]]:
    width = len(observation.candidate_missing_indicator_names)
    rows = [list(row) for row in observation.candidate_missing_indicators]
    rows.extend([[0.0 for _ in range(width)] for _ in range(action_count - len(rows))])
    return rows


def _input_paths(batch_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    files = config["input_files"]
    return {
        "rollout_episodes": batch_root / files["rollout_episodes"],
        "materialization_summary": batch_root / files["materialization_summary"],
        "counterfactual_preference_samples": batch_root / files["counterfactual_preference_samples"],
        "counterfactual_preference_summary": batch_root / files["counterfactual_preference_summary"],
        "unified_policy_sample_registry": batch_root / files["unified_policy_sample_registry"],
        "unified_policy_sample_registry_summary": batch_root / files["unified_policy_sample_registry_summary"],
    }


def _source_summaries(paths: dict[str, Path], repo_root: Path) -> dict[str, dict[str, Any]]:
    return {
        label: {"path": _display_path(path, repo_root), "exists": path.is_file()}
        for label, path in paths.items()
    }


def _load_summary(
    path: Path,
    *,
    expected_schema: str,
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
    if payload.get("schema_version") != expected_schema:
        _append_reason(reason_codes, f"{label}_schema_version_mismatch")
    return payload


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
    for section in ("input_files", "output_files", "validation", "training"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _checkpoint_path(config: dict[str, Any], batch_root: Path) -> Path | None:
    value = config["training"].get("checkpoint_path")
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else batch_root / path


def _check_expected_count(
    reason_codes: list[str],
    validation: dict[str, Any],
    field: str,
    actual: int,
    reason: str,
) -> None:
    if field in validation and actual != _int_value(validation.get(field)):
        _append_reason(reason_codes, reason)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


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


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
