from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG_SCHEMA_VERSION = "counterfactual-preference-training-dry-run-config/v1"
SUMMARY_SCHEMA_VERSION = "counterfactual-preference-training-dry-run-summary/v1"
SAMPLES_SUMMARY_SCHEMA_VERSION = "counterfactual-preference-training-summary/v1"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a local pairwise preference dry-run.")
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples")
    parser.add_argument("--summary")
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
    input_files = config["input_files"]
    samples_path = _resolve_path(args.samples, repo_root) if args.samples else batch_root / input_files["samples"]
    samples_summary_path = (
        _resolve_path(args.summary, repo_root) if args.summary else batch_root / input_files["summary"]
    )
    output_path = batch_root / config["output_files"]["summary"]
    summary = run_counterfactual_preference_dry_run(
        batch_root=batch_root,
        samples_path=samples_path,
        samples_summary_path=samples_summary_path,
        output_path=output_path,
        config=config,
        repo_root=repo_root,
        validate_only=args.validate_only,
    )
    print(
        json.dumps(
            {
                "status": "config validated" if summary["status"] == "passed" else "validation failed",
                "batch_root": _display_path(batch_root, repo_root),
                "reason_codes": summary["reason_codes"],
                "preference_train_sample_count": summary["preference_train_sample_count"],
                "preference_dry_run_status": summary["preference_dry_run_status"],
                "summary": _display_path(output_path, repo_root),
            },
            ensure_ascii=False,
        )
    )
    if not args.validate_only:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 1 if summary["status"] == "failed" else 0


def run_counterfactual_preference_dry_run(
    *,
    batch_root: Path,
    samples_path: Path,
    samples_summary_path: Path,
    output_path: Path,
    config: dict[str, Any],
    repo_root: Path,
    validate_only: bool,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    samples_summary = _load_summary(samples_summary_path, reason_codes=reason_codes)
    if samples_summary.get("status") != "passed":
        _append_reason(reason_codes, "counterfactual_preference_summary_failed")
    samples = _load_samples(samples_path, reason_codes=reason_codes)
    expected_count = config["validation"].get("expected_preference_pair_count")
    if expected_count is not None and len(samples) != _int_value(expected_count):
        _append_reason(reason_codes, "preference_pair_count_mismatch")
    training_result: dict[str, Any] | None = None
    training_error: str | None = None
    if not reason_codes and not validate_only:
        try:
            training_result = _train_pairwise_preference(samples, config=config)
        except Exception as exc:  # noqa: BLE001
            _append_reason(reason_codes, "preference_training_dry_run_failed")
            training_error = str(exc)
    status = "failed" if reason_codes else "passed"
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "preference_dry_run_status": "passed" if status == "passed" else "failed",
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "batch_root": _display_path(batch_root, repo_root),
        "source_summaries": {
            "counterfactual_preference_training_summary": {
                "path": _display_path(samples_summary_path, repo_root),
                "exists": samples_summary_path.is_file(),
                "schema_version": samples_summary.get("schema_version"),
                "status": samples_summary.get("status"),
            },
            "counterfactual_preference_training_samples": {
                "path": _display_path(samples_path, repo_root),
                "exists": samples_path.is_file(),
            },
        },
        "preference_train_sample_count": len(samples),
        "selected_over_alternative_negative_count": _int_value(
            samples_summary.get("selected_over_alternative_negative_count")
        ),
        "tradeoff_preference_pair_count": _int_value(
            samples_summary.get("tradeoff_preference_pair_count")
        ),
        "rejected_binding_or_distance_required_count": _int_value(
            samples_summary.get("rejected_binding_or_distance_required_count")
        ),
        "hard_positive_added_count": _int_value(samples_summary.get("hard_positive_added_count")),
        "training_result": training_result,
        "training_error": training_error,
        "summary": _display_path(output_path, repo_root),
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
        "non_goals": list(config.get("non_goals", [])),
    }


def _train_pairwise_preference(samples: list[dict[str, Any]], *, config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F

    from model_explorer.policy.architectures import build_policy_network_from_metadata

    torch.manual_seed(_int_value(config["training"].get("seed")))
    candidate_features = torch.tensor(
        [
            [sample["selected"]["candidate_features"], sample["alternative"]["candidate_features"]]
            for sample in samples
        ],
        dtype=torch.float32,
    )
    global_features = torch.tensor([sample["global_features"] for sample in samples], dtype=torch.float32)
    action_mask = torch.ones((len(samples), 2), dtype=torch.bool)
    missing = torch.tensor(
        [sample["candidate_missing_indicators"] for sample in samples],
        dtype=torch.float32,
    )
    weights = torch.tensor([float(sample.get("sample_weight", 1.0)) for sample in samples], dtype=torch.float32)
    network = build_policy_network_from_metadata(
        None,
        candidate_feature_count=candidate_features.shape[-1],
        global_feature_count=global_features.shape[-1],
        missing_indicator_count=missing.shape[-1],
        hidden_size=_int_value(config["training"].get("hidden_size")),
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=float(config["training"].get("learning_rate", 1.0e-3)))
    margin = float(config["training"].get("margin", 0.1))
    epoch_losses: list[dict[str, Any]] = []
    epochs = _int_value(config["training"].get("epochs"))
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = network(
            candidate_features=candidate_features,
            global_features=global_features,
            action_mask=action_mask,
            candidate_missing_indicators=missing,
        )
        selected_logits = output.masked_logits[:, 0]
        alternative_logits = output.masked_logits[:, 1]
        per_sample = F.relu(margin - (selected_logits - alternative_logits))
        loss = (per_sample * weights).sum() / weights.sum().clamp_min(1.0)
        loss.backward()
        optimizer.step()
        epoch_losses.append(
            {
                "epoch": epoch + 1,
                "pairwise_preference_loss": float(loss.detach()),
                "mean_selected_minus_alternative_logit": float(
                    (selected_logits - alternative_logits).detach().mean()
                ),
            }
        )
    if not epoch_losses:
        raise ValueError("epochs must be at least 1")
    return {
        "architecture": network.architecture_name,
        "sample_count": len(samples),
        "epochs": epochs,
        "seed": _int_value(config["training"].get("seed")),
        "hidden_size": _int_value(config["training"].get("hidden_size")),
        "learning_rate": float(config["training"].get("learning_rate", 1.0e-3)),
        "margin": margin,
        "epoch_losses": epoch_losses,
        "final_pairwise_preference_loss": epoch_losses[-1]["pairwise_preference_loss"],
    }


def _load_summary(path: Path, *, reason_codes: list[str]) -> dict[str, Any]:
    if not path.is_file():
        _append_reason(reason_codes, "counterfactual_preference_summary_missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _append_reason(reason_codes, "counterfactual_preference_summary_invalid_json")
        return {}
    if not isinstance(payload, dict):
        _append_reason(reason_codes, "counterfactual_preference_summary_invalid_json_root")
        return {}
    if payload.get("schema_version") != SAMPLES_SUMMARY_SCHEMA_VERSION:
        _append_reason(reason_codes, "counterfactual_preference_summary_schema_version_mismatch")
    return payload


def _load_samples(path: Path, *, reason_codes: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        _append_reason(reason_codes, "counterfactual_preference_samples_missing")
        return []
    samples: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                samples.append(payload)
    except json.JSONDecodeError:
        _append_reason(reason_codes, "counterfactual_preference_samples_invalid_json")
    return samples


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
