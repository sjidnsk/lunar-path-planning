from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from git_provenance import git_snapshot as _git_snapshot

try:
    from scripts.run_limited_ppo_update_smoke import run_limited_ppo_update_smoke
except ModuleNotFoundError:
    from run_limited_ppo_update_smoke import run_limited_ppo_update_smoke


CONFIG_SCHEMA_VERSION = "limited-quasi-real-ppo-update-smoke-config/v1"
SUMMARY_SCHEMA_VERSION = "limited-quasi-real-ppo-update-smoke-summary/v1"
GENERIC_UPDATE_SCHEMA_VERSION = "limited-ppo-update-smoke-config/v1"

NEXT_INPUT_INVALID = "limited_quasi_real_ppo_update_input_contract_invalid"
NEXT_POST_UPDATE_GATE_REGRESSION = "limited_quasi_real_ppo_update_post_update_gate_regression"

PostUpdateRunner = Callable[[dict[str, Any]], dict[str, Any]]


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a limited quasi-real PPO update smoke test.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--base-candidate-root", required=True)
    parser.add_argument("--collector-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--quasi-real-root", default=None)
    parser.add_argument("--guarded-teacher-following-root", default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    try:
        config = _load_config(_resolve_path(args.config, repo_root))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.validate_only:
        print(json.dumps({"status": "config validated", "config": args.config}, ensure_ascii=False))
        return 0

    summary = run_limited_quasi_real_ppo_update_smoke(
        source_root=_resolve_path(args.source_root, repo_root),
        base_candidate_root=_resolve_path(args.base_candidate_root, repo_root),
        collector_root=_resolve_path(args.collector_root, repo_root),
        output_root=_resolve_path(args.output_root, repo_root),
        config=config,
        repo_root=repo_root,
        quasi_real_root=None if args.quasi_real_root is None else _resolve_path(args.quasi_real_root, repo_root),
        guarded_teacher_following_root=(
            None
            if args.guarded_teacher_following_root is None
            else _resolve_path(args.guarded_teacher_following_root, repo_root)
        ),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "input_ppo_trainable_transition_count": summary["input_ppo_trainable_transition_count"],
                "optimizer_train_transition_count": summary["optimizer_train_transition_count"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def run_limited_quasi_real_ppo_update_smoke(
    *,
    source_root: Path,
    base_candidate_root: Path,
    collector_root: Path,
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    quasi_real_root: Path | None = None,
    guarded_teacher_following_root: Path | None = None,
    post_update_runner: PostUpdateRunner | None = None,
) -> dict[str, Any]:
    _install_model_explorer_path(repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root, config)
    collector_summary = _load_json(collector_root / config["input_files"]["collector_summary"])
    transition_records = _read_jsonl(collector_root / config["input_files"].get("rollout_transitions", "ppo-rollout-transitions.jsonl"))
    if quasi_real_root is None:
        quasi_real_root = _resolve_optional_root(collector_summary.get("quasi_real_root"), repo_root)
    if guarded_teacher_following_root is None:
        guarded_teacher_following_root = _resolve_optional_root(
            collector_summary.get("guarded_teacher_following_root"),
            repo_root,
        )

    update_summary = run_limited_ppo_update_smoke(
        source_root=source_root,
        base_candidate_root=base_candidate_root,
        collector_root=collector_root,
        output_root=output_root,
        config=_generic_update_config(config),
        repo_root=repo_root,
    )
    post_update_summaries: dict[str, Any] = {}
    if update_summary.get("status") == "passed" and config.get("post_update_gates", {}).get("enabled", True):
        context = {
            "source_root": source_root,
            "base_candidate_root": base_candidate_root,
            "collector_root": collector_root,
            "output_root": output_root,
            "updated_candidate_root": output_root,
            "quasi_real_root": quasi_real_root,
            "guarded_teacher_following_root": guarded_teacher_following_root,
            "config": config,
            "repo_root": repo_root,
        }
        runner = post_update_runner or _run_post_update_gates
        post_update_summaries = runner(context)

    reason_codes = []
    for reason in update_summary.get("reason_codes", []):
        _append_reason(reason_codes, str(reason))
    for reason in _collector_contract_reason_codes(collector_summary, config):
        _append_reason(reason_codes, reason)
    if update_summary.get("status") == "passed":
        for reason in _post_update_reason_codes(post_update_summaries, config):
            _append_reason(reason_codes, reason)

    audit = _transition_audit(update_summary=update_summary, transition_records=transition_records, config=config)
    for field, reason in (
        ("validation_test_optimizer_transition_count", NEXT_INPUT_INVALID),
        ("non_empty_gate_reason_optimizer_transition_count", NEXT_INPUT_INVALID),
        ("disallowed_source_optimizer_transition_count", NEXT_INPUT_INVALID),
    ):
        if int(audit.get(field, 0)):
            _append_reason(reason_codes, reason)

    status = "failed" if reason_codes else "passed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": None if status == "passed" else _next_required_change(reason_codes),
        "source_root": _display_path(source_root, repo_root),
        "base_candidate_root": _display_path(base_candidate_root, repo_root),
        "collector_root": _display_path(collector_root, repo_root),
        "quasi_real_root": _display_path(quasi_real_root, repo_root) if quasi_real_root else None,
        "guarded_teacher_following_root": (
            _display_path(guarded_teacher_following_root, repo_root)
            if guarded_teacher_following_root
            else None
        ),
        "output_root": _display_path(output_root, repo_root),
        "input_collector_summary": _display_path(collector_root / config["input_files"]["collector_summary"], repo_root),
        "input_collector_status": collector_summary.get("status"),
        "input_collector_reason_codes": list(collector_summary.get("reason_codes", [])),
        "collector_episode_count": _int_value(collector_summary.get("episode_count")),
        "collector_step_count": _int_value(collector_summary.get("step_count")),
        "collector_diagnostic_transition_count": _int_value(collector_summary.get("diagnostic_transition_count")),
        "input_ppo_trainable_transition_count": _int_value(update_summary.get("input_ppo_trainable_transition_count")),
        "optimizer_train_transition_count": _int_value(update_summary.get("optimizer_train_transition_count")),
        "collector_ppo_trainable_transition_count": _int_value(collector_summary.get("ppo_trainable_transition_count")),
        "source_fallback_trainable_count": _int_value(update_summary.get("source_fallback_trainable_count")),
        "optimizer_transition_split_counts": audit["optimizer_transition_split_counts"],
        "optimizer_transition_source_counts": audit["optimizer_transition_source_counts"],
        "transition_record_split_counts": audit["transition_record_split_counts"],
        "transition_record_source_counts": audit["transition_record_source_counts"],
        "validation_test_optimizer_transition_count": audit["validation_test_optimizer_transition_count"],
        "non_empty_gate_reason_optimizer_transition_count": audit["non_empty_gate_reason_optimizer_transition_count"],
        "disallowed_source_optimizer_transition_count": audit["disallowed_source_optimizer_transition_count"],
        "old_log_prob_max_abs_error": update_summary.get("old_log_prob_max_abs_error"),
        "old_value_max_abs_error": update_summary.get("old_value_max_abs_error"),
        "loss_non_finite_count": _int_value(update_summary.get("loss_non_finite_count")),
        "non_finite_gradient_count": _int_value(update_summary.get("non_finite_gradient_count")),
        "non_finite_reward_count": _int_value(update_summary.get("non_finite_reward_count")),
        "non_finite_return_count": _int_value(update_summary.get("non_finite_return_count")),
        "non_finite_advantage_count": _int_value(update_summary.get("non_finite_advantage_count")),
        "parameter_l2_delta": update_summary.get("parameter_l2_delta"),
        "approx_kl": update_summary.get("approx_kl"),
        "max_grad_norm_after_clip": update_summary.get("max_grad_norm_after_clip"),
        "limited_ppo_update_smoke_summary": _display_path(paths["update_summary"], repo_root),
        "training_curves": _display_path(paths["training_curves"], repo_root),
        "diagnostics": _display_path(paths["diagnostics"], repo_root),
        "checkpoint_path": _display_path(paths["checkpoint"], repo_root),
        "checkpoint_metadata_path": _display_path(paths["checkpoint_metadata"], repo_root),
        "candidate_summary_path": _display_path(paths["candidate_summary"], repo_root),
        **_post_update_summary_fields(post_update_summaries, repo_root=repo_root),
        "experimental_checkpoint": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "runs_formal_ppo_rollout": False,
        "git_provenance": {"current": _git_snapshot(repo_root), "current_matches_sources": True},
        "summary": _display_path(paths["summary"], repo_root),
        "non_goals": list(config.get("non_goals", [])),
    }
    _write_json(paths["summary"], summary)
    return summary


def _generic_update_config(config: dict[str, Any]) -> dict[str, Any]:
    outputs = config["output_files"]
    return {
        "schema_version": GENERIC_UPDATE_SCHEMA_VERSION,
        "input_files": {
            "rollout_episodes": config["input_files"].get("rollout_episodes", "ppo-rollout-episodes.jsonl"),
            "collector_summary": config["input_files"].get("collector_summary", "ppo-rollout-collector-summary.json"),
            "base_checkpoint": config["input_files"].get("base_checkpoint", "experimental-hybrid-policy-candidate.pt"),
            "base_checkpoint_metadata": config["input_files"].get(
                "base_checkpoint_metadata",
                "experimental-hybrid-policy-candidate-metadata.json",
            ),
            "base_candidate_summary": config["input_files"].get(
                "base_candidate_summary",
                "raw-policy-generalization-candidate-summary.json",
            ),
        },
        "output_files": {
            "summary": outputs.get("update_summary", "limited-ppo-update-smoke-summary.json"),
            "training_curves": outputs.get("training_curves", "limited-ppo-update-training-curves.json"),
            "diagnostics": outputs.get("diagnostics", "limited-ppo-update-diagnostics.json"),
            "checkpoint": outputs.get("checkpoint", "experimental-hybrid-policy-candidate.pt"),
            "checkpoint_metadata": outputs.get(
                "checkpoint_metadata",
                "experimental-hybrid-policy-candidate-metadata.json",
            ),
            "candidate_summary": outputs.get("candidate_summary", "raw-policy-generalization-candidate-summary.json"),
        },
        "training": dict(config.get("training", {})),
        "validation": dict(config.get("validation", {})),
        "evaluation": dict(config.get("evaluation", {})),
        "trainable_filter": dict(config.get("trainable_filter", {})),
        "non_goals": list(config.get("non_goals", [])),
    }


def _transition_audit(
    *,
    update_summary: dict[str, Any],
    transition_records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    optimizer_split_counts = dict(update_summary.get("optimizer_transition_split_counts") or {})
    optimizer_source_counts = dict(update_summary.get("optimizer_transition_source_counts") or {})
    allowed_sources = set(config.get("trainable_filter", {}).get("controlled_choice_sources", []))
    return {
        "optimizer_transition_split_counts": optimizer_split_counts,
        "optimizer_transition_source_counts": optimizer_source_counts,
        "transition_record_split_counts": dict(sorted(Counter(str(r.get("split") or "unknown") for r in transition_records).items())),
        "transition_record_source_counts": dict(
            sorted(Counter(str(r.get("controlled_choice_source") or "unknown") for r in transition_records).items())
        ),
        "validation_test_optimizer_transition_count": sum(
            count for split, count in optimizer_split_counts.items() if split in {"validation", "test"}
        ),
        "non_empty_gate_reason_optimizer_transition_count": _int_value(
            update_summary.get("non_empty_gate_reason_optimizer_transition_count")
        ),
        "disallowed_source_optimizer_transition_count": sum(
            count for source, count in optimizer_source_counts.items() if allowed_sources and source not in allowed_sources
        ),
    }


def _collector_contract_reason_codes(summary: dict[str, Any], config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    validation = config.get("validation", {})
    if summary.get("status") != "passed" or summary.get("reason_codes"):
        reasons.append(NEXT_INPUT_INVALID)
    if _int_value(summary.get("ppo_trainable_transition_count")) < _int_value(
        validation.get("min_optimizer_train_transition_count"),
        1,
    ):
        reasons.append(NEXT_INPUT_INVALID)
    for field in (
        "source_fallback_trainable_count",
        "invalid_action_mask_count",
        "empty_action_mask_count",
        "missing_log_prob_count",
        "missing_value_count",
        "non_finite_reward_count",
        "fallback_or_open_grid_count",
        "safety_regression_count",
        "contract_violation_count",
        "path_cost_regression_count",
        "risk_regression_count",
        "source_selection_regression_count",
    ):
        if _int_value(summary.get(field)):
            reasons.append(NEXT_INPUT_INVALID)
    return _dedup(reasons)


def _post_update_reason_codes(post: dict[str, Any], config: dict[str, Any]) -> list[str]:
    if not config.get("post_update_gates", {}).get("enabled", True):
        return []
    reasons: list[str] = []
    validation = config.get("validation", {})
    raw = _summary(post, "raw_generalization")
    sequential = _summary(post, "sequential_canary")
    generated_collector = _summary(post, "generated_collector")
    teacher = _summary(post, "quasi_real_teacher_following")
    quasi_collector = _summary(post, "quasi_real_collector")
    for item in (sequential, generated_collector, teacher, quasi_collector):
        if item.get("status") != "passed" or item.get("reason_codes"):
            reasons.append(NEXT_POST_UPDATE_GATE_REGRESSION)
    if _raw_generalization_regressed(raw):
        reasons.append(NEXT_POST_UPDATE_GATE_REGRESSION)
    if _int_value(raw.get("raw_test_regression_count")):
        reasons.append(NEXT_POST_UPDATE_GATE_REGRESSION)
    if _int_value(sequential.get("rejected_choice_count")):
        reasons.append(NEXT_POST_UPDATE_GATE_REGRESSION)
    if _int_value(generated_collector.get("ppo_trainable_transition_count")) < _int_value(
        validation.get("min_post_update_ppo_trainable_transition_count"),
        24,
    ):
        reasons.append(NEXT_POST_UPDATE_GATE_REGRESSION)
    if float(teacher.get("teacher_agreement_rate", 0.0)) < float(
        validation.get("min_post_update_teacher_agreement_rate", 0.9)
    ):
        reasons.append(NEXT_POST_UPDATE_GATE_REGRESSION)
    if _int_value(teacher.get("unsafe_disagreement_count")) or _int_value(
        teacher.get("policy_changed_gate_rejected_count")
    ):
        reasons.append(NEXT_POST_UPDATE_GATE_REGRESSION)
    if _int_value(quasi_collector.get("ppo_trainable_transition_count")) < _int_value(
        validation.get("min_post_update_ppo_trainable_transition_count"),
        24,
    ):
        reasons.append(NEXT_POST_UPDATE_GATE_REGRESSION)
    for summary in (generated_collector, quasi_collector):
        for field in (
            "source_fallback_trainable_count",
            "invalid_action_mask_count",
            "empty_action_mask_count",
            "missing_log_prob_count",
            "missing_value_count",
            "non_finite_reward_count",
            "fallback_or_open_grid_count",
            "safety_regression_count",
            "contract_violation_count",
            "path_cost_regression_count",
            "risk_regression_count",
            "source_selection_regression_count",
        ):
            if _int_value(summary.get(field)):
                reasons.append(NEXT_POST_UPDATE_GATE_REGRESSION)
    return _dedup(reasons)


def _post_update_summary_fields(post: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    raw = _summary(post, "raw_generalization")
    sequential = _summary(post, "sequential_canary")
    generated_collector = _summary(post, "generated_collector")
    teacher = _summary(post, "quasi_real_teacher_following")
    quasi_collector = _summary(post, "quasi_real_collector")
    return {
        "post_update_raw_generalization_summary": _summary_path(raw, repo_root),
        "post_update_raw_generalization_status": _effective_raw_generalization_status(raw),
        "post_update_raw_generalization_original_status": raw.get("status"),
        "post_update_raw_generalization_reason_codes": list(raw.get("reason_codes", [])),
        "post_update_raw_test_regression_count": _int_value(raw.get("raw_test_regression_count")),
        "post_update_sequential_canary_summary": _summary_path(sequential, repo_root),
        "post_update_sequential_canary_status": sequential.get("status"),
        "post_update_sequential_rejected_choice_count": _int_value(sequential.get("rejected_choice_count")),
        "post_update_generated_collector_summary": _summary_path(generated_collector, repo_root),
        "post_update_generated_collector_status": generated_collector.get("status"),
        "post_update_generated_collector_trainable_transition_count": _int_value(
            generated_collector.get("ppo_trainable_transition_count")
        ),
        "post_update_quasi_real_teacher_following_summary": _summary_path(teacher, repo_root),
        "post_update_quasi_real_teacher_following_status": teacher.get("status"),
        "post_update_quasi_real_teacher_agreement_rate": float(teacher.get("teacher_agreement_rate", 0.0)),
        "post_update_quasi_real_unsafe_disagreement_count": _int_value(teacher.get("unsafe_disagreement_count")),
        "post_update_quasi_real_collector_summary": _summary_path(quasi_collector, repo_root),
        "post_update_quasi_real_collector_status": quasi_collector.get("status"),
        "post_update_quasi_real_collector_trainable_transition_count": _int_value(
            quasi_collector.get("ppo_trainable_transition_count")
        ),
    }


def _run_post_update_gates(context: dict[str, Any]) -> dict[str, Any]:
    config = context["config"]
    repo_root = Path(context["repo_root"])
    output_root = Path(context["output_root"])
    source_root = Path(context["source_root"])
    updated_candidate_root = Path(context["updated_candidate_root"])
    quasi_real_root = Path(context["quasi_real_root"])
    gates = config.get("post_update_gates", {})
    generated = gates.get("generated", {})
    quasi = gates.get("quasi_real", {})

    sequential_root = output_root / generated.get("sequential_root", "post_update_generated_sequential")
    generated_collector_root = output_root / generated.get("collector_root", "post_update_generated_collector")
    quasi_teacher_root = output_root / quasi.get("teacher_following_root", "post_update_quasi_real_teacher_following")
    quasi_collector_root = output_root / quasi.get("collector_root", "post_update_quasi_real_collector")
    for path in (sequential_root, generated_collector_root, quasi_teacher_root, quasi_collector_root):
        if path.exists():
            shutil.rmtree(path)

    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_raw_policy_generalization_evaluation.py"),
            "--source-root",
            str(source_root),
            "--dev-root",
            str(_resolve_path(generated.get("dev_root", "outputs/path_feedback_batch_sequential_multi_step_opportunity_dev_v1"), repo_root)),
            "--val-root",
            str(_resolve_path(generated.get("val_root", "outputs/path_feedback_batch_sequential_multi_step_opportunity_val_v1"), repo_root)),
            "--test-root",
            str(_resolve_path(generated.get("test_root", "outputs/path_feedback_batch_sequential_multi_step_opportunity_test_v1"), repo_root)),
            "--baseline-candidate-root",
            str(
                _resolve_path(
                    generated.get(
                        "raw_baseline_candidate_root",
                        "outputs/path_feedback_batch_sequential_multi_step_opportunity_baseline_candidate_v1",
                    ),
                    repo_root,
                )
            ),
            "--candidate-root",
            str(updated_candidate_root),
            "--config",
            str(_resolve_path(generated.get("raw_generalization_config", "configs/raw_policy_generalization_evaluation_v1.json"), repo_root)),
        ]
    )
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_policy_gated_sequential_canary_rollout.py"),
            "--source-root",
            str(source_root),
            "--candidate-root",
            str(updated_candidate_root),
            "--batch-root",
            str(sequential_root),
            "--config",
            str(
                _resolve_path(
                    generated.get(
                        "sequential_config",
                        "configs/policy_gated_sequential_multi_step_opportunity_rollout_v1.json",
                    ),
                    repo_root,
                )
            ),
        ]
    )
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_ppo_rollout_collector_dry_run.py"),
            "--sequential-root",
            str(sequential_root),
            "--candidate-root",
            str(updated_candidate_root),
            "--output-root",
            str(generated_collector_root),
            "--config",
            str(_resolve_path(generated.get("collector_config", "configs/ppo_rollout_collector_dry_run_v1.json"), repo_root)),
        ]
    )
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_quasi_real_guarded_teacher_following_pilot.py"),
            "--source-root",
            str(source_root),
            "--candidate-root",
            str(updated_candidate_root),
            "--quasi-real-root",
            str(quasi_real_root),
            "--output-root",
            str(quasi_teacher_root),
            "--config",
            str(
                _resolve_path(
                    quasi.get("teacher_following_config", "configs/quasi_real_guarded_teacher_following_pilot_v1.json"),
                    repo_root,
                )
            ),
        ]
    )
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_quasi_real_ppo_collector_dry_run.py"),
            "--guarded-teacher-following-root",
            str(quasi_teacher_root),
            "--candidate-root",
            str(updated_candidate_root),
            "--quasi-real-root",
            str(quasi_real_root),
            "--output-root",
            str(quasi_collector_root),
            "--config",
            str(_resolve_path(quasi.get("collector_config", "configs/quasi_real_ppo_collector_dry_run_v1.json"), repo_root)),
        ]
    )
    return {
        "raw_generalization": _load_json(updated_candidate_root / "raw-policy-generalization-evaluation-summary.json"),
        "sequential_canary": _load_json(sequential_root / "policy-gated-sequential-canary-rollout-summary.json"),
        "generated_collector": _load_json(generated_collector_root / "ppo-rollout-collector-summary.json"),
        "quasi_real_teacher_following": _load_json(
            quasi_teacher_root / "quasi-real-guarded-teacher-following-pilot-summary.json"
        ),
        "quasi_real_collector": _load_json(quasi_collector_root / "ppo-rollout-collector-summary.json"),
    }


def _paths(output_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["output_files"]
    return {
        "summary": output_root / outputs.get("summary", "limited-quasi-real-ppo-update-smoke-summary.json"),
        "update_summary": output_root / outputs.get("update_summary", "limited-ppo-update-smoke-summary.json"),
        "training_curves": output_root / outputs.get("training_curves", "limited-ppo-update-training-curves.json"),
        "diagnostics": output_root / outputs.get("diagnostics", "limited-ppo-update-diagnostics.json"),
        "checkpoint": output_root / outputs.get("checkpoint", "experimental-hybrid-policy-candidate.pt"),
        "checkpoint_metadata": output_root / outputs.get(
            "checkpoint_metadata",
            "experimental-hybrid-policy-candidate-metadata.json",
        ),
        "candidate_summary": output_root / outputs.get("candidate_summary", "raw-policy-generalization-candidate-summary.json"),
    }


def _load_config(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    for section in ("input_files", "output_files", "training", "validation", "trainable_filter"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _summary(post: dict[str, Any], key: str) -> dict[str, Any]:
    value = post.get(key)
    return value if isinstance(value, dict) else {}


def _summary_path(summary: dict[str, Any], repo_root: Path) -> str | None:
    for key in ("summary", "summary_output"):
        if summary.get(key):
            return _display_path(Path(str(summary[key])), repo_root)
    return None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _run(command: list[str]) -> None:
    subprocess.run(command, check=False)


def _raw_generalization_regressed(summary: dict[str, Any]) -> bool:
    if summary.get("status") == "passed":
        return False
    reason_codes = [str(reason) for reason in summary.get("reason_codes", [])]
    baseline_only = bool(reason_codes) and all(reason.startswith("baseline_") for reason in reason_codes)
    return not (baseline_only and _int_value(summary.get("test_raw_policy_regression_count")) == 0)


def _effective_raw_generalization_status(summary: dict[str, Any]) -> str | None:
    if not summary:
        return None
    return "failed" if _raw_generalization_regressed(summary) else "passed"


def _resolve_optional_root(value: Any, repo_root: Path) -> Path | None:
    if not value:
        return None
    return _resolve_path(str(value), repo_root)


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _install_model_explorer_path(repo_root: Path) -> None:
    source = repo_root / "model-explorer" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _dedup(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _next_required_change(reason_codes: list[str]) -> str:
    if "ppo_update_not_on_collector_policy" in reason_codes:
        return "ppo_update_not_on_collector_policy"
    if NEXT_POST_UPDATE_GATE_REGRESSION in reason_codes:
        return NEXT_POST_UPDATE_GATE_REGRESSION
    return reason_codes[0] if reason_codes else NEXT_INPUT_INVALID


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
