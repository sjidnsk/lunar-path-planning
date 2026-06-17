from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot


CONFIG_SCHEMA_VERSION = "guarded-formal-ppo-post-training-stability-replay-config/v1"
SUMMARY_SCHEMA_VERSION = "guarded-formal-ppo-post-training-stability-replay-summary/v1"
REPLAY_ROW_SCHEMA_VERSION = "guarded-formal-ppo-post-training-stability-replay-row/v1"
EXPECTED_READINESS_STATUS = "guarded_formal_ppo_post_training_stability_replay_evaluated"

TRAINING_RUN_SUMMARY_FILE = "formal-ppo-training-run-summary.json"
TRAINING_RUN_SEED_SUMMARIES_FILE = "formal-ppo-training-run-seed-summaries.jsonl"
SUMMARY_FILE = "formal-ppo-post-training-stability-replay-summary.json"
SEED_SUMMARIES_FILE = "formal-ppo-post-training-stability-replay-seed-summaries.jsonl"
PROGRESS_FILE = "formal-ppo-post-training-stability-replay-progress.jsonl"
DRIFT_REPORT_FILE = "formal-ppo-post-training-stability-replay-drift-report.jsonl"
GATE_AUDIT_FILE = "formal-ppo-post-training-stability-replay-gate-audit.json"
ROLLBACK_MANIFEST_FILE = "formal-ppo-post-training-stability-replay-rollback-manifest.json"
READINESS_FILE = "formal-ppo-post-training-stability-replay-readiness-validate-only.json"
REPORT_FILE = "formal-ppo-post-training-stability-replay-report.md"

ReplayRunner = Callable[..., dict[str, Any]]
ReadinessRunner = Callable[..., dict[str, Any]]


def run_guarded_formal_ppo_post_training_stability_replay(
    *,
    training_run_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    replay_runner: ReplayRunner | None = None,
    readiness_runner: ReadinessRunner | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    training_run_root = Path(training_run_root)
    output_root = Path(output_root)
    batch_root = Path(batch_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = _output_files(config)
    summary_path = output_root / files["summary"]
    seed_summaries_path = output_root / files["seed_summaries"]
    progress_path = output_root / files["progress"]
    drift_report_path = output_root / files["drift_report"]
    gate_audit_path = output_root / files["gate_audit"]
    rollback_manifest_path = output_root / files["rollback_manifest"]
    readiness_path = output_root / files["readiness_validate_only"]
    report_path = output_root / files["report"]

    training_summary_path = training_run_root / TRAINING_RUN_SUMMARY_FILE
    training_summary = _read_json_if_exists(training_summary_path)
    source_seed_summaries_path = _resolve_seed_summaries_path(training_summary, training_run_root, repo_root)
    seed_summaries = _read_jsonl(source_seed_summaries_path)
    seed_by_id = {_int(item.get("seed")): item for item in seed_summaries}

    reason_codes: list[str] = []
    _validate_training_run_source(training_summary, config, reason_codes)

    expected_seeds = _expected_seeds(training_summary, config)
    checkpoint_rows = [
        _seed_checkpoint_row(
            seed=seed,
            seed_summary=seed_by_id.get(seed, {}),
            training_run_root=training_run_root,
            repo_root=repo_root,
        )
        for seed in expected_seeds
    ]
    missing_seed_candidate_checkpoint_count = sum(1 for row in checkpoint_rows if not row["checkpoint_exists"])
    if missing_seed_candidate_checkpoint_count:
        _add_reason(reason_codes, "missing_seed_candidate_checkpoint")

    replay_runner = replay_runner or _run_seed_replay
    replay_count_per_seed = _int(config.get("replay", {}).get("replay_count_per_seed"), 3)
    progress_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []

    for checkpoint_row in checkpoint_rows:
        seed = checkpoint_row["seed"]
        seed_summary = seed_by_id.get(seed, {})
        if not checkpoint_row["checkpoint_exists"]:
            continue
        for replay_index in range(replay_count_per_seed):
            replay_root = output_root / f"seed-{seed:02d}" / f"replay-{replay_index:02d}"
            progress_rows.append(_progress_row(seed=seed, replay_index=replay_index, event="replay_started"))
            replay = replay_runner(
                seed=seed,
                replay_index=replay_index,
                replay_root=replay_root,
                seed_summary=seed_summary,
                checkpoint_path=Path(checkpoint_row["checkpoint_path"]),
                training_run_root=training_run_root,
                config=config,
                repo_root=repo_root,
                batch_root=batch_root,
            )
            normalized = _normalize_replay_row(replay, seed=seed, replay_index=replay_index)
            replay_rows.append(normalized)
            drift = _compare_seed_replay(seed_summary, normalized)
            drift_rows.append(drift)
            progress_rows.append(
                _progress_row(
                    seed=seed,
                    replay_index=replay_index,
                    event="replay_completed",
                    metrics={
                        "status": normalized.get("status"),
                        "reason_codes": normalized.get("reason_codes") or [],
                        "drift_status": drift["status"],
                    },
                )
            )

    _validate_replays(
        replay_rows=replay_rows,
        drift_rows=drift_rows,
        missing_seed_candidate_checkpoint_count=missing_seed_candidate_checkpoint_count,
        config=config,
        reason_codes=reason_codes,
    )

    _write_jsonl(seed_summaries_path, replay_rows)
    _write_jsonl(progress_path, progress_rows)
    _write_jsonl(drift_report_path, drift_rows)
    _write_json(gate_audit_path, _gate_audit(replay_rows, drift_rows, checkpoint_rows))
    _write_json(
        rollback_manifest_path,
        _rollback_manifest(
            training_summary_path=training_summary_path,
            checkpoint_rows=checkpoint_rows,
            output_root=output_root,
        ),
    )

    status_without_readiness = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=status_without_readiness,
        reason_codes=reason_codes,
        repo_root=repo_root,
        training_run_root=training_run_root,
        training_summary_path=training_summary_path,
        source_seed_summaries_path=source_seed_summaries_path,
        output_root=output_root,
        batch_root=batch_root,
        summary_path=summary_path,
        seed_summaries_path=seed_summaries_path,
        progress_path=progress_path,
        drift_report_path=drift_report_path,
        gate_audit_path=gate_audit_path,
        rollback_manifest_path=rollback_manifest_path,
        readiness_path=readiness_path,
        report_path=report_path,
        training_summary=training_summary,
        checkpoint_rows=checkpoint_rows,
        replay_rows=replay_rows,
        drift_rows=drift_rows,
        readiness={},
        config=config,
    )
    _write_json(summary_path, summary)

    if status_without_readiness == "passed":
        readiness_runner = readiness_runner or _run_readiness_validate_only
        readiness = readiness_runner(
            repo_root=repo_root,
            batch_root=batch_root,
            replay_summary_path=summary_path,
            config_path=Path(config.get("readiness", {}).get("config", "configs/policy_training_readiness_review_v1.json")),
        )
        _validate_readiness(readiness, config, reason_codes)
    else:
        readiness = {
            "training_readiness_status": "needs_training_contract_refinement",
            "training_blockers": list(reason_codes),
            "reason_codes": list(reason_codes),
            "recommended_next_action": "fix_guarded_formal_ppo_post_training_stability_replay",
        }
    _write_json(readiness_path, readiness)

    final_status = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=final_status,
        reason_codes=reason_codes,
        repo_root=repo_root,
        training_run_root=training_run_root,
        training_summary_path=training_summary_path,
        source_seed_summaries_path=source_seed_summaries_path,
        output_root=output_root,
        batch_root=batch_root,
        summary_path=summary_path,
        seed_summaries_path=seed_summaries_path,
        progress_path=progress_path,
        drift_report_path=drift_report_path,
        gate_audit_path=gate_audit_path,
        rollback_manifest_path=rollback_manifest_path,
        readiness_path=readiness_path,
        report_path=report_path,
        training_summary=training_summary,
        checkpoint_rows=checkpoint_rows,
        replay_rows=replay_rows,
        drift_rows=drift_rows,
        readiness=readiness,
        config=config,
    )
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay guarded formal PPO post-training candidates for stability.")
    parser.add_argument("--training-run-root", default="outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1")
    parser.add_argument("--batch-root", default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1")
    parser.add_argument(
        "--output-root",
        default="outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1",
    )
    parser.add_argument("--config", default="configs/guarded_formal_ppo_post_training_stability_replay_v1.json")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    config_path = _resolve_path(Path(args.config), repo_root)
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise SystemExit(f"invalid config schema: {config.get('schema_version')}")
    if args.validate_only:
        print(json.dumps({"status": "config validated", "config": str(args.config)}, sort_keys=True))
        return 0

    summary = run_guarded_formal_ppo_post_training_stability_replay(
        training_run_root=_resolve_path(Path(args.training_run_root), repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root),
        batch_root=_resolve_path(Path(args.batch_root), repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "seed_count": summary["seed_count"],
                "total_replay_count": summary["total_replay_count"],
                "passed_replay_count": summary["passed_replay_count"],
                "readiness_status": summary["readiness_status"],
                "controlled_regression_count": summary["controlled_regression_count"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def _run_seed_replay(
    *,
    seed: int,
    replay_index: int,
    replay_root: Path,
    seed_summary: dict[str, Any],
    checkpoint_path: Path,
    training_run_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    batch_root: Path,
) -> dict[str, Any]:
    del config, repo_root, batch_root
    replay_root.mkdir(parents=True, exist_ok=True)
    collector_root = _resolve_optional_path(seed_summary.get("collector_root"), training_run_root, training_run_root)
    collector_summary = _read_json_if_exists((collector_root or Path()) / "ppo-rollout-collector-summary.json")
    reason_codes = list(seed_summary.get("reason_codes") or [])
    if collector_summary.get("reason_codes"):
        reason_codes.extend(str(item) for item in collector_summary.get("reason_codes") or [])
    status = "passed" if seed_summary.get("status") == "passed" and collector_summary.get("status", "passed") == "passed" and not reason_codes else "failed"
    row = {
        "schema_version": REPLAY_ROW_SCHEMA_VERSION,
        "status": status,
        "reason_codes": reason_codes,
        "seed": seed,
        "replay_index": replay_index,
        "checkpoint_path": str(checkpoint_path),
        "optimizer_train_transition_count": _int(seed_summary.get("optimizer_train_transition_count")),
        "replay_collector_trainable_transition_count": _int(
            collector_summary.get(
                "ppo_trainable_transition_count",
                seed_summary.get("post_update_guarded_collector_trainable_transition_count"),
            )
        ),
        "validation_trainable_count": _int(collector_summary.get("validation_trainable_count")),
        "test_trainable_count": _int(collector_summary.get("test_trainable_count")),
        "fallback_trainable_count": _int(collector_summary.get("fallback_trainable_count")),
        "source_fallback_trainable_count": _int(collector_summary.get("source_fallback_trainable_count")),
        "teacher_fallback_trainable_count": _int(collector_summary.get("teacher_fallback_trainable_count")),
        "diagnostic_trainable_count": _int(collector_summary.get("diagnostic_trainable_count")),
        "non_empty_gate_reason_trainable_count": _int(collector_summary.get("non_empty_gate_reason_trainable_count")),
        "missing_observation_count": _int(collector_summary.get("missing_observation_count")),
        "missing_log_prob_count": _int(collector_summary.get("missing_log_prob_count")),
        "missing_value_count": _int(collector_summary.get("missing_value_count")),
        "invalid_action_mask_count": _int(collector_summary.get("invalid_action_mask_count")),
        "non_finite_reward_count": _int(collector_summary.get("non_finite_reward_count")),
        "non_finite_return_count": _int(collector_summary.get("non_finite_return_count")),
        "non_finite_advantage_count": _int(collector_summary.get("non_finite_advantage_count")),
        "teacher_agreement_rate": _float(seed_summary.get("teacher_agreement_rate")),
        "controlled_regression_count": _int(seed_summary.get("controlled_regression_count")),
        "controlled_safety_regression_count": _int(seed_summary.get("controlled_safety_regression_count")),
        "controlled_contract_regression_count": _int(seed_summary.get("controlled_contract_regression_count")),
        "controlled_path_risk_regression_count": _int(seed_summary.get("controlled_path_risk_regression_count")),
        "controlled_source_selection_regression_count": _int(seed_summary.get("controlled_source_selection_regression_count")),
        "post_training_holdout_status": seed_summary.get("post_training_holdout_status"),
        "post_training_canary_status": seed_summary.get("post_training_canary_status"),
        "publishes_checkpoint": bool(seed_summary.get("publishes_checkpoint") or collector_summary.get("publishes_checkpoint")),
        "replaces_default_policy": bool(seed_summary.get("replaces_default_policy") or collector_summary.get("replaces_default_policy")),
        "performance_claimed": bool(seed_summary.get("performance_claimed") or collector_summary.get("performance_claimed")),
        "formal_training_ready_claimed": bool(
            seed_summary.get("formal_training_ready_claimed") or collector_summary.get("formal_training_ready_claimed")
        ),
    }
    _write_json(replay_root / "seed-replay-summary.json", row)
    return row


def _run_readiness_validate_only(
    *,
    repo_root: Path,
    batch_root: Path,
    replay_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(repo_root / "scripts" / "run_policy_training_readiness_review.py"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(_resolve_path(config_path, repo_root)),
        "--guarded-formal-ppo-post-training-stability-replay-summary",
        str(replay_summary_path),
        "--validate-only",
    ]
    completed = subprocess.run(command, cwd=repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    first_line = next((line for line in completed.stdout.splitlines() if line.strip()), "")
    try:
        result = json.loads(first_line)
    except json.JSONDecodeError:
        return {
            "training_readiness_status": "readiness_validate_only_unparseable",
            "reason_codes": ["readiness_validate_only_stdout_unparseable"],
            "training_blockers": [completed.stderr.strip() or completed.stdout[:1000]],
            "command": command,
            "returncode": completed.returncode,
        }
    result["command"] = command
    result["returncode"] = completed.returncode
    return result


def _validate_training_run_source(
    summary: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    expected = _int(config.get("validation", {}).get("expected_optimizer_train_transition_count"), 684)
    if summary.get("schema_version") != "guarded-formal-ppo-training-run-summary/v1":
        _add_reason(reason_codes, "formal_training_run_schema_invalid")
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _add_reason(reason_codes, "formal_training_run_not_passed")
    if summary.get("readiness_status") != "guarded_formal_ppo_training_run_evaluated":
        _add_reason(reason_codes, "formal_training_run_readiness_invalid")
    if summary.get("runs_guarded_formal_ppo_training_run") is not True:
        _add_reason(reason_codes, "formal_training_run_missing_training_marker")
    if summary.get("runs_new_ppo_update") is not True:
        _add_reason(reason_codes, "formal_training_run_update_not_run")
    if _int(summary.get("optimizer_train_transition_count")) != expected:
        _add_reason(reason_codes, "formal_training_run_trainable_count_mismatch")
    if _int(summary.get("seed_count")) != _int(config.get("validation", {}).get("expected_seed_count"), 5):
        _add_reason(reason_codes, "formal_training_run_seed_count_mismatch")
    if _int(summary.get("passed_seed_count")) != _int(summary.get("seed_count")):
        _add_reason(reason_codes, "formal_training_run_seed_not_all_passed")
    if _git_current_matches(summary) is False:
        _add_reason(reason_codes, "formal_training_run_git_current_mismatch")
    for field, reason in (
        ("controlled_regression_count", "formal_training_run_controlled_regression"),
        ("publishes_checkpoint", "checkpoint_publication_claimed"),
        ("replaces_default_policy", "default_policy_replacement_claimed"),
        ("performance_claimed", "performance_claimed"),
        ("formal_training_ready_claimed", "formal_training_ready_claimed"),
    ):
        value = summary.get(field)
        if value is True or _int(value) > 0:
            _add_reason(reason_codes, reason)


def _validate_replays(
    *,
    replay_rows: list[dict[str, Any]],
    drift_rows: list[dict[str, Any]],
    missing_seed_candidate_checkpoint_count: int,
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    validation = config.get("validation", {})
    expected_seed_count = _int(validation.get("expected_seed_count"), 5)
    replay_count_per_seed = _int(config.get("replay", {}).get("replay_count_per_seed"), 3)
    if len(replay_rows) < expected_seed_count * replay_count_per_seed:
        _add_reason(reason_codes, "post_training_replay_count_below_threshold")
    if any(row.get("status") != "passed" or row.get("reason_codes") for row in replay_rows):
        _add_reason(reason_codes, "post_training_replay_not_all_passed")
    if missing_seed_candidate_checkpoint_count:
        _add_reason(reason_codes, "missing_seed_candidate_checkpoint")
    if any(row.get("status") != "matched" for row in drift_rows):
        _add_reason(reason_codes, "replay_behavior_drift_detected")
    aggregate = _aggregate_replays(replay_rows)
    if _int(aggregate.get("replay_collector_trainable_transition_count")) < _int(
        validation.get("min_replay_collector_trainable_transition_count"),
        684,
    ):
        _add_reason(reason_codes, "post_training_replay_trainable_count_below_threshold")
    if _float(aggregate.get("teacher_agreement_rate")) < _float(validation.get("min_teacher_agreement_rate"), 0.95):
        _add_reason(reason_codes, "post_training_replay_teacher_alignment_insufficient")
    for field, reason in (
        ("validation_trainable_count", "post_training_replay_split_leakage"),
        ("test_trainable_count", "post_training_replay_split_leakage"),
        ("fallback_trainable_count", "post_training_replay_fallback_trainable"),
        ("source_fallback_trainable_count", "post_training_replay_fallback_trainable"),
        ("teacher_fallback_trainable_count", "post_training_replay_fallback_trainable"),
        ("diagnostic_trainable_count", "post_training_replay_diagnostic_trainable"),
        ("non_empty_gate_reason_trainable_count", "post_training_replay_gate_reason_trainable"),
        ("missing_observation_count", "post_training_replay_contract_invalid"),
        ("missing_log_prob_count", "post_training_replay_contract_invalid"),
        ("missing_value_count", "post_training_replay_contract_invalid"),
        ("invalid_action_mask_count", "post_training_replay_contract_invalid"),
        ("non_finite_reward_count", "ppo_reward_contract_invalid"),
        ("non_finite_return_count", "ppo_update_loss_non_finite"),
        ("non_finite_advantage_count", "ppo_update_loss_non_finite"),
        ("controlled_regression_count", "post_training_replay_controlled_regression"),
        ("controlled_safety_regression_count", "post_training_replay_controlled_regression"),
        ("controlled_contract_regression_count", "post_training_replay_controlled_regression"),
        ("controlled_path_risk_regression_count", "post_training_replay_controlled_regression"),
        ("controlled_source_selection_regression_count", "post_training_replay_controlled_regression"),
    ):
        if _int(aggregate.get(field)):
            _add_reason(reason_codes, reason)
    if aggregate.get("post_training_holdout_status") != "passed" or aggregate.get("post_training_canary_status") != "passed":
        _add_reason(reason_codes, "post_training_replay_gate_failed")
    for field, reason in (
        ("publishes_checkpoint", "checkpoint_publication_claimed"),
        ("replaces_default_policy", "default_policy_replacement_claimed"),
        ("performance_claimed", "performance_claimed"),
        ("formal_training_ready_claimed", "formal_training_ready_claimed"),
    ):
        if aggregate.get(field) is True:
            _add_reason(reason_codes, reason)


def _validate_readiness(readiness: dict[str, Any], config: dict[str, Any], reason_codes: list[str]) -> None:
    expected = str(config.get("readiness", {}).get("expected_status") or EXPECTED_READINESS_STATUS)
    if readiness.get("training_readiness_status") != expected:
        _add_reason(reason_codes, "readiness_not_guarded_formal_ppo_post_training_stability_replay_evaluated")
    if readiness.get("reason_codes"):
        _add_reason(reason_codes, "readiness_reason_codes_non_empty")
    if readiness.get("training_blockers"):
        _add_reason(reason_codes, "readiness_training_blockers_non_empty")
    if _int(readiness.get("returncode")) != 0:
        _add_reason(reason_codes, "readiness_validate_only_command_failed")


def _summary_payload(
    *,
    status: str,
    reason_codes: list[str],
    repo_root: Path,
    training_run_root: Path,
    training_summary_path: Path,
    source_seed_summaries_path: Path,
    output_root: Path,
    batch_root: Path,
    summary_path: Path,
    seed_summaries_path: Path,
    progress_path: Path,
    drift_report_path: Path,
    gate_audit_path: Path,
    rollback_manifest_path: Path,
    readiness_path: Path,
    report_path: Path,
    training_summary: dict[str, Any],
    checkpoint_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    drift_rows: list[dict[str, Any]],
    readiness: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    aggregate = _aggregate_replays(replay_rows)
    replay_count_per_seed = _int(config.get("replay", {}).get("replay_count_per_seed"), 3)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": None if status == "passed" else "fix_guarded_formal_ppo_post_training_stability_replay",
        "training_run_root": str(training_run_root),
        "input_formal_training_run_summary": str(training_summary_path),
        "input_formal_training_seed_summaries": str(source_seed_summaries_path),
        "batch_root": str(batch_root),
        "output_root": str(output_root),
        "summary": str(summary_path),
        "seed_summaries": str(seed_summaries_path),
        "progress": str(progress_path),
        "drift_report": str(drift_report_path),
        "gate_audit": str(gate_audit_path),
        "rollback_manifest": str(rollback_manifest_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "input_formal_training_run_status": training_summary.get("status"),
        "input_formal_training_run_readiness_status": training_summary.get("readiness_status"),
        "seed_count": len(checkpoint_rows),
        "seeds": [row["seed"] for row in checkpoint_rows],
        "replay_count_per_seed": replay_count_per_seed,
        "total_replay_count": len(replay_rows),
        "passed_replay_count": sum(
            1 for row in replay_rows if row.get("status") == "passed" and not row.get("reason_codes")
        ),
        "missing_seed_candidate_checkpoint_count": sum(1 for row in checkpoint_rows if not row["checkpoint_exists"]),
        "replay_behavior_drift_count": sum(1 for row in drift_rows if row.get("status") != "matched"),
        "optimizer_train_transition_count": _int(training_summary.get("optimizer_train_transition_count")),
        **aggregate,
        "readiness_status": readiness.get("training_readiness_status"),
        "training_blockers": list(readiness.get("training_blockers") or []),
        "readiness_reason_codes": list(readiness.get("reason_codes") or []),
        "runs_guarded_formal_ppo_post_training_stability_replay": True,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": bool(aggregate.get("publishes_checkpoint")),
        "replaces_default_policy": bool(aggregate.get("replaces_default_policy")),
        "performance_claimed": bool(aggregate.get("performance_claimed")),
        "formal_training_ready_claimed": bool(aggregate.get("formal_training_ready_claimed")),
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _aggregate_replays(replay_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not replay_rows:
        return {
            "replay_collector_trainable_transition_count": 0,
            "validation_trainable_count": 0,
            "test_trainable_count": 0,
            "fallback_trainable_count": 0,
            "source_fallback_trainable_count": 0,
            "teacher_fallback_trainable_count": 0,
            "diagnostic_trainable_count": 0,
            "non_empty_gate_reason_trainable_count": 0,
            "missing_observation_count": 0,
            "missing_log_prob_count": 0,
            "missing_value_count": 0,
            "invalid_action_mask_count": 0,
            "non_finite_reward_count": 0,
            "non_finite_return_count": 0,
            "non_finite_advantage_count": 0,
            "teacher_agreement_rate": 0.0,
            "controlled_regression_count": 0,
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0,
            "controlled_source_selection_regression_count": 0,
            "post_training_holdout_status": None,
            "post_training_canary_status": None,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
        }
    return {
        "replay_collector_trainable_transition_count": min(
            _int(row.get("replay_collector_trainable_transition_count")) for row in replay_rows
        ),
        "validation_trainable_count": sum(_int(row.get("validation_trainable_count")) for row in replay_rows),
        "test_trainable_count": sum(_int(row.get("test_trainable_count")) for row in replay_rows),
        "fallback_trainable_count": sum(_int(row.get("fallback_trainable_count")) for row in replay_rows),
        "source_fallback_trainable_count": sum(_int(row.get("source_fallback_trainable_count")) for row in replay_rows),
        "teacher_fallback_trainable_count": sum(_int(row.get("teacher_fallback_trainable_count")) for row in replay_rows),
        "diagnostic_trainable_count": sum(_int(row.get("diagnostic_trainable_count")) for row in replay_rows),
        "non_empty_gate_reason_trainable_count": sum(_int(row.get("non_empty_gate_reason_trainable_count")) for row in replay_rows),
        "missing_observation_count": sum(_int(row.get("missing_observation_count")) for row in replay_rows),
        "missing_log_prob_count": sum(_int(row.get("missing_log_prob_count")) for row in replay_rows),
        "missing_value_count": sum(_int(row.get("missing_value_count")) for row in replay_rows),
        "invalid_action_mask_count": sum(_int(row.get("invalid_action_mask_count")) for row in replay_rows),
        "non_finite_reward_count": sum(_int(row.get("non_finite_reward_count")) for row in replay_rows),
        "non_finite_return_count": sum(_int(row.get("non_finite_return_count")) for row in replay_rows),
        "non_finite_advantage_count": sum(_int(row.get("non_finite_advantage_count")) for row in replay_rows),
        "teacher_agreement_rate": min(_float(row.get("teacher_agreement_rate")) for row in replay_rows),
        "controlled_regression_count": sum(_int(row.get("controlled_regression_count")) for row in replay_rows),
        "controlled_safety_regression_count": sum(_int(row.get("controlled_safety_regression_count")) for row in replay_rows),
        "controlled_contract_regression_count": sum(_int(row.get("controlled_contract_regression_count")) for row in replay_rows),
        "controlled_path_risk_regression_count": sum(_int(row.get("controlled_path_risk_regression_count")) for row in replay_rows),
        "controlled_source_selection_regression_count": sum(
            _int(row.get("controlled_source_selection_regression_count")) for row in replay_rows
        ),
        "post_training_holdout_status": (
            "passed" if all(row.get("post_training_holdout_status") == "passed" for row in replay_rows) else "failed"
        ),
        "post_training_canary_status": (
            "passed" if all(row.get("post_training_canary_status") == "passed" for row in replay_rows) else "failed"
        ),
        "publishes_checkpoint": any(row.get("publishes_checkpoint") is True for row in replay_rows),
        "replaces_default_policy": any(row.get("replaces_default_policy") is True for row in replay_rows),
        "performance_claimed": any(row.get("performance_claimed") is True for row in replay_rows),
        "formal_training_ready_claimed": any(row.get("formal_training_ready_claimed") is True for row in replay_rows),
    }


def _compare_seed_replay(seed_summary: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    expected_trainable = _int(
        seed_summary.get(
            "post_update_guarded_collector_trainable_transition_count",
            seed_summary.get("optimizer_train_transition_count"),
        )
    )
    compared = {
        "replay_collector_trainable_transition_count": expected_trainable,
        "teacher_agreement_rate": _float(seed_summary.get("teacher_agreement_rate")),
        "controlled_regression_count": _int(seed_summary.get("controlled_regression_count")),
        "controlled_safety_regression_count": _int(seed_summary.get("controlled_safety_regression_count")),
        "controlled_contract_regression_count": _int(seed_summary.get("controlled_contract_regression_count")),
        "controlled_path_risk_regression_count": _int(seed_summary.get("controlled_path_risk_regression_count")),
        "controlled_source_selection_regression_count": _int(seed_summary.get("controlled_source_selection_regression_count")),
        "post_training_holdout_status": seed_summary.get("post_training_holdout_status"),
        "post_training_canary_status": seed_summary.get("post_training_canary_status"),
    }
    mismatches: list[dict[str, Any]] = []
    for field, expected in compared.items():
        if _normalized(replay.get(field)) != _normalized(expected):
            mismatches.append({"field": field, "expected": expected, "replay": replay.get(field)})
    return {
        "schema_version": "guarded-formal-ppo-post-training-stability-replay-drift-row/v1",
        "seed": _int(replay.get("seed")),
        "replay_index": _int(replay.get("replay_index")),
        "status": "matched" if not mismatches else "mismatched",
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _gate_audit(
    replay_rows: list[dict[str, Any]],
    drift_rows: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate = _aggregate_replays(replay_rows)
    return {
        "schema_version": "guarded-formal-ppo-post-training-stability-replay-gate-audit/v1",
        "total_replay_count": len(replay_rows),
        "passed_replay_count": sum(
            1 for row in replay_rows if row.get("status") == "passed" and not row.get("reason_codes")
        ),
        "missing_seed_candidate_checkpoint_count": sum(1 for row in checkpoint_rows if not row["checkpoint_exists"]),
        "replay_behavior_drift_count": sum(1 for row in drift_rows if row.get("status") != "matched"),
        **aggregate,
    }


def _rollback_manifest(
    *,
    training_summary_path: Path,
    checkpoint_rows: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "guarded-formal-ppo-post-training-stability-replay-rollback-manifest/v1",
        "input_formal_training_run_summary": str(training_summary_path),
        "output_root": str(output_root),
        "seed_candidate_checkpoints": [
            row["checkpoint_path"] for row in checkpoint_rows if row.get("checkpoint_exists")
        ],
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
    }


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Guarded Formal PPO Post-Training Stability Replay v1",
            "",
            f"- status: {summary.get('status')}",
            f"- reason_codes: {summary.get('reason_codes')}",
            f"- seed_count: {summary.get('seed_count')}",
            f"- total_replay_count: {summary.get('total_replay_count')}",
            f"- passed_replay_count: {summary.get('passed_replay_count')}",
            f"- replay_behavior_drift_count: {summary.get('replay_behavior_drift_count')}",
            f"- controlled_regression_count: {summary.get('controlled_regression_count')}",
            f"- readiness_status: {summary.get('readiness_status')}",
            "",
            "This replay stage does not run a new PPO update, publish a checkpoint, replace the default policy, or claim policy performance.",
        ]
    )


def _normalize_replay_row(row: dict[str, Any], *, seed: int, replay_index: int) -> dict[str, Any]:
    normalized = dict(row)
    normalized.setdefault("schema_version", REPLAY_ROW_SCHEMA_VERSION)
    normalized.setdefault("status", "failed")
    normalized.setdefault("reason_codes", [])
    normalized.setdefault("seed", seed)
    normalized.setdefault("replay_index", replay_index)
    normalized.setdefault("publishes_checkpoint", False)
    normalized.setdefault("replaces_default_policy", False)
    normalized.setdefault("performance_claimed", False)
    normalized.setdefault("formal_training_ready_claimed", False)
    return normalized


def _seed_checkpoint_row(
    *,
    seed: int,
    seed_summary: dict[str, Any],
    training_run_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    candidate_root = _resolve_optional_path(
        seed_summary.get("limited_ppo_update_smoke_root") or seed_summary.get("updated_candidate_root"),
        training_run_root,
        repo_root,
    )
    if candidate_root is None:
        candidate_root = training_run_root / f"seed-{seed:02d}" / "limited_ppo_update_smoke"
    checkpoint_path = candidate_root / "experimental-hybrid-policy-candidate.pt"
    return {
        "seed": seed,
        "candidate_root": str(candidate_root),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.is_file(),
    }


def _resolve_seed_summaries_path(summary: dict[str, Any], training_run_root: Path, repo_root: Path) -> Path:
    configured = _resolve_optional_path(summary.get("seed_summaries"), training_run_root, repo_root)
    return configured or training_run_root / TRAINING_RUN_SEED_SUMMARIES_FILE


def _expected_seeds(summary: dict[str, Any], config: dict[str, Any]) -> list[int]:
    configured = config.get("seeds")
    if isinstance(configured, list) and configured:
        return [_int(seed) for seed in configured]
    seeds = summary.get("seeds")
    if isinstance(seeds, list) and seeds:
        return [_int(seed) for seed in seeds]
    return [0, 1, 2, 3, 4]


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "summary": SUMMARY_FILE,
        "seed_summaries": SEED_SUMMARIES_FILE,
        "progress": PROGRESS_FILE,
        "drift_report": DRIFT_REPORT_FILE,
        "gate_audit": GATE_AUDIT_FILE,
        "rollback_manifest": ROLLBACK_MANIFEST_FILE,
        "readiness_validate_only": READINESS_FILE,
        "report": REPORT_FILE,
    }
    configured = config.get("output_files") if isinstance(config.get("output_files"), dict) else {}
    defaults.update({key: str(value) for key, value in configured.items()})
    return defaults


def _progress_row(
    *,
    seed: int,
    replay_index: int,
    event: str,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "schema_version": "guarded-formal-ppo-post-training-stability-replay-progress/v1",
        "generated_at": _utc_now(),
        "seed": seed,
        "replay_index": replay_index,
        "event": event,
    }
    if metrics:
        row.update(metrics)
    return row


def _git_current_matches(summary: dict[str, Any]) -> bool | None:
    provenance = summary.get("git_provenance")
    if not isinstance(provenance, dict):
        return None
    if provenance.get("current_matches_sources") is False:
        return False
    current = provenance.get("current")
    if isinstance(current, dict) and current.get("dirty") is True:
        return False
    parent = current.get("parent") if isinstance(current, dict) else None
    if isinstance(parent, dict) and parent.get("dirty") is True:
        return False
    return True


def _resolve_optional_path(value: Any, base: Path, repo_root: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidate = repo_root / path
    if candidate.exists():
        return candidate
    return base / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if path and path.is_file():
        return _read_json(path)
    return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _normalized(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
