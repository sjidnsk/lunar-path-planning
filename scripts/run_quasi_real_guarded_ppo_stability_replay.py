from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from git_provenance import git_snapshot
except ModuleNotFoundError:  # pragma: no cover - import path used by unit tests
    from scripts.git_provenance import git_snapshot


CONFIG_SCHEMA_VERSION = "quasi-real-guarded-ppo-stability-replay-config/v1"
SUMMARY_SCHEMA_VERSION = "quasi-real-guarded-ppo-stability-replay-summary/v1"
EXPECTED_READINESS_STATUS = "quasi_real_guarded_ppo_stability_replay_evaluated"

SUMMARY_FILE = "quasi-real-guarded-ppo-stability-replay-summary.json"
COMPARISON_FILE = "stability-replay-comparison.jsonl"
ACCEPTANCE_CONTRACT_FILE = "acceptance-contract-refinement.json"
PROGRESS_EVENTS_FILE = "stability-replay-progress-events.jsonl"
READINESS_FILE = "quasi-real-guarded-ppo-stability-readiness-validate-only.json"
REPORT_FILE = "stability-replay-report.md"

PilotReplayRunner = Callable[..., dict[str, Any]]
ReadinessRunner = Callable[..., dict[str, Any]]


def run_quasi_real_guarded_ppo_stability_replay(
    *,
    freeze_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    pilot_replay_runner: PilotReplayRunner | None = None,
    readiness_runner: ReadinessRunner | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    freeze_root = Path(freeze_root)
    output_root = Path(output_root)
    batch_root = Path(batch_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = _output_files(config)
    summary_path = output_root / files["summary"]
    comparison_path = output_root / files["comparison"]
    contract_path = output_root / files["acceptance_contract"]
    progress_path = output_root / files["progress_events"]
    readiness_path = output_root / files["readiness_validate_only"]
    report_path = output_root / files["report"]

    freeze_summary_path = freeze_root / "quasi-real-guarded-ppo-evidence-freeze-summary.json"
    freeze_manifest_path = freeze_root / "quasi-real-guarded-ppo-evidence-manifest.json"
    freeze_summary = _read_json_if_exists(freeze_summary_path)
    freeze_manifest = _read_json_if_exists(freeze_manifest_path)
    baseline_pilot = _read_baseline_pilot_summary(freeze_summary, repo_root)
    baseline_root = _baseline_pilot_root(freeze_summary, repo_root)
    baseline_steps = _load_step_signatures(
        baseline_root / "quasi-real-guarded-ppo-rollout-steps.jsonl"
    )

    update_root = _path_from_config_or_summary(
        config,
        "update_smoke_root",
        freeze_summary.get("update_root"),
        repo_root,
    )
    candidate_root = _path_from_config_or_summary(
        config,
        "candidate_root",
        baseline_pilot.get("candidate_root") or update_root,
        repo_root,
    )
    quasi_real_root = _path_from_config_or_summary(
        config,
        "quasi_real_root",
        baseline_pilot.get("quasi_real_root")
        or "outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1",
        repo_root,
    )
    pilot_config = _path_from_config_or_summary(
        config,
        "pilot_config",
        config.get("default_paths", {}).get(
            "pilot_config", "configs/quasi_real_guarded_ppo_rollout_pilot_v1.json"
        ),
        repo_root,
    )

    reason_codes: list[str] = []
    _validate_frozen_evidence(freeze_summary, freeze_manifest, reason_codes)
    replay_count = _int_value(config.get("replay", {}).get("replay_count"), 3)
    pilot_replay_runner = pilot_replay_runner or _run_pilot_replay

    progress_events: list[dict[str, Any]] = []
    replay_summaries: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    _add_progress(progress_events, "stability_replay_started", replay_index=None)
    for replay_index in range(replay_count):
        replay_root = output_root / f"replay-{replay_index:02d}"
        started_at = time.perf_counter()
        _add_progress(
            progress_events,
            "replay_started",
            replay_index=replay_index,
            episode_count=0,
            step_count=0,
        )
        replay_summary = pilot_replay_runner(
            output_root=replay_root,
            replay_index=replay_index,
            repo_root=repo_root,
            update_smoke_root=update_root,
            candidate_root=candidate_root,
            quasi_real_root=quasi_real_root,
            config_path=pilot_config,
        )
        replay_summaries.append(replay_summary)
        _validate_replay_summary(replay_summary, config, reason_codes)
        replay_steps = _load_step_signatures(
            replay_root / "quasi-real-guarded-ppo-rollout-steps.jsonl"
        )
        comparison_row = _compare_replay(
            replay_index=replay_index,
            baseline=baseline_pilot or _baseline_from_freeze(freeze_summary),
            replay=replay_summary,
            baseline_steps=baseline_steps,
            replay_steps=replay_steps,
        )
        comparison_rows.append(comparison_row)
        if comparison_row["status"] != "matched":
            _add_reason(reason_codes, "baseline_replay_behavior_drift_detected")
        _add_progress(
            progress_events,
            "replay_finished",
            replay_index=replay_index,
            status=replay_summary.get("status"),
            reason_codes=replay_summary.get("reason_codes") or [],
            episode_count=_int_value(replay_summary.get("episode_count")),
            step_count=_int_value(replay_summary.get("step_count")),
            duration_seconds=round(time.perf_counter() - started_at, 6),
        )

    contract = _acceptance_contract()
    _write_json(contract_path, contract)
    _write_jsonl(comparison_path, comparison_rows)
    _write_jsonl(progress_path, progress_events)

    passed_replay_count = sum(1 for item in replay_summaries if item.get("status") == "passed" and not item.get("reason_codes"))
    aggregate = _aggregate_replays(replay_summaries)
    aggregate["baseline_replay_behavior_drift_count"] = sum(
        1 for row in comparison_rows if row["status"] != "matched"
    )
    if _int_value(aggregate.get("baseline_replay_behavior_drift_count")):
        _add_reason(reason_codes, "baseline_replay_behavior_drift_detected")

    status_without_readiness = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=status_without_readiness,
        reason_codes=reason_codes,
        repo_root=repo_root,
        freeze_root=freeze_root,
        output_root=output_root,
        batch_root=batch_root,
        update_root=update_root,
        candidate_root=candidate_root,
        quasi_real_root=quasi_real_root,
        summary_path=summary_path,
        comparison_path=comparison_path,
        contract_path=contract_path,
        progress_path=progress_path,
        readiness_path=readiness_path,
        report_path=report_path,
        freeze_summary_path=freeze_summary_path,
        freeze_manifest_path=freeze_manifest_path,
        freeze_summary=freeze_summary,
        replay_count=replay_count,
        passed_replay_count=passed_replay_count,
        aggregate=aggregate,
        progress_event_count=len(progress_events),
    )
    _write_json(summary_path, summary)

    readiness_runner = readiness_runner or _run_readiness_validate_only
    readiness = readiness_runner(
        repo_root=repo_root,
        batch_root=batch_root,
        stability_summary_path=summary_path,
        config_path=Path(config.get("readiness", {}).get("config", "configs/policy_training_readiness_review_v1.json")),
    )
    _write_json(readiness_path, readiness)
    _validate_readiness(readiness, config, reason_codes)

    final_status = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=final_status,
        reason_codes=reason_codes,
        repo_root=repo_root,
        freeze_root=freeze_root,
        output_root=output_root,
        batch_root=batch_root,
        update_root=update_root,
        candidate_root=candidate_root,
        quasi_real_root=quasi_real_root,
        summary_path=summary_path,
        comparison_path=comparison_path,
        contract_path=contract_path,
        progress_path=progress_path,
        readiness_path=readiness_path,
        report_path=report_path,
        freeze_summary_path=freeze_summary_path,
        freeze_manifest_path=freeze_manifest_path,
        freeze_summary=freeze_summary,
        replay_count=replay_count,
        passed_replay_count=passed_replay_count,
        aggregate=aggregate,
        progress_event_count=len(progress_events),
        readiness=readiness,
    )
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary, comparison_rows, contract), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay quasi-real guarded PPO rollout pilot evidence for stability."
    )
    parser.add_argument(
        "--freeze-root",
        default="outputs/path_feedback_batch_quasi_real_guarded_ppo_evidence_freeze_v1",
    )
    parser.add_argument(
        "--batch-root",
        default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/path_feedback_batch_quasi_real_guarded_ppo_stability_replay_v1",
    )
    parser.add_argument(
        "--config",
        default="configs/quasi_real_guarded_ppo_stability_replay_v1.json",
    )
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

    summary = run_quasi_real_guarded_ppo_stability_replay(
        freeze_root=_resolve_path(Path(args.freeze_root), repo_root),
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
                "replay_count": summary["replay_count"],
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


def _run_pilot_replay(
    *,
    output_root: Path,
    replay_index: int,
    repo_root: Path,
    update_smoke_root: Path,
    candidate_root: Path,
    quasi_real_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        "bash",
        str(repo_root / "scripts" / "run_quasi_real_guarded_ppo_rollout_pilot.sh"),
        "--update-smoke-root",
        str(update_smoke_root),
        "--candidate-root",
        str(candidate_root),
        "--quasi-real-root",
        str(quasi_real_root),
        "--output-root",
        str(output_root),
        "--config",
        str(config_path),
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    summary_path = output_root / "quasi-real-guarded-ppo-rollout-pilot-summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
    else:
        summary = {
            "schema_version": "quasi-real-guarded-ppo-rollout-pilot-summary/v1",
            "status": "failed",
            "reason_codes": ["pilot_replay_summary_missing"],
        }
    summary["replay_index"] = replay_index
    summary["command"] = command
    summary["returncode"] = completed.returncode
    if completed.returncode != 0 and "pilot_replay_command_failed" not in summary.get("reason_codes", []):
        summary["reason_codes"] = list(summary.get("reason_codes") or []) + ["pilot_replay_command_failed"]
        summary["status"] = "failed"
        summary["stderr"] = completed.stderr[-2000:]
    return summary


def _run_readiness_validate_only(
    *,
    repo_root: Path,
    batch_root: Path,
    stability_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        "bash",
        str(repo_root / "scripts" / "run_policy_training_readiness_review.sh"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(_resolve_path(config_path, repo_root)),
        "--quasi-real-guarded-ppo-stability-replay-summary",
        str(stability_summary_path),
        "--validate-only",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return _readiness_result_from_process(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        command=command,
    )


def _readiness_result_from_process(
    *,
    returncode: int,
    stdout: str,
    stderr: str,
    command: list[str],
) -> dict[str, Any]:
    first_line = next((line for line in stdout.splitlines() if line.strip()), "")
    try:
        result = json.loads(first_line)
    except json.JSONDecodeError:
        return {
            "training_readiness_status": "readiness_validate_only_unparseable",
            "reason_codes": ["readiness_validate_only_stdout_unparseable"],
            "training_blockers": [stderr.strip() or stdout[:1000]],
            "command": command,
            "returncode": returncode,
        }
    result["command"] = command
    result["returncode"] = returncode
    return result


def _summary_payload(
    *,
    status: str,
    reason_codes: list[str],
    repo_root: Path,
    freeze_root: Path,
    output_root: Path,
    batch_root: Path,
    update_root: Path,
    candidate_root: Path,
    quasi_real_root: Path,
    summary_path: Path,
    comparison_path: Path,
    contract_path: Path,
    progress_path: Path,
    readiness_path: Path,
    report_path: Path,
    freeze_summary_path: Path,
    freeze_manifest_path: Path,
    freeze_summary: dict[str, Any],
    replay_count: int,
    passed_replay_count: int,
    aggregate: dict[str, Any],
    progress_event_count: int,
    readiness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    readiness = readiness or {}
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": status,
        "reason_codes": list(reason_codes),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_provenance": {"current": git_snapshot(repo_root)},
        "freeze_root": str(freeze_root),
        "freeze_summary": str(freeze_summary_path),
        "freeze_manifest": str(freeze_manifest_path),
        "batch_root": str(batch_root),
        "update_root": str(update_root),
        "candidate_root": str(candidate_root),
        "quasi_real_root": str(quasi_real_root),
        "output_root": str(output_root),
        "summary": str(summary_path),
        "comparison": str(comparison_path),
        "acceptance_contract": str(contract_path),
        "progress_events": str(progress_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "input_freeze_status": freeze_summary.get("status"),
        "input_freeze_reason_codes": list(freeze_summary.get("reason_codes") or []),
        "input_freeze_readiness_status": freeze_summary.get("readiness_status"),
        "replay_count": replay_count,
        "passed_replay_count": passed_replay_count,
        "progress_event_count": progress_event_count,
        **aggregate,
        "acceptance_contract_refined": True,
        "readiness_status": readiness.get("training_readiness_status"),
        "training_blockers": list(readiness.get("training_blockers") or []),
        "readiness_reason_codes": list(readiness.get("reason_codes") or []),
        "runs_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
    }


def _validate_frozen_evidence(
    freeze_summary: dict[str, Any],
    freeze_manifest: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if freeze_summary.get("status") != "passed" or freeze_summary.get("reason_codes"):
        _add_reason(reason_codes, "input_freeze_summary_not_passed")
    if freeze_summary.get("readiness_status") != "quasi_real_guarded_ppo_rollout_pilot_evaluated":
        _add_reason(reason_codes, "input_freeze_readiness_not_quasi_real_guarded_pilot")
    if _int_value(freeze_summary.get("required_artifact_missing_count")):
        _add_reason(reason_codes, "input_freeze_required_artifact_missing")
    if not freeze_manifest:
        _add_reason(reason_codes, "input_freeze_manifest_missing")
    elif _int_value(freeze_manifest.get("required_artifact_missing_count")):
        _add_reason(reason_codes, "input_freeze_manifest_required_artifact_missing")


def _validate_replay_summary(
    summary: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    validation = config.get("validation", {}) if isinstance(config.get("validation"), dict) else {}
    if summary.get("status") != "passed" or summary.get("reason_codes"):
        _add_reason(reason_codes, "replay_summary_not_passed")
    if _int_value(summary.get("returncode")):
        _add_reason(reason_codes, "replay_command_failed")
    if _int_value(summary.get("episode_count")) != _int_value(validation.get("expected_episode_count"), 36):
        _add_reason(reason_codes, "replay_episode_count_unexpected")
    if _int_value(summary.get("step_count")) != _int_value(validation.get("expected_step_count"), 108):
        _add_reason(reason_codes, "replay_step_count_unexpected")
    trainable = _int_value(summary.get("ppo_trainable_transition_count", summary.get("trainable_transition_count")))
    if trainable < _int_value(validation.get("min_ppo_trainable_transition_count"), 24):
        _add_reason(reason_codes, "replay_trainable_transition_count_below_threshold")
    expected_trainable = validation.get("expected_ppo_trainable_transition_count")
    if expected_trainable is not None and trainable != _int_value(expected_trainable):
        _add_reason(reason_codes, "replay_trainable_transition_count_unexpected")
    expected_diagnostic = validation.get("expected_diagnostic_transition_count")
    if expected_diagnostic is not None and _int_value(summary.get("diagnostic_transition_count")) != _int_value(expected_diagnostic):
        _add_reason(reason_codes, "replay_diagnostic_transition_count_unexpected")
    for field, reason in (
        ("validation_trainable_count", "replay_split_leakage_detected"),
        ("test_trainable_count", "replay_split_leakage_detected"),
        ("source_fallback_trainable_count", "replay_fallback_trainable_detected"),
        ("teacher_fallback_trainable_count", "replay_fallback_trainable_detected"),
        ("missing_observation_count", "replay_materialization_contract_invalid"),
        ("missing_log_prob_count", "replay_materialization_contract_invalid"),
        ("missing_value_count", "replay_materialization_contract_invalid"),
        ("non_finite_reward_count", "replay_non_finite_value_detected"),
        ("non_finite_return_count", "replay_non_finite_value_detected"),
        ("non_finite_advantage_count", "replay_non_finite_value_detected"),
        ("controlled_regression_count", "replay_controlled_regression_detected"),
        ("controlled_safety_regression_count", "replay_controlled_regression_detected"),
        ("controlled_contract_regression_count", "replay_controlled_regression_detected"),
        ("controlled_path_risk_regression_count", "replay_controlled_regression_detected"),
        ("controlled_source_selection_regression_count", "replay_controlled_regression_detected"),
    ):
        if _int_value(summary.get(field)):
            _add_reason(reason_codes, reason)
    if _float_value(summary.get("teacher_agreement_rate")) < float(validation.get("min_teacher_agreement_rate", 0.9)):
        _add_reason(reason_codes, "replay_teacher_agreement_below_threshold")
    if summary.get("quasi_real_collector_replay_status") != "passed":
        _add_reason(reason_codes, "replay_collector_replay_not_passed")
    expected_verdict = validation.get("expected_long_horizon_verdict", "long_horizon_teacher_skill_contract_aligned")
    if summary.get("post_pilot_long_horizon_verdict", summary.get("long_horizon_verdict")) != expected_verdict:
        _add_reason(reason_codes, "replay_long_horizon_verdict_not_aligned")
    for field, reason in (
        ("publishes_checkpoint", "checkpoint_publication_claimed"),
        ("replaces_default_policy", "default_policy_replacement_claimed"),
        ("performance_claimed", "performance_claimed"),
        ("formal_training_ready_claimed", "formal_training_ready_claimed"),
    ):
        if summary.get(field) is True:
            _add_reason(reason_codes, reason)


def _validate_readiness(
    readiness: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    expected = str(config.get("readiness", {}).get("expected_status") or EXPECTED_READINESS_STATUS)
    if readiness.get("training_readiness_status") != expected:
        _add_reason(reason_codes, "readiness_not_quasi_real_guarded_ppo_stability_replay_evaluated")
    if readiness.get("reason_codes"):
        _add_reason(reason_codes, "readiness_reason_codes_non_empty")
    if readiness.get("training_blockers"):
        _add_reason(reason_codes, "readiness_training_blockers_non_empty")
    if _int_value(readiness.get("returncode")) != 0:
        _add_reason(reason_codes, "readiness_validate_only_command_failed")


def _compare_replay(
    *,
    replay_index: int,
    baseline: dict[str, Any],
    replay: dict[str, Any],
    baseline_steps: list[dict[str, Any]],
    replay_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    compared_fields = [
        "episode_count",
        "step_count",
        "ppo_trainable_transition_count",
        "diagnostic_transition_count",
        "validation_trainable_count",
        "test_trainable_count",
        "source_fallback_trainable_count",
        "missing_observation_count",
        "missing_log_prob_count",
        "missing_value_count",
        "non_finite_reward_count",
        "non_finite_return_count",
        "non_finite_advantage_count",
        "controlled_regression_count",
        "controlled_safety_regression_count",
        "controlled_contract_regression_count",
        "controlled_path_risk_regression_count",
        "controlled_source_selection_regression_count",
        "teacher_agreement_rate",
    ]
    mismatches = []
    for field in compared_fields:
        if _normalized_value(baseline.get(field)) != _normalized_value(replay.get(field)):
            mismatches.append(
                {
                    "field": field,
                    "baseline": baseline.get(field),
                    "replay": replay.get(field),
                }
            )
    step_signature_mismatch_count = 0
    if baseline_steps and replay_steps:
        if baseline_steps != replay_steps:
            step_signature_mismatch_count = _step_signature_mismatch_count(baseline_steps, replay_steps)
            mismatches.append(
                {
                    "field": "step_signatures",
                    "baseline_count": len(baseline_steps),
                    "replay_count": len(replay_steps),
                    "mismatch_count": step_signature_mismatch_count,
                }
            )
    return {
        "schema_version": "quasi-real-guarded-ppo-stability-replay-comparison-row/v1",
        "replay_index": replay_index,
        "status": "matched" if not mismatches else "mismatched",
        "mismatch_count": len(mismatches),
        "step_signature_mismatch_count": step_signature_mismatch_count,
        "mismatches": mismatches,
        "allowed_diff_fields": [
            "path",
            "output_root",
            "summary",
            "episodes",
            "steps",
            "generated_at",
            "git_provenance",
            "command",
            "returncode",
        ],
    }


def _aggregate_replays(replay_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    first = replay_summaries[0] if replay_summaries else {}
    return {
        "episode_count": _int_value(first.get("episode_count")),
        "step_count": _int_value(first.get("step_count")),
        "ppo_trainable_transition_count": _int_value(
            first.get("ppo_trainable_transition_count", first.get("trainable_transition_count"))
        ),
        "diagnostic_transition_count": _int_value(first.get("diagnostic_transition_count")),
        "validation_trainable_count": sum(_int_value(item.get("validation_trainable_count")) for item in replay_summaries),
        "test_trainable_count": sum(_int_value(item.get("test_trainable_count")) for item in replay_summaries),
        "source_fallback_trainable_count": sum(_int_value(item.get("source_fallback_trainable_count")) for item in replay_summaries),
        "missing_observation_count": sum(_int_value(item.get("missing_observation_count")) for item in replay_summaries),
        "missing_log_prob_count": sum(_int_value(item.get("missing_log_prob_count")) for item in replay_summaries),
        "missing_value_count": sum(_int_value(item.get("missing_value_count")) for item in replay_summaries),
        "non_finite_reward_count": sum(_int_value(item.get("non_finite_reward_count")) for item in replay_summaries),
        "non_finite_return_count": sum(_int_value(item.get("non_finite_return_count")) for item in replay_summaries),
        "non_finite_advantage_count": sum(_int_value(item.get("non_finite_advantage_count")) for item in replay_summaries),
        "controlled_regression_count": sum(_int_value(item.get("controlled_regression_count")) for item in replay_summaries),
        "controlled_safety_regression_count": sum(_int_value(item.get("controlled_safety_regression_count")) for item in replay_summaries),
        "controlled_contract_regression_count": sum(_int_value(item.get("controlled_contract_regression_count")) for item in replay_summaries),
        "controlled_path_risk_regression_count": sum(_int_value(item.get("controlled_path_risk_regression_count")) for item in replay_summaries),
        "controlled_source_selection_regression_count": sum(_int_value(item.get("controlled_source_selection_regression_count")) for item in replay_summaries),
        "teacher_agreement_rate": min((_float_value(item.get("teacher_agreement_rate")) for item in replay_summaries), default=0.0),
        "baseline_replay_behavior_drift_count": 0,
        "quasi_real_collector_replay_status": first.get("quasi_real_collector_replay_status"),
        "long_horizon_verdict": first.get("post_pilot_long_horizon_verdict", first.get("long_horizon_verdict")),
    }


def _acceptance_contract() -> dict[str, Any]:
    return {
        "schema_version": "quasi-real-guarded-ppo-acceptance-contract-refinement/v1",
        "hard_gates": [
            "freeze_summary_passed",
            "all_replays_passed",
            "episode_count_36",
            "step_count_108",
            "trainable_transition_count_at_least_24_target_36",
            "validation_test_diagnostic_only",
            "source_fallback_trainable_count_zero",
            "missing_observation_log_prob_value_zero",
            "non_finite_reward_return_advantage_zero",
            "controlled_regression_count_zero",
            "teacher_agreement_rate_at_least_0_9",
            "collector_replay_passed",
            "long_horizon_teacher_skill_contract_aligned",
            "readiness_validate_only_passed",
        ],
        "diagnostic_only": [
            "validation_test_diagnostic_only",
            "source_fallback",
            "teacher_fallback",
            "raw_policy_probe_rejection",
            "gate_reason_non_empty",
            "path_planner_or_iris_gcs_diagnostics",
        ],
        "allowed_diff": [
            "output_root",
            "summary_path",
            "artifact_path",
            "generated_at",
            "git_provenance_dirty_fingerprint",
            "command",
            "returncode_zero",
        ],
        "non_goals": [
            "no_new_ppo_update",
            "no_checkpoint_publication",
            "no_default_policy_replacement",
            "no_policy_performance_claim",
            "no_formal_training_ready_claim",
        ],
    }


def _read_baseline_pilot_summary(freeze_summary: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    pilot_root = _baseline_pilot_root(freeze_summary, repo_root)
    pilot_summary_path = pilot_root / "quasi-real-guarded-ppo-rollout-pilot-summary.json"
    if pilot_summary_path.is_file():
        return _read_json(pilot_summary_path)
    return _baseline_from_freeze(freeze_summary)


def _baseline_from_freeze(freeze_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": freeze_summary.get("pilot_status"),
        "reason_codes": [],
        "episode_count": freeze_summary.get("pilot_episode_count"),
        "step_count": freeze_summary.get("pilot_step_count"),
        "ppo_trainable_transition_count": freeze_summary.get("pilot_ppo_trainable_transition_count"),
        "diagnostic_transition_count": freeze_summary.get("pilot_diagnostic_transition_count"),
        "controlled_regression_count": freeze_summary.get("pilot_controlled_regression_count"),
        "teacher_agreement_rate": freeze_summary.get("pilot_teacher_agreement_rate"),
        "quasi_real_collector_replay_status": freeze_summary.get("collector_replay_status"),
        "quasi_real_collector_replay_trainable_transition_count": freeze_summary.get("collector_replay_trainable_transition_count"),
        "post_pilot_long_horizon_verdict": freeze_summary.get("long_horizon_verdict"),
        "validation_trainable_count": 0,
        "test_trainable_count": 0,
        "source_fallback_trainable_count": 0,
        "missing_observation_count": 0,
        "missing_log_prob_count": 0,
        "missing_value_count": 0,
        "non_finite_reward_count": 0,
        "non_finite_return_count": 0,
        "non_finite_advantage_count": 0,
        "controlled_safety_regression_count": 0,
        "controlled_contract_regression_count": 0,
        "controlled_path_risk_regression_count": 0,
        "controlled_source_selection_regression_count": 0,
    }


def _baseline_pilot_root(freeze_summary: dict[str, Any], repo_root: Path) -> Path:
    value = freeze_summary.get("pilot_root") or "outputs/path_feedback_batch_quasi_real_guarded_ppo_rollout_pilot_v1"
    return _resolve_path(Path(str(value)), repo_root)


def _load_step_signatures(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    signatures = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        signatures.append(
            {
                "episode_id": row.get("episode_id"),
                "step_index": row.get("step_index"),
                "context_id": row.get("context_id"),
                "scenario_id": row.get("scenario_id"),
                "split": row.get("split"),
                "controlled_choice_source": row.get("controlled_choice_source"),
                "ppo_trainable": row.get("ppo_trainable"),
                "controlled_action_index": row.get("controlled_action_index"),
                "raw_policy_action_index": row.get("raw_policy_action_index"),
                "teacher_action_index": row.get("teacher_action_index"),
                "gate_reason_codes": row.get("gate_reason_codes") or [],
                "controlled_regression_reason_codes": row.get("controlled_regression_reason_codes") or [],
                "reward": _rounded_float(row.get("reward")),
                "discounted_return": _rounded_float(row.get("discounted_return")),
                "advantage": _rounded_float(row.get("advantage")),
            }
        )
    return signatures


def _step_signature_mismatch_count(
    baseline_steps: list[dict[str, Any]],
    replay_steps: list[dict[str, Any]],
) -> int:
    mismatches = abs(len(baseline_steps) - len(replay_steps))
    for left, right in zip(baseline_steps, replay_steps):
        if left != right:
            mismatches += 1
    return mismatches


def _add_progress(
    events: list[dict[str, Any]],
    stage: str,
    *,
    replay_index: int | None,
    status: str | None = None,
    reason_codes: list[str] | None = None,
    episode_count: int | None = None,
    step_count: int | None = None,
    duration_seconds: float | None = None,
) -> None:
    events.append(
        {
            "schema_version": "quasi-real-guarded-ppo-stability-replay-progress-event/v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "replay_index": replay_index,
            "status": status,
            "reason_codes": reason_codes or [],
            "episode_count": episode_count,
            "step_count": step_count,
            "duration_seconds": duration_seconds,
        }
    )


def _render_report(
    summary: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    contract: dict[str, Any],
) -> str:
    comparisons = "\n".join(
        f"- replay-{row['replay_index']:02d}: status={row['status']} mismatches={row['mismatch_count']}"
        for row in comparison_rows
    )
    hard_gates = "\n".join(f"- `{gate}`" for gate in contract["hard_gates"])
    return (
        "# Quasi-Real Guarded PPO Stability Replay\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Readiness: `{summary.get('readiness_status')}`\n"
        f"- Replays passed: `{summary.get('passed_replay_count')}` / `{summary.get('replay_count')}`\n"
        f"- Episodes / steps: `{summary.get('episode_count')}` / `{summary.get('step_count')}`\n"
        f"- Trainable / diagnostic: `{summary.get('ppo_trainable_transition_count')}` / `{summary.get('diagnostic_transition_count')}`\n"
        f"- Controlled regression count: `{summary.get('controlled_regression_count')}`\n"
        f"- Teacher agreement rate: `{summary.get('teacher_agreement_rate')}`\n\n"
        "## Replay Comparison\n\n"
        f"{comparisons}\n\n"
        "## Acceptance Contract\n\n"
        f"{hard_gates}\n\n"
        "## Non-Goals\n\n"
        "No new PPO update, batch expansion, checkpoint publication, default policy replacement, "
        "gate relaxation, performance claim, or formal training-ready claim is made by this stage.\n"
    )


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "summary": SUMMARY_FILE,
        "comparison": COMPARISON_FILE,
        "acceptance_contract": ACCEPTANCE_CONTRACT_FILE,
        "progress_events": PROGRESS_EVENTS_FILE,
        "readiness_validate_only": READINESS_FILE,
        "report": REPORT_FILE,
    }
    configured = config.get("output_files") if isinstance(config.get("output_files"), dict) else {}
    return {key: str(configured.get(key) or default) for key, default in defaults.items()}


def _path_from_config_or_summary(
    config: dict[str, Any],
    key: str,
    fallback: Any,
    repo_root: Path,
) -> Path:
    configured = config.get("default_paths", {}).get(key)
    value = configured or fallback
    return _resolve_path(Path(str(value)), repo_root)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _rounded_float(value: Any) -> float | None:
    if value is None:
        return None
    return round(_float_value(value), 6)


def _normalized_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, int):
        return value
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return value
    return round(parsed, 6) if math.isfinite(parsed) else value


if __name__ == "__main__":
    raise SystemExit(main())
