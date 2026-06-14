from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from git_provenance import git_snapshot
except ModuleNotFoundError:  # pragma: no cover - import path used by unit tests
    from scripts.git_provenance import git_snapshot


CONFIG_SCHEMA_VERSION = "quasi-real-guarded-ppo-evidence-freeze-config/v1"
SUMMARY_SCHEMA_VERSION = "quasi-real-guarded-ppo-evidence-freeze-summary/v1"
MANIFEST_SCHEMA_VERSION = "quasi-real-guarded-ppo-evidence-manifest/v1"
EXPECTED_READINESS_STATUS = "quasi_real_guarded_ppo_rollout_pilot_evaluated"

SUMMARY_FILE = "quasi-real-guarded-ppo-evidence-freeze-summary.json"
MANIFEST_FILE = "quasi-real-guarded-ppo-evidence-manifest.json"
READINESS_FILE = "quasi-real-guarded-ppo-readiness-validate-only.json"
REPORT_FILE = "quasi-real-guarded-ppo-evidence-freeze-report.md"

ReadinessRunner = Callable[..., dict[str, Any]]


def run_quasi_real_guarded_ppo_evidence_freeze(
    *,
    pilot_root: Path,
    batch_root: Path,
    update_root: Path,
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    readiness_runner: ReadinessRunner | None = None,
) -> dict[str, Any]:
    pilot_root = Path(pilot_root)
    batch_root = Path(batch_root)
    update_root = Path(update_root)
    output_root = Path(output_root)
    repo_root = Path(repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    output_files = _output_files(config)
    summary_path = output_root / output_files["summary"]
    manifest_path = output_root / output_files["manifest"]
    readiness_path = output_root / output_files["readiness_validate_only"]
    report_path = output_root / output_files["report"]

    pilot_summary_path = pilot_root / "quasi-real-guarded-ppo-rollout-pilot-summary.json"
    pilot_summary = _read_json_if_exists(pilot_summary_path)
    episodes_path = _artifact_path(
        pilot_summary.get("episodes"),
        pilot_root / "quasi-real-guarded-ppo-rollout-episodes.jsonl",
        repo_root,
    )
    steps_path = _artifact_path(
        pilot_summary.get("steps"),
        pilot_root / "quasi-real-guarded-ppo-rollout-steps.jsonl",
        repo_root,
    )
    rejection_report_path = _artifact_path(
        pilot_summary.get("rejection_report"),
        pilot_root / "quasi-real-guarded-ppo-rollout-rejection-report.json",
        repo_root,
    )
    reward_audit_path = _artifact_path(
        pilot_summary.get("reward_audit"),
        pilot_root / "quasi-real-guarded-ppo-rollout-reward-audit.json",
        repo_root,
    )
    collector_replay_path = pilot_root / "quasi_real_collector_replay" / "ppo-rollout-collector-summary.json"
    long_horizon_path = _artifact_path(
        pilot_summary.get("long_horizon_summary"),
        update_root / "post_update_long_horizon" / "long-horizon-teacher-skill-contract-summary.json",
        repo_root,
    )
    update_summary_path = update_root / "return-aligned-guarded-ppo-update-smoke-summary.json"

    readiness_runner = readiness_runner or _run_readiness_validate_only
    readiness = readiness_runner(
        repo_root=repo_root,
        batch_root=batch_root,
        pilot_summary_path=pilot_summary_path,
        config_path=Path(config.get("readiness", {}).get("config", "configs/policy_training_readiness_review_v1.json")),
    )
    _write_json(readiness_path, readiness)

    stale_written_readiness_path = batch_root / "policy-training-readiness-review-summary.json"
    stale_written_readiness = _read_json_if_exists(stale_written_readiness_path)
    stale_written_detected = _stale_written_readiness_detected(stale_written_readiness, readiness)

    artifacts = [
        ("pilot_summary", pilot_summary_path, True),
        ("pilot_episodes", episodes_path, True),
        ("pilot_steps", steps_path, True),
        ("pilot_rejection_report", rejection_report_path, True),
        ("pilot_reward_audit", reward_audit_path, True),
        ("collector_replay_summary", collector_replay_path, True),
        ("long_horizon_summary", long_horizon_path, True),
        ("return_aligned_update_smoke_summary", update_summary_path, True),
        ("readiness_validate_only", readiness_path, True),
        ("written_readiness_summary", stale_written_readiness_path, False),
    ]
    manifest = _build_manifest(artifacts)
    _write_json(manifest_path, manifest)

    collector_replay = _read_json_if_exists(collector_replay_path)
    long_horizon = _read_json_if_exists(long_horizon_path)
    update_summary = _read_json_if_exists(update_summary_path)

    reason_codes: list[str] = []
    _validate_artifacts(manifest, reason_codes)
    _validate_pilot_summary(pilot_summary, config, reason_codes)
    _validate_collector_replay(collector_replay, config, reason_codes)
    _validate_long_horizon(long_horizon, config, reason_codes)
    _validate_update_summary(update_summary, reason_codes)
    _validate_readiness(readiness, config, reason_codes)

    status = "passed" if not reason_codes else "failed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": status,
        "reason_codes": reason_codes,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_provenance": {"current": git_snapshot(repo_root)},
        "pilot_root": str(pilot_root),
        "batch_root": str(batch_root),
        "update_root": str(update_root),
        "output_root": str(output_root),
        "summary": str(summary_path),
        "manifest": str(manifest_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "pilot_status": pilot_summary.get("status"),
        "pilot_reason_codes": list(pilot_summary.get("reason_codes") or []),
        "pilot_episode_count": _int_value(pilot_summary.get("episode_count")),
        "pilot_step_count": _int_value(pilot_summary.get("step_count")),
        "pilot_ppo_trainable_transition_count": _int_value(
            pilot_summary.get("ppo_trainable_transition_count")
        ),
        "pilot_diagnostic_transition_count": _int_value(
            pilot_summary.get("diagnostic_transition_count")
        ),
        "pilot_controlled_regression_count": _int_value(
            pilot_summary.get("controlled_regression_count")
        ),
        "pilot_teacher_agreement_rate": _float_value(
            pilot_summary.get("teacher_agreement_rate")
        ),
        "collector_replay_status": collector_replay.get("status")
        or pilot_summary.get("quasi_real_collector_replay_status"),
        "collector_replay_trainable_transition_count": _int_value(
            collector_replay.get(
                "ppo_trainable_transition_count",
                pilot_summary.get("quasi_real_collector_replay_trainable_transition_count"),
            )
        ),
        "long_horizon_status": long_horizon.get("status"),
        "long_horizon_verdict": long_horizon.get(
            "verdict",
            pilot_summary.get("post_pilot_long_horizon_verdict"),
        ),
        "return_aligned_update_smoke_status": update_summary.get("status"),
        "readiness_status": readiness.get("training_readiness_status"),
        "readiness_reason_codes": list(readiness.get("reason_codes") or []),
        "training_blockers": list(readiness.get("training_blockers") or []),
        "stale_written_readiness_summary_detected": stale_written_detected,
        "stale_written_readiness_summary_path": str(stale_written_readiness_path)
        if stale_written_readiness
        else None,
        "manifest_required_artifact_count": manifest["required_artifact_count"],
        "required_artifact_missing_count": manifest["required_artifact_missing_count"],
        "runs_ppo_update": False,
        "publishes_checkpoint": bool(pilot_summary.get("publishes_checkpoint", False)),
        "replaces_default_policy": bool(pilot_summary.get("replaces_default_policy", False)),
        "performance_claimed": bool(pilot_summary.get("performance_claimed", False)),
        "formal_training_ready_claimed": bool(pilot_summary.get("formal_training_ready_claimed", False)),
    }
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary, manifest), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze quasi-real guarded PPO rollout pilot evidence."
    )
    parser.add_argument(
        "--pilot-root",
        default="outputs/path_feedback_batch_quasi_real_guarded_ppo_rollout_pilot_v1",
    )
    parser.add_argument(
        "--batch-root",
        default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1",
    )
    parser.add_argument(
        "--update-root",
        default="outputs/path_feedback_batch_return_aligned_guarded_ppo_update_smoke_v1",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/path_feedback_batch_quasi_real_guarded_ppo_evidence_freeze_v1",
    )
    parser.add_argument(
        "--config",
        default="configs/quasi_real_guarded_ppo_evidence_freeze_v1.json",
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

    summary = run_quasi_real_guarded_ppo_evidence_freeze(
        pilot_root=_resolve_path(Path(args.pilot_root), repo_root),
        batch_root=_resolve_path(Path(args.batch_root), repo_root),
        update_root=_resolve_path(Path(args.update_root), repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "pilot_status": summary["pilot_status"],
                "readiness_status": summary["readiness_status"],
                "stale_written_readiness_summary_detected": summary[
                    "stale_written_readiness_summary_detected"
                ],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def _run_readiness_validate_only(
    *,
    repo_root: Path,
    batch_root: Path,
    pilot_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        "bash",
        str(repo_root / "scripts" / "run_policy_training_readiness_review.sh"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(_resolve_path(config_path, repo_root)),
        "--quasi-real-guarded-ppo-rollout-pilot-summary",
        str(pilot_summary_path),
        "--validate-only",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=None,
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


def _validate_artifacts(manifest: dict[str, Any], reason_codes: list[str]) -> None:
    if manifest["required_artifact_missing_count"]:
        _add_reason(reason_codes, "required_artifact_missing")


def _validate_pilot_summary(
    summary: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    validation = config.get("validation", {}) if isinstance(config.get("validation"), dict) else {}
    if summary.get("status") != "passed" or summary.get("reason_codes"):
        _add_reason(reason_codes, "pilot_summary_not_passed")
    if _int_value(summary.get("episode_count")) < _int_value(validation.get("min_episode_count"), 36):
        _add_reason(reason_codes, "pilot_episode_count_below_threshold")
    if _int_value(summary.get("step_count")) < _int_value(validation.get("min_step_count"), 108):
        _add_reason(reason_codes, "pilot_step_count_below_threshold")
    trainable_count = _int_value(summary.get("ppo_trainable_transition_count"))
    if trainable_count < _int_value(validation.get("min_ppo_trainable_transition_count"), 24):
        _add_reason(reason_codes, "pilot_ppo_trainable_transition_count_below_threshold")
    expected_trainable = validation.get("expected_ppo_trainable_transition_count")
    if expected_trainable is not None and trainable_count != _int_value(expected_trainable):
        _add_reason(reason_codes, "pilot_ppo_trainable_transition_count_unexpected")
    expected_diagnostic = validation.get("expected_diagnostic_transition_count")
    if expected_diagnostic is not None and _int_value(summary.get("diagnostic_transition_count")) != _int_value(expected_diagnostic):
        _add_reason(reason_codes, "pilot_diagnostic_transition_count_unexpected")
    for field, reason in (
        ("validation_trainable_count", "pilot_split_leakage_detected"),
        ("test_trainable_count", "pilot_split_leakage_detected"),
        ("source_fallback_trainable_count", "pilot_fallback_trainable_detected"),
        ("missing_observation_count", "pilot_materialization_contract_invalid"),
        ("missing_log_prob_count", "pilot_materialization_contract_invalid"),
        ("missing_value_count", "pilot_materialization_contract_invalid"),
        ("non_finite_reward_count", "pilot_non_finite_value_detected"),
        ("non_finite_return_count", "pilot_non_finite_value_detected"),
        ("non_finite_advantage_count", "pilot_non_finite_value_detected"),
        ("controlled_regression_count", "pilot_controlled_regression_detected"),
        ("controlled_safety_regression_count", "pilot_controlled_regression_detected"),
        ("controlled_contract_regression_count", "pilot_controlled_regression_detected"),
        ("controlled_path_risk_regression_count", "pilot_controlled_regression_detected"),
        ("controlled_source_selection_regression_count", "pilot_controlled_regression_detected"),
    ):
        if _int_value(summary.get(field)):
            _add_reason(reason_codes, reason)
    if _float_value(summary.get("teacher_agreement_rate")) < float(
        validation.get("min_teacher_agreement_rate", 0.9)
    ):
        _add_reason(reason_codes, "pilot_teacher_agreement_below_threshold")
    for field, reason in (
        ("publishes_checkpoint", "checkpoint_publication_claimed"),
        ("replaces_default_policy", "default_policy_replacement_claimed"),
        ("performance_claimed", "performance_claimed"),
        ("formal_training_ready_claimed", "formal_training_ready_claimed"),
    ):
        if summary.get(field) is True:
            _add_reason(reason_codes, reason)


def _validate_collector_replay(
    summary: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    validation = config.get("validation", {}) if isinstance(config.get("validation"), dict) else {}
    if summary.get("status") != "passed" or summary.get("reason_codes"):
        _add_reason(reason_codes, "collector_replay_not_passed")
    if _int_value(summary.get("ppo_trainable_transition_count")) < _int_value(
        validation.get("min_collector_replay_trainable_transition_count"),
        24,
    ):
        _add_reason(reason_codes, "collector_replay_trainable_count_below_threshold")


def _validate_long_horizon(
    summary: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    validation = config.get("validation", {}) if isinstance(config.get("validation"), dict) else {}
    expected = str(
        validation.get(
            "expected_long_horizon_verdict",
            "long_horizon_teacher_skill_contract_aligned",
        )
    )
    if summary.get("status") != "passed" or summary.get("reason_codes"):
        _add_reason(reason_codes, "long_horizon_summary_not_passed")
    if summary.get("verdict") != expected:
        _add_reason(reason_codes, "long_horizon_verdict_not_aligned")


def _validate_update_summary(summary: dict[str, Any], reason_codes: list[str]) -> None:
    if summary.get("status") != "passed" or summary.get("reason_codes"):
        _add_reason(reason_codes, "return_aligned_update_smoke_summary_not_passed")
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
        _add_reason(reason_codes, "readiness_not_quasi_real_guarded_ppo_rollout_pilot_evaluated")
    if readiness.get("reason_codes"):
        _add_reason(reason_codes, "readiness_reason_codes_non_empty")
    if readiness.get("training_blockers"):
        _add_reason(reason_codes, "readiness_training_blockers_non_empty")
    if _int_value(readiness.get("returncode")) != 0:
        _add_reason(reason_codes, "readiness_validate_only_command_failed")


def _stale_written_readiness_detected(
    written: dict[str, Any],
    explicit: dict[str, Any],
) -> bool:
    if not written:
        return False
    return (
        written.get("training_readiness_status") != explicit.get("training_readiness_status")
        or list(written.get("reason_codes") or []) != list(explicit.get("reason_codes") or [])
        or list(written.get("training_blockers") or []) != list(explicit.get("training_blockers") or [])
        or written.get("status") == "failed"
    )


def _build_manifest(artifact_specs: list[tuple[str, Path, bool]]) -> dict[str, Any]:
    artifacts = [_artifact_payload(label, path, required) for label, path, required in artifact_specs]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifacts": artifacts,
        "required_artifact_count": sum(1 for item in artifacts if item["required"]),
        "required_artifact_missing_count": sum(
            1 for item in artifacts if item["required"] and not item["exists"]
        ),
    }


def _artifact_payload(label: str, path: Path, required: bool) -> dict[str, Any]:
    exists = path.is_file()
    payload: dict[str, Any] = {
        "label": label,
        "path": str(path),
        "required": required,
        "exists": exists,
        "sha256": _sha256(path) if exists else None,
        "schema_version": None,
        "status": None,
        "reason_codes": None,
        "git_provenance": None,
        "summary": {},
    }
    if not exists:
        return payload
    if path.suffix == ".json":
        data = _read_json(path)
        payload["schema_version"] = data.get("schema_version")
        payload["status"] = data.get("status") or data.get("training_readiness_status")
        payload["reason_codes"] = data.get("reason_codes")
        payload["git_provenance"] = data.get("git_provenance")
        payload["summary"] = _key_summary(data)
    elif path.suffix == ".jsonl":
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        first = json.loads(lines[0]) if lines else {}
        payload["schema_version"] = first.get("schema_version")
        payload["status"] = "present"
        payload["summary"] = {"line_count": len(lines)}
    return payload


def _render_report(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    artifacts = "\n".join(
        f"- {item['label']}: exists={item['exists']} sha256={item['sha256'] or 'missing'}"
        for item in manifest["artifacts"]
    )
    return (
        "# Quasi-Real Guarded PPO Evidence Freeze\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Readiness: `{summary.get('readiness_status')}`\n"
        f"- Pilot status: `{summary.get('pilot_status')}`\n"
        f"- Pilot episodes / steps: `{summary.get('pilot_episode_count')}` / `{summary.get('pilot_step_count')}`\n"
        f"- Pilot trainable / diagnostic: `{summary.get('pilot_ppo_trainable_transition_count')}` / `{summary.get('pilot_diagnostic_transition_count')}`\n"
        f"- Controlled regression count: `{summary.get('pilot_controlled_regression_count')}`\n"
        f"- Teacher agreement rate: `{summary.get('pilot_teacher_agreement_rate')}`\n"
        f"- Long horizon verdict: `{summary.get('long_horizon_verdict')}`\n"
        f"- stale written readiness summary detected: `{summary.get('stale_written_readiness_summary_detected')}`\n\n"
        "## Artifacts\n\n"
        f"{artifacts}\n\n"
        "## Non-Goals\n\n"
        "No new PPO update, checkpoint publication, default policy replacement, gate relaxation, "
        "policy performance claim, or formal training-ready claim is made by this freeze.\n"
    )


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "summary": SUMMARY_FILE,
        "manifest": MANIFEST_FILE,
        "readiness_validate_only": READINESS_FILE,
        "report": REPORT_FILE,
    }
    configured = config.get("output_files") if isinstance(config.get("output_files"), dict) else {}
    return {key: str(configured.get(key) or default) for key, default in defaults.items()}


def _artifact_path(value: Any, default: Path, repo_root: Path) -> Path:
    if not value:
        return default
    path = Path(str(value))
    if path.is_absolute():
        return path
    repo_candidate = repo_root / path
    return repo_candidate if repo_candidate.exists() else default.parent / path


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _key_summary(data: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "reason_codes",
        "training_readiness_status",
        "training_blockers",
        "episode_count",
        "step_count",
        "ppo_trainable_transition_count",
        "diagnostic_transition_count",
        "controlled_regression_count",
        "teacher_agreement_rate",
        "verdict",
    )
    return {key: data.get(key) for key in keys if key in data}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


if __name__ == "__main__":
    raise SystemExit(main())
