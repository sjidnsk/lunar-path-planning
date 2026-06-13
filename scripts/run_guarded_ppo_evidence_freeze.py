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


CONFIG_SCHEMA_VERSION = "guarded-ppo-evidence-freeze-config/v1"
SUMMARY_SCHEMA_VERSION = "guarded-ppo-evidence-freeze-summary/v1"
MANIFEST_SCHEMA_VERSION = "guarded-ppo-evidence-manifest/v1"
CONSISTENCY_SCHEMA_VERSION = "guarded-ppo-evidence-consistency-report/v1"
EXPECTED_READINESS_STATUS = "guarded_ppo_rollout_pilot_evaluated"

SUMMARY_FILE = "guarded-ppo-evidence-freeze-summary.json"
MANIFEST_FILE = "evidence-manifest.json"
READINESS_FINAL_FILE = "readiness-final.json"
CONSISTENCY_FILE = "progress-consistency-report.json"
REPORT_FILE = "reproducibility-report.md"


def run_guarded_ppo_evidence_freeze(
    *,
    guarded_root: Path,
    batch_root: Path,
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    readiness_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    guarded_root = Path(guarded_root)
    batch_root = Path(batch_root)
    output_root = Path(output_root)
    repo_root = Path(repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    output_files = _output_files(config)
    guarded_summary_path = guarded_root / "guarded-ppo-rollout-pilot-summary.json"
    progress_summary_path = guarded_root / "training-progress-summary.json"
    progress_events_path = guarded_root / "training-progress-events.jsonl"
    readiness_final_path = output_root / output_files["readiness_final"]
    summary_path = output_root / output_files["summary"]
    manifest_path = output_root / output_files["manifest"]
    consistency_path = output_root / output_files["consistency_report"]
    report_path = output_root / output_files["reproducibility_report"]

    guarded_summary = _read_json_if_exists(guarded_summary_path)
    progress_summary = _read_json_if_exists(progress_summary_path)
    progress_events = _read_jsonl_if_exists(progress_events_path)
    readiness_runner = readiness_runner or _run_readiness_validate_only
    readiness_final = readiness_runner(
        repo_root=repo_root,
        batch_root=batch_root,
        guarded_summary_path=guarded_summary_path,
        config_path=Path(config.get("readiness", {}).get("config", "configs/policy_training_readiness_review_v1.json")),
    )
    _write_json(readiness_final_path, readiness_final)

    stale_readiness_path = batch_root / "policy-training-readiness-review-summary.json"
    stale_readiness = _read_json_if_exists(stale_readiness_path)
    consistency = _build_consistency_report(
        stale_readiness=stale_readiness,
        stale_readiness_path=stale_readiness_path,
        final_readiness=readiness_final,
        readiness_final_path=readiness_final_path,
        progress_summary=progress_summary,
        progress_events=progress_events,
    )
    _write_json(consistency_path, consistency)

    artifact_specs = [
        ("guarded_pilot_summary", guarded_summary_path, True),
        ("training_progress_summary", progress_summary_path, True),
        ("training_progress_events", progress_events_path, True),
        ("readiness_final", readiness_final_path, True),
    ]
    manifest = _build_manifest(artifact_specs)
    _write_json(manifest_path, manifest)

    reason_codes: list[str] = []
    missing = [item for item in manifest["artifacts"] if item["required"] and not item["exists"]]
    if missing:
        _add_reason(reason_codes, "required_artifact_missing")
    if guarded_summary.get("status") != "passed" or guarded_summary.get("reason_codes"):
        _add_reason(reason_codes, "guarded_pilot_summary_not_passed")
    if progress_summary.get("status") != "passed" or int(progress_summary.get("failed_stage_count") or 0) > 0:
        _add_reason(reason_codes, "training_progress_summary_not_passed")
    if not progress_events:
        _add_reason(reason_codes, "training_progress_events_missing_or_empty")
    readiness_status = readiness_final.get("training_readiness_status")
    expected_status = str(config.get("readiness", {}).get("expected_status") or EXPECTED_READINESS_STATUS)
    if readiness_status != expected_status:
        _add_reason(reason_codes, "readiness_not_guarded_ppo_rollout_pilot_evaluated")
    if readiness_final.get("reason_codes"):
        _add_reason(reason_codes, "readiness_reason_codes_non_empty")
    if readiness_final.get("training_blockers"):
        _add_reason(reason_codes, "readiness_training_blockers_non_empty")

    status = "passed" if not reason_codes else "failed"
    debug_artifact = _debug_artifact(missing, guarded_summary_path, progress_summary_path, readiness_final_path, reason_codes)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": status,
        "reason_codes": reason_codes,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_provenance": {"current": git_snapshot(repo_root)},
        "guarded_root": str(guarded_root),
        "batch_root": str(batch_root),
        "output_root": str(output_root),
        "guarded_pilot_summary": str(guarded_summary_path),
        "training_progress_summary": str(progress_summary_path),
        "training_progress_events": str(progress_events_path),
        "readiness_final": str(readiness_final_path),
        "manifest": str(manifest_path),
        "consistency_report": str(consistency_path),
        "reproducibility_report": str(report_path),
        "guarded_pilot_status": guarded_summary.get("status"),
        "guarded_pilot_reason_codes": list(guarded_summary.get("reason_codes") or []),
        "guarded_rollout_pilot_passed": bool(guarded_summary.get("guarded_rollout_pilot_passed")),
        "ppo_trainable_transition_count": _int_value(guarded_summary.get("ppo_trainable_transition_count")),
        "optimizer_train_transition_count": _int_value(guarded_summary.get("optimizer_train_transition_count")),
        "post_update_controlled_sequential_regression_count": _int_value(
            guarded_summary.get("post_update_controlled_sequential_regression_count")
        ),
        "post_update_quasi_real_collector_trainable_transition_count": _int_value(
            guarded_summary.get("post_update_quasi_real_collector_trainable_transition_count")
        ),
        "progress_status": progress_summary.get("status"),
        "progress_failed_stage_count": _int_value(progress_summary.get("failed_stage_count")),
        "progress_event_count": len(progress_events),
        "training_readiness_status": readiness_status,
        "training_blockers": list(readiness_final.get("training_blockers") or []),
        "readiness_reason_codes": list(readiness_final.get("reason_codes") or []),
        "required_artifact_missing_count": len(missing),
        "manifest_required_artifact_count": sum(1 for item in manifest["artifacts"] if item["required"]),
        "stale_readiness_detected": bool(consistency["stale_readiness_detected"]),
        "recommended_debug_artifact": debug_artifact,
        "publishes_checkpoint": bool(guarded_summary.get("publishes_checkpoint", False)),
        "replaces_default_policy": bool(guarded_summary.get("replaces_default_policy", False)),
        "performance_claimed": bool(guarded_summary.get("performance_claimed", False)),
        "formal_training_ready_claimed": bool(guarded_summary.get("formal_training_ready_claimed", False)),
    }
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary, manifest, consistency), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze guarded PPO rollout pilot evidence into an auditable package.")
    parser.add_argument("--guarded-root", default="outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1")
    parser.add_argument("--batch-root", default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1")
    parser.add_argument("--output-root", default="outputs/path_feedback_batch_guarded_ppo_evidence_freeze_v1")
    parser.add_argument("--config", default="configs/guarded_ppo_evidence_freeze_v1.json")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    config = _read_json(Path(args.config))
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise SystemExit(f"invalid config schema: {config.get('schema_version')}")
    if args.validate_only:
        print(json.dumps({"status": "config validated", "config": args.config}, sort_keys=True))
        return 0
    summary = run_guarded_ppo_evidence_freeze(
        guarded_root=Path(args.guarded_root),
        batch_root=Path(args.batch_root),
        output_root=Path(args.output_root),
        config=config,
        repo_root=repo_root,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def _run_readiness_validate_only(
    *,
    repo_root: Path,
    batch_root: Path,
    guarded_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        "bash",
        str(repo_root / "scripts" / "run_policy_training_readiness_review.sh"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(config_path),
        "--guarded-ppo-rollout-pilot-summary",
        str(guarded_summary_path),
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
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        if returncode != 0:
            return {
                "training_readiness_status": "readiness_validate_only_failed",
                "reason_codes": ["readiness_validate_only_command_failed"],
                "training_blockers": [stderr.strip()],
                "command": command,
            }
        return {
            "training_readiness_status": "readiness_validate_only_unparseable",
            "reason_codes": ["readiness_validate_only_stdout_unparseable"],
            "training_blockers": [stdout[:1000]],
        }
    result["command"] = command
    return result


def _build_manifest(artifact_specs: list[tuple[str, Path, bool]]) -> dict[str, Any]:
    artifacts = []
    for label, path, required in artifact_specs:
        payload = _artifact_payload(label, path, required)
        artifacts.append(payload)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifacts": artifacts,
        "required_artifact_count": sum(1 for item in artifacts if item["required"]),
        "required_artifact_missing_count": sum(1 for item in artifacts if item["required"] and not item["exists"]),
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
        "summary": {},
    }
    if not exists:
        return payload
    if path.suffix == ".json":
        data = _read_json(path)
        payload["schema_version"] = data.get("schema_version")
        payload["status"] = data.get("status") or data.get("training_readiness_status")
        payload["summary"] = _key_summary(data)
    elif path.suffix == ".jsonl":
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        first = json.loads(lines[0]) if lines else {}
        payload["schema_version"] = first.get("schema_version")
        payload["status"] = "present"
        payload["summary"] = {"line_count": len(lines), "first_stage": first.get("stage")}
    return payload


def _build_consistency_report(
    *,
    stale_readiness: dict[str, Any],
    stale_readiness_path: Path,
    final_readiness: dict[str, Any],
    readiness_final_path: Path,
    progress_summary: dict[str, Any],
    progress_events: list[dict[str, Any]],
) -> dict[str, Any]:
    stale_status = stale_readiness.get("training_readiness_status")
    final_status = final_readiness.get("training_readiness_status")
    stale_detected = bool(stale_status and final_status and stale_status != final_status)
    return {
        "schema_version": CONSISTENCY_SCHEMA_VERSION,
        "final_readiness_source": "explicit_guarded_summary_validate_only",
        "final_readiness_path": str(readiness_final_path),
        "final_training_readiness_status": final_status,
        "stale_readiness_path": str(stale_readiness_path) if stale_readiness else None,
        "stale_training_readiness_status": stale_status,
        "stale_readiness_detected": stale_detected,
        "progress_readiness_status": progress_summary.get("readiness_status"),
        "progress_status": progress_summary.get("status"),
        "progress_failed_stage_count": _int_value(progress_summary.get("failed_stage_count")),
        "progress_event_count": len(progress_events),
    }


def _render_report(summary: dict[str, Any], manifest: dict[str, Any], consistency: dict[str, Any]) -> str:
    artifacts = "\n".join(
        f"- {item['label']}: exists={item['exists']} sha256={item['sha256'] or 'missing'}"
        for item in manifest["artifacts"]
    )
    return (
        "# Guarded PPO Evidence Freeze\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Readiness: `{summary.get('training_readiness_status')}`\n"
        f"- Guarded trainable transitions: `{summary.get('ppo_trainable_transition_count')}`\n"
        f"- Optimizer transitions: `{summary.get('optimizer_train_transition_count')}`\n"
        f"- Progress events: `{summary.get('progress_event_count')}`\n"
        f"- Stale readiness detected: `{consistency.get('stale_readiness_detected')}`\n\n"
        "## Artifacts\n\n"
        f"{artifacts}\n\n"
        "## Non-Goals\n\n"
        "No PPO update, checkpoint publication, default policy replacement, gate relaxation, "
        "or formal training readiness claim is made by this freeze.\n"
    )


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "summary": SUMMARY_FILE,
        "manifest": MANIFEST_FILE,
        "readiness_final": READINESS_FINAL_FILE,
        "consistency_report": CONSISTENCY_FILE,
        "reproducibility_report": REPORT_FILE,
    }
    configured = config.get("output_files") if isinstance(config.get("output_files"), dict) else {}
    return {key: str(configured.get(key) or default) for key, default in defaults.items()}


def _debug_artifact(
    missing: list[dict[str, Any]],
    guarded_summary_path: Path,
    progress_summary_path: Path,
    readiness_final_path: Path,
    reason_codes: list[str],
) -> str | None:
    if missing:
        return str(missing[0]["path"])
    if "guarded_pilot_summary_not_passed" in reason_codes:
        return str(guarded_summary_path)
    if "training_progress_summary_not_passed" in reason_codes:
        return str(progress_summary_path)
    if any(reason.startswith("readiness_") for reason in reason_codes):
        return str(readiness_final_path)
    return None


def _key_summary(data: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "reason_codes",
        "training_readiness_status",
        "training_blockers",
        "ppo_trainable_transition_count",
        "optimizer_train_transition_count",
        "failed_stage_count",
        "event_count",
    )
    return {key: data.get(key) for key in keys if key in data}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


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


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


if __name__ == "__main__":
    raise SystemExit(main())
