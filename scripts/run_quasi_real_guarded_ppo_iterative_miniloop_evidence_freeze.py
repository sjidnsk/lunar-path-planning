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
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot


CONFIG_SCHEMA_VERSION = "quasi-real-guarded-ppo-iterative-miniloop-evidence-freeze-config/v1"
SUMMARY_SCHEMA_VERSION = "quasi-real-guarded-ppo-iterative-miniloop-evidence-freeze-summary/v1"
MANIFEST_SCHEMA_VERSION = "quasi-real-guarded-ppo-iterative-miniloop-evidence-manifest/v1"
MINILOOP_SUMMARY_SCHEMA_VERSION = "quasi-real-guarded-ppo-iterative-miniloop-stability-summary/v1"
EXPECTED_READINESS_STATUS = "quasi_real_guarded_ppo_iterative_miniloop_stability_evaluated"

SUMMARY_FILE = "quasi-real-guarded-ppo-iterative-miniloop-evidence-freeze-summary.json"
MANIFEST_FILE = "quasi-real-guarded-ppo-iterative-miniloop-evidence-manifest.json"
READINESS_FILE = "quasi-real-guarded-ppo-iterative-miniloop-readiness-validate-only.json"
REPORT_FILE = "quasi-real-guarded-ppo-iterative-miniloop-evidence-freeze-report.md"

MINILOOP_SUMMARY_FILE = "quasi-real-guarded-ppo-iterative-miniloop-stability-summary.json"
MINILOOP_ITERATION_SUMMARIES_FILE = "iterative-miniloop-iteration-summaries.jsonl"
MINILOOP_PROGRESS_FILE = "iterative-miniloop-progress.jsonl"
MINILOOP_READINESS_FILE = "iterative-miniloop-readiness-validate-only.json"
MINILOOP_REPORT_FILE = "iterative-miniloop-stability-report.md"

ReadinessRunner = Callable[..., dict[str, Any]]


def run_quasi_real_guarded_ppo_iterative_miniloop_evidence_freeze(
    *,
    miniloop_root: Path,
    batch_root: Path,
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    readiness_runner: ReadinessRunner | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    miniloop_root = Path(miniloop_root)
    batch_root = Path(batch_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = _output_files(config)
    summary_path = output_root / files["summary"]
    manifest_path = output_root / files["manifest"]
    readiness_path = output_root / files["readiness_validate_only"]
    report_path = output_root / files["report"]

    miniloop_summary_path = miniloop_root / MINILOOP_SUMMARY_FILE
    miniloop_summary = _read_json_if_exists(miniloop_summary_path)
    progress_path = _artifact_path(
        miniloop_summary.get("progress_jsonl"),
        miniloop_root / MINILOOP_PROGRESS_FILE,
        repo_root,
    )
    iteration_summaries_path = _artifact_path(
        miniloop_summary.get("iteration_summaries"),
        miniloop_root / MINILOOP_ITERATION_SUMMARIES_FILE,
        repo_root,
    )
    input_readiness_path = _artifact_path(
        miniloop_summary.get("readiness_validate_only"),
        miniloop_root / MINILOOP_READINESS_FILE,
        repo_root,
    )
    miniloop_report_path = _artifact_path(
        miniloop_summary.get("report"),
        miniloop_root / MINILOOP_REPORT_FILE,
        repo_root,
    )

    runner = readiness_runner or _run_readiness_validate_only
    readiness = runner(
        repo_root=repo_root,
        batch_root=batch_root,
        iterative_summary_path=miniloop_summary_path,
        config_path=Path(
            config.get("readiness", {}).get(
                "config", "configs/policy_training_readiness_review_v1.json"
            )
        ),
    )
    _write_json(readiness_path, readiness)

    progress_rows = _read_jsonl(progress_path)
    iteration_rows = _read_jsonl(iteration_summaries_path)
    input_readiness = _read_json_if_exists(input_readiness_path)

    artifacts = [
        ("miniloop_summary", miniloop_summary_path, True),
        ("miniloop_progress_jsonl", progress_path, True),
        ("miniloop_iteration_summaries_jsonl", iteration_summaries_path, True),
        ("miniloop_input_readiness_validate_only", input_readiness_path, True),
        ("miniloop_stability_report", miniloop_report_path, True),
        ("readiness_validate_only", readiness_path, True),
    ]
    for rel_path in config.get("tracked_source_files", []):
        artifacts.append((_source_artifact_name(str(rel_path)), _resolve_path(Path(str(rel_path)), repo_root), True))

    manifest = _build_manifest(artifacts, repo_root=repo_root)
    _write_json(manifest_path, manifest)

    reason_codes: list[str] = []
    _validate_manifest(manifest, reason_codes)
    _validate_miniloop_summary(
        miniloop_summary=miniloop_summary,
        progress_rows=progress_rows,
        iteration_rows=iteration_rows,
        config=config,
        reason_codes=reason_codes,
    )
    _validate_readiness(readiness, config, reason_codes)
    _validate_input_readiness(input_readiness, reason_codes)

    status = "passed" if not reason_codes else "failed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": status,
        "reason_codes": reason_codes,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_provenance": {"current": git_snapshot(repo_root)},
        "miniloop_root": str(miniloop_root),
        "batch_root": str(batch_root),
        "output_root": str(output_root),
        "summary": str(summary_path),
        "manifest": str(manifest_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "miniloop_summary": str(miniloop_summary_path),
        "miniloop_progress_jsonl": str(progress_path),
        "miniloop_iteration_summaries": str(iteration_summaries_path),
        "miniloop_input_readiness_validate_only": str(input_readiness_path),
        "miniloop_status": miniloop_summary.get("status"),
        "miniloop_reason_codes": list(miniloop_summary.get("reason_codes") or []),
        "readiness_status": readiness.get("training_readiness_status"),
        "readiness_reason_codes": list(readiness.get("reason_codes") or []),
        "training_blockers": list(readiness.get("training_blockers") or []),
        "input_trainable_transition_count": _int_value(
            miniloop_summary.get("input_trainable_transition_count")
        ),
        "unique_trainable_context_count": _int_value(
            miniloop_summary.get("unique_trainable_context_count")
        ),
        "ppo_trainable_transition_count": _int_value(
            miniloop_summary.get("ppo_trainable_transition_count")
        ),
        "seed_count": _int_value(miniloop_summary.get("seed_count")),
        "iteration_count": _int_value(miniloop_summary.get("iteration_count")),
        "passed_iteration_count": _int_value(miniloop_summary.get("passed_iteration_count")),
        "failed_iteration_count": _int_value(miniloop_summary.get("failed_iteration_count")),
        "progress_row_count": len(progress_rows),
        "iteration_summary_row_count": len(iteration_rows),
        "max_old_log_prob_abs_error": _float_value(
            miniloop_summary.get("max_old_log_prob_abs_error")
        ),
        "max_old_value_abs_error": _float_value(miniloop_summary.get("max_old_value_abs_error")),
        "max_abs_approx_kl": _float_value(miniloop_summary.get("max_abs_approx_kl")),
        "max_grad_norm_after_clip": _float_value(
            miniloop_summary.get("max_grad_norm_after_clip")
        ),
        "controlled_regression_count": _int_value(
            miniloop_summary.get("controlled_regression_count")
        ),
        "behavior_drift_count": _int_value(miniloop_summary.get("behavior_drift_count")),
        "manifest_required_artifact_count": manifest["required_artifact_count"],
        "required_artifact_missing_count": manifest["required_artifact_missing_count"],
        "runs_formal_ppo_rollout": bool(
            miniloop_summary.get("runs_formal_ppo_rollout", False)
        ),
        "publishes_checkpoint": bool(miniloop_summary.get("publishes_checkpoint", False)),
        "replaces_default_policy": bool(miniloop_summary.get("replaces_default_policy", False)),
        "performance_claimed": bool(miniloop_summary.get("performance_claimed", False)),
        "formal_training_ready_claimed": bool(
            miniloop_summary.get("formal_training_ready_claimed", False)
        ),
        "baseline_frozen": status == "passed",
    }
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary, manifest), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze current quasi-real guarded PPO iterative mini-loop evidence."
    )
    parser.add_argument(
        "--miniloop-root",
        default="outputs/path_feedback_batch_quasi_real_guarded_ppo_iterative_miniloop_stability_v1",
    )
    parser.add_argument(
        "--batch-root",
        default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/path_feedback_batch_quasi_real_guarded_ppo_iterative_miniloop_evidence_freeze_v1",
    )
    parser.add_argument(
        "--config",
        default="configs/quasi_real_guarded_ppo_iterative_miniloop_evidence_freeze_v1.json",
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

    summary = run_quasi_real_guarded_ppo_iterative_miniloop_evidence_freeze(
        miniloop_root=_resolve_path(Path(args.miniloop_root), repo_root),
        batch_root=_resolve_path(Path(args.batch_root), repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "miniloop_status": summary["miniloop_status"],
                "readiness_status": summary["readiness_status"],
                "input_trainable_transition_count": summary["input_trainable_transition_count"],
                "progress_row_count": summary["progress_row_count"],
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
    iterative_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        "bash",
        str(repo_root / "scripts" / "run_policy_training_readiness_review.sh"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(_resolve_path(config_path, repo_root)),
        "--quasi-real-guarded-ppo-iterative-miniloop-stability-summary",
        str(iterative_summary_path),
        "--validate-only",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode == 0:
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError:
            pass
    return {
        "training_readiness_status": "readiness_validate_only_failed",
        "reason_codes": ["readiness_validate_only_failed"],
        "training_blockers": ["readiness_validate_only_failed"],
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    files = dict(config.get("output_files") or {})
    return {
        "summary": files.get("summary", SUMMARY_FILE),
        "manifest": files.get("manifest", MANIFEST_FILE),
        "readiness_validate_only": files.get("readiness_validate_only", READINESS_FILE),
        "report": files.get("report", REPORT_FILE),
    }


def _validate_manifest(manifest: dict[str, Any], reason_codes: list[str]) -> None:
    if _int_value(manifest.get("required_artifact_missing_count")):
        _append_reason(reason_codes, "required_artifact_missing")


def _validate_miniloop_summary(
    *,
    miniloop_summary: dict[str, Any],
    progress_rows: list[dict[str, Any]],
    iteration_rows: list[dict[str, Any]],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    validation = dict(config.get("validation") or {})
    if miniloop_summary.get("schema_version") != MINILOOP_SUMMARY_SCHEMA_VERSION:
        _append_reason(reason_codes, "miniloop_summary_schema_invalid")
    if miniloop_summary.get("status") != "passed" or miniloop_summary.get("reason_codes"):
        _append_reason(reason_codes, "miniloop_summary_not_passed")

    expected_trainable = _int_value(validation.get("expected_trainable_transition_count"))
    if expected_trainable and _int_value(miniloop_summary.get("input_trainable_transition_count")) != expected_trainable:
        _append_reason(reason_codes, "miniloop_trainable_transition_count_mismatch")
    expected_unique = _int_value(validation.get("expected_unique_trainable_context_count"))
    if expected_unique and _int_value(miniloop_summary.get("unique_trainable_context_count")) != expected_unique:
        _append_reason(reason_codes, "miniloop_unique_context_count_mismatch")
    expected_seed_count = _int_value(validation.get("expected_seed_count"))
    if expected_seed_count and _int_value(miniloop_summary.get("seed_count")) != expected_seed_count:
        _append_reason(reason_codes, "miniloop_seed_count_mismatch")
    expected_iteration_count = _int_value(validation.get("expected_iteration_count"))
    if expected_iteration_count and _int_value(miniloop_summary.get("iteration_count")) != expected_iteration_count:
        _append_reason(reason_codes, "miniloop_iteration_count_mismatch")
    expected_passed_iterations = _int_value(validation.get("expected_passed_iteration_count"))
    passed_iterations = _int_value(miniloop_summary.get("passed_iteration_count"))
    if expected_passed_iterations and passed_iterations != expected_passed_iterations:
        _append_reason(reason_codes, "miniloop_passed_iteration_count_mismatch")
    if _int_value(miniloop_summary.get("failed_iteration_count")):
        _append_reason(reason_codes, "miniloop_failed_iteration_count_nonzero")

    if len(progress_rows) != passed_iterations:
        _append_reason(reason_codes, "miniloop_progress_row_count_mismatch")
    if len(iteration_rows) != passed_iterations:
        _append_reason(reason_codes, "miniloop_iteration_summary_row_count_mismatch")
    if any(row.get("status") != "passed" for row in progress_rows):
        _append_reason(reason_codes, "miniloop_progress_row_not_passed")
    if any(row.get("status") != "passed" for row in iteration_rows):
        _append_reason(reason_codes, "miniloop_iteration_summary_row_not_passed")

    for field, reason in (
        ("validation_trainable_count", "miniloop_split_leakage"),
        ("test_trainable_count", "miniloop_split_leakage"),
        ("source_fallback_trainable_count", "miniloop_fallback_trainable"),
        ("teacher_fallback_trainable_count", "miniloop_fallback_trainable"),
        ("non_empty_gate_reason_trainable_count", "miniloop_gate_reason_trainable"),
        ("missing_observation_count", "miniloop_contract_invalid"),
        ("missing_log_prob_count", "miniloop_contract_invalid"),
        ("missing_value_count", "miniloop_contract_invalid"),
        ("non_finite_reward_count", "miniloop_non_finite_value"),
        ("non_finite_return_count", "miniloop_non_finite_value"),
        ("non_finite_advantage_count", "miniloop_non_finite_value"),
        ("loss_non_finite_count", "miniloop_non_finite_value"),
        ("non_finite_gradient_count", "miniloop_non_finite_value"),
        ("controlled_regression_count", "miniloop_controlled_regression"),
        ("controlled_safety_regression_count", "miniloop_controlled_regression"),
        ("controlled_contract_regression_count", "miniloop_controlled_regression"),
        ("controlled_path_risk_regression_count", "miniloop_controlled_regression"),
        ("controlled_source_selection_regression_count", "miniloop_controlled_regression"),
        ("behavior_drift_count", "miniloop_behavior_drift"),
    ):
        if _int_value(miniloop_summary.get(field)):
            _append_reason(reason_codes, reason)

    if bool(miniloop_summary.get("runs_formal_ppo_rollout", False)):
        _append_reason(reason_codes, "formal_ppo_rollout_unexpected")
    if bool(miniloop_summary.get("publishes_checkpoint", False)):
        _append_reason(reason_codes, "checkpoint_publication_claimed")
    if bool(miniloop_summary.get("replaces_default_policy", False)):
        _append_reason(reason_codes, "default_policy_replacement_claimed")
    if bool(miniloop_summary.get("performance_claimed", False)):
        _append_reason(reason_codes, "policy_performance_claimed")
    if bool(miniloop_summary.get("formal_training_ready_claimed", False)):
        _append_reason(reason_codes, "formal_training_ready_claimed")


def _validate_readiness(
    readiness: dict[str, Any], config: dict[str, Any], reason_codes: list[str]
) -> None:
    expected = str(config.get("readiness", {}).get("expected_status", EXPECTED_READINESS_STATUS))
    if readiness.get("training_readiness_status") != expected:
        _append_reason(reason_codes, "readiness_status_mismatch")
    if readiness.get("reason_codes"):
        _append_reason(reason_codes, "readiness_reason_codes_non_empty")
    if readiness.get("training_blockers"):
        _append_reason(reason_codes, "readiness_training_blockers_non_empty")


def _validate_input_readiness(
    input_readiness: dict[str, Any], reason_codes: list[str]
) -> None:
    if input_readiness and input_readiness.get("training_readiness_status") != EXPECTED_READINESS_STATUS:
        _append_reason(reason_codes, "input_readiness_status_mismatch")


def _build_manifest(
    artifacts: list[tuple[str, Path, bool]], *, repo_root: Path
) -> dict[str, Any]:
    rows = []
    for name, path, required in artifacts:
        exists = path.is_file()
        rows.append(
            {
                "name": name,
                "path": _display_path(path, repo_root),
                "required": required,
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else 0,
                "sha256": _sha256(path) if exists else None,
            }
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "required_artifact_count": sum(1 for row in rows if row["required"]),
        "required_artifact_missing_count": sum(
            1 for row in rows if row["required"] and not row["exists"]
        ),
        "artifacts": rows,
    }


def _render_report(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    missing = [
        item["name"]
        for item in manifest.get("artifacts", [])
        if item.get("required") and not item.get("exists")
    ]
    return "\n".join(
        [
            "# Quasi-Real Guarded PPO Iterative Mini-Loop Evidence Freeze",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- readiness: `{summary.get('readiness_status')}`",
            f"- trainable transitions: `{summary.get('input_trainable_transition_count')}`",
            f"- unique trainable contexts: `{summary.get('unique_trainable_context_count')}`",
            f"- seeds x iterations: `{summary.get('seed_count')} x {summary.get('iteration_count')}`",
            f"- passed iterations: `{summary.get('passed_iteration_count')}`",
            f"- progress rows: `{summary.get('progress_row_count')}`",
            f"- controlled regression count: `{summary.get('controlled_regression_count')}`",
            f"- behavior drift count: `{summary.get('behavior_drift_count')}`",
            f"- required artifact missing count: `{summary.get('required_artifact_missing_count')}`",
            f"- missing required artifacts: `{missing}`",
            "",
            "This is a baseline freeze only. It does not run formal PPO, publish a checkpoint,",
            "replace the default policy, or claim policy performance.",
            "",
        ]
    )


def _source_artifact_name(path: str) -> str:
    mapping = {
        "README.md": "readme_doc",
        "docs/算法设计与系统架构报告.md": "architecture_report_doc",
        "tests/test_quasi_real_guarded_ppo_iterative_miniloop_stability.py": "iterative_miniloop_test",
    }
    return mapping.get(path, path.replace("/", "_").replace(".", "_"))


def _artifact_path(value: Any, fallback: Path, repo_root: Path) -> Path:
    if isinstance(value, str) and value:
        return _resolve_path(Path(value), repo_root)
    return fallback


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return _read_json(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _append_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


if __name__ == "__main__":
    raise SystemExit(main())
