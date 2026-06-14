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

try:
    from scripts.run_quasi_real_guarded_ppo_scale512_multiseed_preflight import (
        _run_seed_smoke,
        _unique_trainable_steps,
    )
except ModuleNotFoundError:  # pragma: no cover
    from run_quasi_real_guarded_ppo_scale512_multiseed_preflight import (
        _run_seed_smoke,
        _unique_trainable_steps,
    )


CONFIG_SCHEMA_VERSION = "quasi-real-guarded-formal-ppo-preflight-config/v1"
SUMMARY_SCHEMA_VERSION = "quasi-real-guarded-formal-ppo-preflight-summary/v1"
SEED_SCHEMA_VERSION = "quasi-real-guarded-formal-ppo-preflight-seed-summary/v1"
EXPECTED_READINESS_STATUS = "quasi_real_guarded_formal_ppo_preflight_evaluated"

SUMMARY_FILE = "quasi-real-guarded-formal-ppo-preflight-summary.json"
SEED_SUMMARIES_FILE = "formal-preflight-seed-summaries.jsonl"
TRAINING_CURVES_FILE = "formal-preflight-training-curves.json"
GATE_AUDIT_FILE = "formal-preflight-gate-audit.json"
ROLLBACK_MANIFEST_FILE = "formal-preflight-rollback-manifest.json"
READINESS_FILE = "formal-preflight-readiness-validate-only.json"
REPORT_FILE = "formal-preflight-report.md"

FREEZE_SUMMARY_FILE = "quasi-real-guarded-ppo-iterative-miniloop-evidence-freeze-summary.json"

SeedPreflightRunner = Callable[..., dict[str, Any]]
ReadinessRunner = Callable[..., dict[str, Any]]


def run_quasi_real_guarded_formal_ppo_preflight(
    *,
    freeze_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    seed_preflight_runner: SeedPreflightRunner | None = None,
    readiness_runner: ReadinessRunner | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    freeze_root = Path(freeze_root)
    output_root = Path(output_root)
    batch_root = Path(batch_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = _output_files(config)
    summary_path = output_root / files["summary"]
    seed_summaries_path = output_root / files["seed_summaries"]
    training_curves_path = output_root / files["training_curves"]
    gate_audit_path = output_root / files["gate_audit"]
    rollback_manifest_path = output_root / files["rollback_manifest"]
    readiness_path = output_root / files["readiness_validate_only"]
    report_path = output_root / files["report"]

    freeze_summary_path = freeze_root / FREEZE_SUMMARY_FILE
    freeze_summary = _read_json_if_exists(freeze_summary_path)
    manifest_path = _resolve_path(
        Path(str(freeze_summary.get("manifest") or freeze_root / "quasi-real-guarded-ppo-iterative-miniloop-evidence-manifest.json")),
        repo_root,
    )
    manifest = _read_json_if_exists(manifest_path)
    miniloop_summary_path = _resolve_miniloop_summary_path(
        freeze_summary=freeze_summary,
        manifest=manifest,
        repo_root=repo_root,
    )
    miniloop_summary = _read_json_if_exists(miniloop_summary_path)
    steps_path = _resolve_steps_path(miniloop_summary, miniloop_summary_path, repo_root)
    steps = _read_jsonl(steps_path)
    trainable_steps = _unique_trainable_steps(steps)
    counters = _input_counters(steps, trainable_steps)

    reason_codes: list[str] = []
    _validate_frozen_inputs(
        freeze_summary=freeze_summary,
        manifest=manifest,
        miniloop_summary=miniloop_summary,
        counters=counters,
        config=config,
        reason_codes=reason_codes,
    )

    seed_summaries: list[dict[str, Any]] = []
    if not reason_codes:
        runner = seed_preflight_runner or _run_seed_preflight
        for seed in _seeds(config):
            seed_summary = runner(
                seed=seed,
                trainable_steps=trainable_steps,
                output_root=output_root,
                config=config,
                repo_root=repo_root,
                batch_root=batch_root,
            )
            seed_summaries.append(_normalize_seed_summary(seed_summary, seed=seed))
        _validate_seed_summaries(seed_summaries, counters, config, reason_codes)

    _write_jsonl(seed_summaries_path, seed_summaries)
    training_curves = _training_curves(seed_summaries)
    gate_audit = _gate_audit(counters, seed_summaries)
    rollback_manifest = _rollback_manifest(seed_summaries, output_root=output_root)
    _write_json(training_curves_path, training_curves)
    _write_json(gate_audit_path, gate_audit)
    _write_json(rollback_manifest_path, rollback_manifest)

    status_without_readiness = "passed" if not reason_codes else "failed"
    pre_readiness_summary = _summary_payload(
        status=status_without_readiness,
        reason_codes=reason_codes,
        repo_root=repo_root,
        freeze_root=freeze_root,
        freeze_summary_path=freeze_summary_path,
        manifest_path=manifest_path,
        miniloop_summary_path=miniloop_summary_path,
        steps_path=steps_path,
        output_root=output_root,
        batch_root=batch_root,
        summary_path=summary_path,
        seed_summaries_path=seed_summaries_path,
        training_curves_path=training_curves_path,
        gate_audit_path=gate_audit_path,
        rollback_manifest_path=rollback_manifest_path,
        readiness_path=readiness_path,
        report_path=report_path,
        counters=counters,
        seed_summaries=seed_summaries,
        readiness={},
    )
    _write_json(summary_path, pre_readiness_summary)

    readiness: dict[str, Any]
    if status_without_readiness == "passed":
        runner = readiness_runner or _run_readiness_validate_only
        readiness = runner(
            repo_root=repo_root,
            batch_root=batch_root,
            formal_preflight_summary_path=summary_path,
            config_path=Path(
                config.get("readiness", {}).get(
                    "config", "configs/policy_training_readiness_review_v1.json"
                )
            ),
        )
        _validate_readiness(readiness, config, reason_codes)
    else:
        readiness = {
            "training_readiness_status": "needs_training_contract_refinement",
            "training_blockers": list(reason_codes),
            "reason_codes": list(reason_codes),
            "recommended_next_action": "fix_quasi_real_guarded_formal_ppo_preflight",
        }
    _write_json(readiness_path, readiness)

    final_status = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=final_status,
        reason_codes=reason_codes,
        repo_root=repo_root,
        freeze_root=freeze_root,
        freeze_summary_path=freeze_summary_path,
        manifest_path=manifest_path,
        miniloop_summary_path=miniloop_summary_path,
        steps_path=steps_path,
        output_root=output_root,
        batch_root=batch_root,
        summary_path=summary_path,
        seed_summaries_path=seed_summaries_path,
        training_curves_path=training_curves_path,
        gate_audit_path=gate_audit_path,
        rollback_manifest_path=rollback_manifest_path,
        readiness_path=readiness_path,
        report_path=report_path,
        counters=counters,
        seed_summaries=seed_summaries,
        readiness=readiness,
    )
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run quasi-real guarded formal PPO preflight."
    )
    parser.add_argument(
        "--freeze-root",
        default="outputs/path_feedback_batch_quasi_real_guarded_ppo_iterative_miniloop_evidence_freeze_v1",
    )
    parser.add_argument(
        "--batch-root",
        default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_preflight_v1",
    )
    parser.add_argument(
        "--config",
        default="configs/quasi_real_guarded_formal_ppo_preflight_v1.json",
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

    summary = run_quasi_real_guarded_formal_ppo_preflight(
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
                "input_trainable_transition_count": summary["input_trainable_transition_count"],
                "optimizer_train_transition_count": summary["optimizer_train_transition_count"],
                "seed_count": summary["seed_count"],
                "passed_seed_count": summary["passed_seed_count"],
                "readiness_status": summary["readiness_status"],
                "controlled_regression_count": summary["controlled_regression_count"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def _run_seed_preflight(
    *,
    seed: int,
    trainable_steps: list[dict[str, Any]],
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    batch_root: Path,
) -> dict[str, Any]:
    seed_summary = _run_seed_smoke(
        seed=seed,
        trainable_steps=trainable_steps,
        output_root=output_root,
        config=config,
        repo_root=repo_root,
        batch_root=batch_root,
    )
    formal = dict(seed_summary)
    formal["schema_version"] = SEED_SCHEMA_VERSION
    formal["updated_candidate_root"] = seed_summary.get(
        "updated_candidate_root", seed_summary.get("limited_ppo_update_smoke_root")
    )
    formal["training_curve_records"] = _seed_training_curve_records(formal)
    return formal


def _run_readiness_validate_only(
    *,
    repo_root: Path,
    batch_root: Path,
    formal_preflight_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        "bash",
        str(repo_root / "scripts" / "run_policy_training_readiness_review.sh"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(_resolve_path(config_path, repo_root)),
        "--quasi-real-guarded-formal-ppo-preflight-summary",
        str(formal_preflight_summary_path),
        "--validate-only",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
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


def _validate_frozen_inputs(
    *,
    freeze_summary: dict[str, Any],
    manifest: dict[str, Any],
    miniloop_summary: dict[str, Any],
    counters: Counter[str],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if freeze_summary.get("status") != "passed" or _string_list(freeze_summary.get("reason_codes")):
        _add_reason(reason_codes, "input_freeze_summary_not_passed")
    if freeze_summary.get("baseline_frozen") is not True:
        _add_reason(reason_codes, "input_freeze_baseline_not_frozen")
    if _int(freeze_summary.get("required_artifact_missing_count")):
        _add_reason(reason_codes, "input_freeze_required_artifact_missing")
    if _int(manifest.get("required_artifact_missing_count")):
        _add_reason(reason_codes, "input_freeze_manifest_required_artifact_missing")
    if miniloop_summary.get("status") != "passed" or _string_list(miniloop_summary.get("reason_codes")):
        _add_reason(reason_codes, "input_miniloop_summary_not_passed")
    if miniloop_summary.get("readiness_status") not in (
        None,
        "quasi_real_guarded_ppo_iterative_miniloop_stability_evaluated",
    ):
        _add_reason(reason_codes, "input_miniloop_readiness_invalid")

    expected = _int(config.get("validation", {}).get("expected_trainable_transition_count"), 684)
    if counters["input_trainable_transition_count"] != expected:
        _add_reason(reason_codes, "formal_preflight_trainable_transition_count_mismatch")
    if counters["unique_trainable_context_count"] != expected:
        _add_reason(reason_codes, "formal_preflight_unique_context_count_mismatch")
    for field, reason in (
        ("validation_trainable_count", "formal_preflight_split_leakage"),
        ("test_trainable_count", "formal_preflight_split_leakage"),
        ("fallback_trainable_count", "formal_preflight_fallback_trainable"),
        ("non_empty_gate_reason_trainable_count", "formal_preflight_gate_reason_trainable"),
        ("missing_observation_count", "formal_preflight_contract_invalid"),
        ("missing_log_prob_count", "formal_preflight_contract_invalid"),
        ("missing_value_count", "formal_preflight_contract_invalid"),
        ("non_finite_reward_count", "formal_preflight_non_finite"),
        ("non_finite_return_count", "formal_preflight_non_finite"),
        ("non_finite_advantage_count", "formal_preflight_non_finite"),
        ("controlled_regression_count", "formal_preflight_controlled_regression"),
        ("controlled_safety_regression_count", "formal_preflight_controlled_regression"),
        ("controlled_contract_regression_count", "formal_preflight_controlled_regression"),
        ("controlled_path_risk_regression_count", "formal_preflight_controlled_regression"),
        ("controlled_source_selection_regression_count", "formal_preflight_controlled_regression"),
    ):
        if counters[field]:
            _add_reason(reason_codes, reason)


def _validate_seed_summaries(
    seed_summaries: list[dict[str, Any]],
    counters: Counter[str],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if len(seed_summaries) != len(_seeds(config)):
        _add_reason(reason_codes, "formal_preflight_seed_not_all_passed")
    validation = config.get("validation", {})
    expected = counters["input_trainable_transition_count"]
    for summary in seed_summaries:
        if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
            _add_reason(reason_codes, "formal_preflight_seed_not_all_passed")
        if _int(summary.get("optimizer_train_transition_count")) != expected:
            _add_reason(reason_codes, "formal_preflight_optimizer_train_count_mismatch")
        if _float(summary.get("old_log_prob_max_abs_error"), math.inf) > _float(
            validation.get("max_old_log_prob_abs_error"), 1.0e-4
        ):
            _add_reason(reason_codes, "formal_preflight_old_policy_reconstruction_error")
        if _float(summary.get("old_value_max_abs_error"), math.inf) > _float(
            validation.get("max_old_value_abs_error"), 1.0e-4
        ):
            _add_reason(reason_codes, "formal_preflight_old_policy_reconstruction_error")
        if abs(_float(summary.get("approx_kl"), math.inf)) > _float(validation.get("max_abs_approx_kl"), 0.25):
            _add_reason(reason_codes, "formal_preflight_seed_kl_too_large")
        if _float(summary.get("max_grad_norm_after_clip"), math.inf) > _float(
            validation.get("max_grad_norm_after_clip"), 1.0
        ) + 1.0e-8:
            _add_reason(reason_codes, "formal_preflight_seed_grad_norm_too_large")
        if _float(summary.get("parameter_l2_delta"), 0.0) <= 0.0:
            _add_reason(reason_codes, "formal_preflight_seed_parameter_delta_missing")
        if _float(summary.get("teacher_agreement_rate"), 0.0) < _float(
            validation.get("min_teacher_agreement_rate"), 0.95
        ):
            _add_reason(reason_codes, "formal_preflight_teacher_alignment_insufficient")
        for field, reason in (
            ("loss_non_finite_count", "formal_preflight_non_finite"),
            ("non_finite_gradient_count", "formal_preflight_non_finite"),
            ("non_finite_reward_count", "formal_preflight_non_finite"),
            ("non_finite_return_count", "formal_preflight_non_finite"),
            ("non_finite_advantage_count", "formal_preflight_non_finite"),
            ("controlled_regression_count", "formal_preflight_controlled_regression"),
            ("controlled_safety_regression_count", "formal_preflight_controlled_regression"),
            ("controlled_contract_regression_count", "formal_preflight_controlled_regression"),
            ("controlled_path_risk_regression_count", "formal_preflight_controlled_regression"),
            ("controlled_source_selection_regression_count", "formal_preflight_controlled_regression"),
        ):
            if _int(summary.get(field)):
                _add_reason(reason_codes, reason)
        for field, reason in (
            ("publishes_checkpoint", "formal_preflight_checkpoint_publication_claimed"),
            ("replaces_default_policy", "formal_preflight_default_policy_replacement_claimed"),
            ("performance_claimed", "formal_preflight_policy_performance_claimed"),
            ("formal_training_ready_claimed", "formal_preflight_formal_training_ready_claimed"),
        ):
            if summary.get(field) is True:
                _add_reason(reason_codes, reason)


def _validate_readiness(
    readiness: dict[str, Any], config: dict[str, Any], reason_codes: list[str]
) -> None:
    expected = str(config.get("readiness", {}).get("expected_status") or EXPECTED_READINESS_STATUS)
    if readiness.get("training_readiness_status") != expected:
        _add_reason(reason_codes, "readiness_not_quasi_real_guarded_formal_ppo_preflight_evaluated")
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
    freeze_root: Path,
    freeze_summary_path: Path,
    manifest_path: Path,
    miniloop_summary_path: Path,
    steps_path: Path,
    output_root: Path,
    batch_root: Path,
    summary_path: Path,
    seed_summaries_path: Path,
    training_curves_path: Path,
    gate_audit_path: Path,
    rollback_manifest_path: Path,
    readiness_path: Path,
    report_path: Path,
    counters: Counter[str],
    seed_summaries: list[dict[str, Any]],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    metrics = _seed_metrics(seed_summaries)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": None if status == "passed" else "fix_quasi_real_guarded_formal_ppo_preflight",
        "freeze_root": str(freeze_root),
        "freeze_summary": str(freeze_summary_path),
        "freeze_manifest": str(manifest_path),
        "miniloop_summary": str(miniloop_summary_path),
        "steps": str(steps_path),
        "batch_root": str(batch_root),
        "output_root": str(output_root),
        "summary": str(summary_path),
        "seed_summaries": str(seed_summaries_path),
        "training_curves": str(training_curves_path),
        "gate_audit": str(gate_audit_path),
        "rollback_manifest": str(rollback_manifest_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "input_trainable_transition_count": counters["input_trainable_transition_count"],
        "optimizer_train_transition_count": metrics["optimizer_train_transition_count"],
        "unique_trainable_context_count": counters["unique_trainable_context_count"],
        "step_count": counters["step_count"],
        "validation_trainable_count": counters["validation_trainable_count"],
        "test_trainable_count": counters["test_trainable_count"],
        "fallback_trainable_count": counters["fallback_trainable_count"],
        "source_fallback_trainable_count": counters["source_fallback_trainable_count"],
        "teacher_fallback_trainable_count": counters["teacher_fallback_trainable_count"],
        "non_empty_gate_reason_trainable_count": counters["non_empty_gate_reason_trainable_count"],
        "missing_observation_count": counters["missing_observation_count"],
        "missing_log_prob_count": counters["missing_log_prob_count"],
        "missing_value_count": counters["missing_value_count"],
        "non_finite_reward_count": counters["non_finite_reward_count"],
        "non_finite_return_count": counters["non_finite_return_count"],
        "non_finite_advantage_count": counters["non_finite_advantage_count"],
        **metrics,
        "readiness_status": readiness.get("training_readiness_status"),
        "training_blockers": list(readiness.get("training_blockers") or []),
        "readiness_reason_codes": list(readiness.get("reason_codes") or []),
        "runs_formal_ppo_rollout": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _seed_metrics(seed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not seed_summaries:
        return {
            "passed_seed_count": 0,
            "seed_failure_count": 0,
            "optimizer_train_transition_count": 0,
            "max_old_log_prob_abs_error": 0.0,
            "max_old_value_abs_error": 0.0,
            "loss_non_finite_count": 0,
            "non_finite_gradient_count": 0,
            "non_finite_reward_count": 0,
            "non_finite_return_count": 0,
            "non_finite_advantage_count": 0,
            "max_abs_approx_kl": 0.0,
            "max_grad_norm_after_clip": 0.0,
            "min_parameter_l2_delta": 0.0,
            "teacher_agreement_rate": 0.0,
            "controlled_regression_count": 0,
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0,
            "controlled_source_selection_regression_count": 0,
        }
    return {
        "seed_count": len(seed_summaries),
        "passed_seed_count": sum(1 for item in seed_summaries if item.get("status") == "passed" and not item.get("reason_codes")),
        "seed_failure_count": sum(1 for item in seed_summaries if item.get("status") != "passed" or item.get("reason_codes")),
        "optimizer_train_transition_count": min(_int(item.get("optimizer_train_transition_count")) for item in seed_summaries),
        "max_old_log_prob_abs_error": max(_float(item.get("old_log_prob_max_abs_error")) for item in seed_summaries),
        "max_old_value_abs_error": max(_float(item.get("old_value_max_abs_error")) for item in seed_summaries),
        "loss_non_finite_count": sum(_int(item.get("loss_non_finite_count")) for item in seed_summaries),
        "non_finite_gradient_count": sum(_int(item.get("non_finite_gradient_count")) for item in seed_summaries),
        "non_finite_reward_count": sum(_int(item.get("non_finite_reward_count")) for item in seed_summaries),
        "non_finite_return_count": sum(_int(item.get("non_finite_return_count")) for item in seed_summaries),
        "non_finite_advantage_count": sum(_int(item.get("non_finite_advantage_count")) for item in seed_summaries),
        "max_abs_approx_kl": max(abs(_float(item.get("approx_kl"))) for item in seed_summaries),
        "max_grad_norm_after_clip": max(_float(item.get("max_grad_norm_after_clip")) for item in seed_summaries),
        "min_parameter_l2_delta": min(_float(item.get("parameter_l2_delta")) for item in seed_summaries),
        "teacher_agreement_rate": min(_float(item.get("teacher_agreement_rate")) for item in seed_summaries),
        "controlled_regression_count": sum(_int(item.get("controlled_regression_count")) for item in seed_summaries),
        "controlled_safety_regression_count": sum(_int(item.get("controlled_safety_regression_count")) for item in seed_summaries),
        "controlled_contract_regression_count": sum(_int(item.get("controlled_contract_regression_count")) for item in seed_summaries),
        "controlled_path_risk_regression_count": sum(_int(item.get("controlled_path_risk_regression_count")) for item in seed_summaries),
        "controlled_source_selection_regression_count": sum(_int(item.get("controlled_source_selection_regression_count")) for item in seed_summaries),
    }


def _input_counters(steps: list[dict[str, Any]], trainable_steps: list[dict[str, Any]]) -> Counter[str]:
    counters: Counter[str] = Counter()
    counters["step_count"] = len(steps)
    counters["input_trainable_transition_count"] = len(trainable_steps)
    counters["unique_trainable_context_count"] = len({step.get("context_id") for step in trainable_steps})
    for step in steps:
        if step.get("ppo_trainable") is True and step.get("split") == "validation":
            counters["validation_trainable_count"] += 1
        if step.get("ppo_trainable") is True and step.get("split") == "test":
            counters["test_trainable_count"] += 1
        if step.get("ppo_trainable") is True and str(step.get("controlled_choice_source")) in {
            "source_fallback",
            "teacher_fallback",
        }:
            counters["fallback_trainable_count"] += 1
        if step.get("ppo_trainable") is True and step.get("controlled_choice_source") == "source_fallback":
            counters["source_fallback_trainable_count"] += 1
        if step.get("ppo_trainable") is True and step.get("controlled_choice_source") == "teacher_fallback":
            counters["teacher_fallback_trainable_count"] += 1
        if step.get("ppo_trainable") is True and _string_list(step.get("gate_reason_codes")):
            counters["non_empty_gate_reason_trainable_count"] += 1
        if step.get("ppo_trainable") is True and (step.get("observation") is None or step.get("missing_observation") is True):
            counters["missing_observation_count"] += 1
        if step.get("ppo_trainable") is True and not _finite(step.get("log_prob")):
            counters["missing_log_prob_count"] += 1
        if step.get("ppo_trainable") is True and not _finite(step.get("value")):
            counters["missing_value_count"] += 1
        if step.get("ppo_trainable") is True and not _finite(step.get("reward")):
            counters["non_finite_reward_count"] += 1
        if step.get("ppo_trainable") is True and not _finite(step.get("discounted_return")):
            counters["non_finite_return_count"] += 1
        if step.get("ppo_trainable") is True and not _finite(step.get("advantage")):
            counters["non_finite_advantage_count"] += 1
        reasons = set(_string_list(step.get("controlled_regression_reason_codes")))
        if reasons:
            counters["controlled_regression_count"] += 1
        if "safety_regression" in reasons:
            counters["controlled_safety_regression_count"] += 1
        if "contract_violation" in reasons or "contract_regression" in reasons:
            counters["controlled_contract_regression_count"] += 1
        if "path_cost_regression" in reasons or "risk_regression" in reasons:
            counters["controlled_path_risk_regression_count"] += 1
        if "source_selection_regression" in reasons:
            counters["controlled_source_selection_regression_count"] += 1
    return counters


def _training_curves(seed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for summary in seed_summaries:
        seed_records = summary.get("training_curve_records")
        if isinstance(seed_records, list):
            records.extend(row for row in seed_records if isinstance(row, dict))
        else:
            records.append(
                {
                    "seed": _int(summary.get("seed")),
                    "approx_kl": _float(summary.get("approx_kl")),
                    "max_grad_norm_after_clip": _float(summary.get("max_grad_norm_after_clip")),
                    "optimizer_train_transition_count": _int(summary.get("optimizer_train_transition_count")),
                }
            )
    return {
        "schema_version": "quasi-real-guarded-formal-ppo-preflight-training-curves/v1",
        "records": records,
    }


def _gate_audit(counters: Counter[str], seed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "quasi-real-guarded-formal-ppo-preflight-gate-audit/v1",
        "validation_trainable_count": counters["validation_trainable_count"],
        "test_trainable_count": counters["test_trainable_count"],
        "fallback_trainable_count": counters["fallback_trainable_count"],
        "non_empty_gate_reason_trainable_count": counters["non_empty_gate_reason_trainable_count"],
        "controlled_regression_count": sum(_int(item.get("controlled_regression_count")) for item in seed_summaries),
        "teacher_agreement_rate": min(
            [_float(item.get("teacher_agreement_rate")) for item in seed_summaries] or [0.0]
        ),
    }


def _rollback_manifest(seed_summaries: list[dict[str, Any]], *, output_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "quasi-real-guarded-formal-ppo-preflight-rollback-manifest/v1",
        "output_root": str(output_root),
        "experimental_candidate_roots": [
            str(item.get("updated_candidate_root"))
            for item in seed_summaries
            if item.get("updated_candidate_root")
        ],
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
    }


def _normalize_seed_summary(summary: dict[str, Any], *, seed: int) -> dict[str, Any]:
    row = dict(summary)
    row["schema_version"] = SEED_SCHEMA_VERSION
    row["seed"] = int(seed)
    for field in (
        "publishes_checkpoint",
        "replaces_default_policy",
        "performance_claimed",
        "formal_training_ready_claimed",
    ):
        row[field] = bool(row.get(field, False))
    return row


def _seed_training_curve_records(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "seed": _int(summary.get("seed")),
            "optimizer_train_transition_count": _int(summary.get("optimizer_train_transition_count")),
            "approx_kl": _float(summary.get("approx_kl")),
            "max_grad_norm_after_clip": _float(summary.get("max_grad_norm_after_clip")),
        }
    ]


def _resolve_miniloop_summary_path(
    *, freeze_summary: dict[str, Any], manifest: dict[str, Any], repo_root: Path
) -> Path:
    if freeze_summary.get("miniloop_summary"):
        return _resolve_path(Path(str(freeze_summary["miniloop_summary"])), repo_root)
    for item in manifest.get("artifacts", []):
        if isinstance(item, dict) and item.get("name") == "miniloop_summary":
            return _resolve_path(Path(str(item.get("path"))), repo_root)
    return repo_root / "missing-miniloop-summary.json"


def _resolve_steps_path(summary: dict[str, Any], summary_path: Path, repo_root: Path) -> Path:
    value = summary.get("steps")
    if not value:
        return summary_path.parent / "quasi-real-trainable-context-expansion-steps.jsonl"
    path = Path(str(value))
    if path.is_absolute():
        return path
    local = summary_path.parent / path
    return local if local.is_file() else repo_root / path


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "summary": SUMMARY_FILE,
        "seed_summaries": SEED_SUMMARIES_FILE,
        "training_curves": TRAINING_CURVES_FILE,
        "gate_audit": GATE_AUDIT_FILE,
        "rollback_manifest": ROLLBACK_MANIFEST_FILE,
        "readiness_validate_only": READINESS_FILE,
        "report": REPORT_FILE,
    }
    configured = config.get("output_files") if isinstance(config.get("output_files"), dict) else {}
    return {key: str(configured.get(key) or default) for key, default in defaults.items()}


def _seeds(config: dict[str, Any]) -> list[int]:
    seeds = config.get("seeds")
    return [int(seed) for seed in seeds] if isinstance(seeds, list) and seeds else [0, 1, 2]


def _render_report(summary: dict[str, Any]) -> str:
    return (
        "# Quasi-Real Guarded Formal PPO Preflight\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Reason codes: `{summary.get('reason_codes')}`\n"
        f"- Trainable contexts: `{summary.get('input_trainable_transition_count')}`\n"
        f"- Optimizer train transitions: `{summary.get('optimizer_train_transition_count')}`\n"
        f"- Seeds passed: `{summary.get('passed_seed_count')}` / `{summary.get('seed_count')}`\n"
        f"- Readiness: `{summary.get('readiness_status')}`\n\n"
        "This is formal PPO preflight only. It does not publish checkpoints, "
        "replace the default policy, or claim formal training readiness.\n"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [value] if value else []
    return [str(value)]


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
