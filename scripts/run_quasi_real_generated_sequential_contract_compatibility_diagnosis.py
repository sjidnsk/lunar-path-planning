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

from git_provenance import git_snapshot as _git_snapshot


CONFIG_SCHEMA_VERSION = "quasi-real-generated-sequential-contract-compatibility-diagnosis-config/v1"
SUMMARY_SCHEMA_VERSION = "quasi-real-generated-sequential-contract-compatibility-summary/v1"

NEXT_MISSING_INPUT = "quasi_real_generated_sequential_diagnosis_input_missing"

ReplayRunner = Callable[[dict[str, Any]], dict[str, Any]]


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose quasi-real vs generated sequential contract compatibility."
    )
    parser.add_argument("--update-smoke-root", required=True)
    parser.add_argument("--base-candidate-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
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

    summary = run_quasi_real_generated_sequential_contract_compatibility_diagnosis(
        update_smoke_root=_resolve_path(args.update_smoke_root, repo_root),
        base_candidate_root=_resolve_path(args.base_candidate_root, repo_root),
        source_root=_resolve_path(args.source_root, repo_root),
        output_root=_resolve_path(args.output_root, repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "diagnosis_verdict": summary["diagnosis_verdict"],
                "failed_step_count": summary["failed_step_count"],
                "base_generated_sequential_status": summary["base_generated_sequential_status"],
                "updated_generated_sequential_status": summary["updated_generated_sequential_status"],
                "recommended_next_action": summary["recommended_next_action"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def run_quasi_real_generated_sequential_contract_compatibility_diagnosis(
    *,
    update_smoke_root: Path,
    base_candidate_root: Path,
    output_root: Path,
    source_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    replay_runner: ReplayRunner | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root, config)
    inputs = _input_paths(update_smoke_root, base_candidate_root, config)
    reason_codes: list[str] = []

    update_smoke_summary = _load_json_required(inputs["update_smoke_summary"], reason_codes, "update_smoke_summary")
    original_updated_summary = _load_json_required(
        inputs["generated_sequential_summary"],
        reason_codes,
        "generated_sequential_summary",
    )
    original_rejection_report = _load_json_required(
        inputs["generated_sequential_rejection_report"],
        reason_codes,
        "generated_sequential_rejection_report",
    )
    original_steps = _read_jsonl_required(inputs["generated_sequential_steps"], reason_codes, "generated_sequential_steps")
    quasi_teacher = _load_json_required(
        inputs["quasi_real_teacher_following_summary"],
        reason_codes,
        "quasi_real_teacher_following_summary",
    )
    quasi_collector = _load_json_required(
        inputs["quasi_real_collector_summary"],
        reason_codes,
        "quasi_real_collector_summary",
    )

    diagnostic_base_root = output_root / config.get("replay", {}).get(
        "diagnostic_base_candidate_root",
        "diagnostic-base-candidate",
    )
    clone_ok = _write_diagnostic_base_candidate(
        base_candidate_root=base_candidate_root,
        output_root=diagnostic_base_root,
        repo_root=repo_root,
        config=config,
        reason_codes=reason_codes,
    )

    replay = _run_or_reuse_replays(
        update_smoke_root=update_smoke_root,
        diagnostic_base_root=diagnostic_base_root,
        output_root=output_root,
        source_root=source_root,
        config=config,
        repo_root=repo_root,
        replay_runner=replay_runner,
    )
    base_root = Path(replay["base_root"])
    updated_root = Path(replay["updated_root"])
    base_summary = replay.get("base_summary") or _load_json(base_root / "policy-gated-sequential-canary-rollout-summary.json")
    updated_summary = replay.get("updated_summary") or _load_json(
        updated_root / "policy-gated-sequential-canary-rollout-summary.json"
    )
    base_failed_steps = _failed_steps_from_root(base_root)
    updated_failed_steps = _failed_steps_from_root(updated_root)
    if not updated_failed_steps and original_rejection_report:
        updated_failed_steps = _failed_steps(original_rejection_report, original_steps)

    comparison_rows = _comparison_rows(base_failed_steps, updated_failed_steps)
    _write_jsonl(paths["failed_step_comparison"], comparison_rows)
    gate_metric_mismatches = _gate_metric_mismatches(updated_failed_steps, config=config)
    verdict = _diagnosis_verdict(
        reason_codes=reason_codes,
        clone_ok=clone_ok,
        base_summary=base_summary,
        updated_summary=updated_summary,
        base_failed_steps=base_failed_steps,
        updated_failed_steps=updated_failed_steps,
        gate_metric_mismatch_count=len(gate_metric_mismatches),
    )
    recommended_next_action = _recommended_next_action(verdict)
    status = "failed" if reason_codes else "passed"

    baseline_vs_updated = {
        "schema_version": "baseline-vs-updated-generated-sequential-summary/v1",
        "base_generated_sequential_status": base_summary.get("status"),
        "updated_generated_sequential_status": updated_summary.get("status"),
        "base_failed_step_count": len(base_failed_steps),
        "updated_failed_step_count": len(updated_failed_steps),
        "matching_failed_step_count": sum(1 for row in comparison_rows if row["base_failed"] and row["updated_failed"]),
        "updated_only_failed_step_count": sum(1 for row in comparison_rows if row["updated_failed"] and not row["base_failed"]),
        "base_only_failed_step_count": sum(1 for row in comparison_rows if row["base_failed"] and not row["updated_failed"]),
    }
    _write_json(paths["baseline_vs_updated_summary"], baseline_vs_updated)

    reason_counts = Counter(
        reason
        for step in updated_failed_steps
        for reason in _string_list(step.get("canary_rejection_reason_codes"))
    )
    family_counts = Counter(str(step.get("scenario_group") or "unknown") for step in updated_failed_steps)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "diagnosis_verdict": verdict,
        "recommended_next_action": recommended_next_action,
        "update_smoke_root": _display_path(update_smoke_root, repo_root),
        "update_smoke_summary": _display_path(inputs["update_smoke_summary"], repo_root),
        "post_update_generated_sequential_summary": _display_path(inputs["generated_sequential_summary"], repo_root),
        "post_update_generated_sequential_steps": _display_path(inputs["generated_sequential_steps"], repo_root),
        "post_update_generated_sequential_rejection_report": _display_path(
            inputs["generated_sequential_rejection_report"],
            repo_root,
        ),
        "post_update_quasi_real_teacher_following_summary": _display_path(
            inputs["quasi_real_teacher_following_summary"],
            repo_root,
        ),
        "post_update_quasi_real_collector_summary": _display_path(
            inputs["quasi_real_collector_summary"],
            repo_root,
        ),
        "base_candidate_root": _display_path(base_candidate_root, repo_root),
        "source_root": _display_path(source_root, repo_root),
        "output_root": _display_path(output_root, repo_root),
        "diagnostic_base_candidate_root": str(diagnostic_base_root),
        "diagnostic_base_candidate_clone_ok": clone_ok,
        "failed_step_count": len(updated_failed_steps),
        "base_failed_step_count": len(base_failed_steps),
        "updated_failed_step_count": len(updated_failed_steps),
        "gate_metric_mismatch_count": len(gate_metric_mismatches),
        "failed_family_counts": dict(sorted(family_counts.items())),
        "canary_rejection_reason_counts": dict(sorted(reason_counts.items())),
        "base_generated_sequential_status": base_summary.get("status"),
        "base_generated_sequential_reason_codes": _string_list(base_summary.get("reason_codes")),
        "base_generated_sequential_root": _display_path(base_root, repo_root),
        "updated_generated_sequential_status": updated_summary.get("status"),
        "updated_generated_sequential_reason_codes": _string_list(updated_summary.get("reason_codes")),
        "updated_generated_sequential_root": _display_path(updated_root, repo_root),
        "original_updated_generated_sequential_status": original_updated_summary.get("status"),
        "update_smoke_status": update_smoke_summary.get("status"),
        "quasi_real_teacher_following_status": quasi_teacher.get("status"),
        "quasi_real_teacher_agreement_rate": quasi_teacher.get("teacher_agreement_rate"),
        "quasi_real_unsafe_disagreement_count": quasi_teacher.get("unsafe_disagreement_count"),
        "quasi_real_collector_status": quasi_collector.get("status"),
        "quasi_real_collector_ppo_trainable_transition_count": quasi_collector.get("ppo_trainable_transition_count"),
        "quasi_real_collector_diagnostic_transition_count": quasi_collector.get("diagnostic_transition_count"),
        "failed_step_comparison": _display_path(paths["failed_step_comparison"], repo_root),
        "baseline_vs_updated_summary": _display_path(paths["baseline_vs_updated_summary"], repo_root),
        "report": _display_path(paths["report"], repo_root),
        "summary": _display_path(paths["summary"], repo_root),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "runs_ppo_update": False,
        "runs_iterative_ppo": False,
        "git_provenance": {"current": _git_snapshot(repo_root), "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }
    _write_report(paths["report"], summary=summary, comparison_rows=comparison_rows, gate_metric_mismatches=gate_metric_mismatches)
    _write_json(paths["summary"], summary)
    return summary


def _write_diagnostic_base_candidate(
    *,
    base_candidate_root: Path,
    output_root: Path,
    repo_root: Path,
    config: dict[str, Any],
    reason_codes: list[str],
) -> bool:
    output_root.mkdir(parents=True, exist_ok=True)
    inputs = config["input_files"]
    try:
        import torch

        checkpoint_path = base_candidate_root / inputs.get("base_checkpoint", "experimental-hybrid-policy-candidate.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint root is not an object")
        _mark_diagnostic_clone(checkpoint, repo_root)
        torch.save(checkpoint, output_root / "experimental-hybrid-policy-candidate.pt")
    except Exception as exc:  # noqa: BLE001 - summary records diagnostic clone failure.
        _append_reason(reason_codes, "diagnostic_base_candidate_clone_failed")
        (output_root / "clone-error.txt").write_text(str(exc), encoding="utf-8")
        return False

    for filename in (
        inputs.get("base_checkpoint_metadata", "experimental-hybrid-policy-candidate-metadata.json"),
        inputs.get("base_candidate_summary", "raw-policy-generalization-candidate-summary.json"),
    ):
        source = base_candidate_root / filename
        payload = _load_json(source)
        if not payload:
            _append_reason(reason_codes, "diagnostic_base_candidate_metadata_missing")
            continue
        _mark_diagnostic_clone(payload, repo_root)
        _write_json(output_root / filename, payload)
    return True


def _mark_diagnostic_clone(payload: dict[str, Any], repo_root: Path) -> None:
    payload["diagnostic_clone"] = True
    payload["experimental"] = True
    payload["experimental_checkpoint"] = True
    payload["publishes_checkpoint"] = False
    payload["replaces_default_policy"] = False
    payload["performance_claimed"] = False
    payload["formal_training_ready_claimed"] = False
    payload["git_provenance"] = {"current": _git_snapshot(repo_root), "current_matches_sources": True}


def _run_or_reuse_replays(
    *,
    update_smoke_root: Path,
    diagnostic_base_root: Path,
    output_root: Path,
    source_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    replay_runner: ReplayRunner | None,
) -> dict[str, Any]:
    replay = config.get("replay", {})
    base_root = output_root / replay.get("base_replay_root", "base_generated_sequential")
    updated_root = output_root / replay.get("updated_replay_root", "updated_generated_sequential_replay")
    context = {
        "update_smoke_root": update_smoke_root,
        "diagnostic_base_candidate_root": diagnostic_base_root,
        "base_replay_root": base_root,
        "updated_replay_root": updated_root,
        "source_root": source_root,
        "config": config,
        "repo_root": repo_root,
    }
    if replay_runner is not None:
        result = replay_runner(context)
        result.setdefault("base_root", base_root)
        result.setdefault("updated_root", updated_root)
        return result

    _run_generated_sequential(
        source_root=source_root,
        candidate_root=diagnostic_base_root,
        batch_root=base_root,
        config=config,
        repo_root=repo_root,
    )
    existing_updated_root = update_smoke_root / "post_update_generated_sequential"
    if replay.get("reuse_existing_updated_replay", True) and existing_updated_root.is_dir():
        if updated_root.exists():
            shutil.rmtree(updated_root)
        shutil.copytree(existing_updated_root, updated_root)
    else:
        _run_generated_sequential(
            source_root=source_root,
            candidate_root=update_smoke_root,
            batch_root=updated_root,
            config=config,
            repo_root=repo_root,
        )
    return {"base_root": base_root, "updated_root": updated_root}


def _run_generated_sequential(
    *,
    source_root: Path,
    candidate_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> None:
    if batch_root.exists():
        shutil.rmtree(batch_root)
    sequential_config = _resolve_path(
        config.get("replay", {}).get(
            "generated_sequential_config",
            "configs/policy_gated_sequential_multi_step_opportunity_rollout_v1.json",
        ),
        repo_root,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_policy_gated_sequential_canary_rollout.py"),
            "--source-root",
            str(source_root),
            "--candidate-root",
            str(candidate_root),
            "--batch-root",
            str(batch_root),
            "--config",
            str(sequential_config),
        ],
        check=False,
    )


def _failed_steps_from_root(root: Path) -> list[dict[str, Any]]:
    report = _load_json(root / "policy-gated-sequential-canary-rejection-report.json")
    steps = _read_jsonl(root / "policy-gated-sequential-canary-steps.jsonl")
    return _failed_steps(report, steps)


def _failed_steps(report: dict[str, Any], steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failed = report.get("failed_steps") if isinstance(report.get("failed_steps"), list) else []
    if failed:
        return [step for step in failed if isinstance(step, dict)]
    return [
        step
        for step in steps
        if _string_list(step.get("canary_rejection_reason_codes"))
        or str(step.get("decision_class")) == "canary_rejected_policy_choice"
    ]


def _comparison_rows(base_failed_steps: list[dict[str, Any]], updated_failed_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_by_key = {_step_key(step): step for step in base_failed_steps}
    updated_by_key = {_step_key(step): step for step in updated_failed_steps}
    rows = []
    for key in sorted(set(base_by_key) | set(updated_by_key)):
        base = base_by_key.get(key)
        updated = updated_by_key.get(key)
        reference = updated or base or {}
        rows.append(
            {
                "episode_id": key[0],
                "step_index": key[1],
                "scenario_id": reference.get("scenario_id"),
                "scenario_group": reference.get("scenario_group"),
                "base_failed": base is not None,
                "updated_failed": updated is not None,
                "base_reasons": _string_list((base or {}).get("canary_rejection_reason_codes")),
                "updated_reasons": _string_list((updated or {}).get("canary_rejection_reason_codes")),
                "base_raw_policy_logit_margin_vs_source": (base or {}).get("raw_policy_logit_margin_vs_source"),
                "updated_raw_policy_logit_margin_vs_source": (updated or {}).get("raw_policy_logit_margin_vs_source"),
                "base_source_action_index": (base or {}).get("source_selected_action_index"),
                "updated_source_action_index": (updated or {}).get("source_selected_action_index"),
                "base_raw_policy_action_index": (base or {}).get("raw_policy_selected_action_index"),
                "updated_raw_policy_action_index": (updated or {}).get("raw_policy_selected_action_index"),
                "base_policy_action_index": (base or {}).get("policy_selected_action_index"),
                "updated_policy_action_index": (updated or {}).get("policy_selected_action_index"),
                "base_source_policy_logit": (base or {}).get("source_selected_policy_logit"),
                "updated_source_policy_logit": (updated or {}).get("source_selected_policy_logit"),
                "base_raw_policy_logit": (base or {}).get("raw_policy_selected_policy_logit"),
                "updated_raw_policy_logit": (updated or {}).get("raw_policy_selected_policy_logit"),
                "base_policy_logit": (base or {}).get("policy_selected_policy_logit"),
                "updated_policy_logit": (updated or {}).get("policy_selected_policy_logit"),
                "base_policy_target_cell": (base or {}).get("policy_selected_policy_target_cell"),
                "updated_policy_target_cell": (updated or {}).get("policy_selected_policy_target_cell"),
                "base_source_target_cell": (base or {}).get("source_selected_policy_target_cell"),
                "updated_source_target_cell": (updated or {}).get("source_selected_policy_target_cell"),
                "base_policy_execution_goal_cell": (base or {}).get("policy_selected_execution_goal_cell"),
                "updated_policy_execution_goal_cell": (updated or {}).get("policy_selected_execution_goal_cell"),
                "base_policy_path_cost_delta": (base or {}).get("policy_selected_path_cost_delta"),
                "updated_policy_path_cost_delta": (updated or {}).get("policy_selected_path_cost_delta"),
                "base_policy_risk_delta": (base or {}).get("policy_selected_risk_delta"),
                "updated_policy_risk_delta": (updated or {}).get("policy_selected_risk_delta"),
                "base_policy_utility_delta": (base or {}).get("policy_selected_utility_delta"),
                "updated_policy_utility_delta": (updated or {}).get("policy_selected_utility_delta"),
                "base_raw_policy_path_cost_delta": (base or {}).get("raw_policy_selected_path_cost_delta"),
                "updated_raw_policy_path_cost_delta": (updated or {}).get("raw_policy_selected_path_cost_delta"),
                "base_raw_policy_risk_delta": (base or {}).get("raw_policy_selected_risk_delta"),
                "updated_raw_policy_risk_delta": (updated or {}).get("raw_policy_selected_risk_delta"),
                "base_controlled_reasons": _string_list((base or {}).get("controlled_regression_reason_codes")),
                "updated_controlled_reasons": _string_list((updated or {}).get("controlled_regression_reason_codes")),
                "base_raw_policy_reasons": _string_list((base or {}).get("raw_policy_regression_reason_codes")),
                "updated_raw_policy_reasons": _string_list((updated or {}).get("raw_policy_regression_reason_codes")),
            }
        )
    return rows


def _step_key(step: dict[str, Any]) -> tuple[str, int]:
    return str(step.get("episode_id") or ""), _int_value(step.get("step_index"))


def _gate_metric_mismatches(steps: list[dict[str, Any]], *, config: dict[str, Any]) -> list[dict[str, Any]]:
    evaluation = config.get("evaluation", {})
    max_path = float(evaluation.get("max_path_cost_regression", 0.0))
    max_risk = float(evaluation.get("max_risk_regression", 0.0))
    mismatches = []
    for step in steps:
        canary_reasons = set(_string_list(step.get("canary_rejection_reason_codes")))
        controlled_reasons = set(_string_list(step.get("controlled_regression_reason_codes")))
        path_delta = _float_value(step.get("policy_selected_path_cost_delta"))
        risk_delta = _float_value(step.get("policy_selected_risk_delta"))
        raw_path_delta = _float_value(step.get("raw_policy_selected_path_cost_delta"))
        raw_risk_delta = _float_value(step.get("raw_policy_selected_risk_delta"))
        mismatch_reasons = []
        if "path_cost_regression" in canary_reasons and raw_path_delta <= max_path:
            mismatch_reasons.append("raw_policy_path_cost_regression_without_positive_raw_delta")
        if "risk_regression" in canary_reasons and raw_risk_delta <= max_risk:
            mismatch_reasons.append("raw_policy_risk_regression_without_positive_raw_delta")
        if "path_cost_regression" in controlled_reasons and path_delta <= max_path:
            mismatch_reasons.append("controlled_path_cost_regression_without_positive_step_delta")
        if "risk_regression" in controlled_reasons and risk_delta <= max_risk:
            mismatch_reasons.append("controlled_risk_regression_without_positive_step_delta")
        if mismatch_reasons:
            row = dict(step)
            row["gate_metric_mismatch_reasons"] = mismatch_reasons
            mismatches.append(row)
    return mismatches


def _diagnosis_verdict(
    *,
    reason_codes: list[str],
    clone_ok: bool,
    base_summary: dict[str, Any],
    updated_summary: dict[str, Any],
    base_failed_steps: list[dict[str, Any]],
    updated_failed_steps: list[dict[str, Any]],
    gate_metric_mismatch_count: int,
) -> str:
    if reason_codes:
        return "diagnosis_inconclusive"
    if gate_metric_mismatch_count:
        return "gate_accounting_or_metric_mismatch"
    if not clone_ok or not base_summary:
        return "stale_or_unreplayable_base_candidate"
    base_status = str(base_summary.get("status") or "")
    updated_status = str(updated_summary.get("status") or "")
    if base_status == "passed" and updated_status != "passed":
        return "ppo_update_induced_generated_regression"
    if base_status != "passed" and updated_status != "passed":
        base_keys = {_step_key(step) for step in base_failed_steps}
        updated_keys = {_step_key(step) for step in updated_failed_steps}
        if base_keys and base_keys == updated_keys:
            return "pre_existing_generated_sequential_contract_mismatch"
    return "diagnosis_inconclusive"


def _recommended_next_action(verdict: str) -> str:
    return {
        "ppo_update_induced_generated_regression": "update_objective_or_learning_rate_guard_required",
        "pre_existing_generated_sequential_contract_mismatch": "generated_sequential_contract_alignment_required",
        "stale_or_unreplayable_base_candidate": "base_candidate_replay_provenance_refresh_required",
        "gate_accounting_or_metric_mismatch": "generated_sequential_gate_metric_audit_required",
        "diagnosis_inconclusive": "manual_contract_compatibility_review_required",
    }.get(verdict, "manual_contract_compatibility_review_required")


def _write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    gate_metric_mismatches: list[dict[str, Any]],
) -> None:
    lines = [
        "# Quasi-Real / Generated Sequential Contract Compatibility Diagnosis",
        "",
        f"- status: `{summary['status']}`",
        f"- verdict: `{summary['diagnosis_verdict']}`",
        f"- recommended_next_action: `{summary['recommended_next_action']}`",
        f"- failed_step_count: `{summary['failed_step_count']}`",
        f"- base_generated_sequential_status: `{summary['base_generated_sequential_status']}`",
        f"- updated_generated_sequential_status: `{summary['updated_generated_sequential_status']}`",
        "",
        "## Failed Families",
    ]
    for family, count in summary["failed_family_counts"].items():
        lines.append(f"- `{family}`: {count}")
    lines.extend(["", "## Rejection Reasons"])
    for reason, count in summary["canary_rejection_reason_counts"].items():
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Failed Steps"])
    for row in comparison_rows:
        if row["updated_failed"]:
            lines.append(
                f"- `{row['episode_id']}` step `{row['step_index']}` family "
                f"`{row.get('scenario_group')}` reasons `{','.join(row['updated_reasons'])}`"
            )
    if gate_metric_mismatches:
        lines.extend(["", "## Gate Metric Mismatches"])
        for step in gate_metric_mismatches:
            lines.append(
                f"- `{step.get('episode_id')}` step `{step.get('step_index')}`: "
                f"{','.join(step.get('gate_metric_mismatch_reasons', []))}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _paths(output_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["output_files"]
    return {
        "summary": output_root / outputs.get("summary", "quasi-real-generated-sequential-contract-compatibility-summary.json"),
        "failed_step_comparison": output_root / outputs.get("failed_step_comparison", "failed-step-comparison.jsonl"),
        "baseline_vs_updated_summary": output_root / outputs.get("baseline_vs_updated_summary", "baseline-vs-updated-sequential-summary.json"),
        "report": output_root / outputs.get("report", "compatibility-diagnosis-report.md"),
    }


def _input_paths(update_smoke_root: Path, base_candidate_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    inputs = config["input_files"]
    return {
        "update_smoke_summary": update_smoke_root / inputs.get("update_smoke_summary", "limited-quasi-real-ppo-update-smoke-summary.json"),
        "generated_sequential_summary": update_smoke_root / inputs.get("generated_sequential_summary", "post_update_generated_sequential/policy-gated-sequential-canary-rollout-summary.json"),
        "generated_sequential_steps": update_smoke_root / inputs.get("generated_sequential_steps", "post_update_generated_sequential/policy-gated-sequential-canary-steps.jsonl"),
        "generated_sequential_rejection_report": update_smoke_root / inputs.get("generated_sequential_rejection_report", "post_update_generated_sequential/policy-gated-sequential-canary-rejection-report.json"),
        "quasi_real_teacher_following_summary": update_smoke_root / inputs.get("quasi_real_teacher_following_summary", "post_update_quasi_real_teacher_following/quasi-real-guarded-teacher-following-pilot-summary.json"),
        "quasi_real_collector_summary": update_smoke_root / inputs.get("quasi_real_collector_summary", "post_update_quasi_real_collector/ppo-rollout-collector-summary.json"),
        "base_checkpoint": base_candidate_root / inputs.get("base_checkpoint", "experimental-hybrid-policy-candidate.pt"),
        "base_checkpoint_metadata": base_candidate_root / inputs.get("base_checkpoint_metadata", "experimental-hybrid-policy-candidate-metadata.json"),
        "base_candidate_summary": base_candidate_root / inputs.get("base_candidate_summary", "raw-policy-generalization-candidate-summary.json"),
    }


def _load_config(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    for section in ("input_files", "output_files", "replay", "evaluation", "validation"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _load_json_required(path: Path, reason_codes: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return {}
    payload = _load_json(path)
    if not payload:
        _append_reason(reason_codes, f"{label}_invalid")
    return payload


def _read_jsonl_required(path: Path, reason_codes: list[str], label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return []
    records = _read_jsonl(path)
    if not records:
        _append_reason(reason_codes, f"{label}_empty")
    return records


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    records = []
    for line in lines:
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
