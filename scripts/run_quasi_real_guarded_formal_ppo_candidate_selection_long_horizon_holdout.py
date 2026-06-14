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


CONFIG_SCHEMA_VERSION = (
    "quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-config/v1"
)
SUMMARY_SCHEMA_VERSION = (
    "quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-summary/v1"
)
EXPECTED_READINESS_STATUS = (
    "quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_evaluated"
)

STABILITY_SUMMARY_FILE = (
    "quasi-real-guarded-formal-ppo-stability-holdout-validation-summary.json"
)
SUMMARY_FILE = (
    "quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-summary.json"
)
SELECTION_AUDIT_FILE = "candidate-selection-audit.json"
HOLDOUT_EPISODES_FILE = "long-horizon-holdout-episodes.jsonl"
HOLDOUT_STEPS_FILE = "long-horizon-holdout-steps.jsonl"
RETURN_AUDIT_FILE = "long-horizon-return-audit.json"
SPLIT_REPORT_FILE = "long-horizon-holdout-split-report.json"
FAMILY_REPORT_FILE = "long-horizon-family-report.json"
CANDIDATE_MANIFEST_FILE = "selected-candidate-manifest.json"
ROLLBACK_MANIFEST_FILE = "candidate-selection-rollback-manifest.json"
READINESS_FILE = "candidate-selection-readiness-validate-only.json"
REPORT_FILE = "candidate-selection-long-horizon-holdout-report.md"

ReadinessRunner = Callable[..., dict[str, Any]]


def run_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout(
    *,
    stability_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    readiness_runner: ReadinessRunner | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    stability_root = Path(stability_root)
    output_root = Path(output_root)
    batch_root = Path(batch_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = _output_files(config)
    summary_path = output_root / files["summary"]
    selection_audit_path = output_root / files["selection_audit"]
    holdout_episodes_path = output_root / files["holdout_episodes"]
    holdout_steps_path = output_root / files["holdout_steps"]
    return_audit_path = output_root / files["return_audit"]
    split_report_path = output_root / files["split_report"]
    family_report_path = output_root / files["family_report"]
    candidate_manifest_path = output_root / files["candidate_manifest"]
    rollback_manifest_path = output_root / files["rollback_manifest"]
    readiness_path = output_root / files["readiness_validate_only"]
    report_path = output_root / files["report"]

    stability_summary_path = stability_root / STABILITY_SUMMARY_FILE
    stability_summary = _read_json_if_exists(stability_summary_path)
    matrix_path = _resolve_optional_path(
        stability_summary.get("stability_matrix"), stability_root, repo_root
    ) or (stability_root / "formal-ppo-stability-matrix.jsonl")
    baseline_manifest_path = _resolve_optional_path(
        stability_summary.get("baseline_manifest"), stability_root, repo_root
    )
    rollback_input_path = _resolve_optional_path(
        stability_summary.get("rollback_manifest"), stability_root, repo_root
    )
    baseline_manifest = _read_json_if_exists(baseline_manifest_path)
    rollback_input_manifest = _read_json_if_exists(rollback_input_path)
    steps_path = _resolve_steps_path(stability_summary, stability_root, repo_root)

    matrix_rows = _read_jsonl(matrix_path)
    steps = _read_jsonl(steps_path)
    trainable_steps = _unique_trainable_steps(steps)
    counters = _input_counters(steps, trainable_steps)

    reason_codes: list[str] = []
    _validate_stability_input(
        stability_summary,
        matrix_rows,
        baseline_manifest,
        rollback_input_manifest,
        counters,
        config,
        reason_codes,
    )

    selection = _select_candidate(matrix_rows, counters, config)
    if not selection["selected"]:
        _add_reason(reason_codes, "candidate_selection_no_eligible_candidate")
    _write_json(selection_audit_path, _selection_audit(selection, matrix_path))

    horizon = _horizon(config)
    holdout = _build_long_horizon_holdout(trainable_steps, horizon=horizon, config=config)
    _write_jsonl(holdout_steps_path, holdout["steps"])
    _write_jsonl(holdout_episodes_path, holdout["episodes"])
    _write_json(return_audit_path, _return_audit(holdout, horizon))
    _write_json(split_report_path, _split_report(steps, trainable_steps))
    _write_json(family_report_path, _family_report(trainable_steps, holdout))
    _write_json(
        candidate_manifest_path,
        _candidate_manifest(
            selection=selection,
            stability_root=stability_root,
            stability_summary_path=stability_summary_path,
            matrix_path=matrix_path,
            baseline_manifest_path=baseline_manifest_path,
            repo_root=repo_root,
        ),
    )
    _write_json(
        rollback_manifest_path,
        _rollback_manifest(
            output_root=output_root,
            selected_candidate_root=selection["selected"].get("updated_candidate_root")
            if selection["selected"]
            else None,
            stability_rollback_manifest_path=rollback_input_path,
            candidate_manifest_path=candidate_manifest_path,
        ),
    )

    _validate_holdout(counters, holdout, selection, config, reason_codes)

    status_without_readiness = "passed" if not reason_codes else "failed"
    pre_readiness_summary = _summary_payload(
        status=status_without_readiness,
        reason_codes=reason_codes,
        repo_root=repo_root,
        stability_root=stability_root,
        stability_summary_path=stability_summary_path,
        matrix_path=matrix_path,
        steps_path=steps_path,
        output_root=output_root,
        batch_root=batch_root,
        summary_path=summary_path,
        selection_audit_path=selection_audit_path,
        holdout_episodes_path=holdout_episodes_path,
        holdout_steps_path=holdout_steps_path,
        return_audit_path=return_audit_path,
        split_report_path=split_report_path,
        family_report_path=family_report_path,
        candidate_manifest_path=candidate_manifest_path,
        rollback_manifest_path=rollback_manifest_path,
        readiness_path=readiness_path,
        report_path=report_path,
        counters=counters,
        selection=selection,
        holdout=holdout,
        readiness={},
    )
    _write_json(summary_path, pre_readiness_summary)

    if status_without_readiness == "passed":
        runner = readiness_runner or _run_readiness_validate_only
        readiness = runner(
            repo_root=repo_root,
            batch_root=batch_root,
            candidate_selection_summary_path=summary_path,
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
            "recommended_next_action": (
                "fix_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout"
            ),
        }
    _write_json(readiness_path, readiness)

    final_status = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=final_status,
        reason_codes=reason_codes,
        repo_root=repo_root,
        stability_root=stability_root,
        stability_summary_path=stability_summary_path,
        matrix_path=matrix_path,
        steps_path=steps_path,
        output_root=output_root,
        batch_root=batch_root,
        summary_path=summary_path,
        selection_audit_path=selection_audit_path,
        holdout_episodes_path=holdout_episodes_path,
        holdout_steps_path=holdout_steps_path,
        return_audit_path=return_audit_path,
        split_report_path=split_report_path,
        family_report_path=family_report_path,
        candidate_manifest_path=candidate_manifest_path,
        rollback_manifest_path=rollback_manifest_path,
        readiness_path=readiness_path,
        report_path=report_path,
        counters=counters,
        selection=selection,
        holdout=holdout,
        readiness=readiness,
    )
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run quasi-real guarded formal PPO candidate selection and long-horizon holdout."
    )
    parser.add_argument(
        "--stability-root",
        default=(
            "outputs/"
            "path_feedback_batch_quasi_real_guarded_formal_ppo_stability_holdout_validation_v1"
        ),
    )
    parser.add_argument(
        "--batch-root",
        default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1",
    )
    parser.add_argument(
        "--output-root",
        default=(
            "outputs/"
            "path_feedback_batch_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_v1"
        ),
    )
    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_v1.json"
        ),
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

    summary = run_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout(
        stability_root=_resolve_path(Path(args.stability_root), repo_root),
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
                "selected_seed": summary.get("selected_seed"),
                "selected_budget": summary.get("selected_budget"),
                "horizon": summary["horizon"],
                "readiness_status": summary.get("readiness_status"),
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
    candidate_selection_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        "bash",
        str(repo_root / "scripts" / "run_policy_training_readiness_review.sh"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(_resolve_path(config_path, repo_root)),
        "--quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-summary",
        str(candidate_selection_summary_path),
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


def _validate_stability_input(
    summary: dict[str, Any],
    matrix_rows: list[dict[str, Any]],
    baseline_manifest: dict[str, Any],
    rollback_manifest: dict[str, Any],
    counters: Counter[str],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if summary.get("schema_version") != (
        "quasi-real-guarded-formal-ppo-stability-holdout-validation-summary/v1"
    ):
        _add_reason(reason_codes, "input_formal_ppo_stability_holdout_schema_invalid")
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _add_reason(reason_codes, "input_formal_ppo_stability_holdout_not_passed")
    expected = _expected_trainable(config)
    if _int(summary.get("input_trainable_transition_count")) != expected:
        _add_reason(reason_codes, "input_formal_ppo_stability_holdout_trainable_count_mismatch")
    if _int(summary.get("unique_trainable_context_count")) != expected:
        _add_reason(reason_codes, "input_formal_ppo_stability_holdout_unique_context_count_mismatch")
    if _int(summary.get("run_count")) <= 0 or _int(summary.get("passed_run_count")) != _int(
        summary.get("run_count")
    ):
        _add_reason(reason_codes, "input_formal_ppo_stability_holdout_run_not_all_passed")
    if not matrix_rows:
        _add_reason(reason_codes, "input_formal_ppo_stability_matrix_missing")
    if not baseline_manifest:
        _add_reason(reason_codes, "input_formal_ppo_stability_baseline_manifest_missing")
    if not rollback_manifest:
        _add_reason(reason_codes, "input_formal_ppo_stability_rollback_manifest_missing")
    git = summary.get("git_provenance") if isinstance(summary.get("git_provenance"), dict) else {}
    current_git = git.get("current") if isinstance(git.get("current"), dict) else {}
    if git.get("current_matches_sources") is False:
        _add_reason(reason_codes, "input_formal_ppo_stability_git_provenance_mismatch")
    if current_git.get("dirty") is True:
        _add_reason(reason_codes, "input_formal_ppo_stability_git_provenance_dirty")
    if counters["input_trainable_transition_count"] != expected:
        _add_reason(reason_codes, "candidate_selection_holdout_trainable_count_mismatch")
    if counters["unique_trainable_context_count"] != expected:
        _add_reason(reason_codes, "candidate_selection_holdout_unique_context_count_mismatch")
    for field, reason in (
        ("validation_trainable_count", "candidate_selection_holdout_split_leakage"),
        ("test_trainable_count", "candidate_selection_holdout_split_leakage"),
        ("fallback_trainable_count", "candidate_selection_holdout_fallback_trainable"),
        ("non_empty_gate_reason_trainable_count", "candidate_selection_holdout_gate_reason_trainable"),
        ("missing_observation_count", "candidate_selection_holdout_contract_invalid"),
        ("missing_log_prob_count", "candidate_selection_holdout_contract_invalid"),
        ("missing_value_count", "candidate_selection_holdout_contract_invalid"),
        ("non_finite_reward_count", "candidate_selection_holdout_non_finite"),
        ("non_finite_return_count", "candidate_selection_holdout_non_finite"),
        ("non_finite_advantage_count", "candidate_selection_holdout_non_finite"),
        ("controlled_regression_count", "candidate_selection_holdout_controlled_regression"),
        ("controlled_safety_regression_count", "candidate_selection_holdout_controlled_regression"),
        ("controlled_contract_regression_count", "candidate_selection_holdout_controlled_regression"),
        ("controlled_path_risk_regression_count", "candidate_selection_holdout_controlled_regression"),
        ("controlled_source_selection_regression_count", "candidate_selection_holdout_controlled_regression"),
    ):
        if counters[field]:
            _add_reason(reason_codes, reason)
    for field, reason in (
        ("publishes_checkpoint", "input_formal_ppo_stability_holdout_checkpoint_publication_claimed"),
        ("replaces_default_policy", "input_formal_ppo_stability_holdout_default_policy_replacement_claimed"),
        ("performance_claimed", "input_formal_ppo_stability_holdout_policy_performance_claimed"),
        ("formal_training_ready_claimed", "input_formal_ppo_stability_holdout_formal_ready_claimed"),
    ):
        if summary.get(field) is True:
            _add_reason(reason_codes, reason)


def _select_candidate(
    matrix_rows: list[dict[str, Any]],
    counters: Counter[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    candidate_rows = []
    for index, row in enumerate(matrix_rows):
        rejection_reasons = _candidate_rejection_reasons(row, counters, config)
        candidate_rows.append(
            {
                "index": index,
                "seed": _int(row.get("seed")),
                "budget_name": str(row.get("budget_name") or ""),
                "updated_candidate_root": row.get("updated_candidate_root"),
                "approx_kl": _float(row.get("approx_kl"), math.inf),
                "max_grad_norm_after_clip": _float(row.get("max_grad_norm_after_clip"), math.inf),
                "parameter_l2_delta": _float(row.get("parameter_l2_delta"), 0.0),
                "teacher_agreement_rate": _float(row.get("teacher_agreement_rate"), 0.0),
                "rejection_reasons": rejection_reasons,
                "eligible": not rejection_reasons,
                "source": row,
            }
        )
    eligible = [row for row in candidate_rows if row["eligible"]]
    eligible.sort(
        key=lambda row: (
            abs(_float(row["approx_kl"], math.inf)),
            _float(row["max_grad_norm_after_clip"], math.inf),
            _int(row["seed"]),
            str(row["budget_name"]),
        )
    )
    selected = dict(eligible[0]["source"]) if eligible else {}
    return {
        "selection_rule": (
            "eligible passed runs are filtered by regression, reconstruction, finite numeric, "
            "KL, grad norm, teacher agreement, and parameter-delta gates; ties use lowest "
            "abs(approx_kl), grad norm, seed, then budget name"
        ),
        "candidate_rows": [
            {key: value for key, value in row.items() if key != "source"} for row in candidate_rows
        ],
        "eligible_candidate_count": len(eligible),
        "rejected_candidate_count": len(candidate_rows) - len(eligible),
        "selected": selected,
        "candidate_selection_reproducible": bool(eligible),
        "candidate_selection_uses_single_loss_or_reward": False,
    }


def _candidate_rejection_reasons(
    row: dict[str, Any],
    counters: Counter[str],
    config: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    validation = config.get("validation", {})
    expected = counters["input_trainable_transition_count"]
    if row.get("status") != "passed" or _string_list(row.get("reason_codes")):
        reasons.append("run_not_passed")
    if _int(row.get("optimizer_train_transition_count")) != expected:
        reasons.append("optimizer_count_mismatch")
    if _float(row.get("old_log_prob_max_abs_error"), math.inf) > _float(
        validation.get("max_old_log_prob_abs_error"), 1.0e-4
    ) or _float(row.get("old_value_max_abs_error"), math.inf) > _float(
        validation.get("max_old_value_abs_error"), 1.0e-4
    ):
        reasons.append("old_policy_reconstruction_error")
    if abs(_float(row.get("approx_kl"), math.inf)) > _float(validation.get("max_abs_approx_kl"), 0.25):
        reasons.append("kl_too_large")
    if _float(row.get("max_grad_norm_after_clip"), math.inf) > _float(
        validation.get("max_grad_norm_after_clip"), 1.0
    ) + 1.0e-8:
        reasons.append("grad_norm_too_large")
    if _float(row.get("parameter_l2_delta"), 0.0) <= 0.0:
        reasons.append("parameter_delta_missing")
    if _float(row.get("teacher_agreement_rate"), 0.0) < _float(
        validation.get("min_teacher_agreement_rate"), 0.95
    ):
        reasons.append("teacher_alignment_insufficient")
    if any(
        _int(row.get(field))
        for field in (
            "loss_non_finite_count",
            "non_finite_gradient_count",
            "non_finite_reward_count",
            "non_finite_return_count",
            "non_finite_advantage_count",
        )
    ):
        reasons.append("non_finite_numeric")
    if any(
        _int(row.get(field))
        for field in (
            "controlled_regression_count",
            "train_controlled_regression_count",
            "validation_controlled_regression_count",
            "test_controlled_regression_count",
            "controlled_safety_regression_count",
            "controlled_contract_regression_count",
            "controlled_path_risk_regression_count",
            "controlled_source_selection_regression_count",
        )
    ):
        reasons.append("controlled_regression")
    if _int(row.get("family_regression_count")):
        reasons.append("family_regression")
    if any(
        row.get(field) is True
        for field in (
            "publishes_checkpoint",
            "replaces_default_policy",
            "performance_claimed",
            "formal_training_ready_claimed",
        )
    ):
        reasons.append("publication_or_claim_flag")
    if not row.get("updated_candidate_root"):
        reasons.append("candidate_root_missing")
    return reasons


def _validate_holdout(
    counters: Counter[str],
    holdout: dict[str, Any],
    selection: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if _horizon(config) < _int(config.get("validation", {}).get("min_horizon"), 10):
        _add_reason(reason_codes, "candidate_selection_holdout_horizon_below_threshold")
    if not selection["selected"]:
        _add_reason(reason_codes, "candidate_selection_no_eligible_candidate")
    if holdout["non_finite_long_horizon_return_count"]:
        _add_reason(reason_codes, "candidate_selection_holdout_non_finite")
    if holdout["non_finite_long_horizon_advantage_count"]:
        _add_reason(reason_codes, "candidate_selection_holdout_non_finite")
    if counters["controlled_regression_count"]:
        _add_reason(reason_codes, "candidate_selection_holdout_controlled_regression")


def _validate_readiness(readiness: dict[str, Any], config: dict[str, Any], reason_codes: list[str]) -> None:
    expected = str(config.get("readiness", {}).get("expected_status") or EXPECTED_READINESS_STATUS)
    if readiness.get("training_readiness_status") != expected:
        _add_reason(reason_codes, "readiness_not_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_evaluated")
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
    stability_root: Path,
    stability_summary_path: Path,
    matrix_path: Path,
    steps_path: Path,
    output_root: Path,
    batch_root: Path,
    summary_path: Path,
    selection_audit_path: Path,
    holdout_episodes_path: Path,
    holdout_steps_path: Path,
    return_audit_path: Path,
    split_report_path: Path,
    family_report_path: Path,
    candidate_manifest_path: Path,
    rollback_manifest_path: Path,
    readiness_path: Path,
    report_path: Path,
    counters: Counter[str],
    selection: dict[str, Any],
    holdout: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    selected = selection.get("selected") or {}
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": None
        if status == "passed"
        else "fix_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout",
        "stability_root": str(stability_root),
        "stability_summary": str(stability_summary_path),
        "stability_matrix": str(matrix_path),
        "steps": str(steps_path),
        "batch_root": str(batch_root),
        "output_root": str(output_root),
        "summary": str(summary_path),
        "selection_audit": str(selection_audit_path),
        "holdout_episodes": str(holdout_episodes_path),
        "holdout_steps": str(holdout_steps_path),
        "return_audit": str(return_audit_path),
        "split_report": str(split_report_path),
        "family_report": str(family_report_path),
        "candidate_manifest": str(candidate_manifest_path),
        "rollback_manifest": str(rollback_manifest_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "selected_seed": selected.get("seed"),
        "selected_budget": selected.get("budget_name"),
        "selected_candidate_root": selected.get("updated_candidate_root"),
        "selected_candidate_from_stability_matrix": bool(selected),
        "candidate_selection_reproducible": bool(selection.get("candidate_selection_reproducible")),
        "candidate_selection_uses_single_loss_or_reward": False,
        "eligible_candidate_count": _int(selection.get("eligible_candidate_count")),
        "rejected_candidate_count": _int(selection.get("rejected_candidate_count")),
        "input_trainable_transition_count": counters["input_trainable_transition_count"],
        "long_horizon_trainable_transition_count": counters["input_trainable_transition_count"],
        "optimizer_train_transition_count": 0,
        "unique_trainable_context_count": counters["unique_trainable_context_count"],
        "step_count": counters["step_count"],
        "horizon": holdout["horizon"],
        "long_horizon_episode_count": holdout["episode_count"],
        "completed_long_horizon_episode_count": holdout["completed_long_horizon_episode_count"],
        "long_horizon_step_count": holdout["step_count"],
        "leftover_step_count": holdout["leftover_step_count"],
        "min_long_horizon_discounted_return": holdout["min_discounted_return"],
        "max_long_horizon_discounted_return": holdout["max_discounted_return"],
        "non_finite_long_horizon_return_count": holdout["non_finite_long_horizon_return_count"],
        "non_finite_long_horizon_advantage_count": holdout["non_finite_long_horizon_advantage_count"],
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
        "loss_non_finite_count": 0,
        "non_finite_gradient_count": 0,
        "max_old_log_prob_abs_error": _float(selected.get("old_log_prob_max_abs_error"), 0.0),
        "max_old_value_abs_error": _float(selected.get("old_value_max_abs_error"), 0.0),
        "max_abs_approx_kl": abs(_float(selected.get("approx_kl"), 0.0)),
        "max_grad_norm_after_clip": _float(selected.get("max_grad_norm_after_clip"), 0.0),
        "min_parameter_l2_delta": _float(selected.get("parameter_l2_delta"), 0.0),
        "teacher_agreement_rate": _float(selected.get("teacher_agreement_rate"), 0.0),
        "controlled_regression_count": counters["controlled_regression_count"],
        "train_controlled_regression_count": counters["train_controlled_regression_count"],
        "validation_controlled_regression_count": counters["validation_controlled_regression_count"],
        "test_controlled_regression_count": counters["test_controlled_regression_count"],
        "family_regression_count": counters["family_regression_count"],
        "controlled_safety_regression_count": counters["controlled_safety_regression_count"],
        "controlled_contract_regression_count": counters["controlled_contract_regression_count"],
        "controlled_path_risk_regression_count": counters["controlled_path_risk_regression_count"],
        "controlled_source_selection_regression_count": counters["controlled_source_selection_regression_count"],
        "safe_better_opportunity_count": counters["safe_better_opportunity_count"],
        "readiness_status": readiness.get("training_readiness_status"),
        "training_blockers": list(readiness.get("training_blockers") or []),
        "readiness_reason_codes": list(readiness.get("reason_codes") or []),
        "runs_formal_ppo_candidate_selection_long_horizon_holdout": True,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _build_long_horizon_holdout(
    trainable_steps: list[dict[str, Any]],
    *,
    horizon: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    discount = _float(config.get("holdout", {}).get("discount_factor"), 0.99)
    steps: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    non_finite_return_count = 0
    non_finite_advantage_count = 0
    returns: list[float] = []
    for episode_index, offset in enumerate(range(0, len(trainable_steps), horizon)):
        chunk = trainable_steps[offset : offset + horizon]
        episode_id = f"long-horizon-holdout-{episode_index:04d}"
        rewards = [_float(step.get("reward"), math.nan) for step in chunk]
        discounted_returns = _discounted_returns(rewards, discount)
        episode_return = discounted_returns[0] if discounted_returns else 0.0
        for local_index, (step, discounted_return) in enumerate(zip(chunk, discounted_returns)):
            value = _float(step.get("value"), 0.0)
            advantage = discounted_return - value
            if not _finite(discounted_return):
                non_finite_return_count += 1
            if not _finite(advantage):
                non_finite_advantage_count += 1
            returns.append(discounted_return)
            row = dict(step)
            row.update(
                {
                    "schema_version": "quasi-real-guarded-formal-ppo-long-horizon-holdout-step/v1",
                    "long_horizon_episode_id": episode_id,
                    "long_horizon_step_index": local_index,
                    "long_horizon": horizon,
                    "long_horizon_discounted_return": discounted_return,
                    "long_horizon_advantage": advantage,
                    "diagnostic_only": True,
                    "long_horizon_trainable": step.get("split") == "train",
                }
            )
            steps.append(row)
        episodes.append(
            {
                "schema_version": "quasi-real-guarded-formal-ppo-long-horizon-holdout-episode/v1",
                "episode_id": episode_id,
                "horizon": horizon,
                "step_count": len(chunk),
                "completed_horizon": len(chunk) == horizon,
                "discounted_return": episode_return,
                "scenario_family_counts": dict(
                    Counter(str(step.get("scenario_family") or "unknown") for step in chunk)
                ),
            }
        )
    completed = sum(1 for episode in episodes if episode["completed_horizon"])
    return {
        "horizon": horizon,
        "episode_count": len(episodes),
        "completed_long_horizon_episode_count": completed,
        "step_count": len(steps),
        "leftover_step_count": len(trainable_steps) % horizon,
        "steps": steps,
        "episodes": episodes,
        "min_discounted_return": min(returns) if returns else 0.0,
        "max_discounted_return": max(returns) if returns else 0.0,
        "non_finite_long_horizon_return_count": non_finite_return_count,
        "non_finite_long_horizon_advantage_count": non_finite_advantage_count,
    }


def _discounted_returns(rewards: list[float], discount: float) -> list[float]:
    returns = [0.0 for _ in rewards]
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + discount * running
        returns[index] = running
    return returns


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
        if step.get("controlled_choice_detail") == "policy_safe_better":
            counters["safe_better_opportunity_count"] += 1
        reasons = set(_string_list(step.get("controlled_regression_reason_codes")))
        if reasons:
            counters["controlled_regression_count"] += 1
        if step.get("split") == "train" and reasons:
            counters["train_controlled_regression_count"] += 1
        if step.get("split") == "validation" and reasons:
            counters["validation_controlled_regression_count"] += 1
        if step.get("split") == "test" and reasons:
            counters["test_controlled_regression_count"] += 1
        if "safety_regression" in reasons:
            counters["controlled_safety_regression_count"] += 1
        if "contract_violation" in reasons or "contract_regression" in reasons:
            counters["controlled_contract_regression_count"] += 1
        if "path_cost_regression" in reasons or "risk_regression" in reasons:
            counters["controlled_path_risk_regression_count"] += 1
        if "source_selection_regression" in reasons:
            counters["controlled_source_selection_regression_count"] += 1
    return counters


def _unique_trainable_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_context: dict[str, dict[str, Any]] = {}
    for step in steps:
        context_id = str(step.get("context_id") or "")
        if not context_id or context_id in by_context:
            continue
        if not _is_trainable_step(step):
            continue
        by_context[context_id] = step
    return list(by_context.values())


def _is_trainable_step(step: dict[str, Any]) -> bool:
    return (
        step.get("ppo_trainable") is True
        and step.get("split") == "train"
        and step.get("controlled_choice_source") == "policy"
        and not _string_list(step.get("gate_reason_codes"))
        and not _string_list(step.get("controlled_regression_reason_codes"))
        and step.get("observation") is not None
        and _finite(step.get("log_prob"))
        and _finite(step.get("value"))
        and _finite(step.get("reward"))
        and _finite(step.get("discounted_return"))
        and _finite(step.get("advantage"))
    )


def _selection_audit(selection: dict[str, Any], matrix_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "quasi-real-guarded-formal-ppo-candidate-selection-audit/v1",
        "matrix": str(matrix_path),
        "selection_rule": selection["selection_rule"],
        "eligible_candidate_count": selection["eligible_candidate_count"],
        "rejected_candidate_count": selection["rejected_candidate_count"],
        "selected_seed": selection["selected"].get("seed") if selection["selected"] else None,
        "selected_budget": selection["selected"].get("budget_name") if selection["selected"] else None,
        "candidate_selection_reproducible": selection["candidate_selection_reproducible"],
        "candidate_selection_uses_single_loss_or_reward": False,
        "candidate_rows": selection["candidate_rows"],
    }


def _candidate_manifest(
    *,
    selection: dict[str, Any],
    stability_root: Path,
    stability_summary_path: Path,
    matrix_path: Path,
    baseline_manifest_path: Path | None,
    repo_root: Path,
) -> dict[str, Any]:
    selected = selection["selected"] or {}
    return {
        "schema_version": "quasi-real-guarded-formal-ppo-selected-candidate-manifest/v1",
        "generated_at": _utc_now(),
        "stability_root": str(stability_root),
        "stability_summary": str(stability_summary_path),
        "stability_matrix": str(matrix_path),
        "stability_baseline_manifest": str(baseline_manifest_path) if baseline_manifest_path else None,
        "selected_seed": selected.get("seed"),
        "selected_budget": selected.get("budget_name"),
        "selected_candidate_root": selected.get("updated_candidate_root"),
        "selected_candidate_from_stability_matrix": bool(selected),
        "diagnostic_selection_only": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _rollback_manifest(
    *,
    output_root: Path,
    selected_candidate_root: Any,
    stability_rollback_manifest_path: Path | None,
    candidate_manifest_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "quasi-real-guarded-formal-ppo-candidate-selection-rollback-manifest/v1",
        "output_root": str(output_root),
        "selected_candidate_root": str(selected_candidate_root) if selected_candidate_root else None,
        "stability_rollback_manifest": str(stability_rollback_manifest_path) if stability_rollback_manifest_path else None,
        "candidate_manifest": str(candidate_manifest_path),
        "restores_selected_candidate_only": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
    }


def _return_audit(holdout: dict[str, Any], horizon: int) -> dict[str, Any]:
    return {
        "schema_version": "quasi-real-guarded-formal-ppo-long-horizon-return-audit/v1",
        "horizon": horizon,
        "episode_count": holdout["episode_count"],
        "completed_long_horizon_episode_count": holdout["completed_long_horizon_episode_count"],
        "step_count": holdout["step_count"],
        "leftover_step_count": holdout["leftover_step_count"],
        "min_discounted_return": holdout["min_discounted_return"],
        "max_discounted_return": holdout["max_discounted_return"],
        "non_finite_long_horizon_return_count": holdout["non_finite_long_horizon_return_count"],
        "non_finite_long_horizon_advantage_count": holdout["non_finite_long_horizon_advantage_count"],
        "uses_multistep_discounted_return": True,
        "does_not_use_single_step_greedy_reward": True,
    }


def _split_report(steps: list[dict[str, Any]], trainable_steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "quasi-real-guarded-formal-ppo-long-horizon-split-report/v1",
        "input_split_counts": dict(Counter(str(step.get("split") or "unknown") for step in steps)),
        "trainable_split_counts": dict(Counter(str(step.get("split") or "unknown") for step in trainable_steps)),
        "validation_trainable_count": sum(
            1 for step in steps if step.get("ppo_trainable") is True and step.get("split") == "validation"
        ),
        "test_trainable_count": sum(
            1 for step in steps if step.get("ppo_trainable") is True and step.get("split") == "test"
        ),
        "validation_test_are_diagnostic_only": True,
    }


def _family_report(trainable_steps: list[dict[str, Any]], holdout: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "quasi-real-guarded-formal-ppo-long-horizon-family-report/v1",
        "trainable_family_counts": dict(
            sorted(Counter(str(step.get("scenario_family") or "unknown") for step in trainable_steps).items())
        ),
        "episode_count": holdout["episode_count"],
        "family_regression_count": 0,
    }


def _render_report(summary: dict[str, Any]) -> str:
    return (
        "# Formal PPO Candidate Selection & Long-Horizon Holdout\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Reason codes: `{summary.get('reason_codes')}`\n"
        f"- Selected candidate: seed `{summary.get('selected_seed')}`, budget `{summary.get('selected_budget')}`\n"
        f"- Horizon: `{summary.get('horizon')}`\n"
        f"- Long-horizon steps: `{summary.get('long_horizon_step_count')}`\n"
        f"- Completed long-horizon episodes: `{summary.get('completed_long_horizon_episode_count')}`\n"
        f"- Controlled regression count: `{summary.get('controlled_regression_count')}`\n"
        f"- Teacher agreement: `{summary.get('teacher_agreement_rate')}`\n"
        f"- Readiness: `{summary.get('readiness_status')}`\n\n"
        "This stage selects an auditable experimental candidate from the frozen "
        "stability matrix and evaluates a longer-horizon holdout. It does not run "
        "a new PPO update, publish a checkpoint, replace the default policy, claim "
        "policy performance, or claim formal training readiness.\n"
    )


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "summary": SUMMARY_FILE,
        "selection_audit": SELECTION_AUDIT_FILE,
        "holdout_episodes": HOLDOUT_EPISODES_FILE,
        "holdout_steps": HOLDOUT_STEPS_FILE,
        "return_audit": RETURN_AUDIT_FILE,
        "split_report": SPLIT_REPORT_FILE,
        "family_report": FAMILY_REPORT_FILE,
        "candidate_manifest": CANDIDATE_MANIFEST_FILE,
        "rollback_manifest": ROLLBACK_MANIFEST_FILE,
        "readiness_validate_only": READINESS_FILE,
        "report": REPORT_FILE,
    }
    configured = config.get("output_files") if isinstance(config.get("output_files"), dict) else {}
    return {key: str(configured.get(key) or default) for key, default in defaults.items()}


def _resolve_steps_path(summary: dict[str, Any], stability_root: Path, repo_root: Path) -> Path:
    configured = summary.get("steps") or "quasi-real-trainable-context-expansion-steps.jsonl"
    path = Path(str(configured))
    if path.is_absolute():
        return path
    candidate = stability_root / path
    return candidate if candidate.is_file() else repo_root / path


def _resolve_optional_path(value: Any, base: Path, repo_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidate = base / path
    return candidate if candidate.exists() else repo_root / path


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _expected_trainable(config: dict[str, Any]) -> int:
    return _int(config.get("validation", {}).get("expected_trainable_transition_count"), 684)


def _horizon(config: dict[str, Any]) -> int:
    return _int(config.get("holdout", {}).get("horizon"), 10)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path | None) -> dict[str, Any]:
    return _read_json(path) if path and path.is_file() else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
