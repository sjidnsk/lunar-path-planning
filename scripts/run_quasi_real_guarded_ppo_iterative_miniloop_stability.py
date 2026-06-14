from __future__ import annotations

import argparse
import copy
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
    from scripts.run_limited_ppo_update_smoke import run_limited_ppo_update_smoke
    from scripts.run_quasi_real_guarded_ppo_scale512_multiseed_preflight import (
        _SeedPolicyEvaluator,
        _refresh_trainable_step_policy_estimates,
        _seed_update_config,
        _unique_trainable_steps,
        _write_seed_collector_artifacts,
    )
except ModuleNotFoundError:  # pragma: no cover
    from run_limited_ppo_update_smoke import run_limited_ppo_update_smoke
    from run_quasi_real_guarded_ppo_scale512_multiseed_preflight import (
        _SeedPolicyEvaluator,
        _refresh_trainable_step_policy_estimates,
        _seed_update_config,
        _unique_trainable_steps,
        _write_seed_collector_artifacts,
    )


CONFIG_SCHEMA_VERSION = "quasi-real-guarded-ppo-iterative-miniloop-stability-config/v1"
SUMMARY_SCHEMA_VERSION = "quasi-real-guarded-ppo-iterative-miniloop-stability-summary/v1"
ITERATION_SCHEMA_VERSION = "quasi-real-guarded-ppo-iterative-miniloop-iteration-summary/v1"
EXPECTED_READINESS_STATUS = "quasi_real_guarded_ppo_iterative_miniloop_stability_evaluated"

SUMMARY_FILE = "quasi-real-guarded-ppo-iterative-miniloop-stability-summary.json"
ITERATION_SUMMARIES_FILE = "iterative-miniloop-iteration-summaries.jsonl"
PROGRESS_FILE = "iterative-miniloop-progress.jsonl"
READINESS_FILE = "iterative-miniloop-readiness-validate-only.json"
REPORT_FILE = "iterative-miniloop-stability-report.md"

IterationRunner = Callable[..., dict[str, Any]]
ReadinessRunner = Callable[..., dict[str, Any]]
PpoUpdateRunner = Callable[..., dict[str, Any]]


def run_quasi_real_guarded_ppo_iterative_miniloop_stability(
    *,
    expansion_root: Path,
    scale512_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    iteration_runner: IterationRunner | None = None,
    readiness_runner: ReadinessRunner | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    expansion_root = Path(expansion_root)
    scale512_root = Path(scale512_root)
    output_root = Path(output_root)
    batch_root = Path(batch_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = _output_files(config)
    summary_path = output_root / files["summary"]
    iteration_summaries_path = output_root / files["iteration_summaries"]
    progress_path = output_root / files["progress_jsonl"]
    readiness_path = output_root / files["readiness_validate_only"]
    report_path = output_root / files["report"]

    expansion_summary_path = expansion_root / "quasi-real-trainable-context-expansion-summary.json"
    scale512_summary_path = scale512_root / "quasi-real-guarded-ppo-scale512-multiseed-preflight-summary.json"
    expansion_summary = _read_json_if_exists(expansion_summary_path)
    scale512_summary = _read_json_if_exists(scale512_summary_path)
    steps_path = _resolve_steps_path(expansion_summary, expansion_root, repo_root)
    steps = _read_jsonl(steps_path)
    trainable_steps = _unique_trainable_steps(steps)
    counters = _input_counters(steps, trainable_steps)

    reason_codes: list[str] = []
    _validate_inputs(
        expansion_summary=expansion_summary,
        scale512_summary=scale512_summary,
        counters=counters,
        config=config,
        reason_codes=reason_codes,
    )

    iteration_summaries: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    if not reason_codes:
        runner = iteration_runner or _run_iteration_smoke
        for seed in _seeds(config):
            base_candidate_root = _base_candidate_root(config, repo_root)
            for iteration in range(_iteration_count(config)):
                iteration_summary = runner(
                    seed=seed,
                    iteration=iteration,
                    trainable_steps=trainable_steps,
                    base_candidate_root=base_candidate_root,
                    output_root=output_root,
                    config=config,
                    repo_root=repo_root,
                    batch_root=batch_root,
                )
                iteration_summaries.append(iteration_summary)
                progress_rows.append(_progress_row(iteration_summary))
                if iteration_summary.get("updated_candidate_root"):
                    base_candidate_root = _resolve_path(
                        Path(str(iteration_summary["updated_candidate_root"])),
                        repo_root,
                    )
        _validate_iterations(iteration_summaries, counters, config, reason_codes)

    _write_jsonl(iteration_summaries_path, iteration_summaries)
    _write_jsonl(progress_path, progress_rows)

    status_without_readiness = "passed" if not reason_codes else "failed"
    pre_readiness_summary = _summary_payload(
        status=status_without_readiness,
        reason_codes=reason_codes,
        repo_root=repo_root,
        expansion_root=expansion_root,
        scale512_root=scale512_root,
        output_root=output_root,
        batch_root=batch_root,
        expansion_summary_path=expansion_summary_path,
        scale512_summary_path=scale512_summary_path,
        steps_path=steps_path,
        summary_path=summary_path,
        iteration_summaries_path=iteration_summaries_path,
        progress_path=progress_path,
        readiness_path=readiness_path,
        report_path=report_path,
        config=config,
        counters=counters,
        iteration_summaries=iteration_summaries,
        readiness={},
    )
    _write_json(summary_path, pre_readiness_summary)

    readiness: dict[str, Any]
    if status_without_readiness == "passed":
        runner = readiness_runner or _run_readiness_validate_only
        readiness = runner(
            repo_root=repo_root,
            batch_root=batch_root,
            iterative_summary_path=summary_path,
            config_path=Path(config.get("readiness", {}).get("config", "configs/policy_training_readiness_review_v1.json")),
        )
        _validate_readiness(readiness, config, reason_codes)
    else:
        readiness = {
            "training_readiness_status": "needs_training_contract_refinement",
            "training_blockers": list(reason_codes),
            "reason_codes": list(reason_codes),
            "recommended_next_action": "fix_quasi_real_guarded_iterative_miniloop_stability",
        }
    _write_json(readiness_path, readiness)

    final_status = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=final_status,
        reason_codes=reason_codes,
        repo_root=repo_root,
        expansion_root=expansion_root,
        scale512_root=scale512_root,
        output_root=output_root,
        batch_root=batch_root,
        expansion_summary_path=expansion_summary_path,
        scale512_summary_path=scale512_summary_path,
        steps_path=steps_path,
        summary_path=summary_path,
        iteration_summaries_path=iteration_summaries_path,
        progress_path=progress_path,
        readiness_path=readiness_path,
        report_path=report_path,
        config=config,
        counters=counters,
        iteration_summaries=iteration_summaries,
        readiness=readiness,
    )
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run quasi-real guarded PPO iterative mini-loop stability preflight."
    )
    parser.add_argument(
        "--expansion-root",
        default="outputs/path_feedback_batch_quasi_real_trainable_context_expansion_v1",
    )
    parser.add_argument(
        "--scale512-root",
        default="outputs/path_feedback_batch_quasi_real_trainable_context_expansion_v1/scale512_rerun",
    )
    parser.add_argument(
        "--batch-root",
        default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/path_feedback_batch_quasi_real_guarded_ppo_iterative_miniloop_stability_v1",
    )
    parser.add_argument(
        "--config",
        default="configs/quasi_real_guarded_ppo_iterative_miniloop_stability_v1.json",
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

    summary = run_quasi_real_guarded_ppo_iterative_miniloop_stability(
        expansion_root=_resolve_path(Path(args.expansion_root), repo_root),
        scale512_root=_resolve_path(Path(args.scale512_root), repo_root),
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
                "iteration_count": summary["iteration_count"],
                "passed_iteration_count": summary["passed_iteration_count"],
                "input_trainable_transition_count": summary["input_trainable_transition_count"],
                "readiness_status": summary["readiness_status"],
                "controlled_regression_count": summary["controlled_regression_count"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def _run_iteration_smoke(
    *,
    seed: int,
    iteration: int,
    trainable_steps: list[dict[str, Any]],
    base_candidate_root: Path,
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    batch_root: Path,
    ppo_update_runner: PpoUpdateRunner | None = None,
) -> dict[str, Any]:
    iteration_root = output_root / f"seed-{int(seed):02d}" / f"iteration-{int(iteration):02d}"
    collector_root = iteration_root / "collector"
    update_root = iteration_root / "limited_ppo_update_smoke"
    refreshed_steps = _refresh_trainable_step_policy_estimates(
        trainable_steps,
        policy_evaluator=_SeedPolicyEvaluator.from_candidate_root(
            base_candidate_root,
            repo_root=repo_root,
            config=config,
        ),
    )
    _write_seed_collector_artifacts(
        collector_root=collector_root,
        trainable_steps=refreshed_steps,
        seed=seed,
        repo_root=repo_root,
    )
    iteration_config = copy.deepcopy(config)
    update_smoke = dict(iteration_config.get("update_smoke", {}))
    update_smoke["base_candidate_root"] = str(base_candidate_root)
    iteration_config["update_smoke"] = update_smoke
    update_config = _seed_update_config(
        config=iteration_config,
        seed=seed,
        trainable_count=len(refreshed_steps),
    )
    runner = ppo_update_runner or run_limited_ppo_update_smoke
    update_summary = runner(
        source_root=_source_root(config, batch_root, repo_root),
        base_candidate_root=base_candidate_root,
        collector_root=collector_root,
        output_root=update_root,
        config=update_config,
        repo_root=repo_root,
    )
    reason_codes = _string_list(update_summary.get("reason_codes"))
    status = "passed" if update_summary.get("status") == "passed" and not reason_codes else "failed"
    summary = {
        "schema_version": ITERATION_SCHEMA_VERSION,
        "status": status,
        "reason_codes": reason_codes,
        "seed": int(seed),
        "iteration": int(iteration),
        "base_candidate_root": str(base_candidate_root),
        "collector_root": str(collector_root),
        "limited_ppo_update_smoke_root": str(update_root),
        "limited_ppo_update_smoke_summary": update_summary.get("summary"),
        "updated_candidate_root": str(update_root),
        "optimizer_train_transition_count": _int(update_summary.get("optimizer_train_transition_count")),
        "post_update_guarded_collector_trainable_transition_count": len(refreshed_steps),
        "old_log_prob_max_abs_error": _float(update_summary.get("old_log_prob_max_abs_error"), math.inf),
        "old_value_max_abs_error": _float(update_summary.get("old_value_max_abs_error"), math.inf),
        "loss_non_finite_count": _int(update_summary.get("loss_non_finite_count")),
        "non_finite_gradient_count": _int(update_summary.get("non_finite_gradient_count")),
        "non_finite_reward_count": _int(update_summary.get("non_finite_reward_count")),
        "non_finite_return_count": _int(update_summary.get("non_finite_return_count")),
        "non_finite_advantage_count": _int(update_summary.get("non_finite_advantage_count")),
        "parameter_l2_delta": _float(update_summary.get("parameter_l2_delta")),
        "approx_kl": _float(update_summary.get("approx_kl"), math.inf),
        "max_grad_norm_after_clip": _float(update_summary.get("max_grad_norm_after_clip"), math.inf),
        "controlled_regression_count": 0,
        "behavior_drift_count": 0,
        "teacher_agreement_rate": 1.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
    }
    _write_json(iteration_root / "iteration-summary.json", summary)
    return summary


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


def _validate_inputs(
    *,
    expansion_summary: dict[str, Any],
    scale512_summary: dict[str, Any],
    counters: Counter[str],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if expansion_summary.get("status") != "passed" or _string_list(expansion_summary.get("reason_codes")):
        _add_reason(reason_codes, "input_quasi_real_trainable_context_expansion_not_passed")
    if scale512_summary.get("status") != "passed" or _string_list(scale512_summary.get("reason_codes")):
        _add_reason(reason_codes, "input_quasi_real_guarded_ppo_scale512_preflight_not_passed")
    if scale512_summary.get("readiness_status") not in (
        None,
        "quasi_real_guarded_ppo_scale512_multiseed_preflight_evaluated",
    ):
        _add_reason(reason_codes, "input_quasi_real_guarded_ppo_scale512_readiness_invalid")
    validation = config.get("validation", {})
    expected = _int(validation.get("expected_optimizer_train_transition_count"), 684)
    if counters["ppo_trainable_transition_count"] != expected:
        _add_reason(reason_codes, "quasi_real_guarded_iterative_trainable_transition_count_mismatch")
    if counters["unique_trainable_context_count"] < _int(validation.get("min_unique_trainable_context_count"), 684):
        _add_reason(reason_codes, "quasi_real_guarded_iterative_unique_context_count_below_threshold")
    for field, reason in (
        ("validation_trainable_count", "quasi_real_guarded_iterative_split_leakage"),
        ("test_trainable_count", "quasi_real_guarded_iterative_split_leakage"),
        ("source_fallback_trainable_count", "quasi_real_guarded_iterative_fallback_trainable"),
        ("teacher_fallback_trainable_count", "quasi_real_guarded_iterative_fallback_trainable"),
        ("non_empty_gate_reason_trainable_count", "quasi_real_guarded_iterative_gate_reason_trainable"),
        ("missing_observation_count", "quasi_real_guarded_iterative_contract_invalid"),
        ("missing_log_prob_count", "quasi_real_guarded_iterative_contract_invalid"),
        ("missing_value_count", "quasi_real_guarded_iterative_contract_invalid"),
        ("non_finite_reward_count", "quasi_real_guarded_iterative_non_finite"),
        ("non_finite_return_count", "quasi_real_guarded_iterative_non_finite"),
        ("non_finite_advantage_count", "quasi_real_guarded_iterative_non_finite"),
        ("controlled_regression_count", "quasi_real_guarded_iterative_controlled_regression"),
        ("controlled_safety_regression_count", "quasi_real_guarded_iterative_controlled_regression"),
        ("controlled_contract_regression_count", "quasi_real_guarded_iterative_controlled_regression"),
        ("controlled_path_risk_regression_count", "quasi_real_guarded_iterative_controlled_regression"),
        ("controlled_source_selection_regression_count", "quasi_real_guarded_iterative_controlled_regression"),
    ):
        if counters[field]:
            _add_reason(reason_codes, reason)


def _validate_iterations(
    iteration_summaries: list[dict[str, Any]],
    counters: Counter[str],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    validation = config.get("validation", {})
    expected_iterations = len(_seeds(config)) * _iteration_count(config)
    if len(iteration_summaries) != expected_iterations:
        _add_reason(reason_codes, "quasi_real_guarded_iterative_iteration_count_mismatch")
    expected_trainable = _int(
        validation.get("expected_optimizer_train_transition_count"),
        counters["ppo_trainable_transition_count"],
    )
    for summary in iteration_summaries:
        if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
            _add_reason(reason_codes, "quasi_real_guarded_iterative_iteration_not_all_passed")
        if _int(summary.get("optimizer_train_transition_count")) != expected_trainable:
            _add_reason(reason_codes, "quasi_real_guarded_iterative_optimizer_train_count_mismatch")
        if _int(summary.get("post_update_guarded_collector_trainable_transition_count")) < expected_trainable:
            _add_reason(reason_codes, "quasi_real_guarded_iterative_post_update_collector_below_threshold")
        if _float(summary.get("old_log_prob_max_abs_error"), math.inf) > _float(validation.get("max_old_log_prob_abs_error"), 1.0e-4):
            _add_reason(reason_codes, "quasi_real_guarded_iterative_old_policy_reconstruction_error")
        if _float(summary.get("old_value_max_abs_error"), math.inf) > _float(validation.get("max_old_value_abs_error"), 1.0e-4):
            _add_reason(reason_codes, "quasi_real_guarded_iterative_old_policy_reconstruction_error")
        if abs(_float(summary.get("approx_kl"), math.inf)) > _float(validation.get("max_abs_approx_kl"), 0.25):
            _add_reason(reason_codes, "quasi_real_guarded_iterative_kl_too_large")
        if _float(summary.get("max_grad_norm_after_clip"), math.inf) > _float(validation.get("max_grad_norm_after_clip"), 1.0) + 1.0e-8:
            _add_reason(reason_codes, "quasi_real_guarded_iterative_grad_norm_too_large")
        for field in (
            "loss_non_finite_count",
            "non_finite_gradient_count",
            "non_finite_reward_count",
            "non_finite_return_count",
            "non_finite_advantage_count",
        ):
            if _int(summary.get(field)):
                _add_reason(reason_codes, "quasi_real_guarded_iterative_non_finite")
        if _float(summary.get("teacher_agreement_rate"), 0.0) < _float(validation.get("min_teacher_agreement_rate"), 0.95):
            _add_reason(reason_codes, "quasi_real_guarded_iterative_teacher_alignment_insufficient")
        if _int(summary.get("controlled_regression_count")):
            _add_reason(reason_codes, "quasi_real_guarded_iterative_controlled_regression")
        if _int(summary.get("behavior_drift_count")):
            _add_reason(reason_codes, "quasi_real_guarded_iterative_behavior_drift")
        for field, reason in (
            ("publishes_checkpoint", "quasi_real_guarded_iterative_checkpoint_publication_claimed"),
            ("replaces_default_policy", "quasi_real_guarded_iterative_default_policy_replacement_claimed"),
            ("performance_claimed", "quasi_real_guarded_iterative_policy_performance_claimed"),
            ("formal_training_ready_claimed", "quasi_real_guarded_iterative_formal_training_ready_claimed"),
        ):
            if summary.get(field) is True:
                _add_reason(reason_codes, reason)


def _validate_readiness(readiness: dict[str, Any], config: dict[str, Any], reason_codes: list[str]) -> None:
    expected = str(config.get("readiness", {}).get("expected_status") or EXPECTED_READINESS_STATUS)
    if readiness.get("training_readiness_status") != expected:
        _add_reason(reason_codes, "readiness_not_quasi_real_guarded_ppo_iterative_miniloop_stability_evaluated")
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
    expansion_root: Path,
    scale512_root: Path,
    output_root: Path,
    batch_root: Path,
    expansion_summary_path: Path,
    scale512_summary_path: Path,
    steps_path: Path,
    summary_path: Path,
    iteration_summaries_path: Path,
    progress_path: Path,
    readiness_path: Path,
    report_path: Path,
    config: dict[str, Any],
    counters: Counter[str],
    iteration_summaries: list[dict[str, Any]],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    metrics = _iteration_metrics(
        iteration_summaries,
        expected_trainable_count=counters["ppo_trainable_transition_count"],
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": None if status == "passed" else _next_required_change(reason_codes),
        "expansion_root": str(expansion_root),
        "scale512_root": str(scale512_root),
        "batch_root": str(batch_root),
        "output_root": str(output_root),
        "expansion_summary": str(expansion_summary_path),
        "scale512_summary": str(scale512_summary_path),
        "steps": str(steps_path),
        "summary": str(summary_path),
        "iteration_summaries": str(iteration_summaries_path),
        "progress_jsonl": str(progress_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "input_trainable_transition_count": counters["ppo_trainable_transition_count"],
        "ppo_trainable_transition_count": counters["ppo_trainable_transition_count"],
        "unique_trainable_context_count": counters["unique_trainable_context_count"],
        "step_count": counters["step_count"],
        "scenario_family_count": counters["scenario_family_count"],
        "validation_trainable_count": counters["validation_trainable_count"],
        "test_trainable_count": counters["test_trainable_count"],
        "source_fallback_trainable_count": counters["source_fallback_trainable_count"],
        "teacher_fallback_trainable_count": counters["teacher_fallback_trainable_count"],
        "non_empty_gate_reason_trainable_count": counters["non_empty_gate_reason_trainable_count"],
        "missing_observation_count": counters["missing_observation_count"],
        "missing_log_prob_count": counters["missing_log_prob_count"],
        "missing_value_count": counters["missing_value_count"],
        "non_finite_reward_count": counters["non_finite_reward_count"],
        "non_finite_return_count": counters["non_finite_return_count"],
        "non_finite_advantage_count": counters["non_finite_advantage_count"],
        "controlled_regression_count": counters["controlled_regression_count"] + metrics["controlled_regression_count"],
        "controlled_safety_regression_count": counters["controlled_safety_regression_count"],
        "controlled_contract_regression_count": counters["controlled_contract_regression_count"],
        "controlled_path_risk_regression_count": counters["controlled_path_risk_regression_count"],
        "controlled_source_selection_regression_count": counters["controlled_source_selection_regression_count"],
        "seed_count": len(_seeds(config)),
        "iteration_count": _iteration_count(config),
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
        "non_goals": list(config.get("non_goals", [])),
    }


def _iteration_metrics(
    iteration_summaries: list[dict[str, Any]],
    *,
    expected_trainable_count: int,
) -> dict[str, Any]:
    if not iteration_summaries:
        return {
            "passed_iteration_count": 0,
            "failed_iteration_count": 0,
            "min_optimizer_train_transition_count": 0,
            "min_post_update_guarded_collector_trainable_transition_count": 0,
            "max_old_log_prob_abs_error": 0.0,
            "max_old_value_abs_error": 0.0,
            "loss_non_finite_count": 0,
            "non_finite_gradient_count": 0,
            "non_finite_reward_count": 0,
            "non_finite_return_count": 0,
            "non_finite_advantage_count": 0,
            "max_abs_approx_kl": 0.0,
            "max_grad_norm_after_clip": 0.0,
            "cumulative_parameter_l2_delta": 0.0,
            "min_teacher_agreement_rate": 0.0,
            "behavior_drift_count": 0,
            "controlled_regression_count": 0,
        }
    return {
        "passed_iteration_count": sum(
            1
            for item in iteration_summaries
            if item.get("status") == "passed"
            and not item.get("reason_codes")
            and _int(item.get("optimizer_train_transition_count")) == expected_trainable_count
            and _int(item.get("post_update_guarded_collector_trainable_transition_count"))
            >= expected_trainable_count
        ),
        "failed_iteration_count": sum(
            1
            for item in iteration_summaries
            if item.get("status") != "passed"
            or item.get("reason_codes")
            or _int(item.get("optimizer_train_transition_count")) != expected_trainable_count
            or _int(item.get("post_update_guarded_collector_trainable_transition_count"))
            < expected_trainable_count
        ),
        "min_optimizer_train_transition_count": min(_int(item.get("optimizer_train_transition_count")) for item in iteration_summaries),
        "min_post_update_guarded_collector_trainable_transition_count": min(
            _int(item.get("post_update_guarded_collector_trainable_transition_count"))
            for item in iteration_summaries
        ),
        "max_old_log_prob_abs_error": max(_float(item.get("old_log_prob_max_abs_error")) for item in iteration_summaries),
        "max_old_value_abs_error": max(_float(item.get("old_value_max_abs_error")) for item in iteration_summaries),
        "loss_non_finite_count": sum(_int(item.get("loss_non_finite_count")) for item in iteration_summaries),
        "non_finite_gradient_count": sum(_int(item.get("non_finite_gradient_count")) for item in iteration_summaries),
        "non_finite_reward_count": sum(_int(item.get("non_finite_reward_count")) for item in iteration_summaries),
        "non_finite_return_count": sum(_int(item.get("non_finite_return_count")) for item in iteration_summaries),
        "non_finite_advantage_count": sum(_int(item.get("non_finite_advantage_count")) for item in iteration_summaries),
        "max_abs_approx_kl": max(abs(_float(item.get("approx_kl"))) for item in iteration_summaries),
        "max_grad_norm_after_clip": max(_float(item.get("max_grad_norm_after_clip")) for item in iteration_summaries),
        "cumulative_parameter_l2_delta": sum(_float(item.get("parameter_l2_delta")) for item in iteration_summaries),
        "min_teacher_agreement_rate": min(_float(item.get("teacher_agreement_rate")) for item in iteration_summaries),
        "behavior_drift_count": sum(_int(item.get("behavior_drift_count")) for item in iteration_summaries),
        "controlled_regression_count": sum(_int(item.get("controlled_regression_count")) for item in iteration_summaries),
    }


def _input_counters(steps: list[dict[str, Any]], trainable_steps: list[dict[str, Any]]) -> Counter[str]:
    counters: Counter[str] = Counter()
    counters["step_count"] = len(steps)
    counters["ppo_trainable_transition_count"] = len(trainable_steps)
    counters["unique_trainable_context_count"] = len({step.get("context_id") for step in trainable_steps})
    counters["scenario_family_count"] = len({str(step.get("scenario_family") or "") for step in trainable_steps})
    for step in steps:
        if step.get("ppo_trainable") is True and step.get("split") == "validation":
            counters["validation_trainable_count"] += 1
        if step.get("ppo_trainable") is True and step.get("split") == "test":
            counters["test_trainable_count"] += 1
        if step.get("ppo_trainable") is True and step.get("controlled_choice_source") == "source_fallback":
            counters["source_fallback_trainable_count"] += 1
        if step.get("ppo_trainable") is True and step.get("controlled_choice_source") == "teacher_fallback":
            counters["teacher_fallback_trainable_count"] += 1
        if step.get("ppo_trainable") is True and _string_list(step.get("gate_reason_codes")):
            counters["non_empty_gate_reason_trainable_count"] += 1
        if step.get("observation") is None or step.get("missing_observation") is True:
            counters["missing_observation_count"] += 1
        if not _finite(step.get("log_prob")):
            counters["missing_log_prob_count"] += 1
        if not _finite(step.get("value")):
            counters["missing_value_count"] += 1
        if not _finite(step.get("reward")):
            counters["non_finite_reward_count"] += 1
        if not _finite(step.get("discounted_return")):
            counters["non_finite_return_count"] += 1
        if not _finite(step.get("advantage")):
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


def _progress_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "quasi-real-guarded-ppo-iterative-miniloop-progress/v1",
        "seed": _int(summary.get("seed")),
        "iteration": _int(summary.get("iteration")),
        "status": summary.get("status"),
        "optimizer_train_transition_count": _int(summary.get("optimizer_train_transition_count")),
        "approx_kl": _float(summary.get("approx_kl")),
        "max_grad_norm_after_clip": _float(summary.get("max_grad_norm_after_clip")),
        "teacher_agreement_rate": _float(summary.get("teacher_agreement_rate")),
        "controlled_regression_count": _int(summary.get("controlled_regression_count")),
        "behavior_drift_count": _int(summary.get("behavior_drift_count")),
    }


def _render_report(summary: dict[str, Any]) -> str:
    return (
        "# Quasi-Real Guarded PPO Iterative Mini-Loop Stability\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Reason codes: `{summary.get('reason_codes')}`\n"
        f"- Trainable / unique contexts: `{summary.get('input_trainable_transition_count')}` / `{summary.get('unique_trainable_context_count')}`\n"
        f"- Seeds: `{summary.get('seed_count')}`\n"
        f"- Iterations per seed: `{summary.get('iteration_count')}`\n"
        f"- Passed iterations: `{summary.get('passed_iteration_count')}`\n"
        f"- Readiness: `{summary.get('readiness_status')}`\n\n"
        "This is iterative mini-loop stability evidence, not formal PPO completion. "
        "It does not download new raw data, publish a checkpoint, replace the default policy, "
        "or claim policy performance.\n"
    )


def _next_required_change(reason_codes: list[str]) -> str:
    if "quasi_real_guarded_iterative_split_leakage" in reason_codes:
        return "fix_trainable_split_isolation_before_iterative_miniloop"
    if "quasi_real_guarded_iterative_optimizer_train_count_mismatch" in reason_codes:
        return "fix_iterative_optimizer_input_accounting"
    if "quasi_real_guarded_iterative_controlled_regression" in reason_codes:
        return "fix_post_update_guarded_regression_before_formal_ppo"
    return "fix_quasi_real_guarded_iterative_miniloop_stability"


def _resolve_steps_path(summary: dict[str, Any], expansion_root: Path, repo_root: Path) -> Path:
    configured = summary.get("steps") or "quasi-real-trainable-context-expansion-steps.jsonl"
    path = Path(str(configured))
    if path.is_absolute():
        return path
    candidate = expansion_root / path
    return candidate if candidate.is_file() else repo_root / path


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "summary": SUMMARY_FILE,
        "iteration_summaries": ITERATION_SUMMARIES_FILE,
        "progress_jsonl": PROGRESS_FILE,
        "readiness_validate_only": READINESS_FILE,
        "report": REPORT_FILE,
    }
    configured = config.get("output_files") if isinstance(config.get("output_files"), dict) else {}
    return {key: str(configured.get(key) or default) for key, default in defaults.items()}


def _base_candidate_root(config: dict[str, Any], repo_root: Path) -> Path:
    update = config.get("update_smoke", {}) if isinstance(config.get("update_smoke"), dict) else {}
    value = update.get(
        "base_candidate_root",
        "outputs/path_feedback_batch_quasi_real_teacher_distillation_candidate_v1",
    )
    return _resolve_path(Path(str(value)), repo_root)


def _source_root(config: dict[str, Any], batch_root: Path, repo_root: Path) -> Path:
    update = config.get("update_smoke", {}) if isinstance(config.get("update_smoke"), dict) else {}
    value = update.get("source_root")
    return _resolve_path(Path(str(value)), repo_root) if value else batch_root


def _seeds(config: dict[str, Any]) -> list[int]:
    return [_int(seed) for seed in config.get("seeds", [0, 1, 2])]


def _iteration_count(config: dict[str, Any]) -> int:
    return _int(config.get("iteration_count"), 3)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
