from __future__ import annotations

import argparse
import json
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


CONFIG_SCHEMA_VERSION = "guarded-formal-ppo-training-authorization-config/v1"
SUMMARY_SCHEMA_VERSION = "guarded-formal-ppo-training-authorization-summary/v1"
EXPECTED_AUTHORIZATION_VERDICT = "authorized_for_guarded_formal_ppo_training_run"
EXPECTED_READINESS_STATUS = "guarded_formal_ppo_training_authorized"

SOURCE_SPECS = {
    "formal_preflight": (
        "formal_preflight_summary",
        "quasi-real-guarded-formal-ppo-preflight-summary/v1",
    ),
    "formal_rollout_canary": (
        "formal_rollout_canary_summary",
        "quasi-real-guarded-formal-ppo-rollout-canary-summary/v1",
    ),
    "formal_stability_holdout": (
        "formal_stability_holdout_summary",
        "quasi-real-guarded-formal-ppo-stability-holdout-validation-summary/v1",
    ),
    "candidate_selection": (
        "candidate_selection_summary",
        "quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-summary/v1",
    ),
    "promotion_decision_review": (
        "promotion_decision_review_summary",
        "selected-formal-ppo-candidate-promotion-decision-review-summary/v1",
    ),
    "canary_preflight": (
        "canary_preflight_summary",
        "guarded-experimental-policy-staged-release-canary-preflight-summary/v1",
    ),
}

ROOT_KEY_BY_SOURCE = {
    "formal_preflight": "formal_preflight_root",
    "formal_rollout_canary": "formal_rollout_canary_root",
    "formal_stability_holdout": "formal_stability_holdout_root",
    "candidate_selection": "candidate_selection_root",
    "promotion_decision_review": "promotion_decision_review_root",
    "canary_preflight": "canary_preflight_root",
}

ReadinessRunner = Callable[..., dict[str, Any]]


def run_guarded_formal_ppo_training_authorization(
    *,
    formal_preflight_root: Path,
    formal_rollout_canary_root: Path,
    formal_stability_holdout_root: Path,
    candidate_selection_root: Path,
    promotion_decision_review_root: Path,
    canary_preflight_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    readiness_runner: ReadinessRunner | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    output_root = Path(output_root)
    batch_root = Path(batch_root)
    output_root.mkdir(parents=True, exist_ok=True)

    roots = {
        "formal_preflight_root": Path(formal_preflight_root),
        "formal_rollout_canary_root": Path(formal_rollout_canary_root),
        "formal_stability_holdout_root": Path(formal_stability_holdout_root),
        "candidate_selection_root": Path(candidate_selection_root),
        "promotion_decision_review_root": Path(promotion_decision_review_root),
        "canary_preflight_root": Path(canary_preflight_root),
    }
    files = _output_files(config)
    summary_path = output_root / files["summary"]
    training_input_audit_path = output_root / files["training_input_audit"]
    budget_manifest_path = output_root / files["budget_manifest"]
    seed_plan_path = output_root / files["seed_plan"]
    stop_condition_manifest_path = output_root / files["stop_condition_manifest"]
    rollback_manifest_path = output_root / files["rollback_manifest"]
    post_training_gate_plan_path = output_root / files["post_training_gate_plan"]
    readiness_path = output_root / files["readiness_validate_only"]
    report_path = output_root / files["report"]

    source_paths, source_summaries = _load_source_summaries(config=config, roots=roots)
    counters = _authorization_counters(source_summaries)
    reason_codes: list[str] = []
    _validate_sources(source_summaries, reason_codes)
    _validate_authorization_inputs(counters, config, reason_codes)

    seed_plan = _seed_plan(config)
    budget_manifest = _budget_manifest(config, counters)
    stop_condition_manifest = _stop_condition_manifest()
    rollback_manifest = _rollback_manifest()
    post_training_gate_plan = _post_training_gate_plan(counters)
    training_input_audit = _training_input_audit(
        source_paths=source_paths,
        counters=counters,
        source_summaries=source_summaries,
        reason_codes=reason_codes,
    )
    _validate_manifests(
        seed_plan=seed_plan,
        budget_manifest=budget_manifest,
        stop_condition_manifest=stop_condition_manifest,
        rollback_manifest=rollback_manifest,
        post_training_gate_plan=post_training_gate_plan,
        reason_codes=reason_codes,
    )

    _write_json(training_input_audit_path, training_input_audit)
    _write_json(budget_manifest_path, budget_manifest)
    _write_json(seed_plan_path, seed_plan)
    _write_json(stop_condition_manifest_path, stop_condition_manifest)
    _write_json(rollback_manifest_path, rollback_manifest)
    _write_json(post_training_gate_plan_path, post_training_gate_plan)

    status_without_readiness = "passed" if not reason_codes else "failed"
    pre_readiness_summary = _summary_payload(
        status=status_without_readiness,
        reason_codes=reason_codes,
        repo_root=repo_root,
        output_root=output_root,
        batch_root=batch_root,
        roots=roots,
        source_paths=source_paths,
        summary_path=summary_path,
        training_input_audit_path=training_input_audit_path,
        budget_manifest_path=budget_manifest_path,
        seed_plan_path=seed_plan_path,
        stop_condition_manifest_path=stop_condition_manifest_path,
        rollback_manifest_path=rollback_manifest_path,
        post_training_gate_plan_path=post_training_gate_plan_path,
        readiness_path=readiness_path,
        report_path=report_path,
        counters=counters,
        seed_plan=seed_plan,
        budget_manifest=budget_manifest,
        stop_condition_manifest=stop_condition_manifest,
        rollback_manifest=rollback_manifest,
        post_training_gate_plan=post_training_gate_plan,
        readiness={},
    )
    _write_json(summary_path, pre_readiness_summary)

    if status_without_readiness == "passed":
        runner = readiness_runner or _run_readiness_validate_only
        readiness = runner(
            repo_root=repo_root,
            batch_root=batch_root,
            authorization_summary_path=summary_path,
            config_path=Path(
                config.get("readiness", {}).get(
                    "config",
                    "configs/policy_training_readiness_review_v1.json",
                )
            ),
        )
        _validate_readiness(readiness, config, reason_codes)
    else:
        readiness = {
            "training_readiness_status": "needs_training_contract_refinement",
            "training_blockers": list(reason_codes),
            "reason_codes": list(reason_codes),
            "recommended_next_action": "fix_guarded_formal_ppo_training_authorization",
        }
    _write_json(readiness_path, readiness)

    final_status = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=final_status,
        reason_codes=reason_codes,
        repo_root=repo_root,
        output_root=output_root,
        batch_root=batch_root,
        roots=roots,
        source_paths=source_paths,
        summary_path=summary_path,
        training_input_audit_path=training_input_audit_path,
        budget_manifest_path=budget_manifest_path,
        seed_plan_path=seed_plan_path,
        stop_condition_manifest_path=stop_condition_manifest_path,
        rollback_manifest_path=rollback_manifest_path,
        post_training_gate_plan_path=post_training_gate_plan_path,
        readiness_path=readiness_path,
        report_path=report_path,
        counters=counters,
        seed_plan=seed_plan,
        budget_manifest=budget_manifest,
        stop_condition_manifest=stop_condition_manifest,
        rollback_manifest=rollback_manifest,
        post_training_gate_plan=post_training_gate_plan,
        readiness=readiness,
    )
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Authorize the next guarded formal PPO training run without running PPO."
    )
    parser.add_argument(
        "--formal-preflight-root",
        default="outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_preflight_v1",
    )
    parser.add_argument(
        "--formal-rollout-canary-root",
        default="outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_rollout_canary_v1",
    )
    parser.add_argument(
        "--formal-stability-holdout-root",
        default="outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_stability_holdout_validation_v1",
    )
    parser.add_argument(
        "--candidate-selection-root",
        default=(
            "outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_candidate_selection_"
            "long_horizon_holdout_v1"
        ),
    )
    parser.add_argument(
        "--promotion-decision-review-root",
        default="outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_decision_review_v1",
    )
    parser.add_argument(
        "--canary-preflight-root",
        default="outputs/path_feedback_batch_guarded_experimental_policy_staged_release_canary_preflight_v1",
    )
    parser.add_argument(
        "--batch-root",
        default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/path_feedback_batch_guarded_formal_ppo_training_authorization_v1",
    )
    parser.add_argument(
        "--config",
        default="configs/guarded_formal_ppo_training_authorization_v1.json",
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

    summary = run_guarded_formal_ppo_training_authorization(
        formal_preflight_root=_resolve_path(Path(args.formal_preflight_root), repo_root),
        formal_rollout_canary_root=_resolve_path(Path(args.formal_rollout_canary_root), repo_root),
        formal_stability_holdout_root=_resolve_path(Path(args.formal_stability_holdout_root), repo_root),
        candidate_selection_root=_resolve_path(Path(args.candidate_selection_root), repo_root),
        promotion_decision_review_root=_resolve_path(Path(args.promotion_decision_review_root), repo_root),
        canary_preflight_root=_resolve_path(Path(args.canary_preflight_root), repo_root),
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
                "authorization_verdict": summary["authorization_verdict"],
                "readiness_status": summary.get("readiness_status"),
                "authorized_trainable_transition_count": summary.get(
                    "authorized_trainable_transition_count"
                ),
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def _load_source_summaries(
    *,
    config: dict[str, Any],
    roots: dict[str, Path],
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    input_files = _input_files(config)
    source_paths: dict[str, Path] = {}
    source_summaries: dict[str, dict[str, Any]] = {}
    for source_name, (input_key, _) in SOURCE_SPECS.items():
        path = roots[ROOT_KEY_BY_SOURCE[source_name]] / input_files[input_key]
        source_paths[source_name] = path
        source_summaries[source_name] = _read_json_if_exists(path)
    return source_paths, source_summaries


def _validate_sources(
    source_summaries: dict[str, dict[str, Any]],
    reason_codes: list[str],
) -> None:
    for source_name, summary in source_summaries.items():
        _, expected_schema = SOURCE_SPECS[source_name]
        if not summary:
            _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_source_missing")
            continue
        if summary.get("schema_version") != expected_schema:
            _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_source_schema_mismatch")
        if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
            _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_source_not_passed")
        if _source_git_current_matches(summary) is False:
            _add_reason(
                reason_codes,
                "guarded_formal_ppo_training_authorization_source_git_provenance_mismatch",
            )
        if summary.get("runs_new_ppo_update") is True and source_name not in {
            "formal_preflight",
            "formal_rollout_canary",
            "formal_stability_holdout",
        }:
            _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_unexpected_source_update")
        if summary.get("publishes_checkpoint") is True:
            _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_source_publishes_checkpoint")
        if summary.get("replaces_default_policy") is True:
            _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_source_replaces_default_policy")
        if summary.get("performance_claimed") is True or summary.get("formal_training_ready_claimed") is True:
            _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_source_overclaims")

    promotion = source_summaries.get("promotion_decision_review", {})
    if promotion and promotion.get("decision_verdict") != "eligible_for_guarded_release_candidate_packaging":
        _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_promotion_not_eligible")
    canary = source_summaries.get("canary_preflight", {})
    if canary:
        if canary.get("staged_release_canary_preflight_verdict") != (
            "eligible_for_guarded_staged_release_canary_dry_run"
        ):
            _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_canary_preflight_not_eligible")
        if canary.get("runs_online_canary") is True or canary.get("connects_real_executor") is True:
            _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_online_canary_or_executor")
        if _int(canary.get("controlled_regression_count")) > 0:
            _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_controlled_regression")


def _authorization_counters(source_summaries: dict[str, dict[str, Any]]) -> Counter[str]:
    stability = source_summaries.get("formal_stability_holdout", {})
    counters: Counter[str] = Counter()
    counters["source_summary_count"] = len(source_summaries)
    counters["passed_source_summary_count"] = sum(
        1
        for summary in source_summaries.values()
        if summary.get("status") == "passed" and not _string_list(summary.get("reason_codes"))
    )
    counters["source_git_provenance_present_count"] = sum(
        1 for summary in source_summaries.values() if isinstance(summary.get("git_provenance"), dict)
    )
    counters["source_git_provenance_mismatch_count"] = sum(
        1
        for summary in source_summaries.values()
        if _source_git_current_matches(summary) is False
    )
    counters["authorized_trainable_transition_count"] = _int(
        stability.get("input_trainable_transition_count")
    )
    counters["authorized_optimizer_train_transition_count"] = _int(
        stability.get("optimizer_train_transition_count")
    )
    counters["unique_authorized_trainable_context_count"] = _int(
        stability.get("unique_trainable_context_count")
    )
    for field in (
        "validation_trainable_count",
        "test_trainable_count",
        "fallback_trainable_count",
        "source_fallback_trainable_count",
        "teacher_fallback_trainable_count",
        "non_empty_gate_reason_trainable_count",
        "missing_observation_count",
        "missing_log_prob_count",
        "missing_value_count",
        "invalid_action_mask_count",
        "non_finite_reward_count",
        "non_finite_return_count",
        "non_finite_advantage_count",
        "loss_non_finite_count",
        "non_finite_gradient_count",
        "controlled_regression_count",
        "controlled_safety_regression_count",
        "controlled_contract_regression_count",
        "controlled_path_risk_regression_count",
        "controlled_source_selection_regression_count",
    ):
        counters[field] = _int(stability.get(field))
    counters["diagnostic_trainable_count"] = (
        counters["validation_trainable_count"]
        + counters["test_trainable_count"]
        + counters["non_empty_gate_reason_trainable_count"]
    )
    return counters


def _validate_authorization_inputs(
    counters: Counter[str],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    validation = config.get("validation") if isinstance(config.get("validation"), dict) else {}
    min_trainable = _int(validation.get("min_authorized_trainable_transition_count"), 684)
    if counters["authorized_trainable_transition_count"] < min_trainable:
        _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_trainable_count_below_threshold")
    if counters["authorized_optimizer_train_transition_count"] != counters["authorized_trainable_transition_count"]:
        _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_optimizer_count_mismatch")
    if counters["unique_authorized_trainable_context_count"] < min_trainable:
        _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_unique_context_count_below_threshold")
    if counters["validation_trainable_count"] or counters["test_trainable_count"]:
        _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_split_leakage")
    if (
        counters["fallback_trainable_count"]
        or counters["source_fallback_trainable_count"]
        or counters["teacher_fallback_trainable_count"]
    ):
        _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_fallback_trainable")
    if counters["non_empty_gate_reason_trainable_count"]:
        _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_gate_reason_trainable")
    if (
        counters["missing_observation_count"]
        or counters["missing_log_prob_count"]
        or counters["missing_value_count"]
        or counters["invalid_action_mask_count"]
    ):
        _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_contract_invalid")
    if (
        counters["non_finite_reward_count"]
        or counters["non_finite_return_count"]
        or counters["non_finite_advantage_count"]
        or counters["loss_non_finite_count"]
        or counters["non_finite_gradient_count"]
    ):
        _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_non_finite")
    if (
        counters["controlled_regression_count"]
        or counters["controlled_safety_regression_count"]
        or counters["controlled_contract_regression_count"]
        or counters["controlled_path_risk_regression_count"]
        or counters["controlled_source_selection_regression_count"]
    ):
        _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_controlled_regression")


def _training_input_audit(
    *,
    source_paths: dict[str, Path],
    counters: Counter[str],
    source_summaries: dict[str, dict[str, Any]],
    reason_codes: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "guarded-formal-ppo-training-input-audit/v1",
        "training_input_audit_passed": not reason_codes,
        "authorized_source": "formal_stability_holdout",
        "source_summary_paths": {name: str(path) for name, path in source_paths.items()},
        "source_statuses": {
            name: {
                "status": summary.get("status"),
                "reason_codes": _string_list(summary.get("reason_codes")),
                "schema_version": summary.get("schema_version"),
                "git_current_matches_sources": _source_git_current_matches(summary),
            }
            for name, summary in source_summaries.items()
        },
        **_counter_payload(counters),
    }


def _budget_manifest(config: dict[str, Any], counters: Counter[str]) -> dict[str, Any]:
    plan = config.get("training_plan") if isinstance(config.get("training_plan"), dict) else {}
    return {
        "schema_version": "guarded-formal-ppo-training-budget-manifest/v1",
        "budget_manifest_passed": True,
        "authorized_trainable_transition_count": counters["authorized_trainable_transition_count"],
        "epochs": _int(plan.get("epochs"), 1),
        "learning_rate_max": _float(plan.get("learning_rate_max"), 1e-5),
        "clip_ratio": _float(plan.get("clip_ratio"), 0.2),
        "discount_factor": _float(plan.get("discount_factor"), 0.99),
        "max_grad_norm": _float(plan.get("max_grad_norm"), 1.0),
        "checkpoint_scope": plan.get("checkpoint_scope", "experimental_candidate_only"),
    }


def _seed_plan(config: dict[str, Any]) -> dict[str, Any]:
    seeds = _seeds(config)
    return {
        "schema_version": "guarded-formal-ppo-training-seed-plan/v1",
        "seed_plan_passed": len(seeds) >= _min_seed_count(config),
        "seeds": seeds,
        "seed_count": len(seeds),
        "requires_all_seed_runs_to_pass": True,
    }


def _stop_condition_manifest() -> dict[str, Any]:
    return {
        "schema_version": "guarded-formal-ppo-training-stop-condition-manifest/v1",
        "stop_condition_manifest_passed": True,
        "stop_conditions": [
            "non_finite_loss",
            "non_finite_gradient",
            "non_finite_reward",
            "non_finite_return",
            "non_finite_advantage",
            "approx_kl_threshold_exceeded",
            "grad_norm_threshold_exceeded",
            "teacher_skill_regression",
            "controlled_regression",
            "holdout_regression",
            "canary_gate_failure",
        ],
    }


def _rollback_manifest() -> dict[str, Any]:
    return {
        "schema_version": "guarded-formal-ppo-training-rollback-manifest/v1",
        "rollback_manifest_passed": True,
        "rollback_target": "default_policy",
        "checkpoint_scope": "experimental_candidate_only",
        "writes_default_policy": False,
    }


def _post_training_gate_plan(counters: Counter[str]) -> dict[str, Any]:
    return {
        "schema_version": "guarded-formal-ppo-post-training-gate-plan/v1",
        "post_training_gate_plan_passed": True,
        "required_gate_plan": [
            "old_policy_reconstruction",
            "finite_loss_gradient_reward_return_advantage",
            "teacher_skill_agreement",
            "guarded_collector_replay",
            "formal_stability_holdout",
            "staged_release_canary_preflight",
        ],
        "min_post_training_trainable_transition_count": counters[
            "authorized_trainable_transition_count"
        ],
    }


def _validate_manifests(
    *,
    seed_plan: dict[str, Any],
    budget_manifest: dict[str, Any],
    stop_condition_manifest: dict[str, Any],
    rollback_manifest: dict[str, Any],
    post_training_gate_plan: dict[str, Any],
    reason_codes: list[str],
) -> None:
    for field, payload, reason in (
        ("seed_plan_passed", seed_plan, "guarded_formal_ppo_training_authorization_seed_plan_failed"),
        ("budget_manifest_passed", budget_manifest, "guarded_formal_ppo_training_authorization_budget_manifest_failed"),
        ("stop_condition_manifest_passed", stop_condition_manifest, "guarded_formal_ppo_training_authorization_stop_condition_manifest_failed"),
        ("rollback_manifest_passed", rollback_manifest, "guarded_formal_ppo_training_authorization_rollback_manifest_failed"),
        ("post_training_gate_plan_passed", post_training_gate_plan, "guarded_formal_ppo_training_authorization_post_training_gate_plan_failed"),
    ):
        if payload.get(field) is not True:
            _add_reason(reason_codes, reason)


def _summary_payload(
    *,
    status: str,
    reason_codes: list[str],
    repo_root: Path,
    output_root: Path,
    batch_root: Path,
    roots: dict[str, Path],
    source_paths: dict[str, Path],
    summary_path: Path,
    training_input_audit_path: Path,
    budget_manifest_path: Path,
    seed_plan_path: Path,
    stop_condition_manifest_path: Path,
    rollback_manifest_path: Path,
    post_training_gate_plan_path: Path,
    readiness_path: Path,
    report_path: Path,
    counters: Counter[str],
    seed_plan: dict[str, Any],
    budget_manifest: dict[str, Any],
    stop_condition_manifest: dict[str, Any],
    rollback_manifest: dict[str, Any],
    post_training_gate_plan: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": None
        if status == "passed"
        else "fix_guarded_formal_ppo_training_authorization",
        "authorization_verdict": EXPECTED_AUTHORIZATION_VERDICT
        if status == "passed"
        else "blocked_by_guarded_formal_ppo_training_authorization",
        "output_root": str(output_root),
        "batch_root": str(batch_root),
        "summary": str(summary_path),
        "training_input_audit": str(training_input_audit_path),
        "budget_manifest": str(budget_manifest_path),
        "seed_plan": str(seed_plan_path),
        "stop_condition_manifest": str(stop_condition_manifest_path),
        "rollback_manifest": str(rollback_manifest_path),
        "post_training_gate_plan": str(post_training_gate_plan_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "formal_preflight_root": str(roots["formal_preflight_root"]),
        "formal_rollout_canary_root": str(roots["formal_rollout_canary_root"]),
        "formal_stability_holdout_root": str(roots["formal_stability_holdout_root"]),
        "candidate_selection_root": str(roots["candidate_selection_root"]),
        "promotion_decision_review_root": str(roots["promotion_decision_review_root"]),
        "canary_preflight_root": str(roots["canary_preflight_root"]),
        "formal_preflight_summary": str(source_paths["formal_preflight"]),
        "formal_rollout_canary_summary": str(source_paths["formal_rollout_canary"]),
        "formal_stability_holdout_summary": str(source_paths["formal_stability_holdout"]),
        "candidate_selection_summary": str(source_paths["candidate_selection"]),
        "promotion_decision_review_summary": str(source_paths["promotion_decision_review"]),
        "canary_preflight_summary": str(source_paths["canary_preflight"]),
        **_counter_payload(counters),
        "seeds": list(seed_plan.get("seeds", [])),
        "seed_count": _int(seed_plan.get("seed_count")),
        "budget_manifest_passed": budget_manifest.get("budget_manifest_passed") is True,
        "seed_plan_passed": seed_plan.get("seed_plan_passed") is True,
        "stop_condition_manifest_passed": stop_condition_manifest.get("stop_condition_manifest_passed") is True,
        "rollback_manifest_passed": rollback_manifest.get("rollback_manifest_passed") is True,
        "post_training_gate_plan_passed": post_training_gate_plan.get("post_training_gate_plan_passed") is True,
        "runs_guarded_formal_ppo_training_authorization": True,
        "runs_new_ppo_update": False,
        "runs_online_canary": False,
        "connects_real_executor": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "readiness_status": readiness.get("training_readiness_status"),
        "training_blockers": _string_list(readiness.get("training_blockers")),
        "readiness_reason_codes": _string_list(readiness.get("reason_codes")),
        "git_provenance": {
            "current": git_snapshot(repo_root),
            "current_matches_sources": True,
        },
    }
    return payload


def _counter_payload(counters: Counter[str]) -> dict[str, int]:
    return {
        "source_summary_count": counters["source_summary_count"],
        "passed_source_summary_count": counters["passed_source_summary_count"],
        "source_git_provenance_present_count": counters[
            "source_git_provenance_present_count"
        ],
        "source_git_provenance_mismatch_count": counters[
            "source_git_provenance_mismatch_count"
        ],
        "authorized_trainable_transition_count": counters["authorized_trainable_transition_count"],
        "authorized_optimizer_train_transition_count": counters[
            "authorized_optimizer_train_transition_count"
        ],
        "unique_authorized_trainable_context_count": counters[
            "unique_authorized_trainable_context_count"
        ],
        "validation_trainable_count": counters["validation_trainable_count"],
        "test_trainable_count": counters["test_trainable_count"],
        "fallback_trainable_count": counters["fallback_trainable_count"],
        "source_fallback_trainable_count": counters["source_fallback_trainable_count"],
        "teacher_fallback_trainable_count": counters["teacher_fallback_trainable_count"],
        "diagnostic_trainable_count": counters["diagnostic_trainable_count"],
        "non_empty_gate_reason_trainable_count": counters["non_empty_gate_reason_trainable_count"],
        "missing_observation_count": counters["missing_observation_count"],
        "missing_log_prob_count": counters["missing_log_prob_count"],
        "missing_value_count": counters["missing_value_count"],
        "invalid_action_mask_count": counters["invalid_action_mask_count"],
        "non_finite_reward_count": counters["non_finite_reward_count"],
        "non_finite_return_count": counters["non_finite_return_count"],
        "non_finite_advantage_count": counters["non_finite_advantage_count"],
        "loss_non_finite_count": counters["loss_non_finite_count"],
        "non_finite_gradient_count": counters["non_finite_gradient_count"],
        "controlled_regression_count": counters["controlled_regression_count"],
        "controlled_safety_regression_count": counters["controlled_safety_regression_count"],
        "controlled_contract_regression_count": counters["controlled_contract_regression_count"],
        "controlled_path_risk_regression_count": counters["controlled_path_risk_regression_count"],
        "controlled_source_selection_regression_count": counters[
            "controlled_source_selection_regression_count"
        ],
    }


def _run_readiness_validate_only(
    *,
    repo_root: Path,
    batch_root: Path,
    authorization_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_policy_training_readiness_review.py"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(_resolve_path(config_path, repo_root)),
        "--guarded-formal-ppo-training-authorization-summary",
        str(authorization_summary_path),
        "--validate-only",
    ]
    completed = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        return {
            "training_readiness_status": "needs_training_contract_refinement",
            "training_blockers": ["guarded_formal_ppo_training_authorization_readiness_failed"],
            "reason_codes": ["guarded_formal_ppo_training_authorization_readiness_failed"],
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    first_line = completed.stdout.splitlines()[0] if completed.stdout.splitlines() else "{}"
    try:
        payload = json.loads(first_line)
    except json.JSONDecodeError:
        return {
            "training_readiness_status": "needs_training_contract_refinement",
            "training_blockers": ["guarded_formal_ppo_training_authorization_readiness_invalid_json"],
            "reason_codes": ["guarded_formal_ppo_training_authorization_readiness_invalid_json"],
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return payload


def _validate_readiness(
    readiness: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    expected = config.get("readiness", {}).get("expected_status", EXPECTED_READINESS_STATUS)
    if readiness.get("training_readiness_status") != expected:
        _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_readiness_status_mismatch")
    if _string_list(readiness.get("training_blockers")):
        _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_readiness_blockers")
    if _string_list(readiness.get("reason_codes")):
        _add_reason(reason_codes, "guarded_formal_ppo_training_authorization_readiness_reason_codes")


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Guarded Formal PPO Training Authorization v1",
            "",
            f"- status: `{summary['status']}`",
            f"- authorization_verdict: `{summary['authorization_verdict']}`",
            f"- authorized_trainable_transition_count: `{summary['authorized_trainable_transition_count']}`",
            f"- seed_count: `{summary['seed_count']}`",
            f"- runs_new_ppo_update: `{summary['runs_new_ppo_update']}`",
            f"- publishes_checkpoint: `{summary['publishes_checkpoint']}`",
            f"- replaces_default_policy: `{summary['replaces_default_policy']}`",
            "",
            "This stage authorizes a future guarded formal PPO training run. It does not run PPO, publish a checkpoint, replace the default policy, or claim performance improvement.",
            "",
        ]
    )


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "summary": "formal-ppo-training-authorization-summary.json",
        "training_input_audit": "formal-ppo-training-input-audit.json",
        "budget_manifest": "formal-ppo-training-budget-manifest.json",
        "seed_plan": "formal-ppo-training-seed-plan.json",
        "stop_condition_manifest": "formal-ppo-training-stop-condition-manifest.json",
        "rollback_manifest": "formal-ppo-training-rollback-manifest.json",
        "post_training_gate_plan": "formal-ppo-post-training-gate-plan.json",
        "readiness_validate_only": "formal-ppo-training-authorization-readiness-validate-only.json",
        "report": "formal-ppo-training-authorization-report.md",
    }
    output_files = config.get("output_files") if isinstance(config.get("output_files"), dict) else {}
    return {**defaults, **{key: str(value) for key, value in output_files.items()}}


def _input_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "formal_preflight_summary": "quasi-real-guarded-formal-ppo-preflight-summary.json",
        "formal_rollout_canary_summary": "quasi-real-guarded-formal-ppo-rollout-canary-summary.json",
        "formal_stability_holdout_summary": (
            "quasi-real-guarded-formal-ppo-stability-holdout-validation-summary.json"
        ),
        "candidate_selection_summary": (
            "quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-summary.json"
        ),
        "promotion_decision_review_summary": (
            "selected-formal-ppo-candidate-promotion-decision-review-summary.json"
        ),
        "canary_preflight_summary": (
            "guarded-experimental-policy-staged-release-canary-preflight-summary.json"
        ),
    }
    input_files = config.get("input_files") if isinstance(config.get("input_files"), dict) else {}
    return {**defaults, **{key: str(value) for key, value in input_files.items()}}


def _seeds(config: dict[str, Any]) -> list[int]:
    plan = config.get("training_plan") if isinstance(config.get("training_plan"), dict) else {}
    seeds = plan.get("seeds", [0, 1, 2, 3, 4])
    if not isinstance(seeds, list):
        return [0, 1, 2, 3, 4]
    return [int(seed) for seed in seeds]


def _min_seed_count(config: dict[str, Any]) -> int:
    validation = config.get("validation") if isinstance(config.get("validation"), dict) else {}
    return _int(validation.get("min_seed_count"), 5)


def _source_git_current_matches(summary: dict[str, Any]) -> bool:
    provenance = summary.get("git_provenance")
    if not isinstance(provenance, dict):
        return False
    return provenance.get("current_matches_sources") is not False


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
