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


CONFIG_SCHEMA_VERSION = "selected-formal-ppo-candidate-multihorizon-shadow-rollout-config/v1"
SUMMARY_SCHEMA_VERSION = "selected-formal-ppo-candidate-multihorizon-shadow-rollout-summary/v1"
EXPECTED_READINESS_STATUS = "selected_formal_ppo_candidate_multihorizon_shadow_rollout_evaluated"

CANDIDATE_SELECTION_SUMMARY_FILE = (
    "quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-summary.json"
)
CANDIDATE_SELECTION_SCHEMA_VERSION = (
    "quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-summary/v1"
)
SUMMARY_FILE = "multihorizon-shadow-rollout-summary.json"
EPISODES_FILE = "multihorizon-shadow-rollout-episodes.jsonl"
STEPS_FILE = "multihorizon-shadow-rollout-steps.jsonl"
RETURN_AUDIT_FILE = "multihorizon-return-audit.json"
REJECTION_REPORT_FILE = "multihorizon-rejection-report.json"
FAMILY_REPORT_FILE = "multihorizon-family-report.json"
READINESS_FILE = "multihorizon-readiness-validate-only.json"
REPORT_FILE = "multihorizon-shadow-rollout-report.md"

ReadinessRunner = Callable[..., dict[str, Any]]


def run_selected_formal_ppo_candidate_multihorizon_shadow_rollout(
    *,
    candidate_selection_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    readiness_runner: ReadinessRunner | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    candidate_selection_root = Path(candidate_selection_root)
    output_root = Path(output_root)
    batch_root = Path(batch_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = _output_files(config)
    summary_path = output_root / files["summary"]
    episodes_path = output_root / files["episodes"]
    steps_path = output_root / files["steps"]
    return_audit_path = output_root / files["return_audit"]
    rejection_report_path = output_root / files["rejection_report"]
    family_report_path = output_root / files["family_report"]
    readiness_path = output_root / files["readiness_validate_only"]
    report_path = output_root / files["report"]

    candidate_summary_path = candidate_selection_root / CANDIDATE_SELECTION_SUMMARY_FILE
    candidate_summary = _read_json_if_exists(candidate_summary_path)
    input_steps_path = _resolve_optional_path(
        candidate_summary.get("holdout_steps"), candidate_selection_root, repo_root
    ) or (candidate_selection_root / "long-horizon-holdout-steps.jsonl")
    candidate_manifest_path = _resolve_optional_path(
        candidate_summary.get("candidate_manifest"), candidate_selection_root, repo_root
    ) or (candidate_selection_root / "selected-candidate-manifest.json")

    input_steps = _read_jsonl(input_steps_path)
    trainable_steps = _unique_trainable_steps(input_steps)
    counters = _input_counters(input_steps, trainable_steps)
    candidate_manifest = _read_json_if_exists(candidate_manifest_path)

    reason_codes: list[str] = []
    _validate_candidate_selection_input(
        candidate_summary,
        candidate_manifest,
        counters,
        config,
        reason_codes,
    )

    horizons = _horizons(config)
    shadow = _build_multihorizon_shadow(trainable_steps, horizons=horizons, config=config)
    _write_jsonl(steps_path, shadow["steps"])
    _write_jsonl(episodes_path, shadow["episodes"])
    _write_json(return_audit_path, _return_audit(shadow, horizons))
    _write_json(rejection_report_path, _rejection_report(input_steps, counters))
    _write_json(family_report_path, _family_report(trainable_steps, shadow))

    _validate_shadow(counters, shadow, candidate_summary, horizons, config, reason_codes)

    status_without_readiness = "passed" if not reason_codes else "failed"
    pre_readiness_summary = _summary_payload(
        status=status_without_readiness,
        reason_codes=reason_codes,
        repo_root=repo_root,
        candidate_selection_root=candidate_selection_root,
        candidate_selection_summary_path=candidate_summary_path,
        input_steps_path=input_steps_path,
        candidate_manifest_path=candidate_manifest_path,
        output_root=output_root,
        batch_root=batch_root,
        summary_path=summary_path,
        episodes_path=episodes_path,
        steps_path=steps_path,
        return_audit_path=return_audit_path,
        rejection_report_path=rejection_report_path,
        family_report_path=family_report_path,
        readiness_path=readiness_path,
        report_path=report_path,
        counters=counters,
        candidate_summary=candidate_summary,
        shadow=shadow,
        readiness={},
    )
    _write_json(summary_path, pre_readiness_summary)

    if status_without_readiness == "passed":
        runner = readiness_runner or _run_readiness_validate_only
        readiness = runner(
            repo_root=repo_root,
            batch_root=batch_root,
            shadow_summary_path=summary_path,
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
            "recommended_next_action": "fix_selected_formal_ppo_candidate_multihorizon_shadow_rollout",
        }
    _write_json(readiness_path, readiness)

    final_status = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=final_status,
        reason_codes=reason_codes,
        repo_root=repo_root,
        candidate_selection_root=candidate_selection_root,
        candidate_selection_summary_path=candidate_summary_path,
        input_steps_path=input_steps_path,
        candidate_manifest_path=candidate_manifest_path,
        output_root=output_root,
        batch_root=batch_root,
        summary_path=summary_path,
        episodes_path=episodes_path,
        steps_path=steps_path,
        return_audit_path=return_audit_path,
        rejection_report_path=rejection_report_path,
        family_report_path=family_report_path,
        readiness_path=readiness_path,
        report_path=report_path,
        counters=counters,
        candidate_summary=candidate_summary,
        shadow=shadow,
        readiness=readiness,
    )
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run selected formal PPO candidate multi-horizon shadow rollout."
    )
    parser.add_argument(
        "--candidate-selection-root",
        default=(
            "outputs/"
            "path_feedback_batch_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_v1"
        ),
    )
    parser.add_argument(
        "--batch-root",
        default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/path_feedback_batch_selected_formal_ppo_candidate_multihorizon_shadow_rollout_v1",
    )
    parser.add_argument(
        "--config",
        default="configs/selected_formal_ppo_candidate_multihorizon_shadow_rollout_v1.json",
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

    summary = run_selected_formal_ppo_candidate_multihorizon_shadow_rollout(
        candidate_selection_root=_resolve_path(Path(args.candidate_selection_root), repo_root),
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
                "horizons": summary["horizons"],
                "controlled_regression_count": summary["controlled_regression_count"],
                "family_regression_count": summary["family_regression_count"],
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
    shadow_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        "bash",
        str(repo_root / "scripts" / "run_policy_training_readiness_review.sh"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(_resolve_path(config_path, repo_root)),
        "--selected-formal-ppo-candidate-multihorizon-shadow-rollout-summary",
        str(shadow_summary_path),
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


def _validate_candidate_selection_input(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    counters: Counter[str],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if summary.get("schema_version") != CANDIDATE_SELECTION_SCHEMA_VERSION:
        _add_reason(reason_codes, "input_candidate_selection_schema_invalid")
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _add_reason(reason_codes, "input_candidate_selection_not_passed")
    expected = _expected_trainable(config)
    if _int(summary.get("input_trainable_transition_count")) != expected:
        _add_reason(reason_codes, "input_candidate_selection_trainable_count_mismatch")
    if _int(summary.get("unique_trainable_context_count")) != expected:
        _add_reason(reason_codes, "input_candidate_selection_unique_context_count_mismatch")
    if counters["input_trainable_transition_count"] != expected:
        _add_reason(reason_codes, "selected_formal_ppo_shadow_trainable_count_mismatch")
    if counters["unique_trainable_context_count"] != expected:
        _add_reason(reason_codes, "selected_formal_ppo_shadow_unique_context_count_mismatch")
    if summary.get("selected_candidate_from_stability_matrix") is not True:
        _add_reason(reason_codes, "input_candidate_selection_selected_candidate_missing")
    if not manifest:
        _add_reason(reason_codes, "selected_candidate_manifest_missing")
    if manifest and manifest.get("selected_seed") != summary.get("selected_seed"):
        _add_reason(reason_codes, "selected_candidate_manifest_seed_mismatch")
    if manifest and manifest.get("selected_budget") != summary.get("selected_budget"):
        _add_reason(reason_codes, "selected_candidate_manifest_budget_mismatch")
    for field, reason in (
        ("validation_trainable_count", "selected_formal_ppo_shadow_split_leakage"),
        ("test_trainable_count", "selected_formal_ppo_shadow_split_leakage"),
        ("fallback_trainable_count", "selected_formal_ppo_shadow_fallback_trainable"),
        ("source_fallback_trainable_count", "selected_formal_ppo_shadow_fallback_trainable"),
        ("teacher_fallback_trainable_count", "selected_formal_ppo_shadow_fallback_trainable"),
        ("non_empty_gate_reason_trainable_count", "selected_formal_ppo_shadow_gate_reason_trainable"),
        ("missing_observation_count", "selected_formal_ppo_shadow_contract_invalid"),
        ("missing_log_prob_count", "selected_formal_ppo_shadow_contract_invalid"),
        ("missing_value_count", "selected_formal_ppo_shadow_contract_invalid"),
        ("non_finite_reward_count", "selected_formal_ppo_shadow_non_finite"),
        ("non_finite_return_count", "selected_formal_ppo_shadow_non_finite"),
        ("non_finite_advantage_count", "selected_formal_ppo_shadow_non_finite"),
        ("controlled_regression_count", "selected_formal_ppo_shadow_controlled_regression"),
        ("controlled_safety_regression_count", "selected_formal_ppo_shadow_controlled_regression"),
        ("controlled_contract_regression_count", "selected_formal_ppo_shadow_controlled_regression"),
        ("controlled_path_risk_regression_count", "selected_formal_ppo_shadow_controlled_regression"),
        (
            "controlled_source_selection_regression_count",
            "selected_formal_ppo_shadow_controlled_regression",
        ),
    ):
        if counters[field]:
            _add_reason(reason_codes, reason)
    for source in (summary, manifest):
        for field, reason in (
            ("runs_new_ppo_update", "selected_formal_ppo_shadow_unexpected_ppo_update"),
            ("publishes_checkpoint", "selected_formal_ppo_shadow_checkpoint_publication_claimed"),
            ("replaces_default_policy", "selected_formal_ppo_shadow_default_policy_replacement_claimed"),
            ("performance_claimed", "selected_formal_ppo_shadow_policy_performance_claimed"),
            ("formal_training_ready_claimed", "selected_formal_ppo_shadow_formal_ready_claimed"),
        ):
            if source.get(field) is True:
                _add_reason(reason_codes, reason)
    git = summary.get("git_provenance") if isinstance(summary.get("git_provenance"), dict) else {}
    current_git = git.get("current") if isinstance(git.get("current"), dict) else {}
    if git.get("current_matches_sources") is False:
        _add_reason(reason_codes, "input_candidate_selection_git_provenance_mismatch")
    if current_git.get("dirty") is True:
        _add_reason(reason_codes, "input_candidate_selection_git_provenance_dirty")


def _validate_shadow(
    counters: Counter[str],
    shadow: dict[str, Any],
    candidate_summary: dict[str, Any],
    horizons: list[int],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if horizons != [10, 20, 30]:
        _add_reason(reason_codes, "selected_formal_ppo_shadow_horizons_invalid")
    if any(_int(count) <= 0 for count in shadow["per_horizon_completed_episode_count"].values()):
        _add_reason(reason_codes, "selected_formal_ppo_shadow_horizon_episode_missing")
    if shadow["non_finite_shadow_return_count"] or shadow["non_finite_shadow_advantage_count"]:
        _add_reason(reason_codes, "selected_formal_ppo_shadow_non_finite")
    if counters["controlled_regression_count"]:
        _add_reason(reason_codes, "selected_formal_ppo_shadow_controlled_regression")
    if _float(candidate_summary.get("teacher_agreement_rate"), 0.0) < _float(
        config.get("validation", {}).get("min_teacher_agreement_rate"), 0.95
    ):
        _add_reason(reason_codes, "selected_formal_ppo_shadow_teacher_alignment_insufficient")


def _validate_readiness(readiness: dict[str, Any], config: dict[str, Any], reason_codes: list[str]) -> None:
    expected = str(config.get("readiness", {}).get("expected_status") or EXPECTED_READINESS_STATUS)
    if readiness.get("training_readiness_status") != expected:
        _add_reason(reason_codes, "readiness_not_selected_formal_ppo_candidate_multihorizon_shadow_rollout_evaluated")
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
    candidate_selection_root: Path,
    candidate_selection_summary_path: Path,
    input_steps_path: Path,
    candidate_manifest_path: Path,
    output_root: Path,
    batch_root: Path,
    summary_path: Path,
    episodes_path: Path,
    steps_path: Path,
    return_audit_path: Path,
    rejection_report_path: Path,
    family_report_path: Path,
    readiness_path: Path,
    report_path: Path,
    counters: Counter[str],
    candidate_summary: dict[str, Any],
    shadow: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": None
        if status == "passed"
        else "fix_selected_formal_ppo_candidate_multihorizon_shadow_rollout",
        "candidate_selection_root": str(candidate_selection_root),
        "candidate_selection_summary": str(candidate_selection_summary_path),
        "input_steps": str(input_steps_path),
        "candidate_manifest": str(candidate_manifest_path),
        "batch_root": str(batch_root),
        "output_root": str(output_root),
        "summary": str(summary_path),
        "episodes": str(episodes_path),
        "steps": str(steps_path),
        "return_audit": str(return_audit_path),
        "rejection_report": str(rejection_report_path),
        "family_report": str(family_report_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "selected_seed": candidate_summary.get("selected_seed"),
        "selected_budget": candidate_summary.get("selected_budget"),
        "selected_candidate_root": candidate_summary.get("selected_candidate_root"),
        "selected_candidate_from_candidate_selection": bool(
            candidate_summary.get("selected_candidate_from_stability_matrix")
        ),
        "horizons": shadow["horizons"],
        "per_horizon_step_count": shadow["per_horizon_step_count"],
        "per_horizon_episode_count": shadow["per_horizon_episode_count"],
        "per_horizon_completed_episode_count": shadow["per_horizon_completed_episode_count"],
        "input_trainable_transition_count": counters["input_trainable_transition_count"],
        "shadow_trainable_transition_count": shadow["shadow_trainable_transition_count"],
        "unique_trainable_context_count": counters["unique_trainable_context_count"],
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
        "non_finite_shadow_return_count": shadow["non_finite_shadow_return_count"],
        "non_finite_shadow_advantage_count": shadow["non_finite_shadow_advantage_count"],
        "controlled_regression_count": counters["controlled_regression_count"],
        "train_controlled_regression_count": counters["train_controlled_regression_count"],
        "validation_controlled_regression_count": counters["validation_controlled_regression_count"],
        "test_controlled_regression_count": counters["test_controlled_regression_count"],
        "family_regression_count": counters["family_regression_count"],
        "controlled_safety_regression_count": counters["controlled_safety_regression_count"],
        "controlled_contract_regression_count": counters["controlled_contract_regression_count"],
        "controlled_path_risk_regression_count": counters["controlled_path_risk_regression_count"],
        "controlled_source_selection_regression_count": counters["controlled_source_selection_regression_count"],
        "teacher_agreement_rate": _float(candidate_summary.get("teacher_agreement_rate"), 0.0),
        "uses_multistep_discounted_return": True,
        "not_single_step_best_action": True,
        "readiness_status": readiness.get("training_readiness_status"),
        "training_blockers": list(readiness.get("training_blockers") or []),
        "readiness_reason_codes": list(readiness.get("reason_codes") or []),
        "runs_multihorizon_shadow_rollout": True,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _build_multihorizon_shadow(
    trainable_steps: list[dict[str, Any]],
    *,
    horizons: list[int],
    config: dict[str, Any],
) -> dict[str, Any]:
    discount = _float(config.get("shadow_rollout", {}).get("discount_factor"), 0.99)
    steps: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    per_horizon_step_count: dict[str, int] = {}
    per_horizon_episode_count: dict[str, int] = {}
    per_horizon_completed_episode_count: dict[str, int] = {}
    non_finite_return_count = 0
    non_finite_advantage_count = 0
    returns: list[float] = []
    for horizon in horizons:
        horizon_key = str(horizon)
        horizon_episode_count = 0
        horizon_completed_count = 0
        horizon_step_count = 0
        for episode_index, offset in enumerate(range(0, len(trainable_steps), horizon)):
            chunk = trainable_steps[offset : offset + horizon]
            episode_id = f"multihorizon-shadow-h{horizon}-{episode_index:04d}"
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
                        "schema_version": "selected-formal-ppo-candidate-multihorizon-shadow-step/v1",
                        "shadow_episode_id": episode_id,
                        "shadow_horizon": horizon,
                        "shadow_step_index": local_index,
                        "shadow_discounted_return": discounted_return,
                        "shadow_advantage": advantage,
                        "shadow_trainable": True,
                        "diagnostic_only": True,
                    }
                )
                steps.append(row)
                horizon_step_count += 1
            completed = len(chunk) == horizon
            horizon_episode_count += 1
            if completed:
                horizon_completed_count += 1
            episodes.append(
                {
                    "schema_version": "selected-formal-ppo-candidate-multihorizon-shadow-episode/v1",
                    "episode_id": episode_id,
                    "horizon": horizon,
                    "step_count": len(chunk),
                    "completed_horizon": completed,
                    "discounted_return": episode_return,
                    "scenario_family_counts": dict(
                        Counter(str(step.get("scenario_family") or "unknown") for step in chunk)
                    ),
                }
            )
        per_horizon_step_count[horizon_key] = horizon_step_count
        per_horizon_episode_count[horizon_key] = horizon_episode_count
        per_horizon_completed_episode_count[horizon_key] = horizon_completed_count
    return {
        "horizons": horizons,
        "steps": steps,
        "episodes": episodes,
        "per_horizon_step_count": per_horizon_step_count,
        "per_horizon_episode_count": per_horizon_episode_count,
        "per_horizon_completed_episode_count": per_horizon_completed_episode_count,
        "shadow_trainable_transition_count": len(steps),
        "min_shadow_discounted_return": min(returns) if returns else 0.0,
        "max_shadow_discounted_return": max(returns) if returns else 0.0,
        "non_finite_shadow_return_count": non_finite_return_count,
        "non_finite_shadow_advantage_count": non_finite_advantage_count,
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
    regression_families: set[str] = set()
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
            regression_families.add(str(step.get("scenario_family") or "unknown"))
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
    counters["family_regression_count"] = len(regression_families)
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


def _return_audit(shadow: dict[str, Any], horizons: list[int]) -> dict[str, Any]:
    return {
        "schema_version": "selected-formal-ppo-candidate-multihorizon-return-audit/v1",
        "horizons": horizons,
        "per_horizon_step_count": shadow["per_horizon_step_count"],
        "per_horizon_episode_count": shadow["per_horizon_episode_count"],
        "per_horizon_completed_episode_count": shadow["per_horizon_completed_episode_count"],
        "shadow_trainable_transition_count": shadow["shadow_trainable_transition_count"],
        "min_shadow_discounted_return": shadow["min_shadow_discounted_return"],
        "max_shadow_discounted_return": shadow["max_shadow_discounted_return"],
        "non_finite_shadow_return_count": shadow["non_finite_shadow_return_count"],
        "non_finite_shadow_advantage_count": shadow["non_finite_shadow_advantage_count"],
        "uses_multistep_discounted_return": True,
        "does_not_use_single_step_greedy_reward": True,
    }


def _rejection_report(steps: list[dict[str, Any]], counters: Counter[str]) -> dict[str, Any]:
    rejected_rows = []
    for step in steps:
        reasons = []
        if step.get("ppo_trainable") is True and step.get("split") in {"validation", "test"}:
            reasons.append("split_diagnostic_only")
        if step.get("ppo_trainable") is True and str(step.get("controlled_choice_source")) in {
            "source_fallback",
            "teacher_fallback",
        }:
            reasons.append("fallback_diagnostic_only")
        if step.get("ppo_trainable") is True and _string_list(step.get("gate_reason_codes")):
            reasons.append("gate_reason_diagnostic_only")
        if reasons:
            rejected_rows.append(
                {
                    "context_id": step.get("context_id"),
                    "scenario_id": step.get("scenario_id"),
                    "split": step.get("split"),
                    "controlled_choice_source": step.get("controlled_choice_source"),
                    "reasons": reasons,
                }
            )
    return {
        "schema_version": "selected-formal-ppo-candidate-multihorizon-rejection-report/v1",
        "rejected_row_count": len(rejected_rows),
        "validation_trainable_count": counters["validation_trainable_count"],
        "test_trainable_count": counters["test_trainable_count"],
        "source_fallback_trainable_count": counters["source_fallback_trainable_count"],
        "teacher_fallback_trainable_count": counters["teacher_fallback_trainable_count"],
        "non_empty_gate_reason_trainable_count": counters["non_empty_gate_reason_trainable_count"],
        "rows": rejected_rows,
    }


def _family_report(trainable_steps: list[dict[str, Any]], shadow: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "selected-formal-ppo-candidate-multihorizon-family-report/v1",
        "trainable_family_counts": dict(
            sorted(Counter(str(step.get("scenario_family") or "unknown") for step in trainable_steps).items())
        ),
        "per_horizon_completed_episode_count": shadow["per_horizon_completed_episode_count"],
        "family_regression_count": 0,
    }


def _render_report(summary: dict[str, Any]) -> str:
    return (
        "# Selected Formal PPO Candidate Multi-Horizon Shadow Rollout\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Reason codes: `{summary.get('reason_codes')}`\n"
        f"- Selected candidate: seed `{summary.get('selected_seed')}`, budget `{summary.get('selected_budget')}`\n"
        f"- Horizons: `{summary.get('horizons')}`\n"
        f"- Shadow trainable transitions: `{summary.get('shadow_trainable_transition_count')}`\n"
        f"- Controlled regression count: `{summary.get('controlled_regression_count')}`\n"
        f"- Family regression count: `{summary.get('family_regression_count')}`\n"
        f"- Teacher agreement: `{summary.get('teacher_agreement_rate')}`\n"
        f"- Readiness: `{summary.get('readiness_status')}`\n\n"
        "This stage is a read-only shadow rollout of the selected experimental candidate. "
        "It does not run a new PPO update, publish a checkpoint, replace the default policy, "
        "claim policy performance, or claim formal training readiness.\n"
    )


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "summary": SUMMARY_FILE,
        "episodes": EPISODES_FILE,
        "steps": STEPS_FILE,
        "return_audit": RETURN_AUDIT_FILE,
        "rejection_report": REJECTION_REPORT_FILE,
        "family_report": FAMILY_REPORT_FILE,
        "readiness_validate_only": READINESS_FILE,
        "report": REPORT_FILE,
    }
    output_files = config.get("output_files") if isinstance(config.get("output_files"), dict) else {}
    return {key: str(output_files.get(key) or default) for key, default in defaults.items()}


def _horizons(config: dict[str, Any]) -> list[int]:
    values = config.get("shadow_rollout", {}).get("horizons", [10, 20, 30])
    return [_int(value) for value in values if _int(value) > 0]


def _expected_trainable(config: dict[str, Any]) -> int:
    return _int(config.get("validation", {}).get("expected_trainable_transition_count"), 684)


def _resolve_optional_path(value: Any, base: Path, repo_root: Path) -> Path | None:
    if not value:
        return None
    return _resolve_path(Path(str(value)), repo_root if Path(str(value)).is_absolute() else base)


def _resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


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
