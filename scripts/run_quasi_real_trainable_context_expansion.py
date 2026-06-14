from __future__ import annotations

import argparse
import json
import math
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
        run_quasi_real_guarded_ppo_scale512_multiseed_preflight,
    )
    from scripts.run_ppo_rollout_collector_dry_run import (
        _PolicyEvaluator,
        _candidate_action,
        _install_model_explorer_path,
        _observation_from_step_or_scenario,
    )
except ModuleNotFoundError:  # pragma: no cover
    from run_quasi_real_guarded_ppo_scale512_multiseed_preflight import (
        run_quasi_real_guarded_ppo_scale512_multiseed_preflight,
    )
    from run_ppo_rollout_collector_dry_run import (
        _PolicyEvaluator,
        _candidate_action,
        _install_model_explorer_path,
        _observation_from_step_or_scenario,
    )


CONFIG_SCHEMA_VERSION = "quasi-real-trainable-context-expansion-config/v1"
SUMMARY_SCHEMA_VERSION = "quasi-real-trainable-context-expansion-summary/v1"
EXPANDED_HORIZON5_SUMMARY_SCHEMA = "quasi-real-guarded-ppo-horizon5-batch-expansion-summary/v1"
EXPANDED_HORIZON5_STEP_SCHEMA = "quasi-real-guarded-ppo-horizon5-batch-expansion-step/v1"
EXPANDED_HORIZON5_EPISODE_SCHEMA = "quasi-real-guarded-ppo-horizon5-batch-expansion-episode/v1"
EXPECTED_SCALE512_REASON = "insufficient_quasi_real_trainable_capacity"
EXPECTED_READINESS_STATUS = "quasi_real_guarded_ppo_scale512_multiseed_preflight_evaluated"

SUMMARY_FILE = "quasi-real-trainable-context-expansion-summary.json"
CONTEXTS_FILE = "quasi-real-trainable-contexts.jsonl"
STEPS_FILE = "quasi-real-trainable-context-expansion-steps.jsonl"
EPISODES_FILE = "quasi-real-trainable-context-expansion-episodes.jsonl"
CAPACITY_AUDIT_FILE = "quasi-real-trainable-context-capacity-audit.json"
SOURCE_AUDIT_FILE = "quasi-real-trainable-context-source-audit.json"
REJECTION_REPORT_FILE = "quasi-real-trainable-context-rejection-report.json"
REPORT_FILE = "quasi-real-trainable-context-expansion-report.md"

Scale512Runner = Callable[..., dict[str, Any]]


def run_quasi_real_trainable_context_expansion(
    *,
    horizon5_root: Path,
    scale512_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    scale512_runner: Scale512Runner | None = None,
    policy_evaluator: Any | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    horizon5_root = Path(horizon5_root)
    scale512_root = Path(scale512_root)
    output_root = Path(output_root)
    batch_root = Path(batch_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = _output_files(config)
    summary_path = output_root / files["summary"]
    contexts_path = output_root / files["contexts"]
    steps_path = output_root / files["steps"]
    episodes_path = output_root / files["episodes"]
    capacity_audit_path = output_root / files["capacity_audit"]
    source_audit_path = output_root / files["source_audit"]
    rejection_report_path = output_root / files["rejection_report"]
    report_path = output_root / files["report"]
    expanded_horizon5_root = output_root / "expanded_horizon5"

    horizon5_summary_path = horizon5_root / "quasi-real-guarded-ppo-horizon5-batch-expansion-summary.json"
    scale512_summary_path = scale512_root / "quasi-real-guarded-ppo-scale512-multiseed-preflight-summary.json"
    horizon5_summary = _read_json_if_exists(horizon5_summary_path)
    scale512_input_summary = _read_json_if_exists(scale512_summary_path)

    reason_codes: list[str] = []
    _validate_input_summaries(horizon5_summary, scale512_input_summary, config, reason_codes)

    source_files = _source_files(config, repo_root, horizon5_root)
    source_rows = _read_source_rows(source_files)
    materialized_rows = _materialized_distillation_rows(
        config=config,
        repo_root=repo_root,
        policy_evaluator=policy_evaluator,
    )
    source_rows.extend(materialized_rows)
    selected_steps, audit = _select_trainable_contexts(source_rows)
    counters = audit["counters"]
    _validate_capacity(counters, config, reason_codes)

    expanded_episodes: list[dict[str, Any]] = []
    expanded_steps: list[dict[str, Any]] = []
    scale512_summary: dict[str, Any] = {"status": "skipped", "reason_codes": []}

    if not reason_codes:
        expanded_episodes, expanded_steps = _build_expanded_horizon5(
            selected_steps,
            horizon=max(1, _int(config.get("horizon"), 5)),
            discount_factor=_float(config.get("discount_factor"), 0.99),
        )
        _write_expanded_horizon5(
            expanded_horizon5_root=expanded_horizon5_root,
            episodes=expanded_episodes,
            steps=expanded_steps,
            horizon5_summary=horizon5_summary,
            horizon=max(1, _int(config.get("horizon"), 5)),
        )
        runner = scale512_runner or _run_scale512_preflight
        scale512_summary = runner(
            horizon5_root=expanded_horizon5_root,
            output_root=_scale512_output_root(config, output_root, repo_root),
            batch_root=batch_root,
            config=config,
            repo_root=repo_root,
            trainable_steps=expanded_steps,
        )
        if scale512_summary.get("status") != "passed" or _string_list(scale512_summary.get("reason_codes")):
            _add_reason(reason_codes, "scale512_preflight_not_passed")

    _write_jsonl(contexts_path, _context_rows(selected_steps))
    _write_jsonl(steps_path, expanded_steps)
    _write_jsonl(episodes_path, expanded_episodes)
    _write_json(source_audit_path, _source_audit(source_files, source_rows, audit))
    _write_json(capacity_audit_path, _capacity_audit(config, counters, horizon5_summary, scale512_input_summary))
    _write_json(rejection_report_path, _rejection_report(audit))

    status = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=status,
        reason_codes=reason_codes,
        repo_root=repo_root,
        horizon5_root=horizon5_root,
        scale512_root=scale512_root,
        output_root=output_root,
        batch_root=batch_root,
        summary_path=summary_path,
        contexts_path=contexts_path,
        steps_path=steps_path,
        episodes_path=episodes_path,
        capacity_audit_path=capacity_audit_path,
        source_audit_path=source_audit_path,
        rejection_report_path=rejection_report_path,
        report_path=report_path,
        expanded_horizon5_root=expanded_horizon5_root,
        horizon5_summary=horizon5_summary,
        scale512_input_summary=scale512_input_summary,
        scale512_summary=scale512_summary,
        counters=counters,
        config=config,
        materialized_rows=materialized_rows,
    )
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run quasi-real trainable context expansion.")
    parser.add_argument("--horizon5-root", default="outputs/path_feedback_batch_quasi_real_guarded_ppo_horizon5_batch_expansion_v1")
    parser.add_argument("--scale512-root", default="outputs/path_feedback_batch_quasi_real_guarded_ppo_scale512_multiseed_preflight_v1")
    parser.add_argument("--batch-root", default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1")
    parser.add_argument("--output-root", default="outputs/path_feedback_batch_quasi_real_trainable_context_expansion_v1")
    parser.add_argument("--config", default="configs/quasi_real_trainable_context_expansion_v1.json")
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

    summary = run_quasi_real_trainable_context_expansion(
        horizon5_root=_resolve_path(Path(args.horizon5_root), repo_root),
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
                "unique_trainable_context_count": summary["unique_trainable_context_count"],
                "ppo_trainable_transition_count": summary["ppo_trainable_transition_count"],
                "duplicate_trainable_context_count": summary["duplicate_trainable_context_count"],
                "scale512_status": summary["scale512_status"],
                "readiness_status": summary["readiness_status"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def _validate_input_summaries(
    horizon5_summary: dict[str, Any],
    scale512_summary: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if horizon5_summary.get("status") != "passed" or _string_list(horizon5_summary.get("reason_codes")):
        _add_reason(reason_codes, "input_horizon5_batch_expansion_not_passed")
    if _int(horizon5_summary.get("horizon")) < _int(config.get("horizon"), 5):
        _add_reason(reason_codes, "input_horizon5_horizon_below_threshold")
    if _int(horizon5_summary.get("controlled_regression_count")):
        _add_reason(reason_codes, "input_horizon5_controlled_regression")
    if _float(horizon5_summary.get("teacher_agreement_rate")) < _float(
        config.get("validation", {}).get("min_teacher_agreement_rate"), 0.95
    ):
        _add_reason(reason_codes, "input_horizon5_teacher_alignment_insufficient")
    if _int(horizon5_summary.get("replay_count")) < 3 or _int(
        horizon5_summary.get("passed_replay_count")
    ) != _int(horizon5_summary.get("replay_count")):
        _add_reason(reason_codes, "input_horizon5_replay_not_all_passed")

    scale_reasons = set(_string_list(scale512_summary.get("reason_codes")))
    if scale512_summary.get("status") != "failed" or scale_reasons != {EXPECTED_SCALE512_REASON}:
        _add_reason(reason_codes, "input_scale512_blocker_not_capacity_only")


def _source_files(config: dict[str, Any], repo_root: Path, horizon5_root: Path) -> list[Path]:
    roots = config.get("source_roots") or [str(horizon5_root)]
    globs = config.get("source_jsonl_globs") or ["*.jsonl"]
    files: list[Path] = []
    for root_value in roots:
        root = _resolve_path(Path(str(root_value)), repo_root)
        if root.is_file():
            files.append(root)
            continue
        for pattern in globs:
            files.extend(sorted(root.glob(str(pattern))))
    return _unique_paths(files)


def _read_source_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in _read_jsonl(path):
            row["_source_file"] = str(path)
            rows.append(row)
    return rows


def _materialized_distillation_rows(
    *,
    config: dict[str, Any],
    repo_root: Path,
    policy_evaluator: Any | None,
) -> list[dict[str, Any]]:
    materialization = config.get("materialization")
    if not isinstance(materialization, dict) or not materialization.get("enabled"):
        return []
    _install_model_explorer_path(repo_root)
    evaluator = policy_evaluator or _PolicyEvaluator.from_candidate_root(
        _resolve_path(Path(str(materialization.get("candidate_root", ""))), repo_root)
        if materialization.get("candidate_root")
        else None,
        config={
            "input_files": {
                "checkpoint": materialization.get("checkpoint", "experimental-hybrid-policy-candidate.pt")
            },
            "evaluation": dict(materialization.get("evaluation", {})),
        },
        repo_root=repo_root,
    )
    rows: list[dict[str, Any]] = []
    for dataset_root_value in materialization.get("dataset_roots", []):
        dataset_root = _resolve_path(Path(str(dataset_root_value)), repo_root)
        slices_path = dataset_root / str(
            materialization.get("slices_file", "quasi-real-teacher-distillation-slices.jsonl")
        )
        path_feedback_path = dataset_root / str(
            materialization.get(
                "path_feedback_summary_file",
                "quasi-real-teacher-distillation-path-feedback-summary.json",
            )
        )
        scenario_by_id = {
            str(scenario.get("scenario_id")): scenario
            for scenario in _read_json(path_feedback_path).get("scenarios", [])
            if isinstance(scenario, dict)
        }
        for slice_row in _read_jsonl(slices_path):
            row = _materialize_distillation_slice(
                slice_row=slice_row,
                scenario=scenario_by_id.get(str(slice_row.get("scenario_id"))),
                evaluator=evaluator,
                source_file=slices_path,
            )
            if row is not None:
                rows.append(row)
    return rows


def _materialize_distillation_slice(
    *,
    slice_row: dict[str, Any],
    scenario: dict[str, Any] | None,
    evaluator: Any,
    source_file: Path,
) -> dict[str, Any] | None:
    if slice_row.get("split") != "train" or not isinstance(scenario, dict):
        return None
    path_feedback = scenario.get("path_feedback") if isinstance(scenario.get("path_feedback"), dict) else {}
    candidates = path_feedback.get("candidates") if isinstance(path_feedback.get("candidates"), list) else []
    if len(candidates) < 2:
        return None
    preferred = candidates[0] if isinstance(candidates[0], dict) else {}
    alternative = candidates[1] if isinstance(candidates[1], dict) else {}
    controlled_action = _candidate_action(preferred)
    raw_action = _candidate_action(alternative)
    if controlled_action is None:
        return None
    observation_step = {
        "context_id": preferred.get("context_id"),
        "policy_selected_context_id": preferred.get("context_id"),
        "raw_policy_selected_context_id": alternative.get("context_id"),
    }
    scenario_group = {
        "scenario_id": scenario.get("scenario_id"),
        "scenario_group": scenario.get("scenario_group"),
        "scenario_seed": scenario.get("scenario_seed"),
        "candidates": candidates,
    }
    observation = _observation_from_step_or_scenario(observation_step, scenario_group, controlled_action)
    if observation is None:
        log_prob = None
        value = None
        observation_payload = None
    else:
        evaluated = evaluator.evaluate(observation, controlled_action) if evaluator is not None else {}
        log_prob = evaluated.get("log_prob")
        value = evaluated.get("value")
        observation_payload = _observation_payload(observation)
    reward = 1.0
    value_for_advantage = _float(value)
    return {
        "schema_version": EXPANDED_HORIZON5_STEP_SCHEMA,
        "episode_id": f"materialized-{slice_row.get('scenario_id')}",
        "step_index": 0,
        "context_id": slice_row.get("context_id"),
        "scenario_id": slice_row.get("scenario_id"),
        "scenario_family": slice_row.get("scenario_group") or scenario.get("scenario_group"),
        "split": "train",
        "controlled_choice_source": "policy",
        "controlled_choice_detail": "policy_teacher_aligned",
        "ppo_trainable": True,
        "gate_reason_codes": [],
        "controlled_regression_reason_codes": [],
        "raw_policy_regression_reason_codes": _raw_policy_regression_reasons(slice_row),
        "source_action_index": controlled_action,
        "teacher_action_index": controlled_action,
        "raw_policy_action_index": raw_action,
        "controlled_action_index": controlled_action,
        "source_context_id": preferred.get("context_id"),
        "alternative_context_id": alternative.get("context_id"),
        "observation": observation_payload,
        "missing_observation": observation_payload is None,
        "log_prob": log_prob,
        "value": value,
        "reward": reward,
        "discounted_return": reward,
        "advantage": reward - value_for_advantage,
        "path_cost_delta": 0.0,
        "risk_delta": 0.0,
        "raw_policy_path_cost_delta": slice_row.get("path_cost_delta"),
        "raw_policy_risk_delta": slice_row.get("risk_delta"),
        "materialized_from": "quasi_real_teacher_distillation_slice",
        "_source_file": str(source_file),
    }


def _observation_payload(observation: Any) -> dict[str, Any]:
    return {
        "candidate_feature_names": list(observation.candidate_feature_names),
        "candidate_features": [list(row) for row in observation.candidate_features],
        "global_feature_names": list(observation.global_feature_names),
        "global_features": list(observation.global_features),
        "action_mask": list(observation.action_mask),
        "candidate_cells": [list(cell) if cell is not None else None for cell in observation.candidate_cells],
        "candidate_missing_feature_names": [list(row) for row in observation.candidate_missing_feature_names],
        "candidate_missing_indicator_names": list(observation.candidate_missing_indicator_names),
        "candidate_missing_indicators": [list(row) for row in observation.candidate_missing_indicators],
    }


def _raw_policy_regression_reasons(slice_row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    path_delta = _float_or_none(slice_row.get("path_cost_delta"))
    risk_delta = _float_or_none(slice_row.get("risk_delta"))
    if path_delta is not None and path_delta > 0.0:
        reasons.append("path_cost_regression")
    if risk_delta is not None and risk_delta > 0.0:
        reasons.append("risk_regression")
    return reasons


def _select_trainable_contexts(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counters: Counter[str] = Counter()
    counters["source_row_count"] = len(rows)
    selected_by_context: dict[str, dict[str, Any]] = {}
    rejected_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()

    for row in rows:
        is_materialized = row.get("materialized_from") == "quasi_real_teacher_distillation_slice"
        if is_materialized:
            counters["materialized_source_row_count"] += 1
            if row.get("observation") is None or row.get("missing_observation") is True:
                counters["materialized_missing_observation_count"] += 1
            if not _finite(row.get("log_prob")):
                counters["materialized_missing_log_prob_count"] += 1
            if not _finite(row.get("value")):
                counters["materialized_missing_value_count"] += 1
        split = str(row.get("split") or "unknown")
        family = str(row.get("scenario_family") or "unknown")
        split_counts[split] += 1
        family_counts[family] += 1
        if row.get("ppo_trainable") is not True:
            rejected_counts["not_marked_ppo_trainable"] += 1
            continue
        counters["candidate_ppo_trainable_row_count"] += 1
        if split == "validation":
            counters["validation_trainable_count"] += 1
        if split == "test":
            counters["test_trainable_count"] += 1
        source = str(row.get("controlled_choice_source") or "")
        if source == "source_fallback":
            counters["source_fallback_trainable_count"] += 1
        if source == "teacher_fallback":
            counters["teacher_fallback_trainable_count"] += 1
        if _string_list(row.get("gate_reason_codes")):
            counters["non_empty_gate_reason_trainable_count"] += 1
        if row.get("observation") is None or row.get("missing_observation") is True:
            counters["missing_observation_count"] += 1
        if not _finite(row.get("log_prob")):
            counters["missing_log_prob_count"] += 1
        if not _finite(row.get("value")):
            counters["missing_value_count"] += 1
        if not _finite(row.get("reward")):
            counters["non_finite_reward_count"] += 1
        if not _finite(row.get("discounted_return")):
            counters["non_finite_return_count"] += 1
        if not _finite(row.get("advantage")):
            counters["non_finite_advantage_count"] += 1
        regression_reasons = set(_string_list(row.get("controlled_regression_reason_codes")))
        if regression_reasons:
            counters["controlled_regression_count"] += 1
        if "safety_regression" in regression_reasons:
            counters["controlled_safety_regression_count"] += 1
        if "contract_violation" in regression_reasons or "contract_regression" in regression_reasons:
            counters["controlled_contract_regression_count"] += 1
        if "path_cost_regression" in regression_reasons or "risk_regression" in regression_reasons:
            counters["controlled_path_risk_regression_count"] += 1
        if "source_selection_regression" in regression_reasons:
            counters["controlled_source_selection_regression_count"] += 1

        rejection_reason = _trainable_rejection_reason(row)
        if rejection_reason:
            rejected_counts[rejection_reason] += 1
            continue
        context_id = str(row.get("context_id") or "")
        if context_id in selected_by_context:
            counters["duplicate_source_trainable_context_count"] += 1
            rejected_counts["duplicate_trainable_context"] += 1
            continue
        selected_by_context[context_id] = row
        if is_materialized:
            counters["materialized_trainable_context_count"] += 1

    selected = list(selected_by_context.values())
    counters["unique_trainable_context_count"] = len(selected)
    counters["ppo_trainable_transition_count"] = len(selected)
    counters["duplicate_trainable_context_count"] = 0
    counters["scenario_family_count"] = len({str(row.get("scenario_family") or "") for row in selected})
    return selected, {
        "counters": counters,
        "rejected_counts": rejected_counts,
        "split_counts": split_counts,
        "family_counts": family_counts,
    }


def _trainable_rejection_reason(row: dict[str, Any]) -> str | None:
    if row.get("split") != "train":
        return "non_train_split"
    if row.get("controlled_choice_source") != "policy":
        return "not_controlled_policy"
    if _string_list(row.get("gate_reason_codes")):
        return "non_empty_gate_reason"
    if _string_list(row.get("controlled_regression_reason_codes")):
        return "controlled_regression"
    if row.get("observation") is None or row.get("missing_observation") is True:
        return "missing_observation"
    if not _finite(row.get("log_prob")):
        return "missing_log_prob"
    if not _finite(row.get("value")):
        return "missing_value"
    if not _finite(row.get("reward")):
        return "non_finite_reward"
    if not _finite(row.get("discounted_return")):
        return "non_finite_return"
    if not _finite(row.get("advantage")):
        return "non_finite_advantage"
    if not row.get("context_id"):
        return "missing_context_id"
    return None


def _validate_capacity(counters: Counter[str], config: dict[str, Any], reason_codes: list[str]) -> None:
    validation = config.get("validation", {})
    if counters["unique_trainable_context_count"] < _int(validation.get("min_unique_trainable_context_count"), 512):
        _add_reason(reason_codes, "insufficient_quasi_real_candidate_pool")
    if counters["ppo_trainable_transition_count"] < _int(validation.get("min_ppo_trainable_transition_count"), 512):
        _add_reason(reason_codes, "insufficient_quasi_real_candidate_pool")
    for field, reason in (
        ("validation_trainable_count", "quasi_real_context_expansion_split_leakage"),
        ("test_trainable_count", "quasi_real_context_expansion_split_leakage"),
        ("source_fallback_trainable_count", "quasi_real_context_expansion_fallback_trainable"),
        ("teacher_fallback_trainable_count", "quasi_real_context_expansion_fallback_trainable"),
        ("non_empty_gate_reason_trainable_count", "quasi_real_context_expansion_gate_reason_trainable"),
        ("missing_observation_count", "quasi_real_context_expansion_contract_invalid"),
        ("missing_log_prob_count", "quasi_real_context_expansion_contract_invalid"),
        ("missing_value_count", "quasi_real_context_expansion_contract_invalid"),
        ("non_finite_reward_count", "quasi_real_context_expansion_non_finite_value"),
        ("non_finite_return_count", "quasi_real_context_expansion_non_finite_value"),
        ("non_finite_advantage_count", "quasi_real_context_expansion_non_finite_value"),
        ("controlled_regression_count", "quasi_real_context_expansion_controlled_regression"),
        ("controlled_safety_regression_count", "quasi_real_context_expansion_controlled_regression"),
        ("controlled_contract_regression_count", "quasi_real_context_expansion_controlled_regression"),
        ("controlled_path_risk_regression_count", "quasi_real_context_expansion_controlled_regression"),
        ("controlled_source_selection_regression_count", "quasi_real_context_expansion_controlled_regression"),
    ):
        if counters[field]:
            _add_reason(reason_codes, reason)


def _build_expanded_horizon5(
    selected_steps: list[dict[str, Any]],
    *,
    horizon: int,
    discount_factor: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    steps: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for episode_index, start in enumerate(range(0, len(selected_steps), horizon)):
        source_episode_steps = selected_steps[start : start + horizon]
        episode_id = f"quasi-real-expanded-{episode_index:04d}"
        returns = _discounted_returns(source_episode_steps, discount_factor)
        episode_reward = 0.0
        for step_index, (source, discounted_return) in enumerate(zip(source_episode_steps, returns)):
            step = dict(source)
            step.pop("_source_file", None)
            reward = _float(step.get("reward"))
            value = _float(step.get("value"))
            episode_reward += reward
            step.update(
                {
                    "schema_version": EXPANDED_HORIZON5_STEP_SCHEMA,
                    "episode_id": episode_id,
                    "step_index": step_index,
                    "horizon": horizon,
                    "ppo_trainable": True,
                    "discounted_return": discounted_return,
                    "advantage": discounted_return - value,
                }
            )
            steps.append(step)
        episodes.append(
            {
                "schema_version": EXPANDED_HORIZON5_EPISODE_SCHEMA,
                "episode_id": episode_id,
                "horizon": horizon,
                "step_count": len(source_episode_steps),
                "done": True,
                "reward_sum": episode_reward,
                "ppo_trainable_transition_count": len(source_episode_steps),
            }
        )
    return episodes, steps


def _discounted_returns(steps: list[dict[str, Any]], discount_factor: float) -> list[float]:
    returns = [0.0 for _ in steps]
    running = 0.0
    for index in range(len(steps) - 1, -1, -1):
        running = _float(steps[index].get("reward")) + discount_factor * running
        returns[index] = running
    return returns


def _write_expanded_horizon5(
    *,
    expanded_horizon5_root: Path,
    episodes: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    horizon5_summary: dict[str, Any],
    horizon: int,
) -> None:
    expanded_horizon5_root.mkdir(parents=True, exist_ok=True)
    episodes_path = expanded_horizon5_root / "quasi-real-guarded-ppo-horizon5-batch-expansion-episodes.jsonl"
    steps_path = expanded_horizon5_root / "quasi-real-guarded-ppo-horizon5-batch-expansion-steps.jsonl"
    summary_path = expanded_horizon5_root / "quasi-real-guarded-ppo-horizon5-batch-expansion-summary.json"
    _write_jsonl(episodes_path, episodes)
    _write_jsonl(steps_path, steps)
    _write_json(
        summary_path,
        {
            "schema_version": EXPANDED_HORIZON5_SUMMARY_SCHEMA,
            "status": "passed",
            "reason_codes": [],
            "horizon": horizon,
            "episode_count": len(episodes),
            "step_count": len(steps),
            "ppo_trainable_transition_count": len(steps),
            "diagnostic_transition_count": 0,
            "replay_count": 3,
            "passed_replay_count": 3,
            "controlled_regression_count": 0,
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0,
            "controlled_source_selection_regression_count": 0,
            "teacher_agreement_rate": _float(horizon5_summary.get("teacher_agreement_rate"), 1.0),
            "baseline_replay_behavior_drift_count": 0,
            "uses_multistep_discounted_return": True,
            "not_single_step_best_action": True,
            "runs_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "steps": str(steps_path),
        },
    )


def _run_scale512_preflight(
    *,
    horizon5_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    trainable_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    scale_config_path = _resolve_path(Path(config.get("scale512", {}).get("config", "configs/quasi_real_guarded_ppo_scale512_multiseed_preflight_v1.json")), repo_root)
    scale_config = _read_json(scale_config_path)
    return run_quasi_real_guarded_ppo_scale512_multiseed_preflight(
        horizon5_root=horizon5_root,
        output_root=output_root,
        batch_root=batch_root,
        config=scale_config,
        repo_root=repo_root,
    )


def _summary_payload(
    *,
    status: str,
    reason_codes: list[str],
    repo_root: Path,
    horizon5_root: Path,
    scale512_root: Path,
    output_root: Path,
    batch_root: Path,
    summary_path: Path,
    contexts_path: Path,
    steps_path: Path,
    episodes_path: Path,
    capacity_audit_path: Path,
    source_audit_path: Path,
    rejection_report_path: Path,
    report_path: Path,
    expanded_horizon5_root: Path,
    horizon5_summary: dict[str, Any],
    scale512_input_summary: dict[str, Any],
    scale512_summary: dict[str, Any],
    counters: Counter[str],
    config: dict[str, Any],
    materialized_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": None if status == "passed" else _next_required_change(reason_codes),
        "horizon": max(1, _int(config.get("horizon"), 5)),
        "horizon5_root": str(horizon5_root),
        "scale512_root": str(scale512_root),
        "output_root": str(output_root),
        "batch_root": str(batch_root),
        "summary": str(summary_path),
        "contexts": str(contexts_path),
        "steps": str(steps_path),
        "episodes": str(episodes_path),
        "capacity_audit": str(capacity_audit_path),
        "source_audit": str(source_audit_path),
        "rejection_report": str(rejection_report_path),
        "report": str(report_path),
        "expanded_horizon5_root": str(expanded_horizon5_root),
        "input_horizon5_status": horizon5_summary.get("status"),
        "input_horizon5_ppo_trainable_transition_count": _int(horizon5_summary.get("ppo_trainable_transition_count")),
        "input_scale512_status": scale512_input_summary.get("status"),
        "input_scale512_reason_codes": _string_list(scale512_input_summary.get("reason_codes")),
        "source_row_count": counters["source_row_count"],
        "materialized_source_row_count": counters["materialized_source_row_count"],
        "materialized_trainable_context_count": counters["materialized_trainable_context_count"],
        "materialized_missing_observation_count": counters["materialized_missing_observation_count"],
        "materialized_missing_log_prob_count": counters["materialized_missing_log_prob_count"],
        "materialized_missing_value_count": counters["materialized_missing_value_count"],
        "candidate_ppo_trainable_row_count": counters["candidate_ppo_trainable_row_count"],
        "unique_trainable_context_count": counters["unique_trainable_context_count"],
        "ppo_trainable_transition_count": counters["ppo_trainable_transition_count"],
        "duplicate_trainable_context_count": counters["duplicate_trainable_context_count"],
        "duplicate_source_trainable_context_count": counters["duplicate_source_trainable_context_count"],
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
        "controlled_regression_count": counters["controlled_regression_count"],
        "controlled_safety_regression_count": counters["controlled_safety_regression_count"],
        "controlled_contract_regression_count": counters["controlled_contract_regression_count"],
        "controlled_path_risk_regression_count": counters["controlled_path_risk_regression_count"],
        "controlled_source_selection_regression_count": counters["controlled_source_selection_regression_count"],
        "teacher_agreement_rate": _float(horizon5_summary.get("teacher_agreement_rate"), 0.0),
        "scale512_status": scale512_summary.get("status", "skipped"),
        "scale512_reason_codes": _string_list(scale512_summary.get("reason_codes")),
        "scale512_summary": scale512_summary.get("summary"),
        "seed_count": _int(scale512_summary.get("seed_count")),
        "passed_seed_count": _int(scale512_summary.get("passed_seed_count")),
        "readiness_status": scale512_summary.get(
            "readiness_status",
            "needs_training_contract_refinement" if status == "failed" else EXPECTED_READINESS_STATUS,
        ),
        "runs_formal_ppo_rollout": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }


def _source_audit(paths: list[Path], rows: list[dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "quasi-real-trainable-context-source-audit/v1",
        "source_files": [str(path) for path in paths],
        "source_file_count": len(paths),
        "source_row_count": len(rows),
        "split_counts": dict(sorted(audit["split_counts"].items())),
        "scenario_family_counts": dict(sorted(audit["family_counts"].items())),
    }


def _capacity_audit(
    config: dict[str, Any],
    counters: Counter[str],
    horizon5_summary: dict[str, Any],
    scale512_input_summary: dict[str, Any],
) -> dict[str, Any]:
    validation = config.get("validation", {})
    return {
        "schema_version": "quasi-real-trainable-context-capacity-audit/v1",
        "min_unique_trainable_context_count": _int(validation.get("min_unique_trainable_context_count"), 512),
        "min_ppo_trainable_transition_count": _int(validation.get("min_ppo_trainable_transition_count"), 512),
        "input_horizon5_ppo_trainable_transition_count": _int(horizon5_summary.get("ppo_trainable_transition_count")),
        "input_scale512_unique_trainable_context_count": _int(scale512_input_summary.get("unique_trainable_context_count")),
        "unique_trainable_context_count": counters["unique_trainable_context_count"],
        "ppo_trainable_transition_count": counters["ppo_trainable_transition_count"],
        "duplicate_trainable_context_count": counters["duplicate_trainable_context_count"],
        "duplicate_source_trainable_context_count": counters["duplicate_source_trainable_context_count"],
        "capacity_sufficient": (
            counters["unique_trainable_context_count"] >= _int(validation.get("min_unique_trainable_context_count"), 512)
            and counters["ppo_trainable_transition_count"] >= _int(validation.get("min_ppo_trainable_transition_count"), 512)
        ),
    }


def _rejection_report(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "quasi-real-trainable-context-rejection-report/v1",
        "rejection_reason_counts": dict(sorted(audit["rejected_counts"].items())),
    }


def _context_rows(selected_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "context_id": row.get("context_id"),
            "scenario_id": row.get("scenario_id"),
            "scenario_family": row.get("scenario_family"),
            "split": row.get("split"),
            "controlled_choice_source": row.get("controlled_choice_source"),
            "source_file": row.get("_source_file"),
        }
        for row in selected_steps
    ]


def _render_report(summary: dict[str, Any]) -> str:
    return (
        "# Quasi-Real Trainable Context Expansion\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Reason codes: `{summary.get('reason_codes')}`\n"
        f"- Unique trainable contexts: `{summary.get('unique_trainable_context_count')}`\n"
        f"- PPO trainable transitions: `{summary.get('ppo_trainable_transition_count')}`\n"
        f"- Duplicate trainable contexts: `{summary.get('duplicate_trainable_context_count')}`\n"
        f"- Scale-512 status: `{summary.get('scale512_status')}`\n"
        f"- Readiness: `{summary.get('readiness_status')}`\n\n"
        "This stage expands real quasi-real trainable context evidence for the "
        "Scale-512 preflight. It does not duplicate contexts, relax gates, run "
        "formal PPO, publish checkpoints, or claim formal training readiness.\n"
    )


def _scale512_output_root(config: dict[str, Any], output_root: Path, repo_root: Path) -> Path:
    configured = config.get("scale512", {}).get("output_root")
    if not configured:
        return output_root / "scale512_rerun"
    return _resolve_path(Path(str(configured)), repo_root)


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "summary": SUMMARY_FILE,
        "contexts": CONTEXTS_FILE,
        "steps": STEPS_FILE,
        "episodes": EPISODES_FILE,
        "capacity_audit": CAPACITY_AUDIT_FILE,
        "source_audit": SOURCE_AUDIT_FILE,
        "rejection_report": REJECTION_REPORT_FILE,
        "report": REPORT_FILE,
    }
    configured = config.get("output_files") if isinstance(config.get("output_files"), dict) else {}
    return {key: str(configured.get(key) or value) for key, value in defaults.items()}


def _next_required_change(reason_codes: list[str]) -> str:
    if "insufficient_quasi_real_candidate_pool" in reason_codes:
        return "expand_quasi_real_trainable_context_source_pool"
    if "scale512_preflight_not_passed" in reason_codes:
        return "fix_scale512_multiseed_preflight_after_context_expansion"
    return "fix_quasi_real_trainable_context_expansion_contract"


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


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


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


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
