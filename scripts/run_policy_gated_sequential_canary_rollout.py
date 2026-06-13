from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from git_provenance import git_snapshot as _git_snapshot

try:  # Imported as scripts.run_policy_gated_sequential_canary_rollout in tests.
    from scripts.run_controlled_hybrid_policy_training_candidate import (
        CHECKPOINT_METADATA_SCHEMA_VERSION,
    )
    from scripts.run_policy_gated_canary_rollout import (
        _canary_decision_record,
        _output_paths as _canary_output_paths,
    )
    from scripts.run_scenario_disjoint_policy_rollout_evaluation import (
        _collect_holdout_scenarios,
        _display_path,
        _score_scenarios,
        _summary_git_matches_current,
    )
except ModuleNotFoundError:  # Executed directly from scripts/.
    from run_controlled_hybrid_policy_training_candidate import (
        CHECKPOINT_METADATA_SCHEMA_VERSION,
    )
    from run_policy_gated_canary_rollout import (
        _canary_decision_record,
        _output_paths as _canary_output_paths,
    )
    from run_scenario_disjoint_policy_rollout_evaluation import (
        _collect_holdout_scenarios,
        _display_path,
        _score_scenarios,
        _summary_git_matches_current,
    )


CONFIG_SCHEMA_VERSION = "policy-gated-sequential-canary-rollout-config/v1"
SUMMARY_SCHEMA_VERSION = "policy-gated-sequential-canary-rollout-summary/v1"
EPISODE_SCHEMA_VERSION = "policy-gated-sequential-canary-episode/v1"
STEP_SCHEMA_VERSION = "policy-gated-sequential-canary-step/v1"
REJECTION_REPORT_SCHEMA_VERSION = "policy-gated-sequential-canary-rejection-report/v1"

NEXT_STATE_CONTINUITY = "sequential_canary_state_continuity_required"
NEXT_OPPORTUNITY_GAP = "sequential_canary_opportunity_generation_gap"
NEXT_OPPORTUNITY_DISTRIBUTION_GAP = "sequential_opportunity_distribution_gap_requires_more_episodes"
NEXT_SAFE_ALIGNMENT = "policy_sequential_safe_choice_alignment_insufficient"
NEXT_CUMULATIVE_REGRESSION = "sequential_canary_cumulative_regression_requires_policy_refinement"
NEXT_PROVENANCE_REFRESH = "clean_head_evidence_refresh_required"

FAMILIES = (
    "mixed_stress_detour",
    "near_blocked_safe_alt",
    "high_risk_tradeoff",
    "dense_choke_safe_bypass",
    "channel_contrast",
    "path_complexity_benefit",
)
VARIANT_SUFFIXES = ("a", "b", "c", "d", "e", "f")


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run policy-gated sequential canary rollout.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    _install_model_explorer_path(repo_root)
    source_root = _resolve_path(args.source_root, repo_root)
    candidate_root = _resolve_path(args.candidate_root, repo_root)
    batch_root = _resolve_path(args.batch_root, repo_root)
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.validate_only:
        print(json.dumps({"status": "config validated", "config": str(config_path)}, ensure_ascii=False))
        return 0

    summary, episodes, steps, rejection_report = run_policy_gated_sequential_canary_rollout(
        source_root=source_root,
        candidate_root=candidate_root,
        batch_root=batch_root,
        config=config,
        repo_root=repo_root,
    )
    outputs = _output_paths(batch_root, config)
    batch_root.mkdir(parents=True, exist_ok=True)
    _write_json(outputs["summary"], summary)
    _write_json(outputs["rejection_report"], rejection_report)
    outputs["episodes"].write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in episodes),
        encoding="utf-8",
    )
    outputs["steps"].write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in steps),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "episode_count": summary["episode_count"],
                "step_count": summary["step_count"],
                "summary": _display_path(outputs["summary"], repo_root),
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def run_policy_gated_sequential_canary_rollout(
    *,
    source_root: Path,
    candidate_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    reason_codes: list[str] = []
    candidate_summary = _load_json(
        candidate_root / config["input_files"]["candidate_summary"],
        reason_codes,
        "candidate_summary",
    )
    checkpoint_metadata = _load_json(
        candidate_root / config["input_files"]["checkpoint_metadata"],
        reason_codes,
        "checkpoint_metadata",
    )
    checkpoint = candidate_root / config["input_files"]["checkpoint"]
    source_batch = _load_json(
        source_root / config["input_files"]["source_batch_summary"],
        reason_codes,
        "source_batch_summary",
    )
    if source_batch.get("failed_count", 0) or source_batch.get("reason_codes"):
        _append_reason(reason_codes, "source_batch_failed")
    if candidate_summary.get("status") != "passed":
        _append_reason(reason_codes, "candidate_summary_failed")
    if checkpoint_metadata.get("schema_version") != CHECKPOINT_METADATA_SCHEMA_VERSION:
        _append_reason(reason_codes, "checkpoint_metadata_schema_version_mismatch")
    if not checkpoint.is_file():
        _append_reason(reason_codes, "experimental_checkpoint_missing")
    current_git = _git_snapshot(repo_root)
    allow_dirty_match = bool(config["validation"].get("allow_dirty_current_git_match"))
    candidate_git_current_matches_sources = _summary_git_matches_current(
        candidate_summary,
        current_git,
        allow_dirty_match=allow_dirty_match,
    )
    checkpoint_metadata_git_current_matches_sources = _summary_git_matches_current(
        checkpoint_metadata,
        current_git,
        allow_dirty_match=allow_dirty_match,
    )
    if config["validation"].get("require_candidate_git_current_match"):
        if not candidate_git_current_matches_sources:
            _append_reason(reason_codes, "candidate_git_current_mismatch")
        if not checkpoint_metadata_git_current_matches_sources:
            _append_reason(reason_codes, "checkpoint_metadata_git_current_mismatch")

    episodes = _episode_templates(config)
    starts = {episode["episode_id"]: list(episode["initial_start_cell"]) for episode in episodes}
    steps: list[dict[str, Any]] = []
    horizon = int(config["generation"].get("episode_horizon", 3))
    if not reason_codes:
        for step_index in range(horizon):
            step_root = batch_root / f"sequential-step-{step_index:02d}"
            spec_path = batch_root / "scenario-specs" / f"sequential-step-{step_index:02d}.json"
            _write_step_scenario_spec(
                spec_path,
                episodes,
                starts,
                step_index=step_index,
                scenario_set=str(config["generation"].get("scenario_set", "policy_canary_value_stability")),
            )
            completed = _run_step_path_feedback(
                spec_path=spec_path,
                step_root=step_root,
                config=config,
                repo_root=repo_root,
            )
            if completed.returncode != 0:
                _append_reason(reason_codes, "sequential_step_path_feedback_failed")
                break
            try:
                canary_decisions = _score_step(
                    checkpoint=checkpoint,
                    step_root=step_root,
                    config=config,
                    repo_root=repo_root,
                )
            except Exception as exc:  # noqa: BLE001
                _append_reason(reason_codes, "sequential_step_policy_scoring_failed")
                steps.append(
                    {
                        "schema_version": STEP_SCHEMA_VERSION,
                        "episode_id": "scoring",
                        "step_index": step_index,
                        "error": str(exc),
                    }
                )
                break
            decisions_by_scenario = {decision["scenario_id"]: decision for decision in canary_decisions}
            for episode in episodes:
                scenario_id = _step_scenario_id(episode, step_index)
                decision = decisions_by_scenario.get(scenario_id)
                if not decision:
                    steps.append(
                        {
                            "schema_version": STEP_SCHEMA_VERSION,
                            "episode_id": episode["episode_id"],
                            "step_index": step_index,
                            "scenario_group": episode["scenario_group"],
                            "input_start_cell": starts[episode["episode_id"]],
                            "decision_class": "missing_decision",
                            "canary_rejection_reason_codes": ["step_decision_missing"],
                        }
                    )
                    continue
                step = build_sequential_step_record(
                    decision,
                    episode_id=episode["episode_id"],
                    step_index=step_index,
                    input_start_cell=starts[episode["episode_id"]],
                )
                steps.append(step)
                controlled_goal = step.get("controlled_execution_goal_cell")
                if _valid_cell(controlled_goal):
                    starts[episode["episode_id"]] = list(controlled_goal)

    summary, rejection_report = summarize_sequential_steps(steps, config=config)
    for reason in reason_codes:
        _append_reason(summary["reason_codes"], reason)
    if reason_codes:
        summary["status"] = "failed"
    summary.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_root": _display_path(source_root, repo_root),
            "candidate_root": _display_path(candidate_root, repo_root),
            "batch_root": _display_path(batch_root, repo_root),
            "evaluation_stage": config.get("evaluation_stage"),
            "candidate_git_current_matches_sources": candidate_git_current_matches_sources,
            "checkpoint_metadata_git_current_matches_sources": checkpoint_metadata_git_current_matches_sources,
            "experimental_checkpoint": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "git_provenance": {"current": current_git, "current_matches_sources": True},
            "non_goals": list(config.get("non_goals", [])),
        }
    )
    summary["next_required_change"] = _next_required_change(summary)
    rejection_report["schema_version"] = REJECTION_REPORT_SCHEMA_VERSION
    rejection_report["reason_codes"] = summary["reason_codes"]
    return summary, episodes, steps, rejection_report


def build_sequential_step_record(
    decision: dict[str, Any],
    *,
    episode_id: str,
    step_index: int,
    input_start_cell: list[int],
) -> dict[str, Any]:
    decision_class = decision.get("decision_class")
    source_goal = _cell(decision.get("source_selected_execution_goal_cell"))
    raw_policy_goal = _cell(decision.get("raw_policy_selected_execution_goal_cell"))
    controlled_policy_goal = _cell(decision.get("policy_selected_execution_goal_cell"))
    policy_goal = raw_policy_goal or controlled_policy_goal
    if decision_class == "canary_accepted_policy_choice":
        controlled_source = "policy"
        controlled_goal = controlled_policy_goal or policy_goal
    elif decision_class == "canary_rejected_policy_choice":
        controlled_source = "source_fallback"
        controlled_goal = source_goal
    else:
        controlled_source = "source"
        controlled_goal = source_goal or controlled_policy_goal
    return {
        **decision,
        "schema_version": STEP_SCHEMA_VERSION,
        "episode_id": episode_id,
        "step_index": step_index,
        "input_start_cell": list(input_start_cell),
        "source_execution_goal_cell": source_goal,
        "policy_execution_goal_cell": policy_goal,
        "controlled_execution_goal_cell": controlled_goal,
        "controlled_choice_source": controlled_source,
    }


def summarize_sequential_steps(
    steps: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    validation = config.get("validation", {})
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        by_episode[str(step.get("episode_id", ""))].append(step)
    continuity_violations: list[dict[str, Any]] = []
    completed_episode_count = 0
    multi_step_accepted_episodes: set[str] = set()
    family_with_multi_step: set[str] = set()
    accepted_families: set[str] = set()
    canary_rejection_counts: Counter[str] = Counter()
    raw_policy_regression_counts: Counter[str] = Counter()
    controlled_regression_counts: Counter[str] = Counter()
    regression_counts: Counter[str] = Counter()
    episode_fallbacks: set[str] = set()
    for episode_id, episode_steps in by_episode.items():
        episode_steps.sort(key=lambda item: int(item.get("step_index", 0)))
        expected_horizon = int(config.get("generation", {}).get("episode_horizon") or len(episode_steps))
        if len(episode_steps) >= expected_horizon:
            completed_episode_count += 1
        accepted_count = 0
        family = str(episode_steps[0].get("scenario_group", "unknown")) if episode_steps else "unknown"
        previous_goal = None
        for step in episode_steps:
            step_index = int(step.get("step_index", 0))
            if step_index > 0 and _cell(step.get("input_start_cell")) != previous_goal:
                continuity_violations.append(
                    {
                        "episode_id": episode_id,
                        "step_index": step_index,
                        "expected_start_cell": previous_goal,
                        "actual_start_cell": _cell(step.get("input_start_cell")),
                    }
                )
            previous_goal = _cell(step.get("controlled_execution_goal_cell"))
            canary_reason_set = set(step.get("canary_rejection_reason_codes") or [])
            raw_reason_set = set(step.get("raw_policy_regression_reason_codes") or [])
            controlled_reason_set = set(step.get("controlled_regression_reason_codes") or [])
            canary_rejection_counts.update(canary_reason_set)
            raw_policy_regression_counts.update(raw_reason_set)
            controlled_regression_counts.update(controlled_reason_set)
            if not bool(step.get("action_mask_valid", True)):
                regression_counts["invalid_action_mask"] += 1
            for reason in controlled_reason_set:
                if reason in {
                    "fallback_or_open_grid",
                    "safety_regression",
                    "contract_violation",
                    "path_cost_regression",
                    "risk_regression",
                    "source_selection_regression",
                }:
                    regression_counts[reason] += 1
            if "fallback_or_open_grid" in controlled_reason_set:
                episode_fallbacks.add(episode_id)
            if step.get("decision_class") == "canary_accepted_policy_choice":
                accepted_count += 1
                accepted_families.add(family)
        if accepted_count >= 2:
            multi_step_accepted_episodes.add(episode_id)
            family_with_multi_step.add(family)

    decision_class_counts = Counter(step.get("decision_class") for step in steps)
    accepted_better_step_count = sum(
        1
        for step in steps
        if step.get("decision_class") == "canary_accepted_policy_choice"
        and step.get("accepted_choice_value_class") == "accepted_better"
    )
    policy_takeover_step_count = (
        decision_class_counts.get("canary_accepted_policy_choice", 0)
        + decision_class_counts.get("canary_rejected_policy_choice", 0)
    )
    reason_codes: list[str] = []
    metrics = {
        "episode_count": len(by_episode),
        "step_count": len(steps),
        "completed_episode_count": completed_episode_count,
        "policy_takeover_step_count": policy_takeover_step_count,
        "accepted_takeover_step_count": decision_class_counts.get("canary_accepted_policy_choice", 0),
        "accepted_better_step_count": accepted_better_step_count,
        "source_fallback_step_count": sum(1 for step in steps if step.get("controlled_choice_source") == "source_fallback"),
        "multi_step_accepted_episode_count": len(multi_step_accepted_episodes),
        "family_with_multi_step_accepted_episode_count": len(family_with_multi_step),
        "accepted_takeover_family_count": len(accepted_families),
        "state_continuity_violation_count": len(continuity_violations),
        "episode_fallback_count": len(episode_fallbacks),
        "canary_rejected_policy_choice_count": decision_class_counts.get("canary_rejected_policy_choice", 0),
        "invalid_action_mask_count": regression_counts.get("invalid_action_mask", 0),
        "fallback_or_open_grid_count": regression_counts.get("fallback_or_open_grid", 0),
        "cumulative_safety_regression_count": regression_counts.get("safety_regression", 0),
        "cumulative_contract_violation_count": regression_counts.get("contract_violation", 0),
        "cumulative_path_cost_regression_count": regression_counts.get("path_cost_regression", 0),
        "cumulative_risk_regression_count": regression_counts.get("risk_regression", 0),
        "cumulative_source_selection_regression_count": regression_counts.get("source_selection_regression", 0),
        "raw_policy_path_cost_regression_count": raw_policy_regression_counts.get("path_cost_regression", 0),
        "raw_policy_risk_regression_count": raw_policy_regression_counts.get("risk_regression", 0),
        "controlled_path_cost_regression_count": controlled_regression_counts.get("path_cost_regression", 0),
        "controlled_risk_regression_count": controlled_regression_counts.get("risk_regression", 0),
    }
    _check_min(metrics, validation, reason_codes, "episode_count")
    _check_min(metrics, validation, reason_codes, "step_count")
    _check_min(metrics, validation, reason_codes, "completed_episode_count")
    _check_min(metrics, validation, reason_codes, "policy_takeover_step_count")
    _check_min(metrics, validation, reason_codes, "accepted_takeover_step_count")
    _check_min(metrics, validation, reason_codes, "accepted_better_step_count")
    _check_min(metrics, validation, reason_codes, "accepted_takeover_family_count")
    _check_min(metrics, validation, reason_codes, "multi_step_accepted_episode_count")
    _check_min(metrics, validation, reason_codes, "family_with_multi_step_accepted_episode_count")
    _check_max(metrics, validation, reason_codes, "state_continuity_violation_count")
    _check_max(metrics, validation, reason_codes, "episode_fallback_count")
    _check_max(metrics, validation, reason_codes, "canary_rejected_policy_choice_count")
    _check_max(metrics, validation, reason_codes, "invalid_action_mask_count")
    _check_max(metrics, validation, reason_codes, "fallback_or_open_grid_count")
    _check_max(metrics, validation, reason_codes, "cumulative_safety_regression_count")
    _check_max(metrics, validation, reason_codes, "cumulative_contract_violation_count")
    _check_max(metrics, validation, reason_codes, "cumulative_path_cost_regression_count")
    _check_max(metrics, validation, reason_codes, "cumulative_risk_regression_count")
    _check_max(metrics, validation, reason_codes, "cumulative_source_selection_regression_count")
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "failed" if reason_codes else "passed",
        "reason_codes": reason_codes,
        **metrics,
        "canary_rejection_reason_counts": dict(sorted(canary_rejection_counts.items())),
        "raw_policy_regression_reason_counts": dict(sorted(raw_policy_regression_counts.items())),
        "controlled_regression_reason_counts": dict(sorted(controlled_regression_counts.items())),
        "next_required_change": None,
        "experimental_checkpoint": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "non_goals": list(config.get("non_goals", [])),
    }
    summary["next_required_change"] = _next_required_change(summary)
    rejection_report = {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "state_continuity_violations": continuity_violations,
        "canary_rejection_reason_counts": dict(sorted(canary_rejection_counts.items())),
        "raw_policy_regression_reason_counts": dict(sorted(raw_policy_regression_counts.items())),
        "controlled_regression_reason_counts": dict(sorted(controlled_regression_counts.items())),
        "failed_steps": [
            _step_with_reason_origin_counts(step)
            for step in steps
            if step.get("decision_class") in {"canary_rejected_policy_choice", "missing_decision"}
            or step.get("controlled_regression_reason_codes")
            or step.get("raw_policy_regression_reason_codes")
        ],
    }
    return summary, rejection_report


def _step_with_reason_origin_counts(step: dict[str, Any]) -> dict[str, Any]:
    row = dict(step)
    row["reason_origin_counts"] = {
        "canary_gate": _reason_count_map(step.get("canary_rejection_reason_codes")),
        "raw_policy_probe": _reason_count_map(step.get("raw_policy_regression_reason_codes")),
        "controlled_rollout": _reason_count_map(step.get("controlled_regression_reason_codes")),
    }
    return row


def _reason_count_map(value: Any) -> dict[str, int]:
    return dict(sorted(Counter(set(str(item) for item in (value or []) if str(item))).items()))


def _score_step(*, checkpoint: Path, step_root: Path, config: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    scenario_groups = _collect_holdout_scenarios(step_root, repo_root)
    base_config = {
        "evaluation": dict(config["evaluation"]),
        "validation": dict(config["validation"]),
    }
    raw_decisions = _score_scenarios(
        checkpoint_path=checkpoint,
        scenario_groups=scenario_groups,
        config=base_config,
        repo_root=repo_root,
    )
    return [_canary_decision_record(decision, config=base_config) for decision in raw_decisions]


def _episode_templates(config: dict[str, Any]) -> list[dict[str, Any]]:
    generation = config.get("generation", {})
    families = generation.get("families") or list(FAMILIES)
    variants = generation.get("variant_suffixes") or list(VARIANT_SUFFIXES)
    initial_start_cell = generation.get("initial_start_cell", [1, 6])
    initial_start_cells_by_episode = generation.get("initial_start_cells_by_episode")
    if not isinstance(initial_start_cells_by_episode, dict):
        initial_start_cells_by_episode = {}
    template_prefix = str(generation.get("template_scenario_id_prefix", "npz_canary_value_stability"))
    episodes: list[dict[str, Any]] = []
    for family in families:
        for suffix in variants:
            episode_id = f"seq-{family}-{suffix}"
            start_cell = initial_start_cells_by_episode.get(episode_id, initial_start_cell)
            episodes.append(
                {
                    "schema_version": EPISODE_SCHEMA_VERSION,
                    "episode_id": episode_id,
                    "scenario_group": str(family),
                    "variant_suffix": str(suffix),
                    "template_scenario_id": f"{template_prefix}_{family}_{suffix}",
                    "initial_start_cell": list(start_cell),
                }
            )
    return episodes


def _write_step_scenario_spec(
    path: Path,
    episodes: list[dict[str, Any]],
    starts: dict[str, list[int]],
    *,
    step_index: int,
    scenario_set: str = "policy_canary_value_stability",
) -> None:
    scenarios = []
    seed_base = 14000 + step_index * 1000
    for index, episode in enumerate(episodes):
        scenario_id = _step_scenario_id(episode, step_index)
        scenarios.append(
            {
                "template_scenario_id": episode["template_scenario_id"],
                "scenario_id": scenario_id,
                "scenario_group": episode["scenario_group"],
                "scenario_seed": seed_base + index,
                "scenario_variant_id": f"{scenario_id}-seed-{seed_base + index}",
                "start_cell": starts[episode["episode_id"]],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "npz-validation-explicit-scenario-spec/v1",
                "scenario_set": scenario_set,
                "scenarios": scenarios,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _run_step_path_feedback(
    *,
    spec_path: Path,
    step_root: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> subprocess.CompletedProcess[str]:
    generation = config.get("generation", {})
    script = repo_root / "scripts" / "run_path_feedback_validation.sh"
    argv = [
        "bash",
        str(script),
        "--scenario-set",
        str(generation.get("scenario_set", "policy_canary_value_stability")),
        "--scenario-spec-json",
        str(spec_path),
        "--diagnostic-profile",
        str(generation.get("diagnostic_profile", "execution")),
        "--top-k",
        str(generation.get("top_k", 3)),
        "--output-root",
        str(step_root),
        "--anchor-projection-candidate-generation",
        "--anchor-projection-contract-aware-trainable-target-generation",
        "--anchor-projection-prefer-contract-safe-trainable-targets",
        "--anchor-projection-planner-validated-trainable-target-mining",
        "--anchor-projection-allow-planner-validated-distance-exception",
    ]
    argv.extend(str(item) for item in generation.get("planner_extra_args", []))
    return subprocess.run(
        argv,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _step_scenario_id(episode: dict[str, Any], step_index: int) -> str:
    return f"npz_seq_canary_{episode['scenario_group']}_{episode['variant_suffix']}_step{step_index:02d}"


def _next_required_change(summary: dict[str, Any]) -> str | None:
    reasons = set(summary.get("reason_codes", []))
    if not reasons:
        return None
    if {"candidate_git_current_mismatch", "checkpoint_metadata_git_current_mismatch"} & reasons:
        return NEXT_PROVENANCE_REFRESH
    if "state_continuity_violation" in reasons:
        return NEXT_STATE_CONTINUITY
    if "episode_count_below_threshold" in reasons or "step_count_below_threshold" in reasons:
        return NEXT_OPPORTUNITY_GAP
    if any(reason.startswith("cumulative_") or reason in {"invalid_action_mask_count_above_threshold", "fallback_or_open_grid_count_above_threshold"} for reason in reasons):
        return NEXT_CUMULATIVE_REGRESSION
    multi_step_reasons = {
        "multi_step_accepted_episode_count_below_threshold",
        "family_with_multi_step_accepted_episode_count_below_threshold",
    }
    if (
        reasons <= multi_step_reasons
        and int(summary.get("canary_rejected_policy_choice_count", 0)) == 0
        and int(summary.get("cumulative_path_cost_regression_count", 0)) == 0
        and int(summary.get("cumulative_risk_regression_count", 0)) == 0
        and int(summary.get("accepted_takeover_family_count", 0)) >= 1
    ):
        return NEXT_OPPORTUNITY_DISTRIBUTION_GAP
    if {
        "policy_takeover_step_count_below_threshold",
        "accepted_takeover_step_count_below_threshold",
        "accepted_better_step_count_below_threshold",
        "accepted_takeover_family_count_below_threshold",
        "multi_step_accepted_episode_count_below_threshold",
        "family_with_multi_step_accepted_episode_count_below_threshold",
    } & reasons:
        return NEXT_SAFE_ALIGNMENT
    return "policy_gated_sequential_canary_rollout_refinement_required"


def _check_min(metrics: dict[str, int], validation: dict[str, Any], reasons: list[str], key: str) -> None:
    threshold = validation.get(f"min_{key}")
    if threshold is not None and int(metrics.get(key, 0)) < int(threshold):
        _append_reason(reasons, f"{key}_below_threshold")


def _check_max(metrics: dict[str, int], validation: dict[str, Any], reasons: list[str], key: str) -> None:
    threshold = validation.get(f"max_{key}")
    if threshold is not None and int(metrics.get(key, 0)) > int(threshold):
        _append_reason(reasons, f"{key}_above_threshold")
    if key == "state_continuity_violation_count" and int(metrics.get(key, 0)) > 0:
        _append_reason(reasons, "state_continuity_violation")


def _output_paths(batch_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["output_files"]
    return {
        "episodes": batch_root / outputs["episodes"],
        "steps": batch_root / outputs["steps"],
        "rejection_report": batch_root / outputs["rejection_report"],
        "summary": batch_root / outputs["summary"],
    }


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {exc}") from exc
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    for section in ("input_files", "output_files", "generation", "evaluation", "validation"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _load_json(path: Path, reason_codes: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _append_reason(reason_codes, f"{label}_invalid_json")
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _cell(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return [int(value[0]), int(value[1])]
        except (TypeError, ValueError):
            return None
    return None


def _valid_cell(value: Any) -> bool:
    return _cell(value) is not None


def _resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _install_model_explorer_path(repo_root: Path) -> None:
    source = repo_root / "model-explorer" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


if __name__ == "__main__":
    raise SystemExit(main())
