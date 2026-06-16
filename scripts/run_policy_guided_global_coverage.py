from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from math import isfinite
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODEL_EXPLORER_SRC = REPO_ROOT / "model-explorer" / "src"
for import_path in (SCRIPT_DIR, MODEL_EXPLORER_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

try:
    from frontier_coverage_planner_common import (
        coverage_rate,
        enumerate_frontier_candidates,
        frontier_cells,
        score_frontier_candidate,
        select_frontier_candidate,
    )
    from git_provenance import git_snapshot
    from global_99_coverage_contract import (
        Cell,
        ConfigError,
        TOLERANCE,
        coverage_footprint,
        load_global_99_config,
        resolve_path,
        scenario_geometry,
        unique_sorted,
        utc_now,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.frontier_coverage_planner_common import (
        coverage_rate,
        enumerate_frontier_candidates,
        frontier_cells,
        score_frontier_candidate,
        select_frontier_candidate,
    )
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import (
        Cell,
        ConfigError,
        TOLERANCE,
        coverage_footprint,
        load_global_99_config,
        resolve_path,
        scenario_geometry,
        unique_sorted,
        utc_now,
        write_json,
        write_jsonl,
    )

from model_explorer.core.interfaces import (
    ConstraintSummary,
    GoalCandidate,
    GridSummary,
    ModelExplorerContract,
    MODEL_EXPLORER_SCHEMA_VERSION,
)
from model_explorer.policy.architectures import build_policy_network
from model_explorer.policy.features import extract_policy_observation
from model_explorer.policy.torch_policy import TorchPolicyScorer


CONFIG_SCHEMA_VERSION = "policy-guided-global-coverage-config/v1"
FRONTIER_CONFIG_SCHEMA_VERSION = "frontier-coverage-planner-baseline-config/v1"
MEMORY_CONFIG_SCHEMA_VERSION = "coverage-memory-replanning-loop-config/v1"
SUMMARY_SCHEMA_VERSION = "policy-guided-global-coverage-summary/v1"
MANIFEST_SCHEMA_VERSION = "policy-guided-global-coverage-manifest/v1"
DECISION_ROW_SCHEMA_VERSION = "policy-guided-global-coverage-decision-row/v1"
LEDGER_ROW_SCHEMA_VERSION = "policy-guided-global-coverage-ledger-row/v1"
SNAPSHOT_ROW_SCHEMA_VERSION = "policy-guided-global-coverage-memory-snapshot-row/v1"
POLICY_SCORE_AUDIT_SCHEMA_VERSION = "policy-guided-global-coverage-policy-score-audit/v1"
GUARD_AUDIT_SCHEMA_VERSION = "policy-guided-global-coverage-guard-audit/v1"
BUDGET_AUDIT_SCHEMA_VERSION = "policy-guided-global-coverage-budget-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "policy-guided-global-coverage-rejection-report/v1"

DEFAULT_CONFIG = "configs/policy_guided_global_coverage_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_policy_guided_global_coverage_v1"

SUMMARY_FILE = "policy-guided-global-coverage-summary.json"
MANIFEST_FILE = "policy-guided-global-coverage-manifest.json"
DECISIONS_FILE = "policy-guided-global-coverage-decisions.jsonl"
LEDGER_FILE = "policy-guided-global-coverage-ledger.jsonl"
SNAPSHOTS_FILE = "policy-guided-global-coverage-memory-snapshots.jsonl"
POLICY_SCORE_AUDIT_FILE = "policy-guided-global-coverage-policy-score-audit.json"
GUARD_AUDIT_FILE = "policy-guided-global-coverage-guard-audit.json"
BUDGET_AUDIT_FILE = "policy-guided-global-coverage-budget-audit.json"
REJECTION_REPORT_FILE = "policy-guided-global-coverage-rejection-report.json"
REPORT_FILE = "policy-guided-global-coverage-report.md"

PASS_NEXT_REQUIRED_CHANGE = "global_99_multi_map_generalization"
FAIL_NEXT_REQUIRED_CHANGE = "fix_policy_guided_global_coverage"


class CheckpointPolicyBundle:
    def __init__(
        self,
        *,
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        state_key: str,
    ) -> None:
        self.checkpoint = checkpoint
        self.metadata = metadata
        self.state_key = state_key
        self._scorer: TorchPolicyScorer | None = None

    def score(self, observation) -> tuple[float, ...]:
        if self._scorer is None:
            state_dict = self.checkpoint[self.state_key]
            training = self.checkpoint.get("training") if isinstance(self.checkpoint.get("training"), dict) else {}
            architecture = self.metadata.get("architecture") or self.checkpoint.get("architecture")
            hidden_size = (
                self.checkpoint.get("hidden_size")
                or training.get("hidden_size")
                or self.metadata.get("hidden_size")
            )
            if hidden_size is None:
                raise ValueError("checkpoint is missing hidden_size")
            network = build_policy_network(
                architecture,
                observation=observation,
                hidden_size=int(hidden_size),
                architecture_config=self.metadata.get("architecture_config"),
            )
            network.load_state_dict(state_dict)
            network.eval()
            self._scorer = TorchPolicyScorer(network)
        return self._scorer.score(observation)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Policy-Guided Global Coverage v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_policy_guided_global_coverage(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "achieved_coverage_rate": summary["achieved_coverage_rate"],
                "coverage_target_met": summary["coverage_target_met"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_policy_guided_global_coverage(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config_path = Path(config_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = _load_config(config_path)
    source_config_path = _resolve_config_reference(config["source_global_99_config"], config_path, repo_root)
    frontier_config_path = _resolve_config_reference(config["source_frontier_baseline_config"], config_path, repo_root)
    memory_config_path = _resolve_config_reference(config["source_coverage_memory_config"], config_path, repo_root)
    source_config = load_global_99_config(source_config_path)
    frontier_config = _load_frontier_config(frontier_config_path)
    memory_config = _load_memory_config(memory_config_path)
    policy_bundle, policy_load = _load_policy_bundle(config, config_path, repo_root)

    target = float(config["target_coverage_rate"])
    path_budget_m = float(config["path_budget_m"])
    scenario_results = [
        _run_policy_guided_scenario(
            scenario,
            target_coverage_rate=target,
            path_budget_m=path_budget_m,
            coverage_radius_cells=int(config["coverage_radius_cells"]),
            replanning_cycle_limit=int(config["replanning_cycle_limit"]),
            segment_step_limit=int(config["segment_step_limit"]),
            memory_snapshot_interval=int(config["memory_snapshot_interval"]),
            max_policy_candidates=int(config["max_policy_candidates"]),
            policy_logit_weight=float(config["policy_logit_weight"]),
            revisit_penalty_weight=float(frontier_config["revisit_penalty_weight"]),
            new_coverage_weight=float(frontier_config["new_coverage_weight"]),
            policy_bundle=policy_bundle,
        )
        for scenario in source_config["scenarios"]
    ]

    decision_rows = [row for result in scenario_results for row in result["decision_rows"]]
    ledger_rows = [row for result in scenario_results for row in result["ledger_rows"]]
    snapshot_rows = [row for result in scenario_results for row in result["snapshot_rows"]]
    total_denominator = sum(result["reachable_safe_cell_count"] for result in scenario_results)
    total_covered = sum(result["covered_reachable_safe_cell_count"] for result in scenario_results)
    achieved = (total_covered / total_denominator) if total_denominator else 0.0
    coverage_target_met = (
        total_denominator > 0
        and achieved + TOLERANCE >= target
        and all(result["coverage_target_met"] for result in scenario_results)
    )
    policy_scored_candidate_count = sum(result["policy_scored_candidate_count"] for result in scenario_results)
    policy_guided_decision_count = sum(result["policy_guided_decision_count"] for result in scenario_results)
    policy_selected_decision_count = sum(result["policy_selected_decision_count"] for result in scenario_results)
    policy_guard_fallback_count = sum(result["policy_guard_fallback_count"] for result in scenario_results)
    controlled_regression_count = sum(result["controlled_regression_count"] for result in scenario_results)
    baseline_agreement_denominator = sum(result["baseline_agreement_denominator"] for result in scenario_results)
    baseline_agreement_count = sum(result["baseline_agreement_count"] for result in scenario_results)
    baseline_agreement_rate = (
        baseline_agreement_count / baseline_agreement_denominator if baseline_agreement_denominator else 0.0
    )
    policy_guidance_applied = policy_load["policy_loaded"] and policy_guided_decision_count > 0
    reason_codes = _summary_reason_codes(
        scenario_results=scenario_results,
        coverage_target_met=coverage_target_met,
        total_denominator=total_denominator,
        policy_load=policy_load,
        policy_guidance_applied=policy_guidance_applied,
        policy_scored_candidate_count=policy_scored_candidate_count,
    )
    status = "passed" if not reason_codes else "failed"
    next_required_change = PASS_NEXT_REQUIRED_CHANGE if status == "passed" else FAIL_NEXT_REQUIRED_CHANGE
    planned_path_cost_m = sum(result["planned_path_cost_m"] for result in scenario_results)
    infeasible_reason_codes = unique_sorted(
        reason for result in scenario_results for reason in result["infeasible_reason_codes"]
    )
    paths = {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "decisions": output_root / DECISIONS_FILE,
        "ledger": output_root / LEDGER_FILE,
        "snapshots": output_root / SNAPSHOTS_FILE,
        "policy_score_audit": output_root / POLICY_SCORE_AUDIT_FILE,
        "guard_audit": output_root / GUARD_AUDIT_FILE,
        "budget_audit": output_root / BUDGET_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "config": str(config_path),
        "source_global_99_config": str(source_config_path),
        "source_frontier_baseline_config": str(frontier_config_path),
        "source_coverage_memory_config": str(memory_config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "decisions": str(paths["decisions"]),
        "ledger": str(paths["ledger"]),
        "memory_snapshots": str(paths["snapshots"]),
        "policy_score_audit": str(paths["policy_score_audit"]),
        "guard_audit": str(paths["guard_audit"]),
        "budget_audit": str(paths["budget_audit"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "target_coverage_rate": target,
        "achieved_coverage_rate": achieved,
        "coverage_target_met": coverage_target_met,
        "reachable_safe_cell_count": total_denominator,
        "covered_reachable_safe_cell_count": total_covered,
        "policy_loaded": policy_load["policy_loaded"],
        "policy_checkpoint_state_key": policy_load.get("policy_checkpoint_state_key"),
        "policy_guidance_applied": policy_guidance_applied,
        "policy_scored_candidate_count": policy_scored_candidate_count,
        "policy_guided_decision_count": policy_guided_decision_count,
        "policy_selected_decision_count": policy_selected_decision_count,
        "policy_guard_fallback_count": policy_guard_fallback_count,
        "baseline_agreement_rate": baseline_agreement_rate,
        "controlled_regression_count": controlled_regression_count,
        "planned_path_cost_m": planned_path_cost_m,
        "path_budget_m": path_budget_m,
        "path_budget_exhausted": any(result["path_budget_exhausted"] for result in scenario_results),
        "coverage_memory_complete": status == "passed",
        "scenario_count": len(scenario_results),
        "passed_scenario_count": sum(1 for result in scenario_results if result["status"] == "passed"),
        "failed_scenario_count": sum(1 for result in scenario_results if result["status"] == "failed"),
        "infeasible_reason_codes": infeasible_reason_codes,
        "coverage_radius_cells": config["coverage_radius_cells"],
        "replanning_cycle_limit": config["replanning_cycle_limit"],
        "segment_step_limit": config["segment_step_limit"],
        "memory_snapshot_interval": config["memory_snapshot_interval"],
        "max_policy_candidates": config["max_policy_candidates"],
        "policy_logit_weight": config["policy_logit_weight"],
        "revisit_penalty_weight": frontier_config["revisit_penalty_weight"],
        "new_coverage_weight": frontier_config["new_coverage_weight"],
        "source_memory_config_schema_version": memory_config["schema_version"],
        "next_required_change": next_required_change,
        "scenario_summaries": [_public_scenario_summary(result) for result in scenario_results],
        "policy_load": policy_load,
        "uses_policy_guidance": True,
        "uses_checkpoint_inference": True,
        "uses_ppo_policy": True,
        "policy_read_only": True,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "checkpoint_publication_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "runs_new_ppo_update": False,
        "modifies_network": False,
        "modifies_action_space": False,
        "modifies_default_astar": False,
        "ackermann_feasible_trajectory_claimed": False,
        "performance_claimed": False,
        "real_world_performance_claimed": False,
        "uses_default_policy": False,
        "uses_path_planner": False,
        "uses_npz_or_sidecar": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }
    policy_score_audit = _policy_score_audit(config, policy_load, scenario_results)
    guard_audit = _guard_audit(scenario_results)
    budget_audit = _budget_audit(config, frontier_config, scenario_results, planned_path_cost_m)
    rejection_report = _rejection_report(reason_codes, scenario_results)
    manifest = _manifest(config_path, output_root, paths, summary)

    write_jsonl(paths["decisions"], decision_rows)
    write_jsonl(paths["ledger"], ledger_rows)
    write_jsonl(paths["snapshots"], snapshot_rows)
    write_json(paths["policy_score_audit"], policy_score_audit)
    write_json(paths["guard_audit"], guard_audit)
    write_json(paths["budget_audit"], budget_audit)
    write_json(paths["rejection_report"], rejection_report)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, rejection_report), encoding="utf-8")
    return summary


def _run_policy_guided_scenario(
    scenario: dict[str, Any],
    *,
    target_coverage_rate: float,
    path_budget_m: float,
    coverage_radius_cells: int,
    replanning_cycle_limit: int,
    segment_step_limit: int,
    memory_snapshot_interval: int,
    max_policy_candidates: int,
    policy_logit_weight: float,
    revisit_penalty_weight: float,
    new_coverage_weight: float,
    policy_bundle: CheckpointPolicyBundle | None,
) -> dict[str, Any]:
    geometry = scenario_geometry(scenario)
    scenario_id = geometry["scenario_id"]
    width = geometry["width"]
    height = geometry["height"]
    resolution_m = geometry["resolution_m"]
    navigation_cells: set[Cell] = geometry["navigation_cells"]
    target_cells: set[Cell] = geometry["reachable_safe_cells"]
    start: Cell = geometry["start"]
    decision_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    reason_codes: list[str] = []
    path_budget_exhausted = False
    replanning_cycle_limit_exhausted = False
    frontier_exhausted = False
    frontier_unreachable = False
    invalid_policy_scores = False
    no_memory_progress = False
    planned_path_cost_m = 0.0
    completed_cycles = 0
    current_cell = start
    policy_scored_candidate_count = 0
    policy_guided_decision_count = 0
    policy_selected_decision_count = 0
    policy_guard_fallback_count = 0
    baseline_agreement_count = 0
    baseline_agreement_denominator = 0
    controlled_regression_count = 0

    coverage_memory_cells = coverage_footprint(start, width, height, coverage_radius_cells) & navigation_cells
    covered_target_cells = coverage_memory_cells & target_cells
    _append_decision_and_ledger_row(
        decision_rows=decision_rows,
        ledger_rows=ledger_rows,
        scenario_id=scenario_id,
        cycle_index=0,
        event_id=f"{scenario_id}-initial-policy-memory",
        selected_waypoint=start,
        baseline_waypoint=start,
        policy_waypoint=None,
        path=[start] if start in navigation_cells else [],
        event_cells=coverage_memory_cells,
        counted=True,
        attempted_path_cost_m=0.0,
        planned_path_cost_m=0.0,
        path_budget_m=path_budget_m,
        path_cost_m=0.0,
        baseline_score=0.0,
        policy_logit=None,
        combined_score=None,
        revisited_path_cell_count=0,
        new_target_cell_count=len(covered_target_cells),
        target_cells=target_cells,
        covered_target_cells=covered_target_cells,
        frontier_cluster_count=0,
        choice_source="initial_memory",
        guard_reason_codes=[],
    )
    _append_snapshot_row(
        snapshot_rows=snapshot_rows,
        scenario_id=scenario_id,
        cycle_index=0,
        current_cell=current_cell,
        coverage_memory_cells=coverage_memory_cells,
        target_cells=target_cells,
        planned_path_cost_m=planned_path_cost_m,
        path_budget_m=path_budget_m,
    )

    while coverage_rate(covered_target_cells, target_cells) + TOLERANCE < target_coverage_rate:
        if completed_cycles >= replanning_cycle_limit:
            replanning_cycle_limit_exhausted = True
            break
        frontier = frontier_cells(coverage_memory_cells, navigation_cells)
        if not frontier:
            frontier_exhausted = True
            break
        baseline_candidate = select_frontier_candidate(
            current_cell=current_cell,
            frontier=frontier,
            navigation_cells=navigation_cells,
            target_cells=target_cells,
            covered_target_cells=covered_target_cells,
            coverage_map_cells=coverage_memory_cells,
            width=width,
            height=height,
            resolution_m=resolution_m,
            coverage_radius_cells=coverage_radius_cells,
            revisit_penalty_weight=revisit_penalty_weight,
            new_coverage_weight=new_coverage_weight,
        )
        candidates = enumerate_frontier_candidates(
            current_cell=current_cell,
            frontier=frontier,
            navigation_cells=navigation_cells,
            target_cells=target_cells,
            covered_target_cells=covered_target_cells,
            coverage_map_cells=coverage_memory_cells,
            width=width,
            height=height,
            resolution_m=resolution_m,
            coverage_radius_cells=coverage_radius_cells,
            revisit_penalty_weight=revisit_penalty_weight,
            new_coverage_weight=new_coverage_weight,
            component_best_only=False,
        )
        if baseline_candidate is None or not candidates:
            frontier_unreachable = True
            break
        candidates = _policy_candidate_subset(baseline_candidate, candidates, max_policy_candidates)
        policy_choice, score_rows, score_error = _policy_choice(
            policy_bundle=policy_bundle,
            candidates=candidates,
            baseline_candidate=baseline_candidate,
            geometry=geometry,
            current_cell=current_cell,
            covered_target_cells=covered_target_cells,
            target_cells=target_cells,
            coverage_memory_cells=coverage_memory_cells,
            cycle_index=completed_cycles,
            replanning_cycle_limit=replanning_cycle_limit,
            policy_logit_weight=policy_logit_weight,
        )
        if score_error is not None:
            invalid_policy_scores = True
            break

        policy_scored_candidate_count += len(score_rows)
        policy_guided_decision_count += 1
        baseline_agreement_denominator += 1
        baseline_agreement = _same_waypoint(policy_choice, baseline_candidate)
        if baseline_agreement:
            baseline_agreement_count += 1
        guard_reason_codes: list[str] = []
        selected_candidate = policy_choice
        attempted_policy_path_cost_m = planned_path_cost_m + policy_choice["path_cost_m"]
        if attempted_policy_path_cost_m > path_budget_m + TOLERANCE:
            guard_reason_codes.append("policy_choice_over_budget")
        if (
            policy_choice["new_target_cell_count"] <= 0
            and baseline_candidate["new_target_cell_count"] > 0
        ):
            guard_reason_codes.append("policy_choice_no_coverage_progress")
        if (
            policy_choice["path_cost_m"] > baseline_candidate["path_cost_m"] + TOLERANCE
            and policy_choice["new_target_cell_count"] <= baseline_candidate["new_target_cell_count"]
        ):
            guard_reason_codes.append("policy_choice_controlled_regression")
        if guard_reason_codes:
            policy_guard_fallback_count += 1
            selected_candidate = baseline_candidate
        else:
            policy_selected_decision_count += 1
            if (
                not baseline_agreement
                and policy_choice["path_cost_m"] > baseline_candidate["path_cost_m"] + TOLERANCE
                and policy_choice["new_target_cell_count"] <= baseline_candidate["new_target_cell_count"]
            ):
                controlled_regression_count += 1

        segment_path = selected_candidate["path"][: segment_step_limit + 1]
        if not segment_path:
            frontier_unreachable = True
            break
        selected_waypoint = segment_path[-1]
        path_cells = set(segment_path)
        footprint = coverage_footprint(selected_waypoint, width, height, coverage_radius_cells) & navigation_cells
        event_cells = footprint | path_cells
        new_target_cells = (event_cells & target_cells) - covered_target_cells
        revisited_path_cell_count = sum(1 for path_cell in segment_path if path_cell in coverage_memory_cells)
        path_cost_m = (len(segment_path) - 1) * resolution_m
        selected_score = score_frontier_candidate(
            path_cost_m=path_cost_m,
            revisited_path_cell_count=revisited_path_cell_count,
            new_covered_cell_count=len(new_target_cells),
            revisit_penalty_weight=revisit_penalty_weight,
            new_coverage_weight=new_coverage_weight,
        )
        attempted_path_cost_m = planned_path_cost_m + path_cost_m
        cycle_index = completed_cycles + 1
        policy_score_row = _score_row_for_candidate(score_rows, policy_choice)
        if attempted_path_cost_m > path_budget_m + TOLERANCE:
            path_budget_exhausted = True
            _append_decision_and_ledger_row(
                decision_rows=decision_rows,
                ledger_rows=ledger_rows,
                scenario_id=scenario_id,
                cycle_index=cycle_index,
                event_id=f"{scenario_id}-policy-{cycle_index:04d}",
                selected_waypoint=selected_waypoint,
                baseline_waypoint=baseline_candidate["selected_waypoint"],
                policy_waypoint=policy_choice["selected_waypoint"],
                path=segment_path,
                event_cells=event_cells,
                counted=False,
                attempted_path_cost_m=attempted_path_cost_m,
                planned_path_cost_m=planned_path_cost_m,
                path_budget_m=path_budget_m,
                path_cost_m=path_cost_m,
                baseline_score=baseline_candidate["score"],
                policy_logit=policy_score_row.get("policy_logit") if policy_score_row else None,
                combined_score=policy_score_row.get("combined_score") if policy_score_row else None,
                revisited_path_cell_count=revisited_path_cell_count,
                new_target_cell_count=0,
                target_cells=target_cells,
                covered_target_cells=covered_target_cells,
                frontier_cluster_count=selected_candidate["frontier_cluster_count"],
                choice_source="path_budget_exhausted",
                guard_reason_codes=guard_reason_codes,
            )
            break

        before_count = len(covered_target_cells)
        before_memory_count = len(coverage_memory_cells)
        planned_path_cost_m = attempted_path_cost_m
        current_cell = selected_waypoint
        coverage_memory_cells |= event_cells
        covered_target_cells = coverage_memory_cells & target_cells
        new_target_cell_count = len(covered_target_cells) - before_count
        if len(coverage_memory_cells) == before_memory_count:
            no_memory_progress = True
            break
        _append_decision_and_ledger_row(
            decision_rows=decision_rows,
            ledger_rows=ledger_rows,
            scenario_id=scenario_id,
            cycle_index=cycle_index,
            event_id=f"{scenario_id}-policy-{cycle_index:04d}",
            selected_waypoint=selected_waypoint,
            baseline_waypoint=baseline_candidate["selected_waypoint"],
            policy_waypoint=policy_choice["selected_waypoint"],
            path=segment_path,
            event_cells=event_cells,
            counted=True,
            attempted_path_cost_m=attempted_path_cost_m,
            planned_path_cost_m=planned_path_cost_m,
            path_budget_m=path_budget_m,
            path_cost_m=path_cost_m,
            baseline_score=baseline_candidate["score"],
            policy_logit=policy_score_row.get("policy_logit") if policy_score_row else None,
            combined_score=policy_score_row.get("combined_score") if policy_score_row else None,
            revisited_path_cell_count=revisited_path_cell_count,
            new_target_cell_count=new_target_cell_count,
            target_cells=target_cells,
            covered_target_cells=covered_target_cells,
            frontier_cluster_count=selected_candidate["frontier_cluster_count"],
            choice_source="policy_guided_global_coverage",
            guard_reason_codes=guard_reason_codes,
        )
        completed_cycles = cycle_index
        if completed_cycles % memory_snapshot_interval == 0:
            _append_snapshot_row(
                snapshot_rows=snapshot_rows,
                scenario_id=scenario_id,
                cycle_index=completed_cycles,
                current_cell=current_cell,
                coverage_memory_cells=coverage_memory_cells,
                target_cells=target_cells,
                planned_path_cost_m=planned_path_cost_m,
                path_budget_m=path_budget_m,
            )

    target_met = bool(target_cells) and coverage_rate(covered_target_cells, target_cells) + TOLERANCE >= target_coverage_rate
    if not target_cells:
        reason_codes.append("coverage_denominator_invalid")
    if target_cells and not target_met:
        reason_codes.append("coverage_target_not_met")
    if path_budget_exhausted:
        reason_codes.append("insufficient_budget")
    if replanning_cycle_limit_exhausted:
        reason_codes.append("replanning_cycle_limit_exhausted")
    if frontier_exhausted:
        reason_codes.append("frontier_exhausted")
    if frontier_unreachable:
        reason_codes.append("frontier_unreachable")
    if invalid_policy_scores:
        reason_codes.append("invalid_policy_scores")
    if no_memory_progress:
        reason_codes.append("coverage_memory_no_progress")

    infeasible_reason_codes: list[str] = []
    if geometry["unsafe_roi_cells"]:
        infeasible_reason_codes.append("unsafe_roi_cells")
    if geometry["unreachable_roi_cells"]:
        infeasible_reason_codes.append("unreachable_roi_cells")
    if path_budget_exhausted:
        infeasible_reason_codes.append("insufficient_budget")
    if not target_cells:
        infeasible_reason_codes.append("coverage_denominator_invalid")
    if target_cells and not target_met:
        infeasible_reason_codes.append("coverage_target_not_met")

    return {
        "scenario_id": scenario_id,
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": unique_sorted(reason_codes),
        "infeasible_reason_codes": unique_sorted(infeasible_reason_codes),
        "target_coverage_rate": target_coverage_rate,
        "achieved_coverage_rate": coverage_rate(covered_target_cells, target_cells),
        "coverage_target_met": target_met,
        "reachable_safe_cell_count": len(target_cells),
        "covered_reachable_safe_cell_count": len(covered_target_cells),
        "policy_scored_candidate_count": policy_scored_candidate_count,
        "policy_guided_decision_count": policy_guided_decision_count,
        "policy_selected_decision_count": policy_selected_decision_count,
        "policy_guard_fallback_count": policy_guard_fallback_count,
        "baseline_agreement_count": baseline_agreement_count,
        "baseline_agreement_denominator": baseline_agreement_denominator,
        "controlled_regression_count": controlled_regression_count,
        "planned_path_cost_m": planned_path_cost_m,
        "path_budget_m": path_budget_m,
        "path_budget_exhausted": path_budget_exhausted,
        "coverage_memory_complete": target_met and not reason_codes,
        "replanning_cycle_count": completed_cycles,
        "memory_snapshot_count": len(snapshot_rows),
        "replanning_cycle_limit_exhausted": replanning_cycle_limit_exhausted,
        "frontier_exhausted": frontier_exhausted,
        "frontier_unreachable": frontier_unreachable,
        "roi_cell_count": len(geometry["roi_cells"]),
        "blocked_roi_cell_count": len(geometry["roi_cells"] & geometry["blocked_cells"]),
        "unsafe_roi_cell_count": len(geometry["roi_cells"] & geometry["unsafe_cells"]),
        "unreachable_roi_cell_count": len(geometry["unreachable_roi_cells"]),
        "decision_rows": decision_rows,
        "ledger_rows": ledger_rows,
        "snapshot_rows": snapshot_rows,
    }


def _policy_choice(
    *,
    policy_bundle: CheckpointPolicyBundle | None,
    candidates: list[dict[str, Any]],
    baseline_candidate: dict[str, Any],
    geometry: dict[str, Any],
    current_cell: Cell,
    covered_target_cells: set[Cell],
    target_cells: set[Cell],
    coverage_memory_cells: set[Cell],
    cycle_index: int,
    replanning_cycle_limit: int,
    policy_logit_weight: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    if policy_bundle is None:
        return baseline_candidate, [], "policy_guidance_unavailable"
    contract = _contract_for_candidates(
        candidates,
        geometry=geometry,
        current_cell=current_cell,
        covered_target_cells=covered_target_cells,
        target_cells=target_cells,
        coverage_memory_cells=coverage_memory_cells,
    )
    observation = extract_policy_observation(
        contract,
        current_cell=current_cell,
        step_index=cycle_index,
        remaining_steps=max(replanning_cycle_limit - cycle_index, 0),
        max_candidates=len(candidates),
    )
    try:
        logits = policy_bundle.score(observation)
    except Exception:
        return baseline_candidate, [], "invalid_policy_scores"
    if len(logits) != len(candidates) or any(not isfinite(float(logit)) for logit in logits):
        return baseline_candidate, [], "invalid_policy_scores"
    normalized_logits = _min_max_normalize(tuple(float(logit) for logit in logits))
    score_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        combined_score = candidate["score"] - policy_logit_weight * normalized_logits[index]
        score_rows.append(
            {
                "candidate_index": index,
                "selected_waypoint": list(candidate["selected_waypoint"]),
                "baseline_score": candidate["score"],
                "policy_logit": float(logits[index]),
                "normalized_policy_logit": normalized_logits[index],
                "combined_score": combined_score,
                "path_cost_m": candidate["path_cost_m"],
                "new_target_cell_count": candidate["new_target_cell_count"],
            }
        )
    ranked = sorted(
        zip(candidates, score_rows),
        key=lambda item: (
            item[1]["combined_score"],
            -item[0]["new_target_cell_count"],
            item[0]["path_cost_m"],
            item[0]["selected_waypoint"][1],
            item[0]["selected_waypoint"][0],
        ),
    )
    return ranked[0][0], score_rows, None


def _contract_for_candidates(
    candidates: list[dict[str, Any]],
    *,
    geometry: dict[str, Any],
    current_cell: Cell,
    covered_target_cells: set[Cell],
    target_cells: set[Cell],
    coverage_memory_cells: set[Cell],
) -> ModelExplorerContract:
    width = int(geometry["width"])
    height = int(geometry["height"])
    resolution_m = float(geometry["resolution_m"])
    grid_area = max(width * height, 1)
    target_denominator = max(len(target_cells), 1)
    max_new = max((candidate["new_target_cell_count"] for candidate in candidates), default=0)
    max_new = max(max_new, 1)
    goals = []
    for candidate in candidates:
        new_count = int(candidate["new_target_cell_count"])
        rate_delta = new_count / target_denominator
        utility = new_count / max_new
        goals.append(
            GoalCandidate(
                cell=candidate["selected_waypoint"],
                utility=utility,
                reachable=True,
                experimental={
                    "expected_coverage_rate_delta": rate_delta,
                    "expected_new_coverage_area": float(new_count),
                    "information_gain": rate_delta,
                    "confidence_gain": rate_delta,
                    "value": utility,
                    "path_cost": float(candidate["path_cost_m"]),
                    "energy_cost": float(candidate["path_cost_m"]),
                    "risk": 0.0,
                },
            )
        )
    return ModelExplorerContract(
        schema_version=MODEL_EXPLORER_SCHEMA_VERSION,
        grid=GridSummary(
            width=width,
            height=height,
            resolution=resolution_m,
            frame_id="global_99_synthetic",
            origin=(0.0, 0.0),
            layers=("coverage_memory", "reachable_safe_exploration_cells"),
        ),
        constraints=ConstraintSummary(
            violation_count=grid_area - len(geometry["navigation_cells"]),
            passable_ratio=len(geometry["navigation_cells"]) / grid_area,
            reason_counts={},
        ),
        top_goals=tuple(goals),
        top_sequences=(),
        observation_update={
            "coverage_rate": coverage_rate(covered_target_cells, target_cells),
            "coverage_memory_rate": len(coverage_memory_cells) / grid_area,
            "current_cell": list(current_cell),
        },
    )


def _append_decision_and_ledger_row(
    *,
    decision_rows: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    scenario_id: str,
    cycle_index: int,
    event_id: str,
    selected_waypoint: Cell,
    baseline_waypoint: Cell,
    policy_waypoint: Cell | None,
    path: list[Cell],
    event_cells: set[Cell],
    counted: bool,
    attempted_path_cost_m: float,
    planned_path_cost_m: float,
    path_budget_m: float,
    path_cost_m: float,
    baseline_score: float,
    policy_logit: float | None,
    combined_score: float | None,
    revisited_path_cell_count: int,
    new_target_cell_count: int,
    target_cells: set[Cell],
    covered_target_cells: set[Cell],
    frontier_cluster_count: int,
    choice_source: str,
    guard_reason_codes: list[str],
) -> None:
    achieved = coverage_rate(covered_target_cells, target_cells)
    decision_rows.append(
        {
            "schema_version": DECISION_ROW_SCHEMA_VERSION,
            "scenario_id": scenario_id,
            "cycle_index": cycle_index,
            "event_id": event_id,
            "choice_source": choice_source,
            "selected_waypoint": list(selected_waypoint),
            "baseline_waypoint": list(baseline_waypoint),
            "policy_waypoint": None if policy_waypoint is None else list(policy_waypoint),
            "guard_reason_codes": guard_reason_codes,
            "path_cell_count": len(path),
            "path_cost_m": path_cost_m,
            "attempted_path_cost_m": attempted_path_cost_m,
            "planned_path_cost_m": planned_path_cost_m,
            "path_budget_m": path_budget_m,
            "counted": counted,
            "frontier_cluster_count": frontier_cluster_count,
            "revisited_path_cell_count": revisited_path_cell_count,
            "new_covered_reachable_safe_cell_count": new_target_cell_count,
            "baseline_score": baseline_score,
            "policy_logit": policy_logit,
            "combined_score": combined_score,
            "covered_reachable_safe_cell_count": len(covered_target_cells),
            "reachable_safe_cell_count": len(target_cells),
            "achieved_coverage_rate": achieved,
        }
    )
    ledger_rows.append(
        {
            "schema_version": LEDGER_ROW_SCHEMA_VERSION,
            "scenario_id": scenario_id,
            "cycle_index": cycle_index,
            "event_id": event_id,
            "selected_waypoint": list(selected_waypoint),
            "counted": counted,
            "event_cell_count": len(event_cells),
            "event_reachable_safe_cell_count": len(event_cells & target_cells),
            "new_covered_reachable_safe_cell_count": new_target_cell_count,
            "cumulative_covered_reachable_safe_cell_count": len(covered_target_cells),
            "reachable_safe_cell_count": len(target_cells),
            "achieved_coverage_rate": achieved,
            "path_cost_m": path_cost_m,
            "attempted_path_cost_m": attempted_path_cost_m,
            "planned_path_cost_m": planned_path_cost_m,
            "path_budget_m": path_budget_m,
        }
    )


def _append_snapshot_row(
    *,
    snapshot_rows: list[dict[str, Any]],
    scenario_id: str,
    cycle_index: int,
    current_cell: Cell,
    coverage_memory_cells: set[Cell],
    target_cells: set[Cell],
    planned_path_cost_m: float,
    path_budget_m: float,
) -> None:
    covered_target_cells = coverage_memory_cells & target_cells
    snapshot_rows.append(
        {
            "schema_version": SNAPSHOT_ROW_SCHEMA_VERSION,
            "scenario_id": scenario_id,
            "cycle_index": cycle_index,
            "current_cell": list(current_cell),
            "planned_path_cost_m": planned_path_cost_m,
            "path_budget_m": path_budget_m,
            "coverage_memory_cell_count": len(coverage_memory_cells),
            "covered_reachable_safe_cell_count": len(covered_target_cells),
            "reachable_safe_cell_count": len(target_cells),
            "achieved_coverage_rate": coverage_rate(covered_target_cells, target_cells),
        }
    )


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    for key in (
        "source_global_99_config",
        "source_frontier_baseline_config",
        "source_coverage_memory_config",
        "policy_candidate_root",
        "policy_checkpoint",
        "policy_checkpoint_metadata",
        "policy_candidate_summary",
    ):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
    normalized = dict(payload)
    normalized["target_coverage_rate"] = _bounded_float(payload.get("target_coverage_rate"), "target_coverage_rate")
    normalized["path_budget_m"] = _nonnegative_float(payload.get("path_budget_m"), "path_budget_m")
    normalized["coverage_radius_cells"] = _nonnegative_int(payload.get("coverage_radius_cells"), "coverage_radius_cells")
    normalized["replanning_cycle_limit"] = _nonnegative_int(payload.get("replanning_cycle_limit"), "replanning_cycle_limit")
    normalized["segment_step_limit"] = _positive_int(payload.get("segment_step_limit"), "segment_step_limit")
    normalized["memory_snapshot_interval"] = _positive_int(
        payload.get("memory_snapshot_interval"),
        "memory_snapshot_interval",
    )
    normalized["max_policy_candidates"] = _positive_int(payload.get("max_policy_candidates"), "max_policy_candidates")
    normalized["policy_logit_weight"] = _nonnegative_float(payload.get("policy_logit_weight"), "policy_logit_weight")
    normalized["allow_policy_candidate_git_provenance_mismatch"] = bool(
        payload.get("allow_policy_candidate_git_provenance_mismatch", False)
    )
    return normalized


def _load_frontier_config(path: Path) -> dict[str, Any]:
    payload = _read_json(path, "frontier config")
    if payload.get("schema_version") != FRONTIER_CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"frontier schema_version must be {FRONTIER_CONFIG_SCHEMA_VERSION!r}")
    normalized = dict(payload)
    normalized["revisit_penalty_weight"] = _nonnegative_float(
        payload.get("revisit_penalty_weight"),
        "revisit_penalty_weight",
    )
    normalized["new_coverage_weight"] = _nonnegative_float(payload.get("new_coverage_weight"), "new_coverage_weight")
    return normalized


def _load_memory_config(path: Path) -> dict[str, Any]:
    payload = _read_json(path, "coverage memory config")
    if payload.get("schema_version") != MEMORY_CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"coverage memory schema_version must be {MEMORY_CONFIG_SCHEMA_VERSION!r}")
    return payload


def _load_policy_bundle(
    config: dict[str, Any],
    config_path: Path,
    repo_root: Path,
) -> tuple[CheckpointPolicyBundle | None, dict[str, Any]]:
    candidate_root = _resolve_config_reference(config["policy_candidate_root"], config_path, repo_root)
    summary_path = candidate_root / config["policy_candidate_summary"]
    metadata_path = candidate_root / config["policy_checkpoint_metadata"]
    checkpoint_path = candidate_root / config["policy_checkpoint"]
    load_summary: dict[str, Any] = {
        "policy_candidate_root": str(candidate_root),
        "policy_candidate_summary": str(summary_path),
        "policy_checkpoint_metadata": str(metadata_path),
        "policy_checkpoint": str(checkpoint_path),
        "policy_loaded": False,
        "policy_checkpoint_state_key": None,
        "reason_codes": [],
    }
    try:
        candidate_summary = _read_json(summary_path, "policy candidate summary")
        metadata = _read_json(metadata_path, "policy checkpoint metadata")
    except ConfigError as exc:
        load_summary["reason_codes"].append("policy_guidance_unavailable")
        load_summary["load_error"] = str(exc)
        return None, load_summary
    if candidate_summary.get("status") != "passed":
        load_summary["reason_codes"].append("policy_guidance_unavailable")
        load_summary["candidate_summary_status"] = candidate_summary.get("status")
        return None, load_summary
    for payload in (candidate_summary, metadata):
        if payload.get("publishes_checkpoint") or payload.get("replaces_default_policy") or payload.get("performance_claimed"):
            load_summary["reason_codes"].append("policy_boundary_violation")
            return None, load_summary
    if not checkpoint_path.is_file():
        load_summary["reason_codes"].append("policy_guidance_unavailable")
        load_summary["load_error"] = f"checkpoint does not exist: {checkpoint_path}"
        return None, load_summary
    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - exercised through summary failure path
        load_summary["reason_codes"].append("policy_guidance_unavailable")
        load_summary["load_error"] = str(exc)
        return None, load_summary
    if not isinstance(checkpoint, dict):
        load_summary["reason_codes"].append("policy_guidance_unavailable")
        load_summary["load_error"] = "checkpoint root must be a mapping"
        return None, load_summary
    state_key = "state_dict" if "state_dict" in checkpoint else "model_state_dict" if "model_state_dict" in checkpoint else None
    if state_key is None:
        load_summary["reason_codes"].append("policy_guidance_unavailable")
        load_summary["load_error"] = "checkpoint missing state_dict/model_state_dict"
        return None, load_summary
    load_summary["policy_loaded"] = True
    load_summary["policy_checkpoint_state_key"] = state_key
    load_summary["architecture"] = metadata.get("architecture") or checkpoint.get("architecture")
    load_summary["hidden_size"] = (
        checkpoint.get("hidden_size")
        or (checkpoint.get("training") if isinstance(checkpoint.get("training"), dict) else {}).get("hidden_size")
        or metadata.get("hidden_size")
    )
    return CheckpointPolicyBundle(checkpoint=checkpoint, metadata=metadata, state_key=state_key), load_summary


def _policy_candidate_subset(
    baseline_candidate: dict[str, Any],
    candidates: list[dict[str, Any]],
    max_policy_candidates: int,
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[Cell] = set()
    for candidate in [baseline_candidate, *candidates]:
        waypoint = candidate["selected_waypoint"]
        if waypoint in seen:
            continue
        seen.add(waypoint)
        deduped.append(candidate)
        if len(deduped) >= max_policy_candidates:
            break
    return deduped


def _score_row_for_candidate(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> dict[str, Any] | None:
    waypoint = list(candidate["selected_waypoint"])
    for row in rows:
        if row["selected_waypoint"] == waypoint:
            return row
    return None


def _same_waypoint(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left["selected_waypoint"] == right["selected_waypoint"]


def _summary_reason_codes(
    *,
    scenario_results: list[dict[str, Any]],
    coverage_target_met: bool,
    total_denominator: int,
    policy_load: dict[str, Any],
    policy_guidance_applied: bool,
    policy_scored_candidate_count: int,
) -> list[str]:
    reasons: list[str] = []
    for result in scenario_results:
        reasons.extend(result["reason_codes"])
    reasons.extend(policy_load.get("reason_codes", []))
    if total_denominator <= 0:
        reasons.append("coverage_denominator_invalid")
    if total_denominator > 0 and not coverage_target_met:
        reasons.append("coverage_target_not_met")
    if not policy_load.get("policy_loaded") or not policy_guidance_applied or policy_scored_candidate_count <= 0:
        reasons.append("policy_guidance_unavailable")
    return unique_sorted(reasons)


def _public_scenario_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "scenario_id",
            "status",
            "reason_codes",
            "infeasible_reason_codes",
            "target_coverage_rate",
            "achieved_coverage_rate",
            "coverage_target_met",
            "reachable_safe_cell_count",
            "covered_reachable_safe_cell_count",
            "policy_scored_candidate_count",
            "policy_guided_decision_count",
            "policy_selected_decision_count",
            "policy_guard_fallback_count",
            "controlled_regression_count",
            "planned_path_cost_m",
            "path_budget_m",
            "path_budget_exhausted",
            "coverage_memory_complete",
            "replanning_cycle_count",
            "memory_snapshot_count",
            "frontier_exhausted",
            "frontier_unreachable",
            "roi_cell_count",
            "blocked_roi_cell_count",
            "unsafe_roi_cell_count",
            "unreachable_roi_cell_count",
        )
    }


def _policy_score_audit(
    config: dict[str, Any],
    policy_load: dict[str, Any],
    scenario_results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": POLICY_SCORE_AUDIT_SCHEMA_VERSION,
        "policy_logit_weight": config["policy_logit_weight"],
        "max_policy_candidates": config["max_policy_candidates"],
        "policy_load": policy_load,
        "policy_scored_candidate_count": sum(result["policy_scored_candidate_count"] for result in scenario_results),
        "policy_guided_decision_count": sum(result["policy_guided_decision_count"] for result in scenario_results),
        "policy_selected_decision_count": sum(result["policy_selected_decision_count"] for result in scenario_results),
        "baseline_agreement_count": sum(result["baseline_agreement_count"] for result in scenario_results),
        "baseline_agreement_denominator": sum(result["baseline_agreement_denominator"] for result in scenario_results),
    }


def _guard_audit(scenario_results: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counter = Counter(
        reason
        for result in scenario_results
        for row in result["decision_rows"]
        for reason in row.get("guard_reason_codes", [])
    )
    return {
        "schema_version": GUARD_AUDIT_SCHEMA_VERSION,
        "policy_guard_fallback_count": sum(result["policy_guard_fallback_count"] for result in scenario_results),
        "controlled_regression_count": sum(result["controlled_regression_count"] for result in scenario_results),
        "guard_reason_code_counts": dict(reason_counter),
    }


def _budget_audit(
    config: dict[str, Any],
    frontier_config: dict[str, Any],
    scenario_results: list[dict[str, Any]],
    planned_path_cost_m: float,
) -> dict[str, Any]:
    return {
        "schema_version": BUDGET_AUDIT_SCHEMA_VERSION,
        "target_coverage_rate": config["target_coverage_rate"],
        "path_budget_m": config["path_budget_m"],
        "planned_path_cost_m": planned_path_cost_m,
        "path_budget_exhausted": any(result["path_budget_exhausted"] for result in scenario_results),
        "coverage_radius_cells": config["coverage_radius_cells"],
        "replanning_cycle_limit": config["replanning_cycle_limit"],
        "segment_step_limit": config["segment_step_limit"],
        "revisit_penalty_weight": frontier_config["revisit_penalty_weight"],
        "new_coverage_weight": frontier_config["new_coverage_weight"],
        "scenario_budget": [
            {
                "scenario_id": result["scenario_id"],
                "planned_path_cost_m": result["planned_path_cost_m"],
                "path_budget_m": result["path_budget_m"],
                "path_budget_exhausted": result["path_budget_exhausted"],
                "replanning_cycle_count": result["replanning_cycle_count"],
            }
            for result in scenario_results
        ],
    }


def _rejection_report(reason_codes: list[str], scenario_results: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_reason_counts = Counter(reason for result in scenario_results for reason in result["reason_codes"])
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(Counter(reason_codes)),
        "scenario_failure_reason_code_counts": dict(scenario_reason_counts),
        "scenario_rejections": [
            {
                "scenario_id": result["scenario_id"],
                "status": result["status"],
                "reason_codes": result["reason_codes"],
                "infeasible_reason_codes": result["infeasible_reason_codes"],
            }
            for result in scenario_results
        ],
    }


def _manifest(
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": summary["generated_at"],
        "stage": "Policy-Guided Global Coverage v1",
        "config": str(config_path),
        "output_root": str(output_root),
        "artifacts": {key: str(value) for key, value in paths.items()},
        "summary_status": summary["status"],
        "next_required_change": summary["next_required_change"],
        "boundary": {
            "uses_policy_guidance": True,
            "uses_checkpoint_inference": True,
            "policy_read_only": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
            "uses_path_planner": False,
            "uses_npz_or_sidecar": False,
        },
    }


def _render_report(summary: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    lines = [
        "# Policy-Guided Global Coverage v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- achieved_coverage_rate: `{summary['achieved_coverage_rate']}`",
        f"- coverage_target_met: `{summary['coverage_target_met']}`",
        f"- policy_loaded: `{summary['policy_loaded']}`",
        f"- policy_guidance_applied: `{summary['policy_guidance_applied']}`",
        f"- policy_scored_candidate_count: `{summary['policy_scored_candidate_count']}`",
        f"- policy_guard_fallback_count: `{summary['policy_guard_fallback_count']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        "",
        "## Scenario Summaries",
        "",
    ]
    for scenario in summary["scenario_summaries"]:
        lines.extend(
            [
                f"### {scenario['scenario_id']}",
                "",
                f"- status: `{scenario['status']}`",
                f"- achieved_coverage_rate: `{scenario['achieved_coverage_rate']}`",
                f"- policy_guided_decision_count: `{scenario['policy_guided_decision_count']}`",
                f"- policy_guard_fallback_count: `{scenario['policy_guard_fallback_count']}`",
                f"- infeasible_reason_codes: `{scenario['infeasible_reason_codes']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Rejection Report",
            "",
            f"- failure_reason_code_counts: `{rejection_report['failure_reason_code_counts']}`",
            "",
            "This stage uses read-only experimental checkpoint inference to rank synthetic frontier candidates. It does not train PPO, publish a checkpoint, replace default policy, connect a real executor, call path-planner, use NPZ/sidecar maps, or modify network/action space/default A*.",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve_config_reference(value: str, config_path: Path, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    config_relative = config_path.parent / path
    if config_relative.exists():
        return config_relative
    return resolve_path(path, repo_root)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"{label} file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{label} JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"{label} root must be an object")
    return payload


def _min_max_normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        return ()
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return tuple(0.0 for _ in values)
    return tuple((value - minimum) / (maximum - minimum) for value in values)


def _bounded_float(value: Any, label: str) -> float:
    numeric = _nonnegative_float(value, label)
    if numeric <= 0.0 or numeric > 1.0:
        raise ConfigError(f"{label} must be > 0 and <= 1")
    return numeric


def _nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{label} must be numeric")
    numeric = float(value)
    if numeric < 0.0:
        raise ConfigError(f"{label} must be >= 0")
    return numeric


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    if value < 0:
        raise ConfigError(f"{label} must be >= 0")
    return value


def _positive_int(value: Any, label: str) -> int:
    numeric = _nonnegative_int(value, label)
    if numeric <= 0:
        raise ConfigError(f"{label} must be > 0")
    return numeric


if __name__ == "__main__":
    raise SystemExit(main())
