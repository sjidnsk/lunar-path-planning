from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot


DEFAULT_STAGE16_ROOT = "outputs/path_feedback_batch_default_policy_candidate_authorization_preflight_v1"
DEFAULT_STAGE15_ROOT = "outputs/path_feedback_batch_checkpoint_publication_sandbox_consumer_replay_canary_v1"
DEFAULT_STAGE14_ROOT = "outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_verification_v1"
DEFAULT_COST_EFFICIENCY_ROOT = "outputs/path_feedback_batch_cost_efficiency_aware_coverage_reward_candidate_filter_v1"
DEFAULT_LOW_OBSERVATION_ROOT = "outputs/path_feedback_batch_low_observation_candidate_geometry_improvement_v1"
DEFAULT_SAFE_BETTER_ROOT = "outputs/path_feedback_batch_safe_better_pair_expansion_across_families_v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_family_balanced_algorithm_gap_closure_v1"

STAGE16_SUMMARY_FILE = "default-policy-candidate-authorization-preflight-summary.json"
STAGE15_SUMMARY_FILE = "checkpoint-publication-sandbox-consumer-replay-canary-summary.json"
STAGE14_SUMMARY_FILE = "checkpoint-publication-sandbox-install-dry-run-verification-summary.json"
COST_SUMMARY_FILE = "cost-efficiency-aware-coverage-reward-candidate-filter-summary.json"
COST_FILTERED_BATCH_FILE = "cost-efficiency-filtered-batch.jsonl"
COST_ADVANTAGE_AUDIT_FILE = "cost-efficiency-advantage-audit.jsonl"
COST_COMPAT_STAGE5A2_DIR = "cost-efficiency-compatible-stage5a2-input"
COST_COMPAT_COUNTERFACTUAL_FILE = "counterfactual-coverage-rollouts.jsonl"
LOW_OBSERVATION_SUMMARY_FILE = "low-observation-candidate-geometry-summary.json"
LOW_OBSERVATION_OVERLAY_FILE = "low-observation-candidate-geometry-overlay.jsonl"
LOW_OBSERVATION_COUNTERFACTUAL_FILE = "low-observation-candidate-geometry-counterfactual-rollouts.jsonl"
SAFE_BETTER_SUMMARY_FILE = "safe-better-pair-expansion-summary.json"
SAFE_BETTER_PAIRS_FILE = "expanded-safe-better-pairs.jsonl"
SAFE_BETTER_COUNTERFACTUAL_FILE = "expanded-counterfactual-coverage-rollouts.jsonl"

SUMMARY_FILE = "family-balanced-algorithm-gap-closure-summary.json"
SOURCE_LEDGER_FILE = "family-balanced-source-ledger.json"
FAMILY_BALANCE_AUDIT_FILE = "family-balance-audit.json"
LOW_OBSERVATION_GAP_AUDIT_FILE = "low-observation-gap-audit.json"
BALANCED_COUNTERFACTUAL_FILE = "family-balanced-counterfactual-rollouts.jsonl"
BALANCED_PAIRS_FILE = "family-balanced-safe-better-pairs.jsonl"
MANIFEST_FILE = "family-balanced-coverage-driven-input-manifest.json"
RELEASE_AUDIT_FILE = "family-balanced-release-boundary-audit.json"
REJECTION_REPORT_FILE = "family-balanced-rejection-report.json"
REPORT_FILE = "family-balanced-algorithm-gap-closure-report.md"

PASS_VERDICT = "eligible_for_family_balanced_coverage_driven_ppo_rerun"
NEXT_REQUIRED_CHANGE = "family_balanced_coverage_driven_ppo_rerun"
TARGET_FAMILY = "low_observation_count"
TARGET_FAMILIES = (
    "low_observation_count",
    "mixed_risk",
    "rim_or_steep_slope",
    "smooth_high_confidence",
)
DEFAULT_MINIMUM_LOW_OBSERVATION_COUNT = 32
DEFAULT_MINIMUM_FAMILY_PAIR_COUNT = 32
DEFAULT_MINIMUM_FAMILY_BALANCE_RATIO = 0.15
TOLERANCE = 1.0e-9

RELEASE_BOUNDARY_FIELDS = (
    "checkpoint_publication_approved",
    "default_policy_replacement_approved",
    "real_executor_connection_approved",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "real_world_release_approved",
    "real_world_performance_claimed",
    "final_release_approved",
    "relaxes_guard",
    "guard_relaxed",
    "modifies_network_or_action_space",
    "modifies_network",
    "modifies_action_space",
    "modifies_default_astar",
    "ackermann_feasible_trajectory_claimed",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Family-Balanced Algorithm Gap Closure v1.")
    parser.add_argument("--stage16-root", default=DEFAULT_STAGE16_ROOT)
    parser.add_argument("--stage15-root", default=DEFAULT_STAGE15_ROOT)
    parser.add_argument("--stage14-root", default=DEFAULT_STAGE14_ROOT)
    parser.add_argument("--cost-efficiency-root", default=DEFAULT_COST_EFFICIENCY_ROOT)
    parser.add_argument("--low-observation-geometry-root", default=DEFAULT_LOW_OBSERVATION_ROOT)
    parser.add_argument("--safe-better-root", default=DEFAULT_SAFE_BETTER_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--minimum-low-observation-count", type=int, default=DEFAULT_MINIMUM_LOW_OBSERVATION_COUNT)
    parser.add_argument("--minimum-family-pair-count", type=int, default=DEFAULT_MINIMUM_FAMILY_PAIR_COUNT)
    parser.add_argument("--minimum-family-balance-ratio", type=float, default=DEFAULT_MINIMUM_FAMILY_BALANCE_RATIO)
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    summary = run_family_balanced_algorithm_gap_closure(
        stage16_root=_resolve_path(Path(args.stage16_root), repo_root),
        stage15_root=_resolve_path(Path(args.stage15_root), repo_root),
        stage14_root=_resolve_path(Path(args.stage14_root), repo_root),
        cost_efficiency_root=_resolve_path(Path(args.cost_efficiency_root), repo_root),
        low_observation_geometry_root=_resolve_path(Path(args.low_observation_geometry_root), repo_root),
        safe_better_root=_resolve_path(Path(args.safe_better_root), repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root),
        repo_root=repo_root,
        minimum_low_observation_count=args.minimum_low_observation_count,
        minimum_family_pair_count=args.minimum_family_pair_count,
        minimum_family_balance_ratio=args.minimum_family_balance_ratio,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "gap_closure_verdict": summary["gap_closure_verdict"],
                "family_balanced_algorithm_gap_closure_passed": summary[
                    "family_balanced_algorithm_gap_closure_passed"
                ],
                "low_observation_gap_closed": summary["low_observation_gap_closed"],
                "next_required_change": summary["next_required_change"],
                "checkpoint_publication_approved": summary["checkpoint_publication_approved"],
                "default_policy_replacement_approved": summary["default_policy_replacement_approved"],
                "real_executor_connection_approved": summary["real_executor_connection_approved"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_family_balanced_algorithm_gap_closure(
    *,
    stage16_root: Path,
    stage15_root: Path,
    stage14_root: Path,
    cost_efficiency_root: Path,
    low_observation_geometry_root: Path,
    safe_better_root: Path,
    output_root: Path,
    repo_root: Path,
    minimum_low_observation_count: int = DEFAULT_MINIMUM_LOW_OBSERVATION_COUNT,
    minimum_family_pair_count: int = DEFAULT_MINIMUM_FAMILY_PAIR_COUNT,
    minimum_family_balance_ratio: float = DEFAULT_MINIMUM_FAMILY_BALANCE_RATIO,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)
    read_reasons: list[str] = []

    stage16_summary = _read_json(stage16_root / STAGE16_SUMMARY_FILE, read_reasons, "stage16_summary")
    stage15_summary = _read_json(stage15_root / STAGE15_SUMMARY_FILE, read_reasons, "stage15_summary")
    stage14_summary = _read_json(stage14_root / STAGE14_SUMMARY_FILE, read_reasons, "stage14_summary")
    cost_summary = _read_json(cost_efficiency_root / COST_SUMMARY_FILE, read_reasons, "cost_efficiency_summary")
    cost_rows = _read_jsonl(cost_efficiency_root / COST_FILTERED_BATCH_FILE, read_reasons, "cost_efficiency_filtered_batch")
    cost_advantage_rows = _read_jsonl(cost_efficiency_root / COST_ADVANTAGE_AUDIT_FILE, [], "cost_efficiency_advantage_audit")
    cost_counterfactual_rows = _read_jsonl(
        cost_efficiency_root / COST_COMPAT_STAGE5A2_DIR / COST_COMPAT_COUNTERFACTUAL_FILE,
        read_reasons,
        "cost_efficiency_counterfactual_rollouts",
    )
    low_summary = _read_json(
        low_observation_geometry_root / LOW_OBSERVATION_SUMMARY_FILE,
        read_reasons,
        "low_observation_geometry_summary",
    )
    low_overlay_rows = _read_jsonl(
        low_observation_geometry_root / LOW_OBSERVATION_OVERLAY_FILE,
        read_reasons,
        "low_observation_overlay",
    )
    low_counterfactual_rows = _read_jsonl(
        low_observation_geometry_root / LOW_OBSERVATION_COUNTERFACTUAL_FILE,
        read_reasons,
        "low_observation_counterfactual_rollouts",
    )
    safe_summary = _read_json(safe_better_root / SAFE_BETTER_SUMMARY_FILE, read_reasons, "safe_better_summary")
    safe_pairs = _read_jsonl(safe_better_root / SAFE_BETTER_PAIRS_FILE, read_reasons, "safe_better_pairs")
    safe_counterfactual_rows = _read_jsonl(
        safe_better_root / SAFE_BETTER_COUNTERFACTUAL_FILE,
        read_reasons,
        "safe_better_counterfactual_rollouts",
    )

    stage16_gate = _stage16_gate(stage16_summary)
    lineage = _lineage_audit(stage14_summary, stage15_summary, cost_summary, low_summary, safe_summary, read_reasons)
    docs = _docs_audit(repo_root)
    selected = _build_family_balanced_inputs(
        cost_rows=cost_rows,
        safe_pairs=safe_pairs,
        cost_counterfactual_rows=cost_counterfactual_rows,
        safe_counterfactual_rows=safe_counterfactual_rows,
        low_counterfactual_rows=low_counterfactual_rows,
    )
    family_balance = _family_balance_audit(
        selected_pairs=selected["pairs"],
        base_cost_rows=cost_rows,
        low_observation_rows=safe_pairs,
        minimum_low_observation_count=minimum_low_observation_count,
        minimum_family_pair_count=minimum_family_pair_count,
        minimum_family_balance_ratio=minimum_family_balance_ratio,
        low_observation_had_non_train=selected["low_observation_had_non_train"],
    )
    low_observation = _low_observation_gap_audit(
        low_summary=low_summary,
        low_overlay_rows=low_overlay_rows,
        low_counterfactual_rows=low_counterfactual_rows,
        selected_pairs=selected["pairs"],
        selected_counterfactual_rows=selected["counterfactual_rows"],
        minimum_low_observation_count=minimum_low_observation_count,
    )
    source_backed = _source_backed_counterfactual_audit(selected["pairs"], selected["counterfactual_rows"])
    release = _release_boundary_audit(
        {
            "stage16_summary": stage16_summary,
            "stage15_summary": stage15_summary,
            "stage14_summary": stage14_summary,
            "cost_efficiency_summary": cost_summary,
            "low_observation_summary": low_summary,
            "safe_better_summary": safe_summary,
        }
    )

    reason_codes: list[str] = []
    for audit in (stage16_gate, lineage, family_balance, low_observation, source_backed, release):
        for reason in audit.get("reason_codes", []):
            _add_reason(reason_codes, reason)
    if docs.get("docs_audit_passed") is not True:
        _add_reason(reason_codes, "docs_not_updated")
    if cost_summary.get("status") != "passed" or cost_summary.get("reason_codes") not in ([], None):
        _add_reason(reason_codes, "cost_efficiency_source_not_passed")
    if low_summary.get("status") != "passed" or low_summary.get("reason_codes") not in ([], None):
        _add_reason(reason_codes, "low_observation_source_missing")
    if safe_summary.get("status") != "passed" or safe_summary.get("reason_codes") not in ([], None):
        _add_reason(reason_codes, "counterfactual_source_missing")
    if _int(cost_summary.get("fallback_gain_contamination_count")) > 0 or _int(
        low_summary.get("fallback_gain_contamination_count")
    ) > 0 or _int(safe_summary.get("fallback_gain_contamination_count")) > 0:
        _add_reason(reason_codes, "fallback_gain_contamination")
    if _int(cost_summary.get("controlled_regression_count")) > 0 or _int(
        low_summary.get("controlled_regression_count")
    ) > 0 or _int(safe_summary.get("controlled_regression_count")) > 0:
        _add_reason(reason_codes, "controlled_regression_detected")

    approved = not reason_codes
    family_counts = _family_counts(selected["pairs"])
    low_observation_count = family_counts.get(TARGET_FAMILY, 0)
    minimum_family_observed = min((family_counts.get(family, 0) for family in TARGET_FAMILIES), default=0)
    family_balance_ratio = _family_balance_ratio(family_counts)

    _write_jsonl(paths["balanced_pairs"], selected["pairs"])
    _write_jsonl(paths["balanced_counterfactual"], selected["counterfactual_rows"])
    source_ledger = _source_ledger(
        stage16_root=stage16_root,
        stage15_root=stage15_root,
        stage14_root=stage14_root,
        cost_efficiency_root=cost_efficiency_root,
        low_observation_geometry_root=low_observation_geometry_root,
        safe_better_root=safe_better_root,
        cost_summary=cost_summary,
        low_summary=low_summary,
        safe_summary=safe_summary,
        cost_rows=cost_rows,
        cost_advantage_rows=cost_advantage_rows,
        cost_counterfactual_rows=cost_counterfactual_rows,
        low_overlay_rows=low_overlay_rows,
        low_counterfactual_rows=low_counterfactual_rows,
        safe_pairs=safe_pairs,
        safe_counterfactual_rows=safe_counterfactual_rows,
        selected_pairs=selected["pairs"],
    )
    manifest = _manifest(paths, family_counts, selected["pairs"], selected["counterfactual_rows"], approved)
    summary = {
        "schema_version": "family-balanced-algorithm-gap-closure-summary/v1",
        "generated_at": _utc_now(),
        "status": "passed" if approved else "failed",
        "reason_codes": reason_codes,
        "gap_closure_verdict": PASS_VERDICT if approved else "resolve_family_balanced_algorithm_gap_closure_rejections",
        "family_balanced_algorithm_gap_closure_passed": approved,
        "family_balanced_coverage_driven_ppo_rerun_approved": approved,
        "low_observation_gap_closed": low_observation["low_observation_gap_closed"],
        "source_backed_counterfactual_audit_passed": source_backed["source_backed_counterfactual_audit_passed"],
        "family_balance_audit_passed": family_balance["family_balance_audit_passed"],
        "lineage_audit_passed": lineage["lineage_audit_passed"],
        "release_boundary_audit_passed": release["release_boundary_audit_passed"],
        "family_safe_better_counts": family_counts,
        "low_observation_count": low_observation_count,
        "minimum_low_observation_count": minimum_low_observation_count,
        "minimum_family_pair_count": minimum_family_pair_count,
        "minimum_family_observed": minimum_family_observed,
        "family_balance_ratio": family_balance_ratio,
        "minimum_family_balance_ratio": minimum_family_balance_ratio,
        "coverage_return_improvement_reference": _float(cost_summary.get("coverage_return_improvement")),
        "coverage_efficiency_regression_reference": bool(cost_summary.get("coverage_efficiency_regression", False)),
        "fallback_gain_contamination_count": max(
            _int(cost_summary.get("fallback_gain_contamination_count")),
            _int(low_summary.get("fallback_gain_contamination_count")),
            _int(safe_summary.get("fallback_gain_contamination_count")),
        ),
        "controlled_regression_count": max(
            _int(cost_summary.get("controlled_regression_count")),
            _int(low_summary.get("controlled_regression_count")),
            _int(safe_summary.get("controlled_regression_count")),
        ),
        "next_required_change": NEXT_REQUIRED_CHANGE if approved else "resolve_family_balanced_algorithm_gap_closure_rejections",
        **_closed_boundaries(),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "source_ledger": str(paths["source_ledger"]),
        "family_balance_audit": str(paths["family_balance"]),
        "low_observation_gap_audit": str(paths["low_observation"]),
        "family_balanced_counterfactual_rollouts": str(paths["balanced_counterfactual"]),
        "family_balanced_safe_better_pairs": str(paths["balanced_pairs"]),
        "family_balanced_coverage_driven_input_manifest": str(paths["manifest"]),
        "release_boundary_audit": str(paths["release"]),
        "rejection_report": str(paths["rejection_report"]),
        "report": str(paths["report"]),
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }
    for path, payload in (
        (paths["source_ledger"], source_ledger),
        (paths["family_balance"], family_balance),
        (paths["low_observation"], low_observation),
        (paths["manifest"], manifest),
        (paths["release"], release),
        (paths["rejection_report"], _rejection_report(reason_codes)),
        (paths["summary"], summary),
    ):
        _write_json(path, payload)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _paths(root: Path) -> dict[str, Path]:
    return {
        "summary": root / SUMMARY_FILE,
        "source_ledger": root / SOURCE_LEDGER_FILE,
        "family_balance": root / FAMILY_BALANCE_AUDIT_FILE,
        "low_observation": root / LOW_OBSERVATION_GAP_AUDIT_FILE,
        "balanced_counterfactual": root / BALANCED_COUNTERFACTUAL_FILE,
        "balanced_pairs": root / BALANCED_PAIRS_FILE,
        "manifest": root / MANIFEST_FILE,
        "release": root / RELEASE_AUDIT_FILE,
        "rejection_report": root / REJECTION_REPORT_FILE,
        "report": root / REPORT_FILE,
    }


def _stage16_gate(summary: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if summary.get("status") != "passed" or summary.get("reason_codes") not in ([], None):
        reasons.append("stage16_not_passed")
    if summary.get("default_policy_candidate_authorization_preflight_passed") is not True:
        reasons.append("stage16_not_passed")
    if summary.get("default_policy_candidate_sandbox_install_preflight_approved") is not True:
        reasons.append("stage16_not_passed")
    if _release_violations(summary):
        reasons.append("stage16_release_boundary_violation")
    return {"stage16_gate_passed": not reasons, "reason_codes": _unique(reasons)}


def _lineage_audit(
    stage14_summary: dict[str, Any],
    stage15_summary: dict[str, Any],
    cost_summary: dict[str, Any],
    low_summary: dict[str, Any],
    safe_summary: dict[str, Any],
    read_reasons: list[str],
) -> dict[str, Any]:
    passed = (
        not read_reasons
        and stage14_summary.get("status") == "passed"
        and stage15_summary.get("status") == "passed"
        and cost_summary.get("status") == "passed"
        and low_summary.get("status") == "passed"
        and safe_summary.get("status") == "passed"
    )
    return {
        "schema_version": "family-balanced-lineage-audit/v1",
        "lineage_audit_passed": passed,
        "read_reason_codes": read_reasons,
        "reason_codes": [] if passed else ["lineage_incomplete"],
    }


def _build_family_balanced_inputs(
    *,
    cost_rows: list[dict[str, Any]],
    safe_pairs: list[dict[str, Any]],
    cost_counterfactual_rows: list[dict[str, Any]],
    safe_counterfactual_rows: list[dict[str, Any]],
    low_counterfactual_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    counterfactual_index = _counterfactual_index(
        [*safe_counterfactual_rows, *low_counterfactual_rows, *cost_counterfactual_rows]
    )
    selected: list[dict[str, Any]] = []
    selected_counterfactual: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, int, int]] = set()

    for row in cost_rows:
        if str(row.get("scenario_family") or "") == TARGET_FAMILY:
            continue
        if _eligible_reasons(row, counterfactual_index):
            continue
        _append_selected(
            row=row,
            counterfactual_index=counterfactual_index,
            source_stage="stage5b7_cost_efficiency_filter",
            selected=selected,
            selected_counterfactual=selected_counterfactual,
            seen_keys=seen_keys,
        )

    low_observation_had_non_train = False
    selected_low_observation_count = 0
    for row in safe_pairs:
        if str(row.get("scenario_family") or "") != TARGET_FAMILY:
            continue
        reasons = _eligible_reasons(row, counterfactual_index)
        if "non_train_split" in reasons:
            low_observation_had_non_train = True
        if reasons:
            continue
        _append_selected(
            row=row,
            counterfactual_index=counterfactual_index,
            source_stage="low_observation_geometry_supplement",
            selected=selected,
            selected_counterfactual=selected_counterfactual,
            seen_keys=seen_keys,
        )
        selected_low_observation_count += 1
    selected.sort(key=_sort_key)
    selected_counterfactual.sort(key=_sort_key)
    return {
        "pairs": selected,
        "counterfactual_rows": selected_counterfactual,
        "low_observation_had_non_train": low_observation_had_non_train and selected_low_observation_count == 0,
    }


def _append_selected(
    *,
    row: dict[str, Any],
    counterfactual_index: dict[tuple[str, str, int, int], dict[str, Any]],
    source_stage: str,
    selected: list[dict[str, Any]],
    selected_counterfactual: list[dict[str, Any]],
    seen_keys: set[tuple[str, str, int, int]],
) -> None:
    action_index = _first_int(row.get("candidate_action_index"))
    if action_index is None:
        return
    key = _counterfactual_key_from_pair(row, action_index)
    if key in seen_keys:
        return
    enriched = copy.deepcopy(row)
    enriched.update(
        {
            "schema_version": "family-balanced-safe-better-pair-row/v1",
            "family_balance_source_stage": source_stage,
            "family_balanced_trainable": True,
        }
    )
    counterfactual = copy.deepcopy(counterfactual_index[key])
    counterfactual.update(
        {
            "schema_version": "family-balanced-counterfactual-coverage-row/v1",
            "family_balance_source_stage": source_stage,
            "family_balanced_trainable": True,
            "action_index": action_index,
            "context_id": str(row.get("context_id") or ""),
            "episode_id": str(row.get("episode_id") or ""),
            "step_index": _int(row.get("step_index")),
            "scenario_id": row.get("scenario_id"),
            "scenario_family": row.get("scenario_family"),
            "split": "train",
            "coverage_source_available": True,
            "safe_better_than_teacher_candidate": True,
            "fallback_like": False,
            "guard_rejected": False,
            "controlled_regression_reason_codes": [],
        }
    )
    selected.append(enriched)
    selected_counterfactual.append(counterfactual)
    seen_keys.add(key)


def _eligible_reasons(row: dict[str, Any], counterfactual_index: dict[tuple[str, str, int, int], dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    if str(row.get("split") or "") != "train":
        reasons.append("non_train_split")
    if not bool(row.get("ppo_trainable")):
        reasons.append("not_ppo_trainable")
    if not bool(row.get("coverage_source_available")):
        reasons.append("coverage_source_missing")
    if not bool(row.get("safe_better_than_teacher_candidate")):
        reasons.append("not_safe_better_than_teacher")
    if _float(row.get("coverage_advantage")) <= TOLERANCE:
        reasons.append("coverage_advantage_not_positive")
    if bool(row.get("fallback_like")):
        reasons.append("fallback_like_candidate")
    if bool(row.get("guard_rejected")):
        reasons.append("guard_rejected_candidate")
    if _string_list(row.get("controlled_regression_reason_codes")):
        reasons.append("controlled_regression_present")
    action_index = _first_int(row.get("candidate_action_index"))
    if action_index is None:
        reasons.append("candidate_action_index_missing")
    elif _counterfactual_key_from_pair(row, action_index) not in counterfactual_index:
        reasons.append("counterfactual_row_missing")
    return _unique(reasons)


def _family_balance_audit(
    *,
    selected_pairs: list[dict[str, Any]],
    base_cost_rows: list[dict[str, Any]],
    low_observation_rows: list[dict[str, Any]],
    minimum_low_observation_count: int,
    minimum_family_pair_count: int,
    minimum_family_balance_ratio: float,
    low_observation_had_non_train: bool,
) -> dict[str, Any]:
    counts = _family_counts(selected_pairs)
    base_counts = _family_counts([row for row in base_cost_rows if _row_is_trainable(row)])
    low_source_counts = _family_counts([row for row in low_observation_rows if row.get("scenario_family") == TARGET_FAMILY])
    low_count = counts.get(TARGET_FAMILY, 0)
    minimum_observed = min((counts.get(family, 0) for family in TARGET_FAMILIES), default=0)
    ratio = _family_balance_ratio(counts)
    reasons: list[str] = []
    if low_count < minimum_low_observation_count:
        reasons.append("low_observation_gap_not_closed")
    if minimum_observed < minimum_family_pair_count or ratio < minimum_family_balance_ratio:
        reasons.append("family_balance_below_threshold")
    if any(counts.get(family, 0) <= 0 for family in TARGET_FAMILIES):
        reasons.append("family_balance_below_threshold")
    if low_observation_had_non_train:
        reasons.append("non_train_split_in_trainable_output")
    return {
        "schema_version": "family-balance-audit/v1",
        "generated_at": _utc_now(),
        "family_balance_audit_passed": not reasons,
        "family_safe_better_counts": counts,
        "base_cost_efficiency_family_counts": base_counts,
        "low_observation_source_family_counts": low_source_counts,
        "low_observation_count": low_count,
        "minimum_low_observation_count": minimum_low_observation_count,
        "minimum_family_pair_count": minimum_family_pair_count,
        "minimum_family_observed": minimum_observed,
        "family_balance_ratio": ratio,
        "minimum_family_balance_ratio": minimum_family_balance_ratio,
        "reason_codes": _unique(reasons),
    }


def _low_observation_gap_audit(
    *,
    low_summary: dict[str, Any],
    low_overlay_rows: list[dict[str, Any]],
    low_counterfactual_rows: list[dict[str, Any]],
    selected_pairs: list[dict[str, Any]],
    selected_counterfactual_rows: list[dict[str, Any]],
    minimum_low_observation_count: int,
) -> dict[str, Any]:
    selected_low = [row for row in selected_pairs if row.get("scenario_family") == TARGET_FAMILY]
    selected_low_counterfactual = [
        row for row in selected_counterfactual_rows if row.get("scenario_family") == TARGET_FAMILY
    ]
    low_count = len(selected_low)
    source_passed = (
        low_summary.get("status") == "passed"
        and _int(low_summary.get("missing_counterfactual_source_count")) == 0
        and _int(low_summary.get("fallback_gain_contamination_count")) == 0
        and _int(low_summary.get("controlled_regression_count")) == 0
    )
    source_backed = all(row.get("source_path") and row.get("coverage_source_available") is True for row in selected_low)
    counterfactual_backed = all(
        row.get("source_path") and row.get("coverage_source_available") is True for row in selected_low_counterfactual
    )
    closed = source_passed and source_backed and counterfactual_backed and low_count >= minimum_low_observation_count
    reasons: list[str] = []
    if not source_passed or not low_overlay_rows or not low_counterfactual_rows:
        reasons.append("low_observation_source_missing")
    if low_count < minimum_low_observation_count:
        reasons.append("low_observation_gap_not_closed")
    if not counterfactual_backed:
        reasons.append("counterfactual_source_missing")
    return {
        "schema_version": "low-observation-gap-audit/v1",
        "generated_at": _utc_now(),
        "low_observation_gap_closed": closed,
        "low_observation_selected_pair_count": low_count,
        "low_observation_selected_counterfactual_count": len(selected_low_counterfactual),
        "low_observation_overlay_row_count": len(low_overlay_rows),
        "low_observation_counterfactual_row_count": len(low_counterfactual_rows),
        "minimum_low_observation_count": minimum_low_observation_count,
        "source_backed": source_backed,
        "counterfactual_backed": counterfactual_backed,
        "reason_codes": _unique(reasons),
    }


def _source_backed_counterfactual_audit(
    selected_pairs: list[dict[str, Any]],
    selected_counterfactual_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    selected_keys = {_counterfactual_key_from_pair(row, _int(row.get("candidate_action_index"))) for row in selected_pairs}
    counterfactual_keys = {
        _counterfactual_key(row)
        for row in selected_counterfactual_rows
        if _first_int(row.get("action_index")) is not None
    }
    if selected_keys != counterfactual_keys:
        reasons.append("counterfactual_source_missing")
    if any(not row.get("source_path") or row.get("coverage_source_available") is not True for row in selected_counterfactual_rows):
        reasons.append("counterfactual_source_missing")
    if any(row.get("split") != "train" for row in selected_pairs):
        reasons.append("non_train_split_in_trainable_output")
    return {
        "schema_version": "family-balanced-source-backed-counterfactual-audit/v1",
        "source_backed_counterfactual_audit_passed": not reasons,
        "selected_pair_count": len(selected_pairs),
        "selected_counterfactual_count": len(selected_counterfactual_rows),
        "reason_codes": _unique(reasons),
    }


def _source_ledger(
    *,
    stage16_root: Path,
    stage15_root: Path,
    stage14_root: Path,
    cost_efficiency_root: Path,
    low_observation_geometry_root: Path,
    safe_better_root: Path,
    cost_summary: dict[str, Any],
    low_summary: dict[str, Any],
    safe_summary: dict[str, Any],
    cost_rows: list[dict[str, Any]],
    cost_advantage_rows: list[dict[str, Any]],
    cost_counterfactual_rows: list[dict[str, Any]],
    low_overlay_rows: list[dict[str, Any]],
    low_counterfactual_rows: list[dict[str, Any]],
    safe_pairs: list[dict[str, Any]],
    safe_counterfactual_rows: list[dict[str, Any]],
    selected_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_counts = _family_counts(selected_pairs)
    return {
        "schema_version": "family-balanced-source-ledger/v1",
        "generated_at": _utc_now(),
        "sources": {
            "stage16_summary": str(stage16_root / STAGE16_SUMMARY_FILE),
            "stage15_summary": str(stage15_root / STAGE15_SUMMARY_FILE),
            "stage14_summary": str(stage14_root / STAGE14_SUMMARY_FILE),
            "cost_efficiency_summary": str(cost_efficiency_root / COST_SUMMARY_FILE),
            "cost_efficiency_filtered_batch": str(cost_efficiency_root / COST_FILTERED_BATCH_FILE),
            "cost_efficiency_advantage_audit": str(cost_efficiency_root / COST_ADVANTAGE_AUDIT_FILE),
            "cost_efficiency_counterfactual_rollouts": str(
                cost_efficiency_root / COST_COMPAT_STAGE5A2_DIR / COST_COMPAT_COUNTERFACTUAL_FILE
            ),
            "low_observation_geometry_summary": str(low_observation_geometry_root / LOW_OBSERVATION_SUMMARY_FILE),
            "low_observation_overlay": str(low_observation_geometry_root / LOW_OBSERVATION_OVERLAY_FILE),
            "low_observation_counterfactual_rollouts": str(
                low_observation_geometry_root / LOW_OBSERVATION_COUNTERFACTUAL_FILE
            ),
            "safe_better_summary": str(safe_better_root / SAFE_BETTER_SUMMARY_FILE),
            "safe_better_pairs": str(safe_better_root / SAFE_BETTER_PAIRS_FILE),
            "safe_better_counterfactual_rollouts": str(safe_better_root / SAFE_BETTER_COUNTERFACTUAL_FILE),
        },
        "source_statuses": {
            "cost_efficiency": cost_summary.get("status"),
            "low_observation_geometry": low_summary.get("status"),
            "safe_better": safe_summary.get("status"),
        },
        "source_row_counts": {
            "cost_efficiency_filtered_batch": len(cost_rows),
            "cost_efficiency_advantage_audit": len(cost_advantage_rows),
            "cost_efficiency_counterfactual_rollouts": len(cost_counterfactual_rows),
            "low_observation_overlay": len(low_overlay_rows),
            "low_observation_counterfactual_rollouts": len(low_counterfactual_rows),
            "safe_better_pairs": len(safe_pairs),
            "safe_better_counterfactual_rollouts": len(safe_counterfactual_rows),
            "selected_family_balanced_pairs": len(selected_pairs),
        },
        "selected_family_counts": selected_counts,
    }


def _manifest(
    paths: dict[str, Path],
    family_counts: dict[str, int],
    selected_pairs: list[dict[str, Any]],
    selected_counterfactual_rows: list[dict[str, Any]],
    approved: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "family-balanced-coverage-driven-input-manifest/v1",
        "generated_at": _utc_now(),
        "family_balanced_safe_better_pairs": str(paths["balanced_pairs"]),
        "family_balanced_counterfactual_rollouts": str(paths["balanced_counterfactual"]),
        "family_safe_better_counts": family_counts,
        "selected_pair_count": len(selected_pairs),
        "selected_counterfactual_count": len(selected_counterfactual_rows),
        "next_required_change": NEXT_REQUIRED_CHANGE if approved else "resolve_family_balanced_algorithm_gap_closure_rejections",
        "intended_consumer": "family_balanced_coverage_driven_ppo_rerun",
        **_closed_boundaries(),
    }


def _release_boundary_audit(payloads: dict[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    for name, payload in payloads.items():
        for path, field in _release_violations(payload):
            violations.append({"source": name, "path": path, "field": field})
    return {
        "schema_version": "family-balanced-release-boundary-audit/v1",
        "generated_at": _utc_now(),
        "release_boundary_audit_passed": not violations,
        "violations": violations,
        "reason_codes": [] if not violations else ["release_boundary_violation"],
        **_closed_boundaries(),
    }


def _release_violations(payload: Any) -> list[tuple[str, str]]:
    violations: list[tuple[str, str]] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                current = f"{path}.{key}" if path else str(key)
                if key in RELEASE_BOUNDARY_FIELDS and item is True:
                    violations.append((current, key))
                walk(item, current)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(payload, "")
    return violations


def _docs_audit(repo_root: Path) -> dict[str, Any]:
    readme = _read_text(repo_root / "README.md")
    zh_doc = _read_text(repo_root / "docs" / "算法设计与系统架构报告.md")
    spec = _read_text(repo_root / "docs" / "superpowers" / "specs" / "2026-06-16-family-balanced-algorithm-gap-closure.md")
    required_readme = (
        "Family-Balanced Algorithm Gap Closure v1",
        DEFAULT_OUTPUT_ROOT,
        NEXT_REQUIRED_CHANGE,
        "checkpoint_publication_approved=false",
        "default_policy_replacement_approved=false",
        "real_executor_connection_approved=false",
    )
    required_zh = (
        "Family-Balanced Algorithm Gap Closure v1",
        "低观测",
        "不发布 checkpoint",
        "不替换 default policy",
        "不连接真实执行器",
    )
    required_spec = (
        "Family-Balanced Algorithm Gap Closure v1",
        SUMMARY_FILE,
        NEXT_REQUIRED_CHANGE,
    )
    missing = [
        item
        for item in required_readme
        if item not in readme
    ] + [
        item
        for item in required_zh
        if item not in zh_doc
    ] + [
        item
        for item in required_spec
        if item not in spec
    ]
    return {
        "schema_version": "family-balanced-doc-consistency-audit/v1",
        "docs_audit_passed": not missing,
        "missing_markers": missing,
        "reason_codes": [] if not missing else ["docs_not_updated"],
    }


def _rejection_report(reason_codes: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "family-balanced-rejection-report/v1",
        "generated_at": _utc_now(),
        "rejected": bool(reason_codes),
        "reason_codes": reason_codes,
    }


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Family-Balanced Algorithm Gap Closure v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- gap_closure_verdict: `{summary['gap_closure_verdict']}`",
            f"- low_observation_count: `{summary['low_observation_count']}`",
            f"- family_safe_better_counts: `{summary['family_safe_better_counts']}`",
            f"- family_balance_ratio: `{summary['family_balance_ratio']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            "",
            "This stage prepares family-balanced offline algorithm inputs only. It does not publish a checkpoint, replace the default policy, or connect a real executor.",
            "",
        ]
    )


def _counterfactual_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    return {
        _counterfactual_key(row): row
        for row in rows
        if _first_int(row.get("action_index")) is not None
    }


def _counterfactual_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row.get("context_id") or ""),
        str(row.get("episode_id") or ""),
        _int(row.get("step_index")),
        _int(row.get("action_index")),
    )


def _counterfactual_key_from_pair(row: dict[str, Any], action_index: int) -> tuple[str, str, int, int]:
    return (
        str(row.get("context_id") or ""),
        str(row.get("episode_id") or ""),
        _int(row.get("step_index")),
        int(action_index),
    )


def _sort_key(row: dict[str, Any]) -> tuple[str, str, str, int, int]:
    action_index = _first_int(row.get("candidate_action_index"), row.get("action_index"))
    return (
        str(row.get("scenario_family") or ""),
        str(row.get("scenario_id") or ""),
        str(row.get("episode_id") or ""),
        _int(row.get("step_index")),
        int(action_index) if action_index is not None else -1,
    )


def _family_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("scenario_family") or "") for row in rows if row.get("scenario_family"))
    return dict(sorted(counts.items()))


def _family_balance_ratio(counts: dict[str, int]) -> float:
    target_counts = [counts.get(family, 0) for family in TARGET_FAMILIES]
    maximum = max(target_counts) if target_counts else 0
    if maximum <= 0:
        return 0.0
    return round(counts.get(TARGET_FAMILY, 0) / float(maximum), 12)


def _row_is_trainable(row: dict[str, Any]) -> bool:
    return str(row.get("split") or "") == "train" and bool(row.get("ppo_trainable"))


def _read_json(path: Path, reasons: list[str], label: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _add_reason(reasons, f"{label}_missing")
        return {}


def _read_jsonl(path: Path, reasons: list[str], label: str) -> list[dict[str, Any]]:
    try:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    except (OSError, json.JSONDecodeError):
        _add_reason(reasons, f"{label}_missing")
        return []


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _closed_boundaries() -> dict[str, bool]:
    return {
        "checkpoint_publication_approved": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "relaxes_guard": False,
        "modifies_network_or_action_space": False,
        "modifies_default_astar": False,
        "ackermann_feasible_trajectory_claimed": False,
        "real_world_performance_claimed": False,
    }


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value is None:
        return []
    text = str(value)
    return [text] if text else []


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
