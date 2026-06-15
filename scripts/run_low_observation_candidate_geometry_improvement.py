from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
    from run_low_observation_counterfactual_opportunity_generation import (
        run_low_observation_counterfactual_opportunity_generation,
    )
    from run_quasi_real_map_path_feedback_bridge import run_quasi_real_map_path_feedback_bridge
    from run_safe_better_pair_expansion_across_families import (
        DEFAULT_FORMAL_ROOT,
        DEFAULT_OUTPUT_ROOT as DEFAULT_SAFE_BETTER_ROOT,
        DEFAULT_QUASI_REAL_ROOT,
        DEFAULT_REPLAY_ROOT,
        DEFAULT_REWARD_ROOT,
        DEFAULT_SIGNAL_ROOT,
        DEFAULT_STAGE5A2_ROOT,
        DEFAULT_TUNING_ROOT,
        run_safe_better_pair_expansion_across_families,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.run_low_observation_counterfactual_opportunity_generation import (
        run_low_observation_counterfactual_opportunity_generation,
    )
    from scripts.run_quasi_real_map_path_feedback_bridge import run_quasi_real_map_path_feedback_bridge
    from scripts.run_safe_better_pair_expansion_across_families import (
        DEFAULT_FORMAL_ROOT,
        DEFAULT_OUTPUT_ROOT as DEFAULT_SAFE_BETTER_ROOT,
        DEFAULT_QUASI_REAL_ROOT,
        DEFAULT_REPLAY_ROOT,
        DEFAULT_REWARD_ROOT,
        DEFAULT_SIGNAL_ROOT,
        DEFAULT_STAGE5A2_ROOT,
        DEFAULT_TUNING_ROOT,
        run_safe_better_pair_expansion_across_families,
    )


CONFIG_SCHEMA_VERSION = "quasi-real-low-observation-candidate-geometry-config/v1"
SUMMARY_SCHEMA_VERSION = "low-observation-candidate-geometry-summary/v1"
OVERLAY_ROW_SCHEMA_VERSION = "low-observation-candidate-geometry-overlay-row/v1"
COUNTERFACTUAL_ROW_SCHEMA_VERSION = "low-observation-candidate-geometry-counterfactual-row/v1"
GAP_SCHEMA_VERSION = "low-observation-candidate-geometry-family-gap-report/v1"

DEFAULT_CONFIG = "configs/quasi_real_low_observation_candidate_geometry_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_low_observation_candidate_geometry_improvement_v1"
DEFAULT_SOURCE_MATRIX = "model-explorer/data/manifests/lunar_south_pole_lro_lola_selection_matrix_v1.json"
DEFAULT_OPPORTUNITY_ROOT = "low-observation-opportunity-audit"
TARGET_FAMILY = "low_observation_count"
MIN_TRAINABLE_SAFE_BETTER_PAIR_COUNT = 8
TOLERANCE = 1.0e-9

SUMMARY_FILE = "low-observation-candidate-geometry-summary.json"
OVERLAY_FILE = "low-observation-candidate-geometry-overlay.jsonl"
COUNTERFACTUAL_FILE = "low-observation-candidate-geometry-counterfactual-rollouts.jsonl"
FAMILY_GAP_FILE = "low-observation-candidate-geometry-family-gap-report.json"
REPORT_FILE = "low-observation-candidate-geometry-report.md"
MATRIX_FILE = "low-observation-candidate-geometry-matrix.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Improve low-observation candidate geometry.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--path-feedback-summary", action="append", default=[])
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tuning-root", default=DEFAULT_TUNING_ROOT)
    parser.add_argument("--stage5a2-root", default=DEFAULT_STAGE5A2_ROOT)
    parser.add_argument("--quasi-real-root", default=DEFAULT_QUASI_REAL_ROOT)
    parser.add_argument("--formal-training-root", default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--post-training-replay-root", default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--coverage-signal-root", default=DEFAULT_SIGNAL_ROOT)
    parser.add_argument("--reward-refinement-root", default=DEFAULT_REWARD_ROOT)
    parser.add_argument("--safe-better-output-root", default=DEFAULT_SAFE_BETTER_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--skip-path-feedback-run", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    summary = run_low_observation_candidate_geometry_improvement(
        config_path=_resolve_path(Path(args.config), repo_root),
        path_feedback_summary_paths=[
            _resolve_path(Path(path), repo_root) for path in args.path_feedback_summary
        ],
        output_root=_resolve_path(Path(args.output_root), repo_root),
        repo_root=repo_root,
        tuning_root=_resolve_path(Path(args.tuning_root), repo_root),
        stage5a2_root=_resolve_path(Path(args.stage5a2_root), repo_root),
        quasi_real_root=_resolve_path(Path(args.quasi_real_root), repo_root),
        formal_training_root=_resolve_path(Path(args.formal_training_root), repo_root),
        post_training_replay_root=_resolve_path(Path(args.post_training_replay_root), repo_root),
        coverage_signal_root=_resolve_path(Path(args.coverage_signal_root), repo_root),
        reward_refinement_root=_resolve_path(Path(args.reward_refinement_root), repo_root),
        safe_better_output_root=_resolve_path(Path(args.safe_better_output_root), repo_root),
        execute_path_feedback=not args.skip_path_feedback_run,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "low_observation_trainable_safe_better_pair_count": summary[
                    "low_observation_trainable_safe_better_pair_count"
                ],
                "safe_better_than_teacher_family_count": summary[
                    "safe_better_than_teacher_family_count"
                ],
                "performance_claimed": summary["performance_claimed"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_low_observation_candidate_geometry_improvement(
    *,
    config_path: Path,
    path_feedback_summary_paths: list[Path] | None,
    output_root: Path,
    repo_root: Path,
    tuning_root: Path,
    stage5a2_root: Path,
    quasi_real_root: Path,
    formal_training_root: Path,
    post_training_replay_root: Path,
    coverage_signal_root: Path,
    reward_refinement_root: Path,
    safe_better_output_root: Path,
    execute_path_feedback: bool = True,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_root)
    input_reasons: list[str] = []
    config = _read_json(config_path, input_reasons, "low_observation_candidate_geometry_config_missing")
    target_family = str(config.get("target_family") or TARGET_FAMILY)
    top_k = _int(config.get("top_k"), default=8)
    min_trainable = _int(
        config.get("minimum_trainable_safe_better_pair_count"),
        default=MIN_TRAINABLE_SAFE_BETTER_PAIR_COUNT,
    )

    generated_artifacts: dict[str, Any] = {}
    path_feedback_paths = [
        _resolve_path(Path(path), repo_root) for path in (path_feedback_summary_paths or [])
    ]
    if not path_feedback_paths and execute_path_feedback:
        path_feedback_paths, generated_artifacts = _run_path_feedback_generation(
            config=config,
            output_root=output_root,
            repo_root=repo_root,
            target_family=target_family,
        )
    if not path_feedback_paths:
        input_reasons.append("low_observation_path_feedback_summary_missing")

    geometry = _materialize_geometry_overlay(
        path_feedback_summary_paths=path_feedback_paths,
        target_family=target_family,
        top_k=top_k,
        input_reasons=input_reasons,
    )
    _write_jsonl(paths["overlay"], geometry["overlay_rows"])
    _write_jsonl(paths["counterfactual"], geometry["counterfactual_rows"])

    opportunity_root = output_root / DEFAULT_OPPORTUNITY_ROOT
    opportunity_summary = run_low_observation_counterfactual_opportunity_generation(
        config_path=config_path,
        source_overlay_paths=[paths["overlay"]],
        source_counterfactual_paths=[paths["counterfactual"]],
        output_root=opportunity_root,
        repo_root=repo_root,
    )
    safe_better_summary = run_safe_better_pair_expansion_across_families(
        tuning_root=tuning_root,
        stage5a2_root=stage5a2_root,
        quasi_real_root=quasi_real_root,
        formal_training_root=formal_training_root,
        post_training_replay_root=post_training_replay_root,
        coverage_signal_root=coverage_signal_root,
        reward_refinement_root=reward_refinement_root,
        output_root=safe_better_output_root,
        repo_root=repo_root,
        supplemental_overlay_paths=[
            Path(opportunity_summary["produced_artifacts"]["supplemental_overlay"])
        ],
        supplemental_counterfactual_paths=[
            Path(opportunity_summary["produced_artifacts"]["supplemental_counterfactual_rollouts"])
        ],
    )

    reason_codes = _reason_codes(
        input_reasons=input_reasons,
        geometry=geometry,
        opportunity_summary=opportunity_summary,
        safe_better_summary=safe_better_summary,
        min_trainable=min_trainable,
    )
    status = "passed" if not reason_codes else "failed"
    family_gap = _family_gap_report(
        target_family=target_family,
        geometry=geometry,
        opportunity_summary=opportunity_summary,
        safe_better_summary=safe_better_summary,
        min_trainable=min_trainable,
    )
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason_codes": reason_codes,
        "next_required_change": None
        if status == "passed"
        else "refine_low_observation_candidate_geometry_or_frontier_ranking",
        "target_family": target_family,
        "config_path": str(config_path),
        "output_root": str(output_root),
        "path_feedback_summary_paths": [str(path) for path in path_feedback_paths],
        "generated_artifacts": generated_artifacts,
        "produced_artifacts": {
            "candidate_overlay": str(paths["overlay"]),
            "counterfactual_rollouts": str(paths["counterfactual"]),
            "low_observation_opportunity_summary": str(opportunity_root / "low-observation-counterfactual-opportunity-summary.json"),
            "low_observation_opportunity_root": str(opportunity_root),
            "safe_better_pair_expansion_summary": str(safe_better_output_root / "safe-better-pair-expansion-summary.json"),
            "family_gap_report": str(paths["family_gap"]),
            "report": str(paths["report"]),
        },
        "source_path_feedback_scenario_count": geometry["scenario_count"],
        "low_observation_geometry_context_count": geometry["context_count"],
        "low_observation_geometry_candidate_count": geometry["candidate_count"],
        "geometry_ranked_candidate_count": geometry["ranked_candidate_count"],
        "frontier_unknown_adjacent_candidate_count": geometry["frontier_unknown_adjacent_candidate_count"],
        "low_observation_trainable_safe_better_pair_count": _int(
            opportunity_summary.get("low_observation_trainable_safe_better_pair_count")
        ),
        "low_observation_diagnostic_safe_better_pair_count": _int(
            opportunity_summary.get("low_observation_diagnostic_safe_better_pair_count")
        ),
        "safe_better_than_teacher_candidate_count": _int(
            safe_better_summary.get("safe_better_than_teacher_candidate_count")
        ),
        "safe_better_than_teacher_family_count": _int(
            safe_better_summary.get("safe_better_than_teacher_family_count")
        ),
        "family_safe_better_counts": safe_better_summary.get("family_safe_better_counts", {}),
        "missing_counterfactual_source_count": max(
            geometry["missing_counterfactual_source_count"],
            _int(opportunity_summary.get("missing_counterfactual_source_count")),
        ),
        "fallback_gain_contamination_count": max(
            geometry["fallback_gain_contamination_count"],
            _int(opportunity_summary.get("fallback_gain_contamination_count")),
        ),
        "controlled_regression_count": max(
            geometry["controlled_regression_count"],
            _int(opportunity_summary.get("controlled_regression_count")),
        ),
        "guard_rejected_candidate_count": geometry["guard_rejected_candidate_count"],
        "non_positive_coverage_advantage_count": geometry["non_positive_coverage_advantage_count"],
        "opportunity_status": opportunity_summary.get("status"),
        "opportunity_reason_codes": opportunity_summary.get("reason_codes", []),
        "safe_better_status": safe_better_summary.get("status"),
        "safe_better_reason_codes": safe_better_summary.get("reason_codes", []),
        "runs_model_explorer_path_feedback": bool(generated_artifacts),
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "modifies_network_or_action_space": False,
        "modifies_default_astar": False,
        "relaxes_guard": False,
        "performance_claimed": False,
        "formal_release_claimed": False,
        "git_provenance": git_snapshot(repo_root),
        "summary": _summary_sentence(status=status, reason_codes=reason_codes, opportunity=opportunity_summary),
    }
    _write_json(paths["family_gap"], family_gap)
    _write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, family_gap), encoding="utf-8")
    return summary


def _run_path_feedback_generation(
    *,
    config: dict[str, Any],
    output_root: Path,
    repo_root: Path,
    target_family: str,
) -> tuple[list[Path], dict[str, Any]]:
    matrix_path = output_root / MATRIX_FILE
    matrix_payload = _low_observation_matrix_payload(config=config, repo_root=repo_root, target_family=target_family)
    _write_json(matrix_path, matrix_payload)
    bridge_summary = run_quasi_real_map_path_feedback_bridge(
        matrix_manifest_path=matrix_path,
        output_root=output_root,
        config=config,
        repo_root=repo_root,
    )
    manifest_path = Path(bridge_summary["path_feedback_manifest"])
    env = dict(os.environ)
    src = str(repo_root / "model-explorer" / "src")
    env["PYTHONPATH"] = f"{src}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src
    stdout_path = output_root / "path-feedback-run-stdout.json"
    stderr_path = output_root / "path-feedback-run-stderr.txt"
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr_handle:
        subprocess.run(
            [sys.executable, "-m", "model_explorer", "path-feedback", "run", str(manifest_path)],
            cwd=repo_root,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=True,
        )
    summary_path = output_root / "quasi-real-map-path-feedback-summary.json"
    return [summary_path], {
        "matrix_manifest": str(matrix_path),
        "bridge_summary": str(output_root / "quasi-real-map-path-feedback-bridge-summary.json"),
        "path_feedback_manifest": str(manifest_path),
        "path_feedback_summary": str(summary_path),
        "path_feedback_stdout": str(stdout_path),
        "path_feedback_stderr": str(stderr_path),
    }


def _low_observation_matrix_payload(
    *,
    config: dict[str, Any],
    repo_root: Path,
    target_family: str,
) -> dict[str, Any]:
    _ensure_model_explorer_path(repo_root)
    from model_explorer.data.evaluation_matrix import load_quasi_real_evaluation_manifest

    source_matrix = _resolve_path(Path(config.get("source_matrix_manifest", DEFAULT_SOURCE_MATRIX)), repo_root)
    source_manifest = load_quasi_real_evaluation_manifest(source_matrix)
    starts = [_cell_tuple(item) for item in config.get("start_cells", [[0, 0]])]
    offsets = [_cell_tuple(item) for item in config.get("roi_offsets", [[0, 0]])]
    starts = [item for item in starts if item is not None]
    offsets = [item for item in offsets if item is not None]
    rois: list[dict[str, Any]] = []
    candidate_count = _int(config.get("candidate_count"), default=16)
    base_seed = _int(config.get("seed"), default=20260615)
    for roi_index, roi in enumerate(source_manifest.rois):
        if str(roi.name) != target_family:
            rois.append(
                {
                    "name": str(roi.name),
                    "split": str(roi.split),
                    "roi_x": int(roi.roi_x),
                    "roi_y": int(roi.roi_y),
                    "roi_width": int(roi.roi_width),
                    "roi_height": int(roi.roi_height),
                    "candidate_count": max(1, min(candidate_count, int(getattr(roi, "candidate_count", 1)))),
                    "episode_count": 1,
                    "seed": int(getattr(roi, "seed", base_seed)),
                    "start_cell": [int(roi.start_cell[0]), int(roi.start_cell[1])],
                    "scenario_id": f"qreal_low_observation_geometry_required_{roi.name}_{roi.split}_{roi_index:03d}",
                    "scenario_variant_id": (
                        f"qreal_low_observation_geometry_required_{roi.name}_{roi.split}_{roi_index:03d}"
                        f"-seed-{int(getattr(roi, 'seed', base_seed))}-start-{int(roi.start_cell[0])}-{int(roi.start_cell[1])}"
                    ),
                }
            )
            continue
        width = int(roi.roi_width)
        height = int(roi.roi_height)
        valid_starts = [start for start in starts if 0 <= start[0] < width and 0 <= start[1] < height]
        for start_index, start in enumerate(valid_starts):
            offset = offsets[start_index % len(offsets)]
            roi_x = max(0, int(roi.roi_x) + offset[0])
            roi_y = max(0, int(roi.roi_y) + offset[1])
            seed = int(getattr(roi, "seed", base_seed)) + start_index * 101
            scenario_id = f"qreal_low_observation_geometry_{roi.split}_{roi_index:03d}_s{start_index:02d}"
            rois.append(
                {
                    "name": str(roi.name),
                    "split": str(roi.split),
                    "roi_x": roi_x,
                    "roi_y": roi_y,
                    "roi_width": width,
                    "roi_height": height,
                    "candidate_count": candidate_count,
                    "episode_count": 1,
                    "seed": seed,
                    "start_cell": [start[0], start[1]],
                    "scenario_id": scenario_id,
                    "scenario_variant_id": f"{scenario_id}-seed-{seed}-start-{start[0]}-{start[1]}",
                }
            )
    return {
        "schema_version": "model-explorer-quasi-real-evaluation/v1",
        "name": "qreal-low-observation-candidate-geometry-v1",
        "run_id": "low-observation-candidate-geometry-v1",
        "dataset_manifest": str(source_manifest.dataset_manifest),
        "output_root": str(config.get("processed_output_root", "../processed/qreal_low_observation_candidate_geometry_v1")),
        "candidate_count": candidate_count,
        "episode_count": 1,
        "seed": base_seed,
        "source_matrix_manifest": str(source_matrix),
        "rois": rois,
        "mask_stress": {"enabled": False},
        "selection": {"enabled": False},
        "dataset_validation": dict(config.get("dataset_validation", {})),
        "train": dict(config.get("train", {})),
    }


def _materialize_geometry_overlay(
    *,
    path_feedback_summary_paths: list[Path],
    target_family: str,
    top_k: int,
    input_reasons: list[str],
) -> dict[str, Any]:
    overlay_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    scenario_count = 0
    context_ids: set[str] = set()
    candidate_count = 0
    ranked_candidate_count = 0
    frontier_unknown_adjacent_candidate_count = 0
    missing_counterfactual_source_count = 0
    fallback_gain_contamination_count = 0
    controlled_regression_count = 0
    guard_rejected_candidate_count = 0
    non_positive_coverage_advantage_count = 0

    for summary_path in path_feedback_summary_paths:
        payload = _read_json(summary_path, input_reasons, "path_feedback_summary_missing")
        scenarios = payload.get("scenarios", []) if isinstance(payload.get("scenarios"), list) else []
        for scenario_index, scenario in enumerate(scenarios):
            if _scenario_family(scenario) != target_family:
                continue
            contract_goals = _contract_goals_for_scenario(summary_path=summary_path, scenario=scenario)
            candidates = [
                _enrich_candidate_from_contract(candidate, contract_goals)
                for candidate in _scenario_candidates(scenario)
            ]
            if not candidates:
                continue
            teacher = _teacher_candidate(scenario, candidates)
            if teacher is None:
                continue
            scenario_count += 1
            split = _split_from_scenario(scenario)
            context_id = _context_id(scenario, teacher)
            context_ids.add(context_id)
            teacher_row = _overlay_row(
                scenario=scenario,
                candidate=teacher,
                source_path=summary_path,
                context_id=context_id,
                split=split,
                step_index=scenario_index,
                is_teacher=True,
                teacher_action_index=_action_index(teacher),
            )
            overlay_rows.append(teacher_row)
            teacher_coverage = _coverage_value(teacher)
            non_teacher = [candidate for candidate in candidates if _action_index(candidate) != _action_index(teacher)]
            ranked = sorted(non_teacher, key=_geometry_priority, reverse=True)[: max(top_k, 0)]
            ranked_candidate_count += len(ranked)
            for candidate in ranked:
                candidate_count += 1
                if _is_frontier_unknown_adjacent(candidate):
                    frontier_unknown_adjacent_candidate_count += 1
                candidate_coverage = _coverage_value(candidate)
                source_available = candidate_coverage is not None
                if not source_available:
                    missing_counterfactual_source_count += 1
                fallback_like = _is_fallback_like(scenario, candidate)
                coverage_advantage = (
                    candidate_coverage - teacher_coverage
                    if candidate_coverage is not None and teacher_coverage is not None
                    else None
                )
                if coverage_advantage is not None and coverage_advantage <= TOLERANCE:
                    non_positive_coverage_advantage_count += 1
                if coverage_advantage is not None and coverage_advantage > TOLERANCE and fallback_like:
                    fallback_gain_contamination_count += 1
                if _controlled_regression_reasons(candidate):
                    controlled_regression_count += 1
                if _guard_rejected(candidate):
                    guard_rejected_candidate_count += 1
                overlay_rows.append(
                    _overlay_row(
                        scenario=scenario,
                        candidate=candidate,
                        source_path=summary_path,
                        context_id=context_id,
                        split=split,
                        step_index=scenario_index,
                        is_teacher=False,
                        teacher_action_index=_action_index(teacher),
                    )
                )
                counterfactual_rows.append(
                    _counterfactual_row(
                        scenario=scenario,
                        candidate=candidate,
                        source_path=summary_path,
                        context_id=context_id,
                        split=split,
                        step_index=scenario_index,
                        teacher=teacher,
                    )
                )

    return {
        "overlay_rows": overlay_rows,
        "counterfactual_rows": counterfactual_rows,
        "scenario_count": scenario_count,
        "context_count": len(context_ids),
        "candidate_count": candidate_count,
        "ranked_candidate_count": ranked_candidate_count,
        "frontier_unknown_adjacent_candidate_count": frontier_unknown_adjacent_candidate_count,
        "missing_counterfactual_source_count": missing_counterfactual_source_count,
        "fallback_gain_contamination_count": fallback_gain_contamination_count,
        "controlled_regression_count": controlled_regression_count,
        "guard_rejected_candidate_count": guard_rejected_candidate_count,
        "non_positive_coverage_advantage_count": non_positive_coverage_advantage_count,
    }


def _overlay_row(
    *,
    scenario: dict[str, Any],
    candidate: dict[str, Any],
    source_path: Path,
    context_id: str,
    split: str,
    step_index: int,
    is_teacher: bool,
    teacher_action_index: int | None,
) -> dict[str, Any]:
    coverage = _coverage_value(candidate)
    return {
        "schema_version": OVERLAY_ROW_SCHEMA_VERSION,
        "context_id": context_id,
        "episode_id": f"low-observation-geometry-{scenario.get('scenario_id')}",
        "step_index": step_index,
        "scenario_id": scenario.get("scenario_id"),
        "scenario_family": _scenario_family(scenario),
        "split": split,
        "action_index": _action_index(candidate),
        "candidate_cell": candidate.get("cell") or candidate.get("candidate_cell"),
        "action_mask_valid": not _guard_rejected(candidate),
        "teacher_action_index": teacher_action_index,
        "is_teacher_action": bool(is_teacher),
        "is_pre_improvement_selected_action": bool(is_teacher),
        "is_post_update_raw_action": bool(is_teacher),
        "is_post_update_controlled_action": bool(is_teacher),
        "counterfactual_coverage_observed": bool(is_teacher),
        "fallback_like": _is_fallback_like(scenario, candidate),
        "guard_rejected": _guard_rejected(candidate),
        "policy_action_accepted": not _guard_rejected(candidate),
        "controlled_regression_reason_codes": _controlled_regression_reasons(candidate),
        "coverage_source_available": coverage is not None,
        "expected_coverage_rate_delta": coverage,
        "expected_new_coverage_area": _first_float(candidate.get("expected_new_coverage_area"), coverage),
        "information_gain": _first_float(candidate.get("information_gain"), coverage),
        "value": _first_float(candidate.get("value"), candidate.get("valuable_coverage_proxy")),
        "valuable_coverage_proxy": _first_float(candidate.get("valuable_coverage_proxy"), candidate.get("value"), coverage),
        "path_cost": _float(candidate.get("path_cost")),
        "risk": _float(candidate.get("risk")),
        "energy_cost": _first_float(candidate.get("energy_cost"), 0.0),
        "source_path": str(source_path),
        "match_method": "low_observation_geometry_path_feedback",
        "source_confidence": 1.0 if coverage is not None else 0.0,
        "source_execution_type": "low_observation_geometry_counterfactual_path_feedback",
        "missing_reason_codes": [] if coverage is not None else ["low_observation_candidate_coverage_source_missing"],
        "geometry_priority": _geometry_priority(candidate),
        "geometry_tags": _geometry_tags(candidate),
    }


def _counterfactual_row(
    *,
    scenario: dict[str, Any],
    candidate: dict[str, Any],
    source_path: Path,
    context_id: str,
    split: str,
    step_index: int,
    teacher: dict[str, Any],
) -> dict[str, Any]:
    coverage = _coverage_value(candidate)
    teacher_coverage = _coverage_value(teacher)
    return {
        "schema_version": COUNTERFACTUAL_ROW_SCHEMA_VERSION,
        "context_id": context_id,
        "episode_id": f"low-observation-geometry-{scenario.get('scenario_id')}",
        "step_index": step_index,
        "scenario_id": scenario.get("scenario_id"),
        "scenario_family": _scenario_family(scenario),
        "split": split,
        "action_index": _action_index(candidate),
        "candidate_cell": candidate.get("cell") or candidate.get("candidate_cell"),
        "teacher_action_index": _action_index(teacher),
        "coverage_source_available": coverage is not None,
        "expected_coverage_rate_delta": coverage,
        "teacher_expected_coverage_rate_delta": teacher_coverage,
        "coverage_advantage": coverage - teacher_coverage
        if coverage is not None and teacher_coverage is not None
        else None,
        "expected_new_coverage_area": _first_float(candidate.get("expected_new_coverage_area"), coverage),
        "information_gain": _first_float(candidate.get("information_gain"), coverage),
        "valuable_coverage_proxy": _first_float(candidate.get("valuable_coverage_proxy"), candidate.get("value"), coverage),
        "match_method": "low_observation_geometry_path_feedback",
        "source_path": str(source_path),
        "fallback_like": _is_fallback_like(scenario, candidate),
        "guard_rejected": _guard_rejected(candidate),
        "controlled_regression_reason_codes": _controlled_regression_reasons(candidate),
    }


def _reason_codes(
    *,
    input_reasons: list[str],
    geometry: dict[str, Any],
    opportunity_summary: dict[str, Any],
    safe_better_summary: dict[str, Any],
    min_trainable: int,
) -> list[str]:
    reasons: list[str] = []
    for reason in input_reasons:
        _add_reason(reasons, reason)
    trainable = _int(opportunity_summary.get("low_observation_trainable_safe_better_pair_count"))
    if trainable < min_trainable:
        _add_reason(reasons, "low_observation_geometry_no_positive_coverage_advantage")
    if geometry["missing_counterfactual_source_count"] > 0 or _int(opportunity_summary.get("missing_counterfactual_source_count")) > 0:
        _add_reason(reasons, "low_observation_counterfactual_source_missing")
    if geometry["fallback_gain_contamination_count"] > 0 or _int(opportunity_summary.get("fallback_gain_contamination_count")) > 0:
        _add_reason(reasons, "fallback_gain_contamination")
    if geometry["controlled_regression_count"] > 0 or _int(opportunity_summary.get("controlled_regression_count")) > 0:
        _add_reason(reasons, "controlled_regression")
    if _int(safe_better_summary.get("safe_better_than_teacher_family_count")) < 4:
        _add_reason(reasons, "safe_better_family_count_below_threshold")
    family_counts = safe_better_summary.get("family_safe_better_counts", {})
    if _int(family_counts.get(TARGET_FAMILY)) < min_trainable:
        _add_reason(reasons, "family_safe_better_gap_low_observation_count")
    return reasons


def _family_gap_report(
    *,
    target_family: str,
    geometry: dict[str, Any],
    opportunity_summary: dict[str, Any],
    safe_better_summary: dict[str, Any],
    min_trainable: int,
) -> dict[str, Any]:
    return {
        "schema_version": GAP_SCHEMA_VERSION,
        "target_family": target_family,
        "minimum_trainable_safe_better_pair_count": min_trainable,
        "geometry_candidate_count": geometry["candidate_count"],
        "low_observation_trainable_safe_better_pair_count": _int(
            opportunity_summary.get("low_observation_trainable_safe_better_pair_count")
        ),
        "safe_better_than_teacher_family_count": _int(
            safe_better_summary.get("safe_better_than_teacher_family_count")
        ),
        "family_safe_better_counts": safe_better_summary.get("family_safe_better_counts", {}),
    }


def _render_report(summary: dict[str, Any], family_gap: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Low-Observation Candidate Geometry Improvement v1",
            "",
            "## Summary",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- low_observation_trainable_safe_better_pair_count: `{summary['low_observation_trainable_safe_better_pair_count']}`",
            f"- safe_better_than_teacher_family_count: `{summary['safe_better_than_teacher_family_count']}`",
            f"- missing_counterfactual_source_count: `{summary['missing_counterfactual_source_count']}`",
            f"- fallback_gain_contamination_count: `{summary['fallback_gain_contamination_count']}`",
            f"- performance_claimed: `{summary['performance_claimed']}`",
            "",
            "## Family Gap",
            "",
            f"- target_family: `{family_gap['target_family']}`",
            f"- family_safe_better_counts: `{family_gap['family_safe_better_counts']}`",
            "",
            "## Non-Goals",
            "",
            "- no PPO update",
            "- no checkpoint publication",
            "- no default policy replacement",
            "- no real executor connection",
            "- no network/action-space/default-A* change",
            "- no guard relaxation",
            "- no performance claim",
            "",
        ]
    )


def _summary_sentence(
    *,
    status: str,
    reason_codes: list[str],
    opportunity: dict[str, Any],
) -> str:
    count = _int(opportunity.get("low_observation_trainable_safe_better_pair_count"))
    if status == "passed":
        return f"Low-observation geometry passed with {count} trainable safe-better pairs."
    return f"Low-observation geometry failed because {reason_codes}; trainable low-observation pairs={count}."


def _scenario_candidates(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    path_feedback = scenario.get("path_feedback")
    candidates = path_feedback.get("candidates", []) if isinstance(path_feedback, dict) else []
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


def _teacher_candidate(
    scenario: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    selected = scenario.get("selected_cell_after_path_feedback") or scenario.get("selected_cell_before_path_feedback")
    if selected is not None:
        for candidate in candidates:
            if candidate.get("cell") == selected or candidate.get("candidate_cell") == selected:
                return candidate
    for candidate in candidates:
        if _action_index(candidate) == 0:
            return candidate
    return candidates[0] if candidates else None


def _contract_goals_for_scenario(
    *,
    summary_path: Path,
    scenario: dict[str, Any],
) -> dict[tuple[int, int], dict[str, Any]]:
    scenario_id = scenario.get("scenario_id")
    if not scenario_id:
        return {}
    contract_path = summary_path.parent / "path_planner_sidecars" / f"{scenario_id}.contract.json"
    if not contract_path.is_file():
        return {}
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    goals: dict[tuple[int, int], dict[str, Any]] = {}
    for goal in payload.get("top_goals", []) if isinstance(payload.get("top_goals"), list) else []:
        if not isinstance(goal, dict):
            continue
        cell = _cell_tuple(goal.get("cell"))
        if cell is not None:
            goals[cell] = goal
    return goals


def _enrich_candidate_from_contract(
    candidate: dict[str, Any],
    contract_goals: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    cell = _cell_tuple(candidate.get("cell") or candidate.get("candidate_cell"))
    if cell is None or cell not in contract_goals:
        return candidate
    goal = contract_goals[cell]
    enriched = dict(candidate)
    for field in (
        "expected_coverage_rate_delta",
        "expected_new_coverage_area",
        "information_gain",
        "confidence_gain",
        "value",
        "valuable_coverage_proxy",
        "risk",
        "path_cost",
        "energy_cost",
        "observation_count",
        "data_confidence",
    ):
        if enriched.get(field) is None and goal.get(field) is not None:
            enriched[field] = goal.get(field)
    if enriched.get("valuable_coverage_proxy") is None and goal.get("value") is not None:
        enriched["valuable_coverage_proxy"] = goal.get("value")
    return enriched


def _scenario_family(scenario: dict[str, Any]) -> str:
    return str(
        scenario.get("scenario_family")
        or scenario.get("scenario_group")
        or scenario.get("roi_group")
        or ""
    )


def _split_from_scenario(scenario: dict[str, Any]) -> str:
    split = scenario.get("split")
    if split:
        return str(split)
    scenario_id = str(scenario.get("scenario_id") or "")
    for value in ("train", "validation", "test"):
        if f"_{value}_" in scenario_id or scenario_id.endswith(f"_{value}"):
            return value
    return "unknown"


def _context_id(scenario: dict[str, Any], teacher: dict[str, Any]) -> str:
    value = teacher.get("context_id") or scenario.get("context_id")
    if value:
        return str(value)
    fields = {
        "schema_version": "low-observation-candidate-geometry-context/v1",
        "scenario_id": scenario.get("scenario_id"),
        "scenario_family": _scenario_family(scenario),
        "split": _split_from_scenario(scenario),
        "teacher_action_index": _action_index(teacher),
        "teacher_cell": teacher.get("cell") or teacher.get("candidate_cell"),
    }
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _geometry_priority(candidate: dict[str, Any]) -> float:
    coverage = _coverage_value(candidate)
    coverage_score = coverage if coverage is not None else -1.0
    information_gain = _first_float(candidate.get("information_gain"), 0.0) or 0.0
    valuable = _first_float(candidate.get("valuable_coverage_proxy"), candidate.get("value"), 0.0) or 0.0
    risk = _first_float(candidate.get("risk"), 0.0) or 0.0
    path_cost = _first_float(candidate.get("path_cost"), 0.0) or 0.0
    return 10.0 * coverage_score + 2.0 * information_gain + valuable - 0.05 * risk - 0.0001 * path_cost


def _is_frontier_unknown_adjacent(candidate: dict[str, Any]) -> bool:
    tags = _geometry_tags(candidate)
    return bool({"frontier", "unknown_adjacent", "low_observation"} & set(tags))


def _geometry_tags(candidate: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if _coverage_value(candidate) is not None and (_coverage_value(candidate) or 0.0) > 0:
        tags.append("low_observation")
    if (_first_float(candidate.get("information_gain"), 0.0) or 0.0) > 0:
        tags.append("unknown_adjacent")
    if (_first_float(candidate.get("risk"), 0.0) or 0.0) > 0.6:
        tags.append("rim_adjacent")
    return tags


def _coverage_value(candidate: dict[str, Any]) -> float | None:
    return _float_or_none(
        candidate.get("expected_coverage_rate_delta"),
        candidate.get("coverage_rate_delta"),
        candidate.get("new_area_covered"),
    )


def _is_fallback_like(scenario: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return (
        bool(scenario.get("open_grid_fallback_used"))
        or bool(candidate.get("open_grid_fallback_used"))
        or bool(candidate.get("fallback_like"))
        or str(candidate.get("source_execution_type") or "").startswith("fallback")
    )


def _guard_rejected(candidate: dict[str, Any]) -> bool:
    if bool(candidate.get("guard_rejected")):
        return True
    if candidate.get("reachable") is False:
        return True
    if candidate.get("failure_reason"):
        return True
    return False


def _controlled_regression_reasons(candidate: dict[str, Any]) -> list[Any]:
    reasons = candidate.get("controlled_regression_reason_codes")
    return reasons if isinstance(reasons, list) else []


def _action_index(candidate: dict[str, Any]) -> int | None:
    return _int_or_none(candidate.get("action_index"))


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "overlay": output_root / OVERLAY_FILE,
        "counterfactual": output_root / COUNTERFACTUAL_FILE,
        "family_gap": output_root / FAMILY_GAP_FILE,
        "report": output_root / REPORT_FILE,
    }


def _read_json(path: Path, reasons: list[str], missing_reason: str) -> dict[str, Any]:
    if not path.is_file():
        _add_reason(reasons, missing_reason)
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _add_reason(reasons, f"{missing_reason}_invalid_json")
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _ensure_model_explorer_path(repo_root: Path) -> None:
    src = repo_root / "model-explorer" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _cell_tuple(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return (int(value[0]), int(value[1]))


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _first_float(*values: Any) -> float | None:
    for value in values:
        numeric = _float_or_none(value)
        if numeric is not None:
            return numeric
    return None


def _float(value: Any, *, default: float = 0.0) -> float:
    numeric = _float_or_none(value)
    return default if numeric is None else numeric


def _float_or_none(*values: Any) -> float | None:
    for value in values:
        try:
            if value is None:
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _int(value: Any, *, default: int = 0) -> int:
    numeric = _int_or_none(value)
    return default if numeric is None else numeric


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
