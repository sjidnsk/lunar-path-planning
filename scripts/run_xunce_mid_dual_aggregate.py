"""Independently recalculate the reduced midterm dual-gate from raw rows.

The three source summaries are never used as metric inputs.  Each source
manifest is verified first; its manifest-bound config then provides the
expected scale/input/code lineage for every raw row.  Stored summaries are
read only after recalculation and act as fail-closed consistency checks.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import xunce_artifact_io as artifact_io
from xunce_artifact_paths import (
    MID_DUAL_CONFIG,
    MID_DUAL_MANIFEST,
    MID_DUAL_RESULTS,
    MID_DUAL_SUMMARY,
    artifact_path,
)
from xunce_mid_dual_artifacts import MidDualRunStore
from xunce_mid_dual_contracts import (
    ClosedLoopStepRow,
    CoverageEpisodeRow,
    FINAL_COVERAGE_THRESHOLD,
    FINAL_TIME_MS,
    G1_EPISODES_PER_FORMAL_SPLIT,
    G2_FORMAL_CALLS,
    G2_KILOMETER_REQUESTS,
    G2_PLATFORMS,
    G2_REPEATS,
    G2_STANDARD_REQUESTS,
    InterfaceReplayRow,
    MID_COVERAGE_THRESHOLD,
    MID_TIME_MS,
    PlanningCallRow,
    SCALE_PROFILE,
    evaluate_formal_g2,
    evaluate_g1_split,
    nearest_rank,
)


AGGREGATE_SCHEMA_VERSION = "xunce-mid-dual-independent-aggregate/v1"
AGGREGATE_CONFIG_SCHEMA_VERSION = "xunce-mid-dual-aggregate-config/v1"
_SOURCE_GATES = ("g1", "g2", "g3")
_SHA256_LENGTH = 64


class AggregateBlocked(ValueError):
    """A stable evidence blocker, distinct from a measured gate failure."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(char in "0123456789abcdef" for char in value)
    )


def _exact_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise AggregateBlocked(f"{field_name}_invalid")
    return value


def _exact_nonnegative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise AggregateBlocked(f"{field_name}_invalid")
    return value


def _finite_number(value: object, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise AggregateBlocked(f"{field_name}_nonfinite_or_invalid")
    return float(value)


def _dataclass_kwargs(row: Mapping[str, Any], row_type: type[object]) -> dict[str, Any]:
    names = tuple(item.name for item in fields(row_type))
    missing = [name for name in names if name not in row]
    if missing:
        raise AggregateBlocked(f"row_schema_missing:{','.join(missing)}")
    return {name: row[name] for name in names}


def _status_for(midterm_passed: bool) -> str:
    return "passed" if midterm_passed else "failed"


def midterm_reduced_truth_table(g1: bool, g2: bool, g3: bool) -> bool:
    """The reduced midterm gate requires all three independently recalculated gates."""
    return type(g1) is bool and type(g2) is bool and type(g3) is bool and g1 and g2 and g3


def final_threshold_reduced_truth_table(g1: bool, g2: bool, g3: bool) -> bool:
    """The reduced final-threshold gate also requires all three gates."""
    return type(g1) is bool and type(g2) is bool and type(g3) is bool and g1 and g2 and g3


def blocked_summary(blockers: Iterable[str]) -> dict[str, Any]:
    """Build terminal evidence-not-ready state without a failed/pass alias."""
    unique_blockers = sorted({str(reason) for reason in blockers if str(reason)})
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "scale_profile": SCALE_PROFILE,
        "status": "blocked",
        "formal_evidence_eligible": False,
        "midterm_reduced_gate_passed": False,
        "final_threshold_reduced_gate_passed": False,
        "blockers": unique_blockers,
        "gates": {},
        "sample_counts": {
            "g1_total_episodes": 0,
            "g1_test_q24": 0,
            "g1_unseen24": 0,
            "g2_formal_calls": 0,
            "g3_wheel_episodes": 0,
            "g3_interface_replays": 0,
        },
    }


def _load_verified_source(root: str | Path, gate_id: str) -> dict[str, Any]:
    source_root = Path(root)
    if not source_root.is_absolute():
        raise AggregateBlocked(f"{gate_id}_root_not_absolute")
    try:
        # This is deliberately the first read operation for a source root.
        MidDualRunStore.verify_manifest(source_root)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise AggregateBlocked(f"{gate_id}_manifest_invalid:{type(exc).__name__}") from exc

    manifest_path = artifact_path(source_root, MID_DUAL_MANIFEST)
    config = artifact_io.read_json(artifact_path(source_root, MID_DUAL_CONFIG))
    rows = artifact_io.read_jsonl(artifact_path(source_root, MID_DUAL_RESULTS))
    stored_summary = artifact_io.read_json(artifact_path(source_root, MID_DUAL_SUMMARY))
    manifest = artifact_io.read_json(manifest_path)
    if manifest.get("formal_evidence_eligible") is not True:
        raise AggregateBlocked(f"{gate_id}_source_blocked")
    if stored_summary.get("status") == "blocked":
        raise AggregateBlocked(f"{gate_id}_source_blocked")
    if config.get("gate_id") != gate_id:
        raise AggregateBlocked(f"{gate_id}_config_gate_id_drift")
    if config.get("scale_profile") != SCALE_PROFILE:
        raise AggregateBlocked(f"{gate_id}_config_scale_profile_drift")
    if not _is_sha256(config.get("config_sha256")):
        raise AggregateBlocked(f"{gate_id}_config_sha256_invalid")
    if not _is_sha256(config.get("input_sha256")):
        raise AggregateBlocked(f"{gate_id}_config_input_sha256_invalid")
    if not _is_sha256(config.get("code_sha256")):
        raise AggregateBlocked(f"{gate_id}_config_code_sha256_invalid")
    return {
        "root": source_root,
        "config": config,
        "rows": rows,
        "stored_summary": stored_summary,
        "manifest_sha256": _sha256_bytes(artifact_io.read_bytes(manifest_path)),
    }


def _validate_row_lineage(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    gate_id: str,
) -> None:
    expected_config = config["config_sha256"]
    expected_input = config["input_sha256"]
    expected_code = config["code_sha256"]
    for index, row in enumerate(rows):
        prefix = f"{gate_id}_row_{index}"
        if row.get("scale_profile") != SCALE_PROFILE:
            raise AggregateBlocked(f"{prefix}_scale_profile_drift")
        if row.get("config_sha256") != expected_config:
            raise AggregateBlocked(f"{prefix}_config_sha256_drift")
        if row.get("input_sha256") != expected_input:
            raise AggregateBlocked(f"{prefix}_input_sha256_drift")
        if row.get("code_sha256") != expected_code:
            raise AggregateBlocked(f"{prefix}_code_sha256_drift")
        if row.get("source_sha256") != expected_code:
            raise AggregateBlocked(f"{prefix}_source_sha256_drift")


def _timing_statistics(values: Sequence[object]) -> dict[str, Any]:
    materialized = tuple(_finite_number(value, "timing") for value in values)
    if not materialized:
        raise AggregateBlocked("timing_partition_empty")
    mean_value = statistics.mean(materialized)
    p95 = nearest_rank(materialized, 0.95)
    maximum = max(materialized)
    midterm = mean_value <= MID_TIME_MS and p95 <= MID_TIME_MS and maximum <= MID_TIME_MS
    final = (
        mean_value <= FINAL_TIME_MS
        and p95 <= FINAL_TIME_MS
        and sum(value <= FINAL_TIME_MS for value in materialized) / len(materialized) >= 0.95
        and maximum <= MID_TIME_MS
    )
    return {
        "sample_count": len(materialized),
        "mean_ms": mean_value,
        "p50_ms": nearest_rank(materialized, 0.50),
        "p95_ms": p95,
        "p99_ms": nearest_rank(materialized, 0.99),
        "sample_stddev_ms": statistics.stdev(materialized) if len(materialized) > 1 else 0.0,
        "min_ms": min(materialized),
        "max_ms": maximum,
        "at_or_below_1000_count": sum(value <= FINAL_TIME_MS for value in materialized),
        "over_2000_count": sum(value > MID_TIME_MS for value in materialized),
        "midterm_reduced_passed": midterm,
        "final_threshold_reduced_passed": final,
    }


def recompute_g1(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    """Recalculate both formal G1 splits from their 48 episode rows."""
    if len(rows) != 2 * G1_EPISODES_PER_FORMAL_SPLIT:
        raise AggregateBlocked("g1_formal_episode_count_mismatch")
    if any(row.get("row_kind") != "g1_coverage_episode" for row in rows):
        raise AggregateBlocked("g1_row_kind_mismatch")
    _validate_row_lineage(rows, config, "g1")
    episode_ids = [row.get("episode_id") for row in rows]
    scenario_ids = [row.get("scenario_id") for row in rows]
    if len(set(episode_ids)) != len(rows):
        raise AggregateBlocked("g1_duplicate_episode_id")
    if len(set(scenario_ids)) != len(rows):
        raise AggregateBlocked("g1_duplicate_scenario_id")

    typed_rows: list[tuple[CoverageEpisodeRow, str, str]] = []
    for row in rows:
        try:
            typed = CoverageEpisodeRow(**_dataclass_kwargs(row, CoverageEpisodeRow))
        except (TypeError, ValueError) as exc:
            raise AggregateBlocked(f"g1_row_schema_invalid:{type(exc).__name__}") from exc
        split = row.get("split")
        lane_id = row.get("lane_id")
        if split not in {"test_q24", "unseen24"} or not isinstance(lane_id, str) or not lane_id:
            raise AggregateBlocked("g1_split_or_lane_invalid")
        typed_rows.append((typed, split, lane_id))

    split_results: dict[str, dict[str, Any]] = {}
    for split in ("test_q24", "unseen24"):
        selected = [(row, lane) for row, row_split, lane in typed_rows if row_split == split]
        if len(selected) != G1_EPISODES_PER_FORMAL_SPLIT:
            raise AggregateBlocked(f"g1_{split}_count_mismatch")
        statistics_result = evaluate_g1_split(
            [row.episode_id for row, _ in selected],
            [row.coverage for row, _ in selected],
            lane_ids=[lane for _, lane in selected],
        )
        if statistics_result["status"] == "blocked":
            raise AggregateBlocked(f"g1_{split}_{statistics_result['blocking_reason']}")
        split_results[split] = statistics_result

    safety_clean = all(row.safety_violation_count == 0 for row, _, _ in typed_rows)
    midterm = safety_clean and all(
        result["midterm_reduced_passed"] for result in split_results.values()
    )
    final = safety_clean and all(
        result["final_threshold_reduced_passed"] for result in split_results.values()
    )
    return {
        "status": _status_for(midterm),
        "sample_count": len(rows),
        "g1_coverage_80_passed": midterm,
        "g1_coverage_99_passed": final,
        "safety_clean": safety_clean,
        "splits": split_results,
    }


def recompute_g2(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    """Recalculate correctness and every platform/scale/outcome timing partition."""
    if len(rows) != G2_FORMAL_CALLS:
        raise AggregateBlocked("g2_formal_call_count_mismatch")
    if any(row.get("row_kind") != "g2_planning_call" for row in rows):
        raise AggregateBlocked("g2_row_kind_mismatch")
    _validate_row_lineage(rows, config, "g2")
    typed_rows: list[PlanningCallRow] = []
    repeat_indices_by_request: dict[tuple[object, object], list[int]] = {}
    for row in rows:
        if _exact_bool(row.get("formal_sample"), "g2_formal_sample") is not True:
            raise AggregateBlocked("g2_nonformal_row_in_distribution")
        repeat_index = row.get("repeat_index")
        if type(repeat_index) is not int or repeat_index not in range(G2_REPEATS):
            raise AggregateBlocked("g2_repeat_index_invalid")
        repeat_indices_by_request.setdefault(
            (row.get("platform"), row.get("request_id")), []
        ).append(repeat_index)
        try:
            typed_rows.append(PlanningCallRow(**_dataclass_kwargs(row, PlanningCallRow)))
        except (TypeError, ValueError) as exc:
            raise AggregateBlocked(f"g2_row_schema_invalid:{type(exc).__name__}") from exc
    if any(
        sorted(indices) != list(range(G2_REPEATS))
        for indices in repeat_indices_by_request.values()
    ):
        raise AggregateBlocked("g2_duplicate_or_missing_repeat_index")

    formal = evaluate_formal_g2(typed_rows)
    if formal["status"] == "blocked":
        raise AggregateBlocked(str(formal["blocking_reason"]))

    timing_by_platform_scale: dict[str, dict[str, Any]] = {}
    timing_by_platform_scale_outcome: dict[str, dict[str, Any]] = {}
    for platform in G2_PLATFORMS:
        for scale, expected_count in (
            ("standard", G2_STANDARD_REQUESTS * G2_REPEATS),
            ("kilometer", G2_KILOMETER_REQUESTS * G2_REPEATS),
        ):
            selected = [row for row in typed_rows if row.platform == platform and row.scale == scale]
            if len(selected) != expected_count:
                raise AggregateBlocked("g2_platform_scale_count_mismatch")
            timing_by_platform_scale[f"{platform}/{scale}"] = _timing_statistics(
                [row.elapsed_ms for row in selected]
            )
            for outcome_kind in sorted({row.outcome_kind for row in selected}):
                outcome_rows = [row for row in selected if row.outcome_kind == outcome_kind]
                timing_by_platform_scale_outcome[
                    f"{platform}/{scale}/{outcome_kind}"
                ] = _timing_statistics([row.elapsed_ms for row in outcome_rows])

    unreachable_correct = all(
        not row.provider_success and not row.route_l2_valid
        for row in typed_rows
        if row.outcome_kind == "unreachable"
    )
    correctness = formal["reachable_correctness"]["midterm_reduced_passed"] and unreachable_correct
    all_partitions = (
        *timing_by_platform_scale.values(),
        *timing_by_platform_scale_outcome.values(),
    )
    midterm = correctness and all(item["midterm_reduced_passed"] for item in all_partitions)
    final = correctness and all(item["final_threshold_reduced_passed"] for item in all_partitions)
    return {
        "status": _status_for(midterm),
        "formal_call_count": len(rows),
        "g2_all_platforms_2s_passed": midterm,
        "g2_all_platforms_1s_passed": final,
        "correctness_passed": correctness,
        "timing_by_platform_scale": timing_by_platform_scale,
        "timing_by_platform_scale_outcome": timing_by_platform_scale_outcome,
        "reachable_correctness": formal["reachable_correctness"],
    }


def recompute_g3(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    """Recalculate ten wheel episodes plus the six platform-correct interface replays."""
    if any(row.get("row_kind") not in {"g3_wheel_step", "g3_interface_replay"} for row in rows):
        raise AggregateBlocked("g3_row_kind_mismatch")
    _validate_row_lineage(rows, config, "g3")
    wheel_rows = [row for row in rows if row["row_kind"] == "g3_wheel_step"]
    interface_rows = [row for row in rows if row["row_kind"] == "g3_interface_replay"]
    if len(interface_rows) != 6:
        raise AggregateBlocked("g3_interface_replay_count_mismatch")

    typed_steps: list[tuple[ClosedLoopStepRow, Mapping[str, Any]]] = []
    for row in wheel_rows:
        try:
            typed_steps.append(
                (ClosedLoopStepRow(**_dataclass_kwargs(row, ClosedLoopStepRow)), row)
            )
        except (TypeError, ValueError) as exc:
            raise AggregateBlocked(f"g3_wheel_row_schema_invalid:{type(exc).__name__}") from exc
    if len({typed.step_id for typed, _ in typed_steps}) != len(typed_steps):
        raise AggregateBlocked("g3_duplicate_step_id")
    episode_ids = {typed.episode_id for typed, _ in typed_steps}
    if len(episode_ids) != 10:
        raise AggregateBlocked("g3_wheel_episode_count_mismatch")

    terminal_rows: list[tuple[ClosedLoopStepRow, Mapping[str, Any]]] = []
    for episode_id in sorted(episode_ids):
        episode = [(typed, raw) for typed, raw in typed_steps if typed.episode_id == episode_id]
        terminals = [
            (typed, raw)
            for typed, raw in episode
            if _exact_bool(raw.get("is_terminal"), "g3_is_terminal")
        ]
        if len(terminals) != 1:
            raise AggregateBlocked("g3_terminal_row_count_mismatch")
        terminal_rows.extend(terminals)
    split_counts = {
        split: len({typed.episode_id for typed, raw in typed_steps if raw.get("split") == split})
        for split in ("test_q24", "unseen24")
    }
    if split_counts != {"test_q24": 5, "unseen24": 5}:
        raise AggregateBlocked("g3_frozen_split_count_mismatch")

    final_coverages = [typed.coverage for typed, _ in terminal_rows]
    paired_coverages = [
        _finite_number(raw.get("paired_g1_coverage"), "g3_paired_g1_coverage")
        for _, raw in terminal_rows
    ]
    coverage_mean = statistics.mean(final_coverages)
    paired_delta_mean = statistics.mean(
        actual - reference
        for actual, reference in zip(final_coverages, paired_coverages, strict=True)
    )
    timing = _timing_statistics([typed.planner_elapsed_ms for typed, _ in typed_steps])
    wheel_integrity = True
    for _, raw in typed_steps:
        request_id = raw.get("request_id")
        wheel_integrity = wheel_integrity and (
            _exact_bool(raw.get("join_valid"), "g3_join_valid")
            and isinstance(request_id, str)
            and request_id
            and raw.get("route_request_id") == request_id
            and raw.get("feedback_request_id") == request_id
            and _exact_nonnegative_int(
                raw.get("safety_violation_count"), "g3_safety_violation_count"
            )
            == 0
            and _exact_nonnegative_int(
                raw.get("masked_action_count"), "g3_masked_action_count"
            )
            == 0
            and _exact_nonnegative_int(
                raw.get("candidate_route_mismatch_count"),
                "g3_candidate_route_mismatch_count",
            )
            == 0
        )

    typed_replays: list[tuple[InterfaceReplayRow, Mapping[str, Any]]] = []
    for row in interface_rows:
        try:
            typed_replays.append(
                (InterfaceReplayRow(**_dataclass_kwargs(row, InterfaceReplayRow)), row)
            )
        except (TypeError, ValueError) as exc:
            raise AggregateBlocked(f"g3_interface_row_schema_invalid:{type(exc).__name__}") from exc
    if len({typed.replay_id for typed, _ in typed_replays}) != 6:
        raise AggregateBlocked("g3_duplicate_interface_replay_id")
    platform_counts = {
        platform: sum(raw.get("platform") == platform for _, raw in typed_replays)
        for platform in ("legged", "hopper")
    }
    if platform_counts != {"legged": 3, "hopper": 3}:
        raise AggregateBlocked("g3_interface_platform_count_mismatch")
    if any(
        _exact_bool(raw.get("formal_input_eligible"), "g3_formal_input_eligible") is not True
        for _, raw in typed_replays
    ):
        raise AggregateBlocked("g3_formal_input_ineligible")
    interface_correct = all(
        raw.get("platform") in {"legged", "hopper"}
        and _exact_bool(raw.get("platform_match"), "g3_platform_match")
        and _exact_bool(raw.get("request_match"), "g3_request_match")
        and _exact_bool(raw.get("result_match"), "g3_result_match")
        and _exact_bool(raw.get("timing_contract_valid"), "g3_timing_contract_valid")
        for _, raw in typed_replays
    )
    interface_timing = _timing_statistics([typed.elapsed_ms for typed, _ in typed_replays])

    coverage_mid = (
        coverage_mean >= MID_COVERAGE_THRESHOLD
        and all(value >= MID_COVERAGE_THRESHOLD for value in final_coverages)
    )
    coverage_final = (
        coverage_mean >= FINAL_COVERAGE_THRESHOLD
        and all(value >= FINAL_COVERAGE_THRESHOLD for value in final_coverages)
    )
    shared = wheel_integrity and interface_correct and paired_delta_mean >= -0.01
    midterm = (
        shared
        and coverage_mid
        and timing["midterm_reduced_passed"]
        and interface_timing["midterm_reduced_passed"]
    )
    final = (
        shared
        and coverage_final
        and timing["final_threshold_reduced_passed"]
        and interface_timing["final_threshold_reduced_passed"]
    )
    return {
        "status": _status_for(midterm),
        "wheel_episode_count": len(episode_ids),
        "wheel_step_count": len(wheel_rows),
        "interface_replay_count": len(interface_rows),
        "g3_midterm_crosscheck_passed": midterm,
        "g3_final_crosscheck_passed": final,
        "wheel_coverage_mean": coverage_mean,
        "wheel_coverage_80_count": sum(
            value >= MID_COVERAGE_THRESHOLD for value in final_coverages
        ),
        "wheel_coverage_99_count": sum(
            value >= FINAL_COVERAGE_THRESHOLD for value in final_coverages
        ),
        "paired_g1_coverage_delta_mean": paired_delta_mean,
        "wheel_timing": timing,
        "interface_timing": interface_timing,
        "wheel_integrity_passed": wheel_integrity,
        "interface_correctness_passed": interface_correct,
    }


_SUMMARY_KEYS = {
    "g1": (
        "status",
        "sample_count",
        "g1_coverage_80_passed",
        "g1_coverage_99_passed",
    ),
    "g2": (
        "status",
        "formal_call_count",
        "g2_all_platforms_2s_passed",
        "g2_all_platforms_1s_passed",
    ),
    "g3": (
        "status",
        "wheel_episode_count",
        "interface_replay_count",
        "g3_midterm_crosscheck_passed",
        "g3_final_crosscheck_passed",
    ),
}


def _stored_summary_matches(
    gate_id: str,
    recomputed: Mapping[str, Any],
    stored: Mapping[str, Any],
) -> bool:
    return (
        stored.get("scale_profile") == SCALE_PROFILE
        and stored.get("gate_id") == gate_id
        and all(stored.get(key) == recomputed.get(key) for key in _SUMMARY_KEYS[gate_id])
    )


def aggregate_completed_roots(
    g1_root: str | Path,
    g2_root: str | Path,
    g3_root: str | Path,
) -> dict[str, Any]:
    """Verify all source manifests, then independently recalculate the three gates."""
    gates: dict[str, dict[str, Any]] = {}
    source_manifests: dict[str, str] = {}
    blockers: list[str] = []
    recomputers = {"g1": recompute_g1, "g2": recompute_g2, "g3": recompute_g3}
    roots = {"g1": g1_root, "g2": g2_root, "g3": g3_root}

    for gate_id in _SOURCE_GATES:
        try:
            source = _load_verified_source(roots[gate_id], gate_id)
            recomputed = recomputers[gate_id](source["rows"], source["config"])
            gates[gate_id] = recomputed
            source_manifests[gate_id] = source["manifest_sha256"]
            if not _stored_summary_matches(
                gate_id, recomputed, source["stored_summary"]
            ):
                blockers.append(f"{gate_id}_stored_summary_mismatch")
        except AggregateBlocked as exc:
            blockers.append(str(exc))
            gates.setdefault(
                gate_id,
                {
                    "status": "blocked",
                    "midterm_reduced_passed": False,
                    "final_threshold_reduced_passed": False,
                },
            )

    sample_counts = {
        "g1_total_episodes": int(gates.get("g1", {}).get("sample_count", 0)),
        "g1_test_q24": int(
            gates.get("g1", {}).get("splits", {}).get("test_q24", {}).get("sample_count", 0)
        ),
        "g1_unseen24": int(
            gates.get("g1", {}).get("splits", {}).get("unseen24", {}).get("sample_count", 0)
        ),
        "g2_formal_calls": int(gates.get("g2", {}).get("formal_call_count", 0)),
        "g3_wheel_episodes": int(gates.get("g3", {}).get("wheel_episode_count", 0)),
        "g3_interface_replays": int(
            gates.get("g3", {}).get("interface_replay_count", 0)
        ),
    }
    if blockers:
        return {
            **blocked_summary(blockers),
            "gates": gates,
            "sample_counts": sample_counts,
            "source_manifest_sha256": source_manifests,
        }

    midterm = midterm_reduced_truth_table(
        gates["g1"]["g1_coverage_80_passed"],
        gates["g2"]["g2_all_platforms_2s_passed"],
        gates["g3"]["g3_midterm_crosscheck_passed"],
    )
    final = final_threshold_reduced_truth_table(
        gates["g1"]["g1_coverage_99_passed"],
        gates["g2"]["g2_all_platforms_1s_passed"],
        gates["g3"]["g3_final_crosscheck_passed"],
    )
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "scale_profile": SCALE_PROFILE,
        "status": _status_for(midterm),
        "formal_evidence_eligible": True,
        "midterm_reduced_gate_passed": midterm,
        "final_threshold_reduced_gate_passed": final,
        "blockers": [],
        "gates": gates,
        "sample_counts": sample_counts,
        "source_manifest_sha256": source_manifests,
    }


def render_report(summary: Mapping[str, Any]) -> str:
    """Render the bounded one-page reduced-scale conclusion table."""
    counts = summary.get("sample_counts", {})
    gates = summary.get("gates", {})
    status = summary.get("status")
    if status == "blocked":
        conclusion = "证据未就绪；以下阻塞项关闭前不得描述为指标失败或通过。"
    elif status == "passed":
        conclusion = "缩减规模中期双门槛通过。"
    else:
        conclusion = "完整证据已复算，指标未通过。"
    blockers = summary.get("blockers", [])
    blocker_text = "、".join(str(item) for item in blockers) if blockers else "无"
    lines = [
        "# 缩减规模中期实验（G1 24 场景/split）独立复算报告",
        "",
        f"- 规模合同：`{SCALE_PROFILE}`",
        f"- 结论：{conclusion}",
        f"- 阻塞项：{blocker_text}",
        "",
        "| 证据组 | 精确样本数 |",
        "|---|---:|",
        f"| G1 Test-Q24 | {counts.get('g1_test_q24', 0)} |",
        f"| G1 Unseen-24 | {counts.get('g1_unseen24', 0)} |",
        f"| G2 正式调用 | {counts.get('g2_formal_calls', 0)} |",
        f"| G3 轮式闭环 | {counts.get('g3_wheel_episodes', 0)} |",
        f"| G3 接口回放 | {counts.get('g3_interface_replays', 0)} |",
    ]
    g1_splits = gates.get("g1", {}).get("splits", {})
    if g1_splits:
        lines.extend(("", "| G1 split | mean | >=80% | >=99% |", "|---|---:|---:|---:|"))
        for split, label in (("test_q24", "G1 Test-Q24"), ("unseen24", "G1 Unseen-24")):
            row = g1_splits.get(split, {})
            lines.append(
                f"| {label} | {float(row.get('mean', 0.0)):.6f} | "
                f"{row.get('coverage_80_count', 0)}/24 | "
                f"{row.get('coverage_99_count', 0)}/24 |"
            )
    g2_groups = gates.get("g2", {}).get("timing_by_platform_scale", {})
    if g2_groups:
        lines.extend(
            (
                "",
                "| G2 platform/scale | n | mean ms | P95 ms | P99 ms | 2 s | 1 s |",
                "|---|---:|---:|---:|---:|---|---|",
            )
        )
        for platform in G2_PLATFORMS:
            for scale in ("standard", "kilometer"):
                key = f"{platform}/{scale}"
                row = g2_groups.get(key, {})
                lines.append(
                    f"| {key} | {row.get('sample_count', 0)} | "
                    f"{float(row.get('mean_ms', 0.0)):.3f} | "
                    f"{float(row.get('p95_ms', 0.0)):.3f} | "
                    f"{float(row.get('p99_ms', 0.0)):.3f} | "
                    f"{bool(row.get('midterm_reduced_passed'))} | "
                    f"{bool(row.get('final_threshold_reduced_passed'))} |"
                )
    g3 = gates.get("g3", {})
    if g3:
        lines.extend(
            (
                "",
                "| G3 cross-check | value |",
                "|---|---:|",
                f"| G3 轮式覆盖均值 | {float(g3.get('wheel_coverage_mean', 0.0)):.6f} |",
                f"| G3 轮式规划 P95 ms | {float(g3.get('wheel_timing', {}).get('p95_ms', 0.0)):.3f} |",
                f"| G3 G1 配对覆盖差均值 | {float(g3.get('paired_g1_coverage_delta_mean', 0.0)):.6f} |",
                f"| G3 接口对应正确 | {bool(g3.get('interface_correctness_passed'))} |",
            )
        )
    lines.extend(
        (
            "",
            "| 路由 | 结果 |",
            "|---|---|",
            f"| 中期缩减门槛（80% / 2 s） | {bool(summary.get('midterm_reduced_gate_passed'))} |",
            f"| 最终阈值缩减门槛（99% / 1 s） | {bool(summary.get('final_threshold_reduced_gate_passed'))} |",
            "",
            "本报告仅代表上述缩减规模样本，不等同于项目全规模完成验收。",
            "",
        )
    )
    return "\n".join(lines)


def _environment_probe() -> dict[str, Any]:
    memory_bytes = 1
    try:
        import psutil

        memory_bytes = int(psutil.virtual_memory().total)
    except (ImportError, OSError, ValueError):
        pass
    gpu_model = gpu_driver = gpu_cuda = "unavailable"
    try:
        probe = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            gpu_model, gpu_driver = (
                item.strip() for item in probe.stdout.splitlines()[0].split(",", 1)
            )
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    try:
        import torch

        gpu_cuda = str(torch.version.cuda or "unavailable")
    except ImportError:
        pass
    thread_names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "windows_version": platform.platform() or "unavailable",
        "cpu_model": platform.processor() or platform.machine() or "unavailable",
        "cpu_logical_count": int(os.cpu_count() or 1),
        "memory_bytes": max(1, memory_bytes),
        "gpu": {"model": gpu_model, "driver": gpu_driver, "cuda": gpu_cuda},
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "frozen_dependencies": [
            f"python=={platform.python_version()}",
            f"pytest-environment={os.environ.get('PYTEST_VERSION', 'not-running')}",
        ],
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset"),
        "thread_variables": {name: os.environ.get(name, "unset") for name in thread_names},
        "worker_start_method": "spawn",
        "power_mode": os.environ.get("XUNCE_POWER_MODE", "not-recorded"),
    }


def _git_output(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=cwd or Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _load_aggregate_config(path: str | Path) -> dict[str, Any]:
    config = artifact_io.read_json(path)
    if config.get("schema_version") != AGGREGATE_CONFIG_SCHEMA_VERSION:
        raise ValueError("aggregate config schema_version is unsupported")
    if config.get("scale_profile") != SCALE_PROFILE:
        raise ValueError("aggregate config scale_profile drift")
    output_root = config.get("output_root")
    if not isinstance(output_root, str) or not Path(output_root).is_absolute():
        raise ValueError("aggregate output_root must be absolute")
    if config.get("required_phase_ids") != ["p01"]:
        raise ValueError("aggregate required_phase_ids must be ['p01']")
    expected_actual = {
        "g1_episodes_per_split": 24,
        "g1_split_count": 2,
        "g2_formal_calls": 645,
        "g3_interface_replays": 6,
        "g3_wheel_episodes": 10,
    }
    if config.get("actual_sample_counts") != expected_actual:
        raise ValueError("aggregate actual_sample_counts drift")
    expected_reference = {
        "g1_episodes_per_split": 64,
        "g1_split_count": 2,
        "g2_formal_calls": 1950,
        "g3_interface_replays": 18,
        "g3_wheel_episodes": 30,
    }
    if config.get("reference_sample_counts") != expected_reference:
        raise ValueError("aggregate reference_sample_counts drift")
    return config


def run_aggregate(
    *,
    config_path: str | Path,
    g1_root: str | Path,
    g2_root: str | Path,
    g3_root: str | Path,
    run_id: str,
) -> Path:
    """Run one non-overwriting aggregate attempt and write canonical artifacts."""
    if not isinstance(run_id, str) or not run_id or any(char not in "-_." and not char.isalnum() for char in run_id):
        raise ValueError("run_id must be a non-empty filesystem-safe identifier")
    for name, root in (("g1", g1_root), ("g2", g2_root), ("g3", g3_root)):
        if not Path(root).is_absolute():
            raise ValueError(f"{name}_root must be absolute")
    config = _load_aggregate_config(config_path)
    output_root = Path(config["output_root"]) / run_id
    summary = aggregate_completed_roots(g1_root, g2_root, g3_root)
    effective_config = {
        **config,
        "run_id": run_id,
        "g1_root": str(Path(g1_root)),
        "g2_root": str(Path(g2_root)),
        "g3_root": str(Path(g3_root)),
    }
    store = MidDualRunStore.create_new(output_root, effective_config)
    source_rows = [
        {
            "row_kind": "independent_gate_recalculation",
            "gate_id": gate_id,
            "recalculated": summary.get("gates", {}).get(gate_id, {"status": "blocked"}),
        }
        for gate_id in _SOURCE_GATES
    ]
    if summary["status"] != "blocked":
        attempt = store.write_phase_attempt(
            "p01", source_rows, {"kind": "independent_aggregate"}
        )
        store.accept_phase(
            "p01", attempt, store.phase_attempt_row_sha256("p01", attempt)
        )
    repository_root = Path(_git_output("rev-parse", "--show-toplevel") or Path.cwd())
    required_sources = [
        Path(config_path),
        Path(__file__),
        repository_root / "scripts" / "xunce_mid_dual_contracts.py",
        repository_root / "scripts" / "xunce_mid_dual_artifacts.py",
        repository_root / "scripts" / "xunce_artifact_io.py",
        repository_root / "scripts" / "xunce_artifact_paths.py",
    ]
    store.capture_lineage(
        required_sources,
        _git_output("rev-parse", "HEAD"),
        _git_output("-C", "path-planner", "rev-parse", "HEAD"),
    )
    store.capture_environment(_environment_probe)
    routing = {
        "schema_version": "xunce-mid-dual-aggregate-routing/v1",
        "scale_profile": SCALE_PROFILE,
        "status": summary["status"],
        "midterm_reduced_gate_passed": summary["midterm_reduced_gate_passed"],
        "final_threshold_reduced_gate_passed": summary[
            "final_threshold_reduced_gate_passed"
        ],
        "blockers": summary["blockers"],
    }
    store.finalize(
        summary,
        routing,
        render_report(summary),
        {
            "source_roots": {
                "g1_root": str(Path(g1_root)),
                "g2_root": str(Path(g2_root)),
                "g3_root": str(Path(g3_root)),
                "source_manifest_sha256": summary.get("source_manifest_sha256", {}),
            }
        },
    )
    MidDualRunStore.verify_manifest(output_root)
    return output_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently recalculate the reduced midterm dual-gate."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--g1-root", required=True)
    parser.add_argument("--g2-root", required=True)
    parser.add_argument("--g3-root", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = run_aggregate(
        config_path=args.config,
        g1_root=args.g1_root,
        g2_root=args.g2_root,
        g3_root=args.g3_root,
        run_id=args.run_id,
    )
    print(
        json.dumps(
            {"output_root": str(output_root), "status": "complete"},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
