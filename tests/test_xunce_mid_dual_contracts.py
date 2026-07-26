"""Behavioral contracts for the reduced midterm dual-gate evidence chain."""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from xunce_mid_dual_contracts import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    CoverageEpisodeRow,
    FINAL_COVERAGE_THRESHOLD,
    FINAL_TIME_MS,
    G1_EPISODES_PER_FORMAL_SPLIT,
    G1_LANE_SIZES,
    G2_FORMAL_CALLS,
    G2_KILOMETER_REQUESTS,
    G2_PLATFORMS,
    G2_REPEATS,
    G2_STANDARD_REQUESTS,
    MID_COVERAGE_THRESHOLD,
    MID_TIME_MS,
    PlanningCallRow,
    SCALE_PROFILE,
    evaluate_formal_g2,
    evaluate_g1_split,
    evaluate_g2_platform,
    episode_bootstrap_ci,
    nearest_rank,
    reachable_request_success_rate,
    route_gate,
)


_HASH = "a" * 64


def _formal_g2_rows() -> list[PlanningCallRow]:
    rows: list[PlanningCallRow] = []
    for platform in G2_PLATFORMS:
        request_index = 0
        for scale, count in (("standard", 33), ("kilometer", 10)):
            for _ in range(count):
                outcome_kind = "reachable" if request_index < 38 else "unreachable"
                request_id = f"{platform}-request-{request_index:02d}"
                for repeat in range(G2_REPEATS):
                    rows.append(
                        PlanningCallRow(
                            schema_version="planning-call-row/v1",
                            scale_profile=SCALE_PROFILE,
                            run_id="run-1",
                            episode_id="episode-1",
                            request_id=request_id,
                            call_id=f"{platform}-call-{request_index:02d}-{repeat}",
                            platform=platform,
                            scale=scale,
                            outcome_kind=outcome_kind,
                            source_sha256=_HASH,
                            config_sha256=_HASH,
                            request_sha256=_HASH,
                            provider_sha256=_HASH,
                            oracle_sha256=_HASH,
                            elapsed_ms=10.0,
                            input_validation_ms=1.0,
                            platform_instantiation_ms=1.0,
                            search_ms=6.0,
                            complete_route_validation_ms=1.0,
                            result_assembly_ms=1.0,
                            provider_success=True,
                            route_l2_valid=True,
                            semantic_digest=f"semantic-{platform}-{request_index:02d}",
                        )
                    )
                request_index += 1
    return rows


def _coverage_row() -> CoverageEpisodeRow:
    return CoverageEpisodeRow(
        schema_version="coverage-episode-row/v1",
        scale_profile=SCALE_PROFILE,
        run_id="run-1",
        episode_id="episode-1",
        scenario_id="scenario-1",
        source_sha256=_HASH,
        config_sha256=_HASH,
        checkpoint_sha256=_HASH,
        denominator_sha256=_HASH,
        coverage=0.80,
        elapsed_ms=10.0,
        safety_violation_count=0,
        masked_action_count=0,
    )


def test_scale_profile_freezes_w8x3_update80_contract() -> None:
    """Catch accidental widening, reshaping, or checkpoint-update drift."""
    assert SCALE_PROFILE == "midterm_reduced_w8x3_update80/v1"
    assert G1_EPISODES_PER_FORMAL_SPLIT == 24
    assert G1_LANE_SIZES == (3, 3, 3, 3, 3, 3, 3, 3)
    assert sum(G1_LANE_SIZES) == G1_EPISODES_PER_FORMAL_SPLIT


def test_g1_requires_exactly_24_unique_jobs_and_eight_lanes_of_three() -> None:
    """Catch a G1 denominator with duplicate/missing jobs or unequal lanes."""
    jobs = tuple(f"scene-{index:02d}" for index in range(24))
    balanced_lanes = tuple(f"lane-{index // 3}" for index in range(24))
    assert evaluate_g1_split(jobs, [0.80] * 24, lane_ids=balanced_lanes)["status"] == "passed"

    duplicate_jobs = (*jobs[:-1], jobs[-2])
    assert evaluate_g1_split(duplicate_jobs, [0.80] * 24, lane_ids=balanced_lanes)["status"] == "blocked"
    assert evaluate_g1_split(jobs[:-1], [0.80] * 23, lane_ids=balanced_lanes[:-1])["status"] == "blocked"

    unbalanced_lanes = ("lane-0",) * 24
    assert evaluate_g1_split(jobs, [0.80] * 24, lane_ids=unbalanced_lanes)["status"] == "blocked"


def test_g2_requires_exactly_645_formal_calls() -> None:
    """Catch changes that silently reduce the formal G2 sample matrix."""
    assert G2_PLATFORMS == ("wheel", "legged", "hopper")
    assert G2_STANDARD_REQUESTS == 33
    assert G2_KILOMETER_REQUESTS == 10
    assert G2_REPEATS == 5
    assert G2_FORMAL_CALLS == 645
    assert len(G2_PLATFORMS) * (G2_STANDARD_REQUESTS + G2_KILOMETER_REQUESTS) * G2_REPEATS == G2_FORMAL_CALLS


def test_threshold_boundaries_are_inclusive_at_080_099_1000_2000() -> None:
    """Catch strict comparisons that reject values exactly at a gate boundary."""
    lanes = tuple(f"lane-{index // 3}" for index in range(24))
    coverage = evaluate_g1_split(tuple(f"scene-{index}" for index in range(24)), [0.80] * 24, lane_ids=lanes)
    assert coverage["midterm_reduced_passed"] is True
    assert coverage["final_threshold_reduced_passed"] is False

    final_coverage = evaluate_g1_split(tuple(f"final-{index}" for index in range(24)), [0.99] * 24, lane_ids=lanes)
    assert final_coverage["final_threshold_reduced_passed"] is True

    mid_timing = evaluate_g2_platform(
        platform="wheel",
        scale="standard",
        outcome_kind="success",
        elapsed_ms=[2000.0] * 5,
    )
    assert mid_timing["midterm_reduced_passed"] is True
    final_timing = evaluate_g2_platform(
        platform="wheel",
        scale="standard",
        outcome_kind="success",
        elapsed_ms=[1000.0] * 5,
    )
    assert final_timing["final_threshold_reduced_passed"] is True
    assert MID_COVERAGE_THRESHOLD == 0.80
    assert FINAL_COVERAGE_THRESHOLD == 0.99
    assert MID_TIME_MS == 2000.0
    assert FINAL_TIME_MS == 1000.0


def test_nearest_rank_uses_ceil_qn_one_based_index() -> None:
    """Catch interpolation or zero-based percentile indexing in timing reports."""
    assert nearest_rank([10.0, 20.0, 30.0, 40.0], 0.95) == 40.0
    assert nearest_rank([10.0, 20.0, 30.0, 40.0], 0.50) == 20.0
    assert nearest_rank([30.0, 10.0, 20.0], 0.01) == 10.0


def test_episode_bootstrap_is_fixed_seed_and_uses_2000_resamples() -> None:
    """Catch nondeterministic or wrongly sized episode-level bootstrap output."""
    values = [0.10, 0.20, 0.30, 0.40]
    first = episode_bootstrap_ci(values)
    second = episode_bootstrap_ci(values)
    assert BOOTSTRAP_RESAMPLES == 2000
    assert first == second
    assert first["resamples"] == 2000
    assert first["unit"] == "episode"
    assert first["lower"] <= 0.25 <= first["upper"]


def test_nonfinite_values_block_instead_of_being_filtered() -> None:
    """Catch NaN/inf filtering that would understate failed formal samples."""
    jobs = tuple(f"scene-{index}" for index in range(24))
    lanes = tuple(f"lane-{index // 3}" for index in range(24))
    assert evaluate_g1_split(jobs, [0.80] * 23 + [math.nan], lane_ids=lanes)["status"] == "blocked"
    assert evaluate_g2_platform(
        platform="wheel",
        scale="standard",
        outcome_kind="success",
        elapsed_ms=[1.0, math.inf],
    )["status"] == "blocked"


def test_formal_g2_blocks_missing_extra_duplicate_and_incomplete_platform_matrix() -> None:
    """Catch any 645-call matrix that is not exactly three 43-request platforms."""
    rows = _formal_g2_rows()
    assert evaluate_formal_g2(rows)["status"] == "passed"
    assert evaluate_formal_g2(rows[:-1])["status"] == "blocked"

    extra = [*rows, replace(rows[0], call_id="wheel-extra-call")]
    assert evaluate_formal_g2(extra)["status"] == "blocked"

    duplicate_call_id = [*rows]
    duplicate_call_id[-1] = replace(duplicate_call_id[-1], call_id=duplicate_call_id[0].call_id)
    assert evaluate_formal_g2(duplicate_call_id)["status"] == "blocked"

    incomplete_platform = [*rows]
    incomplete_platform[-1] = replace(
        incomplete_platform[-1],
        platform="wheel",
        request_id="wheel-request-extra",
        call_id="wheel-call-extra",
        semantic_digest="semantic-wheel-extra",
    )
    assert evaluate_formal_g2(incomplete_platform)["status"] == "blocked"


def test_formal_g2_gates_each_platform_scale_outcome_partition() -> None:
    """Catch a slow outcome partition hidden by a fast platform-scale aggregate p95."""
    rows = _formal_g2_rows()
    slow_indices = [
        index
        for index, row in enumerate(rows)
        if row.platform == "wheel" and row.scale == "kilometer" and row.outcome_kind == "unreachable"
    ][:2]
    for index in slow_indices:
        rows[index] = replace(rows[index], elapsed_ms=2001.0, search_ms=1997.0)

    routed = evaluate_formal_g2(rows)
    assert routed["midterm_reduced_passed"] is False
    assert routed["final_threshold_reduced_passed"] is False
    partition = routed["timing_by_platform_scale_outcome"][("wheel", "kilometer", "unreachable")]
    assert partition["p95_ms"] == 2001.0


@pytest.mark.parametrize("hash_field", ["provider_sha256", "oracle_sha256"])
def test_formal_g2_blocks_repeat_provider_or_oracle_provenance_drift(hash_field: str) -> None:
    """Catch a 645-call matrix whose repeated request changes executor provenance."""
    rows = _formal_g2_rows()
    rows[1] = replace(rows[1], **{hash_field: "b" * 64})
    routed = evaluate_formal_g2(rows)
    assert routed["status"] == "blocked"
    assert routed["blocking_reason"] == "g2_request_repeat_or_provenance_mismatch"


def test_reachable_consensus_requires_per_platform_38_unique_repeated_requests() -> None:
    """Catch undersized, duplicate, cross-platform, or 37/38 reachable evidence."""
    rows = _formal_g2_rows()
    assert reachable_request_success_rate(rows)["status"] == "passed"
    assert reachable_request_success_rate(rows[:5])["status"] == "blocked"

    duplicate_call_id = [*rows]
    duplicate_call_id[1] = replace(duplicate_call_id[1], call_id=duplicate_call_id[0].call_id)
    assert reachable_request_success_rate(duplicate_call_id)["status"] == "blocked"

    cross_platform_request_id = [*rows]
    cross_platform_request_id[215] = replace(cross_platform_request_id[215], request_id=rows[0].request_id)
    assert reachable_request_success_rate(cross_platform_request_id)["status"] == "blocked"

    only_37_reachable = [*rows]
    for index, row in enumerate(only_37_reachable):
        if row.platform == "hopper" and row.request_id == "hopper-request-37":
            only_37_reachable[index] = replace(row, outcome_kind="unreachable")
    assert reachable_request_success_rate(only_37_reachable)["status"] == "blocked"


@pytest.mark.parametrize("malformed", [True, "1.0", None])
def test_statistics_boundaries_block_malformed_values_without_coercion(malformed: object) -> None:
    """Catch bool/string/None coercion or leaked conversion errors at a gate boundary."""
    lanes = tuple(f"lane-{index // 3}" for index in range(24))
    assert evaluate_g1_split(tuple(f"scene-{index}" for index in range(24)), [0.80] * 23 + [malformed], lane_ids=lanes)["status"] == "blocked"
    assert evaluate_g2_platform(
        platform="wheel",
        scale="standard",
        outcome_kind="success",
        elapsed_ms=[10.0, malformed],
    )["status"] == "blocked"


@pytest.mark.parametrize("malformed", [0, "", []])
def test_g1_blocks_malformed_job_ids_before_set_operations(malformed: object) -> None:
    """Catch non-string/empty/unhashable formal G1 identifiers leaking or raising."""
    jobs: list[object] = [f"scene-{index}" for index in range(24)]
    jobs[0] = malformed
    lanes = tuple(f"lane-{index // 3}" for index in range(24))
    assert evaluate_g1_split(jobs, [0.80] * 24, lane_ids=lanes)["status"] == "blocked"


def test_episode_bootstrap_rejects_seed_override() -> None:
    """Catch callers changing the frozen bootstrap random seed."""
    with pytest.raises(TypeError):
        episode_bootstrap_ci([0.10, 0.20], seed=1)


@pytest.mark.parametrize("field_name", ["provider_success", "route_l2_valid"])
@pytest.mark.parametrize("malformed", ["false", 1, None])
def test_planning_call_row_rejects_non_bool_success_and_validity_evidence(field_name: str, malformed: object) -> None:
    """Catch truthy strings/integers leaking through G2 success and validity evidence."""
    with pytest.raises(ValueError):
        replace(_formal_g2_rows()[0], **{field_name: malformed})


@pytest.mark.parametrize("field_name", ["safety_violation_count", "masked_action_count"])
@pytest.mark.parametrize("malformed", [True, 1.5, None, "0"])
def test_coverage_episode_row_rejects_non_integer_count_evidence(field_name: str, malformed: object) -> None:
    """Catch bool, fractional, null, and string evidence in coverage count fields."""
    with pytest.raises(ValueError):
        replace(_coverage_row(), **{field_name: malformed})


def test_reduced_pass_fields_have_no_unqualified_pass_alias() -> None:
    """Catch a bare pass field that loses the reduced-scale qualification."""
    routing = route_gate(midterm=True, final=False)
    assert routing == {
        "status": "passed",
        "midterm_reduced_passed": True,
        "final_threshold_reduced_passed": False,
    }
    assert "passed" not in routing
