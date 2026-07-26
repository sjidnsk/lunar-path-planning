"""Behavioral contracts for the reduced midterm dual-gate evidence chain."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from xunce_mid_dual_contracts import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
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
    SCALE_PROFILE,
    evaluate_g1_split,
    evaluate_g2_platform,
    episode_bootstrap_ci,
    nearest_rank,
    route_gate,
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
    assert evaluate_g1_split(jobs, [0.80] * 24)["status"] == "passed"

    duplicate_jobs = (*jobs[:-1], jobs[-2])
    assert evaluate_g1_split(duplicate_jobs, [0.80] * 24)["status"] == "blocked"
    assert evaluate_g1_split(jobs[:-1], [0.80] * 23)["status"] == "blocked"


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
    coverage = evaluate_g1_split(tuple(f"scene-{index}" for index in range(24)), [0.80] * 24)
    assert coverage["midterm_reduced_passed"] is True
    assert coverage["final_threshold_reduced_passed"] is False

    final_coverage = evaluate_g1_split(tuple(f"final-{index}" for index in range(24)), [0.99] * 24)
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
    assert evaluate_g1_split(jobs, [0.80] * 23 + [math.nan])["status"] == "blocked"
    assert evaluate_g2_platform(
        platform="wheel",
        scale="standard",
        outcome_kind="success",
        elapsed_ms=[1.0, math.inf],
    )["status"] == "blocked"


def test_reduced_pass_fields_have_no_unqualified_pass_alias() -> None:
    """Catch a bare pass field that loses the reduced-scale qualification."""
    routing = route_gate(midterm=True, final=False)
    assert routing == {
        "status": "passed",
        "midterm_reduced_passed": True,
        "final_threshold_reduced_passed": False,
    }
    assert "passed" not in routing
