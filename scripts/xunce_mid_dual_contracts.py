"""Frozen common contracts for the reduced midterm dual-gate experiment.

This module is intentionally free of runner and artifact I/O.  It owns the
small, replayable facts that every later stage must use when it evaluates raw
rows and routes a formal gate result.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
import random
from statistics import mean, median, stdev
from typing import Iterable, Mapping, Sequence


SCALE_PROFILE = "midterm_reduced_w8x3_update80/v1"
G1_EPISODES_PER_FORMAL_SPLIT = 24
G1_LANE_SIZES = (3, 3, 3, 3, 3, 3, 3, 3)
G2_PLATFORMS = ("wheel", "legged", "hopper")
G2_STANDARD_REQUESTS = 33
G2_KILOMETER_REQUESTS = 10
G2_REPEATS = 5
G2_FORMAL_CALLS = 645
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260726
MID_COVERAGE_THRESHOLD = 0.80
FINAL_COVERAGE_THRESHOLD = 0.99
MID_TIME_MS = 2000.0
FINAL_TIME_MS = 1000.0

_SHA256_LENGTH = 64


def _require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_sha256(value: str, field_name: str) -> None:
    _require_nonempty(value, field_name)
    if len(value) != _SHA256_LENGTH or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")


def _require_finite(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite")


def _require_common_row_contract(row: object) -> None:
    _require_nonempty(getattr(row, "schema_version"), "schema_version")
    if getattr(row, "scale_profile") != SCALE_PROFILE:
        raise ValueError("scale_profile does not match the frozen reduced-scale contract")
    for field_name in ("run_id", "episode_id"):
        _require_nonempty(getattr(row, field_name), field_name)
    for field_name in ("source_sha256", "config_sha256"):
        _require_sha256(getattr(row, field_name), field_name)
    for item in fields(row):
        value = getattr(row, item.name)
        if isinstance(value, float):
            _require_finite(value, item.name)


@dataclass(frozen=True, slots=True)
class CoverageEpisodeRow:
    """One G1 episode result, including its immutable evaluation provenance."""

    schema_version: str
    scale_profile: str
    run_id: str
    episode_id: str
    scenario_id: str
    source_sha256: str
    config_sha256: str
    checkpoint_sha256: str
    denominator_sha256: str
    coverage: float
    elapsed_ms: float
    safety_violation_count: int
    masked_action_count: int

    def __post_init__(self) -> None:
        _require_common_row_contract(self)
        for field_name in ("scenario_id",):
            _require_nonempty(getattr(self, field_name), field_name)
        for field_name in ("checkpoint_sha256", "denominator_sha256"):
            _require_sha256(getattr(self, field_name), field_name)
        for field_name in ("coverage", "elapsed_ms"):
            _require_finite(getattr(self, field_name), field_name)
        if self.safety_violation_count < 0 or self.masked_action_count < 0:
            raise ValueError("count fields must be non-negative")


@dataclass(frozen=True, slots=True)
class PlanningCallRow:
    """One G2 call result; repeated calls share a request ID but not a call ID."""

    schema_version: str
    scale_profile: str
    run_id: str
    episode_id: str
    request_id: str
    call_id: str
    platform: str
    scale: str
    outcome_kind: str
    source_sha256: str
    config_sha256: str
    request_sha256: str
    provider_sha256: str
    oracle_sha256: str
    elapsed_ms: float
    input_validation_ms: float
    platform_instantiation_ms: float
    search_ms: float
    complete_route_validation_ms: float
    result_assembly_ms: float
    provider_success: bool
    route_l2_valid: bool
    semantic_digest: str

    def __post_init__(self) -> None:
        _require_common_row_contract(self)
        for field_name in ("request_id", "call_id", "platform", "scale", "outcome_kind", "semantic_digest"):
            _require_nonempty(getattr(self, field_name), field_name)
        for field_name in ("request_sha256", "provider_sha256", "oracle_sha256"):
            _require_sha256(getattr(self, field_name), field_name)
        components = (
            self.input_validation_ms,
            self.platform_instantiation_ms,
            self.search_ms,
            self.complete_route_validation_ms,
            self.result_assembly_ms,
        )
        for field_name, value in zip(
            (
                "elapsed_ms",
                "input_validation_ms",
                "platform_instantiation_ms",
                "search_ms",
                "complete_route_validation_ms",
                "result_assembly_ms",
            ),
            (self.elapsed_ms, *components),
            strict=True,
        ):
            _require_finite(value, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if not math.isclose(self.elapsed_ms, sum(components), abs_tol=0.001):
            raise ValueError("elapsed_ms must equal the five timing components")


@dataclass(frozen=True, slots=True)
class ClosedLoopStepRow:
    """One G3 closed-loop decision/planning step with stable provenance IDs."""

    schema_version: str
    scale_profile: str
    run_id: str
    episode_id: str
    step_id: str
    candidate_id: str
    source_sha256: str
    config_sha256: str
    planner_sha256: str
    coverage: float
    planner_elapsed_ms: float

    def __post_init__(self) -> None:
        _require_common_row_contract(self)
        for field_name in ("step_id", "candidate_id"):
            _require_nonempty(getattr(self, field_name), field_name)
        _require_sha256(self.planner_sha256, "planner_sha256")
        _require_finite(self.coverage, "coverage")
        _require_finite(self.planner_elapsed_ms, "planner_elapsed_ms")
        if self.planner_elapsed_ms < 0:
            raise ValueError("planner_elapsed_ms must be non-negative")


@dataclass(frozen=True, slots=True)
class InterfaceReplayRow:
    """A deterministic interface replay row used for G3 cross-check evidence."""

    schema_version: str
    scale_profile: str
    run_id: str
    episode_id: str
    replay_id: str
    request_id: str
    source_sha256: str
    config_sha256: str
    request_sha256: str
    result_sha256: str
    elapsed_ms: float

    def __post_init__(self) -> None:
        _require_common_row_contract(self)
        for field_name in ("replay_id", "request_id"):
            _require_nonempty(getattr(self, field_name), field_name)
        for field_name in ("request_sha256", "result_sha256"):
            _require_sha256(getattr(self, field_name), field_name)
        _require_finite(self.elapsed_ms, "elapsed_ms")
        if self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be non-negative")


def _finite_values(values: Iterable[float]) -> tuple[float, ...] | None:
    materialized = tuple(float(value) for value in values)
    if not materialized or not all(math.isfinite(value) for value in materialized):
        return None
    return materialized


def nearest_rank(values: Sequence[float], q: float) -> float:
    """Return the one-based nearest-rank percentile: ``ceil(q*n)``."""
    finite = _finite_values(values)
    if finite is None:
        raise ValueError("values must be non-empty and finite")
    if not math.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("q must be finite and within [0, 1]")
    index = max(1, math.ceil(q * len(finite))) - 1
    return sorted(finite)[index]


def episode_bootstrap_ci(values: Sequence[float], *, seed: int = BOOTSTRAP_SEED) -> dict[str, float | int | str]:
    """Return replayable 95% bootstrap CI using whole episodes as units."""
    finite = _finite_values(values)
    if finite is None:
        raise ValueError("bootstrap values must be non-empty and finite")
    generator = random.Random(seed)
    sampled_means = sorted(
        mean(generator.choice(finite) for _ in range(len(finite))) for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return {
        "unit": "episode",
        "seed": seed,
        "resamples": BOOTSTRAP_RESAMPLES,
        "confidence_level": 0.95,
        "lower": nearest_rank(sampled_means, 0.025),
        "upper": nearest_rank(sampled_means, 0.975),
    }


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    finite = _finite_values(values)
    if finite is None:
        raise ValueError("statistics values must be non-empty and finite")
    return {
        "sample_count": len(finite),
        "mean": mean(finite),
        "median": median(finite),
        "sample_stddev": stdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
    }


def route_gate(*, midterm: bool, final: bool, blocked: bool = False) -> dict[str, str | bool]:
    """Route one qualification gate without emitting an unqualified pass alias."""
    if blocked:
        return {
            "status": "blocked",
            "midterm_reduced_passed": False,
            "final_threshold_reduced_passed": False,
        }
    return {
        "status": "passed" if midterm else "failed",
        "midterm_reduced_passed": bool(midterm),
        "final_threshold_reduced_passed": bool(final),
    }


def g1_split_statistics(job_ids: Sequence[str], coverage_values: Sequence[float]) -> dict[str, object]:
    """Compute one split's G1 thresholds; malformed formal evidence blocks closed."""
    if len(job_ids) != G1_EPISODES_PER_FORMAL_SPLIT or len(set(job_ids)) != G1_EPISODES_PER_FORMAL_SPLIT:
        return {**route_gate(midterm=False, final=False, blocked=True), "blocking_reason": "g1_jobs_incomplete_or_duplicate"}
    values = _finite_values(coverage_values)
    if values is None or len(values) != G1_EPISODES_PER_FORMAL_SPLIT:
        return {**route_gate(midterm=False, final=False, blocked=True), "blocking_reason": "g1_nonfinite_or_incomplete_coverage"}
    statistics = _summary(values)
    midterm_count = sum(value >= MID_COVERAGE_THRESHOLD for value in values)
    final_count = sum(value >= FINAL_COVERAGE_THRESHOLD for value in values)
    midterm = statistics["mean"] >= MID_COVERAGE_THRESHOLD and midterm_count >= 23
    final = statistics["mean"] >= FINAL_COVERAGE_THRESHOLD and final_count >= 23
    return {
        **route_gate(midterm=midterm, final=final),
        **statistics,
        "coverage_80_count": midterm_count,
        "coverage_99_count": final_count,
        "bootstrap_ci": episode_bootstrap_ci(values),
    }


evaluate_g1_split = g1_split_statistics


def evaluate_g2_platform(*, platform: str, scale: str, outcome_kind: str, elapsed_ms: Sequence[float]) -> dict[str, object]:
    """Summarize one G2 platform/scale/outcome partition and its time gates."""
    if platform not in G2_PLATFORMS or not scale or not outcome_kind:
        return {**route_gate(midterm=False, final=False, blocked=True), "blocking_reason": "invalid_g2_partition"}
    values = _finite_values(elapsed_ms)
    if values is None:
        return {**route_gate(midterm=False, final=False, blocked=True), "blocking_reason": "g2_nonfinite_or_empty_time"}
    statistics = _summary(values)
    p95 = nearest_rank(values, 0.95)
    midterm = p95 <= MID_TIME_MS
    final = p95 <= FINAL_TIME_MS
    return {
        **route_gate(midterm=midterm, final=final),
        **statistics,
        "platform": platform,
        "scale": scale,
        "outcome_kind": outcome_kind,
        "p95_ms": p95,
        "bootstrap_ci": episode_bootstrap_ci(values),
    }


def g2_statistics(rows: Sequence[PlanningCallRow]) -> dict[tuple[str, str, str], dict[str, object]]:
    """Keep G2 timing statistics partitioned by platform, scale, and outcome."""
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row.platform, row.scale, row.outcome_kind), []).append(row.elapsed_ms)
    return {
        key: evaluate_g2_platform(platform=key[0], scale=key[1], outcome_kind=key[2], elapsed_ms=values)
        for key, values in grouped.items()
    }


def unique_request_semantic_consensus(rows: Sequence[PlanningCallRow]) -> dict[str, bool]:
    """Apply the G2 five-repeat, all-success, identical-digest correctness rule."""
    grouped: dict[str, list[PlanningCallRow]] = {}
    for row in rows:
        grouped.setdefault(row.request_id, []).append(row)
    return {
        request_id: len(request_rows) == G2_REPEATS
        and all(row.provider_success and row.route_l2_valid for row in request_rows)
        and len({row.semantic_digest for row in request_rows}) == 1
        for request_id, request_rows in grouped.items()
    }


def reachable_request_success_rate(rows: Sequence[PlanningCallRow]) -> dict[str, object]:
    """Calculate reachable-request correctness from unique requests, never calls."""
    reachable = [row for row in rows if row.outcome_kind == "reachable"]
    consensus = unique_request_semantic_consensus(reachable)
    if not consensus:
        return {"status": "blocked", "blocking_reason": "no_reachable_unique_requests"}
    success_count = sum(consensus.values())
    total = len(consensus)
    return {
        "status": "passed" if success_count / total >= FINAL_COVERAGE_THRESHOLD else "failed",
        "reachable_unique_request_count": total,
        "reachable_unique_success_count": success_count,
        "reachable_unique_success_rate": success_count / total,
    }
