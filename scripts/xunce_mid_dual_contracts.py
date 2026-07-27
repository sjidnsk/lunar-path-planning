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
from typing import Iterable, Sequence


SCALE_PROFILE = "midterm_reduced_w8x3_update80/v1"
G1_EPISODES_PER_FORMAL_SPLIT = 24
G1_LANE_SIZES = (3, 3, 3, 3, 3, 3, 3, 3)
G2_PLATFORMS = ("wheel", "legged", "hopper")
G2_PLATFORM_INVARIANTS = {
    platform: {"max_traversable_slope_deg": 30.0}
    for platform in G2_PLATFORMS
}
G2_STANDARD_REQUESTS = 33
G2_KILOMETER_REQUESTS = 10
G2_REPEATS = 5
G2_FORMAL_CALLS = 645
G2_REQUEST_CLASS_COUNTS = {
    "standard": {
        "normal_reachable": 23,
        "hard_reachable": 7,
        "unreachable": 3,
    },
    "kilometer": {
        "normal_reachable": 6,
        "hard_reachable": 2,
        "unreachable": 2,
    },
}
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
        for field_name in ("safety_violation_count", "masked_action_count"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")


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
    request_class: str
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
        for field_name in (
            "request_id",
            "call_id",
            "platform",
            "scale",
            "request_class",
            "outcome_kind",
            "semantic_digest",
        ):
            _require_nonempty(getattr(self, field_name), field_name)
        if self.request_class not in {
            "normal_reachable",
            "hard_reachable",
            "unreachable",
        }:
            raise ValueError("request_class is outside the frozen G2 taxonomy")
        expected_outcome = (
            "unreachable"
            if self.request_class == "unreachable"
            else "reachable"
        )
        if self.outcome_kind != expected_outcome:
            raise ValueError("outcome_kind does not match request_class")
        for field_name in ("request_sha256", "provider_sha256", "oracle_sha256"):
            _require_sha256(getattr(self, field_name), field_name)
        for field_name in ("provider_success", "route_l2_valid"):
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"{field_name} must be an exact bool")
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


def _finite_values(values: Iterable[object]) -> tuple[float, ...] | None:
    """Accept only finite real-number evidence; never coerce formal inputs."""
    materialized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        materialized.append(number)
    return tuple(materialized) if materialized else None


def nearest_rank(values: Sequence[object], q: object) -> float:
    """Return the one-based nearest-rank percentile: ``ceil(q*n)``."""
    finite = _finite_values(values)
    if finite is None:
        raise ValueError("values must be non-empty and finite")
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(float(q)) or not 0.0 <= q <= 1.0:
        raise ValueError("q must be finite and within [0, 1]")
    index = max(1, math.ceil(q * len(finite))) - 1
    return sorted(finite)[index]


def episode_bootstrap_ci(values: Sequence[object]) -> dict[str, float | int | str]:
    """Return replayable 95% bootstrap CI using whole episodes as units."""
    finite = _finite_values(values)
    if finite is None:
        raise ValueError("bootstrap values must be non-empty and finite")
    generator = random.Random(BOOTSTRAP_SEED)
    sampled_means = sorted(
        mean(generator.choice(finite) for _ in range(len(finite))) for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return {
        "unit": "episode",
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "confidence_level": 0.95,
        "lower": nearest_rank(sampled_means, 0.025),
        "upper": nearest_rank(sampled_means, 0.975),
    }


def _summary(values: Sequence[object]) -> dict[str, float | int]:
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


def g1_split_statistics(
    job_ids: Sequence[str],
    coverage_values: Sequence[object],
    *,
    lane_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Compute one split's G1 thresholds; malformed formal evidence blocks closed."""
    if len(job_ids) != G1_EPISODES_PER_FORMAL_SPLIT or any(
        not isinstance(job_id, str) or not job_id for job_id in job_ids
    ):
        return {**route_gate(midterm=False, final=False, blocked=True), "blocking_reason": "g1_jobs_incomplete_or_duplicate"}
    if len(set(job_ids)) != G1_EPISODES_PER_FORMAL_SPLIT:
        return {**route_gate(midterm=False, final=False, blocked=True), "blocking_reason": "g1_jobs_incomplete_or_duplicate"}
    if lane_ids is None or len(lane_ids) != G1_EPISODES_PER_FORMAL_SPLIT:
        return {**route_gate(midterm=False, final=False, blocked=True), "blocking_reason": "g1_lanes_incomplete_or_unbalanced"}
    lane_sizes: dict[str, int] = {}
    for lane_id in lane_ids:
        if not isinstance(lane_id, str) or not lane_id:
            return {**route_gate(midterm=False, final=False, blocked=True), "blocking_reason": "g1_lanes_incomplete_or_unbalanced"}
        lane_sizes[lane_id] = lane_sizes.get(lane_id, 0) + 1
    if len(lane_sizes) != len(G1_LANE_SIZES) or tuple(sorted(lane_sizes.values())) != tuple(sorted(G1_LANE_SIZES)):
        return {**route_gate(midterm=False, final=False, blocked=True), "blocking_reason": "g1_lanes_incomplete_or_unbalanced"}
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


def evaluate_g2_platform(
    *,
    platform: str,
    scale: str,
    outcome_kind: str,
    elapsed_ms: Sequence[object],
) -> dict[str, object]:
    """Summarize one G2 timing partition using the complete frozen formula."""
    if platform not in G2_PLATFORMS or not scale or not outcome_kind:
        return {**route_gate(midterm=False, final=False, blocked=True), "blocking_reason": "invalid_g2_partition"}
    values = _finite_values(elapsed_ms)
    if values is None:
        return {**route_gate(midterm=False, final=False, blocked=True), "blocking_reason": "g2_nonfinite_or_empty_time"}
    statistics = _summary(values)
    p95 = nearest_rank(values, 0.95)
    at_or_below = sum(value <= FINAL_TIME_MS for value in values)
    midterm = (
        statistics["mean"] <= MID_TIME_MS
        and p95 <= MID_TIME_MS
        and statistics["max"] <= MID_TIME_MS
    )
    final = (
        statistics["mean"] <= FINAL_TIME_MS
        and p95 <= FINAL_TIME_MS
        and at_or_below / len(values) >= 0.95
        and statistics["max"] <= MID_TIME_MS
    )
    return {
        **route_gate(midterm=midterm, final=final),
        **statistics,
        "mean_ms": statistics["mean"],
        "p50_ms": nearest_rank(values, 0.50),
        "platform": platform,
        "scale": scale,
        "outcome_kind": outcome_kind,
        "p95_ms": p95,
        "p99_ms": nearest_rank(values, 0.99),
        "sample_stddev_ms": statistics["sample_stddev"],
        "min_ms": statistics["min"],
        "max_ms": statistics["max"],
        "at_or_below_1000_count": at_or_below,
        "proportion_at_or_below_1000ms": at_or_below / len(values),
        "over_2000_count": sum(value > MID_TIME_MS for value in values),
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


def _blocked(reason: str) -> dict[str, object]:
    return {**route_gate(midterm=False, final=False, blocked=True), "blocking_reason": reason}


def _group_rows_by_platform_request(
    rows: Sequence[PlanningCallRow],
) -> dict[tuple[str, str], list[PlanningCallRow]]:
    grouped: dict[tuple[str, str], list[PlanningCallRow]] = {}
    for row in rows:
        grouped.setdefault((row.platform, row.request_id), []).append(row)
    return grouped


def _has_cross_platform_request_id(rows: Sequence[PlanningCallRow]) -> bool:
    platforms_by_request: dict[str, set[str]] = {}
    for row in rows:
        platforms_by_request.setdefault(row.request_id, set()).add(row.platform)
    return any(len(platforms) != 1 for platforms in platforms_by_request.values())


def _request_provenance_is_stable(request_rows: Sequence[PlanningCallRow]) -> bool:
    return len(
        {
            (
                row.platform,
                row.scale,
                row.request_class,
                row.outcome_kind,
                row.source_sha256,
                row.config_sha256,
                row.request_sha256,
                row.provider_sha256,
                row.oracle_sha256,
            )
            for row in request_rows
        }
    ) == 1


def _request_repeat_structure_is_valid(request_rows: Sequence[PlanningCallRow]) -> bool:
    return (
        len(request_rows) == G2_REPEATS
        and len({row.call_id for row in request_rows}) == G2_REPEATS
        and _request_provenance_is_stable(request_rows)
    )


def unique_request_semantic_consensus(rows: Sequence[PlanningCallRow]) -> dict[tuple[str, str], bool]:
    """Apply five independent repeats per ``(platform, request_id)`` evidence key."""
    grouped = _group_rows_by_platform_request(rows)
    return {
        key: _request_repeat_structure_is_valid(request_rows)
        and all(row.provider_success and row.route_l2_valid for row in request_rows)
        and len({row.semantic_digest for row in request_rows}) == 1
        for key, request_rows in grouped.items()
    }


def reachable_request_success_rate(rows: Sequence[PlanningCallRow]) -> dict[str, object]:
    """Require exact success/failure semantics for every repeated G2 request."""
    if _has_cross_platform_request_id(rows):
        return _blocked("g2_cross_platform_request_id")
    reachable = [row for row in rows if row.outcome_kind == "reachable"]
    grouped = _group_rows_by_platform_request(reachable)
    keys_by_platform = {platform: {key for key in grouped if key[0] == platform} for platform in G2_PLATFORMS}
    if any(len(keys_by_platform[platform]) != 38 for platform in G2_PLATFORMS):
        return _blocked("g2_reachable_request_count_mismatch")
    if any(not _request_repeat_structure_is_valid(request_rows) for request_rows in grouped.values()):
        return _blocked("g2_reachable_repeat_or_provenance_mismatch")
    consensus = unique_request_semantic_consensus(reachable)
    successes_by_platform = {
        platform: sum(consensus[key] for key in keys_by_platform[platform]) for platform in G2_PLATFORMS
    }
    unreachable = [row for row in rows if row.outcome_kind == "unreachable"]
    unreachable_groups = _group_rows_by_platform_request(unreachable)
    unreachable_counts = {
        platform: sum(key[0] == platform for key in unreachable_groups)
        for platform in G2_PLATFORMS
    }
    unreachable_correct = (
        unreachable_counts == {platform: 5 for platform in G2_PLATFORMS}
        and all(
            _request_repeat_structure_is_valid(request_rows)
            and all(
                row.provider_success is False
                and row.route_l2_valid is False
                for row in request_rows
            )
            and len({row.semantic_digest for row in request_rows}) == 1
            for request_rows in unreachable_groups.values()
        )
    )
    all_succeeded = (
        all(successes_by_platform[platform] == 38 for platform in G2_PLATFORMS)
        and unreachable_correct
    )
    return {
        **route_gate(midterm=all_succeeded, final=all_succeeded),
        "reachable_unique_request_count_by_platform": {platform: 38 for platform in G2_PLATFORMS},
        "reachable_unique_success_count_by_platform": successes_by_platform,
        "reachable_unique_success_rate_by_platform": {
            platform: successes_by_platform[platform] / 38 for platform in G2_PLATFORMS
        },
        "unreachable_correct": unreachable_correct,
        "semantic_consensus": (
            all(consensus.values())
            and all(
                len({row.semantic_digest for row in request_rows}) == 1
                for request_rows in unreachable_groups.values()
            )
        ),
        "truth_provider_crosswalk_valid": True,
    }


def _formal_g2_matrix_blocking_reason(rows: Sequence[PlanningCallRow]) -> str | None:
    if len(rows) != G2_FORMAL_CALLS:
        return "g2_formal_call_count_mismatch"
    if any(not isinstance(row, PlanningCallRow) for row in rows):
        return "g2_formal_row_type_mismatch"
    if len({row.call_id for row in rows}) != G2_FORMAL_CALLS:
        return "g2_duplicate_call_id"
    if _has_cross_platform_request_id(rows):
        return "g2_cross_platform_request_id"
    groups = _group_rows_by_platform_request(rows)
    for platform in G2_PLATFORMS:
        platform_groups = {key: value for key, value in groups.items() if key[0] == platform}
        if len(platform_groups) != G2_STANDARD_REQUESTS + G2_KILOMETER_REQUESTS:
            return "g2_platform_request_matrix_incomplete"
        scale_counts = {"standard": 0, "kilometer": 0}
        class_counts = {
            scale: {
                "normal_reachable": 0,
                "hard_reachable": 0,
                "unreachable": 0,
            }
            for scale in G2_REQUEST_CLASS_COUNTS
        }
        for request_rows in platform_groups.values():
            if not _request_repeat_structure_is_valid(request_rows):
                return "g2_request_repeat_or_provenance_mismatch"
            scale = request_rows[0].scale
            if scale not in scale_counts:
                return "g2_platform_request_matrix_incomplete"
            scale_counts[scale] += 1
            request_class = request_rows[0].request_class
            if request_class not in class_counts[scale]:
                return "g2_platform_request_matrix_incomplete"
            class_counts[scale][request_class] += 1
        if scale_counts != {"standard": G2_STANDARD_REQUESTS, "kilometer": G2_KILOMETER_REQUESTS}:
            return "g2_platform_request_matrix_incomplete"
        if class_counts != G2_REQUEST_CLASS_COUNTS:
            return "g2_platform_request_matrix_incomplete"
    return None


def _g2_request_bootstrap_ci(
    rows: Sequence[PlanningCallRow],
) -> dict[str, float | int | str]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.request_id, []).append(row.elapsed_ms)
    request_means = [
        mean(values) for _request_id, values in sorted(grouped.items())
    ]
    generator = random.Random(BOOTSTRAP_SEED)
    sampled_means = sorted(
        mean(generator.choice(request_means) for _ in request_means)
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return {
        "unit": "unique_request",
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "confidence_level": 0.95,
        "lower_mean_ms": nearest_rank(sampled_means, 0.025),
        "upper_mean_ms": nearest_rank(sampled_means, 0.975),
    }


def _g2_partition_statistics(
    rows: Sequence[PlanningCallRow],
) -> dict[str, object]:
    if not rows:
        raise ValueError("G2 timing partition must not be empty")
    platform = rows[0].platform
    scale = rows[0].scale
    outcome = rows[0].outcome_kind
    result = evaluate_g2_platform(
        platform=platform,
        scale=scale,
        outcome_kind=outcome,
        elapsed_ms=[row.elapsed_ms for row in rows],
    )
    if result["status"] == "blocked":
        raise ValueError("G2 timing partition is invalid")
    return {
        key: value
        for key, value in result.items()
        if key
        in {
            "sample_count",
            "mean_ms",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "sample_stddev_ms",
            "min_ms",
            "max_ms",
            "at_or_below_1000_count",
            "proportion_at_or_below_1000ms",
            "over_2000_count",
            "midterm_reduced_passed",
            "final_threshold_reduced_passed",
        }
    } | {
        "unique_request_count": len({row.request_id for row in rows}),
        "bootstrap_ci": _g2_request_bootstrap_ci(rows),
    }


def evaluate_formal_g2(rows: Sequence[PlanningCallRow]) -> dict[str, object]:
    """Route only a complete 645-call G2 matrix; partial timing is never formal pass evidence."""
    blocking_reason = _formal_g2_matrix_blocking_reason(rows)
    if blocking_reason is not None:
        return _blocked(blocking_reason)
    correctness = reachable_request_success_rate(rows)
    if correctness["status"] == "blocked":
        return correctness
    timing_by_platform_scale: dict[str, dict[str, object]] = {}
    timing_by_platform_scale_outcome: dict[str, dict[str, object]] = {}
    timing_by_platform_scale_class: dict[str, dict[str, object]] = {}
    class_counts: dict[str, dict[str, int]] = {}
    for platform in G2_PLATFORMS:
        for scale in ("standard", "kilometer"):
            selected = [
                row
                for row in rows
                if row.platform == platform and row.scale == scale
            ]
            timing_by_platform_scale[
                f"{platform}/{scale}"
            ] = _g2_partition_statistics(selected)
            class_counts[f"{platform}/{scale}"] = dict(
                G2_REQUEST_CLASS_COUNTS[scale]
            )
            for outcome in ("reachable", "unreachable"):
                subset = [
                    row for row in selected if row.outcome_kind == outcome
                ]
                timing_by_platform_scale_outcome[
                    f"{platform}/{scale}/{outcome}"
                ] = _g2_partition_statistics(subset)
            for request_class in (
                "normal_reachable",
                "hard_reachable",
                "unreachable",
            ):
                subset = [
                    row
                    for row in selected
                    if row.request_class == request_class
                ]
                timing_by_platform_scale_class[
                    f"{platform}/{scale}/{request_class}"
                ] = _g2_partition_statistics(subset)
    partitions = (
        *timing_by_platform_scale.values(),
        *timing_by_platform_scale_outcome.values(),
        *timing_by_platform_scale_class.values(),
    )
    midterm = correctness["midterm_reduced_passed"] and all(
        statistics["midterm_reduced_passed"] for statistics in partitions
    )
    final = correctness["final_threshold_reduced_passed"] and all(
        statistics["final_threshold_reduced_passed"] for statistics in partitions
    )
    return {
        **route_gate(midterm=midterm, final=final),
        "formal_call_count": len(rows),
        "unique_request_count": len(
            {(row.platform, row.request_id) for row in rows}
        ),
        "active_platforms": list(G2_PLATFORMS),
        "platform_invariants": G2_PLATFORM_INVARIANTS,
        "g2_all_platforms_2s_passed": bool(midterm),
        "g2_all_platforms_1s_passed": bool(final),
        "correctness_passed": bool(
            correctness["midterm_reduced_passed"]
        ),
        "request_class_counts_by_platform_scale": class_counts,
        "reachable_correctness": correctness,
        "timing_by_platform_scale": timing_by_platform_scale,
        "timing_by_platform_scale_outcome": timing_by_platform_scale_outcome,
        "timing_by_platform_scale_class": timing_by_platform_scale_class,
    }
