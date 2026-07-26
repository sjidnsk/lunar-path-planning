"""中期双门槛 G3 闭环计时与跨门强联接 runner。

本模块把 G3 的判定保持为纯 raw-row 复算：轮式闭环行必须逐步形成
candidate -> request -> route -> feedback -> next-state 哈希链，并与 G1 的
整数覆盖分母结果强联接；Legged/Hopper 回放必须与 G2 的五次重复结果强联接。
任何输入、联接或计时合同缺口均 fail closed 为 ``blocked``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Mapping, Sequence

import xunce_artifact_io as artifact_io


SCALE_PROFILE = "midterm_reduced_w8x3_update80/v1"
RUNNER_ID = "run_xunce_mid_dual_g3_closed_loop/v1"
UPDATE80_CHECKPOINT_SHA256 = (
    "35e04c86f9f973af028fb08f0175d42ab45378d09e1f2b96ee6aad6e4c12b5b5"
)
UPDATE80_POLICY_STATE_SHA256 = (
    "3123e6adde99be41e3bd5cc2f3843068e5416892395926d759c2c6a243ffd381"
)
MID_COVERAGE_THRESHOLD = 0.80
FINAL_COVERAGE_THRESHOLD = 0.99
MID_TIME_MS = 2000.0
FINAL_TIME_MS = 1000.0
PAIRED_G1_DELTA_MIN = -0.01
TIMING_CONTRACT_ID = "five-phase-sequential-ns/v1"
TIMING_COMPONENT_FIELDS = (
    "input_validation_ns",
    "platform_instantiation_ns",
    "search_ns",
    "complete_route_validation_ns",
    "result_assembly_ns",
)
_G3_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "gate_id",
        "runner_id",
        "scale_profile",
        "output_base",
        "required_phase_ids",
        "required_phases",
        "formal_counts",
        "thresholds",
        "checkpoint_sha256",
        "policy_state_sha256",
        "timing_contract_id",
    }
)

_G1_ROW_FIELDS = frozenset(
    {
        "row_kind",
        "schema_version",
        "gate_id",
        "runner_id",
        "phase_id",
        "phase_name",
        "scale_profile",
        "run_id",
        "split",
        "episode_index",
        "episode_id",
        "scenario_id",
        "lane_id",
        "source_sha256",
        "config_sha256",
        "input_sha256",
        "code_sha256",
        "scenario_manifest_sha256",
        "checkpoint_sha256",
        "policy_state_sha256",
        "denominator_source",
        "denominator_algorithm",
        "denominator_sha256",
        "denominator_cell_count",
        "initial_covered_cell_count",
        "final_covered_cell_count",
        "coverage",
        "elapsed_ms",
        "steps_executed",
        "termination_reason",
        "safety_violation_count",
        "masked_action_count",
    }
)
_WHEEL_ROW_FIELDS = frozenset(
    {
        "row_kind",
        "schema_version",
        "scale_profile",
        "run_id",
        "split",
        "episode_index",
        "episode_id",
        "scenario_id",
        "lane_id",
        "step_id",
        "step_index",
        "is_terminal",
        "termination_reason",
        "decision_sha256",
        "candidate_id",
        "candidate_sha256",
        "request_candidate_sha256",
        "selected_candidate_cell_xy",
        "planned_path_cells",
        "planner_path_length_m",
        "route_endpoint_cell_xy",
        "feedback_pose_cell_xy",
        "selected_theta",
        "route_endpoint_theta",
        "feedback_theta",
        "pre_snapshot_sha256",
        "request_id",
        "request_sha256",
        "route_request_id",
        "route_request_sha256",
        "route_result_sha256",
        "feedback_request_id",
        "feedback_route_result_sha256",
        "feedback_sha256",
        "post_snapshot_parent_sha256",
        "post_snapshot_sha256",
        "coverage",
        "paired_g1_coverage",
        "safety_violation_count",
        "masked_action_count",
        "candidate_route_mismatch_count",
        "planner_success",
        "planner_failure_reason",
        *TIMING_COMPONENT_FIELDS,
        "total_ns",
    }
)
_INTERFACE_ROW_FIELDS = frozenset(
    {
        "row_kind",
        "schema_version",
        "scale_profile",
        "run_id",
        "replay_id",
        "platform",
        "request_id",
        "request_sha256",
        "g2_reference_call_ids",
        "g2_semantic_digest",
        "replay_semantic_digest",
        "timing_contract_id",
        "formal_input_eligible",
        *TIMING_COMPONENT_FIELDS,
        "total_ns",
    }
)
_G2_REFERENCE_FIELDS = frozenset(
    {
        "row_kind",
        "platform",
        "request_id",
        "request_sha256",
        "call_id",
        "repeat_index",
        "semantic_digest",
        "provider_success",
        "route_l2_valid",
        "timing_contract_id",
    }
)


class G3Blocked(ValueError):
    """G3 证据不能形成正式判定。"""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def validate_g3_config_payload(
    payload: object,
) -> dict[str, object]:
    """验证仓库内 G3 canonical config，不接受运行时放宽或 override。"""

    if not isinstance(payload, Mapping) or set(payload) != set(
        _G3_CONFIG_FIELDS
    ):
        raise G3Blocked("g3_config_invalid")
    expected = {
        "schema_version": "mid-dual-g3-config/v1",
        "gate_id": "g3",
        "runner_id": RUNNER_ID,
        "scale_profile": SCALE_PROFILE,
        "output_base": "D:/xunce/out/mid_dual/g3",
        "required_phase_ids": ["p01", "p02"],
        "required_phases": ["wheel_closed_loop", "interface_replay"],
        "formal_counts": {
            "wheel_episodes": 10,
            "test_q24": 5,
            "unseen24": 5,
            "legged_replays": 3,
            "hopper_replays": 3,
        },
        "thresholds": {
            "midterm_coverage": MID_COVERAGE_THRESHOLD,
            "final_coverage": FINAL_COVERAGE_THRESHOLD,
            "paired_g1_delta_min": PAIRED_G1_DELTA_MIN,
            "midterm_planner_ms": MID_TIME_MS,
            "final_planner_ms": FINAL_TIME_MS,
            "absolute_max_planner_ms": MID_TIME_MS,
        },
        "checkpoint_sha256": UPDATE80_CHECKPOINT_SHA256,
        "policy_state_sha256": UPDATE80_POLICY_STATE_SHA256,
        "timing_contract_id": TIMING_CONTRACT_ID,
    }
    copied = json.loads(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    if copied != expected:
        raise G3Blocked("g3_config_invalid")
    return copied


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, reason: str) -> str:
    if not _is_sha256(value):
        raise G3Blocked(reason)
    return str(value)


def _require_nonempty(value: object, reason: str) -> str:
    if not isinstance(value, str) or not value:
        raise G3Blocked(reason)
    return value


def _require_exact_int(value: object, reason: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise G3Blocked(reason)
    return value


def _require_exact_bool(value: object, reason: str) -> bool:
    if type(value) is not bool:
        raise G3Blocked(reason)
    return value


def _finite_number(
    value: object,
    reason: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise G3Blocked(reason)
    number = float(value)
    if minimum is not None and number < minimum:
        raise G3Blocked(reason)
    if maximum is not None and number > maximum:
        raise G3Blocked(reason)
    return number


def _require_exact_keys(
    row: object,
    expected: frozenset[str],
    reason: str,
) -> Mapping[str, object]:
    if not isinstance(row, Mapping) or set(row) != set(expected):
        raise G3Blocked(reason)
    return row


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise G3Blocked("g3_timing_contract")
    index = max(1, math.ceil(quantile * len(values))) - 1
    return sorted(values)[index]


def _timing_statistics(values: Sequence[float]) -> dict[str, object]:
    rows = tuple(values)
    if not rows:
        raise G3Blocked("g3_timing_contract")
    p95 = _nearest_rank(rows, 0.95)
    p99 = _nearest_rank(rows, 0.99)
    average = mean(rows)
    return {
        "sample_count": len(rows),
        "mean_ms": average,
        "median_ms": median(rows),
        "sample_stddev_ms": stdev(rows) if len(rows) > 1 else 0.0,
        "min_ms": min(rows),
        "max_ms": max(rows),
        "p95_ms": p95,
        "p99_ms": p99,
        "count_le_1000ms": sum(value <= FINAL_TIME_MS for value in rows),
        "count_le_2000ms": sum(value <= MID_TIME_MS for value in rows),
        "midterm_reduced_passed": (
            average <= MID_TIME_MS
            and p95 <= MID_TIME_MS
            and max(rows) <= MID_TIME_MS
        ),
        "final_threshold_reduced_passed": (
            average <= FINAL_TIME_MS
            and p95 <= FINAL_TIME_MS
            and sum(value <= FINAL_TIME_MS for value in rows)
            >= math.ceil(0.95 * len(rows))
            and max(rows) <= MID_TIME_MS
        ),
    }


def _timing_ms(row: Mapping[str, object]) -> float:
    components = tuple(
        _require_exact_int(row.get(field), "g3_timing_contract")
        for field in TIMING_COMPONENT_FIELDS
    )
    total = _require_exact_int(row.get("total_ns"), "g3_timing_contract")
    if total != sum(components):
        raise G3Blocked("g3_timing_contract")
    return total / 1_000_000.0


def deterministic_request_id(
    *,
    scenario_id: str,
    step_index: int,
    pre_snapshot_sha256: str,
) -> str:
    """从冻结三元组域分隔生成 G3 planner request ID。"""

    scenario = _require_nonempty(scenario_id, "g3_request_id")
    step = _require_exact_int(step_index, "g3_request_id")
    snapshot = _require_sha256(pre_snapshot_sha256, "g3_request_id")
    encoded = (
        "mid-dual-g3-request/v1\0"
        f"{scenario}\0{step}\0{snapshot}"
    ).encode("utf-8")
    return f"g3-wheel-{hashlib.sha256(encoded).hexdigest()}"


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise G3Blocked("g3_wheel_hash_chain") from exc
    return hashlib.sha256(encoded).hexdigest()


def deterministic_candidate_sha256(
    *,
    scenario_id: str,
    step_index: int,
    pre_snapshot_sha256: str,
    decision_sha256: str,
    candidate_id: str,
    selected_candidate_cell_xy: Sequence[int],
    selected_theta: float,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "mid-dual-g3-candidate-binding/v1",
            "scenario_id": _require_nonempty(
                scenario_id,
                "g3_wheel_hash_chain",
            ),
            "step_index": _require_exact_int(
                step_index,
                "g3_wheel_hash_chain",
            ),
            "pre_snapshot_sha256": _require_sha256(
                pre_snapshot_sha256,
                "g3_wheel_hash_chain",
            ),
            "decision_sha256": _require_sha256(
                decision_sha256,
                "g3_wheel_hash_chain",
            ),
            "candidate_id": _require_nonempty(
                candidate_id,
                "g3_wheel_hash_chain",
            ),
            "selected_candidate_cell_xy": list(
                _cell(
                    selected_candidate_cell_xy,
                    "g3_wheel_hash_chain",
                )
            ),
            "selected_theta": _theta(
                selected_theta,
                "g3_wheel_hash_chain",
            ),
        }
    )


def deterministic_request_sha256(
    *,
    request_id: str,
    candidate_sha256: str,
    pre_snapshot_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "mid-dual-g3-planner-request-binding/v1",
            "request_id": _require_nonempty(
                request_id,
                "g3_wheel_hash_chain",
            ),
            "candidate_sha256": _require_sha256(
                candidate_sha256,
                "g3_wheel_hash_chain",
            ),
            "pre_snapshot_sha256": _require_sha256(
                pre_snapshot_sha256,
                "g3_wheel_hash_chain",
            ),
        }
    )


def deterministic_route_result_sha256(
    *,
    request_sha256: str,
    planner_success: bool,
    planner_failure_reason: str,
    planned_path_cells: Sequence[Sequence[int]],
    planner_path_length_m: float,
    route_endpoint_cell_xy: Sequence[int],
    route_endpoint_theta: float,
) -> str:
    if not isinstance(planned_path_cells, (list, tuple)):
        raise G3Blocked("g3_wheel_hash_chain")
    path = [
        list(_cell(value, "g3_wheel_hash_chain"))
        for value in planned_path_cells
    ]
    return _canonical_sha256(
        {
            "schema_version": "mid-dual-g3-route-result-binding/v1",
            "request_sha256": _require_sha256(
                request_sha256,
                "g3_wheel_hash_chain",
            ),
            "planner_success": _require_exact_bool(
                planner_success,
                "g3_wheel_hash_chain",
            ),
            "planner_failure_reason": _require_nonempty(
                planner_failure_reason,
                "g3_wheel_hash_chain",
            ),
            "planned_path_cells": path,
            "planner_path_length_m": _finite_number(
                planner_path_length_m,
                "g3_wheel_hash_chain",
                minimum=0.0,
            ),
            "route_endpoint_cell_xy": list(
                _cell(
                    route_endpoint_cell_xy,
                    "g3_wheel_hash_chain",
                )
            ),
            "route_endpoint_theta": _theta(
                route_endpoint_theta,
                "g3_wheel_hash_chain",
            ),
        }
    )


def deterministic_feedback_sha256(
    *,
    route_result_sha256: str,
    feedback_pose_cell_xy: Sequence[int],
    feedback_theta: float,
    coverage: float,
    safety_violation_count: int,
    masked_action_count: int,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "mid-dual-g3-feedback-binding/v1",
            "route_result_sha256": _require_sha256(
                route_result_sha256,
                "g3_wheel_hash_chain",
            ),
            "feedback_pose_cell_xy": list(
                _cell(
                    feedback_pose_cell_xy,
                    "g3_wheel_hash_chain",
                )
            ),
            "feedback_theta": _theta(
                feedback_theta,
                "g3_wheel_hash_chain",
            ),
            "coverage": _finite_number(
                coverage,
                "g3_wheel_hash_chain",
                minimum=0.0,
                maximum=1.0,
            ),
            "safety_violation_count": _require_exact_int(
                safety_violation_count,
                "g3_wheel_hash_chain",
            ),
            "masked_action_count": _require_exact_int(
                masked_action_count,
                "g3_wheel_hash_chain",
            ),
        }
    )


def validate_g3_manifest(
    frozen_manifest: Mapping[str, object],
) -> dict[str, tuple[str, ...]]:
    """只从冻结清单取得 5+5；禁止从 G1/G2 结果选择场景。"""

    if (
        not isinstance(frozen_manifest, Mapping)
        or frozen_manifest.get("schema_version")
        != "mid-dual-scenario-freeze/v1"
        or frozen_manifest.get("completion_status") != "complete"
    ):
        raise G3Blocked("g3_frozen_manifest")
    cohorts = frozen_manifest.get("cohorts")
    if not isinstance(cohorts, Mapping):
        raise G3Blocked("g3_frozen_manifest")
    required = {
        "test_q24": (24, "test/"),
        "unseen24": (24, "unseen/"),
        "g3_test_q5": (5, "test/"),
        "g3_unseen5": (5, "unseen/"),
    }
    parsed: dict[str, tuple[str, ...]] = {}
    for name, (count, prefix) in required.items():
        values = cohorts.get(name)
        if (
            not isinstance(values, (list, tuple))
            or len(values) != count
            or len(set(values)) != count
            or any(
                not isinstance(value, str)
                or not value.startswith(prefix)
                for value in values
            )
        ):
            raise G3Blocked("g3_frozen_manifest")
        parsed[name] = tuple(values)
    if (
        not set(parsed["g3_test_q5"]).issubset(parsed["test_q24"])
        or not set(parsed["g3_unseen5"]).issubset(parsed["unseen24"])
        or set(parsed["g3_test_q5"]).intersection(parsed["g3_unseen5"])
    ):
        raise G3Blocked("g3_frozen_manifest")
    return {
        "test_q24": parsed["g3_test_q5"],
        "unseen24": parsed["g3_unseen5"],
    }


def _g1_index(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], Mapping[str, object]]:
    if len(rows) != 48:
        raise G3Blocked("g3_g1_cross_root_join")
    result: dict[tuple[str, str], Mapping[str, object]] = {}
    split_counts = {"test_q24": 0, "unseen24": 0}
    for raw in rows:
        row = _require_exact_keys(raw, _G1_ROW_FIELDS, "g3_g1_cross_root_join")
        if (
            row.get("row_kind") != "coverage_episode"
            or row.get("schema_version")
            != "xunce-mid-dual-g1-coverage-episode/v1"
            or row.get("gate_id") != "g1"
            or row.get("runner_id")
            != "run_xunce_mid_dual_g1_coverage/v1"
            or row.get("scale_profile") != SCALE_PROFILE
        ):
            raise G3Blocked("g3_g1_cross_root_join")
        split = row.get("split")
        if split not in split_counts:
            raise G3Blocked("g3_g1_cross_root_join")
        episode_index = _require_exact_int(
            row.get("episode_index"),
            "g3_g1_cross_root_join",
        )
        if episode_index >= 24 or row.get("lane_id") != f"lane-{episode_index % 8}":
            raise G3Blocked("g3_g1_cross_root_join")
        expected_phase = "p03" if split == "test_q24" else "p04"
        if (
            row.get("phase_id") != expected_phase
            or row.get("phase_name") != split
            or row.get("episode_id")
            != f"{split}-episode-{episode_index:02d}"
        ):
            raise G3Blocked("g3_g1_cross_root_join")
        scenario_id = _require_nonempty(
            row.get("scenario_id"),
            "g3_g1_cross_root_join",
        )
        expected_prefix = "test/" if split == "test_q24" else "unseen/"
        if not scenario_id.startswith(expected_prefix):
            raise G3Blocked("g3_g1_cross_root_join")
        _require_nonempty(row.get("run_id"), "g3_g1_cross_root_join")
        _require_nonempty(row.get("episode_id"), "g3_g1_cross_root_join")
        if (
            row.get("checkpoint_sha256") != UPDATE80_CHECKPOINT_SHA256
            or row.get("policy_state_sha256") != UPDATE80_POLICY_STATE_SHA256
        ):
            raise G3Blocked("g3_g1_update80_binding")
        for name in (
            "source_sha256",
            "config_sha256",
            "input_sha256",
            "code_sha256",
            "scenario_manifest_sha256",
        ):
            _require_sha256(row.get(name), "g3_g1_cross_root_join")
        if row.get("source_sha256") != row.get("code_sha256"):
            raise G3Blocked("g3_g1_cross_root_join")
        if (
            row.get("denominator_source")
            != "reachable_observable_free_highres_cells/v1"
            or row.get("denominator_algorithm")
            != "exact_reachable_safe_pose_range_los/v1"
        ):
            raise G3Blocked("g3_g1_cross_root_join")
        _require_sha256(
            row.get("denominator_sha256"),
            "g3_g1_cross_root_join",
        )
        denominator = _require_exact_int(
            row.get("denominator_cell_count"),
            "g3_g1_cross_root_join",
            minimum=1,
        )
        initial = _require_exact_int(
            row.get("initial_covered_cell_count"),
            "g3_g1_cross_root_join",
        )
        covered = _require_exact_int(
            row.get("final_covered_cell_count"),
            "g3_g1_cross_root_join",
        )
        if initial > covered or covered > denominator:
            raise G3Blocked("g3_g1_cross_root_join")
        _finite_number(
            row.get("elapsed_ms"),
            "g3_g1_cross_root_join",
            minimum=0.0,
        )
        _require_exact_int(row.get("steps_executed"), "g3_g1_cross_root_join")
        _require_nonempty(
            row.get("termination_reason"),
            "g3_g1_cross_root_join",
        )
        if (
            _require_exact_int(
                row.get("safety_violation_count"),
                "g3_g1_cross_root_join",
            )
            != 0
            or _require_exact_int(
                row.get("masked_action_count"),
                "g3_g1_cross_root_join",
            )
            != 0
        ):
            raise G3Blocked("g3_g1_cross_root_join")
        coverage = _finite_number(
            row.get("coverage"),
            "g3_g1_cross_root_join",
            minimum=0.0,
            maximum=1.0,
        )
        if not math.isclose(
            coverage,
            covered / denominator,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise G3Blocked("g3_g1_cross_root_join")
        key = (str(split), scenario_id)
        if key in result:
            raise G3Blocked("g3_g1_cross_root_join")
        result[key] = row
        split_counts[str(split)] += 1
    if split_counts != {"test_q24": 24, "unseen24": 24}:
        raise G3Blocked("g3_g1_cross_root_join")
    return result


def _cell(value: object, reason: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        raise G3Blocked(reason)
    return int(value[0]), int(value[1])


def _theta(value: object, reason: str) -> float:
    return _finite_number(value, reason)


def _evaluate_wheel(
    *,
    selected: Mapping[str, tuple[str, ...]],
    wheel_rows: Sequence[Mapping[str, object]],
    g1_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    g1 = _g1_index(g1_rows)
    if not wheel_rows:
        raise G3Blocked("g3_wheel_episode_matrix")
    episodes: dict[str, list[Mapping[str, object]]] = {}
    timings: list[float] = []
    expected_scenarios = {
        (split, scenario_id)
        for split, scenario_ids in selected.items()
        for scenario_id in scenario_ids
    }
    observed_scenarios: set[tuple[str, str]] = set()
    seen_steps: set[str] = set()
    for raw in wheel_rows:
        row = _require_exact_keys(raw, _WHEEL_ROW_FIELDS, "g3_wheel_row_schema")
        if (
            row.get("row_kind") != "g3_wheel_step"
            or row.get("schema_version") != "mid-dual-g3-wheel-step/v1"
            or row.get("scale_profile") != SCALE_PROFILE
        ):
            raise G3Blocked("g3_wheel_row_schema")
        _require_nonempty(row.get("run_id"), "g3_wheel_row_schema")
        split = row.get("split")
        scenario_id = row.get("scenario_id")
        if (
            split not in selected
            or not isinstance(scenario_id, str)
            or (str(split), scenario_id) not in expected_scenarios
        ):
            raise G3Blocked("g3_wheel_episode_matrix")
        episode_id = _require_nonempty(
            row.get("episode_id"),
            "g3_wheel_episode_matrix",
        )
        episode_index = _require_exact_int(
            row.get("episode_index"),
            "g3_wheel_episode_matrix",
        )
        g1_row = g1.get((str(split), scenario_id))
        if (
            g1_row is None
            or episode_index != g1_row.get("episode_index")
            or row.get("lane_id") != g1_row.get("lane_id")
        ):
            raise G3Blocked("g3_g1_cross_root_join")
        step_id = _require_nonempty(row.get("step_id"), "g3_wheel_step_chain")
        if step_id in seen_steps:
            raise G3Blocked("g3_wheel_step_chain")
        seen_steps.add(step_id)
        _require_exact_int(row.get("step_index"), "g3_wheel_step_chain")
        _require_exact_bool(row.get("is_terminal"), "g3_wheel_step_chain")
        _require_nonempty(row.get("termination_reason"), "g3_wheel_step_chain")
        pre = _require_sha256(row.get("pre_snapshot_sha256"), "g3_request_id")
        expected_request_id = deterministic_request_id(
            scenario_id=scenario_id,
            step_index=int(row["step_index"]),
            pre_snapshot_sha256=pre,
        )
        request_id = _require_nonempty(row.get("request_id"), "g3_request_id")
        if request_id != expected_request_id:
            raise G3Blocked("g3_request_id")
        candidate_sha = _require_sha256(
            row.get("candidate_sha256"),
            "g3_wheel_hash_chain",
        )
        decision_sha = _require_sha256(
            row.get("decision_sha256"),
            "g3_wheel_hash_chain",
        )
        candidate_id = _require_nonempty(
            row.get("candidate_id"),
            "g3_wheel_hash_chain",
        )
        request_sha = _require_sha256(
            row.get("request_sha256"),
            "g3_wheel_hash_chain",
        )
        route_sha = _require_sha256(
            row.get("route_result_sha256"),
            "g3_wheel_hash_chain",
        )
        feedback_sha = _require_sha256(
            row.get("feedback_sha256"),
            "g3_wheel_hash_chain",
        )
        _require_sha256(
            row.get("post_snapshot_sha256"),
            "g3_wheel_hash_chain",
        )
        selected_cell = _cell(
            row.get("selected_candidate_cell_xy"),
            "g3_wheel_endpoint_mismatch",
        )
        planned_path_value = row.get("planned_path_cells")
        if not isinstance(planned_path_value, (list, tuple)):
            raise G3Blocked("g3_wheel_hash_chain")
        planned_path = [
            list(_cell(value, "g3_wheel_hash_chain"))
            for value in planned_path_value
        ]
        planner_length = _finite_number(
            row.get("planner_path_length_m"),
            "g3_wheel_hash_chain",
            minimum=0.0,
        )
        route_endpoint = _cell(
            row.get("route_endpoint_cell_xy"),
            "g3_wheel_endpoint_mismatch",
        )
        feedback_pose = _cell(
            row.get("feedback_pose_cell_xy"),
            "g3_wheel_endpoint_mismatch",
        )
        selected_theta = _theta(
            row.get("selected_theta"),
            "g3_wheel_endpoint_mismatch",
        )
        route_theta = _theta(
            row.get("route_endpoint_theta"),
            "g3_wheel_endpoint_mismatch",
        )
        feedback_theta = _theta(
            row.get("feedback_theta"),
            "g3_wheel_endpoint_mismatch",
        )
        planner_success = _require_exact_bool(
            row.get("planner_success"),
            "g3_planner_failure",
        )
        planner_failure_reason = _require_nonempty(
            row.get("planner_failure_reason"),
            "g3_planner_failure",
        )
        safety_count = _require_exact_int(
            row.get("safety_violation_count"),
            "g3_wheel_integrity",
        )
        masked_count = _require_exact_int(
            row.get("masked_action_count"),
            "g3_wheel_integrity",
        )
        coverage = _finite_number(
            row.get("coverage"),
            "g3_wheel_coverage",
            minimum=0.0,
            maximum=1.0,
        )
        if candidate_sha != deterministic_candidate_sha256(
            scenario_id=scenario_id,
            step_index=int(row["step_index"]),
            pre_snapshot_sha256=pre,
            decision_sha256=decision_sha,
            candidate_id=candidate_id,
            selected_candidate_cell_xy=selected_cell,
            selected_theta=selected_theta,
        ):
            raise G3Blocked("g3_wheel_hash_chain")
        if request_sha != deterministic_request_sha256(
            request_id=request_id,
            candidate_sha256=candidate_sha,
            pre_snapshot_sha256=pre,
        ):
            raise G3Blocked("g3_wheel_hash_chain")
        if route_sha != deterministic_route_result_sha256(
            request_sha256=request_sha,
            planner_success=planner_success,
            planner_failure_reason=planner_failure_reason,
            planned_path_cells=planned_path,
            planner_path_length_m=planner_length,
            route_endpoint_cell_xy=route_endpoint,
            route_endpoint_theta=route_theta,
        ):
            raise G3Blocked("g3_wheel_hash_chain")
        if feedback_sha != deterministic_feedback_sha256(
            route_result_sha256=route_sha,
            feedback_pose_cell_xy=feedback_pose,
            feedback_theta=feedback_theta,
            coverage=coverage,
            safety_violation_count=safety_count,
            masked_action_count=masked_count,
        ):
            raise G3Blocked("g3_wheel_hash_chain")
        if (
            row.get("request_candidate_sha256") != candidate_sha
            or row.get("route_request_id") != request_id
            or row.get("route_request_sha256") != request_sha
            or row.get("feedback_request_id") != request_id
            or row.get("feedback_route_result_sha256") != route_sha
            or row.get("post_snapshot_parent_sha256") != feedback_sha
        ):
            raise G3Blocked("g3_wheel_hash_chain")
        if (
            route_endpoint != selected_cell
            or feedback_pose != selected_cell
            or not math.isclose(
                selected_theta,
                route_theta,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                selected_theta,
                feedback_theta,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise G3Blocked("g3_wheel_endpoint_mismatch")
        if (
            planner_success is not True
            or planner_failure_reason != "none"
        ):
            raise G3Blocked("g3_planner_failure")
        mismatch_count = _require_exact_int(
            row.get("candidate_route_mismatch_count"),
            "g3_wheel_integrity",
        )
        if safety_count != 0 or masked_count != 0 or mismatch_count != 0:
            raise G3Blocked("g3_wheel_integrity")
        paired = _finite_number(
            row.get("paired_g1_coverage"),
            "g3_g1_cross_root_join",
            minimum=0.0,
            maximum=1.0,
        )
        g1_coverage = float(g1_row["coverage"])
        if not math.isclose(
            paired,
            g1_coverage,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise G3Blocked("g3_g1_cross_root_join")
        timings.append(_timing_ms(row))
        episodes.setdefault(episode_id, []).append(row)
        observed_scenarios.add((str(split), scenario_id))

    if len(episodes) != 10 or observed_scenarios != expected_scenarios:
        raise G3Blocked("g3_wheel_episode_matrix")
    terminal_rows: list[Mapping[str, object]] = []
    for episode_rows in episodes.values():
        ordered = sorted(
            episode_rows,
            key=lambda row: int(row["step_index"]),
        )
        if [row["step_index"] for row in ordered] != list(range(len(ordered))):
            raise G3Blocked("g3_wheel_step_chain")
        stable = {
            (
                row["scenario_id"],
                row["split"],
                row["episode_index"],
                row["episode_id"],
                row["lane_id"],
            )
            for row in ordered
        }
        terminals = [
            row for row in ordered if row.get("is_terminal") is True
        ]
        if (
            len(stable) != 1
            or len(terminals) != 1
            or terminals[0] is not ordered[-1]
        ):
            raise G3Blocked("g3_wheel_step_chain")
        for left, right in zip(ordered[:-1], ordered[1:], strict=True):
            if left["post_snapshot_sha256"] != right["pre_snapshot_sha256"]:
                raise G3Blocked("g3_wheel_step_chain")
        terminal_rows.append(terminals[0])

    final_coverages = [float(row["coverage"]) for row in terminal_rows]
    deltas = [
        float(row["coverage"]) - float(row["paired_g1_coverage"])
        for row in terminal_rows
    ]
    paired_delta_mean = round(mean(deltas), 15)
    timing = _timing_statistics(timings)
    coverage_mean = mean(final_coverages)
    coverage_80_count = sum(
        value >= MID_COVERAGE_THRESHOLD for value in final_coverages
    )
    coverage_99_count = sum(
        value >= FINAL_COVERAGE_THRESHOLD for value in final_coverages
    )
    return {
        "wheel_episode_count": 10,
        "wheel_step_count": len(wheel_rows),
        "coverage_mean": coverage_mean,
        "coverage_80_count": coverage_80_count,
        "coverage_99_count": coverage_99_count,
        "paired_g1_coverage_delta_mean": paired_delta_mean,
        "timing": timing,
        "midterm_reduced_passed": (
            coverage_mean >= MID_COVERAGE_THRESHOLD
            and coverage_80_count == 10
            and paired_delta_mean >= PAIRED_G1_DELTA_MIN
            and timing["midterm_reduced_passed"] is True
        ),
        "final_threshold_reduced_passed": (
            coverage_mean >= FINAL_COVERAGE_THRESHOLD
            and coverage_99_count == 10
            and paired_delta_mean >= PAIRED_G1_DELTA_MIN
            and timing["final_threshold_reduced_passed"] is True
        ),
    }


def execute_update80_wheel_rows(
    *,
    catalog: object,
    frozen_manifest: object,
    raw_frozen_manifest: Mapping[str, object],
    g1_rows: Sequence[Mapping[str, object]],
    safety_contract: object,
    config_sha256: str,
    run_id: str,
    evaluation_seed_start: int,
    policy: object | None = None,
) -> list[dict[str, object]]:
    """顺序重跑冻结 5+5 update80 episode，并保留每个 planner step。

    该路径刻意不调用 Standard 的 16/24/64 调度器；G3 的正式样本恰为十个
    episode，不通过补跑额外场景来凑 worker 数。
    """

    import numpy as np
    import torch
    from torch import nn

    from lunar_exploration_ppo.env.standard_training import (
        StandardEvaluationEnv,
    )
    from lunar_exploration_ppo.eval import standard
    from lunar_exploration_ppo.eval.midterm_reduced import (
        load_midterm_update80_policy,
    )
    from lunar_exploration_ppo.policy.cross_attention import (
        batch_policy_observations,
    )

    selected = validate_g3_manifest(raw_frozen_manifest)
    g1 = _g1_index(g1_rows)
    records = {
        f"{record.scenario_id}/standard-proxy/v1": record
        for record in getattr(catalog, "records", ())
    }
    if set(selected["test_q24"] + selected["unseen24"]) - set(records):
        raise G3Blocked("g3_frozen_manifest_catalog_join")
    frozen_cohorts = getattr(frozen_manifest, "cohorts", None)
    if (
        not isinstance(frozen_cohorts, Mapping)
        or tuple(frozen_cohorts.get("g3_test_q5", ()))
        != selected["test_q24"]
        or tuple(frozen_cohorts.get("g3_unseen5", ()))
        != selected["unseen24"]
    ):
        raise G3Blocked("g3_frozen_manifest_catalog_join")
    if policy is None:
        policy = load_midterm_update80_policy()
    if not isinstance(policy, nn.Module):
        raise G3Blocked("g3_update80_policy_invalid")

    sequence_index = 0
    output_rows: list[dict[str, object]] = []
    original_training = policy.training
    policy.eval()
    try:
        for split in ("test_q24", "unseen24"):
            for scenario_id in selected[split]:
                g1_row = g1.get((split, scenario_id))
                if g1_row is None:
                    raise G3Blocked("g3_g1_cross_root_join")
                episode_index = int(g1_row["episode_index"])
                record = records[scenario_id]
                expected_source_split = (
                    "test" if split == "test_q24" else "unseen"
                )
                if record.split != expected_source_split:
                    raise G3Blocked("g3_frozen_manifest_catalog_join")
                environment = StandardEvaluationEnv(
                    catalog,
                    split=record.split,
                    scenario_ids=(record.scenario_id,),
                    safety_contract=safety_contract,
                    config_sha256=config_sha256,
                )
                try:
                    observation = environment.reset()
                    if environment.is_done or not environment.needs_policy:
                        raise G3Blocked("g3_wheel_reset_terminal")
                    episode_id = (
                        f"g3-{split}-episode-{episode_index:02d}"
                    )
                    step_index = 0
                    while not environment.is_done and step_index < 128:
                        batch = batch_policy_observations(
                            (observation,),
                            device="cuda",
                        )
                        with torch.inference_mode():
                            policy_output = policy(batch)
                        action = standard.select_standard_evaluation_actions(
                            method="ppo_policy",
                            observations=(observation,),
                            evaluation_seeds=(
                                evaluation_seed_start
                                + episode_index * 128
                                + step_index,
                            ),
                            policy_output=policy_output,
                        )[0]
                        decision = standard.build_standard_decision_record(
                            method="ppo_policy",
                            sequence_index=sequence_index,
                            episode_index=episode_index,
                            step_index=step_index,
                            worker_index=episode_index % 8,
                            observation=observation,
                            action=action,
                            selector_inputs={
                                "observation_batch": batch,
                                "policy_output": policy_output,
                            },
                            policy_batch_row_index=0,
                        )
                        sequence_index += 1
                        valid_indices = tuple(
                            int(value)
                            for value in np.flatnonzero(
                                np.asarray(
                                    observation.candidate_mask,
                                    dtype=bool,
                                )
                            )
                        )
                        try:
                            selected_rank = valid_indices.index(
                                action.candidate_index
                            )
                            selected_cell = environment.frontier_cells[
                                selected_rank
                            ]
                        except (IndexError, ValueError) as exc:
                            raise G3Blocked(
                                "g3_selected_candidate_invalid"
                            ) from exc
                        pre_snapshot_sha256 = _require_sha256(
                            decision.get("policy_observation_sha256"),
                            "g3_wheel_hash_chain",
                        )
                        decision_sha256 = _require_sha256(
                            decision.get("decision_sha256"),
                            "g3_wheel_hash_chain",
                        )
                        candidate_id = (
                            f"g3-candidate:{split}:{episode_index:02d}:"
                            f"{step_index:03d}:{selected_cell.x}:"
                            f"{selected_cell.y}"
                        )
                        selected_cell_xy = [
                            int(selected_cell.x),
                            int(selected_cell.y),
                        ]
                        candidate_sha256 = deterministic_candidate_sha256(
                            scenario_id=scenario_id,
                            step_index=step_index,
                            pre_snapshot_sha256=pre_snapshot_sha256,
                            decision_sha256=decision_sha256,
                            candidate_id=candidate_id,
                            selected_candidate_cell_xy=selected_cell_xy,
                            selected_theta=float(action.target_theta),
                        )
                        request_id = deterministic_request_id(
                            scenario_id=scenario_id,
                            step_index=step_index,
                            pre_snapshot_sha256=pre_snapshot_sha256,
                        )
                        request_sha256 = deterministic_request_sha256(
                            request_id=request_id,
                            candidate_sha256=candidate_sha256,
                            pre_snapshot_sha256=pre_snapshot_sha256,
                        )

                        step = environment.step(action)
                        diagnostics = step.diagnostics.planner
                        if not isinstance(diagnostics, Mapping):
                            raise G3Blocked("g3_timing_contract")
                        timing = {
                            name: _require_exact_int(
                                diagnostics.get(name),
                                "g3_timing_contract",
                            )
                            for name in TIMING_COMPONENT_FIELDS
                        }
                        timing["total_ns"] = _require_exact_int(
                            diagnostics.get("total_ns"),
                            "g3_timing_contract",
                        )
                        if timing["total_ns"] != sum(
                            timing[name]
                            for name in TIMING_COMPONENT_FIELDS
                        ):
                            raise G3Blocked("g3_timing_contract")
                        planned_path_cells = [
                            [int(cell.x), int(cell.y)]
                            for cell in step.diagnostics.planned_path_cells
                        ]
                        planner_failure_reason = str(
                            diagnostics.get("failure_reason", "unknown")
                        )
                        planner_success = (
                            planner_failure_reason == "none"
                            and not step.diagnostics.invalid_action
                            and bool(planned_path_cells)
                        )
                        route_endpoint = (
                            planned_path_cells[-1]
                            if planned_path_cells
                            else selected_cell_xy
                        )
                        route_theta = float(action.target_theta)
                        route_result_sha256 = (
                            deterministic_route_result_sha256(
                                request_sha256=request_sha256,
                                planner_success=planner_success,
                                planner_failure_reason=(
                                    planner_failure_reason
                                ),
                                planned_path_cells=planned_path_cells,
                                planner_path_length_m=float(
                                    step.diagnostics.path_length_m
                                ),
                                route_endpoint_cell_xy=route_endpoint,
                                route_endpoint_theta=route_theta,
                            )
                        )
                        feedback_pose = [
                            int(environment.pose.cell.x),
                            int(environment.pose.cell.y),
                        ]
                        feedback_theta = float(environment.pose.theta)
                        safety_count = int(
                            step.diagnostics.safety_violation
                        )
                        masked_count = int(
                            step.diagnostics.invalid_action
                        )
                        mismatch_count = int(
                            route_endpoint != selected_cell_xy
                            or feedback_pose != selected_cell_xy
                            or not math.isclose(
                                route_theta,
                                float(action.target_theta),
                                rel_tol=0.0,
                                abs_tol=1.0e-12,
                            )
                            or not math.isclose(
                                feedback_theta,
                                float(action.target_theta),
                                rel_tol=0.0,
                                abs_tol=1.0e-12,
                            )
                        )
                        coverage = float(step.coverage_rate)
                        feedback_sha256 = deterministic_feedback_sha256(
                            route_result_sha256=route_result_sha256,
                            feedback_pose_cell_xy=feedback_pose,
                            feedback_theta=feedback_theta,
                            coverage=coverage,
                            safety_violation_count=safety_count,
                            masked_action_count=masked_count,
                        )
                        post_snapshot_sha256 = _require_sha256(
                            standard._policy_observation_sha256(  # noqa: SLF001
                                step.observation
                            ),
                            "g3_wheel_hash_chain",
                        )
                        output_rows.append(
                            {
                                "row_kind": "g3_wheel_step",
                                "schema_version": (
                                    "mid-dual-g3-wheel-step/v1"
                                ),
                                "scale_profile": SCALE_PROFILE,
                                "run_id": run_id,
                                "split": split,
                                "episode_index": episode_index,
                                "episode_id": episode_id,
                                "scenario_id": scenario_id,
                                "lane_id": g1_row["lane_id"],
                                "step_id": (
                                    f"{episode_id}:step:{step_index:03d}"
                                ),
                                "step_index": step_index,
                                "is_terminal": bool(step.done),
                                "termination_reason": (
                                    str(step.reason)
                                    if step.done
                                    else "none"
                                ),
                                "decision_sha256": decision_sha256,
                                "candidate_id": candidate_id,
                                "candidate_sha256": candidate_sha256,
                                "request_candidate_sha256": (
                                    candidate_sha256
                                ),
                                "selected_candidate_cell_xy": (
                                    selected_cell_xy
                                ),
                                "planned_path_cells": planned_path_cells,
                                "planner_path_length_m": float(
                                    step.diagnostics.path_length_m
                                ),
                                "route_endpoint_cell_xy": route_endpoint,
                                "feedback_pose_cell_xy": feedback_pose,
                                "selected_theta": float(
                                    action.target_theta
                                ),
                                "route_endpoint_theta": route_theta,
                                "feedback_theta": feedback_theta,
                                "pre_snapshot_sha256": (
                                    pre_snapshot_sha256
                                ),
                                "request_id": request_id,
                                "request_sha256": request_sha256,
                                "route_request_id": request_id,
                                "route_request_sha256": request_sha256,
                                "route_result_sha256": (
                                    route_result_sha256
                                ),
                                "feedback_request_id": request_id,
                                "feedback_route_result_sha256": (
                                    route_result_sha256
                                ),
                                "feedback_sha256": feedback_sha256,
                                "post_snapshot_parent_sha256": (
                                    feedback_sha256
                                ),
                                "post_snapshot_sha256": (
                                    post_snapshot_sha256
                                ),
                                "coverage": coverage,
                                "paired_g1_coverage": float(
                                    g1_row["coverage"]
                                ),
                                "safety_violation_count": safety_count,
                                "masked_action_count": masked_count,
                                "candidate_route_mismatch_count": (
                                    mismatch_count
                                ),
                                "planner_success": planner_success,
                                "planner_failure_reason": (
                                    planner_failure_reason
                                ),
                                **timing,
                            }
                        )
                        observation = step.observation
                        step_index += 1
                    if not environment.is_done:
                        raise G3Blocked(
                            "g3_wheel_episode_did_not_terminate"
                        )
                    episode_rows = [
                        row
                        for row in output_rows
                        if row["episode_id"] == episode_id
                    ]
                    if (
                        not episode_rows
                        or episode_rows[-1]["is_terminal"] is not True
                    ):
                        raise G3Blocked("g3_wheel_step_chain")
                finally:
                    environment.close()
    finally:
        policy.train(original_training)

    _evaluate_wheel(
        selected=selected,
        wheel_rows=output_rows,
        g1_rows=g1_rows,
    )
    return output_rows


def _g2_call_index(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for raw in rows:
        row = _require_exact_keys(
            raw,
            _G2_REFERENCE_FIELDS,
            "g3_g2_cross_root_join",
        )
        if (
            row.get("row_kind") != "g2_planning_call"
            or row.get("platform") not in {"wheel", "legged", "hopper"}
        ):
            raise G3Blocked("g3_g2_cross_root_join")
        call_id = _require_nonempty(
            row.get("call_id"),
            "g3_g2_cross_root_join",
        )
        if call_id in result:
            raise G3Blocked("g3_g2_cross_root_join")
        _require_nonempty(row.get("request_id"), "g3_g2_cross_root_join")
        _require_sha256(row.get("request_sha256"), "g3_g2_cross_root_join")
        _require_sha256(row.get("semantic_digest"), "g3_g2_cross_root_join")
        repeat = _require_exact_int(
            row.get("repeat_index"),
            "g3_g2_cross_root_join",
        )
        if repeat >= 5:
            raise G3Blocked("g3_g2_cross_root_join")
        if (
            _require_exact_bool(
                row.get("provider_success"),
                "g3_g2_cross_root_join",
            )
            is not True
            or _require_exact_bool(
                row.get("route_l2_valid"),
                "g3_g2_cross_root_join",
            )
            is not True
            or row.get("timing_contract_id") != TIMING_CONTRACT_ID
        ):
            raise G3Blocked("g3_g2_cross_root_join")
        result[call_id] = row
    return result


def _evaluate_interface(
    *,
    interface_rows: Sequence[Mapping[str, object]],
    g2_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(interface_rows) != 6:
        raise G3Blocked("g3_interface_request_matrix")
    calls = _g2_call_index(g2_rows)
    replay_ids: set[str] = set()
    request_keys: set[tuple[str, str, str]] = set()
    platform_counts = {"legged": 0, "hopper": 0}
    timings: list[float] = []
    for raw in interface_rows:
        row = _require_exact_keys(
            raw,
            _INTERFACE_ROW_FIELDS,
            "g3_interface_row_schema",
        )
        if (
            row.get("row_kind") != "g3_interface_replay"
            or row.get("schema_version")
            != "mid-dual-g3-interface-replay/v1"
            or row.get("scale_profile") != SCALE_PROFILE
        ):
            raise G3Blocked("g3_interface_row_schema")
        _require_nonempty(row.get("run_id"), "g3_interface_row_schema")
        replay_id = _require_nonempty(
            row.get("replay_id"),
            "g3_interface_request_matrix",
        )
        if replay_id in replay_ids:
            raise G3Blocked("g3_interface_request_matrix")
        replay_ids.add(replay_id)
        platform = row.get("platform")
        if platform not in platform_counts:
            raise G3Blocked("g3_interface_request_matrix")
        request_id = _require_nonempty(
            row.get("request_id"),
            "g3_interface_request_matrix",
        )
        request_sha = _require_sha256(
            row.get("request_sha256"),
            "g3_interface_request_matrix",
        )
        key = (str(platform), request_id, request_sha)
        if key in request_keys:
            raise G3Blocked("g3_interface_request_matrix")
        request_keys.add(key)
        references = row.get("g2_reference_call_ids")
        if (
            not isinstance(references, (list, tuple))
            or len(references) != 5
            or len(set(references)) != 5
            or any(not isinstance(item, str) or not item for item in references)
        ):
            raise G3Blocked("g3_g2_cross_root_join")
        reference_rows = [calls.get(str(call_id)) for call_id in references]
        if any(reference is None for reference in reference_rows):
            raise G3Blocked("g3_g2_cross_root_join")
        typed_references = [
            reference for reference in reference_rows if reference is not None
        ]
        if (
            {reference["platform"] for reference in typed_references}
            != {platform}
            or {reference["request_id"] for reference in typed_references}
            != {request_id}
            or {reference["request_sha256"] for reference in typed_references}
            != {request_sha}
            or {reference["repeat_index"] for reference in typed_references}
            != set(range(5))
            or len(
                {
                    reference["semantic_digest"]
                    for reference in typed_references
                }
            )
            != 1
        ):
            raise G3Blocked("g3_g2_cross_root_join")
        semantic = typed_references[0]["semantic_digest"]
        if (
            row.get("g2_semantic_digest") != semantic
            or row.get("replay_semantic_digest") != semantic
            or row.get("timing_contract_id") != TIMING_CONTRACT_ID
        ):
            raise G3Blocked("g3_g2_cross_root_join")
        eligible = _require_exact_bool(
            row.get("formal_input_eligible"),
            "g3_interface_formal_input",
        )
        if not eligible:
            if platform == "hopper":
                raise G3Blocked("g3_hopper_formal_input_ineligible")
            raise G3Blocked("g3_interface_formal_input_ineligible")
        timings.append(_timing_ms(row))
        platform_counts[str(platform)] += 1
    if platform_counts != {"legged": 3, "hopper": 3}:
        raise G3Blocked("g3_interface_request_matrix")
    timing = _timing_statistics(timings)
    return {
        "interface_replay_count": 6,
        "platform_counts": platform_counts,
        "cross_root_join_passed": True,
        "timing": timing,
        "midterm_reduced_passed": timing["midterm_reduced_passed"],
        "final_threshold_reduced_passed": timing[
            "final_threshold_reduced_passed"
        ],
    }


def evaluate_g3_evidence(
    *,
    frozen_manifest: Mapping[str, object],
    wheel_rows: Sequence[Mapping[str, object]],
    g1_rows: Sequence[Mapping[str, object]],
    interface_rows: Sequence[Mapping[str, object]],
    g2_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """从冻结清单和三组 raw rows 独立复算 G3；不信任 stored pass 字段。"""

    try:
        selected = validate_g3_manifest(frozen_manifest)
        wheel = _evaluate_wheel(
            selected=selected,
            wheel_rows=wheel_rows,
            g1_rows=g1_rows,
        )
        interface = _evaluate_interface(
            interface_rows=interface_rows,
            g2_rows=g2_rows,
        )
    except G3Blocked as exc:
        return {
            "schema_version": "mid-dual-g3-summary/v1",
            "scale_profile": SCALE_PROFILE,
            "status": "blocked",
            "formal_evidence_eligible": False,
            "blockers": [exc.reason],
            "g3_midterm_crosscheck_passed": False,
            "g3_final_crosscheck_passed": False,
        }
    midterm = (
        wheel["midterm_reduced_passed"] is True
        and interface["midterm_reduced_passed"] is True
    )
    final = (
        wheel["final_threshold_reduced_passed"] is True
        and interface["final_threshold_reduced_passed"] is True
    )
    return {
        "schema_version": "mid-dual-g3-summary/v1",
        "scale_profile": SCALE_PROFILE,
        "status": "passed" if midterm else "failed",
        "formal_evidence_eligible": True,
        "blockers": [],
        "g3_midterm_crosscheck_passed": midterm,
        "g3_final_crosscheck_passed": final,
        "selected_scenarios": {
            split: list(values) for split, values in selected.items()
        },
        "wheel": wheel,
        "interface": interface,
    }


def _load_json(path: str | Path) -> object:
    return artifact_io.read_json(path)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the reduced midterm G3 closed-loop cross-check",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--frozen-manifest")
    parser.add_argument("--g1-root")
    parser.add_argument("--g2-root")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--mode",
        choices=("preflight", "formal", "resume"),
        default="preflight",
    )
    args = parser.parse_args(argv)
    blockers: list[str] = []
    try:
        config = _load_json(args.config)
        validate_g3_config_payload(config)
    except (OSError, ValueError, TypeError):
        blockers.append("g3_config_missing_or_invalid")
    for name, value in (
        ("frozen_manifest", args.frozen_manifest),
        ("g1_root", args.g1_root),
        ("g2_root", args.g2_root),
    ):
        if value is None:
            blockers.append(f"g3_{name}_missing")
    if args.mode != "preflight" and blockers:
        execution_status = "blocked"
    elif args.mode == "preflight":
        execution_status = "complete"
    else:
        blockers.append("g3_formal_execution_not_materialized")
        execution_status = "blocked"
    print(
        json.dumps(
            {
                "execution_status": execution_status,
                "gate_status": "blocked",
                "formal_evidence_eligible": False,
                "blockers": blockers,
                "run_id": args.run_id,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if args.mode == "preflight" else int(bool(blockers))


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "G3Blocked",
    "RUNNER_ID",
    "SCALE_PROFILE",
    "TIMING_COMPONENT_FIELDS",
    "TIMING_CONTRACT_ID",
    "deterministic_candidate_sha256",
    "deterministic_feedback_sha256",
    "deterministic_request_id",
    "deterministic_request_sha256",
    "deterministic_route_result_sha256",
    "evaluate_g3_evidence",
    "execute_update80_wheel_rows",
    "validate_g3_config_payload",
    "validate_g3_manifest",
]
