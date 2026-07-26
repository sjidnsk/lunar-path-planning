"""G3 闭环计时与跨门强联接合同。"""

from __future__ import annotations

import copy
from contextlib import redirect_stdout
import hashlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_xunce_mid_dual_g3_closed_loop.py"
CONFIG = ROOT / "configs/xunce_mid_dual_g3_closed_loop_v1.json"
SCALE_PROFILE = "midterm_reduced_w8x3_update80/v1"
UPDATE80_CHECKPOINT_SHA256 = (
    "35e04c86f9f973af028fb08f0175d42ab45378d09e1f2b96ee6aad6e4c12b5b5"
)
UPDATE80_POLICY_STATE_SHA256 = (
    "3123e6adde99be41e3bd5cc2f3843068e5416892395926d759c2c6a243ffd381"
)


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mid_dual_g3", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    scripts_path = str(ROOT / "scripts")
    sys.path.insert(0, scripts_path)
    try:
        spec.loader.exec_module(module)
    finally:
        assert sys.path[0] == scripts_path
        sys.path.pop(0)
    return module


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _scenario_id(split: str, index: int) -> str:
    return f"{split}/scenario-{index:02d}/standard-proxy/v1"


def _frozen_manifest() -> dict[str, object]:
    test_q24 = [_scenario_id("test", index) for index in range(24)]
    unseen24 = [_scenario_id("unseen", index) for index in range(24)]
    return {
        "schema_version": "mid-dual-scenario-freeze/v1",
        "completion_status": "complete",
        "cohorts": {
            "test_q24": test_q24,
            "unseen24": unseen24,
            "g3_test_q5": test_q24[:5],
            "g3_unseen5": unseen24[:5],
        },
    }


def _g1_rows(*, coverage: float = 0.99) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, source_split in (("test_q24", "test"), ("unseen24", "unseen")):
        for index in range(24):
            denominator = 10_000
            covered = int(round(coverage * denominator))
            code_sha = _sha("g1-code")
            rows.append(
                {
                    "row_kind": "coverage_episode",
                    "schema_version": (
                        "xunce-mid-dual-g1-coverage-episode/v1"
                    ),
                    "gate_id": "g1",
                    "runner_id": "run_xunce_mid_dual_g1_coverage/v1",
                    "phase_id": "p03" if split == "test_q24" else "p04",
                    "phase_name": split,
                    "scale_profile": SCALE_PROFILE,
                    "run_id": "g1-formal-run",
                    "split": split,
                    "episode_index": index,
                    "episode_id": f"{split}-episode-{index:02d}",
                    "scenario_id": _scenario_id(source_split, index),
                    "lane_id": f"lane-{index % 8}",
                    "source_sha256": code_sha,
                    "config_sha256": _sha("g1-config"),
                    "input_sha256": _sha("g1-input"),
                    "code_sha256": code_sha,
                    "scenario_manifest_sha256": _sha(
                        "frozen-scenario-manifest"
                    ),
                    "checkpoint_sha256": UPDATE80_CHECKPOINT_SHA256,
                    "policy_state_sha256": UPDATE80_POLICY_STATE_SHA256,
                    "denominator_source": (
                        "reachable_observable_free_highres_cells/v1"
                    ),
                    "denominator_algorithm": (
                        "exact_reachable_safe_pose_range_los/v1"
                    ),
                    "denominator_sha256": _sha(
                        f"mask:{source_split}:{index}"
                    ),
                    "denominator_cell_count": denominator,
                    "initial_covered_cell_count": 0,
                    "final_covered_cell_count": covered,
                    "coverage": covered / denominator,
                    "elapsed_ms": 100.0,
                    "steps_executed": 1,
                    "termination_reason": "success_done",
                    "safety_violation_count": 0,
                    "masked_action_count": 0,
                }
            )
    return rows


def _timing_ns(total_ns: int = 100_000_000) -> dict[str, int]:
    component = total_ns // 5
    return {
        "input_validation_ns": component,
        "platform_instantiation_ns": component,
        "search_ns": component,
        "complete_route_validation_ns": component,
        "result_assembly_ns": total_ns - component * 4,
        "total_ns": total_ns,
    }


def _wheel_rows(
    module: ModuleType,
    *,
    coverage: float = 0.99,
    total_ns: int = 100_000_000,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, source_split in (("test_q24", "test"), ("unseen24", "unseen")):
        for index in range(5):
            scenario_id = _scenario_id(source_split, index)
            episode_id = f"g3:{split}:{index:02d}"
            pre = _sha(f"pre:{scenario_id}:0")
            request_id = module.deterministic_request_id(
                scenario_id=scenario_id,
                step_index=0,
                pre_snapshot_sha256=pre,
            )
            candidate_id = f"candidate-{split}-{index:02d}"
            decision_sha = _sha(f"decision:{scenario_id}:0")
            candidate_cell = [index + 1, index + 2]
            candidate_sha = module.deterministic_candidate_sha256(
                scenario_id=scenario_id,
                step_index=0,
                pre_snapshot_sha256=pre,
                decision_sha256=decision_sha,
                candidate_id=candidate_id,
                selected_candidate_cell_xy=candidate_cell,
                selected_theta=0.0,
            )
            request_sha = module.deterministic_request_sha256(
                request_id=request_id,
                candidate_sha256=candidate_sha,
                pre_snapshot_sha256=pre,
            )
            planned_path = [[0, 0], candidate_cell]
            route_sha = module.deterministic_route_result_sha256(
                request_sha256=request_sha,
                planner_success=True,
                planner_failure_reason="none",
                planned_path_cells=planned_path,
                planner_path_length_m=1.0,
                route_endpoint_cell_xy=candidate_cell,
                route_endpoint_theta=0.0,
            )
            feedback_sha = module.deterministic_feedback_sha256(
                route_result_sha256=route_sha,
                feedback_pose_cell_xy=candidate_cell,
                feedback_theta=0.0,
                coverage=coverage,
                safety_violation_count=0,
                masked_action_count=0,
            )
            post = _sha(f"post:{feedback_sha}")
            rows.append(
                {
                    "row_kind": "g3_wheel_step",
                    "schema_version": "mid-dual-g3-wheel-step/v1",
                    "scale_profile": SCALE_PROFILE,
                    "run_id": "g3-formal-run",
                    "split": split,
                    "episode_index": index,
                    "episode_id": episode_id,
                    "scenario_id": scenario_id,
                    "lane_id": f"lane-{index % 8}",
                    "step_id": f"{episode_id}:step:000",
                    "step_index": 0,
                    "is_terminal": True,
                    "termination_reason": "success_done",
                    "decision_sha256": decision_sha,
                    "candidate_id": candidate_id,
                    "candidate_sha256": candidate_sha,
                    "request_candidate_sha256": candidate_sha,
                    "selected_candidate_cell_xy": candidate_cell,
                    "planned_path_cells": planned_path,
                    "planner_path_length_m": 1.0,
                    "route_endpoint_cell_xy": candidate_cell,
                    "feedback_pose_cell_xy": candidate_cell,
                    "selected_theta": 0.0,
                    "route_endpoint_theta": 0.0,
                    "feedback_theta": 0.0,
                    "pre_snapshot_sha256": pre,
                    "request_id": request_id,
                    "request_sha256": request_sha,
                    "route_request_id": request_id,
                    "route_request_sha256": request_sha,
                    "route_result_sha256": route_sha,
                    "feedback_request_id": request_id,
                    "feedback_route_result_sha256": route_sha,
                    "feedback_sha256": feedback_sha,
                    "post_snapshot_parent_sha256": feedback_sha,
                    "post_snapshot_sha256": post,
                    "coverage": coverage,
                    "paired_g1_coverage": coverage,
                    "safety_violation_count": 0,
                    "masked_action_count": 0,
                    "candidate_route_mismatch_count": 0,
                    "planner_success": True,
                    "planner_failure_reason": "none",
                    **_timing_ns(total_ns),
                }
            )
    return rows


def _g2_reference_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for platform in ("legged", "hopper"):
        for index in range(3):
            request_id = f"g2-{platform}-{index:02d}"
            request_sha = _sha(f"g2-request:{platform}:{index}")
            semantic = _sha(f"g2-semantic:{platform}:{index}")
            for repeat in range(5):
                rows.append(
                    {
                        "row_kind": "g2_planning_call",
                        "platform": platform,
                        "request_id": request_id,
                        "request_sha256": request_sha,
                        "call_id": f"{request_id}:repeat:{repeat}",
                        "repeat_index": repeat,
                        "semantic_digest": semantic,
                        "provider_success": True,
                        "route_l2_valid": True,
                        "timing_contract_id": "five-phase-sequential-ns/v1",
                    }
                )
    return rows


def _rebind_route_and_feedback(
    module: ModuleType,
    row: dict[str, object],
) -> None:
    route_sha = module.deterministic_route_result_sha256(
        request_sha256=row["request_sha256"],
        planner_success=row["planner_success"],
        planner_failure_reason=row["planner_failure_reason"],
        planned_path_cells=row["planned_path_cells"],
        planner_path_length_m=row["planner_path_length_m"],
        route_endpoint_cell_xy=row["route_endpoint_cell_xy"],
        route_endpoint_theta=row["route_endpoint_theta"],
    )
    row["route_result_sha256"] = route_sha
    row["feedback_route_result_sha256"] = route_sha
    feedback_sha = module.deterministic_feedback_sha256(
        route_result_sha256=route_sha,
        feedback_pose_cell_xy=row["feedback_pose_cell_xy"],
        feedback_theta=row["feedback_theta"],
        coverage=row["coverage"],
        safety_violation_count=row["safety_violation_count"],
        masked_action_count=row["masked_action_count"],
    )
    row["feedback_sha256"] = feedback_sha
    row["post_snapshot_parent_sha256"] = feedback_sha


def _interface_rows(
    *,
    formal_input_eligible: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for platform in ("legged", "hopper"):
        for index in range(3):
            request_id = f"g2-{platform}-{index:02d}"
            request_sha = _sha(f"g2-request:{platform}:{index}")
            semantic = _sha(f"g2-semantic:{platform}:{index}")
            rows.append(
                {
                    "row_kind": "g3_interface_replay",
                    "schema_version": "mid-dual-g3-interface-replay/v1",
                    "scale_profile": SCALE_PROFILE,
                    "run_id": "g3-formal-run",
                    "replay_id": f"g3-replay-{platform}-{index:02d}",
                    "platform": platform,
                    "request_id": request_id,
                    "request_sha256": request_sha,
                    "g2_reference_call_ids": [
                        f"{request_id}:repeat:{repeat}" for repeat in range(5)
                    ],
                    "g2_semantic_digest": semantic,
                    "replay_semantic_digest": semantic,
                    "timing_contract_id": "five-phase-sequential-ns/v1",
                    "formal_input_eligible": formal_input_eligible,
                    **_timing_ns(),
                }
            )
    return rows


def _evaluate(
    module: ModuleType,
    *,
    wheel_rows: list[dict[str, object]] | None = None,
    g1_rows: list[dict[str, object]] | None = None,
    interface_rows: list[dict[str, object]] | None = None,
    g2_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return module.evaluate_g3_evidence(
        frozen_manifest=_frozen_manifest(),
        wheel_rows=_wheel_rows(module) if wheel_rows is None else wheel_rows,
        g1_rows=_g1_rows() if g1_rows is None else g1_rows,
        interface_rows=(
            _interface_rows() if interface_rows is None else interface_rows
        ),
        g2_rows=_g2_reference_rows() if g2_rows is None else g2_rows,
    )


def test_g3_manifest_is_exactly_five_q24_plus_five_unseen24() -> None:
    module = _module()
    selected = module.validate_g3_manifest(_frozen_manifest())
    assert selected == {
        "test_q24": tuple(_scenario_id("test", index) for index in range(5)),
        "unseen24": tuple(_scenario_id("unseen", index) for index in range(5)),
    }

    duplicated = _frozen_manifest()
    duplicated["cohorts"]["g3_test_q5"][4] = duplicated["cohorts"][
        "g3_test_q5"
    ][0]
    result = module.evaluate_g3_evidence(
        frozen_manifest=duplicated,
        wheel_rows=[],
        g1_rows=[],
        interface_rows=[],
        g2_rows=[],
    )
    assert result["status"] == "blocked"
    assert "g3_frozen_manifest" in result["blockers"]


def test_wheel_step_joins_candidate_request_route_feedback_one_to_one() -> None:
    module = _module()
    assert _evaluate(module)["status"] == "passed"

    rows = _wheel_rows(module)
    rows[0]["feedback_route_result_sha256"] = _sha("wrong-route")
    result = _evaluate(module, wheel_rows=rows)
    assert result["status"] == "blocked"
    assert "g3_wheel_hash_chain" in result["blockers"]

    rows = _wheel_rows(module)
    rows[0]["request_id"] = "self-reported-request"
    result = _evaluate(module, wheel_rows=rows)
    assert result["status"] == "blocked"
    assert "g3_request_id" in result["blockers"]


def test_wheel_gate_requires_ten_of_ten_threshold_successes() -> None:
    module = _module()
    rows = _wheel_rows(module, coverage=0.80)
    rows[0]["coverage"] = 0.7999
    _rebind_route_and_feedback(module, rows[0])
    result = _evaluate(
        module,
        wheel_rows=rows,
        g1_rows=_g1_rows(coverage=0.80),
    )
    assert result["status"] == "failed"
    assert result["wheel"]["coverage_80_count"] == 9
    assert result["g3_midterm_crosscheck_passed"] is False


def test_wheel_g1_paired_coverage_delta_mean_is_at_least_minus_001() -> None:
    module = _module()
    rows = _wheel_rows(module, coverage=0.98)
    for row in rows:
        row["paired_g1_coverage"] = 0.99
    result = _evaluate(module, wheel_rows=rows)
    assert result["wheel"]["paired_g1_coverage_delta_mean"] == -0.01
    assert result["status"] == "passed"

    rows[0]["coverage"] = 0.9799
    _rebind_route_and_feedback(module, rows[0])
    result = _evaluate(module, wheel_rows=rows)
    assert result["status"] == "failed"
    assert result["wheel"]["paired_g1_coverage_delta_mean"] < -0.01


def test_every_planner_call_preserves_the_g2_timing_field_contract() -> None:
    module = _module()
    rows = _wheel_rows(module)
    rows[0]["total_ns"] += 1
    result = _evaluate(module, wheel_rows=rows)
    assert result["status"] == "blocked"
    assert "g3_timing_contract" in result["blockers"]

    interface = _interface_rows()
    interface[0]["search_ns"] = -1
    result = _evaluate(module, interface_rows=interface)
    assert result["status"] == "blocked"
    assert "g3_timing_contract" in result["blockers"]


def test_legged_and_hopper_each_have_three_platform_correct_replays() -> None:
    module = _module()
    result = _evaluate(module)
    assert result["interface"]["platform_counts"] == {
        "legged": 3,
        "hopper": 3,
    }
    assert result["interface"]["cross_root_join_passed"] is True

    interface = _interface_rows()
    interface[0]["request_id"] = interface[1]["request_id"]
    interface[0]["request_sha256"] = interface[1]["request_sha256"]
    interface[0]["g2_reference_call_ids"] = copy.deepcopy(
        interface[1]["g2_reference_call_ids"]
    )
    interface[0]["g2_semantic_digest"] = interface[1][
        "g2_semantic_digest"
    ]
    interface[0]["replay_semantic_digest"] = interface[1][
        "replay_semantic_digest"
    ]
    result = _evaluate(module, interface_rows=interface)
    assert result["status"] == "blocked"
    assert "g3_interface_request_matrix" in result["blockers"]


def test_no_candidate_unreachable_timeout_and_stale_snapshot_fail_closed() -> None:
    module = _module()
    for field, value, blocker in (
        ("planner_success", False, "g3_planner_failure"),
        ("planner_failure_reason", "timeout", "g3_planner_failure"),
        (
            "post_snapshot_parent_sha256",
            _sha("stale"),
            "g3_wheel_hash_chain",
        ),
    ):
        rows = _wheel_rows(module)
        rows[0][field] = value
        if field in {"planner_success", "planner_failure_reason"}:
            _rebind_route_and_feedback(module, rows[0])
        result = _evaluate(module, wheel_rows=rows)
        assert result["status"] == "blocked"
        assert blocker in result["blockers"]


def test_hopper_diagnostic_fixture_cannot_make_g3_formal_pass() -> None:
    module = _module()
    interface = _interface_rows()
    for row in interface:
        if row["platform"] == "hopper":
            row["formal_input_eligible"] = False
    result = _evaluate(module, interface_rows=interface)
    assert result["status"] == "blocked"
    assert "g3_hopper_formal_input_ineligible" in result["blockers"]
    assert result["g3_midterm_crosscheck_passed"] is False
    assert result["g3_final_crosscheck_passed"] is False


def test_g3_config_freezes_exact_scale_counts_thresholds_and_two_phases() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert _module().validate_g3_config_payload(payload) == payload
    assert payload["schema_version"] == "mid-dual-g3-config/v1"
    assert payload["runner_id"] == "run_xunce_mid_dual_g3_closed_loop/v1"
    assert payload["scale_profile"] == SCALE_PROFILE
    assert payload["output_base"] == "D:/xunce/out/mid_dual/g3"
    assert payload["required_phase_ids"] == ["p01", "p02"]
    assert payload["required_phases"] == [
        "wheel_closed_loop",
        "interface_replay",
    ]
    assert payload["formal_counts"] == {
        "wheel_episodes": 10,
        "test_q24": 5,
        "unseen24": 5,
        "legged_replays": 3,
        "hopper_replays": 3,
    }
    assert payload["checkpoint_sha256"] == UPDATE80_CHECKPOINT_SHA256
    assert payload["policy_state_sha256"] == UPDATE80_POLICY_STATE_SHA256


def test_g3_production_uses_exact_ten_without_standard_schedule_padding() -> None:
    module = _module()
    source = inspect.getsource(module.execute_update80_wheel_rows)
    assert "run_standard_evaluation_jobs" not in source
    assert "build_standard_evaluation_schedule" not in source
    assert 'for split in ("test_q24", "unseen24")' in source
    assert "selected = validate_g3_manifest" in source
    assert "g1 = _g1_index" in source


def test_g3_runner_uses_artifact_io_and_missing_preflight_stays_blocked() -> None:
    module = _module()
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in (
        ".read_text(",
        ".write_text(",
        ".read_bytes(",
        ".write_bytes(",
        ".open(",
        ".mkdir(",
        ".exists(",
        ".is_file(",
    ):
        assert forbidden not in source

    stream = io.StringIO()
    with redirect_stdout(stream):
        return_code = module._main(
            [
                "--config",
                str(CONFIG),
                "--run-id",
                "g3-implementation-preflight",
                "--mode",
                "preflight",
            ]
        )
    result = json.loads(stream.getvalue())
    assert return_code == 0
    assert result["execution_status"] == "complete"
    assert result["gate_status"] == "blocked"
    assert result["formal_evidence_eligible"] is False
    assert set(result["blockers"]) == {
        "g3_frozen_manifest_missing",
        "g3_g1_root_missing",
        "g3_g2_root_missing",
    }


def test_timing_diagnostics_have_five_nonoverlapping_ns_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.integrations import (
        path_planner_adapter as adapter_module,
    )
    from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry

    ticks = iter((0, 2, 2, 5, 5, 10, 10, 17, 17, 28))
    monkeypatch.setattr(adapter_module, "perf_counter_ns", lambda: next(ticks))

    result = adapter_module.PathPlannerAdapter(
        GridGeometry(3, 3, 0.5)
    ).validate(
        np.ones((3, 3), dtype=bool),
        CellXY(0, 0),
        CellXY(2, 2),
        0.5,
    )

    timing_names = (
        "input_validation_ns",
        "platform_instantiation_ns",
        "search_ns",
        "complete_route_validation_ns",
        "result_assembly_ns",
    )
    assert result.valid is True
    assert {name: result.diagnostics[name] for name in timing_names} == {
        "input_validation_ns": 2,
        "platform_instantiation_ns": 3,
        "search_ns": 5,
        "complete_route_validation_ns": 7,
        "result_assembly_ns": 11,
    }
    assert result.diagnostics["total_ns"] == 28


def test_timing_diagnostics_sum_to_total_ns() -> None:
    from lunar_exploration_ppo.integrations.path_planner_adapter import (
        PathPlannerAdapter,
    )
    from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry

    result = PathPlannerAdapter(GridGeometry(3, 3, 0.5)).validate(
        np.ones((3, 3), dtype=bool),
        CellXY(0, 0),
        CellXY(2, 2),
        0.5,
    )
    values = tuple(
        result.diagnostics[name]
        for name in (
            "input_validation_ns",
            "platform_instantiation_ns",
            "search_ns",
            "complete_route_validation_ns",
            "result_assembly_ns",
        )
    )
    assert all(type(value) is int and value >= 0 for value in values)
    assert result.diagnostics["total_ns"] == sum(values)


def test_timing_does_not_change_route_cells_length_theta_or_failure_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.integrations import (
        path_planner_adapter as adapter_module,
    )
    from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry

    adapter = adapter_module.PathPlannerAdapter(GridGeometry(3, 3, 0.5))
    mask = np.ones((3, 3), dtype=bool)
    baseline = adapter.validate(mask, CellXY(0, 0), CellXY(2, 2), 0.5)
    ticks = iter((100, 103, 103, 108, 108, 115, 115, 126, 126, 139))
    monkeypatch.setattr(adapter_module, "perf_counter_ns", lambda: next(ticks))

    timed = adapter.validate(mask, CellXY(0, 0), CellXY(2, 2), 0.5)
    assert (
        timed.valid,
        timed.failure_reason,
        timed.failure_classification,
        timed.path_cells,
        timed.path_world,
        timed.target_theta,
        timed.path_length_m,
    ) == (
        baseline.valid,
        baseline.failure_reason,
        baseline.failure_classification,
        baseline.path_cells,
        baseline.path_world,
        baseline.target_theta,
        baseline.path_length_m,
    )


def test_early_failure_records_zero_for_unentered_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.integrations import (
        path_planner_adapter as adapter_module,
    )
    from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry

    ticks = iter((0, 5, 5, 12))
    monkeypatch.setattr(adapter_module, "perf_counter_ns", lambda: next(ticks))

    result = adapter_module.PathPlannerAdapter(
        GridGeometry(3, 3, 0.5)
    ).validate(
        np.ones((3, 3), dtype=bool),
        CellXY(0, 0),
        CellXY(9, 9),
        0.5,
    )
    assert result.valid is False
    assert result.failure_reason == "target_out_of_bounds"
    assert result.diagnostics["input_validation_ns"] == 5
    assert result.diagnostics["platform_instantiation_ns"] == 0
    assert result.diagnostics["search_ns"] == 0
    assert result.diagnostics["complete_route_validation_ns"] == 0
    assert result.diagnostics["result_assembly_ns"] == 7
    assert result.diagnostics["total_ns"] == 12
