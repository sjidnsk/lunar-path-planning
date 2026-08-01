from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from lunar_exploration_ppo.env.frontier import FrontierGenerator
from lunar_exploration_ppo.env.frontier_oracle import audit_frontier_opportunities
from lunar_exploration_ppo.env.map_state import ObservedMapState
from lunar_exploration_ppo.env.scenario import LowResolutionPrior
from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, PoseXYTheta


def _isolated_safe_start_fixture() -> tuple[
    ObservedMapState,
    LowResolutionPrior,
    PoseXYTheta,
]:
    state = ObservedMapState.empty(
        GridGeometry(width=25, height=25, resolution_m=0.5)
    )
    state.observed_mask[3:22, 3:22] = True
    state.confidence[state.observed_mask] = 1.0
    state.traversability[state.observed_mask] = 1.0
    pose = PoseXYTheta(CellXY(12, 12), 1.2)
    state.observed_safe_mask[pose.cell.y, pose.cell.x] = True
    prior = LowResolutionPrior(
        channels=np.zeros((7, 5, 5), dtype=np.float32),
        resolution_m=2.5,
        value_prior_source="constant_neutral/v1",
    )
    return state, prior, pose


def test_double_empty_frontier_emits_one_safe_start_candidate() -> None:
    state, prior, pose = _isolated_safe_start_fixture()

    actions = FrontierGenerator(top_m=16).extract(state, prior, pose)

    assert np.count_nonzero(state.planning_safe_mask) == 1
    assert actions.cells == (pose.cell,)
    assert actions.candidate_count == 1
    assert np.flatnonzero(actions.candidate_mask).tolist() == [0]
    row = actions.frontier_features[0]
    assert row[2] == 0.0
    assert np.array_equal(row[10:13], np.zeros((3,), dtype=np.float32))
    assert row[13] == 1.0
    assert np.allclose(row[14:16], (0.0, 1.0), atol=1.0e-7)
    assert row[18] == 0.0
    assert row[21] == 1.0
    assert actions.diagnostics["start_fallback_activated"] is True
    assert (
        actions.diagnostics["start_fallback_source"]
        == "safe_current_pose_best_of_8_absolute_headings/v1"
    )
    assert actions.diagnostics["start_fallback_heading_index"] == 0
    assert actions.diagnostics["start_fallback_gain_cells"] > 0


def test_oracle_counts_the_same_safe_start_opportunity_once() -> None:
    state, _, pose = _isolated_safe_start_fixture()

    audit = audit_frontier_opportunities(
        state,
        pose,
        planning_safe_mask=state.planning_safe_mask,
        sensor_range_m=20.0,
    )

    assert audit.component_size == 1
    assert audit.unknown_count > 0
    assert audit.opportunity_cells == (pose.cell,)


def test_start_fallback_requires_positive_gain_and_safe_start() -> None:
    no_gain, prior, pose = _isolated_safe_start_fixture()
    no_gain.observed_mask[...] = True
    no_gain.confidence[...] = 1.0
    no_gain.traversability[...] = 1.0
    no_gain_actions = FrontierGenerator(top_m=16).extract(
        no_gain,
        prior,
        pose,
    )

    assert no_gain_actions.cells == ()
    assert no_gain_actions.diagnostics["start_fallback_activated"] is False
    assert no_gain_actions.diagnostics["start_fallback_heading_index"] is None
    assert no_gain_actions.diagnostics["start_fallback_gain_cells"] == 0

    unsafe, prior, pose = _isolated_safe_start_fixture()
    unsafe.observed_safe_mask[pose.cell.y, pose.cell.x] = False
    unsafe_actions = FrontierGenerator(top_m=16).extract(unsafe, prior, pose)

    assert unsafe_actions.cells == ()
    assert unsafe_actions.diagnostics["start_fallback_activated"] is False
    assert unsafe_actions.diagnostics["start_fallback_heading_index"] is None


def test_oracle_and_generator_independently_agree_on_start_eligibility() -> None:
    cases: list[tuple[str, bool]] = [
        ("open_unknown", True),
        ("fully_observed", False),
        ("unsafe_start", False),
        ("observed_blocker_ring", False),
    ]
    for case, expected in cases:
        state, prior, pose = _isolated_safe_start_fixture()
        if case == "fully_observed":
            state.observed_mask[...] = True
            state.confidence[...] = 1.0
            state.traversability[...] = 1.0
        elif case == "unsafe_start":
            state.observed_safe_mask[pose.cell.y, pose.cell.x] = False
        elif case == "observed_blocker_ring":
            state.obstacle[3, 3:22] = True
            state.obstacle[21, 3:22] = True
            state.obstacle[3:22, 3] = True
            state.obstacle[3:22, 21] = True

        actions = FrontierGenerator(top_m=16).extract(state, prior, pose)
        audit = audit_frontier_opportunities(
            state,
            pose,
            planning_safe_mask=state.planning_safe_mask,
            sensor_range_m=20.0,
        )

        assert (actions.cells == (pose.cell,)) is expected, case
        assert (audit.opportunity_cells == (pose.cell,)) is expected, case


def test_oracle_keeps_out_of_bounds_pose_a_safe_empty_audit() -> None:
    state, _, _ = _isolated_safe_start_fixture()
    pose = PoseXYTheta(CellXY(state.geometry.width, 12), 0.0)

    audit = audit_frontier_opportunities(
        state,
        pose,
        planning_safe_mask=state.planning_safe_mask,
        sensor_range_m=20.0,
    )

    assert audit.component_size == 0
    assert audit.opportunity_cells == ()


@pytest.mark.skipif(
    os.environ.get("RUN_G1_FROZEN_INTEGRATION") != "1",
    reason="requires the frozen G1 manifest and three production resets",
)
def test_frozen_g1_failures_reset_to_one_safe_start_candidate() -> None:
    from lunar_exploration_ppo.configs.stage6 import (
        SafetyContract,
        load_stage6_config,
    )
    from lunar_exploration_ppo.env.standard_training import (
        StandardEvaluationEnv,
        build_standard_catalog,
    )
    from lunar_exploration_ppo.eval.midterm_reduced import (
        build_midterm_reduced_job_plan,
        load_frozen_scenario_manifest,
    )
    from lunar_exploration_ppo.env.reachability import reachable_component

    repo_root = Path(__file__).resolve().parents[2]
    formal_root = Path(
        "D:/xunce/out/mid_dual/g1/g1-formal-20260727-0650-cst"
    )
    manifest_path = Path(
        "D:/xunce/inputs/mid_dual/scenarios/37a3e639b4decfaf/manifest.json"
    )
    formal_config = json.loads(
        (formal_root / "config.json").read_text(encoding="utf-8")
    )
    stage6_config = load_stage6_config(
        repo_root / "configs/ppo_highres_frontier_stage6_v1.json"
    )
    catalog = build_standard_catalog(verify_hashes=True)
    frozen = load_frozen_scenario_manifest(
        bundle_root=manifest_path.parent,
        expected_manifest_sha256=formal_config["scenario_manifest"]["sha256"],
    )
    plan = build_midterm_reduced_job_plan(
        catalog=catalog,
        frozen_manifest=frozen,
        cohort="unseen24",
        evaluation_seed_start=2026072700,
    )
    expected = {
        3: ("unseen/scenario-0036", 2026072703, 3, 1331),
        10: ("unseen/scenario-0018", 2026072710, 7, 1331),
        23: ("unseen/scenario-0062", 2026072723, 7, 1331),
    }
    safety_contract = SafetyContract.from_stage6_config(stage6_config)
    for episode_index, (
        scenario_id,
        evaluation_seed,
        heading_index,
        gain_cells,
    ) in expected.items():
        job = plan.jobs[episode_index]
        assert job.scenario_id == scenario_id
        assert job.evaluation_seed == evaluation_seed
        environment = StandardEvaluationEnv(
            catalog,
            split=job.split,
            scenario_ids=(job.scenario_id,),
            safety_contract=safety_contract,
            config_sha256=formal_config["config_sha256"],
        )
        try:
            observation = environment.reset()
            actions = environment.current_action_set
            component = reachable_component(
                environment.observed_state.planning_safe_mask,
                environment.pose.cell,
            )
            assert np.count_nonzero(component) == 1
            assert actions.cells == (environment.pose.cell,)
            assert np.flatnonzero(observation.candidate_mask).tolist() == [0]
            assert actions.diagnostics["start_fallback_activated"] is True
            assert (
                actions.diagnostics["start_fallback_heading_index"]
                == heading_index
            )
            assert (
                actions.diagnostics["start_fallback_gain_cells"]
                == gain_cells
            )
            actual_theta = math.atan2(
                float(actions.frontier_features[0, 14]),
                float(actions.frontier_features[0, 15]),
            )
            expected_theta = math.atan2(
                math.sin(heading_index * math.pi / 4.0),
                math.cos(heading_index * math.pi / 4.0),
            )
            assert actual_theta == pytest.approx(expected_theta, abs=1.0e-7)
            assert environment.is_done is False
            assert environment.needs_policy is True
            assert environment.terminal_reason == "none"
        finally:
            environment.close()
