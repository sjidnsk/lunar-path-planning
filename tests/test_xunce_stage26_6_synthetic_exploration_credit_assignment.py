from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_synthetic_credit_feature_exposure_reuses_eight_network_slots() -> None:
    from scripts.xunce_synthetic_exploration_credit import (
        SYNTHETIC_CREDIT_FEATURE_NAMES,
        build_synthetic_credit_feature_rows,
    )

    rows = build_synthetic_credit_feature_rows(
        candidate_count=3,
        relative_distances=[1.0, 2.0, 4.0],
        hybrid_reachable_flags=[True, False, True],
        obstacle_aware_new_visible_cell_counts=[4, 0, 8],
        obstacle_aware_gain_per_hybrid_costs=[0.4, None, 1.2],
        hybrid_astar_path_costs=[10.0, None, 40.0],
        synthetic_los_blocker_candidate_counts=[0, 3, 6],
        synthetic_hard_obstacle_candidate_counts=[1, 0, 4],
        risk_values=[0.0, 0.5, 1.0],
    )

    assert SYNTHETIC_CREDIT_FEATURE_NAMES == (
        "relative_distance_norm",
        "hybrid_reachable",
        "obstacle_aware_new_visible_norm",
        "obstacle_aware_gain_per_hybrid_cost_norm",
        "hybrid_astar_path_cost_norm",
        "synthetic_los_blocker_pressure_norm",
        "synthetic_hard_obstacle_pressure_norm",
        "risk_or_clearance_proxy_norm",
    )
    assert len(rows) == 3
    assert all(len(row) == 8 for row in rows)
    assert rows[0][1] == 1.0
    assert rows[1][1] == 0.0
    assert rows[2][3] == 1.0
    assert all(0.0 <= value <= 1.0 for row in rows for value in row)


def test_synthetic_credit_target_respects_masks_and_scores_best_candidate() -> None:
    from scripts.xunce_synthetic_exploration_credit import select_synthetic_credit_target

    rows = [
        [0.2, 1.0, 0.4, 0.5, 0.1, 0.0, 0.0, 0.0],
        [0.1, 1.0, 0.9, 0.9, 0.2, 0.0, 0.0, 0.0],
        [0.1, 1.0, 0.95, 1.0, 0.1, 0.0, 0.0, 0.0],
    ]
    target = select_synthetic_credit_target(
        rows,
        action_mask=[True, True, True],
        sampling_mask=[True, True, False],
        hard_risk_clean_mask=[True, True, True],
        hybrid_reachable_flags=[True, True, True],
    )

    assert target["synthetic_credit_target_index"] == 1
    assert target["synthetic_credit_target_score"] > 0.0
    assert target["synthetic_credit_target_reason"] == "best_synthetic_credit_score"


def test_synthetic_credit_target_requires_explicit_hybrid_reachable_flags() -> None:
    from scripts.xunce_synthetic_exploration_credit import select_synthetic_credit_target

    rows = [[0.1, 1.0, 1.0, 1.0, 0.1, 0.0, 0.0, 0.0]]
    target = select_synthetic_credit_target(
        rows,
        action_mask=[True],
        sampling_mask=[True],
        hard_risk_clean_mask=[True],
        hybrid_reachable_flags=None,
    )

    assert target["synthetic_credit_target_index"] is None
    assert target["synthetic_credit_target_reason"] == "missing_or_mismatched_hybrid_reachable_flags"


def test_behavior_logprob_uses_mixture_probability_for_selected_target() -> None:
    from scripts.xunce_synthetic_exploration_credit import synthetic_credit_behavior_logprob

    result = synthetic_credit_behavior_logprob(
        policy_probs=[0.2, 0.3, 0.5],
        action_index=1,
        target_index=1,
        mixture_probability=0.35,
        theta_log_prob=-0.7,
    )

    expected_point_prob = (1.0 - 0.35) * 0.3 + 0.35
    assert math.isclose(result["old_policy_point_log_prob"], math.log(0.3), rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(result["old_behavior_point_log_prob"], math.log(expected_point_prob), rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(result["old_log_prob"], math.log(expected_point_prob) - 0.7, rel_tol=0.0, abs_tol=1e-9)
    assert result["synthetic_credit_target_selected"] is True


def test_stage21_3_rejects_bad_synthetic_credit_behavior_logprob() -> None:
    import scripts.run_xunce_stage21_3_ppo_batch_validation as s21_3

    row = {
        "action_index": 1,
        "old_log_prob": -10.0,
        "old_theta_log_prob": -0.7,
        "old_behavior_point_log_prob": -1.0,
        "synthetic_credit_mixture_probability": 0.35,
        "synthetic_credit_target_index": 1,
        "behavior_policy_id": "synthetic_credit_mixture_policy/v1",
        "info": {
            "old_action_probs": [0.2, 0.3, 0.5],
            "old_behavior_point_log_prob": -1.0,
            "old_theta_log_prob": -0.7,
            "old_log_prob": -10.0,
            "synthetic_credit_mixture_probability": 0.35,
            "synthetic_credit_target_index": 1,
            "behavior_policy_id": "synthetic_credit_mixture_policy/v1",
        },
    }

    assert s21_3._synthetic_credit_behavior_logprob_missing_count([row]) == 1
    fixed = dict(row)
    fixed["old_behavior_point_log_prob"] = math.log((1.0 - 0.35) * 0.3 + 0.35)
    fixed["old_policy_point_log_prob"] = math.log(0.3)
    fixed["old_policy_log_prob"] = math.log(0.3) - 0.7
    fixed["old_log_prob"] = fixed["old_behavior_point_log_prob"] - 0.7
    fixed["info"] = dict(
        row["info"],
        old_behavior_point_log_prob=fixed["old_behavior_point_log_prob"],
        old_behavior_log_prob=fixed["old_log_prob"],
        old_policy_point_log_prob=fixed["old_policy_point_log_prob"],
        old_policy_log_prob=fixed["old_policy_log_prob"],
        old_log_prob=fixed["old_log_prob"],
    )
    assert s21_3._synthetic_credit_behavior_logprob_missing_count([fixed]) == 0

    mismatched_behavior_total = dict(fixed)
    mismatched_behavior_total["old_behavior_log_prob"] = fixed["old_log_prob"] + 0.1
    mismatched_behavior_total["info"] = dict(fixed["info"], old_behavior_log_prob=fixed["old_log_prob"] + 0.1)
    assert s21_3._synthetic_credit_behavior_logprob_missing_count([mismatched_behavior_total]) == 1


def test_stage21_3_requires_policy_logprob_audit_for_synthetic_credit_behavior() -> None:
    import scripts.run_xunce_stage21_3_ppo_batch_validation as s21_3

    point = math.log((1.0 - 0.35) * 0.3 + 0.35)
    row = {
        "action_index": 1,
        "old_log_prob": point - 0.7,
        "old_theta_log_prob": -0.7,
        "old_behavior_point_log_prob": point,
        "old_behavior_log_prob": point - 0.7,
        "synthetic_credit_mixture_probability": 0.35,
        "synthetic_credit_target_index": 1,
        "behavior_policy_id": "synthetic_credit_mixture_policy/v1",
        "info": {
            "old_action_probs": [0.2, 0.3, 0.5],
            "old_behavior_point_log_prob": point,
            "old_behavior_log_prob": point - 0.7,
            "old_theta_log_prob": -0.7,
            "old_log_prob": point - 0.7,
            "synthetic_credit_mixture_probability": 0.35,
            "synthetic_credit_target_index": 1,
            "behavior_policy_id": "synthetic_credit_mixture_policy/v1",
        },
    }

    assert s21_3._synthetic_credit_behavior_logprob_missing_count([row]) == 1


def test_stage21_3_rejects_synthetic_credit_behavior_when_not_allowed() -> None:
    import scripts.run_xunce_stage21_3_ppo_batch_validation as s21_3

    row = {
        "action_index": 1,
        "old_log_prob": -1.0,
        "old_value": 0.0,
        "reward": 0.0,
        "return": 0.0,
        "advantage": 0.0,
        "info": {},
        "behavior_policy_id": "synthetic_credit_mixture_policy/v1",
    }
    reasons = s21_3._contract_rejections(
        [row],
        audit_rows=[{"transition_id": "t"}],
        splits={"overlap_transition_count": 0},
        config={
            "min_trainable_transition_count": 1,
            "allow_synthetic_credit_behavior_policy": False,
            "require_theta_aware_viewpoint_contract": False,
            "require_theta_aware_reward_contract": False,
            "require_slope_obstacle_aware_theta_reward_contract": False,
            "require_hybrid_astar_path_cost_contract": False,
            "require_synthetic_terrain_contract": False,
            "require_continuous_theta_action_contract": False,
            "max_old_log_prob_recompute_abs_error": 1.0,
        },
    )

    assert "synthetic_credit_behavior_policy_not_allowed" in reasons


def test_stage21_1_sampler_writes_behavior_logprob_for_synthetic_credit_target() -> None:
    import scripts.run_xunce_stage21_1_on_policy_ppo_rollout_collector as s21_1

    assert s21_1._int_or_none("1") == 1
    assert s21_1._int_or_none(None) is None

    class FakeOutput:
        logits = torch.tensor([[0.0, 0.4, 0.1]], dtype=torch.float32)
        masked_logits = torch.tensor([[0.0, 0.4, 0.1]], dtype=torch.float32)
        value = torch.tensor([0.25], dtype=torch.float32)

    class FakeModel:
        def __call__(self, **_: object) -> FakeOutput:
            return FakeOutput()

    detail = s21_1._sample_xunce_action(
        FakeModel(),
        {"candidate_features": torch.zeros((1, 3, 8), dtype=torch.float32)},
        sampling_mask=(True, True, True),
        temperature=1.0,
        synthetic_credit_target_index=1,
        synthetic_credit_mixture_probability=1.0,
    )

    assert detail["action_index"] == 1
    assert detail["behavior_policy_id"] == "synthetic_credit_mixture_policy/v1"
    assert detail["synthetic_credit_target_selected"] is True
    assert math.isclose(detail["old_behavior_point_log_prob"], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(detail["old_log_prob"], detail["old_behavior_log_prob"], rel_tol=0.0, abs_tol=1e-9)
    assert detail["old_policy_log_prob"] < detail["old_behavior_log_prob"]


def test_stage21_1_continuous_credit_target_keeps_theta_policy_logprob() -> None:
    import scripts.run_xunce_stage21_1_on_policy_ppo_rollout_collector as s21_1

    class FakeOutput:
        logits = torch.tensor([[0.0, 0.4, 0.1]], dtype=torch.float32)
        masked_logits = torch.tensor([[0.0, 0.4, 0.1]], dtype=torch.float32)
        value = torch.tensor([0.25], dtype=torch.float32)
        theta_mu_rad = torch.zeros((1, 3), dtype=torch.float32)
        theta_kappa = torch.ones((1, 3), dtype=torch.float32)

    class FakeModel:
        def __call__(self, **_: object) -> FakeOutput:
            return FakeOutput()

    detail = s21_1._sample_xunce_action(
        FakeModel(),
        {"candidate_features": torch.zeros((1, 3, 8), dtype=torch.float32)},
        sampling_mask=(True, True, True),
        temperature=1.0,
        continuous_theta_action_space_enabled=True,
        synthetic_credit_target_index=1,
        synthetic_credit_target_theta_deg=45.0,
        synthetic_credit_mixture_probability=1.0,
    )

    assert detail["action_index"] == 1
    assert detail["synthetic_credit_target_selected"] is True
    assert detail["synthetic_credit_target_theta_deg"] == pytest.approx(45.0)
    assert detail["synthetic_credit_theta_behavior"] == "policy_von_mises_sample/v1"
    assert math.isfinite(detail["old_theta_log_prob"])
    assert math.isfinite(detail["old_log_prob"])
    assert math.isclose(
        detail["old_log_prob"],
        detail["old_behavior_point_log_prob"] + detail["old_theta_log_prob"],
        rel_tol=0.0,
        abs_tol=1.0e-6,
    )


def test_stage21_1_builds_synthetic_credit_step_metadata_and_feature_override() -> None:
    import scripts.run_xunce_stage21_1_on_policy_ppo_rollout_collector as s21_1

    xunce_batch = {"candidate_features": torch.zeros((1, 3, 8), dtype=torch.float32)}
    result = s21_1._synthetic_credit_step_metadata(
        xunce_batch=xunce_batch,
        observation_payload={
            "candidate_feature_names": ["relative_distance", "risk"],
            "candidate_features": [[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]],
        },
        slope_theta_metadata={
            "obstacle_aware_new_visible_cell_counts": [1, 10, 2],
            "obstacle_aware_theta_coverage_gain_per_path_costs": [0.1, 0.9, 0.2],
        },
        hybrid_path_metadata={
            "hybrid_astar_reachable_flags": [True, True, True],
            "hybrid_astar_path_costs": [3.0, 2.0, 1.0],
            "hybrid_astar_reachability_probe_theta_deg": 30.0,
        },
        synthetic_terrain_metadata={
            "synthetic_los_blocker_candidate_counts": [0, 1, 3],
            "synthetic_hard_obstacle_candidate_counts": [0, 0, 2],
            "synthetic_los_blocker_cell_count": 5,
            "synthetic_hard_obstacle_cell_count": 3,
        },
        action_mask=(True, True, True),
        sampling_mask=(True, True, True),
        hard_risk_clean_mask=(True, True, True),
        enabled=True,
    )

    assert result["synthetic_credit_target_index"] == 1
    assert result["synthetic_credit_target_theta_deg"] == pytest.approx(30.0)
    assert result["xunce_batch_feature_semantic_map"]["feature_names"][4] == "hybrid_astar_path_cost_norm"
    assert result["updated_xunce_batch"]["candidate_features"].shape == (1, 3, 8)
    assert float(result["updated_xunce_batch"]["candidate_features"][0, 1, 3]) == pytest.approx(1.0)


def test_stage26_6_routes_to_credit_sampler_when_target_not_selected(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_6_synthetic_exploration_credit_assignment as s26

    stage26_5 = tmp_path / "stage26_5"
    stage26_5.mkdir()
    _write_json(
        stage26_5 / "xunce-stage26-5-summary.json",
        {
            "status": "failed",
            "next_required_change": "repair_stage26_synthetic_exploration_credit_assignment",
            "counterfactual_best_not_directly_credited": True,
            "candidate_feature_signal_missing": True,
            "synthetic_terrain_hash": "hash",
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        },
    )
    config = tmp_path / "config.json"
    _write_json(
        config,
        {
            "schema_version": "xunce-stage26-6-synthetic-exploration-credit-assignment-config/v1",
            "stage_id": "xunce-stage26-6-synthetic-exploration-credit-assignment",
            "stage26_5_root": str(stage26_5),
            "run_stage26_chain": False,
            "min_synthetic_credit_target_selected_count": 4,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )

    summary = s26.run_xunce_stage26_6_synthetic_exploration_credit_assignment(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_6_synthetic_credit_sampler"
    assert summary["publishes_checkpoint"] is False


def test_stage26_6_stage26_3_config_preserves_continuous_theta_and_feature_exposure(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_6_synthetic_exploration_credit_assignment as s26

    base = tmp_path / "stage26_3_base.json"
    _write_json(base, {"schema_version": "stage26-3-test-base"})
    config = {
        "stage26_3_base_config": str(base),
        "required_scenario_count": 2,
        "eval_rollout_steps": 4,
        "hybrid_astar_candidate_eval_workers": 4,
    }

    stage26_3_config = s26._build_stage26_3_config(config, tmp_path / "s26_2")

    assert stage26_3_config["stage26_2_root"] == str(tmp_path / "s26_2")
    assert stage26_3_config["action_space_type"] == "hybrid_discrete_xy_continuous_theta/v1"
    assert stage26_3_config["continuous_theta_action_space_enabled"] is True
    assert stage26_3_config["synthetic_credit_feature_exposure_enabled"] is True
    assert stage26_3_config["hybrid_astar_candidate_eval_workers"] == 4


def test_stage26_6_preserves_stage26_3_specific_repair_route() -> None:
    import scripts.run_xunce_stage26_6_synthetic_exploration_credit_assignment as s26

    route = s26._route(
        boundary_rejections=[],
        input_rejections=[],
        config={"min_synthetic_credit_target_selected_count": 1},
        feature_audit={"candidate_feature_signal_missing": False},
        behavior_audit={"behavior_logprob_recomputable": True},
        target_rows=[{"synthetic_credit_target_selected": True}],
        stage26_1_summary={"status": "passed"},
        stage26_2_summary={"status": "passed"},
        stage26_3_summary={"status": "failed", "next_required_change": "repair_stage26_3_synthetic_eval_safety_regression"},
    )

    assert route == "repair_stage26_3_synthetic_eval_safety_regression"


def test_stage26_6_routes_stage26_3_incomplete_eval_to_feature_repair() -> None:
    import scripts.run_xunce_stage26_6_synthetic_exploration_credit_assignment as s26

    route = s26._route(
        boundary_rejections=[],
        input_rejections=[],
        config={"min_synthetic_credit_target_selected_count": 1},
        feature_audit={"candidate_feature_signal_missing": False},
        behavior_audit={"behavior_logprob_recomputable": True},
        target_rows=[{"synthetic_credit_target_selected": True}],
        stage26_1_summary={"status": "passed"},
        stage26_2_summary={"status": "passed"},
        stage26_3_summary={
            "status": "failed",
            "next_required_change": "rerun_stage26_3_required_inputs",
            "stage21_5_execution_incomplete": True,
            "synthetic_inference_required_field_missing_count": 4,
            "hybrid_path_contract_mismatch_count": 4,
        },
    )

    assert route == "repair_stage26_6_candidate_feature_exposure"


def test_stage26_6_feature_audit_requires_nonzero_candidate_signals(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_6_synthetic_exploration_credit_assignment as s26

    semantic_map = {
        "feature_contract_id": "synthetic_credit_candidate_features/v1",
        "feature_names": [
            "relative_distance_norm",
            "hybrid_reachable",
            "obstacle_aware_new_visible_norm",
            "obstacle_aware_gain_per_hybrid_cost_norm",
            "hybrid_astar_path_cost_norm",
            "synthetic_los_blocker_pressure_norm",
            "synthetic_hard_obstacle_pressure_norm",
            "risk_or_clearance_proxy_norm",
        ],
    }
    empty_signal = {
        "relative_distance_norm": 2,
        "hybrid_reachable": 1,
        "obstacle_aware_new_visible_norm": 0,
        "obstacle_aware_gain_per_hybrid_cost_norm": 0,
        "hybrid_astar_path_cost_norm": 1,
        "synthetic_los_blocker_pressure_norm": 0,
        "synthetic_hard_obstacle_pressure_norm": 0,
        "risk_or_clearance_proxy_norm": 2,
    }
    audit = s26._feature_audit_from_stage26_1(
        tmp_path,
        [{"xunce_batch_feature_semantic_map": semantic_map, "candidate_feature_nonzero_counts": empty_signal}],
    )

    assert audit["candidate_feature_signal_missing"] is True
    assert audit["coverage_signal_row_count"] == 0
    assert audit["synthetic_pressure_signal_row_count"] == 0

    full_signal = dict(empty_signal)
    full_signal["obstacle_aware_gain_per_hybrid_cost_norm"] = 1
    full_signal["synthetic_los_blocker_pressure_norm"] = 1
    audit = s26._feature_audit_from_stage26_1(
        tmp_path,
        [{"xunce_batch_feature_semantic_map": semantic_map, "candidate_feature_nonzero_counts": full_signal}],
    )

    assert audit["candidate_feature_signal_missing"] is False


def test_candidate_pressure_values_do_not_repeat_global_map_counts() -> None:
    from scripts.xunce_synthetic_exploration_credit import candidate_pressure_values

    assert candidate_pressure_values(None, 3) == [0.0, 0.0, 0.0]
    assert candidate_pressure_values([1, 2, 3], 3) == [1.0, 2.0, 3.0]
    assert candidate_pressure_values([7], 3) == [0.0, 0.0, 0.0]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
