from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_theta_proposal_selects_reachable_probe_when_policy_theta_unreachable() -> None:
    from scripts.xunce_synthetic_exploration_credit import (
        REACHABILITY_THETA_POLICY_ID,
        select_reachable_synthetic_credit_theta,
    )

    result = select_reachable_synthetic_credit_theta(
        policy_theta_deg=12.0,
        reachability_probe_theta_deg=45.0,
        current_theta_deg=0.0,
        theta_step_deg=45.0,
        reachable_theta_degs=[45.0],
    )

    assert result["synthetic_credit_theta_policy_id"] == REACHABILITY_THETA_POLICY_ID
    assert result["synthetic_credit_target_theta_deg"] == 45.0
    assert result["synthetic_credit_theta_selected_proposal_index"] >= 0
    assert result["synthetic_credit_target_skipped_unreachable"] is False
    assert result["synthetic_credit_theta_reachable_proposal_count"] == 1
    assert math.isclose(result["old_behavior_theta_log_prob"], 0.0, rel_tol=0.0, abs_tol=1.0e-9)


def test_theta_proposal_reports_skip_when_no_candidate_theta_is_reachable() -> None:
    from scripts.xunce_synthetic_exploration_credit import select_reachable_synthetic_credit_theta

    result = select_reachable_synthetic_credit_theta(
        policy_theta_deg=12.0,
        reachability_probe_theta_deg=45.0,
        current_theta_deg=0.0,
        theta_step_deg=45.0,
        reachable_theta_degs=[],
    )

    assert result["synthetic_credit_target_theta_deg"] is None
    assert result["synthetic_credit_target_skipped_unreachable"] is True
    assert result["old_behavior_theta_log_prob"] is None


def test_theta_proposal_uses_precomputed_reachable_support() -> None:
    from scripts.xunce_synthetic_exploration_credit import select_reachable_synthetic_credit_theta

    result = select_reachable_synthetic_credit_theta(
        policy_theta_deg=12.0,
        reachability_probe_theta_deg=45.0,
        current_theta_deg=0.0,
        theta_step_deg=45.0,
        proposal_theta_degs=[0.0, 45.0, 90.0],
        reachable_theta_degs=[45.0, 90.0],
    )

    assert result["synthetic_credit_theta_proposals_deg"] == [0.0, 45.0, 90.0]
    assert result["synthetic_credit_target_theta_deg"] == 45.0
    assert result["synthetic_credit_theta_selected_proposal_index"] == 1
    assert result["synthetic_credit_theta_reachable_proposal_count"] == 2
    assert result["old_behavior_theta_log_prob"] == pytest.approx(-math.log(2.0), abs=1.0e-9)


def test_behavior_logprob_can_use_behavior_theta_proposal_logprob() -> None:
    from scripts.xunce_synthetic_exploration_credit import synthetic_credit_behavior_logprob

    result = synthetic_credit_behavior_logprob(
        policy_probs=[0.2, 0.3, 0.5],
        action_index=1,
        target_index=1,
        mixture_probability=0.35,
        theta_log_prob=-0.7,
        behavior_theta_log_prob=-1.2,
    )

    expected_point_prob = (1.0 - 0.35) * 0.3 + 0.35
    assert math.isclose(result["old_policy_log_prob"], math.log(0.3) - 0.7, abs_tol=1e-9)
    assert math.isclose(result["old_behavior_log_prob"], math.log(expected_point_prob) - 1.2, abs_tol=1e-9)
    assert math.isclose(result["old_log_prob"], result["old_behavior_log_prob"], abs_tol=1e-9)
    assert result["old_behavior_theta_log_prob"] == -1.2


def test_stage21_3_rejects_bad_behavior_theta_logprob() -> None:
    import scripts.run_xunce_stage21_3_ppo_batch_validation as s21_3
    from scripts.xunce_synthetic_exploration_credit import synthetic_credit_behavior_logprob

    behavior = synthetic_credit_behavior_logprob(
        policy_probs=[0.2, 0.3, 0.5],
        action_index=1,
        target_index=1,
        mixture_probability=1.0,
        theta_log_prob=-0.7,
        behavior_theta_log_prob=-math.log(2.0),
    )
    row = {
        "action_index": 1,
        "behavior_policy_id": "synthetic_credit_mixture_policy/v1",
        "synthetic_credit_mixture_probability": 1.0,
        "synthetic_credit_target_index": 1,
        "old_theta_log_prob": -0.7,
        **behavior,
        "info": {
            "old_action_probs": [0.2, 0.3, 0.5],
            "synthetic_credit_theta_policy_id": "reachability_theta_proposal_mixture/v1",
            "synthetic_credit_theta_proposals_deg": [0.0, 45.0],
            "synthetic_credit_theta_selected_proposal_index": 1,
            "old_behavior_theta_log_prob": -math.log(2.0),
            **behavior,
        },
    }

    assert s21_3._synthetic_credit_behavior_logprob_missing_count([row]) == 0
    bad = dict(row)
    bad["old_behavior_theta_log_prob"] = -0.1
    bad["info"] = dict(row["info"], old_behavior_theta_log_prob=-0.1)
    assert s21_3._synthetic_credit_behavior_logprob_missing_count([bad]) == 1


def test_stage21_1_continuous_credit_target_uses_reachability_theta_behavior() -> None:
    import torch
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
        synthetic_credit_theta_proposals_deg=[0.0, 45.0, 90.0],
        synthetic_credit_theta_selected_proposal_index=1,
        synthetic_credit_theta_reachable_proposal_count=2,
        synthetic_credit_theta_policy_id="reachability_theta_proposal_mixture/v1",
        synthetic_credit_behavior_theta_log_prob=-math.log(2.0),
        synthetic_credit_mixture_probability=1.0,
    )

    assert detail["action_index"] == 1
    assert detail["selected_theta_deg"] == pytest.approx(45.0, abs=1.0e-4)
    assert detail["synthetic_credit_theta_policy_id"] == "reachability_theta_proposal_mixture/v1"
    assert detail["old_behavior_theta_log_prob"] == pytest.approx(-math.log(2.0), abs=1.0e-9)
    assert detail["old_theta_log_prob"] == pytest.approx(-math.log(2.0), abs=1.0e-9)
    assert math.isclose(
        detail["old_log_prob"],
        detail["old_behavior_point_log_prob"] + detail["old_behavior_theta_log_prob"],
        abs_tol=1.0e-6,
    )


def test_stage21_1_metadata_skips_credit_target_when_no_reachable_theta() -> None:
    import torch
    import scripts.run_xunce_stage21_1_on_policy_ppo_rollout_collector as s21_1

    result = s21_1._synthetic_credit_step_metadata(
        xunce_batch={"candidate_features": torch.zeros((1, 1, 8), dtype=torch.float32)},
        observation_payload={"candidate_feature_names": ["relative_distance"], "candidate_features": [[1.0]]},
        slope_theta_metadata={
            "obstacle_aware_new_visible_cell_counts": [4],
            "obstacle_aware_theta_coverage_gain_per_path_costs": [1.0],
        },
        hybrid_path_metadata={
            "hybrid_astar_reachable_flags": [True],
            "hybrid_astar_path_costs": [5.0],
        },
        synthetic_terrain_metadata={},
        action_mask=(True,),
        sampling_mask=(True,),
        hard_risk_clean_mask=(True,),
        enabled=True,
        score_version="path_efficiency_v2",
    )

    assert result["synthetic_credit_target_index"] is None
    assert result["synthetic_credit_target_skipped_unreachable"] is True


def test_stage21_1_metadata_tries_next_target_when_best_xy_has_no_reachable_theta() -> None:
    import torch
    import scripts.run_xunce_stage21_1_on_policy_ppo_rollout_collector as s21_1

    result = s21_1._synthetic_credit_step_metadata(
        xunce_batch={"candidate_features": torch.zeros((1, 2, 8), dtype=torch.float32)},
        observation_payload={
            "candidate_feature_names": ["relative_distance"],
            "candidate_features": [[1.0], [2.0]],
        },
        slope_theta_metadata={
            "obstacle_aware_new_visible_cell_counts": [10, 5],
            "obstacle_aware_theta_coverage_gain_per_path_costs": [2.0, 1.0],
        },
        hybrid_path_metadata={
            "hybrid_astar_reachable_flags": [True, True],
            "hybrid_astar_path_costs": [5.0, 6.0],
            "hybrid_astar_theta_proposals_deg_by_candidate": [[0.0, 45.0], [90.0, 135.0]],
            "hybrid_astar_reachable_theta_degs_by_candidate": [[], [135.0]],
            "hybrid_astar_reachability_probe_theta_deg": 0.0,
        },
        synthetic_terrain_metadata={},
        action_mask=(True, True),
        sampling_mask=(True, True),
        hard_risk_clean_mask=(True, True),
        enabled=True,
        score_version="coverage_proxy_v1",
    )

    assert result["synthetic_credit_target_index"] == 1
    assert result["synthetic_credit_target_theta_deg"] == 135.0
    assert result["synthetic_credit_theta_selected_proposal_index"] == 1
    assert result["synthetic_credit_theta_reachable_proposal_count"] == 1


def test_stage26_7d_routes_untrusted_stage26_7c_root(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_7d_repair_synthetic_credit_sampler_continuous_theta_reachability as s26

    stage26_7c = tmp_path / "stage26_7c"
    stage26_7c.mkdir()
    _write_json(
        stage26_7c / "xunce-stage26-7c-summary.json",
        {"status": "passed", "next_required_change": "run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot"},
    )
    config = tmp_path / "config.json"
    _write_json(
        config,
        {
            "schema_version": "xunce-stage26-7d-repair-synthetic-credit-sampler-continuous-theta-reachability-config/v1",
            "stage_id": "xunce-stage26-7d-repair-synthetic-credit-sampler-continuous-theta-reachability",
            "stage26_7c_root": str(stage26_7c),
            "run_stage26_chain": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )

    summary = s26.run_xunce_stage26_7d_repair_synthetic_credit_sampler_continuous_theta_reachability(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_7d_required_inputs"
    assert summary["coverage_source"] == "endpoint_theta_slope_obstacle_los/v1"
    assert summary["path_cost_source"] == "hybrid_astar_pose_path/v1"
    assert summary["coverage_denominator_source"] == "main_coverable_cells/v1"
    assert summary["synthetic_source_kind"] == "synthetic_terrain_obstacle_proxy/v1"
    assert summary["action_space_type"] == "hybrid_discrete_xy_continuous_theta/v1"
    assert summary["max_traversable_slope_deg"] == 30.0
    assert summary["hybrid_astar_candidate_eval_workers"] == 4


def test_stage26_7d_uses_short_chain_root_when_default_output_path_is_too_long(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_7d_repair_synthetic_credit_sampler_continuous_theta_reachability as s26

    long_output_root = tmp_path / ("very_long_stage26_7d_output_root_" + "x" * 150)
    config = {
        "hybrid_astar_candidate_eval_workers": 4,
    }

    root = s26._chain_output_root(long_output_root, config, REPO_ROOT)

    assert root == long_output_root.parent / "s26_7d_chain_v1"
    assert root.exists()
    assert s26._worst_case_stage21_1_summary_path_length(root) < s26._worst_case_stage21_1_summary_path_length(
        long_output_root / "r"
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
