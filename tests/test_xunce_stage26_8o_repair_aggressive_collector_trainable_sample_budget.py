import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_reachability_guard_resamples_theta_for_same_candidate() -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    detail = _continuous_detail(action_index=0, selected_theta_deg=10.0)
    updated = runner._apply_selected_continuous_theta_reachability_guard(
        detail,
        sampling_mask=(True, True),
        hybrid_path_metadata=_hybrid_metadata(
            proposals=[[10.0, 45.0], [90.0]],
            reachable=[[45.0], [90.0]],
            flags=[[False, True], [True]],
            costs=[[None, 2.0], [5.0]],
        ),
        sampling_seed=260801,
    )

    assert updated["action_index"] == 0
    assert updated["selected_theta_deg"] == 45.0
    assert updated["selected_theta_resampled_for_reachability"] is True
    assert updated["selected_candidate_resampled_for_reachability"] is False
    assert updated["behavior_policy_id"] == runner.REACHABILITY_GUARD_BEHAVIOR_POLICY_ID
    assert updated["synthetic_credit_theta_policy_id"] == runner.REACHABILITY_GUARD_THETA_POLICY_ID
    assert math.isclose(updated["old_log_prob"], updated["old_behavior_log_prob"], abs_tol=1.0e-9)


def test_reachability_guard_theta_logprob_matches_all_reachable_support() -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    detail = _continuous_detail(action_index=0, selected_theta_deg=10.0)
    updated = runner._apply_selected_continuous_theta_reachability_guard(
        detail,
        sampling_mask=(True, True),
        hybrid_path_metadata=_hybrid_metadata(
            proposals=[[10.0, 45.0, 90.0], [135.0]],
            reachable=[[45.0, 90.0], [135.0]],
            flags=[[False, True, True], [True]],
            costs=[[None, 1.0, 3.0], [5.0]],
        ),
        sampling_seed=260801,
    )

    assert updated["synthetic_credit_theta_reachable_proposal_count"] == 2
    assert updated["old_behavior_theta_log_prob"] == -math.log(2)
    assert updated["selected_theta_deg"] in {45.0, 90.0}


def test_reachability_guard_switches_to_next_reachable_candidate() -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    detail = _continuous_detail(action_index=0, selected_theta_deg=10.0)
    updated = runner._apply_selected_continuous_theta_reachability_guard(
        detail,
        sampling_mask=(True, True),
        hybrid_path_metadata=_hybrid_metadata(
            proposals=[[10.0], [90.0, 135.0]],
            reachable=[[], [135.0]],
            flags=[[False], [False, True]],
            costs=[[None], [None, 3.0]],
        ),
        sampling_seed=260802,
    )

    assert updated["action_index"] == 1
    assert updated["selected_theta_deg"] == 135.0
    assert updated["selected_candidate_resampled_for_reachability"] is True
    assert updated["selected_continuous_theta_resample_point_log_prob_source"] == "uniform_reachable_candidate_support"


def test_reachability_guard_marks_terminal_when_no_candidate_reachable() -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    detail = _continuous_detail(action_index=0, selected_theta_deg=10.0)
    updated = runner._apply_selected_continuous_theta_reachability_guard(
        detail,
        sampling_mask=(True, True),
        hybrid_path_metadata=_hybrid_metadata(
            proposals=[[10.0], [90.0]],
            reachable=[[], []],
            flags=[[False], [False]],
            costs=[[None], [None]],
        ),
        sampling_seed=260803,
    )

    assert updated["selected_pose_reachability_guard_failed"] is True
    assert updated["selected_pose_reachability_guard_failure_reason"] == "no_selected_reachable_pose_candidate_terminal"


def test_stage21_3_accepts_reachability_guard_behavior_logprob() -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as stage21_1
    from scripts import run_xunce_stage21_3_ppo_batch_validation as stage21_3

    detail = _continuous_detail(action_index=0, selected_theta_deg=10.0)
    updated = stage21_1._apply_selected_continuous_theta_reachability_guard(
        detail,
        sampling_mask=(True, True),
        hybrid_path_metadata=_hybrid_metadata(
            proposals=[[10.0, 45.0], [90.0]],
            reachable=[[45.0], [90.0]],
            flags=[[False, True], [True]],
            costs=[[None, 2.0], [5.0]],
        ),
        sampling_seed=260801,
    )
    row = _batch_row_from_detail(updated)

    assert stage21_3._row_has_continuous_theta_action_contract(row) is True
    bad_point = json.loads(json.dumps(row))
    bad_point["old_behavior_point_log_prob"] = 123.0
    bad_point["old_point_log_prob"] = 123.0
    bad_point["old_log_prob"] = 123.0 + bad_point["old_behavior_theta_log_prob"]
    bad_point["old_behavior_log_prob"] = bad_point["old_log_prob"]
    bad_point["info"]["old_behavior_point_log_prob"] = 123.0
    bad_point["info"]["old_point_log_prob"] = 123.0
    bad_point["info"]["old_log_prob"] = bad_point["old_log_prob"]
    bad_point["info"]["old_behavior_log_prob"] = bad_point["old_log_prob"]
    assert stage21_3._row_has_continuous_theta_action_contract(bad_point) is False
    row["old_behavior_theta_log_prob"] = 123.0
    row["old_theta_log_prob"] = 123.0
    row["old_log_prob"] = row["old_behavior_point_log_prob"] + 123.0
    row["old_behavior_log_prob"] = row["old_log_prob"]
    row["info"]["old_behavior_theta_log_prob"] = 123.0
    row["info"]["old_theta_log_prob"] = 123.0
    row["info"]["old_log_prob"] = row["old_log_prob"]
    row["info"]["old_behavior_log_prob"] = row["old_log_prob"]
    assert stage21_3._row_has_continuous_theta_action_contract(row) is False


def test_stage26_8o_rejects_wrong_stage26_8n_route(tmp_path: Path) -> None:
    from scripts import run_xunce_stage26_8o_repair_aggressive_collector_trainable_sample_budget as runner

    config = _write_8o_config(tmp_path, stage26_8n_route="not_expected")
    summary = runner.run_xunce_stage26_8o_repair_aggressive_collector_trainable_sample_budget(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8o_required_inputs"


def test_stage26_8o_routes_to_expand_when_rows_still_short(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage26_8o_repair_aggressive_collector_trainable_sample_budget as runner

    config = _write_8o_config(tmp_path)
    monkeypatch.setattr(runner.stage26_1, "run_xunce_stage26_1_synthetic_terrain_collector_smoke", _fake_stage26_1(80))

    summary = runner.run_xunce_stage26_8o_repair_aggressive_collector_trainable_sample_budget(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["next_required_change"] == "expand_stage26_8o_sample_budget_or_start_pool"
    assert summary["trainable_transition_count"] == 80


def test_stage26_8o_routes_to_rerun_8n_when_collector_repaired(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage26_8o_repair_aggressive_collector_trainable_sample_budget as runner

    config = _write_8o_config(tmp_path)
    monkeypatch.setattr(runner.stage26_1, "run_xunce_stage26_1_synthetic_terrain_collector_smoke", _fake_stage26_1(120))

    summary = runner.run_xunce_stage26_8o_repair_aggressive_collector_trainable_sample_budget(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "rerun_stage26_8n_aggressive_update_sweep_with_repaired_collector"
    assert summary["publishes_checkpoint"] is False
    assert summary["starts_online_canary"] is False


def _continuous_detail(*, action_index: int, selected_theta_deg: float) -> dict:
    logits = [0.0, 1.0]
    probs = [0.2689414213699951, 0.7310585786300049]
    theta_mu = [0.0, 0.0]
    theta_kappa = [1.0, 1.0]
    return {
        "action_index": action_index,
        "selected_base_candidate_index": action_index,
        "selected_theta_deg": selected_theta_deg,
        "selected_theta_rad": math.radians(selected_theta_deg),
        "old_sampling_logits": logits,
        "old_action_probs": probs,
        "theta_mu_rad": theta_mu,
        "theta_kappa": theta_kappa,
        "old_point_log_prob": math.log(probs[action_index]),
        "old_theta_log_prob": -1.0,
        "old_log_prob": math.log(probs[action_index]) - 1.0,
        "old_value": 0.0,
        "selected_probability": probs[action_index],
        "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
    }


def _hybrid_metadata(*, proposals: list[list[float]], reachable: list[list[float]], flags: list[list[bool]], costs: list[list[float | None]]) -> dict:
    return {
        "hybrid_astar_theta_proposals_deg_by_candidate": proposals,
        "hybrid_astar_reachable_theta_degs_by_candidate": reachable,
        "hybrid_astar_theta_probe_reachable_flags_by_candidate": flags,
        "hybrid_astar_theta_probe_path_costs_by_candidate": costs,
    }


def _batch_row_from_detail(detail: dict) -> dict:
    row = {
        "action_index": detail["action_index"],
        "selected_base_candidate_index": detail["selected_base_candidate_index"],
        "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
        "selected_theta_rad": detail["selected_theta_rad"],
        "selected_theta_deg": detail["selected_theta_deg"],
        "base_candidate_set_hash": "base-hash",
        "action_sample_hash": "sample-hash",
        "old_point_log_prob": detail["old_point_log_prob"],
        "old_theta_log_prob": detail["old_theta_log_prob"],
        "old_log_prob": detail["old_log_prob"],
        "old_policy_point_log_prob": detail["old_policy_point_log_prob"],
        "old_policy_theta_log_prob": detail["old_policy_theta_log_prob"],
        "old_policy_log_prob": detail["old_policy_log_prob"],
        "old_behavior_point_log_prob": detail["old_behavior_point_log_prob"],
        "old_behavior_theta_log_prob": detail["old_behavior_theta_log_prob"],
        "old_behavior_log_prob": detail["old_behavior_log_prob"],
        "behavior_policy_id": detail["behavior_policy_id"],
        "synthetic_credit_theta_policy_id": detail["synthetic_credit_theta_policy_id"],
        "synthetic_credit_theta_behavior": detail["synthetic_credit_theta_behavior"],
        "synthetic_credit_theta_proposals_deg": detail["synthetic_credit_theta_proposals_deg"],
        "synthetic_credit_theta_selected_proposal_index": detail["synthetic_credit_theta_selected_proposal_index"],
        "synthetic_credit_theta_reachable_proposal_count": detail["synthetic_credit_theta_reachable_proposal_count"],
        "selected_continuous_theta_resample_point_log_prob_source": detail[
            "selected_continuous_theta_resample_point_log_prob_source"
        ],
        "selected_continuous_theta_reachable_candidate_indices": detail.get(
            "selected_continuous_theta_reachable_candidate_indices"
        ),
        "selected_continuous_theta_guard_original_behavior_policy_id": detail.get(
            "selected_continuous_theta_guard_original_behavior_policy_id"
        ),
    }
    row["info"] = {
        "action_index": row["action_index"],
        "selected_theta_rad": row["selected_theta_rad"],
        "old_sampling_logits": detail["old_sampling_logits"],
        "theta_mu_rad": detail["theta_mu_rad"],
        "theta_kappa": detail["theta_kappa"],
        **{key: value for key, value in row.items() if key != "info"},
    }
    return row


def _write_8o_config(tmp_path: Path, *, stage26_8n_route: str = "expand_stage26_8n_aggressive_sample_budget") -> Path:
    stage26_8n_root = tmp_path / "stage26_8n"
    fixture_root = tmp_path / "fixture"
    stage26_8n_root.mkdir()
    fixture_root.mkdir()
    (stage26_8n_root / "xunce-stage26-8n-summary.json").write_text(
        json.dumps({"status": "failed", "next_required_change": stage26_8n_route}),
        encoding="utf-8",
    )
    config = {
        "schema_version": "xunce-stage26-8o-repair-aggressive-collector-trainable-sample-budget-config/v1",
        "stage_id": "xunce-stage26-8o-repair-aggressive-collector-trainable-sample-budget",
        "stage26_8n_root": str(stage26_8n_root),
        "source_scenario_fixture_root": str(fixture_root),
        "base_stage26_1_config": "configs/xunce_stage26_1_synthetic_terrain_collector_smoke_v1.json",
        "run_stage26_1": True,
        "required_scenario_count": 6,
        "rollout_steps": 20,
        "min_trainable_transition_count": 100,
        "sampling_seed": 260801,
        "scenario_seed_base": 260801,
        "selected_continuous_theta_reachability_guard_enabled": True,
        "selected_continuous_theta_unreachable_resample_policy": "reachable_theta_proposal/v1",
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
        "hybrid_astar_candidate_eval_workers": 4,
        "max_traversable_slope_deg": 30.0,
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _fake_stage26_1(trainable_count: int):
    def run_xunce_stage26_1_synthetic_terrain_collector_smoke(**_kwargs):
        return {
            "status": "passed",
            "next_required_change": "run_stage26_2_synthetic_terrain_ppo_update_smoke",
            "transition_count": trainable_count,
            "batch_row_count": trainable_count,
            "stage21_1_status": "passed",
            "stage21_2_status": "passed",
            "stage21_3_status": "passed",
            "stage21_1_source_roi_expansion_root_match": True,
            "hard_risk_violation_count": 0,
            "mask_violation_count": 0,
            "path_planning_failure_count": 0,
            "open_grid_fallback_count": 0,
            "stage21_1_selected_continuous_theta_unreachable_attempt_count": 12,
            "stage21_1_selected_continuous_theta_resample_success_count": 9,
            "stage21_1_selected_candidate_resample_success_count": 3,
            "stage21_1_selected_pose_unreachable_terminal_count": 0,
            "stage21_1_synthetic_credit_target_selected_count_from_transition_rows": 10,
        }

    return run_xunce_stage26_1_synthetic_terrain_collector_smoke
