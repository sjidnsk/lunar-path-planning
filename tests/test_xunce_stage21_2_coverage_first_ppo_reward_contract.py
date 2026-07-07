import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
MODEL_EXPLORER_SRC = str(REPO_ROOT / "model-explorer" / "src")
if MODEL_EXPLORER_SRC not in sys.path:
    sys.path.insert(0, MODEL_EXPLORER_SRC)


def test_stage21_2_evaluates_stage21_1_batch_and_routes_to_stage21_3(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract import (
        run_xunce_stage21_2_coverage_first_ppo_reward_contract,
    )

    collector_root = _write_stage21_1_root(tmp_path, final_coverage=0.995, rollout_steps=4)
    config = _write_config(tmp_path, collector_root)

    summary = run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "implement_stage21_3_ppo_batch_validation"
    assert summary["reward_evaluation_row_count"] == 1
    assert summary["hard_risk_positive_reward_count"] == 0
    assert summary["stage21_2_authorized"] is False
    assert (tmp_path / "out" / "xunce-stage21-2-reward-contract-evaluation.jsonl").is_file()


def test_stage21_2_routes_40_step_below_99pct_to_horizon_scaling(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract import (
        run_xunce_stage21_2_coverage_first_ppo_reward_contract,
    )

    collector_root = _write_stage21_1_root(tmp_path, final_coverage=0.55, rollout_steps=40)
    config = _write_config(tmp_path, collector_root)

    summary = run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "stage21_12_rollout_horizon_or_mission_budget_scaling_for_99pct_coverage"


def test_stage21_2_boundary_flag_hard_fails(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract import (
        run_xunce_stage21_2_coverage_first_ppo_reward_contract,
    )

    collector_root = _write_stage21_1_root(tmp_path)
    config = _write_config(tmp_path, collector_root, runs_new_ppo_update=True)

    summary = run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage21_2_coverage_first_reward_boundary_rejections"
    assert "runs_new_ppo_update" in summary["blocking_reason_codes"]


def test_stage21_2_uses_theta_aware_reward_contract_when_required(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract import (
        run_xunce_stage21_2_coverage_first_ppo_reward_contract,
    )

    collector_root = _write_stage21_1_root(tmp_path, theta_reward_contract=True)
    config = _write_config(
        tmp_path,
        collector_root,
        require_theta_aware_reward_contract=True,
        theta_coverage_denominator_cells=100.0,
    )

    summary = run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    rows = _read_jsonl(tmp_path / "out" / "xunce-stage21-2-reward-contract-evaluation.jsonl")
    assert summary["status"] == "passed"
    assert summary["theta_aware_reward_contract_required"] is True
    assert summary["theta_reward_contract_missing_count"] == 0
    assert rows[0]["theta_aware_reward_contract"] is True
    assert rows[0]["coverage_source"] == "theta_aware_sensor_footprint/v1"
    assert rows[0]["metrics"]["coverage_rate_delta"] == 0.06
    assert rows[0]["candidate_viewpoint"] == [1, 0, 90]
    assert rows[0]["point_only_reward_fallback_used"] is False


def test_stage21_2_rejects_missing_theta_reward_contract_when_required(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract import (
        run_xunce_stage21_2_coverage_first_ppo_reward_contract,
    )

    collector_root = _write_stage21_1_root(tmp_path)
    config = _write_config(tmp_path, collector_root, require_theta_aware_reward_contract=True)

    summary = run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_2_coverage_first_reward_contract"
    assert "theta_aware_reward_contract_missing" in summary["blocking_reason_codes"]


def test_stage21_2_uses_slope_obstacle_theta_reward_contract_when_enabled(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract import (
        run_xunce_stage21_2_coverage_first_ppo_reward_contract,
    )

    collector_root = _write_stage21_1_root(tmp_path, theta_reward_contract=True, slope_reward_contract=True)
    config = _write_config(
        tmp_path,
        collector_root,
        require_theta_aware_reward_contract=True,
        slope_obstacle_aware_theta_reward_enabled=True,
        theta_coverage_denominator_cells=100.0,
    )

    summary = run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    rows = _read_jsonl(tmp_path / "out" / "xunce-stage21-2-reward-contract-evaluation.jsonl")
    assert summary["status"] == "passed"
    assert summary["slope_obstacle_aware_theta_reward_enabled"] is True
    assert summary["slope_obstacle_reward_contract_missing_count"] == 0
    assert rows[0]["slope_obstacle_aware_theta_reward_contract"] is True
    assert rows[0]["coverage_source"] == "endpoint_theta_slope_obstacle_los/v1"
    assert rows[0]["metrics"]["coverage_rate_delta"] == 0.04
    assert rows[0]["obstacle_aware_new_visible_cell_count"] == 4
    assert rows[0]["slope_obstacle_source_hash"] == "slope-source-hash"
    assert rows[0]["platform_contract_hash"] == "platform-hash"
    assert rows[0]["max_traversable_slope_deg"] == 30.0
    assert rows[0]["slope_blocked_source_kind"] == "slope_blocked_as_obstacle_proxy"
    assert rows[0]["unobstructed_theta_reward_fallback_used"] is False


def test_stage21_2_rejects_unobstructed_theta_when_slope_contract_enabled(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract import (
        run_xunce_stage21_2_coverage_first_ppo_reward_contract,
    )

    collector_root = _write_stage21_1_root(tmp_path, theta_reward_contract=True)
    config = _write_config(
        tmp_path,
        collector_root,
        require_theta_aware_reward_contract=True,
        slope_obstacle_aware_theta_reward_enabled=True,
    )

    summary = run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_2_coverage_first_reward_contract"
    assert "slope_obstacle_aware_theta_reward_contract_missing" in summary["blocking_reason_codes"]
    assert "unobstructed_theta_reward_fallback_used" in summary["blocking_reason_codes"]


def test_stage21_2_does_not_treat_obstacle_total_visible_as_new_coverage(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract import (
        run_xunce_stage21_2_coverage_first_ppo_reward_contract,
    )

    collector_root = _write_stage21_1_root(
        tmp_path,
        theta_reward_contract=True,
        slope_reward_contract=True,
        slope_reward_total_visible_only=True,
    )
    config = _write_config(
        tmp_path,
        collector_root,
        require_theta_aware_reward_contract=True,
        slope_obstacle_aware_theta_reward_enabled=True,
    )

    summary = run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert "slope_obstacle_aware_theta_reward_contract_missing" in summary["blocking_reason_codes"]


def test_stage21_2_rejects_20deg_sensitivity_as_default_slope_reward_contract(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract import (
        run_xunce_stage21_2_coverage_first_ppo_reward_contract,
    )

    collector_root = _write_stage21_1_root(
        tmp_path,
        theta_reward_contract=True,
        slope_reward_contract=True,
        slope_max_traversable_slope_deg=20.0,
    )
    config = _write_config(
        tmp_path,
        collector_root,
        require_theta_aware_reward_contract=True,
        slope_obstacle_aware_theta_reward_enabled=True,
    )

    summary = run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert "slope_obstacle_aware_theta_reward_contract_missing" in summary["blocking_reason_codes"]


def test_stage21_2_uses_hybrid_astar_path_cost_contract_when_required(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract import (
        run_xunce_stage21_2_coverage_first_ppo_reward_contract,
    )

    collector_root = _write_stage21_1_root(
        tmp_path,
        theta_reward_contract=True,
        slope_reward_contract=True,
        hybrid_path_cost_contract=True,
    )
    config = _write_config(
        tmp_path,
        collector_root,
        require_theta_aware_reward_contract=True,
        slope_obstacle_aware_theta_reward_enabled=True,
        require_hybrid_astar_path_cost_contract=True,
        theta_coverage_denominator_cells=100.0,
    )

    summary = run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    rows = _read_jsonl(tmp_path / "out" / "xunce-stage21-2-reward-contract-evaluation.jsonl")
    assert summary["status"] == "passed"
    assert summary["hybrid_astar_path_cost_reward_contract_required"] is True
    assert summary["hybrid_astar_path_cost_contract_missing_count"] == 0
    assert rows[0]["hybrid_astar_path_cost_reward_contract"] is True
    assert rows[0]["path_cost_source"] == "hybrid_astar_pose_path/v1"
    assert rows[0]["metrics"]["path_cost_m"] == 2.5
    assert rows[0]["metrics"]["coverage_per_cost"] == 1.6
    assert rows[0]["legacy_grid_astar_path_cost"] == 10.0
    assert rows[0]["hybrid_vs_grid_path_cost_delta"] == -7.5
    assert rows[0]["point_grid_path_cost_fallback_used"] is False
    assert rows[0]["default_astar_replaced"] is False
    assert rows[0]["hybrid_astar_ackermann_feasible_claimed"] is False


def test_stage21_2_rejects_grid_only_path_cost_when_hybrid_required(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract import (
        run_xunce_stage21_2_coverage_first_ppo_reward_contract,
    )

    collector_root = _write_stage21_1_root(tmp_path, theta_reward_contract=True, slope_reward_contract=True)
    config = _write_config(
        tmp_path,
        collector_root,
        require_theta_aware_reward_contract=True,
        slope_obstacle_aware_theta_reward_enabled=True,
        require_hybrid_astar_path_cost_contract=True,
    )

    summary = run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert "hybrid_astar_path_cost_contract_missing" in summary["blocking_reason_codes"]
    assert summary["point_grid_path_cost_fallback_used_count"] == 1


def test_stage21_2_rejects_hybrid_ackermann_claim_when_required(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract import (
        run_xunce_stage21_2_coverage_first_ppo_reward_contract,
    )

    collector_root = _write_stage21_1_root(
        tmp_path,
        theta_reward_contract=True,
        slope_reward_contract=True,
        hybrid_path_cost_contract=True,
        hybrid_ackermann_feasible_claimed=True,
    )
    config = _write_config(
        tmp_path,
        collector_root,
        require_theta_aware_reward_contract=True,
        slope_obstacle_aware_theta_reward_enabled=True,
        require_hybrid_astar_path_cost_contract=True,
    )

    summary = run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert "hybrid_astar_path_cost_contract_missing" in summary["blocking_reason_codes"]


def test_stage21_2_requires_stage21_1_passed(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract import (
        run_xunce_stage21_2_coverage_first_ppo_reward_contract,
    )

    collector_root = _write_stage21_1_root(tmp_path, status="failed")
    config = _write_config(tmp_path, collector_root)

    summary = run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage21_2_required_inputs"
    assert "stage21_1_collector_not_passed" in summary["blocking_reason_codes"]


def test_stage21_2_accepts_stage26_10_terminal_aware_v3_reward_profile(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract import (
        run_xunce_stage21_2_coverage_first_ppo_reward_contract,
    )

    collector_root = _write_stage21_1_root(tmp_path, final_coverage=0.5, dead_end_action_mask_zero=True)
    config = _write_config(
        tmp_path,
        collector_root,
        coverage_first_reward_profile="configs/xunce_stage26_10_terminal_aware_ppo_reward_profile_v3.json",
    )

    summary = run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )
    rows = _read_jsonl(tmp_path / "out" / "xunce-stage21-2-reward-contract-evaluation.jsonl")

    assert summary["status"] == "passed"
    assert summary["profile_version"] == "stage26-10-terminal-aware-v3"
    assert summary["terminal_aware_reward_profile_hash"] == summary["profile_hash"]
    assert summary["collector_canonical_reward_profile_hash"] == "interim-v3"
    assert summary["reward_trainable_false_count"] == 0
    assert rows[0]["trainable"] is True
    assert rows[0]["components"]["incomplete_terminal_penalty_component"] < 0.0
    assert rows[0]["components"]["dead_end_penalty_component"] == -2.0
    assert "terminal_incomplete_below_99pct_target" in rows[0]["reason_codes"]
    assert "dead_end_penalty_applied" in rows[0]["reason_codes"]
    assert rows[0]["terminal_reason"] == "no_hybrid_astar_pose_reachable_candidate_for_action_mask"
    assert rows[0]["dead_end_attribution_source"] == "next_state_action_mask_all_false/v1"


def _write_stage21_1_root(
    tmp_path: Path,
    *,
    final_coverage: float = 0.995,
    rollout_steps: int = 4,
    status: str = "passed",
    theta_reward_contract: bool = False,
    slope_reward_contract: bool = False,
    slope_reward_total_visible_only: bool = False,
    slope_max_traversable_slope_deg: float = 30.0,
    hybrid_path_cost_contract: bool = False,
    hybrid_astar_path_cost: float = 2.5,
    hybrid_ackermann_feasible_claimed: bool = False,
    dead_end_action_mask_zero: bool = False,
) -> Path:
    root = tmp_path / "stage21_1"
    root.mkdir()
    (root / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json").write_text(
        json.dumps(
            {
                "schema_version": "xunce-stage21-1-on-policy-ppo-rollout-collector-summary/v1",
                "status": status,
                "next_required_change": "implement_stage21_2_coverage_first_ppo_reward_contract",
                "profile_hash": "interim-v3",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    transition = {
        "schema_version": "xunce-stage21-1-ppo-transition/v1",
        "transition_id": "s1:0",
        "scenario_id": "s1",
        "step_index": 0,
        "done": True,
        "action_index": 0,
        "trainable": True,
        "reward": 0.1,
        "info": {
            "coverage_rate_delta": 0.1,
            "final_coverage_rate_after_step": final_coverage,
            "path_cost": 10.0,
            "soft_risk_exposure": 1.0,
            "hard_risk_violation": False,
        },
    }
    if dead_end_action_mask_zero:
        transition["info"].update(
            {
                "terminal_reason": "no_hybrid_astar_pose_reachable_candidate_for_action_mask",
                "next_action_mask_true_count": 0,
                "next_action_mask_zero": True,
                "next_grid_action_mask_true_count": 2,
                "dead_end_attribution_source": "next_state_action_mask_all_false/v1",
            }
        )
    if theta_reward_contract:
        transition["info"].update(
            {
                "candidate_viewpoints": [[1, 0, 90]],
                "candidate_theta_deg": [90],
                "theta_new_visible_cell_counts": [6],
                "theta_coverage_hashes": ["theta-hash"],
                "theta_coverage_gain_per_path_costs": [0.6],
                "coverage_source": "theta_aware_sensor_footprint/v1",
            }
        )
    if slope_reward_contract:
        slope_fields = {
            "obstacle_aware_theta_coverage_hashes": ["slope-los-hash"],
            "obstacle_aware_theta_coverage_gain_per_path_costs": [0.4],
            "slope_obstacle_source_hash": "slope-source-hash",
            "platform_contract_hash": "platform-hash",
            "max_traversable_slope_deg": slope_max_traversable_slope_deg,
            "slope_blocked_source_kind": "slope_blocked_as_obstacle_proxy",
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        }
        if slope_reward_total_visible_only:
            slope_fields["obstacle_aware_visible_cell_counts"] = [4]
        else:
            slope_fields["obstacle_aware_new_visible_cell_counts"] = [4]
        transition["info"].update(slope_fields)
    if hybrid_path_cost_contract:
        transition["info"].update(
            {
                "path_cost_source": "hybrid_astar_pose_path/v1",
                "hybrid_astar_path_cost": hybrid_astar_path_cost,
                "hybrid_astar_pose_path_hash": "hybrid-pose-path-hash",
                "hybrid_astar_trajectory_kind": "hybrid_astar_pose_path",
                "legacy_grid_astar_path_cost": 10.0,
                "hybrid_vs_grid_path_cost_delta": hybrid_astar_path_cost - 10.0,
                "default_astar_replaced": False,
                "hybrid_astar_ackermann_feasible_claimed": hybrid_ackermann_feasible_claimed,
            }
        )
    (root / "xunce-stage21-1-ppo-trainable-batch.jsonl").write_text(
        json.dumps(transition, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    episode = {
        "schema_version": "xunce-stage21-1-ppo-rollout-episode/v1",
        "scenario_id": "s1",
        "rollout_steps": rollout_steps,
        "final_coverage_rate": final_coverage,
    }
    (root / "xunce-stage21-1-ppo-rollout-episodes.jsonl").write_text(
        json.dumps(episode, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return root


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_config(tmp_path: Path, collector_root: Path, **overrides) -> Path:
    payload = {
        "schema_version": "xunce-stage21-2-coverage-first-ppo-reward-contract-config/v1",
        "stage21_1_collector_root": str(collector_root),
        "coverage_first_reward_profile": "configs/xunce_stage21_coverage_first_ppo_reward_profile_v1.json",
        "stage21_2_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage21_2_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
