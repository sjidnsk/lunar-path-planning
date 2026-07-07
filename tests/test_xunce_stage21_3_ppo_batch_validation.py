import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage21_3_validates_batch_and_routes_to_stage21_4(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path)
    config = _write_config(tmp_path, stage21_1, stage21_2)

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "implement_stage21_4_tiny_ppo_update_smoke"
    assert summary["trainable_transition_count"] == 2
    assert summary["return_finite_count"] == 2
    assert summary["advantage_finite_count"] == 2
    assert summary["invalid_action_count"] == 0
    assert summary["hard_risk_violation_count"] == 0
    assert summary["duplicate_transition_id_count"] == 0
    assert summary["duplicate_reward_transition_id_count"] == 0
    assert summary["reward_trainable_false_count"] == 0
    assert summary["runs_new_ppo_update"] is False
    batch_path = tmp_path / "out" / "xunce-stage21-3-ppo-trainable-batch.jsonl"
    rows = _read_jsonl(batch_path)
    assert {row["stage21_3_split"] for row in rows} <= {"train", "validation"}
    assert all("advantage_normalization_scope" in row for row in rows)


def test_stage21_3_accepts_terminal_aware_v3_nonblocking_reward_reason_codes(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path)
    reward_path = stage21_2 / "xunce-stage21-2-reward-contract-evaluation.jsonl"
    rewards = _read_jsonl(reward_path)
    for reward in rewards:
        reward["profile_version"] = "stage26-10-terminal-aware-v3"
        reward["profile_hash"] = "terminal-aware-hash"
        reward["components"] = {
            "incomplete_terminal_penalty_component": -0.5,
            "dead_end_penalty_component": -2.0,
        }
        reward["reason_codes"] = ["terminal_incomplete_below_99pct_target", "dead_end_penalty_applied"]
        reward["terminal_reason"] = "no_hybrid_astar_pose_reachable_candidate_for_action_mask"
        reward["dead_end_attribution_source"] = "next_state_action_mask_all_false/v1"
    _write_jsonl(reward_path, rewards)
    summary_path = stage21_2 / "xunce-stage21-2-coverage-first-reward-summary.json"
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_payload["profile_hash"] = "terminal-aware-hash"
    summary_payload["profile_version"] = "stage26-10-terminal-aware-v3"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False), encoding="utf-8")
    config = _write_config(tmp_path, stage21_1, stage21_2)

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )
    batch_rows = _read_jsonl(tmp_path / "out" / "xunce-stage21-3-ppo-trainable-batch.jsonl")

    assert summary["status"] == "passed"
    assert summary["reward_trainable_false_count"] == 0
    assert summary["nonblocking_reward_reason_codes"] == [
        "terminal_incomplete_below_99pct_target",
        "dead_end_penalty_applied",
        "coverage_per_cost_component_inactive_without_coverage_gain",
    ]
    assert summary["nonblocking_reward_reason_code_count"] == 4
    assert summary["hard_risk_reward_rejection_count"] == 0
    assert batch_rows[0]["reward_reason_codes"] == [
        "terminal_incomplete_below_99pct_target",
        "dead_end_penalty_applied",
    ]
    assert batch_rows[0]["reward_components"]["dead_end_penalty_component"] == -2.0


def test_stage21_3_boundary_flag_hard_fails(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path)
    config = _write_config(tmp_path, stage21_1, stage21_2, starts_online_canary=True)

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage21_3_ppo_batch_validation_boundary_rejections"
    assert "starts_online_canary" in summary["blocking_reason_codes"]


def test_stage21_3_detects_reward_lineage_mismatch(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path, omit_second_reward=True)
    config = _write_config(tmp_path, stage21_1, stage21_2)

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_3_batch_lineage"
    assert "missing_reward_transition_rows" in summary["blocking_reason_codes"]


def test_stage21_3_detects_duplicate_transition_ids(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path, duplicate_transition=True)
    config = _write_config(tmp_path, stage21_1, stage21_2)

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_3_duplicate_lineage_ids"
    assert "duplicate_transition_ids" in summary["blocking_reason_codes"]


def test_stage21_3_detects_episode_without_terminal(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path, no_terminal=True)
    config = _write_config(tmp_path, stage21_1, stage21_2)

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_3_episode_boundary_contract"
    assert "episode_terminal_count_mismatch" in summary["blocking_reason_codes"]


def test_stage21_3_detects_nontrainable_reward(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path, reward_trainable=False)
    config = _write_config(tmp_path, stage21_1, stage21_2)

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_3_reward_trainability_contract"
    assert "reward_trainable_false" in summary["blocking_reason_codes"]


def test_stage21_3_detects_invalid_action_mask_and_negative_action(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path, negative_action=True)
    config = _write_config(tmp_path, stage21_1, stage21_2)

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_3_ppo_batch_contract"
    assert "invalid_action_detected" in summary["blocking_reason_codes"]


def test_stage21_3_gates_old_log_prob_recompute_error(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path, old_log_error=0.5)
    config = _write_config(tmp_path, stage21_1, stage21_2, max_old_log_prob_recompute_abs_error=1.0e-6)

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_3_ppo_batch_contract"
    assert "old_log_prob_recompute_error_exceeds_threshold" in summary["blocking_reason_codes"]


def test_stage21_3_requires_theta_viewpoint_contract_when_enabled(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path)
    config = _write_config(tmp_path, stage21_1, stage21_2, require_theta_aware_viewpoint_contract=True)

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_3_ppo_batch_contract"
    assert "theta_viewpoint_contract_missing" in summary["blocking_reason_codes"]


def test_stage21_3_accepts_theta_viewpoint_contract_when_enabled(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path, theta_viewpoint_contract=True)
    config = _write_config(tmp_path, stage21_1, stage21_2, require_theta_aware_viewpoint_contract=True)

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["theta_aware_viewpoint_contract_required"] is True
    assert summary["theta_aware_viewpoint_contract_missing_count"] == 0


def test_stage21_3_requires_theta_reward_contract_when_enabled(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path, theta_viewpoint_contract=True)
    config = _write_config(
        tmp_path,
        stage21_1,
        stage21_2,
        require_theta_aware_viewpoint_contract=True,
        require_theta_aware_reward_contract=True,
    )

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_3_ppo_batch_contract"
    assert "theta_reward_contract_missing" in summary["blocking_reason_codes"]


def test_stage21_3_accepts_theta_reward_contract_when_enabled(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path, theta_viewpoint_contract=True, theta_reward_contract=True)
    config = _write_config(
        tmp_path,
        stage21_1,
        stage21_2,
        require_theta_aware_viewpoint_contract=True,
        require_theta_aware_reward_contract=True,
    )

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    rows = _read_jsonl(tmp_path / "out" / "xunce-stage21-3-ppo-trainable-batch.jsonl")
    assert summary["status"] == "passed"
    assert summary["theta_aware_reward_contract_required"] is True
    assert summary["theta_aware_reward_contract_missing_count"] == 0
    assert rows[0]["coverage_source"] == "theta_aware_sensor_footprint/v1"


def test_stage21_3_rejects_theta_reward_contract_mismatched_to_selected_action(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(
        tmp_path,
        theta_viewpoint_contract=True,
        theta_reward_contract=True,
        theta_reward_viewpoint=[2, 0, 90],
    )
    config = _write_config(
        tmp_path,
        stage21_1,
        stage21_2,
        require_theta_aware_viewpoint_contract=True,
        require_theta_aware_reward_contract=True,
    )

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert "theta_reward_contract_missing" in summary["blocking_reason_codes"]


def test_stage21_3_rejects_theta_reward_without_transition_viewpoint_contract(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path, theta_viewpoint_contract=False, theta_reward_contract=True)
    config = _write_config(tmp_path, stage21_1, stage21_2, require_theta_aware_reward_contract=True)

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert "theta_reward_contract_missing" in summary["blocking_reason_codes"]


def test_stage21_3_rejects_theta_reward_without_denominator(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(
        tmp_path,
        theta_viewpoint_contract=True,
        theta_reward_contract=True,
        theta_reward_omit_denominator=True,
    )
    config = _write_config(
        tmp_path,
        stage21_1,
        stage21_2,
        require_theta_aware_viewpoint_contract=True,
        require_theta_aware_reward_contract=True,
    )

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert "theta_reward_contract_missing" in summary["blocking_reason_codes"]


def test_stage21_3_accepts_slope_obstacle_theta_reward_contract_when_required(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(
        tmp_path,
        theta_viewpoint_contract=True,
        theta_reward_contract=True,
        slope_reward_contract=True,
    )
    config = _write_config(
        tmp_path,
        stage21_1,
        stage21_2,
        require_theta_aware_viewpoint_contract=True,
        require_theta_aware_reward_contract=True,
        require_slope_obstacle_aware_theta_reward_contract=True,
    )

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    rows = _read_jsonl(tmp_path / "out" / "xunce-stage21-3-ppo-trainable-batch.jsonl")
    assert summary["status"] == "passed"
    assert summary["slope_obstacle_aware_theta_reward_contract_required"] is True
    assert summary["slope_obstacle_aware_theta_reward_contract_missing_count"] == 0
    assert rows[0]["coverage_source"] == "endpoint_theta_slope_obstacle_los/v1"
    assert rows[0]["slope_obstacle_source_hash"] == "slope-source-hash"
    assert rows[0]["platform_contract_hash"] == "platform-hash"
    assert rows[0]["max_traversable_slope_deg"] == 30.0


def test_stage21_3_preserves_synthetic_credit_path_efficiency_fields(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation
    from scripts.xunce_synthetic_exploration_credit import synthetic_credit_behavior_logprob

    stage21_1, stage21_2 = _write_roots(tmp_path)
    transition_path = stage21_1 / "xunce-stage21-1-ppo-trainable-batch.jsonl"
    transitions = _read_jsonl(transition_path)
    for transition in transitions:
        logprob = synthetic_credit_behavior_logprob(
            policy_probs=[1.0],
            action_index=0,
            target_index=0,
            mixture_probability=1.0,
        )
        fields = {
            **logprob,
            "synthetic_credit_score": 0.42,
            "synthetic_credit_score_version": "path_efficiency_v2",
            "path_efficiency_filter_relaxed": False,
            "selected_target_hybrid_cost_norm": 0.33,
            "selected_target_gain_per_cost_norm": 0.77,
        }
        transition.update(fields)
        transition["info"] = {
            **transition["info"],
            **fields,
            "old_action_probs": [1.0],
        }
    _write_jsonl(transition_path, transitions)
    config = _write_config(
        tmp_path,
        stage21_1,
        stage21_2,
        allow_synthetic_credit_behavior_policy=True,
    )

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    rows = _read_jsonl(tmp_path / "out" / "xunce-stage21-3-ppo-trainable-batch.jsonl")
    assert summary["status"] == "passed"
    assert rows[0]["synthetic_credit_score_version"] == "path_efficiency_v2"
    assert rows[0]["path_efficiency_filter_relaxed"] is False
    assert rows[0]["selected_target_hybrid_cost_norm"] == 0.33
    assert rows[0]["selected_target_gain_per_cost_norm"] == 0.77


def test_stage21_3_rejects_unobstructed_theta_reward_when_slope_required(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path, theta_viewpoint_contract=True, theta_reward_contract=True)
    config = _write_config(
        tmp_path,
        stage21_1,
        stage21_2,
        require_theta_aware_viewpoint_contract=True,
        require_theta_aware_reward_contract=True,
        require_slope_obstacle_aware_theta_reward_contract=True,
    )

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_3_ppo_batch_contract"
    assert "slope_obstacle_aware_theta_reward_contract_missing" in summary["blocking_reason_codes"]


def test_stage21_3_rejects_20deg_sensitivity_reward_when_slope_required(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(
        tmp_path,
        theta_viewpoint_contract=True,
        theta_reward_contract=True,
        slope_reward_contract=True,
        slope_max_traversable_slope_deg=20.0,
    )
    config = _write_config(
        tmp_path,
        stage21_1,
        stage21_2,
        require_theta_aware_viewpoint_contract=True,
        require_theta_aware_reward_contract=True,
        require_slope_obstacle_aware_theta_reward_contract=True,
    )

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert "slope_obstacle_aware_theta_reward_contract_missing" in summary["blocking_reason_codes"]


def test_stage21_3_accepts_hybrid_astar_path_cost_contract_when_required(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(
        tmp_path,
        theta_viewpoint_contract=True,
        theta_reward_contract=True,
        slope_reward_contract=True,
        hybrid_path_cost_contract=True,
    )
    config = _write_config(
        tmp_path,
        stage21_1,
        stage21_2,
        require_theta_aware_viewpoint_contract=True,
        require_theta_aware_reward_contract=True,
        require_slope_obstacle_aware_theta_reward_contract=True,
        require_hybrid_astar_path_cost_contract=True,
    )

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    rows = _read_jsonl(tmp_path / "out" / "xunce-stage21-3-ppo-trainable-batch.jsonl")
    assert summary["status"] == "passed"
    assert summary["hybrid_astar_path_cost_contract_required"] is True
    assert summary["hybrid_astar_path_cost_contract_missing_count"] == 0
    assert rows[0]["path_cost_source"] == "hybrid_astar_pose_path/v1"
    assert rows[0]["hybrid_astar_path_cost"] == 2.5
    assert rows[0]["legacy_grid_astar_path_cost"] == 10.0
    assert rows[0]["point_grid_path_cost_fallback_used"] is False


def test_stage21_3_rejects_grid_only_path_cost_when_hybrid_required(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(
        tmp_path,
        theta_viewpoint_contract=True,
        theta_reward_contract=True,
        slope_reward_contract=True,
    )
    config = _write_config(
        tmp_path,
        stage21_1,
        stage21_2,
        require_theta_aware_viewpoint_contract=True,
        require_theta_aware_reward_contract=True,
        require_slope_obstacle_aware_theta_reward_contract=True,
        require_hybrid_astar_path_cost_contract=True,
    )

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_3_ppo_batch_contract"
    assert "hybrid_astar_path_cost_contract_missing" in summary["blocking_reason_codes"]


def test_stage21_3_rejects_hybrid_ackermann_claim_when_hybrid_required(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(
        tmp_path,
        theta_viewpoint_contract=True,
        theta_reward_contract=True,
        slope_reward_contract=True,
        hybrid_path_cost_contract=True,
        hybrid_ackermann_feasible_claimed=True,
    )
    config = _write_config(
        tmp_path,
        stage21_1,
        stage21_2,
        require_theta_aware_viewpoint_contract=True,
        require_theta_aware_reward_contract=True,
        require_slope_obstacle_aware_theta_reward_contract=True,
        require_hybrid_astar_path_cost_contract=True,
    )

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert "hybrid_astar_path_cost_contract_missing" in summary["blocking_reason_codes"]


def _write_roots(
    tmp_path: Path,
    *,
    omit_second_reward: bool = False,
    duplicate_transition: bool = False,
    no_terminal: bool = False,
    reward_trainable: bool = True,
    negative_action: bool = False,
    old_log_error: float = 0.0,
    theta_viewpoint_contract: bool = False,
    theta_reward_contract: bool = False,
    slope_reward_contract: bool = False,
    slope_max_traversable_slope_deg: float = 30.0,
    theta_reward_viewpoint: list[int] | None = None,
    theta_reward_omit_denominator: bool = False,
    hybrid_path_cost_contract: bool = False,
    hybrid_ackermann_feasible_claimed: bool = False,
) -> tuple[Path, Path]:
    stage21_1 = tmp_path / "stage21_1"
    stage21_2 = tmp_path / "stage21_2"
    stage21_1.mkdir()
    stage21_2.mkdir()
    (stage21_1 / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json").write_text(
        json.dumps(
            {
                "schema_version": "xunce-stage21-1-on-policy-ppo-rollout-collector-summary/v1",
                "status": "passed",
                "next_required_change": "implement_stage21_2_coverage_first_ppo_reward_contract",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (stage21_2 / "xunce-stage21-2-coverage-first-reward-summary.json").write_text(
        json.dumps(
            {
                "schema_version": "xunce-stage21-2-coverage-first-reward-summary/v1",
                "status": "passed",
                "next_required_change": "implement_stage21_3_ppo_batch_validation",
                "profile_hash": "reward-hash",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    transitions = [
        _transition(
            "s1:0",
            step=0,
            done=False,
            reward=1.0,
            next_observation={"action_mask": [True]},
            old_log_error=old_log_error,
            theta_viewpoint_contract=theta_viewpoint_contract,
        ),
        _transition(
            "s1:1" if not duplicate_transition else "s1:0",
            step=1,
            done=not no_terminal,
            reward=2.0,
            next_observation=None if not no_terminal else {"action_mask": [True]},
            negative_action=negative_action,
            theta_viewpoint_contract=theta_viewpoint_contract,
        ),
    ]
    _write_jsonl(stage21_1 / "xunce-stage21-1-ppo-trainable-batch.jsonl", transitions)
    rewards = [
        _reward(
            "s1:0",
            step=0,
            done=False,
            reward=1.0,
            trainable=reward_trainable,
            theta_reward_contract=theta_reward_contract,
            slope_reward_contract=slope_reward_contract,
            slope_max_traversable_slope_deg=slope_max_traversable_slope_deg,
            theta_reward_viewpoint=theta_reward_viewpoint,
            theta_reward_omit_denominator=theta_reward_omit_denominator,
            hybrid_path_cost_contract=hybrid_path_cost_contract,
            hybrid_ackermann_feasible_claimed=hybrid_ackermann_feasible_claimed,
        )
    ]
    if not omit_second_reward:
        rewards.append(
            _reward(
                "s1:1",
                step=1,
                done=not no_terminal,
                reward=2.0,
                trainable=reward_trainable,
                theta_reward_contract=theta_reward_contract,
                slope_reward_contract=slope_reward_contract,
                slope_max_traversable_slope_deg=slope_max_traversable_slope_deg,
                theta_reward_viewpoint=theta_reward_viewpoint,
                theta_reward_omit_denominator=theta_reward_omit_denominator,
                hybrid_path_cost_contract=hybrid_path_cost_contract,
                hybrid_ackermann_feasible_claimed=hybrid_ackermann_feasible_claimed,
            )
        )
    _write_jsonl(stage21_2 / "xunce-stage21-2-reward-contract-evaluation.jsonl", rewards)
    return stage21_1, stage21_2


def _transition(
    transition_id: str,
    *,
    step: int,
    done: bool,
    reward: float,
    next_observation,
    negative_action: bool = False,
    old_log_error: float = 0.0,
    theta_viewpoint_contract: bool = False,
) -> dict:
    action_index = -1 if negative_action else 0
    candidate_cells = [[1, 0, 0]] if theta_viewpoint_contract else [[1, 0]]
    info = {
        "action_mask": [True],
        "candidate_cells": candidate_cells,
        "sampling_mask": [True],
        "hard_risk_clean_mask": [True],
        "hard_risk_violation": False,
        "old_log_prob_recompute_abs_error": old_log_error,
    }
    if theta_viewpoint_contract:
        info["candidate_theta_deg"] = [0]
        info["candidate_viewpoints"] = candidate_cells
    return {
        "schema_version": "xunce-stage21-1-ppo-transition/v1",
        "transition_id": transition_id,
        "scenario_id": "s1",
        "step_index": step,
        "observation": {"action_mask": [True], "candidate_cells": candidate_cells},
        "xunce_batch": {"action_mask": {"shape": [1, 1], "dtype": "bool", "values": [[True]]}},
        "action_index": action_index,
        "old_log_prob": -0.1,
        "old_value": 0.5,
        "reward": reward,
        "next_observation": next_observation,
        "next_xunce_batch": None if done else {"action_mask": {"shape": [1, 1], "dtype": "bool", "values": [[True]]}},
        "done": done,
        "trainable": True,
        "info": info,
    }


def _reward(
    transition_id: str,
    *,
    step: int,
    done: bool,
    reward: float,
    trainable: bool,
    theta_reward_contract: bool = False,
    slope_reward_contract: bool = False,
    slope_max_traversable_slope_deg: float = 30.0,
    theta_reward_viewpoint: list[int] | None = None,
    theta_reward_omit_denominator: bool = False,
    hybrid_path_cost_contract: bool = False,
    hybrid_ackermann_feasible_claimed: bool = False,
) -> dict:
    row = {
        "schema_version": "xunce-stage21-2-reward-contract-evaluation/v1",
        "transition_id": transition_id,
        "scenario_id": "s1",
        "step_index": step,
        "done": done,
        "reward": reward,
        "components": {},
        "trainable": trainable,
        "reason_codes": [] if trainable else ["hard_risk_rejected"],
        "profile_id": "profile",
        "profile_version": "stage21-coverage-first-v1",
        "profile_hash": "reward-hash",
    }
    if theta_reward_contract:
        viewpoint = theta_reward_viewpoint or [1, 0, 0]
        coverage_source = "endpoint_theta_slope_obstacle_los/v1" if slope_reward_contract else "theta_aware_sensor_footprint/v1"
        row.update(
            {
                "theta_aware_reward_contract": True,
                "coverage_source": coverage_source,
                "candidate_viewpoint": viewpoint,
                "candidate_theta_deg": viewpoint[2],
                "theta_new_visible_cell_count": 3,
                "theta_coverage_hash": "theta-hash",
                "theta_coverage_gain_per_path_cost": 0.3,
                "metrics": {"coverage_rate_delta": 0.03},
                "point_only_reward_fallback_used": False,
            }
        )
        if slope_reward_contract:
            row.update(
                {
                    "slope_obstacle_aware_theta_reward_contract": True,
                    "obstacle_aware_new_visible_cell_count": 3,
                    "obstacle_aware_theta_coverage_hash": "slope-los-hash",
                    "obstacle_aware_theta_coverage_gain_per_path_cost": 0.3,
                    "obstacle_aware_theta_coverage_denominator_cells": 100.0,
                    "slope_obstacle_source_hash": "slope-source-hash",
                    "platform_contract_hash": "platform-hash",
                    "max_traversable_slope_deg": slope_max_traversable_slope_deg,
                    "slope_blocked_source_kind": "slope_blocked_as_obstacle_proxy",
                    "unobstructed_theta_reward_fallback_used": False,
                }
            )
        if hybrid_path_cost_contract:
            metrics = dict(row.get("metrics") or {})
            metrics["path_cost_m"] = 2.5
            metrics["coverage_per_cost"] = 1.2
            row.update(
                {
                    "hybrid_astar_path_cost_reward_contract": True,
                    "path_cost_source": "hybrid_astar_pose_path/v1",
                    "hybrid_astar_path_cost": 2.5,
                    "hybrid_astar_pose_path_hash": "hybrid-pose-path-hash",
                    "hybrid_astar_trajectory_kind": "hybrid_astar_pose_path",
                    "legacy_grid_astar_path_cost": 10.0,
                    "hybrid_vs_grid_path_cost_delta": -7.5,
                    "default_astar_replaced": False,
                    "hybrid_astar_ackermann_feasible_claimed": hybrid_ackermann_feasible_claimed,
                    "point_grid_path_cost_fallback_used": False,
                    "metrics": metrics,
                }
            )
        if not theta_reward_omit_denominator:
            row["theta_coverage_denominator_cells"] = 100.0
    return row


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_config(tmp_path: Path, stage21_1: Path, stage21_2: Path, **overrides) -> Path:
    payload = {
        "schema_version": "xunce-stage21-3-ppo-batch-validation-config/v1",
        "stage21_1_collector_root": str(stage21_1),
        "stage21_2_reward_contract_root": str(stage21_2),
        "discount_factor": 0.99,
        "normalize_advantages": True,
        "validation_fraction": 0.25,
        "min_trainable_transition_count": 2,
        "max_old_log_prob_recompute_abs_error": 1.0e-6,
        "require_validation_split": False,
        "stage21_3_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage21_3_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
