from __future__ import annotations

import json
from pathlib import Path

import scripts.run_xunce_stage21_7_reward_collector_advantage_horizon_repair as stage21_7
from scripts.run_xunce_stage21_7_reward_collector_advantage_horizon_repair import (
    ROUTE_ADVANTAGE,
    ROUTE_INPUTS,
    ROUTE_REWARD,
    ROUTE_RERUN_21_6,
    ROUTE_SCALE,
    ROUTE_UPDATE,
    run_xunce_stage21_7_reward_collector_advantage_horizon_repair,
)


def test_stage21_7_routes_to_scale_and_writes_repaired_config(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_6_root(root, trainable_transition_count_total=9)
    config = _write_config(root, execute_repaired_pilot=False)

    summary = run_xunce_stage21_7_reward_collector_advantage_horizon_repair(
        config_path=config,
        output_root=root / "stage21_7",
        repo_root=root,
    )

    assert summary["status"] == "partial"
    assert summary["next_required_change"] == ROUTE_SCALE
    assert summary["primary_root_cause"] == "collector_or_horizon"
    assert "insufficient_trainable_transition_count" in summary["diagnostic_reason_codes"]
    assert "holdout_horizon_too_short" in summary["diagnostic_reason_codes"]
    repaired = json.loads((root / "stage21_7" / "xunce-stage21-7-repaired-stage21-6-config.json").read_text())
    assert repaired["required_scenario_count"] == 8
    assert repaired["rollout_steps"] == 10
    assert repaired["publishes_checkpoint"] is False
    repaired_stage21_4 = json.loads((root / "stage21_7" / "xunce-stage21-7-repaired-stage21-4-base-config.json").read_text())
    assert repaired_stage21_4["epochs"] == 2
    assert repaired_stage21_4["learning_rate"] == 0.00002


def test_stage21_7_rejects_missing_stage21_6_inputs(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    config = _write_config(root, execute_repaired_pilot=False)

    summary = run_xunce_stage21_7_reward_collector_advantage_horizon_repair(
        config_path=config,
        output_root=root / "stage21_7",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_INPUTS
    assert "missing_stage21_6_summary" in summary["input_reason_codes"]


def test_stage21_7_rejects_open_config_boundary_flags(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_6_root(root)
    config = _write_config(root, execute_repaired_pilot=False, publishes_checkpoint=True)

    summary = run_xunce_stage21_7_reward_collector_advantage_horizon_repair(
        config_path=config,
        output_root=root / "stage21_7",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_INPUTS
    assert "stage21_7_config_publishes_checkpoint_true" in summary["boundary_reason_codes"]


def test_stage21_7_rejects_stage21_6_execution_failure_counts(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_6_root(root, aggregate_overrides={"open_grid_fallback_total": 1})
    config = _write_config(root, execute_repaired_pilot=False)

    summary = run_xunce_stage21_7_reward_collector_advantage_horizon_repair(
        config_path=config,
        output_root=root / "stage21_7",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_INPUTS
    assert "stage21_6_open_grid_fallback_total_nonzero" in summary["boundary_reason_codes"]


def test_stage21_7_routes_reward_signal_first(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_6_root(root, reward_mode="flat")
    config = _write_config(
        root,
        execute_repaired_pilot=False,
        min_transition_count_for_performance_claim=1,
        current_holdout_rollout_steps=10,
    )

    summary = run_xunce_stage21_7_reward_collector_advantage_horizon_repair(
        config_path=config,
        output_root=root / "stage21_7",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_REWARD
    assert summary["primary_root_cause"] == "reward"
    assert "weak_or_misaligned_reward_signal" in summary["diagnostic_reason_codes"]


def test_stage21_7_routes_advantage_after_reward_passes(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_6_root(root, advantage_mode="flat")
    config = _write_config(
        root,
        execute_repaired_pilot=False,
        min_transition_count_for_performance_claim=1,
        current_holdout_rollout_steps=10,
    )

    summary = run_xunce_stage21_7_reward_collector_advantage_horizon_repair(
        config_path=config,
        output_root=root / "stage21_7",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_ADVANTAGE
    assert summary["primary_root_cause"] == "advantage"


def test_stage21_7_routes_ppo_update_strength_when_policy_shift_is_tiny(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_6_root(root, policy_shift="tiny")
    config = _write_config(
        root,
        execute_repaired_pilot=False,
        min_transition_count_for_performance_claim=1,
        current_holdout_rollout_steps=10,
    )

    summary = run_xunce_stage21_7_reward_collector_advantage_horizon_repair(
        config_path=config,
        output_root=root / "stage21_7",
        repo_root=root,
    )

    assert summary["status"] == "partial"
    assert summary["next_required_change"] == ROUTE_UPDATE
    assert summary["primary_root_cause"] == "ppo_update_strength"
    assert "ppo_update_too_small" in summary["diagnostic_reason_codes"]


def test_stage21_7_uses_repaired_pilot_passed_result(monkeypatch, tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_6_root(root, trainable_transition_count_total=9)
    config = _write_config(root, execute_repaired_pilot=True)

    def fake_stage21_6(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, object]:
        return {
            "status": "passed",
            "next_required_change": "prepare_stage22_formal_pure_ppo_training_run",
            "mean_final_coverage_delta": 0.02,
            "mean_coverage_auc_delta": 0.02,
            "sample_count_too_low_for_performance_claim": False,
            "reason_codes": [],
            "summary": str(output_root / "summary.json"),
        }

    monkeypatch.setattr(stage21_7, "run_xunce_stage21_6_multi_seed_ppo_pilot", fake_stage21_6)

    summary = run_xunce_stage21_7_reward_collector_advantage_horizon_repair(
        config_path=config,
        output_root=root / "stage21_7",
        repo_root=root,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == ROUTE_RERUN_21_6
    assert summary["repaired_pilot_executed"] is True
    assert summary["repaired_pilot_status"] == "passed"


def test_stage21_7_routes_repaired_grad_instability_to_update_calibration(monkeypatch, tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_6_root(root, trainable_transition_count_total=9)
    config = _write_config(root, execute_repaired_pilot=True)

    def fake_stage21_6(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, object]:
        return {
            "status": "failed",
            "next_required_change": "repair_stage21_6_ppo_numerical_stability",
            "mean_final_coverage_delta": 0.0,
            "mean_coverage_auc_delta": 0.0,
            "sample_count_too_low_for_performance_claim": False,
            "reason_codes": ["seed_2101_grad_unstable"],
            "summary": str(output_root / "summary.json"),
        }

    monkeypatch.setattr(stage21_7, "run_xunce_stage21_6_multi_seed_ppo_pilot", fake_stage21_6)

    summary = run_xunce_stage21_7_reward_collector_advantage_horizon_repair(
        config_path=config,
        output_root=root / "stage21_7",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_UPDATE
    assert summary["primary_root_cause"] == "repaired_pilot_ppo_numerical_stability"


def test_stage21_7_reuses_existing_repaired_pilot_summary(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_6_root(root, trainable_transition_count_total=9)
    repaired_root = root / "repaired_stage21_6"
    repaired_root.mkdir()
    _json(
        repaired_root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json",
        {
            "status": "failed",
            "next_required_change": "repair_stage21_6_ppo_numerical_stability",
            "mean_final_coverage_delta": 0.0,
            "mean_coverage_auc_delta": 0.0,
            "sample_count_too_low_for_performance_claim": False,
            "reason_codes": ["seed_2101_grad_unstable"],
        },
    )
    config = _write_config(root, execute_repaired_pilot=True, reuse_completed_repaired_pilot_result=True)

    summary = run_xunce_stage21_7_reward_collector_advantage_horizon_repair(
        config_path=config,
        output_root=root / "stage21_7",
        repo_root=root,
    )
    repaired = json.loads((root / "stage21_7" / "xunce-stage21-7-repaired-stage21-6-result-summary.json").read_text())

    assert summary["next_required_change"] == ROUTE_UPDATE
    assert repaired["reused_existing_result"] is True


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "configs").mkdir()
    _json(root / "configs" / "stage21_6.json", {"schema_version": "xunce-stage21-6-multi-seed-ppo-pilot-config/v1"})
    _json(root / "configs" / "stage21_4.json", {"schema_version": "xunce-stage21-4-tiny-ppo-update-smoke-config/v1"})
    return root


def _write_config(root: Path, **overrides: object) -> Path:
    payload = {
        "schema_version": "xunce-stage21-7-reward-collector-advantage-horizon-repair-config/v1",
        "stage21_6_root": str(root / "stage21_6"),
        "stage21_6_base_config": str(root / "configs" / "stage21_6.json"),
        "stage21_4_base_config": str(root / "configs" / "stage21_4.json"),
        "min_transition_count_for_performance_claim": 200,
        "current_holdout_rollout_steps": 4,
        "min_rollout_steps_for_horizon_confidence": 10,
        "min_reward_std": 0.01,
        "min_reward_coverage_correlation": 0.1,
        "min_coverage_reward_nonzero_rate": 0.2,
        "min_advantage_std": 0.01,
        "min_advantage_coverage_correlation": 0.1,
        "min_abs_kl_for_policy_shift": 0.001,
        "min_selected_probability_delta": 0.001,
        "min_parameter_delta_l2": 0.001,
        "execute_repaired_pilot": False,
        "repaired_seed_list": [2101, 2102, 2103],
        "repaired_required_scenario_count": 8,
        "repaired_rollout_steps": 10,
        "repaired_dynamic_max_candidates_per_step": 36,
        "repaired_dynamic_proposal_pool_limit_per_step": 288,
        "repaired_stage21_4_epochs": 2,
        "repaired_stage21_4_learning_rate": 0.00002,
        "repaired_stage21_6_output_root": str(root / "repaired_stage21_6"),
    }
    payload.update(overrides)
    path = root / "config.json"
    _json(path, payload)
    return path


def _write_stage21_6_root(
    root: Path,
    *,
    trainable_transition_count_total: int = 240,
    reward_mode: str = "good",
    advantage_mode: str = "good",
    policy_shift: str = "visible",
    aggregate_overrides: dict[str, object] | None = None,
) -> None:
    stage21_6 = root / "stage21_6"
    stage21_6.mkdir()
    seeds = [2101, 2102, 2103]
    _json(
        stage21_6 / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json",
        {
            "status": "failed",
            "next_required_change": "repair_stage21_6_reward_collector_advantage_or_horizon",
            "mean_final_coverage_delta": 0.0,
            "mean_coverage_auc_delta": 0.0,
            "stage21_6_authorized": False,
            "training_or_release_authorized": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    aggregate = {
        "trainable_transition_count_total": trainable_transition_count_total,
        "hard_risk_violation_total": 0,
        "safety_boundary_violation_total": 0,
        "execution_boundary_violation_total": 0,
        "model_inference_failure_total": 0,
        "model_inference_mask_violation_total": 0,
        "open_grid_fallback_total": 0,
        "path_planning_failure_total": 0,
        "unreachable_selected_total": 0,
        "checkpoint_reload_failed_count": 0,
        "checkpoint_not_experimental_only_count": 0,
        "seed_release_boundary_violation_total": 0,
    }
    if aggregate_overrides:
        aggregate.update(aggregate_overrides)
    _json(stage21_6 / "xunce-stage21-6-aggregate-metrics.json", aggregate)
    _json(stage21_6 / "xunce-stage21-6-lineage-audit.json", {"passed": True})
    _json(stage21_6 / "xunce-stage21-6-next-stage-routing.json", {"status": "failed"})
    rows = []
    for seed in seeds:
        seed_root = stage21_6 / f"seed_{seed}"
        _write_seed(seed_root, seed=seed, reward_mode=reward_mode, advantage_mode=advantage_mode, policy_shift=policy_shift)
        rows.append({"seed": seed, "sampling_seed": seed, "seed_root": str(seed_root)})
    _jsonl(stage21_6 / "xunce-stage21-6-seed-results.jsonl", rows)


def _write_seed(root: Path, *, seed: int, reward_mode: str, advantage_mode: str, policy_shift: str) -> None:
    stage21_3 = root / "stage21_3"
    stage21_4 = root / "stage21_4"
    stage21_5 = root / "stage21_5"
    pre = stage21_5 / "pre_ppo_xunce"
    post = stage21_5 / "post_ppo_xunce"
    for path in (stage21_3, stage21_4, stage21_5, pre, post):
        path.mkdir(parents=True, exist_ok=True)
    coverage = [0.1, 0.2, 0.3]
    rewards = [1.0, 2.0, 3.0] if reward_mode == "good" else [1.0, 1.0, 1.0]
    advantages = [-1.0, 0.5, 1.0] if advantage_mode == "good" else [0.0, 0.0, 0.0]
    batch_rows = []
    for index, cov in enumerate(coverage):
        batch_rows.append(
            {
                "transition_id": f"{seed}-{index}",
                "reward": rewards[index],
                "advantage": advantages[index],
                "raw_advantage": advantages[index],
                "reward_components": {"step_coverage_gain_component": cov},
                "info": {"coverage_rate_delta": cov},
            }
        )
    _jsonl(stage21_3 / "xunce-stage21-3-ppo-trainable-batch.jsonl", batch_rows)
    _jsonl(
        stage21_4 / "xunce-stage21-4-ppo-loss-audit.jsonl",
        [
            {
                "post_update_approx_kl": 0.01 if policy_shift == "visible" else 0.00001,
                "ratio_min": 0.99,
                "ratio_max": 1.01,
                "entropy": 1.0,
            }
        ],
    )
    _json(stage21_4 / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json", {"parameter_delta_l2": 0.01 if policy_shift == "visible" else 0.0001})
    _json(
        stage21_5 / "xunce-stage21-5-post-update-evaluation-summary.json",
        {
            "pre_evaluation_root": str(pre),
            "post_evaluation_root": str(post),
            "rollout_steps": 4,
            "required_scenario_count": 2,
        },
    )
    _json(stage21_5 / "xunce-stage21-5-trajectory-delta.json", {"final_coverage_rate_mean_delta": 0.0})
    _write_inference(pre, post, policy_shift=policy_shift)


def _write_inference(pre: Path, post: Path, *, policy_shift: str) -> None:
    pre_rows = []
    post_rows = []
    for step in range(3):
        pre_rows.append(
            {
                "scenario_id": "s0",
                "step_index": step,
                "candidate_set_hash": f"h{step}",
                "detail": {"selected_action_index": 0, "selected_rank": 1, "selected_probability": 0.5, "logits": [1.0, 0.0]},
            }
        )
        post_rows.append(
            {
                "scenario_id": "s0",
                "step_index": step,
                "candidate_set_hash": f"h{step}",
                "detail": {
                    "selected_action_index": 1 if policy_shift == "visible" and step == 0 else 0,
                    "selected_rank": 2 if policy_shift == "visible" and step == 0 else 1,
                    "selected_probability": 0.7 if policy_shift == "visible" else 0.50001,
                    "logits": [0.0, 1.0] if policy_shift == "visible" else [1.00001, 0.0],
                },
            }
        )
    _jsonl(pre / "xunce-exploration-coverage-model-inference.jsonl", pre_rows)
    _jsonl(post / "xunce-exploration-coverage-model-inference.jsonl", post_rows)


def _json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
