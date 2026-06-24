from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_EXPLORER_SRC = REPO_ROOT / "model-explorer" / "src"
if str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))

from model_explorer.policy.coverage_first_reward import load_coverage_first_reward_profile

from scripts.run_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke import (
    ROUTE_BOUNDARY,
    ROUTE_INPUTS,
    ROUTE_POLICY_SIGNAL,
    ROUTE_RETURN_ADVANTAGE,
    ROUTE_SCALE,
    run_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke,
)


def test_stage21_13_copies_stage21_12_recommended_config_and_rejects_zero_delta(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    out = root / "out"
    _write_stage21_6_artifacts(out / "s6", raw_delta=0.0, capped_delta=0.0)
    config = _write_config(root)

    summary = run_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke(
        config_path=config,
        output_root=out,
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_RETURN_ADVANTAGE
    stage21_6_config = json.loads(Path(summary["stage21_6_config"]).read_text(encoding="utf-8"))
    assert stage21_6_config["reward_profile"].endswith("configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json")
    assert stage21_6_config["stage21_2_base_config"] == str(root / "stage21_12" / "xunce-stage21-12-recommended-stage21-2-config.json")
    assert stage21_6_config["seed_list"] == [2101, 2102, 2103]
    assert stage21_6_config["required_scenario_count"] == 8
    assert stage21_6_config["rollout_steps"] == 10
    assert stage21_6_config["dynamic_max_candidates_per_step"] == 36
    assert stage21_6_config["dynamic_proposal_pool_limit_per_step"] == 288
    assert stage21_6_config["runs_new_ppo_update"] is True
    assert stage21_6_config["publishes_checkpoint"] is False
    assert summary["stage21_6_lineage_passed"] is True
    assert summary["policy_shift_observable"] is True


def test_stage21_13_positive_raw_and_capped_delta_routes_to_scale(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    out = root / "out"
    _write_stage21_6_artifacts(out / "s6", raw_delta=0.01, capped_delta=0.01)
    config = _write_config(root)

    summary = run_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke(
        config_path=config,
        output_root=out,
        repo_root=root,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == ROUTE_SCALE
    assert summary["mean_final_coverage_rate_capped_delta"] == 0.01
    assert summary["mean_coverage_curve_auc_capped_delta"] == 0.01
    assert summary["stage21_13_authorized"] is False


def test_stage21_13_rejects_raw_positive_capped_zero(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    out = root / "out"
    _write_stage21_6_artifacts(out / "s6", raw_delta=0.01, capped_delta=0.0)
    config = _write_config(root)

    summary = run_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke(
        config_path=config,
        output_root=out,
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_RETURN_ADVANTAGE
    assert summary["mean_final_coverage_delta"] == 0.01
    assert summary["mean_final_coverage_rate_capped_delta"] == 0.0


def test_stage21_13_routes_policy_signal_when_parameter_delta_is_zero(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    out = root / "out"
    _write_stage21_6_artifacts(out / "s6", raw_delta=0.01, capped_delta=0.01, parameter_delta_l2=0.0)
    config = _write_config(root)

    summary = run_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke(
        config_path=config,
        output_root=out,
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_POLICY_SIGNAL
    assert summary["policy_shift_observable"] is False


def test_stage21_13_rejects_stage21_12_not_passed(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path, stage21_12_status="failed")
    config = _write_config(root)

    summary = run_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_INPUTS
    assert "stage21_12_not_passed" in summary["input_reason_codes"]


def test_stage21_13_rejects_wrong_reward_profile(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path, reward_profile="configs/xunce_stage21_coverage_first_ppo_reward_profile_v1.json")
    config = _write_config(root)

    summary = run_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_INPUTS
    assert "recommended_stage21_6_reward_profile_not_v2" in summary["input_reason_codes"]


def test_stage21_13_rejects_spoofed_v2_profile_hash(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path, profile_weight_override=41.0)
    config = _write_config(root)

    summary = run_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_INPUTS
    assert "recommended_stage21_6_reward_profile_hash_mismatch" in summary["input_reason_codes"]


def test_stage21_13_boundary_flags_hard_fail(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    config = _write_config(root, starts_online_canary=True)

    summary = run_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_BOUNDARY
    assert "stage21_13_config_starts_online_canary_true" in summary["boundary_reason_codes"]


def test_stage21_13_rejects_stage21_6_execution_boundary_counts(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    out = root / "out"
    _write_stage21_6_artifacts(out / "s6", raw_delta=0.01, capped_delta=0.01, execution_boundary_violation_total=1)
    config = _write_config(root)

    summary = run_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke(
        config_path=config,
        output_root=out,
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_BOUNDARY
    assert "stage21_6_execution_boundary_violation_total_nonzero" in summary["boundary_reason_codes"]


def test_stage21_13_rejects_nonfinite_gradient_flag(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    out = root / "out"
    _write_stage21_6_artifacts(out / "s6", raw_delta=0.01, capped_delta=0.01, grad_norm_finite=False)
    config = _write_config(root)

    summary = run_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke(
        config_path=config,
        output_root=out,
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "continue_stage21_9_gradient_normalization_loss_scaling_repair"


def _fixture_root(
    tmp_path: Path,
    *,
    stage21_12_status: str = "passed",
    reward_profile: str = "configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json",
    profile_weight_override: float | None = None,
) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "configs").mkdir()
    v2_profile_path = root / "configs" / "xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json"
    _write_v2_profile(v2_profile_path, coverage_per_cost_weight=profile_weight_override or 40.0)
    expected_hash = load_coverage_first_reward_profile(v2_profile_path).profile_hash
    stage21_12 = root / "stage21_12"
    stage21_12.mkdir()
    stage21_2_config = stage21_12 / "xunce-stage21-12-recommended-stage21-2-config.json"
    _json(stage21_2_config, {"schema_version": "xunce-stage21-2-coverage-first-ppo-reward-contract-config/v1"})
    stage21_6_config = stage21_12 / "xunce-stage21-12-recommended-stage21-6-config.json"
    _json(
        stage21_6_config,
        {
            "schema_version": "xunce-stage21-6-multi-seed-ppo-pilot-config/v1",
            "reward_profile": reward_profile,
            "stage21_2_base_config": str(stage21_2_config),
            "stage21_4_base_config": str(root / "stage21_10_repaired_stage21_4.json"),
            "seed_list": [2101, 2102, 2103],
            "required_scenario_count": 8,
            "rollout_steps": 10,
            "dynamic_max_candidates_per_step": 36,
            "dynamic_proposal_pool_limit_per_step": 288,
            "max_grad_norm": 25.0,
            "execute_seed_pipeline": True,
            "stage21_6_authorized": False,
            "training_or_release_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    _json(
        stage21_12 / "xunce-stage21-12-summary.json",
        {
            "status": stage21_12_status,
            "next_required_change": "run_stage21_13_coverage_constrained_multi_seed_ppo_smoke",
            "recommended_stage21_2_config": str(stage21_2_config),
            "recommended_stage21_6_config": str(stage21_6_config),
            "profile_v2_id": "xunce-stage21-coverage-constrained-ppo-reward-v2",
            "profile_v2_version": "stage21-coverage-constrained-v2",
            "profile_v2_hash": "expected-original-hash" if profile_weight_override is not None else expected_hash,
            "v2_hard_risk_trainable_count": 0,
        },
    )
    return root


def _write_config(root: Path, **overrides: object) -> Path:
    payload = {
        "schema_version": "xunce-stage21-13-coverage-constrained-multi-seed-ppo-smoke-config/v1",
        "stage21_12_root": str(root / "stage21_12"),
        "execute_stage21_6": False,
        "reuse_existing_stage21_6_result": True,
        "stage21_6_output_subdir": "s6",
        "min_parameter_delta_l2": 1.0e-12,
        "max_abs_approx_kl": 0.5,
        "min_entropy": 0.0,
        "stage21_13_authorized": False,
        "training_or_release_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = root / "stage21_13_config.json"
    _json(path, payload)
    return path


def _write_stage21_6_artifacts(
    root: Path,
    *,
    raw_delta: float,
    capped_delta: float,
    parameter_delta_l2: float = 0.001,
    lineage_passed: bool = True,
    execution_boundary_violation_total: int = 0,
    grad_norm_finite: bool = True,
) -> None:
    root.mkdir(parents=True)
    _json(
        root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json",
        {
            "status": "passed" if raw_delta > 0.0 and capped_delta > 0.0 else "failed",
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    _json(
        root / "xunce-stage21-6-aggregate-metrics.json",
        {
            "seed_count": 3,
            "trainable_transition_count_total": 240,
            "final_coverage_delta_mean": raw_delta,
            "final_coverage_delta_min": raw_delta,
            "coverage_auc_delta_mean": raw_delta,
            "coverage_auc_delta_min": raw_delta,
            "pre_clip_grad_norm_max": 2.0,
            "pre_clip_grad_norm_mean": 2.0,
            "max_post_update_approx_kl_mean": 0.001,
            "min_entropy_mean": 1.0,
            "hard_risk_violation_total": 0,
            "safety_boundary_violation_total": 0,
            "scenario_safety_boundary_regression_total": 0,
            "execution_boundary_violation_total": execution_boundary_violation_total,
            "model_inference_failure_total": 0,
            "model_inference_mask_violation_total": 0,
            "path_planning_failure_total": 0,
            "open_grid_fallback_total": 0,
            "seed_release_boundary_violation_total": 0,
            "checkpoint_reload_failed_count": 0,
            "checkpoint_not_experimental_only_count": 0,
            "sample_count_too_low_for_performance_claim": False,
            "worst_seed_id": 2101,
        },
    )
    _json(root / "xunce-stage21-6-lineage-audit.json", {"passed": lineage_passed})
    rows = []
    for index, seed in enumerate((2101, 2102, 2103)):
        stage21_5_root = root / f"seed_{seed}" / "stage21_5"
        stage21_5_root.mkdir(parents=True)
        _json(
            stage21_5_root / "xunce-stage21-5-post-update-evaluation-summary.json",
            {
                "final_coverage_rate_delta": raw_delta,
                "coverage_curve_auc_delta": raw_delta,
                "final_coverage_rate_capped_delta": capped_delta,
                "coverage_curve_auc_capped_delta": capped_delta,
            },
        )
        rows.append(
            {
                "seed": seed,
                "stage21_5_root": str(stage21_5_root),
                "pre_clip_grad_norm": 2.0,
                "post_clip_grad_norm": 1.0,
                "max_post_update_approx_kl": 0.001,
                "min_entropy": 1.0,
                "grad_norm_finite": grad_norm_finite,
                "parameter_delta_l2": parameter_delta_l2 if index > 0 else parameter_delta_l2,
                "final_coverage_delta": raw_delta,
                "coverage_auc_delta": raw_delta,
                "stage21_4_status": "passed",
                "stage21_5_status": "passed",
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
                "stage21_4_authorized": False,
                "training_or_release_authorized": False,
            }
        )
    _jsonl(root / "xunce-stage21-6-seed-results.jsonl", rows)


def _json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _write_v2_profile(path: Path, *, coverage_per_cost_weight: float) -> None:
    _json(
        path,
        {
            "schema_version": "xunce-stage21-coverage-constrained-ppo-reward-profile/v2",
            "profile_id": "xunce-stage21-coverage-constrained-ppo-reward-v2",
            "profile_version": "stage21-coverage-constrained-v2",
            "target_final_coverage_rate": 0.99,
            "coverage_cap_rate": 1.0,
            "horizon_steps": 40,
            "mission_budget_route_when_below_target": "stage21_12_rollout_horizon_or_mission_budget_scaling_for_99pct_coverage",
            "weights": {
                "coverage_gain": 4.0,
                "coverage_progress": 0.5,
                "coverage_per_cost": coverage_per_cost_weight,
                "final_coverage": 2.0,
                "success_99pct": 5.0,
                "path_cost": 0.2,
                "soft_risk": 0.005,
                "failure": 1.0,
                "hard_risk_failure": 5.0,
            },
            "normalizers": {
                "coverage_rate_delta": 0.05,
                "remaining_coverage_gap": 0.99,
                "coverage_per_cost": 0.025,
                "final_coverage_rate": 0.99,
                "path_cost_m": 100.0,
                "soft_risk_exposure": 25.0,
            },
            "risk_policy": {
                "path_cost_includes_risk_proxy": True,
                "soft_risk_component_mode": "audit_weighted_tiny",
                "max_soft_risk_to_path_cost_penalty_ratio": 0.1,
            },
            "reward_policy": {
                "coverage_per_cost_path_cost_floor": 0.01,
                "coverage_per_cost_component_cap_ratio": 8.0,
                "path_cost_penalty_cap_ratio_when_below_target": 0.5,
            },
            "hard_risk_policy": {
                "reject_before_reward": True,
                "clamp_positive_reward_to_non_positive": True,
                "failure_floor": -1.0,
            },
            "component_source_map": {
                "coverage_gain_component": "transition.info.coverage_rate_delta",
                "coverage_progress_component": "transition.info.final_coverage_rate_after_step",
                "coverage_per_cost_component": "test fixture",
                "path_cost_component": "transition.info.path_cost",
                "soft_risk_component": "transition.info.soft_risk_exposure",
                "final_coverage_bonus_component": "terminal transition final coverage",
                "success_99pct_bonus_component": "terminal transition final coverage >= target_final_coverage_rate",
                "failure_component": "transition failure reason",
                "hard_risk_component": "hard risk rejection signal",
            },
        },
    )
