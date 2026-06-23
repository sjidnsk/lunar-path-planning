from __future__ import annotations

import json
from pathlib import Path

from scripts.run_xunce_stage21_6_multi_seed_ppo_pilot import (
    ROUTE_BOUNDARY,
    ROUTE_EXECUTION,
    ROUTE_HARD_RISK,
    ROUTE_LINEAGE,
    ROUTE_NUMERICS,
    ROUTE_REPAIR,
    ROUTE_STAGE21_5,
    ROUTE_STAGE22,
    run_xunce_stage21_6_multi_seed_ppo_pilot,
)


def test_stage21_6_routes_to_stage22_when_all_seed_smokes_improve(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    seeds = [2101, 2102, 2103]
    _write_stage21_5_prerequisite(root / "stage21_5_prereq")
    for seed in seeds:
        _write_seed(root / "out" / f"seed_{seed}", seed=seed, final_delta=0.01, auc_delta=0.01)
    config = _write_config(root, seeds=seeds)

    summary = run_xunce_stage21_6_multi_seed_ppo_pilot(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == ROUTE_STAGE22
    assert summary["seed_count"] == 3
    assert summary["mean_final_coverage_delta"] == 0.01
    assert summary["sample_count_too_low_for_performance_claim"] is True
    assert summary["stage21_6_authorized"] is False
    assert summary["training_or_release_authorized"] is False
    assert summary["runs_new_ppo_update"] is True
    assert summary["publishes_checkpoint"] is False
    assert summary["replaces_default_policy"] is False
    assert summary["connects_real_executor"] is False
    assert summary["starts_online_canary"] is False


def test_stage21_6_requires_stage21_5_prerequisite(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    config = _write_config(root)

    summary = run_xunce_stage21_6_multi_seed_ppo_pilot(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_STAGE21_5
    assert "missing_stage21_5_summary" in summary["reason_codes"]


def test_stage21_6_boundary_flag_hard_fails(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_5_prerequisite(root / "stage21_5_prereq")
    config = _write_config(root, publishes_checkpoint=True)

    summary = run_xunce_stage21_6_multi_seed_ppo_pilot(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_BOUNDARY
    assert "publishes_checkpoint" in summary["reason_codes"]


def test_stage21_6_rejects_failed_seed_stage(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_5_prerequisite(root / "stage21_5_prereq")
    for seed in [2101, 2102, 2103]:
        _write_seed(root / "out" / f"seed_{seed}", seed=seed, final_delta=0.01, auc_delta=0.01, stage21_4_status="failed" if seed == 2102 else "passed")
    config = _write_config(root)

    summary = run_xunce_stage21_6_multi_seed_ppo_pilot(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_EXECUTION
    assert "seed_2102_stage21_4_status_not_passed" in summary["reason_codes"]


def test_stage21_6_rejects_per_seed_checkpoint_boundary_violation(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_5_prerequisite(root / "stage21_5_prereq")
    for seed in [2101, 2102, 2103]:
        _write_seed(
            root / "out" / f"seed_{seed}",
            seed=seed,
            final_delta=0.01,
            auc_delta=0.01,
            checkpoint_reload_passed=False if seed == 2102 else True,
            experimental_only=False if seed == 2102 else True,
            publishes_checkpoint=True if seed == 2102 else False,
        )
    config = _write_config(root)

    summary = run_xunce_stage21_6_multi_seed_ppo_pilot(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_BOUNDARY
    assert "seed_2102_checkpoint_reload_failed" in summary["reason_codes"]
    assert "seed_2102_checkpoint_not_experimental_only" in summary["reason_codes"]
    assert "seed_2102_publishes_checkpoint_true" in summary["reason_codes"]


def test_stage21_6_rejects_hard_risk_or_execution_boundary(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_5_prerequisite(root / "stage21_5_prereq")
    for seed in [2101, 2102, 2103]:
        _write_seed(root / "out" / f"seed_{seed}", seed=seed, final_delta=0.01, auc_delta=0.01, hard_risk=1 if seed == 2103 else 0)
    config = _write_config(root)

    summary = run_xunce_stage21_6_multi_seed_ppo_pilot(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_HARD_RISK
    assert "hard_risk_or_execution_boundary_regression" in summary["reason_codes"]


def test_stage21_6_rejects_model_inference_execution_boundary(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_5_prerequisite(root / "stage21_5_prereq")
    for seed in [2101, 2102, 2103]:
        _write_seed(
            root / "out" / f"seed_{seed}",
            seed=seed,
            final_delta=0.01,
            auc_delta=0.01,
            model_inference_failure_count=1 if seed == 2103 else 0,
        )
    config = _write_config(root)

    summary = run_xunce_stage21_6_multi_seed_ppo_pilot(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_HARD_RISK
    assert summary["execution_boundary_violation_total"] == 1
    assert "hard_risk_or_execution_boundary_regression" in summary["reason_codes"]


def test_stage21_6_rejects_kl_entropy_or_grad_instability(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_5_prerequisite(root / "stage21_5_prereq")
    for seed in [2101, 2102, 2103]:
        _write_seed(root / "out" / f"seed_{seed}", seed=seed, final_delta=0.01, auc_delta=0.01, kl=0.8 if seed == 2101 else 0.01)
    config = _write_config(root)

    summary = run_xunce_stage21_6_multi_seed_ppo_pilot(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_NUMERICS
    assert "seed_2101_kl_unstable" in summary["reason_codes"]


def test_stage21_6_rejects_reused_batch_fingerprint(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_5_prerequisite(root / "stage21_5_prereq")
    for seed in [2101, 2102, 2103]:
        _write_seed(root / "out" / f"seed_{seed}", seed=seed, final_delta=0.01, auc_delta=0.01, batch_suffix="same")
    config = _write_config(root)

    summary = run_xunce_stage21_6_multi_seed_ppo_pilot(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_LINEAGE
    assert "duplicate_stage21_3_batch_fingerprint" in summary["reason_codes"]


def test_stage21_6_rejects_no_mean_coverage_improvement(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_5_prerequisite(root / "stage21_5_prereq")
    for seed in [2101, 2102, 2103]:
        _write_seed(root / "out" / f"seed_{seed}", seed=seed, final_delta=0.0, auc_delta=0.01)
    config = _write_config(root)

    summary = run_xunce_stage21_6_multi_seed_ppo_pilot(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_REPAIR
    assert "coverage_auc_worst_seed_or_cost_rejected" in summary["reason_codes"]
    assert summary["runs_new_ppo_update"] is True
    assert summary["offline_ppo_update_executed"] is True


def test_stage21_6_computes_low_sample_from_total_transition_threshold(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_5_prerequisite(root / "stage21_5_prereq")
    for seed in [2101, 2102, 2103]:
        _write_seed(
            root / "out" / f"seed_{seed}",
            seed=seed,
            final_delta=0.01,
            auc_delta=0.01,
            stage21_4_sample_count_too_low=False,
        )
    config = _write_config(root)

    summary = run_xunce_stage21_6_multi_seed_ppo_pilot(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == ROUTE_STAGE22
    assert summary["trainable_transition_count_total"] == 24
    assert summary["min_transition_count_for_performance_claim"] == 200
    assert summary["sample_count_too_low_for_performance_claim"] is True


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("configs", "out"):
        (root / name).mkdir(exist_ok=True)
    for filename in (
        "stage21_1.json",
        "stage21_2.json",
        "stage21_3.json",
        "stage21_4.json",
        "stage21_5.json",
    ):
        (root / "configs" / filename).write_text("{}", encoding="utf-8")
    return root


def _write_config(root: Path, *, seeds: list[int] | None = None, **overrides: object) -> Path:
    payload = {
        "schema_version": "xunce-stage21-6-multi-seed-ppo-pilot-config/v1",
        "stage21_5_prerequisite_root": str(root / "stage21_5_prereq"),
        "stage21_1_base_config": str(root / "configs" / "stage21_1.json"),
        "stage21_2_base_config": str(root / "configs" / "stage21_2.json"),
        "stage21_3_base_config": str(root / "configs" / "stage21_3.json"),
        "stage21_4_base_config": str(root / "configs" / "stage21_4.json"),
        "stage21_5_base_config": str(root / "configs" / "stage21_5.json"),
        "execute_seed_pipeline": False,
        "seed_list": seeds or [2101, 2102, 2103],
        "required_scenario_count": 2,
        "rollout_steps": 4,
        "dynamic_max_candidates_per_step": 36,
        "dynamic_proposal_pool_limit_per_step": 288,
        "min_transition_count_for_performance_claim": 200,
        "max_abs_approx_kl": 0.5,
        "min_entropy": 0.0,
        "max_grad_norm": 25.0,
        "max_path_cost_delta_m": 20.0,
        "max_soft_risk_exposure_delta": 25.0,
        "min_mean_coverage_delta": 0.0,
        "min_worst_seed_coverage_delta": 0.0,
        "stage21_6_authorized": False,
        "training_or_release_authorized": False,
        "runs_new_ppo_update": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = root / "config.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_stage21_5_prerequisite(root: Path) -> None:
    root.mkdir(parents=True)
    summary = {
        "status": "passed",
        "next_required_change": "implement_stage21_6_multi_seed_ppo_pilot",
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    routing = dict(summary)
    (root / "xunce-stage21-5-post-update-evaluation-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "xunce-stage21-5-next-stage-routing.json").write_text(json.dumps(routing), encoding="utf-8")


def _write_seed(
    root: Path,
    *,
    seed: int,
    final_delta: float,
    auc_delta: float,
    hard_risk: int = 0,
    kl: float = 0.01,
    entropy: float = 1.0,
    grad: float = 1.0,
    stage21_4_status: str = "passed",
    batch_suffix: str | None = None,
    stage21_4_sample_count_too_low: bool = True,
    checkpoint_reload_passed: bool = True,
    experimental_only: bool = True,
    publishes_checkpoint: bool = False,
    model_inference_failure_count: int = 0,
) -> None:
    for stage in ("stage21_1", "stage21_3", "stage21_4", "stage21_5"):
        (root / stage).mkdir(parents=True, exist_ok=True)
    _json(root / "stage21_1" / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json", {"status": "passed", "sampling_seed": seed, "trainable_transition_count": 8})
    _json(root / "stage21_3" / "xunce-stage21-3-ppo-batch-validation-summary.json", {"status": "passed", "trainable_transition_count": 8})
    suffix = batch_suffix or str(seed)
    batch_rows = [
        {"transition_id": f"{suffix}-t0", "seed": suffix, "value": 1},
        {"transition_id": f"{suffix}-t1", "seed": suffix, "value": 2},
    ]
    (root / "stage21_3" / "xunce-stage21-3-ppo-trainable-batch.jsonl").write_text(
        "\n".join(json.dumps(row) for row in batch_rows) + "\n",
        encoding="utf-8",
    )
    _json(
        root / "stage21_4" / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json",
        {
            "status": stage21_4_status,
            "experimental_checkpoint": experimental_only,
            "stage21_4_authorized": False,
            "training_or_release_authorized": False,
            "parameter_delta_l2": 0.01,
            "sample_count_too_low_for_performance_claim": stage21_4_sample_count_too_low,
            "publishes_checkpoint": publishes_checkpoint,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
        },
    )
    _json(
        root / "stage21_4" / "xunce-stage21-4-gradient-audit.json",
        {
            "grad_norm": grad,
            "pre_clip_grad_norm": grad,
            "post_clip_grad_norm": min(grad, 1.0),
            "grad_norm_finite": True,
        },
    )
    _json(
        root / "stage21_4" / "xunce-stage21-4-checkpoint-audit.json",
        {
            "checkpoint_reload_passed": checkpoint_reload_passed,
            "metadata": {
                "experimental_only": experimental_only,
                "publishes_checkpoint": publishes_checkpoint,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
                "training_or_release_authorized": False,
            },
        },
    )
    (root / "stage21_4" / "xunce-stage21-4-ppo-loss-audit.jsonl").write_text(
        json.dumps({"post_update_approx_kl": kl, "entropy": entropy}) + "\n",
        encoding="utf-8",
    )
    post_eval_root = root / "stage21_5" / "post_ppo_xunce"
    post_eval_root.mkdir(parents=True, exist_ok=True)
    _json(
        post_eval_root / "xunce-exploration-coverage-comparison-summary.json",
        {
            "status": "passed",
            "true_model_inference_executed": True,
            "model_inference_failure_count": model_inference_failure_count,
            "model_inference_mask_violation_count": 0,
            "unreachable_selected_count": 0,
            "path_planning_failure_count": 0,
            "open_grid_fallback_count": 0,
        },
    )
    _json(
        root / "stage21_5" / "xunce-stage21-5-post-update-evaluation-summary.json",
        {
            "status": "passed",
            "post_evaluation_root": str(post_eval_root),
            "final_coverage_rate_delta": final_delta,
            "coverage_curve_auc_delta": auc_delta,
            "path_cost_total_m_delta": 0.0,
            "soft_risk_exposure_total_delta": 0.0,
            "post_hard_risk_violation_count": hard_risk,
            "post_safety_boundary_violation_count": hard_risk,
            "scenario_regression_count": 0,
            "scenario_safety_boundary_regression_count": hard_risk,
        },
    )


def _json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
