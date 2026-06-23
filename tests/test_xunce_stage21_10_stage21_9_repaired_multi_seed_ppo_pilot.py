from __future__ import annotations

import json
from pathlib import Path

from scripts.run_xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot import (
    ROUTE_BOUNDARY,
    ROUTE_INPUTS,
    ROUTE_REWARD_ADVANTAGE,
    ROUTE_SCALE,
    run_xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot,
)


def test_stage21_10_writes_repaired_configs_and_rejects_zero_delta(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    out = root / "out"
    _write_stage21_6_artifacts(out / "stage21_6", raw_delta=0.0, capped_delta=0.0)
    config = _write_config(root)

    summary = run_xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot(
        config_path=config,
        output_root=out,
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_REWARD_ADVANTAGE
    repaired4 = json.loads(Path(summary["repaired_stage21_4_config"]).read_text(encoding="utf-8"))
    assert repaired4["learning_rate"] == 0.000002
    assert repaired4["epochs"] == 1
    assert repaired4["clip_ratio"] == 0.2
    assert repaired4["max_grad_norm"] == 1.0
    assert repaired4["advantage_clip_abs"] == 5.0
    assert repaired4["normalize_minibatch_advantages"] is True
    assert repaired4["loss_scale"] == 0.25
    assert repaired4["value_loss_coefficient"] == 0.1
    assert repaired4["publishes_checkpoint"] is False
    repaired6 = json.loads(Path(summary["repaired_stage21_6_config"]).read_text(encoding="utf-8"))
    assert repaired6["stage21_4_base_config"] == summary["repaired_stage21_4_config"]
    assert repaired6["seed_list"] == [2101, 2102, 2103]
    assert repaired6["required_scenario_count"] == 8
    assert repaired6["rollout_steps"] == 10
    assert repaired6["dynamic_max_candidates_per_step"] == 36
    assert repaired6["dynamic_proposal_pool_limit_per_step"] == 288
    assert repaired6["max_grad_norm"] == 25.0
    assert repaired6["publishes_checkpoint"] is False
    assert summary["stage21_6_lineage_passed"] is True


def test_stage21_10_raw_positive_but_capped_zero_still_rejected(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    out = root / "out"
    _write_stage21_6_artifacts(out / "stage21_6", raw_delta=0.01, capped_delta=0.0)
    config = _write_config(root)

    summary = run_xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot(
        config_path=config,
        output_root=out,
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_REWARD_ADVANTAGE
    assert summary["mean_final_coverage_delta"] == 0.01
    assert summary["mean_final_coverage_rate_capped_delta"] == 0.0


def test_stage21_10_positive_raw_and_capped_delta_routes_to_scale(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    out = root / "out"
    _write_stage21_6_artifacts(out / "stage21_6", raw_delta=0.01, capped_delta=0.01)
    config = _write_config(root)

    summary = run_xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot(
        config_path=config,
        output_root=out,
        repo_root=root,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == ROUTE_SCALE
    assert summary["mean_coverage_curve_auc_capped_delta"] == 0.01
    assert summary["stage21_10_authorized"] is False


def test_stage21_10_rejects_positive_delta_when_stage21_6_status_failed(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    out = root / "out"
    _write_stage21_6_artifacts(out / "stage21_6", raw_delta=0.01, capped_delta=0.01, summary_status="failed")
    config = _write_config(root)

    summary = run_xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot(
        config_path=config,
        output_root=out,
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_REWARD_ADVANTAGE


def test_stage21_10_rejects_auc_worst_seed_regression(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    out = root / "out"
    _write_stage21_6_artifacts(out / "stage21_6", raw_delta=0.01, capped_delta=0.01, auc_min=-0.001)
    config = _write_config(root)

    summary = run_xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot(
        config_path=config,
        output_root=out,
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_REWARD_ADVANTAGE
    assert summary["min_coverage_auc_delta"] == -0.001


def test_stage21_10_rejects_capped_auc_worst_seed_regression(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    out = root / "out"
    _write_stage21_6_artifacts(out / "stage21_6", raw_delta=0.01, capped_delta=0.01, capped_auc_min=-0.001)
    config = _write_config(root)

    summary = run_xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot(
        config_path=config,
        output_root=out,
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_REWARD_ADVANTAGE
    assert summary["min_coverage_curve_auc_capped_delta"] == -0.001


def test_stage21_10_rejects_stage21_6_execution_boundary_counts(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    out = root / "out"
    _write_stage21_6_artifacts(out / "stage21_6", raw_delta=0.01, capped_delta=0.01, execution_boundary_violation_total=1)
    config = _write_config(root)

    summary = run_xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot(
        config_path=config,
        output_root=out,
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_BOUNDARY
    assert "stage21_6_execution_boundary_violation_total_nonzero" in summary["boundary_reason_codes"]


def test_stage21_10_requires_stage21_9_passed_route(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path, stage21_9_status="failed")
    config = _write_config(root)

    summary = run_xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_INPUTS
    assert "stage21_9_not_passed" in summary["input_reason_codes"]


def test_stage21_10_boundary_flags_hard_fail(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    config = _write_config(root, starts_online_canary=True)

    summary = run_xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_BOUNDARY
    assert "stage21_10_config_starts_online_canary_true" in summary["boundary_reason_codes"]


def test_stage21_10_rejects_failed_stage21_6_lineage(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    out = root / "out"
    _write_stage21_6_artifacts(out / "stage21_6", raw_delta=0.01, capped_delta=0.01, lineage_passed=False)
    config = _write_config(root)

    summary = run_xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot(
        config_path=config,
        output_root=out,
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_INPUTS
    assert summary["primary_reason"] == "stage21_6_lineage_not_passed"


def _fixture_root(tmp_path: Path, *, stage21_9_status: str = "passed") -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    stage21_9 = root / "stage21_9"
    stage21_9.mkdir()
    repaired4 = stage21_9 / "repaired_stage21_4.json"
    _json(
        repaired4,
        {
            "schema_version": "xunce-stage21-4-tiny-ppo-update-smoke-config/v1",
            "learning_rate": 0.00001,
            "epochs": 2,
            "clip_ratio": 0.2,
            "max_grad_norm": 1.0,
        },
    )
    _json(
        stage21_9 / "xunce-stage21-9-diagnostic-summary.json",
        {
            "status": stage21_9_status,
            "next_required_change": "rerun_stage21_6_multi_seed_pilot_with_stage21_9_repaired_config",
            "repaired_pre_clip_grad_norm_max": 2.3,
            "repaired_stage21_6_pre_clip_grad_norm_gate": 25.0,
            "repaired_numerically_stable": True,
            "repaired_policy_shift_observable": True,
            "repaired_coverage_auc_not_regressed": True,
            "repaired_stage21_4_config": str(repaired4),
        },
    )
    _json(
        root / "stage21_7_repaired_stage21_6_config.json",
        {
            "schema_version": "xunce-stage21-6-multi-seed-ppo-pilot-config/v1",
            "stage21_4_base_config": "old-stage21-4.json",
            "seed_list": [1],
            "required_scenario_count": 1,
            "rollout_steps": 1,
            "dynamic_max_candidates_per_step": 6,
            "dynamic_proposal_pool_limit_per_step": 48,
            "publishes_checkpoint": True,
        },
    )
    return root


def _write_config(root: Path, **overrides: object) -> Path:
    payload = {
        "schema_version": "xunce-stage21-10-stage21-9-repaired-multi-seed-ppo-pilot-config/v1",
        "stage21_9_root": str(root / "stage21_9"),
        "stage21_7_repaired_stage21_6_config": str(root / "stage21_7_repaired_stage21_6_config.json"),
        "execute_stage21_6": False,
        "reuse_existing_stage21_6_result": True,
        "seed_list": [2101, 2102, 2103],
        "required_scenario_count": 8,
        "rollout_steps": 10,
        "dynamic_max_candidates_per_step": 36,
        "dynamic_proposal_pool_limit_per_step": 288,
        "learning_rate": 0.000002,
        "epochs": 1,
        "clip_ratio": 0.2,
        "stage21_4_max_grad_norm": 1.0,
        "stage21_6_pre_clip_grad_norm_gate": 25.0,
        "advantage_clip_abs": 5.0,
        "normalize_minibatch_advantages": True,
        "loss_scale": 0.25,
        "value_loss_coefficient": 0.1,
        "max_abs_approx_kl": 0.5,
        "min_entropy": 0.0,
        "stage21_10_authorized": False,
        "training_or_release_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = root / "stage21_10_config.json"
    _json(path, payload)
    return path


def _write_stage21_6_artifacts(
    root: Path,
    *,
    raw_delta: float,
    capped_delta: float,
    lineage_passed: bool = True,
    summary_status: str | None = None,
    auc_min: float | None = None,
    capped_auc_min: float | None = None,
    execution_boundary_violation_total: int = 0,
) -> None:
    root.mkdir(parents=True)
    status = summary_status if summary_status is not None else ("passed" if raw_delta > 0 and capped_delta > 0 else "failed")
    auc_min = raw_delta if auc_min is None else auc_min
    capped_auc_min = capped_delta if capped_auc_min is None else capped_auc_min
    _json(
        root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json",
        {
            "status": status,
            "next_required_change": "prepare_stage22_formal_pure_ppo_training_run",
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
            "coverage_auc_delta_min": auc_min,
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
    _json(root / "xunce-stage21-6-lineage-audit.json", {"passed": lineage_passed, "reason_codes": [] if lineage_passed else ["duplicate_sampling_seed"]})
    seed_rows = []
    for index, seed in enumerate((2101, 2102, 2103)):
        stage21_5_root = root / f"seed_{seed}" / "stage21_5"
        stage21_5_root.mkdir(parents=True)
        _json(
            stage21_5_root / "xunce-stage21-5-post-update-evaluation-summary.json",
            {
                "final_coverage_rate_delta": raw_delta,
                "coverage_curve_auc_delta": raw_delta,
                "final_coverage_rate_capped_delta": capped_delta,
                "coverage_curve_auc_capped_delta": capped_auc_min if index == 0 else capped_delta,
            },
        )
        seed_rows.append(
            {
                "seed": seed,
                "stage21_5_root": str(stage21_5_root),
                "pre_clip_grad_norm": 2.0,
                "post_clip_grad_norm": 1.0,
                "max_post_update_approx_kl": 0.001,
                "min_entropy": 1.0,
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
    _jsonl(root / "xunce-stage21-6-seed-results.jsonl", seed_rows)


def _json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
