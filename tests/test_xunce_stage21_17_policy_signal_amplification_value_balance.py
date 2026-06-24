from __future__ import annotations

import json
import subprocess
from pathlib import Path

import scripts.run_xunce_stage21_17_policy_signal_amplification_value_balance as stage21_17
from scripts.run_xunce_stage21_17_policy_signal_amplification_value_balance import (
    ROUTE_BOUNDARY,
    ROUTE_INPUTS,
    ROUTE_MARGIN,
    ROUTE_POLICY_MULTIPLIER,
    ROUTE_SCALE,
    ROUTE_VALUE_BALANCE,
    run_xunce_stage21_17_policy_signal_amplification_value_balance,
)


def test_stage21_17_routes_to_value_balance_when_value_loss_still_dominates(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_6_result(
        root / "sweep" / "v0p05_p1p0_loss0p25" / "stage21_6",
        probability_delta=0.02,
        policy_grad=0.1,
        value_grad=3.0,
    )
    summary = _run(root)

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_VALUE_BALANCE
    assert summary["value_to_policy_grad_norm_ratio_max"] > 3.0
    recommended = json.loads(Path(summary["recommended_stage21_6_config"]).read_text(encoding="utf-8"))
    stage21_4 = json.loads(Path(recommended["stage21_4_base_config"]).read_text(encoding="utf-8"))
    assert stage21_4["value_loss_coefficient"] == 0.02
    assert stage21_4["policy_loss_coefficient"] == 1.0


def test_stage21_17_routes_to_policy_multiplier_when_balanced_but_probability_tiny(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_6_result(
        root / "sweep" / "v0p05_p1p0_loss0p25" / "stage21_6",
        probability_delta=0.0001,
        policy_grad=1.0,
        value_grad=1.0,
    )
    summary = _run(root)

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_POLICY_MULTIPLIER
    assert summary["mean_abs_probability_delta_max"] < 0.005
    routing = _read_json(root / "out" / "xunce-stage21-17-next-stage-routing.json")
    assert routing["stage21_17_authorized"] is False
    assert routing["stage21_authorized"] is False
    assert routing["training_or_release_authorized"] is False


def test_stage21_17_uses_best_balanced_combo_instead_of_worst_prior_combo(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_6_result(
        root / "sweep" / "v0p05_p1p0_loss0p25" / "stage21_6",
        probability_delta=0.0001,
        policy_grad=0.1,
        value_grad=3.0,
    )
    _write_stage21_6_result(
        root / "sweep" / "v0p02_p1p0_loss0p25" / "stage21_6",
        probability_delta=0.0002,
        policy_grad=1.0,
        value_grad=1.0,
    )

    summary = _run(root)

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_POLICY_MULTIPLIER
    assert summary["best_balanced_combo_id"] == "v0p02_p1p0_loss0p25"


def test_stage21_17_routes_to_margin_when_probability_moves_but_rank_does_not(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_6_result(
        root / "sweep" / "v0p05_p1p0_loss0p25" / "stage21_6",
        probability_delta=0.04,
        policy_grad=1.0,
        value_grad=1.0,
        action_shift=False,
    )
    summary = _run(root)

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_MARGIN
    assert summary["argmax_changed_count"] == 0


def test_stage21_17_routes_to_scale_when_coverage_and_auc_improve(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_6_result(
        root / "sweep" / "v0p05_p1p0_loss0p25" / "stage21_6",
        probability_delta=0.04,
        coverage_delta=0.02,
        action_shift=True,
        policy_grad=1.0,
        value_grad=1.0,
        status="passed",
    )
    summary = _run(root)

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == ROUTE_SCALE


def test_stage21_17_rejects_missing_stage21_16_summary(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path, write_stage21_16=False)
    summary = _run(root)

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_INPUTS
    assert "missing_stage21_16_summary" in summary["input_reason_codes"]


def test_stage21_17_rejects_boundary_flags(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    summary = run_xunce_stage21_17_policy_signal_amplification_value_balance(
        config_path=_write_config(root, publishes_checkpoint=True),
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_BOUNDARY
    assert "stage21_17_config_publishes_checkpoint_true" in summary["boundary_reason_codes"]


def test_stage21_17_generates_combo_configs_with_policy_value_coefficients(monkeypatch, tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    config = _write_config(root, execute_sweep=True, max_new_combinations_to_execute=1)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(stage21_17.subprocess, "run", fake_run)

    run_xunce_stage21_17_policy_signal_amplification_value_balance(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    stage21_4 = json.loads((root / "sweep" / "v0p05_p1p0_loss0p25" / "stage21_4_config.json").read_text(encoding="utf-8"))
    stage21_6 = json.loads((root / "sweep" / "v0p05_p1p0_loss0p25" / "stage21_6_config.json").read_text(encoding="utf-8"))
    assert stage21_4["value_loss_coefficient"] == 0.05
    assert stage21_4["policy_loss_coefficient"] == 1.0
    assert stage21_4["learning_rate"] == 0.00002
    assert stage21_4["epochs"] == 8
    assert stage21_6["runs_new_ppo_update"] is True
    assert stage21_6["publishes_checkpoint"] is False
    assert stage21_6["replaces_default_policy"] is False
    assert stage21_6["connects_real_executor"] is False
    assert stage21_6["starts_online_canary"] is False


def _run(root: Path) -> dict[str, object]:
    return run_xunce_stage21_17_policy_signal_amplification_value_balance(
        config_path=_write_config(root, execute_sweep=False),
        output_root=root / "out",
        repo_root=root,
    )


def _fixture_root(tmp_path: Path, *, write_stage21_16: bool = True) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "configs").mkdir()
    _json(root / "configs" / "stage21_4.json", {"schema_version": "xunce-stage21-4-tiny-ppo-update-smoke-config/v1"})
    _json(
        root / "configs" / "stage21_6.json",
        {
            "schema_version": "xunce-stage21-6-multi-seed-ppo-pilot-config/v1",
            "stage21_4_base_config": str(root / "configs" / "stage21_4.json"),
            "seed_list": [2101, 2102, 2103],
            "required_scenario_count": 8,
            "rollout_steps": 10,
            "dynamic_max_candidates_per_step": 36,
            "dynamic_proposal_pool_limit_per_step": 288,
            "max_grad_norm": 25.0,
        },
    )
    stage21_16 = root / "stage21_16"
    stage21_16.mkdir()
    if write_stage21_16:
        _json(
            stage21_16 / "xunce-stage21-16-summary.json",
            {
                "status": "failed",
                "next_required_change": "increase_stage21_policy_update_signal_strength",
                "strong_state_join_available_count": 480,
                "strong_state_binding_unavailable_count": 0,
                "policy_grad_ratio_mean": 0.026,
                "value_to_policy_grad_norm_ratio_max": 69.0,
            },
        )
        _json(stage21_16 / "xunce-stage21-16-recommended-stage21-6-config.json", _read_json(root / "configs" / "stage21_6.json"))
        _json(stage21_16 / "xunce-stage21-16-recommended-stage21-4-config.json", _read_json(root / "configs" / "stage21_4.json"))
    return root


def _write_config(root: Path, **overrides: object) -> Path:
    payload = {
        "schema_version": "xunce-stage21-17-policy-signal-amplification-value-balance-config/v1",
        "stage21_16_root": str(root / "stage21_16"),
        "stage21_6_base_config": str(root / "configs" / "stage21_6.json"),
        "stage21_4_base_config": str(root / "configs" / "stage21_4.json"),
        "seed_list": [2101, 2102, 2103],
        "required_scenario_count": 8,
        "rollout_steps": 10,
        "dynamic_max_candidates_per_step": 36,
        "dynamic_proposal_pool_limit_per_step": 288,
        "execute_sweep": False,
        "sweep_work_root": str(root / "sweep"),
        "max_new_combinations_to_execute": 1,
        "per_combo_timeout_seconds": 7200,
        "stage21_6_pre_clip_grad_norm_gate": 25.0,
        "stage21_4_max_grad_norm": 1.0,
        "advantage_clip_abs": 5.0,
        "normalize_minibatch_advantages": True,
        "probability_delta_threshold": 0.005,
        "large_probability_margin_threshold": 0.05,
        "min_policy_grad_ratio": 0.1,
        "max_value_to_policy_grad_ratio": 3.0,
        "priority_combinations": [
            {
                "combo_id": "v0p05_p1p0_loss0p25",
                "epochs": 8,
                "learning_rate": 0.00002,
                "clip_ratio": 0.2,
                "loss_scale": 0.25,
                "value_loss_coefficient": 0.05,
                "policy_loss_coefficient": 1.0,
            },
            {
                "combo_id": "v0p02_p1p0_loss0p25",
                "epochs": 8,
                "learning_rate": 0.00002,
                "clip_ratio": 0.2,
                "loss_scale": 0.25,
                "value_loss_coefficient": 0.02,
                "policy_loss_coefficient": 1.0,
            },
        ],
        "stage21_17_authorized": False,
        "training_or_release_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = root / "stage21_17_config.json"
    _json(path, payload)
    return path


def _write_stage21_6_result(
    root: Path,
    *,
    probability_delta: float = 0.02,
    coverage_delta: float = 0.0,
    action_shift: bool = False,
    policy_grad: float = 1.0,
    value_grad: float = 1.0,
    status: str = "failed",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _json(
        root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json",
        {
            "status": status,
            "next_required_change": "repair_stage21_return_advantage_credit_assignment",
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
            "trainable_transition_count_total": 240,
            "final_coverage_delta_mean": coverage_delta,
            "final_coverage_delta_min": coverage_delta,
            "coverage_auc_delta_mean": coverage_delta,
            "coverage_auc_delta_min": coverage_delta,
            "pre_clip_grad_norm_max": 4.0,
            "max_post_update_approx_kl_mean": 0.001,
            "min_entropy_mean": 3.0,
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
        },
    )
    _json(root / "xunce-stage21-6-lineage-audit.json", {"passed": True})
    seed_rows = []
    for seed in (2101, 2102, 2103):
        seed_root = root / f"seed_{seed}"
        stage21_2 = seed_root / "stage21_2"
        stage21_3 = seed_root / "stage21_3"
        stage21_4 = seed_root / "stage21_4"
        stage21_5 = seed_root / "stage21_5"
        pre = stage21_5 / "pre_ppo_xunce"
        post = stage21_5 / "post_ppo_xunce"
        pre.mkdir(parents=True, exist_ok=True)
        post.mkdir(parents=True, exist_ok=True)
        _json(
            stage21_5 / "xunce-stage21-5-post-update-evaluation-summary.json",
            {
                "pre_evaluation_root": str(pre),
                "post_evaluation_root": str(post),
                "final_coverage_rate_capped_delta": coverage_delta,
                "coverage_curve_auc_capped_delta": coverage_delta,
            },
        )
        _jsonl(pre / "xunce-exploration-coverage-model-inference.jsonl", [_inference_row(0, 1, 0.60)])
        _jsonl(post / "xunce-exploration-coverage-model-inference.jsonl", [_inference_row(1 if action_shift else 0, 2 if action_shift else 1, 0.60 + probability_delta, best_delta=probability_delta)])
        _jsonl(pre / "xunce-exploration-coverage-candidate-metric-audit.jsonl", _candidate_rows())
        _jsonl(
            stage21_2 / "xunce-stage21-2-reward-contract-evaluation.jsonl",
            [
                {"reward": 1.0, "metrics": {"coverage_per_cost": 1.0}, "trainable": True},
                {"reward": 2.0, "metrics": {"coverage_per_cost": 2.0}, "trainable": True},
            ],
        )
        _jsonl(
            stage21_3 / "xunce-stage21-3-return-advantage-audit.jsonl",
            [
                {"transition_id": "a", "stage21_3_split": "train", "advantage": 1.0, "coverage_per_cost": 1.0},
                {"transition_id": "b", "stage21_3_split": "train", "advantage": 2.0, "coverage_per_cost": 2.0},
            ],
        )
        _json(
            stage21_4 / "xunce-stage21-4-gradient-audit.json",
            {
                "component_grad_norms": {
                    "policy_loss_grad_norm": policy_grad,
                    "value_loss_grad_norm": value_grad,
                    "entropy_loss_grad_norm": 0.01,
                    "total_loss_grad_norm": policy_grad + value_grad,
                },
                "pre_clip_grad_norm": policy_grad + value_grad,
                "post_clip_grad_norm": 1.0,
            },
        )
        seed_rows.append(
            {
                "seed": seed,
                "seed_root": str(seed_root),
                "stage21_2_root": str(stage21_2),
                "stage21_3_root": str(stage21_3),
                "stage21_4_root": str(stage21_4),
                "stage21_5_root": str(stage21_5),
                "pre_clip_grad_norm": 4.0,
                "post_clip_grad_norm": 1.0,
                "max_post_update_approx_kl": 0.001,
                "min_entropy": 3.0,
                "grad_norm_finite": True,
                "parameter_delta_l2": 0.002,
                "final_coverage_delta": coverage_delta,
                "coverage_auc_delta": coverage_delta,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
                "training_or_release_authorized": False,
            }
        )
    _jsonl(root / "xunce-stage21-6-seed-results.jsonl", seed_rows)


def _inference_row(selected: int, rank: int, top_probability: float, *, best_delta: float = 0.0) -> dict[str, object]:
    probs = [top_probability, 0.30 + best_delta, max(0.0, 0.10 - best_delta)]
    if selected == 1:
        probs = [0.35, top_probability, 0.05]
    return {
        "scenario_id": "scenario-a",
        "step_index": 0,
        "current_cell": [0, 0],
        "current_cell_before": [0, 0],
        "covered_cells_hash": "covered-hash",
        "candidate_set_hash": "candidate-hash",
        "candidate_cells": [[0, 1], [1, 0], [1, 1]],
        "action_mask": [True, True, True],
        "selected_action_index": selected,
        "selected_rank": rank,
        "selected_probability": probs[selected],
        "action_probs": probs,
        "logits": [2.0, 1.0, 0.0],
        "masked_logits": [2.0, 1.0, 0.0],
        "value": 0.0,
        "finite_outputs": True,
        "latency_ms": 0.1,
    }


def _candidate_rows() -> list[dict[str, object]]:
    base = {
        "scenario_id": "scenario-a",
        "step_index": 0,
        "current_cell": [0, 0],
        "covered_cells_hash": "covered-hash",
        "candidate_set_hash": "candidate-hash",
        "action_mask_valid": True,
    }
    return [
        {**base, "candidate_index": 0, "coverage_gain_per_path_cost": 1.0, "expected_new_coverage_cell_count": 10.0, "path_cost": 10.0},
        {**base, "candidate_index": 1, "coverage_gain_per_path_cost": 3.0, "expected_new_coverage_cell_count": 12.0, "path_cost": 4.0},
        {**base, "candidate_index": 2, "coverage_gain_per_path_cost": 0.5, "expected_new_coverage_cell_count": 1.0, "path_cost": 2.0},
    ]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
