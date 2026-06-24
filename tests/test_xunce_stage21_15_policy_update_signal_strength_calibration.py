from __future__ import annotations

import json
import subprocess
from pathlib import Path

import scripts.run_xunce_stage21_15_policy_update_signal_strength_calibration as stage21_15
from scripts.run_xunce_stage21_15_policy_update_signal_strength_calibration import (
    ROUTE_BINDING,
    ROUTE_CREDIT,
    ROUTE_INPUTS,
    ROUTE_MARGIN,
    ROUTE_SCALE,
    ROUTE_SIGNAL,
    run_xunce_stage21_15_policy_update_signal_strength_calibration,
)


def test_stage21_15_routes_to_binding_repair_when_strong_fields_are_missing(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    combo_root = root / "sweep" / "e8_lr2e-6_c0p2" / "stage21_6"
    _write_stage21_6_result(combo_root, probability_delta=0.04, coverage_delta=0.0, strong_fields=False)
    config = _write_config(root, execute_sweep=False)

    summary = run_xunce_stage21_15_policy_update_signal_strength_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_BINDING
    assert summary["strong_state_join_available_count"] == 0
    audit = json.loads((root / "out" / "xunce-stage21-15-strong-state-binding-audit.json").read_text())
    assert audit["strong_state_binding_unavailable_count"] == 3


def test_stage21_15_routes_to_binding_repair_when_probability_vector_shape_is_invalid(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    combo_root = root / "sweep" / "e8_lr2e-6_c0p2" / "stage21_6"
    _write_stage21_6_result(combo_root, probability_delta=0.04, coverage_delta=0.0, malformed_probs=True)
    config = _write_config(root, execute_sweep=False)

    summary = run_xunce_stage21_15_policy_update_signal_strength_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_BINDING
    assert summary["strong_state_join_available_count"] == 0


def test_stage21_15_routes_to_signal_strength_when_probabilities_are_tiny(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    combo_root = root / "sweep" / "e8_lr2e-6_c0p2" / "stage21_6"
    _write_stage21_6_result(combo_root, probability_delta=0.0001, coverage_delta=0.0)
    config = _write_config(root, execute_sweep=False)

    summary = run_xunce_stage21_15_policy_update_signal_strength_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_SIGNAL
    recommended = json.loads(Path(summary["recommended_stage21_6_config"]).read_text())
    recommended_stage21_4 = json.loads(Path(recommended["stage21_4_base_config"]).read_text())
    assert recommended_stage21_4["learning_rate"] == 0.000005
    assert recommended_stage21_4["epochs"] == 8
    rows = _read_jsonl(root / "out" / "xunce-stage21-15-policy-signal-sweep-results.jsonl")
    assert rows[0]["mean_abs_probability_delta"] < 0.005


def test_stage21_15_recommends_extrapolated_signal_config_when_all_priority_combos_are_done(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    for combo_id, lr, delta in (
        ("e8_lr2e-6_c0p2", 0.000002, 0.00001),
        ("e8_lr5e-6_c0p2", 0.000005, 0.00002),
        ("e8_lr1e-5_c0p2", 0.00001, 0.00005),
    ):
        combo_root = root / "sweep" / combo_id / "stage21_6"
        _write_stage21_6_result(combo_root, probability_delta=delta, coverage_delta=0.0)
    config = _write_config(
        root,
        execute_sweep=False,
        priority_combinations=[
            {"combo_id": "e8_lr2e-6_c0p2", "epochs": 8, "learning_rate": 0.000002, "clip_ratio": 0.2},
            {"combo_id": "e8_lr5e-6_c0p2", "epochs": 8, "learning_rate": 0.000005, "clip_ratio": 0.2},
            {"combo_id": "e8_lr1e-5_c0p2", "epochs": 8, "learning_rate": 0.00001, "clip_ratio": 0.2},
        ],
    )

    summary = run_xunce_stage21_15_policy_update_signal_strength_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_SIGNAL
    assert summary["recommendation_type"] == "next_extrapolated_policy_signal_combo"
    recommended = json.loads(Path(summary["recommended_stage21_6_config"]).read_text())
    recommended_stage21_4 = json.loads(Path(recommended["stage21_4_base_config"]).read_text())
    assert recommended_stage21_4["learning_rate"] == 0.00002


def test_stage21_15_routes_to_margin_when_probabilities_change_but_rank_does_not(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    combo_root = root / "sweep" / "e8_lr2e-6_c0p2" / "stage21_6"
    _write_stage21_6_result(combo_root, probability_delta=0.03, coverage_delta=0.0, action_shift=False)
    config = _write_config(root, execute_sweep=False)

    summary = run_xunce_stage21_15_policy_update_signal_strength_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_MARGIN
    assert summary["argmax_changed_count"] == 0
    assert summary["selected_rank_changed_count"] == 0


def test_stage21_15_routes_to_credit_when_rank_changes_without_trajectory_gain(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    combo_root = root / "sweep" / "e8_lr2e-6_c0p2" / "stage21_6"
    _write_stage21_6_result(combo_root, probability_delta=0.04, coverage_delta=0.0, action_shift=True)
    config = _write_config(root, execute_sweep=False)

    summary = run_xunce_stage21_15_policy_update_signal_strength_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_CREDIT
    assert summary["argmax_changed_count"] == 3
    assert summary["selected_rank_changed_count"] == 3


def test_stage21_15_routes_to_scale_when_coverage_and_auc_improve(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    combo_root = root / "sweep" / "e8_lr2e-6_c0p2" / "stage21_6"
    _write_stage21_6_result(combo_root, probability_delta=0.04, coverage_delta=0.02, action_shift=True, status="passed")
    config = _write_config(root, execute_sweep=False)

    summary = run_xunce_stage21_15_policy_update_signal_strength_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == ROUTE_SCALE
    assert Path(summary["recommended_stage21_6_config"]).is_file()


def test_stage21_15_rejects_missing_stage21_14_summary(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path, write_stage21_14=False)
    config = _write_config(root)

    summary = run_xunce_stage21_15_policy_update_signal_strength_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_INPUTS
    assert "missing_stage21_14_summary" in summary["input_reason_codes"]


def test_stage21_15_generates_combo_configs_for_offline_smoke(monkeypatch, tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    config = _write_config(root, execute_sweep=True, max_new_combinations_to_execute=1)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(stage21_15.subprocess, "run", fake_run)

    run_xunce_stage21_15_policy_update_signal_strength_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    stage21_4 = json.loads((root / "sweep" / "e8_lr2e-6_c0p2" / "stage21_4_config.json").read_text())
    stage21_6 = json.loads((root / "sweep" / "e8_lr2e-6_c0p2" / "stage21_6_config.json").read_text())
    assert stage21_4["epochs"] == 8
    assert stage21_4["learning_rate"] == 0.000002
    assert stage21_6["runs_new_ppo_update"] is True
    assert stage21_6["publishes_checkpoint"] is False
    assert stage21_6["replaces_default_policy"] is False
    assert stage21_6["connects_real_executor"] is False
    assert stage21_6["starts_online_canary"] is False


def _fixture_root(tmp_path: Path, *, write_stage21_14: bool = True) -> Path:
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
    stage21_14 = root / "stage21_14"
    stage21_14.mkdir()
    if write_stage21_14:
        _json(
            stage21_14 / "xunce-stage21-14-summary.json",
            {
                "status": "failed",
                "next_required_change": "calibrate_stage21_policy_update_signal_strength",
                "completed_stage21_14_sweep_combo_count": 3,
                "strong_state_binding_unavailable_combo_count": 3,
                "recommended_stage21_6_config": str(root / "configs" / "stage21_6.json"),
            },
        )
    return root


def _write_config(root: Path, **overrides: object) -> Path:
    payload = {
        "schema_version": "xunce-stage21-15-policy-update-signal-strength-calibration-config/v1",
        "stage21_14_root": str(root / "stage21_14"),
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
        "loss_scale": 0.25,
        "value_loss_coefficient": 0.1,
        "advantage_clip_abs": 5.0,
        "normalize_minibatch_advantages": True,
        "probability_delta_threshold": 0.005,
        "priority_combinations": [
            {"combo_id": "e8_lr2e-6_c0p2", "epochs": 8, "learning_rate": 0.000002, "clip_ratio": 0.2},
            {"combo_id": "e8_lr5e-6_c0p2", "epochs": 8, "learning_rate": 0.000005, "clip_ratio": 0.2},
        ],
        "stage21_15_authorized": False,
        "training_or_release_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = root / "stage21_15_config.json"
    _json(path, payload)
    return path


def _write_stage21_6_result(
    root: Path,
    *,
    probability_delta: float,
    coverage_delta: float,
    action_shift: bool = False,
    strong_fields: bool = True,
    malformed_probs: bool = False,
    status: str = "failed",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _json(
        root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json",
        {
            "status": status,
            "next_required_change": "repair_stage21_6_reward_collector_advantage_or_horizon",
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
    rows = []
    for seed in (2101, 2102, 2103):
        stage21_5 = root / f"seed_{seed}" / "stage21_5"
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
        _jsonl(
            pre / "xunce-exploration-coverage-model-inference.jsonl",
            [_inference_row(0, 1, 0.50, strong_fields, malformed_probs=malformed_probs)],
        )
        _jsonl(
            post / "xunce-exploration-coverage-model-inference.jsonl",
            [
                _inference_row(
                    1 if action_shift else 0,
                    2 if action_shift else 1,
                    0.50 + probability_delta,
                    strong_fields,
                    malformed_probs=malformed_probs,
                )
            ],
        )
        _jsonl(pre / "xunce-exploration-coverage-candidate-metric-audit.jsonl", _candidate_rows(strong_fields))
        rows.append(
            {
                "seed": seed,
                "stage21_5_root": str(stage21_5),
                "pre_clip_grad_norm": 4.0,
                "post_clip_grad_norm": 1.0,
                "max_post_update_approx_kl": 0.001,
                "min_entropy": 3.0,
                "grad_norm_finite": True,
                "parameter_delta_l2": 0.002,
            }
        )
    _jsonl(root / "xunce-stage21-6-seed-results.jsonl", rows)


def _inference_row(
    index: int,
    rank: int,
    selected_probability: float,
    strong_fields: bool,
    *,
    malformed_probs: bool = False,
) -> dict[str, object]:
    action_probs = (
        [selected_probability, 1.0 - selected_probability]
        if index == 0
        else [1.0 - selected_probability, selected_probability]
    )
    if malformed_probs:
        action_probs = action_probs[:1]
    row: dict[str, object] = {
        "scenario_id": "scenario-a",
        "step_index": 0,
        "candidate_cells": [[0, 1], [1, 0]],
        "candidate_set_hash": "candidate-hash",
        "action_mask": [True, True],
        "selected_action_index": index,
        "selected_rank": rank,
        "action_probs": action_probs,
        "logits": [0.0, 0.0],
        "masked_logits": [0.0, 0.0],
        "value": 0.0,
        "finite_outputs": True,
        "latency_ms": 0.1,
        "detail": {
            "selected_action_index": index,
            "selected_rank": rank,
            "selected_probability": selected_probability,
            "action_probs": action_probs,
            "logits": [0.0, 0.0],
            "masked_logits": [0.0, 0.0],
            "value": 0.0,
            "finite_outputs": True,
            "latency_ms": 0.1,
        },
    }
    if strong_fields:
        row.update({"current_cell": [0, 0], "current_cell_before": [0, 0], "covered_cells_hash": "covered-hash"})
    return row


def _candidate_rows(strong_fields: bool) -> list[dict[str, object]]:
    base = {
        "scenario_id": "scenario-a",
        "step_index": 0,
        "candidate_set_hash": "candidate-hash",
        "action_mask_valid": True,
    }
    if strong_fields:
        base.update({"current_cell": [0, 0], "covered_cells_hash": "covered-hash"})
    return [
        {**base, "candidate_index": 0, "coverage_gain_per_path_cost": 1.0},
        {**base, "candidate_index": 1, "coverage_gain_per_path_cost": 3.0},
    ]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
