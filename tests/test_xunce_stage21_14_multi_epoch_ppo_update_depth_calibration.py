from __future__ import annotations

import json
import subprocess
from pathlib import Path

import scripts.run_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration as stage21_14
from scripts.run_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration import (
    ROUTE_BOUNDARY,
    ROUTE_CREDIT,
    ROUTE_INPUTS,
    ROUTE_POLICY_SIGNAL,
    ROUTE_RUNTIME,
    ROUTE_SCALE,
    run_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration,
)


def test_stage21_14_routes_to_credit_when_probabilities_shift_but_coverage_does_not(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    out = root / "out"
    combo_root = root / "sweep" / "e2_lr2e-6_c0p2" / "stage21_6"
    _write_stage21_6_result(combo_root, action_shift=True, prob_delta=0.03, coverage_delta=0.0)
    config = _write_config(root, execute_sweep=False)

    summary = run_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration(
        config_path=config,
        output_root=out,
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_CREDIT
    assert summary["completed_stage21_14_sweep_combo_count"] == 1
    assert summary["max_argmax_changed_count"] == 3
    assert summary["max_selected_rank_changed_count"] == 3
    rows = _read_jsonl(out / "xunce-stage21-14-sweep-results.jsonl")
    assert rows[0]["epochs"] == 2
    assert rows[0]["action_rank_shift_observable"] is True
    assert rows[0]["strong_state_binding_unavailable_count"] == 0
    assert rows[0]["best_coverage_per_cost_probability_delta_mean"] > 0


def test_stage21_14_routes_to_policy_signal_when_stable_but_action_shift_is_tiny(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    combo_root = root / "sweep" / "e2_lr2e-6_c0p2" / "stage21_6"
    _write_stage21_6_result(combo_root, action_shift=False, prob_delta=0.0, coverage_delta=0.0)
    config = _write_config(root, execute_sweep=False)

    summary = run_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_POLICY_SIGNAL
    audit = json.loads((root / "out" / "xunce-stage21-14-action-rank-shift-audit.json").read_text())
    assert audit["observable_action_shift_combo_count"] == 0


def test_stage21_14_routes_to_scale_when_coverage_and_auc_improve(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    combo_root = root / "sweep" / "e2_lr2e-6_c0p2" / "stage21_6"
    _write_stage21_6_result(combo_root, action_shift=True, prob_delta=0.04, coverage_delta=0.02, status="passed")
    config = _write_config(root, execute_sweep=False)

    summary = run_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == ROUTE_SCALE
    recommended = json.loads(Path(summary["recommended_stage21_6_config"]).read_text())
    assert recommended["max_grad_norm"] == 25.0
    recommended_stage21_4 = json.loads((root / "out" / "xunce-stage21-14-recommended-stage21-4-config.json").read_text())
    assert recommended_stage21_4["epochs"] == 2
    assert recommended_stage21_4["learning_rate"] == 0.000002


def test_stage21_14_routes_to_scale_on_real_coverage_improvement_even_without_strong_action_join(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    combo_root = root / "sweep" / "e2_lr2e-6_c0p2" / "stage21_6"
    _write_stage21_6_result(
        combo_root,
        action_shift=True,
        prob_delta=0.04,
        coverage_delta=0.02,
        status="passed",
        strong_fields=False,
    )
    config = _write_config(root, execute_sweep=False)

    summary = run_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == ROUTE_SCALE
    rows = _read_jsonl(root / "out" / "xunce-stage21-14-sweep-results.jsonl")
    assert rows[0]["coverage_auc_improved"] is True
    assert rows[0]["action_rank_shift_observable"] is False


def test_stage21_14_records_strong_state_binding_unavailable_without_using_weak_join_as_gate(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    combo_root = root / "sweep" / "e2_lr2e-6_c0p2" / "stage21_6"
    _write_stage21_6_result(combo_root, action_shift=True, prob_delta=0.04, coverage_delta=0.0, strong_fields=False)
    config = _write_config(root, execute_sweep=False)

    summary = run_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_POLICY_SIGNAL
    assert summary["primary_reason"] == "strong_state_binding_unavailable_for_action_shift_audit"
    rows = _read_jsonl(root / "out" / "xunce-stage21-14-sweep-results.jsonl")
    assert rows[0]["strong_state_binding_unavailable_count"] == 3
    assert rows[0]["strong_state_join_available"] is False
    assert rows[0]["action_rank_shift_observable"] is False


def test_stage21_14_requires_candidate_set_hash_for_strong_action_join(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    combo_root = root / "sweep" / "e2_lr2e-6_c0p2" / "stage21_6"
    _write_stage21_6_result(combo_root, action_shift=True, prob_delta=0.04, coverage_delta=0.0)
    for path in combo_root.glob("seed_*/stage21_5/*_ppo_xunce/xunce-exploration-coverage-model-inference.jsonl"):
        rows = _read_jsonl(path)
        for row in rows:
            row.pop("candidate_set_hash", None)
        _jsonl(path, rows)
    config = _write_config(root, execute_sweep=False)

    summary = run_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_POLICY_SIGNAL
    assert summary["primary_reason"] == "strong_state_binding_unavailable_for_action_shift_audit"
    rows = _read_jsonl(root / "out" / "xunce-stage21-14-sweep-results.jsonl")
    assert rows[0]["policy_shift_matched_step_count"] == 0
    assert rows[0]["strong_state_binding_unavailable_count"] == 3


def test_stage21_14_cannot_scale_before_required_core_combos_complete(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    combo_root = root / "sweep" / "e2_lr2e-6_c0p2" / "stage21_6"
    _write_stage21_6_result(combo_root, action_shift=True, prob_delta=0.04, coverage_delta=0.02, status="passed")
    config = _write_config(
        root,
        execute_sweep=False,
        required_core_combo_count=3,
        priority_combinations=[
            {"combo_id": "e2_lr2e-6_c0p2", "epochs": 2, "learning_rate": 0.000002, "clip_ratio": 0.2},
            {"combo_id": "e4_lr2e-6_c0p2", "epochs": 4, "learning_rate": 0.000002, "clip_ratio": 0.2},
            {"combo_id": "e8_lr2e-6_c0p2", "epochs": 8, "learning_rate": 0.000002, "clip_ratio": 0.2},
        ],
    )

    summary = run_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "partial"
    assert summary["next_required_change"] == ROUTE_RUNTIME
    assert summary["primary_reason"] == "required_core_stage21_14_sweep_incomplete"


def test_stage21_14_rejects_open_boundary_flags(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    config = _write_config(root, starts_online_canary=True)

    summary = run_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_BOUNDARY
    assert "stage21_14_config_starts_online_canary_true" in summary["boundary_reason_codes"]


def test_stage21_14_rejects_missing_stage21_13_summary(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path, write_stage21_13=False)
    config = _write_config(root)

    summary = run_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_INPUTS
    assert "missing_stage21_13_summary" in summary["input_reason_codes"]


def test_stage21_14_records_runtime_blocker_on_timeout(monkeypatch, tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    config = _write_config(root, execute_sweep=True, max_new_combinations_to_execute=1, per_combo_timeout_seconds=1)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args", "cmd"), timeout=1, output="out", stderr="err")

    monkeypatch.setattr(stage21_14.subprocess, "run", fake_run)

    summary = run_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "partial"
    assert summary["next_required_change"] == ROUTE_RUNTIME
    assert (root / "sweep" / "e2_lr2e-6_c0p2" / "runtime-blocker.json").is_file()
    stage21_4 = json.loads((root / "sweep" / "e2_lr2e-6_c0p2" / "stage21_4_config.json").read_text())
    stage21_6 = json.loads((root / "sweep" / "e2_lr2e-6_c0p2" / "stage21_6_config.json").read_text())
    assert stage21_4["epochs"] == 2
    assert stage21_4["learning_rate"] == 0.000002
    assert stage21_4["clip_ratio"] == 0.2
    assert stage21_4["loss_scale"] == 0.25
    assert stage21_4["value_loss_coefficient"] == 0.1
    assert stage21_4["advantage_clip_abs"] == 5.0
    assert stage21_4["normalize_minibatch_advantages"] is True
    assert stage21_4["max_grad_norm"] == 1.0
    assert stage21_6["stage21_4_base_config"] == str(root / "sweep" / "e2_lr2e-6_c0p2" / "stage21_4_config.json")
    assert stage21_6["max_grad_norm"] == 25.0
    assert stage21_6["runs_new_ppo_update"] is True


def test_stage21_14_does_not_execute_optional_combos_after_core_count_is_complete(monkeypatch, tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    combo_root = root / "sweep" / "e2_lr2e-6_c0p2" / "stage21_6"
    _write_stage21_6_result(combo_root, action_shift=False, prob_delta=0.0, coverage_delta=0.0)
    config = _write_config(
        root,
        execute_sweep=True,
        required_core_combo_count=1,
        max_new_combinations_to_execute=2,
        priority_combinations=[
            {"combo_id": "e2_lr2e-6_c0p2", "epochs": 2, "learning_rate": 0.000002, "clip_ratio": 0.2},
            {"combo_id": "e4_lr5e-6_c0p2", "epochs": 4, "learning_rate": 0.000005, "clip_ratio": 0.2},
        ],
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("optional combo should not execute after required core combo count is complete")

    monkeypatch.setattr(stage21_14.subprocess, "run", fail_if_called)

    summary = run_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["completed_stage21_14_sweep_combo_count"] == 1
    assert not (root / "sweep" / "e4_lr5e-6_c0p2").exists()


def test_stage21_14_executes_missing_core_combo_but_not_following_optional_combo(monkeypatch, tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_stage21_6_result(root / "sweep" / "e2_lr2e-6_c0p2" / "stage21_6", action_shift=False, prob_delta=0.0, coverage_delta=0.0)
    _write_stage21_6_result(root / "sweep" / "e4_lr2e-6_c0p2" / "stage21_6", action_shift=False, prob_delta=0.0, coverage_delta=0.0)
    config = _write_config(
        root,
        execute_sweep=True,
        required_core_combo_count=3,
        max_new_combinations_to_execute=3,
        priority_combinations=[
            {"combo_id": "e2_lr2e-6_c0p2", "epochs": 2, "learning_rate": 0.000002, "clip_ratio": 0.2},
            {"combo_id": "e4_lr2e-6_c0p2", "epochs": 4, "learning_rate": 0.000002, "clip_ratio": 0.2},
            {"combo_id": "e8_lr2e-6_c0p2", "epochs": 8, "learning_rate": 0.000002, "clip_ratio": 0.2},
            {"combo_id": "e4_lr5e-6_c0p2", "epochs": 4, "learning_rate": 0.000005, "clip_ratio": 0.2},
        ],
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(stage21_14.subprocess, "run", fake_run)

    run_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert len(calls) == 1
    assert "e8_lr2e-6_c0p2" in " ".join(str(part) for part in calls[0])
    assert not (root / "sweep" / "e4_lr5e-6_c0p2").exists()


def _fixture_root(tmp_path: Path, *, write_stage21_13: bool = True) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "configs").mkdir()
    _json(root / "configs" / "stage21_4.json", {"schema_version": "xunce-stage21-4-tiny-ppo-update-smoke-config/v1"})
    _json(
        root / "configs" / "stage21_6.json",
        {
            "schema_version": "xunce-stage21-6-multi-seed-ppo-pilot-config/v1",
            "stage21_2_base_config": str(root / "configs" / "stage21_2.json"),
            "stage21_4_base_config": str(root / "configs" / "stage21_4.json"),
            "reward_profile": "configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json",
            "seed_list": [2101, 2102, 2103],
            "required_scenario_count": 8,
            "rollout_steps": 10,
            "dynamic_max_candidates_per_step": 36,
            "dynamic_proposal_pool_limit_per_step": 288,
            "max_grad_norm": 25.0,
        },
    )
    stage21_12 = root / "stage21_12"
    stage21_12.mkdir()
    _json(
        stage21_12 / "xunce-stage21-12-summary.json",
        {
            "status": "passed",
            "next_required_change": "run_stage21_13_coverage_constrained_multi_seed_ppo_smoke",
            "recommended_stage21_6_config": str(root / "configs" / "stage21_6.json"),
        },
    )
    stage21_13 = root / "stage21_13"
    stage21_13.mkdir()
    if write_stage21_13:
        _json(
            stage21_13 / "xunce-stage21-13-summary.json",
            {
                "status": "failed",
                "next_required_change": "repair_stage21_return_advantage_credit_assignment",
                "trainable_transition_count_total": 240,
                "pre_clip_grad_norm_max": 3.6,
                "max_post_update_approx_kl_mean": 0.0000028,
                "parameter_delta_l2_min": 0.00027,
                "mean_final_coverage_delta": 0.0,
                "mean_coverage_auc_delta": 0.0,
            },
        )
        _json(root / "stage21_13" / "xunce-stage21-13-stage21-6-config.json", json.loads((root / "configs" / "stage21_6.json").read_text()))
    return root


def _write_config(root: Path, **overrides: object) -> Path:
    payload = {
        "schema_version": "xunce-stage21-14-multi-epoch-ppo-update-depth-calibration-config/v1",
        "stage21_13_root": str(root / "stage21_13"),
        "stage21_12_root": str(root / "stage21_12"),
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
        "priority_combinations": [
            {"combo_id": "e2_lr2e-6_c0p2", "epochs": 2, "learning_rate": 0.000002, "clip_ratio": 0.2}
        ],
        "stage21_14_authorized": False,
        "training_or_release_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = root / "stage21_14_config.json"
    _json(path, payload)
    return path


def _write_stage21_6_result(
    root: Path,
    *,
    action_shift: bool,
    prob_delta: float,
    coverage_delta: float,
    status: str = "failed",
    strong_fields: bool = True,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _json(
        root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json",
        {
            "status": status,
            "next_required_change": "repair_stage21_return_advantage_credit_assignment",
            "reason_codes": [] if status == "passed" else ["coverage_auc_worst_seed_or_cost_rejected"],
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
            "final_coverage_delta_mean": coverage_delta,
            "final_coverage_delta_min": coverage_delta,
            "coverage_auc_delta_mean": coverage_delta,
            "coverage_auc_delta_min": coverage_delta,
            "pre_clip_grad_norm_mean": 3.0,
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
                "final_coverage_rate_delta": coverage_delta,
                "coverage_curve_auc_delta": coverage_delta,
                "final_coverage_rate_capped_delta": coverage_delta,
                "coverage_curve_auc_capped_delta": coverage_delta,
            },
        )
        _jsonl(pre / "xunce-exploration-coverage-model-inference.jsonl", [_inference_row(0, 1, 0.50, strong_fields)])
        _jsonl(post / "xunce-exploration-coverage-model-inference.jsonl", [_inference_row(1 if action_shift else 0, 2 if action_shift else 1, 0.50 + prob_delta, strong_fields)])
        _jsonl(pre / "xunce-exploration-coverage-candidate-metric-audit.jsonl", _candidate_rows(strong_fields))
        rows.append(
            {
                "seed": seed,
                "stage21_1_status": "passed",
                "stage21_3_status": "passed",
                "stage21_4_status": "passed",
                "stage21_5_status": "passed",
                "stage21_5_root": str(stage21_5),
                "stage21_3_batch_fingerprint": f"batch-{seed}",
                "transition_id_fingerprint": f"trans-{seed}",
                "pre_clip_grad_norm": 4.0,
                "post_clip_grad_norm": 1.0,
                "max_post_update_approx_kl": 0.001,
                "min_entropy": 3.0,
                "grad_norm_finite": True,
                "parameter_delta_l2": 0.002,
                "final_coverage_delta": coverage_delta,
                "coverage_auc_delta": coverage_delta,
            }
        )
    _jsonl(root / "xunce-stage21-6-seed-results.jsonl", rows)


def _inference_row(index: int, rank: int, selected_probability: float, strong_fields: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario_id": "scenario-a",
        "step_index": 0,
        "candidate_set_hash": "candidate-hash",
        "detail": {
            "selected_action_index": index,
            "selected_rank": rank,
            "selected_probability": selected_probability,
            "action_probs": [1.0 - selected_probability, selected_probability],
        },
    }
    if strong_fields:
        row.update({"current_cell": [0, 0], "covered_cells_hash": "covered-hash"})
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
