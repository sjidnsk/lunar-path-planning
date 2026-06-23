from __future__ import annotations

import json
import subprocess
from pathlib import Path

import scripts.run_xunce_stage21_8_ppo_update_strength_calibration as stage21_8
from scripts.run_xunce_stage21_8_ppo_update_strength_calibration import (
    ROUTE_GRAD,
    ROUTE_INPUTS,
    ROUTE_RERUN_21_6,
    ROUTE_RUNTIME,
    run_xunce_stage21_8_ppo_update_strength_calibration,
)


def test_stage21_8_recommends_stable_nonregressing_combo(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    combo_root = root / "out" / "sweep_runs" / "stable" / "stage21_6"
    _write_stage21_6_result(combo_root, pre_clip=20.0, post_clip=1.0, reason_codes=[], action_shift=True)
    config = _write_config(root, execute_sweep=False)

    summary = run_xunce_stage21_8_ppo_update_strength_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == ROUTE_RERUN_21_6
    assert summary["recommended_combo_id"] == "stable"
    recommended = json.loads((root / "out" / "xunce-stage21-8-recommended-stage21-6-config.json").read_text())
    assert recommended["max_grad_norm"] == 25.0
    assert recommended["publishes_checkpoint"] is False
    recommended_stage21_4 = json.loads((root / "out" / "xunce-stage21-8-recommended-stage21-4-config.json").read_text())
    assert recommended_stage21_4["max_grad_norm"] == 1.0


def test_stage21_8_routes_to_gradient_repair_when_no_combo_is_stable(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    combo_root = root / "out" / "sweep_runs" / "stable" / "stage21_6"
    _write_stage21_6_result(combo_root, pre_clip=35.0, post_clip=1.0, reason_codes=["seed_2101_grad_unstable"], action_shift=True)
    config = _write_config(root, execute_sweep=False)

    summary = run_xunce_stage21_8_ppo_update_strength_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_GRAD


def test_stage21_8_separates_pre_clip_gate_from_stage21_4_clip_norm(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    combo_root = root / "out" / "sweep_runs" / "stable" / "stage21_6"
    _write_stage21_6_result(combo_root, pre_clip=20.0, post_clip=2.0, reason_codes=[], action_shift=True)
    config = _write_config(root, execute_sweep=False)

    summary = run_xunce_stage21_8_ppo_update_strength_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    rows = [
        json.loads(line)
        for line in (root / "out" / "xunce-stage21-8-sweep-results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stable_row = next(row for row in rows if row["combo_id"] == "stable")
    assert stable_row["gradient_stable"] is True
    assert stable_row["post_clip_within_stage21_4_max_grad_norm"] is False
    assert stable_row["numerically_stable"] is False
    assert summary["next_required_change"] == ROUTE_GRAD


def test_stage21_8_rejects_missing_fingerprint_lineage(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    combo_root = root / "out" / "sweep_runs" / "stable" / "stage21_6"
    _write_stage21_6_result(
        combo_root,
        pre_clip=20.0,
        post_clip=1.0,
        reason_codes=[],
        action_shift=True,
        include_fingerprints=False,
    )
    config = _write_config(root, execute_sweep=False)

    summary = run_xunce_stage21_8_ppo_update_strength_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    rows = [
        json.loads(line)
        for line in (root / "out" / "xunce-stage21-8-sweep-results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stable_row = next(row for row in rows if row["combo_id"] == "stable")
    assert stable_row["numerically_stable"] is False
    assert "seed_2101_missing_stage21_3_batch_fingerprint" in stable_row["boundary_reason_codes"]
    assert "seed_2101_missing_transition_id_fingerprint" in stable_row["boundary_reason_codes"]
    assert summary["next_required_change"] == ROUTE_GRAD


def test_stage21_8_does_not_recommend_failed_stage21_6_status(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    combo_root = root / "out" / "sweep_runs" / "stable" / "stage21_6"
    _write_stage21_6_result(
        combo_root,
        pre_clip=20.0,
        post_clip=1.0,
        reason_codes=["sample_count_too_low_for_performance_claim"],
        action_shift=True,
    )
    config = _write_config(root, execute_sweep=False)

    summary = run_xunce_stage21_8_ppo_update_strength_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["recommended_combo_id"] is None
    assert summary["next_required_change"] != ROUTE_RERUN_21_6


def test_stage21_8_rejects_open_boundary_flags(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    config = _write_config(root, execute_sweep=False, starts_online_canary=True)

    summary = run_xunce_stage21_8_ppo_update_strength_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_INPUTS
    assert "stage21_8_config_starts_online_canary_true" in summary["boundary_reason_codes"]


def test_stage21_8_records_runtime_blocker_on_combo_timeout(monkeypatch, tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    config = _write_config(root, execute_sweep=True, max_new_combinations_to_execute=1, per_combo_timeout_seconds=1)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args", "cmd"), timeout=1, output="out", stderr="err")

    monkeypatch.setattr(stage21_8.subprocess, "run", fake_run)

    summary = run_xunce_stage21_8_ppo_update_strength_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "partial"
    assert summary["next_required_change"] == ROUTE_RUNTIME
    assert (root / "out" / "sweep_runs" / "stable" / "runtime-blocker.json").is_file()


def test_stage21_8_records_runtime_blocker_on_subprocess_failure(monkeypatch, tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    config = _write_config(root, execute_sweep=True, max_new_combinations_to_execute=1)

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=kwargs.get("args", []), returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(stage21_8.subprocess, "run", fake_run)

    summary = run_xunce_stage21_8_ppo_update_strength_calibration(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    blocker = json.loads((root / "out" / "sweep_runs" / "stable" / "runtime-blocker.json").read_text())
    assert summary["status"] == "partial"
    assert summary["next_required_change"] == ROUTE_RUNTIME
    assert blocker["reason_code"] == "stage21_8_calibration_subprocess_failed"


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "configs").mkdir()
    _json(root / "configs" / "stage21_6.json", {"schema_version": "xunce-stage21-6-multi-seed-ppo-pilot-config/v1"})
    _json(root / "configs" / "stage21_4.json", {"schema_version": "xunce-stage21-4-tiny-ppo-update-smoke-config/v1"})
    stage21_7 = root / "stage21_7"
    stage21_7.mkdir()
    _json(
        stage21_7 / "xunce-stage21-7-diagnostic-summary.json",
        {"status": "failed", "next_required_change": "calibrate_stage21_ppo_update_strength"},
    )
    repaired = root / "repaired"
    _write_stage21_6_result(repaired, pre_clip=42.0, post_clip=1.0, reason_codes=["seed_2101_grad_unstable"], action_shift=False)
    return root


def _write_config(root: Path, **overrides: object) -> Path:
    payload = {
        "schema_version": "xunce-stage21-8-ppo-update-strength-calibration-config/v1",
        "stage21_7_root": str(root / "stage21_7"),
        "stage21_7_repaired_stage21_6_root": str(root / "repaired"),
        "stage21_6_base_config": str(root / "configs" / "stage21_6.json"),
        "stage21_4_base_config": str(root / "configs" / "stage21_4.json"),
        "seed_list": [2101, 2102, 2103],
        "required_scenario_count": 8,
        "rollout_steps": 10,
        "dynamic_max_candidates_per_step": 36,
        "dynamic_proposal_pool_limit_per_step": 288,
        "stage21_7_repaired_baseline_max_grad_norm_gate": 25.0,
        "execute_sweep": False,
        "retry_runtime_blocked_combinations": False,
        "max_new_combinations_to_execute": 1,
        "per_combo_timeout_seconds": 1,
        "priority_combinations": [
            {
                "combo_id": "stable",
                "learning_rate": 0.000005,
                "epochs": 1,
                "clip_ratio": 0.2,
                "stage21_4_max_grad_norm": 1.0,
                "stage21_6_pre_clip_grad_norm_gate": 25.0,
            }
        ],
        "stage21_8_authorized": False,
        "training_or_release_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = root / "config.json"
    _json(path, payload)
    return path


def _write_stage21_6_result(
    root: Path,
    *,
    pre_clip: float,
    post_clip: float,
    reason_codes: list[str],
    action_shift: bool,
    include_fingerprints: bool = True,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _json(
        root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json",
        {
            "status": "failed" if reason_codes else "passed",
            "next_required_change": "repair_stage21_6_ppo_numerical_stability" if reason_codes else "prepare_stage22_formal_pure_ppo_training_run",
            "reason_codes": reason_codes,
            "mean_final_coverage_delta": 0.0,
            "mean_coverage_auc_delta": 0.0,
        },
    )
    _json(
        root / "xunce-stage21-6-aggregate-metrics.json",
        {
            "trainable_transition_count_total": 240,
            "pre_clip_grad_norm_mean": pre_clip,
            "pre_clip_grad_norm_max": pre_clip,
            "max_post_update_approx_kl_mean": 0.0001,
            "min_entropy_mean": 3.0,
            "final_coverage_delta_mean": 0.0,
            "final_coverage_delta_min": 0.0,
            "coverage_auc_delta_mean": 0.0,
            "coverage_auc_delta_min": 0.0,
            "path_cost_delta_m_mean": 0.0,
            "soft_risk_exposure_delta_mean": 0.0,
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
    rows = []
    for index, seed in enumerate((2101, 2102, 2103)):
        stage21_5 = root / f"seed_{seed}" / "stage21_5"
        pre = stage21_5 / "pre"
        post = stage21_5 / "post"
        pre.mkdir(parents=True, exist_ok=True)
        post.mkdir(parents=True, exist_ok=True)
        _json(stage21_5 / "xunce-stage21-5-post-update-evaluation-summary.json", {"pre_evaluation_root": str(pre), "post_evaluation_root": str(post)})
        _jsonl(pre / "xunce-exploration-coverage-model-inference.jsonl", [_inference_row(0, 0, 0.5)])
        _jsonl(post / "xunce-exploration-coverage-model-inference.jsonl", [_inference_row(1 if action_shift else 0, 0, 0.7 if action_shift else 0.5)])
        rows.append(
            {
                "seed": seed,
                "stage21_1_status": "passed",
                "stage21_3_status": "passed",
                "stage21_4_status": "passed",
                "stage21_5_status": "passed",
                "stage21_3_batch_fingerprint": f"batch-{index}" if include_fingerprints else None,
                "transition_id_fingerprint": f"transition-{index}" if include_fingerprints else None,
                "stage21_5_root": str(stage21_5),
                "parameter_delta_l2": 0.01,
                "post_clip_grad_norm": post_clip,
            }
        )
    _jsonl(root / "xunce-stage21-6-seed-results.jsonl", rows)


def _inference_row(action: int, step: int, probability: float) -> dict[str, object]:
    return {
        "scenario_id": "s0",
        "step_index": step,
        "candidate_set_hash": "h0",
        "detail": {"selected_action_index": action, "selected_probability": probability},
    }


def _json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
