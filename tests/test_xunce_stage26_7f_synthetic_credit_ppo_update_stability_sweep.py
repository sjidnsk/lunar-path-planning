from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_7f_rejects_untrusted_stage26_7d_root(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep as s26

    stage26_7d = tmp_path / "s26_7d"
    stage26_7d.mkdir()
    _write_json(
        stage26_7d / "xunce-stage26-7d-summary.json",
        {
            "stage_id": "xunce-stage26-7d-repair-synthetic-credit-sampler-continuous-theta-reachability",
            "status": "failed",
            "next_required_change": "some_other_route",
        },
    )
    config = _write_config(tmp_path, stage26_7d_root=stage26_7d, run_stage26_chain=False)

    summary = s26.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_7f_required_inputs"


def test_stage26_7f_rejects_relaxed_kl_limit(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep as s26

    stage26_7d = _make_valid_stage26_7d_root(tmp_path)
    config = _write_config(tmp_path, stage26_7d_root=stage26_7d, max_abs_approx_kl=1.6)

    with pytest.raises(ValueError, match="max_abs_approx_kl"):
        s26.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep(
            config_path=config,
            output_root=tmp_path / "out",
            repo_root=REPO_ROOT,
        )


def test_stage26_7f_rejects_nonfinite_kl_limit(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep as s26

    stage26_7d = _make_valid_stage26_7d_root(tmp_path)
    config = _write_config(tmp_path, stage26_7d_root=stage26_7d, max_abs_approx_kl=float("nan"))

    with pytest.raises(ValueError, match="max_abs_approx_kl"):
        s26.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep(
            config_path=config,
            output_root=tmp_path / "out",
            repo_root=REPO_ROOT,
        )


def test_stage26_7f_requires_full_stage26_7d_sample_count(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep as s26

    stage26_7d = _make_valid_stage26_7d_root(tmp_path, trainable_count=1, selected_count=1)
    config = _write_config(tmp_path, stage26_7d_root=stage26_7d, run_stage26_chain=False)

    summary = s26.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_7f_required_inputs"


def test_stage26_7f_does_not_allow_input_summary_to_lower_required_counts(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep as s26

    stage26_7d = _make_valid_stage26_7d_root(tmp_path, trainable_count=1, selected_count=36)
    summary_path = stage26_7d / "xunce-stage26-7d-summary.json"
    summary = _read_json(summary_path)
    summary["stage26_7f_required_trainable_transition_count"] = 1
    _write_json(summary_path, summary)
    config = _write_config(tmp_path, stage26_7d_root=stage26_7d, run_stage26_chain=False)

    result = s26.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert result["next_required_change"] == "rerun_stage26_7f_required_inputs"


def test_stage26_7f_requires_credit_selected_count_independently(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep as s26

    stage26_7d = _make_valid_stage26_7d_root(tmp_path, trainable_count=36, selected_count=1)
    config = _write_config(tmp_path, stage26_7d_root=stage26_7d, run_stage26_chain=False)

    result = s26.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert result["next_required_change"] == "rerun_stage26_7f_required_inputs"


def test_stage26_7f_reuses_one_collector_and_evals_only_stable_combo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep as s26

    stage26_7d = _make_valid_stage26_7d_root(tmp_path)
    config = _write_config(
        tmp_path,
        stage26_7d_root=stage26_7d,
        update_sweep=[
            _combo("kl_bad", "u0", epochs=4, learning_rate=1.0e-5),
            _combo("stable", "u1", epochs=1, learning_rate=5.0e-6),
        ],
    )
    s26_2_configs: list[dict[str, object]] = []
    s26_3_configs: list[dict[str, object]] = []

    def fake_stage26_2(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, object]:
        cfg = _read_json(config_path)
        s26_2_configs.append(cfg)
        stage21_4_root = output_root / "s21_4"
        stage21_4_root.mkdir(parents=True)
        _write_json(
            stage21_4_root / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json",
            {"old_log_prob_source": "behavior_policy_when_present"},
        )
        if cfg["learning_rate"] == 1.0e-5:
            return _stage26_2_summary(output_root, status="failed", kl=2.7, reload=False)
        return _stage26_2_summary(output_root, status="passed", kl=0.72, reload=True, parameter_delta=0.004)

    def fake_stage26_3(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, object]:
        cfg = _read_json(config_path)
        s26_3_configs.append(cfg)
        return {
            "status": "failed",
            "next_required_change": "calibrate_stage26_synthetic_discrete_margin_crossing_after_credit",
            "selected_action_changed_count": 0,
            "main_final_coverage_delta": 0.0,
            "main_coverage_auc_delta": 0.0,
            "main_coverage_per_100m_delta": 0.0,
            "hybrid_astar_path_cost_delta": 0.0,
            "hard_risk_violation_count": 0,
            "mask_violation_count": 0,
            "path_planning_failure_count": 0,
            "open_grid_fallback_count": 0,
        }

    monkeypatch.setattr(s26.stage26_2, "run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke", fake_stage26_2)
    monkeypatch.setattr(s26.stage26_3, "run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke", fake_stage26_3)

    summary = s26.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    expected_stage26_1_root = str(stage26_7d / "chain" / "r" / "c1" / "s26_1")
    assert [cfg["stage26_1_root"] for cfg in s26_2_configs] == [expected_stage26_1_root, expected_stage26_1_root]
    assert [cfg["max_abs_approx_kl"] for cfg in s26_2_configs] == [1.5, 1.5]
    assert len(s26_3_configs) == 1
    assert s26_3_configs[0]["coverage_denominator_source"] == "main_coverable_cells/v1"
    assert s26_3_configs[0]["hybrid_astar_candidate_eval_workers"] == 4
    assert summary["stable_combo_count"] == 1
    assert summary["best_stable_combo_id"] == "stable"
    assert summary["next_required_change"] == "calibrate_stage26_synthetic_discrete_margin_crossing_after_credit"


def test_stage26_7f_routes_kl_stable_checkpoint_failure_to_checkpoint_repair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep as s26

    stage26_7d = _make_valid_stage26_7d_root(tmp_path)
    config = _write_config(tmp_path, stage26_7d_root=stage26_7d, update_sweep=[_combo("checkpoint_bad", "u0")])

    def fake_stage26_2(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, object]:
        return _stage26_2_summary(output_root, status="failed", kl=0.8, reload=False, stage21_4_status="passed")

    monkeypatch.setattr(s26.stage26_2, "run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke", fake_stage26_2)

    summary = s26.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["stable_combo_count"] == 0
    assert summary["next_required_change"] == "repair_stage26_7_credit_checkpoint_boundary"


def test_stage26_7f_routes_all_kl_failures_to_reduce_update_strength(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep as s26

    stage26_7d = _make_valid_stage26_7d_root(tmp_path)
    config = _write_config(tmp_path, stage26_7d_root=stage26_7d, update_sweep=[_combo("bad", "u0")])

    def fake_stage26_2(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, object]:
        return _stage26_2_summary(output_root, status="failed", kl=2.0, reload=False)

    monkeypatch.setattr(s26.stage26_2, "run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke", fake_stage26_2)

    summary = s26.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["stable_combo_count"] == 0
    assert summary["next_required_change"] == "reduce_stage26_7_credit_update_strength"


def test_stage26_7f_routes_pre_update_kl_baseline_to_behavior_kl_repair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep as s26

    stage26_7d = _make_valid_stage26_7d_root(tmp_path)
    config = _write_config(tmp_path, stage26_7d_root=stage26_7d, update_sweep=[_combo("baseline_kl_bad", "u0")])

    def fake_stage26_2(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, object]:
        return _stage26_2_summary(output_root, status="failed", kl=2.7, reload=False, pre_kl=2.7)

    monkeypatch.setattr(s26.stage26_2, "run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke", fake_stage26_2)

    summary = s26.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["pre_update_kl_baseline_exceeded"] is True
    assert summary["next_required_change"] == "repair_stage26_7_behavior_policy_kl_baseline"


def test_best_stable_combo_prefers_larger_parameter_delta_when_kl_not_close() -> None:
    import scripts.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep as s26

    details = [
        {"sweep_row": _stable_row("small_delta", kl=0.7, parameter_delta=0.001), "stage26_2_config": {}, "stage26_3_config": {}},
        {"sweep_row": _stable_row("large_delta", kl=0.9, parameter_delta=0.004), "stage26_2_config": {}, "stage26_3_config": {}},
    ]

    best = s26._best_stable_combo(details, 1.5)

    assert best["combo_id"] == "large_delta"


def _make_valid_stage26_7d_root(tmp_path: Path, *, trainable_count: int = 36, selected_count: int = 36) -> Path:
    stage26_7d = tmp_path / "s26_7d"
    stage26_1 = stage26_7d / "chain" / "r" / "c1" / "s26_1"
    stage26_1.mkdir(parents=True)
    _write_json(stage26_1 / "xunce-stage26-1-summary.json", {"status": "passed"})
    _write_json(
        stage26_7d / "xunce-stage26-7d-summary.json",
        {
            "stage_id": "xunce-stage26-7d-repair-synthetic-credit-sampler-continuous-theta-reachability",
            "status": "failed",
            "next_required_change": "repair_stage26_7_credit_ppo_update_stability",
            "stage26_7c_rerun_output_root": str(stage26_7d / "chain"),
            "trainable_transition_count": trainable_count,
            "synthetic_credit_target_selected_count": selected_count,
            "selected_continuous_theta_hybrid_astar_unreachable_count": 0,
            "behavior_theta_logprob_recomputable": True,
        },
    )
    return stage26_7d


def _write_config(
    tmp_path: Path,
    *,
    stage26_7d_root: Path,
    max_abs_approx_kl: float = 1.5,
    run_stage26_chain: bool = True,
    update_sweep: list[dict[str, object]] | None = None,
) -> Path:
    stage26_2_base = tmp_path / "stage26_2_base.json"
    stage26_3_base = tmp_path / "stage26_3_base.json"
    _write_json(stage26_2_base, {"stage26_1_root": "unused"})
    _write_json(stage26_3_base, {"stage26_2_root": "unused"})
    config = tmp_path / "config.json"
    _write_json(
        config,
        {
            "schema_version": "xunce-stage26-7f-synthetic-credit-ppo-update-stability-sweep-config/v1",
            "stage_id": "xunce-stage26-7f-synthetic-credit-ppo-update-stability-sweep",
            "stage26_7d_root": str(stage26_7d_root),
            "stage26_2_base_config": str(stage26_2_base),
            "stage26_3_base_config": str(stage26_3_base),
            "run_stage26_chain": run_stage26_chain,
            "primary_combo_work_dir": "c1",
            "required_scenario_count": 2,
            "eval_rollout_steps": 4,
            "hybrid_astar_candidate_eval_workers": 4,
            "max_abs_approx_kl": max_abs_approx_kl,
            "update_sweep": update_sweep or [_combo("default", "u0")],
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    return config


def _combo(combo_id: str, work_dir: str, *, epochs: int = 1, learning_rate: float = 5.0e-6) -> dict[str, object]:
    return {
        "combo_id": combo_id,
        "work_dir": work_dir,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "policy_loss_coefficient": 1.0,
        "value_loss_coefficient": 0.02,
        "loss_scale": 0.25,
    }


def _stage26_2_summary(
    output_root: Path,
    *,
    status: str,
    kl: float,
    reload: bool,
    parameter_delta: float = 0.0,
    stage21_4_status: str | None = None,
    pre_kl: float | None = None,
) -> dict[str, object]:
    stage21_4_root = output_root / "s21_4"
    stage21_4_root.mkdir(parents=True, exist_ok=True)
    loss_audit = stage21_4_root / "xunce-stage21-4-ppo-loss-audit.jsonl"
    if pre_kl is not None:
        loss_audit.write_text(
            json.dumps({"pre_update_approx_kl": pre_kl, "post_update_approx_kl": kl, "clip_fraction": 1.0}) + "\n",
            encoding="utf-8",
        )
        summary_path = stage21_4_root / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json"
        if summary_path.exists():
            summary = _read_json(summary_path)
        else:
            summary = {}
        summary["loss_audit"] = str(loss_audit)
        summary.setdefault("old_log_prob_source", "behavior_policy_when_present")
        _write_json(summary_path, summary)
    return {
        "status": status,
        "next_required_change": "run_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke" if status == "passed" else "repair_stage26_7_credit_ppo_update_stability",
        "stage21_4_status": stage21_4_status or status,
        "stage21_4_next_required_change": "stage21_4_complete" if status == "passed" else "post_update_approx_kl_exceeded",
        "stage21_4_root": str(stage21_4_root),
        "final_post_update_approx_kl": kl,
        "loss_finite": True,
        "gradient_finite": True,
        "final_total_loss": 0.1,
        "pre_clip_grad_norm": 0.2,
        "post_clip_grad_norm": 0.2,
        "parameter_delta_l2": parameter_delta,
        "checkpoint_exists": reload,
        "checkpoint_reload_passed": reload,
        "experimental_only": reload,
        "checkpoint_boundary_passed": reload,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _stable_row(combo_id: str, *, kl: float, parameter_delta: float) -> dict[str, object]:
    return {
        "combo_id": combo_id,
        "stage26_2_status": "passed",
        "stage21_4_status": "passed",
        "checkpoint_reload_passed": True,
        "experimental_only": True,
        "checkpoint_boundary_passed": True,
        "final_post_update_approx_kl": kl,
        "loss_finite": True,
        "gradient_finite": True,
        "parameter_delta_l2": parameter_delta,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
