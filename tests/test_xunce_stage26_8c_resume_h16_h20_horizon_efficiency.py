from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_8c_rejects_non_resume_stage26_8b_root(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8c_resume_h16_h20_horizon_efficiency as s26

    stage26_8b_root = _make_stage26_8b_root(tmp_path, next_required_change="repair_stage26_8b_terminal_sampling_mask_diagnostics")
    config = _write_config(tmp_path, stage26_8b_root=stage26_8b_root, run_horizon_chain=False)

    summary = s26.run_xunce_stage26_8c_resume_h16_h20_horizon_efficiency(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8c_required_inputs"
    assert "stage26_8b_route_not_resume_h16_h20" in summary["input_rejections"]


def test_stage26_8c_runs_only_h16_h20_and_routes_stage26_9_on_majority_positive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import scripts.run_xunce_stage26_8c_resume_h16_h20_horizon_efficiency as s26

    stage26_8a_root = _make_stage26_8a_root(tmp_path)
    stage26_8b_root = _make_stage26_8b_root(tmp_path, stage26_8a_root=stage26_8a_root)
    base_config = _write_stage26_8_base_config(tmp_path)
    config = _write_config(tmp_path, stage26_8b_root=stage26_8b_root, stage26_8_base_config=base_config)
    calls: list[tuple[int, Path]] = []

    def fake_stage26_8(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        horizon = int(cfg["collector_rollout_steps"])
        calls.append((horizon, output_root))
        assert cfg["collector_rollout_steps"] == cfg["eval_rollout_steps"]
        assert cfg["stage21_5_timeout_seconds"] == 0.0
        assert cfg["stop_on_first_seed_blocker"] is False
        assert cfg["reuse_existing_seed_outputs"] is True
        assert cfg["seeds"] == [260801, 260802, 260803]
        assert cfg["coverage_denominator_source"] == "main_coverable_cells/v1"
        if horizon == 20:
            return _stage26_8_summary(positive=2, nonnegative=3, mean_per100=0.6, status="passed", route="run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot")
        return _stage26_8_summary(positive=0, nonnegative=3, mean_per100=0.0)

    monkeypatch.setattr(s26.stage26_8, "run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot", fake_stage26_8)

    summary = s26.run_xunce_stage26_8c_resume_h16_h20_horizon_efficiency(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot"
    assert summary["h12_baseline_available"] is True
    assert summary["horizons"] == [16, 20]
    assert calls == [
        (16, tmp_path / "out" / "h16" / "s26_8"),
        (20, tmp_path / "out" / "h20" / "s26_8"),
    ]
    rows = _read_jsonl(tmp_path / "out" / "xunce-stage26-8c-horizon-results.jsonl")
    assert [row["horizon_steps"] for row in rows] == [16, 20]


def test_stage26_8c_does_not_reuse_old_h16_failed_output(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8c_resume_h16_h20_horizon_efficiency as s26

    stage26_8a_root = _make_stage26_8a_root(tmp_path)
    old_h16 = stage26_8a_root / "h16" / "s26_8"
    old_h16.mkdir(parents=True)
    _write_json(old_h16 / "xunce-stage26-8-summary.json", _stage26_8_summary(positive=0, nonnegative=0, mean_per100=0.0, route="repair_stage26_8_update_stability"))
    stage26_8b_root = _make_stage26_8b_root(tmp_path, stage26_8a_root=stage26_8a_root)
    config = _write_config(tmp_path, stage26_8b_root=stage26_8b_root)
    called = 0

    def fake_stage26_8(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        nonlocal called
        called += 1
        assert output_root != old_h16
        return _stage26_8_summary(positive=0, nonnegative=3, mean_per100=0.0)

    monkeypatch.setattr(s26.stage26_8, "run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot", fake_stage26_8)

    summary = s26.run_xunce_stage26_8c_resume_h16_h20_horizon_efficiency(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert called == 2
    assert summary["next_required_change"] == "repair_stage26_synthetic_policy_update_signal_strength"


def test_stage26_8c_routes_expand_seed_when_single_positive(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8c_resume_h16_h20_horizon_efficiency as s26

    stage26_8b_root = _make_stage26_8b_root(tmp_path, stage26_8a_root=_make_stage26_8a_root(tmp_path))
    config = _write_config(tmp_path, stage26_8b_root=stage26_8b_root)

    def fake_stage26_8(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        horizon = json.loads(config_path.read_text(encoding="utf-8"))["collector_rollout_steps"]
        if horizon == 16:
            return _stage26_8_summary(positive=1, nonnegative=3, mean_per100=0.2)
        return _stage26_8_summary(positive=0, nonnegative=3, mean_per100=0.0)

    monkeypatch.setattr(s26.stage26_8, "run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot", fake_stage26_8)

    summary = s26.run_xunce_stage26_8c_resume_h16_h20_horizon_efficiency(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    recommended = json.loads((tmp_path / "out" / "xunce-stage26-8c-recommended-next-config.json").read_text(encoding="utf-8"))
    assert summary["next_required_change"] == "expand_stage26_8c_seed_budget_at_best_horizon"
    assert recommended["recommended_horizon_steps"] == 16
    assert recommended["recommended_new_seeds"] == [260804, 260805]


def test_stage26_8c_routes_binding_update_credit_and_boundary() -> None:
    import scripts.run_xunce_stage26_8c_resume_h16_h20_horizon_efficiency as s26

    binding = [_horizon_row(16, positive=2, binding=1, mean_per100=0.4)]
    update = [_horizon_row(16, positive=2, update=1, mean_per100=0.4)]
    negative = [_horizon_row(16, positive=0, negative=2, mean_per100=-0.2)]

    assert s26._route(boundary_rejections=["starts_online_canary"], input_rejections=[], horizon_rows=[], aggregate=s26._horizon_aggregate([])) == "resolve_stage26_8c_boundary_rejections"
    assert s26._route(boundary_rejections=[], input_rejections=[], horizon_rows=binding, aggregate=s26._horizon_aggregate(binding)) == "repair_stage26_8c_horizon_eval_binding_or_safety"
    assert s26._route(boundary_rejections=[], input_rejections=[], horizon_rows=update, aggregate=s26._horizon_aggregate(update)) == "repair_stage26_8c_update_stability"
    assert s26._route(boundary_rejections=[], input_rejections=[], horizon_rows=negative, aggregate=s26._horizon_aggregate(negative)) == "repair_stage26_synthetic_credit_assignment"


def test_stage26_8c_boundary_rejection(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8c_resume_h16_h20_horizon_efficiency as s26

    stage26_8b_root = _make_stage26_8b_root(tmp_path, stage26_8a_root=_make_stage26_8a_root(tmp_path))
    config = _write_config(tmp_path, stage26_8b_root=stage26_8b_root, starts_online_canary=True)

    summary = s26.run_xunce_stage26_8c_resume_h16_h20_horizon_efficiency(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage26_8c_boundary_rejections"


def _make_stage26_8a_root(tmp_path: Path) -> Path:
    root = tmp_path / "stage26_8a"
    h12 = root / "h12" / "s26_8"
    h12.mkdir(parents=True)
    _write_json(h12 / "xunce-stage26-8-summary.json", _stage26_8_summary(positive=0, nonnegative=3, mean_per100=0.0))
    return root


def _make_stage26_8b_root(tmp_path: Path, **overrides: object) -> Path:
    root = tmp_path / "stage26_8b"
    stage26_8a_root = overrides.pop("stage26_8a_root", None)
    if stage26_8a_root is None:
        stage26_8a_root = _make_stage26_8a_root(tmp_path)
    summary = {
        "schema_version": "xunce-stage26-8b-summary/v1",
        "stage_id": "xunce-stage26-8b-repair-horizon-collector-terminal-reachability",
        "status": "passed",
        "next_required_change": "resume_stage26_8a_from_h16_h20",
        "stage26_8a_root": str(stage26_8a_root),
        "h16_repaired_status": "passed",
        "h16_repaired_transition_count": 45,
        "h16_repaired_stage21_1_terminal_count": 3,
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "max_traversable_slope_deg": 30.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    summary.update(overrides)
    root.mkdir(parents=True)
    _write_json(root / "xunce-stage26-8b-summary.json", summary)
    return root


def _write_stage26_8_base_config(tmp_path: Path) -> Path:
    path = tmp_path / "stage26_8_base.json"
    _write_json(
        path,
        {
            "schema_version": "xunce-stage26-8-synthetic-terrain-multi-seed-coverage-efficiency-pilot-config/v1",
            "stage_id": "xunce-stage26-8-synthetic-terrain-multi-seed-coverage-efficiency-pilot",
            "stage26_7h_root": str(tmp_path / "stage26_7h"),
            "stage26_0_root": str(tmp_path / "stage26_0"),
            "stage26_1_base_config": str(tmp_path / "stage26_1.json"),
            "stage26_2_base_config": str(tmp_path / "stage26_2.json"),
            "stage26_3_base_config": str(tmp_path / "stage26_3.json"),
            "seeds": [260801, 260802, 260803],
            "run_stage26_chain": True,
            "required_scenario_count": 3,
            "collector_rollout_steps": 8,
            "eval_rollout_steps": 8,
            "dynamic_max_candidates_per_step": 36,
            "dynamic_proposal_pool_limit_per_step": 288,
            "hybrid_astar_candidate_eval_workers": 4,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    return path


def _write_config(tmp_path: Path, *, stage26_8b_root: Path, **overrides: object) -> Path:
    payload = {
        "schema_version": "xunce-stage26-8c-resume-h16-h20-horizon-efficiency-config/v1",
        "stage_id": "xunce-stage26-8c-resume-h16-h20-horizon-efficiency",
        "stage26_8b_root": str(stage26_8b_root),
        "stage26_8_base_config": str(overrides.pop("stage26_8_base_config", _write_stage26_8_base_config(tmp_path))),
        "horizons": [16, 20],
        "seeds": [260801, 260802, 260803],
        "run_horizon_chain": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage26_8c_config.json"
    _write_json(path, payload)
    return path


def _stage26_8_summary(
    *,
    positive: int,
    nonnegative: int,
    mean_per100: float,
    status: str = "failed",
    route: str = "expand_stage26_8_seed_or_horizon_budget",
) -> dict[str, object]:
    return {
        "schema_version": "xunce-stage26-8-summary/v1",
        "stage_id": "xunce-stage26-8-synthetic-terrain-multi-seed-coverage-efficiency-pilot",
        "status": status,
        "next_required_change": route,
        "seed_count": 3,
        "completed_seed_count": 3,
        "clean_seed_count": 3,
        "positive_efficiency_seed_count": positive,
        "nonnegative_efficiency_seed_count": nonnegative,
        "negative_efficiency_seed_count": 3 - nonnegative,
        "binding_or_safety_failure_seed_count": 0,
        "execution_failure_seed_count": 0,
        "update_failure_seed_count": 0,
        "mean_main_coverage_per_100m_delta": mean_per100,
        "mean_main_final_coverage_delta": max(mean_per100, 0.0) / 1000.0,
        "mean_main_coverage_auc_delta_diagnostic": -0.01,
        "mean_hybrid_astar_path_cost_delta_diagnostic": 2.0,
        "coverage_denominator_source": "main_coverable_cells/v1",
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
        "max_traversable_slope_deg": 30.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _horizon_row(
    horizon: int,
    *,
    positive: int,
    negative: int = 0,
    binding: int = 0,
    execution: int = 0,
    update: int = 0,
    mean_per100: float,
) -> dict[str, object]:
    majority = 2
    return {
        "horizon_steps": horizon,
        "horizon_label": f"h{horizon}",
        "seed_count": 3,
        "positive_efficiency_seed_count": positive,
        "negative_efficiency_seed_count": negative,
        "binding_or_safety_failure_seed_count": binding,
        "execution_failure_seed_count": execution,
        "update_failure_seed_count": update,
        "mean_main_coverage_per_100m_delta": mean_per100,
        "majority_positive": positive >= majority,
        "single_positive_without_majority": 0 < positive < majority,
        "majority_negative": negative >= majority,
        "clean_zero_delta": binding == 0 and execution == 0 and update == 0 and positive == 0 and negative == 0 and mean_per100 == 0.0,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
