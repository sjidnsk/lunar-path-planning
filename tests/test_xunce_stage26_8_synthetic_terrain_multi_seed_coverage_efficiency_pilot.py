from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_8_rejects_unclean_stage26_7h_input(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as s26

    stage26_7h = _make_stage26_7h_root(tmp_path, hybrid_path_missing_provenance_count=1)
    config = _write_config(tmp_path, stage26_7h_root=stage26_7h, run_stage26_chain=False)

    summary = s26.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8_required_inputs"
    assert "stage26_7h_hybrid_path_missing_provenance_nonzero" in summary["input_rejections"]


def test_stage26_8_rejects_stage26_7h_missing_required_zero_count(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as s26

    stage26_7h = _make_stage26_7h_root(tmp_path)
    summary_path = stage26_7h / "xunce-stage26-7h-summary.json"
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_payload.pop("hard_risk_violation_count")
    _write_json(summary_path, summary_payload)
    config = _write_config(tmp_path, stage26_7h_root=stage26_7h, run_stage26_chain=False)

    summary = s26.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert "stage26_7h_hard_risk_violation_missing" in summary["input_rejections"]


def test_stage26_8_accepts_clean_failed_credit_route_input_gate(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as s26

    stage26_7h = _make_stage26_7h_root(
        tmp_path,
        status="failed",
        next_required_change="repair_stage26_synthetic_credit_assignment",
        main_coverage_per_100m_delta=-0.019,
    )
    config = _write_config(tmp_path, stage26_7h_root=stage26_7h, run_stage26_chain=False)

    summary = s26.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["input_rejections"] == []


def test_stage26_8_rejects_unrelated_stage26_7h_route(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as s26

    stage26_7h = _make_stage26_7h_root(
        tmp_path,
        status="failed",
        next_required_change="repair_stage26_7h_pre_policy_unreachable_baseline",
    )
    config = _write_config(tmp_path, stage26_7h_root=stage26_7h, run_stage26_chain=False)

    summary = s26.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert "stage26_7h_route_not_accepted_for_efficiency_pilot" in summary["input_rejections"]


def test_stage26_8_rejects_stage26_7h_safety_or_boundary_leak(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as s26

    stage26_7h = _make_stage26_7h_root(tmp_path, hard_risk_violation_count=1, starts_online_canary=True)
    config = _write_config(tmp_path, stage26_7h_root=stage26_7h, run_stage26_chain=False)

    summary = s26.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert "stage26_7h_hard_risk_violation_nonzero" in summary["input_rejections"]
    assert "stage26_7h_starts_online_canary" in summary["input_rejections"]


def test_stage26_8_rejects_stage26_7h_lineage_mismatch(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as s26

    stage26_7h = _make_stage26_7h_root(tmp_path, path_cost_source="legacy_grid_astar/v0")
    config = _write_config(tmp_path, stage26_7h_root=stage26_7h, run_stage26_chain=False)

    summary = s26.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert "stage26_7h_path_cost_source_mismatch" in summary["input_rejections"]


def test_stage26_8_rejects_stage26_7h_missing_action_space_type(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as s26

    stage26_7h = _make_stage26_7h_root(tmp_path)
    summary_path = stage26_7h / "xunce-stage26-7h-summary.json"
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_payload.pop("action_space_type")
    _write_json(summary_path, summary_payload)
    config = _write_config(tmp_path, stage26_7h_root=stage26_7h, run_stage26_chain=False)

    summary = s26.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert "stage26_7h_action_space_type_mismatch" in summary["input_rejections"]


def test_stage26_8_rejects_stage26_7h_missing_efficiency_metric(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as s26

    stage26_7h = _make_stage26_7h_root(tmp_path)
    summary_path = stage26_7h / "xunce-stage26-7h-summary.json"
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_payload.pop("post_update_success_metric")
    _write_json(summary_path, summary_payload)
    config = _write_config(tmp_path, stage26_7h_root=stage26_7h, run_stage26_chain=False)

    summary = s26.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert "stage26_7h_post_update_success_metric_mismatch" in summary["input_rejections"]


def test_stage26_8_runs_seed_chain_and_routes_long_horizon_when_majority_efficiency_positive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as s26

    stage26_7h = _make_stage26_7h_root(tmp_path)
    config = _write_config(tmp_path, stage26_7h_root=stage26_7h, seeds=[260801, 260802, 260803])
    calls: list[tuple[str, int]] = []

    def fake_stage26_1(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        seed = int(cfg["stage26_8_seed"])
        calls.append(("s26_1", seed))
        stage21_1_base = json.loads(Path(cfg["stage21_1_base_config"]).read_text(encoding="utf-8"))
        assert stage21_1_base["sampling_seed"] == seed
        assert cfg["rollout_steps"] == 8
        assert cfg["hybrid_astar_candidate_eval_workers"] == 4
        summary = _stage26_1_summary(seed)
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json(output_root / "xunce-stage26-1-summary.json", summary)
        return summary

    def fake_stage26_2(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        seed = int(cfg["stage26_8_seed"])
        calls.append(("s26_2", seed))
        assert cfg["epochs"] == 4
        assert cfg["learning_rate"] == 1.0e-5
        summary = _stage26_2_summary(output_root / "s21_4")
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json(output_root / "xunce-stage26-2-summary.json", summary)
        return summary

    def fake_stage26_3(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        seed = int(cfg["stage26_8_seed"])
        calls.append(("s26_3", seed))
        assert cfg["post_update_success_metric"] == "main_coverable_coverage_efficiency/v1"
        assert cfg["coverage_denominator_source"] == "main_coverable_cells/v1"
        assert cfg["rollout_steps"] == 8
        assert cfg["stage21_5_timeout_seconds"] == 0.0
        per_100m = {260801: 1.5, 260802: -0.2, 260803: 0.8}[seed]
        status = "passed" if per_100m >= 0.0 else "failed"
        route = (
            "run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot"
            if status == "passed"
            else "repair_stage26_synthetic_credit_assignment"
        )
        summary = _stage26_3_summary(status=status, route=route, per_100m=per_100m, final_delta=max(per_100m, 0.0) / 1000.0)
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json(output_root / "xunce-stage26-3-summary.json", summary)
        return summary

    monkeypatch.setattr(s26.stage26_1, "run_xunce_stage26_1_synthetic_terrain_collector_smoke", fake_stage26_1)
    monkeypatch.setattr(s26.stage26_2, "run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke", fake_stage26_2)
    monkeypatch.setattr(s26.stage26_3, "run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke", fake_stage26_3)

    summary = s26.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["schema_version"] == "xunce-stage26-8-summary/v1"
    assert summary["next_required_change"] == "run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot"
    assert summary["seed_count"] == 3
    assert summary["positive_efficiency_seed_count"] == 2
    assert summary["binding_or_safety_failure_seed_count"] == 0
    assert (tmp_path / "out" / "xunce-stage26-8-scenario-efficiency-rows.jsonl").is_file()
    assert calls == [
        ("s26_1", 260801),
        ("s26_2", 260801),
        ("s26_3", 260801),
        ("s26_1", 260802),
        ("s26_2", 260802),
        ("s26_3", 260802),
        ("s26_1", 260803),
        ("s26_2", 260803),
        ("s26_3", 260803),
    ]


def test_stage26_8_reuses_existing_seed_collector_and_update_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as s26

    stage26_7h = _make_stage26_7h_root(tmp_path)
    config = _write_config(
        tmp_path,
        stage26_7h_root=stage26_7h,
        seeds=[260801],
        reuse_existing_seed_outputs=True,
    )
    seed_root = tmp_path / "out" / "s0"
    (seed_root / "s26_1").mkdir(parents=True)
    (seed_root / "s26_2").mkdir(parents=True)
    _write_json(seed_root / "s26_1" / "xunce-stage26-1-summary.json", _stage26_1_summary(260801))
    _write_json(seed_root / "s26_2" / "xunce-stage26-2-summary.json", _stage26_2_summary(seed_root / "s26_2" / "s21_4"))
    calls: list[str] = []

    def fail_stage26_1(**_: object) -> dict:
        raise AssertionError("Stage26.1 should be reused")

    def fail_stage26_2(**_: object) -> dict:
        raise AssertionError("Stage26.2 should be reused")

    def fake_stage26_3(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        calls.append("s26_3")
        summary = _stage26_3_summary(
            status="passed",
            route="run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot",
            per_100m=1.0,
            final_delta=0.001,
        )
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json(output_root / "xunce-stage26-3-summary.json", summary)
        return summary

    monkeypatch.setattr(s26.stage26_1, "run_xunce_stage26_1_synthetic_terrain_collector_smoke", fail_stage26_1)
    monkeypatch.setattr(s26.stage26_2, "run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke", fail_stage26_2)
    monkeypatch.setattr(s26.stage26_3, "run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke", fake_stage26_3)

    summary = s26.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "xunce-stage26-8-seed-results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert calls == ["s26_3"]
    assert summary["positive_efficiency_seed_count"] == 1
    assert rows[0]["reused_existing_stage26_1"] is True
    assert rows[0]["reused_existing_stage26_2"] is True
    assert rows[0]["reused_existing_stage26_3"] is False


def test_stage26_8_routes_credit_repair_when_majority_efficiency_negative(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as s26

    rows = [
        {"seed": 1, "stage26_3_status": "failed", "main_coverage_per_100m_delta": -0.3},
        {"seed": 2, "stage26_3_status": "failed", "main_coverage_per_100m_delta": -0.2},
        {"seed": 3, "stage26_3_status": "passed", "main_coverage_per_100m_delta": 0.1},
    ]

    route = s26._route(boundary_rejections=[], input_rejections=[], seed_rows=rows, aggregate=s26._efficiency_aggregate(rows))

    assert route == "repair_stage26_synthetic_credit_assignment"


def test_stage26_8_does_not_count_failed_zero_delta_seed_as_efficiency_positive() -> None:
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as s26

    rows = [
        _seed_row(status="failed", route="calibrate_stage26_synthetic_discrete_margin_crossing_after_credit", per_100m=0.0),
        _seed_row(status="failed", route="calibrate_stage26_synthetic_discrete_margin_crossing_after_credit", per_100m=0.0),
        _seed_row(status="passed", route="run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot", per_100m=0.2, passed=True),
    ]

    aggregate = s26._efficiency_aggregate(rows)
    route = s26._route(boundary_rejections=[], input_rejections=[], seed_rows=rows, aggregate=aggregate)

    assert aggregate["positive_efficiency_seed_count"] == 1
    assert aggregate["nonnegative_efficiency_seed_count"] == 3
    assert route == "expand_stage26_8_seed_or_horizon_budget"


def test_stage26_8_does_not_route_stage26_9_when_majority_clean_zero_delta() -> None:
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as s26

    rows = [
        _seed_row(status="passed", route="run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot", per_100m=0.0, passed=True),
        _seed_row(status="passed", route="run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot", per_100m=0.0, passed=True),
        _seed_row(status="passed", route="run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot", per_100m=0.2, passed=True),
    ]

    aggregate = s26._efficiency_aggregate(rows)
    route = s26._route(boundary_rejections=[], input_rejections=[], seed_rows=rows, aggregate=aggregate)

    assert aggregate["positive_efficiency_seed_count"] == 1
    assert aggregate["nonnegative_efficiency_seed_count"] == 3
    assert route == "expand_stage26_8_seed_or_horizon_budget"


def test_stage26_8_routes_binding_repair_on_seed_lineage_mismatch() -> None:
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as s26

    rows = [
        _seed_row(status="passed", route="run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot", per_100m=0.5, passed=True),
        _seed_row(status="passed", route="run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot", per_100m=0.4, passed=True, lineage_mismatch=True),
    ]

    aggregate = s26._efficiency_aggregate(rows)
    route = s26._route(boundary_rejections=[], input_rejections=[], seed_rows=rows, aggregate=aggregate)

    assert aggregate["binding_or_safety_failure_seed_count"] == 1
    assert route == "repair_stage26_8_multiseed_eval_binding_or_safety"


def test_stage26_8_update_failure_without_stage26_3_summary_is_not_binding_failure() -> None:
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as s26

    row = {
        "stage26_1_status": "passed",
        "stage26_2_status": "failed",
        "stage26_3_status": None,
        "lineage_mismatch": False,
        "binding_or_safety_failure": False,
        "update_failure": True,
        "main_coverage_per_100m_delta": 0.0,
    }

    aggregate = s26._efficiency_aggregate([row])
    route = s26._route(boundary_rejections=[], input_rejections=[], seed_rows=[row], aggregate=aggregate)

    assert aggregate["binding_or_safety_failure_seed_count"] == 0
    assert aggregate["update_failure_seed_count"] == 1
    assert route == "repair_stage26_8_update_stability"


def test_stage26_8_collector_failure_routes_to_binding_repair() -> None:
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as s26

    row = {
        "stage26_1_status": "failed",
        "stage26_1_next_required_change": "repair_stage26_1_collector_synthetic_map_binding",
        "stage26_2_status": None,
        "stage26_3_status": None,
        "lineage_mismatch": False,
        "binding_or_safety_failure": True,
        "collector_failure": True,
        "update_failure": False,
        "main_coverage_per_100m_delta": 0.0,
    }

    aggregate = s26._efficiency_aggregate([row])
    route = s26._route(boundary_rejections=[], input_rejections=[], seed_rows=[row], aggregate=aggregate)

    assert aggregate["binding_or_safety_failure_seed_count"] == 1
    assert aggregate["update_failure_seed_count"] == 0
    assert route == "repair_stage26_8_multiseed_eval_binding_or_safety"


def test_stage26_8_routes_binding_repair_on_stage21_5_runtime_execution_failure() -> None:
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as s26

    row = _seed_row(
        status="failed",
        route="rerun_stage26_3_required_inputs",
        per_100m=0.0,
        execution_failure=True,
    )

    aggregate = s26._efficiency_aggregate([row])
    route = s26._route(boundary_rejections=[], input_rejections=[], seed_rows=[row], aggregate=aggregate)

    assert aggregate["clean_seed_count"] == 0
    assert aggregate["execution_failure_seed_count"] == 1
    assert route == "repair_stage26_8_multiseed_eval_binding_or_safety"


def test_stage26_8_treats_seed_success_metric_mismatch_as_lineage_failure() -> None:
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as s26

    row = _stage26_3_summary(
        status="passed",
        route="run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot",
        per_100m=0.5,
        final_delta=0.1,
    )
    row["post_update_success_metric"] = "coverage_curve_auc/v1"

    assert s26._lineage_mismatch(row) is True
    assert s26._efficiency_seed_passed(row) is False


def _make_stage26_7h_root(tmp_path: Path, **overrides: object) -> Path:
    root = tmp_path / "s26_7h"
    summary = {
        "schema_version": "xunce-stage26-7h-summary/v1",
        "stage_id": "xunce-stage26-7h-repair-credit-post-update-eval-binding",
        "status": "passed",
        "next_required_change": "run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot",
        "post_update_success_metric": "main_coverable_coverage_efficiency/v1",
        "synthetic_inference_required_field_missing_count": 0,
        "hybrid_path_missing_provenance_count": 0,
        "hybrid_path_contract_mismatch_count": 0,
        "explicit_unreachable_selected_provenance_count": 0,
        "pre_unreachable_selected_count": 0,
        "post_unreachable_selected_count": 0,
        "hard_risk_violation_count": 0,
        "mask_violation_count": 0,
        "path_planning_failure_count": 0,
        "open_grid_fallback_count": 0,
        "main_coverage_per_100m_delta": 2.6762,
        "main_final_coverage_delta": 0.0039,
        "coverage_denominator_source": "main_coverable_cells/v1",
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
        "synthetic_terrain_hash": "e628d24c6fb0f961c50294ea0a018a2558a79a0b30ad21dd31b6e34fc0867e85",
        "platform_contract_hash": "platform-hash",
        "max_traversable_slope_deg": 30.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    summary.update(overrides)
    root.mkdir(parents=True)
    _write_json(root / "xunce-stage26-7h-summary.json", summary)
    return root


def _write_config(tmp_path: Path, *, stage26_7h_root: Path, seeds: list[int] | None = None, **overrides: object) -> Path:
    stage21_1_base = tmp_path / "stage21_1_base.json"
    stage21_1_base.write_text(json.dumps({"schema_version": "stage21-1-base/v1"}), encoding="utf-8")
    stage26_1_base = tmp_path / "stage26_1_base.json"
    _write_json(
        stage26_1_base,
        {
            "schema_version": "xunce-stage26-1-synthetic-terrain-collector-smoke-config/v1",
            "stage_id": "xunce-stage26-1-synthetic-terrain-collector-smoke",
            "stage26_0_root": str(tmp_path / "stage26_0"),
            "stage21_1_base_config": str(stage21_1_base),
            "stage21_2_base_config": str(tmp_path / "stage21_2.json"),
            "stage21_3_base_config": str(tmp_path / "stage21_3.json"),
            "coverage_first_reward_profile": str(tmp_path / "reward.json"),
            "required_scenario_count": 2,
            "rollout_steps": 4,
            "min_trainable_transition_count": 1,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    payload = {
        "schema_version": "xunce-stage26-8-synthetic-terrain-multi-seed-coverage-efficiency-pilot-config/v1",
        "stage_id": "xunce-stage26-8-synthetic-terrain-multi-seed-coverage-efficiency-pilot",
        "stage26_7h_root": str(stage26_7h_root),
        "stage26_1_base_config": str(stage26_1_base),
        "stage26_2_base_config": "configs/xunce_stage26_2_synthetic_terrain_ppo_update_smoke_v1.json",
        "stage26_3_base_config": "configs/xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke_v1.json",
        "seeds": seeds or [260801, 260802, 260803],
        "run_stage26_chain": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage26_8_config.json"
    _write_json(path, payload)
    return path


def _stage26_1_summary(seed: int) -> dict:
    return {
        "schema_version": "xunce-stage26-1-summary/v1",
        "status": "passed",
        "next_required_change": "run_stage26_2_synthetic_terrain_ppo_update_smoke",
        "synthetic_terrain_hash": "e628d24c6fb0f961c50294ea0a018a2558a79a0b30ad21dd31b6e34fc0867e85",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "stage26_8_seed": seed,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _stage26_2_summary(stage21_4_root: Path) -> dict:
    return {
        "schema_version": "xunce-stage26-2-summary/v1",
        "status": "passed",
        "next_required_change": "run_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke",
        "stage21_4_root": str(stage21_4_root),
        "synthetic_terrain_hash": "e628d24c6fb0f961c50294ea0a018a2558a79a0b30ad21dd31b6e34fc0867e85",
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _stage26_3_summary(*, status: str, route: str, per_100m: float, final_delta: float) -> dict:
    return {
        "schema_version": "xunce-stage26-3-summary/v1",
        "status": status,
        "next_required_change": route,
        "synthetic_inference_required_field_missing_count": 0,
        "hybrid_path_missing_provenance_count": 0,
        "hybrid_path_contract_mismatch_count": 0,
        "pre_unreachable_selected_count": 0,
        "post_unreachable_selected_count": 0,
        "explicit_unreachable_selected_provenance_count": 0,
        "hard_risk_violation_count": 0,
        "mask_violation_count": 0,
        "path_planning_failure_count": 0,
        "open_grid_fallback_count": 0,
        "coverage_denominator_source": "main_coverable_cells/v1",
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "synthetic_terrain_hash": "e628d24c6fb0f961c50294ea0a018a2558a79a0b30ad21dd31b6e34fc0867e85",
        "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
        "platform_contract_hash": "platform-hash",
        "max_traversable_slope_deg": 30.0,
        "coverage_per_100m_delta": per_100m,
        "post_update_success_metric": "main_coverable_coverage_efficiency/v1",
        "final_coverage_delta": final_delta,
        "coverage_auc_delta": -0.01,
        "selected_action_changed_count": 1,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _seed_row(
    *,
    status: str,
    route: str,
    per_100m: float,
    passed: bool = False,
    lineage_mismatch: bool = False,
    execution_failure: bool = False,
) -> dict:
    return {
        "seed": 1,
        "stage26_3_status": status,
        "stage26_3_next_required_change": route,
        "main_coverage_per_100m_delta": per_100m,
        "efficiency_seed_passed": passed,
        "binding_or_safety_failure": lineage_mismatch,
        "execution_failure": execution_failure,
        "update_failure": False,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
