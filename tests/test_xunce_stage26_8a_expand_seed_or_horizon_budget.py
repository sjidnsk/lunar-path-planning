from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_8a_rejects_non_expand_stage26_8_root(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8a_expand_seed_or_horizon_budget as s26

    stage26_8_root = _make_stage26_8_root(tmp_path, next_required_change="repair_stage26_8_update_stability")
    config = _write_config(tmp_path, stage26_8_root=stage26_8_root, run_horizon_chain=False)

    summary = s26.run_xunce_stage26_8a_expand_seed_or_horizon_budget(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8a_required_inputs"
    assert "stage26_8_route_not_expand_budget" in summary["input_rejections"]


def test_stage26_8a_rejects_stage26_8_lineage_mismatch(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8a_expand_seed_or_horizon_budget as s26

    stage26_8_root = _make_stage26_8_root(tmp_path, path_cost_source="legacy_grid_astar/v0")
    config = _write_config(tmp_path, stage26_8_root=stage26_8_root, run_horizon_chain=False)

    summary = s26.run_xunce_stage26_8a_expand_seed_or_horizon_budget(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8a_required_inputs"
    assert "stage26_8_path_cost_source_mismatch" in summary["input_rejections"]


@pytest.mark.parametrize(
    ("field", "bad_value", "expected_reason"),
    [
        ("coverage_denominator_source", "roi_valid_cells/v0", "stage26_8_coverage_denominator_source_mismatch"),
        ("coverage_source", "theta_aware_sensor_footprint/v1", "stage26_8_coverage_source_mismatch"),
        ("path_cost_source", "legacy_grid_astar/v0", "stage26_8_path_cost_source_mismatch"),
        ("synthetic_source_kind", "physical_obstacle_cells", "stage26_8_synthetic_source_kind_mismatch"),
        ("action_space_type", "discrete_xy_theta_bin/v0", "stage26_8_action_space_type_mismatch"),
        ("max_traversable_slope_deg", 20.0, "stage26_8_max_traversable_slope_deg_mismatch"),
        ("publishes_checkpoint", True, "stage26_8_publishes_checkpoint"),
        ("replaces_default_policy", True, "stage26_8_replaces_default_policy"),
        ("connects_real_executor", True, "stage26_8_connects_real_executor"),
        ("starts_online_canary", True, "stage26_8_starts_online_canary"),
        ("canary_traffic_fraction", 0.1, "stage26_8_canary_traffic_fraction"),
    ],
)
def test_stage26_8a_rejects_each_stage26_8_lineage_or_boundary_mismatch(
    tmp_path: Path,
    field: str,
    bad_value: object,
    expected_reason: str,
) -> None:
    import scripts.run_xunce_stage26_8a_expand_seed_or_horizon_budget as s26

    stage26_8_root = _make_stage26_8_root(tmp_path, **{field: bad_value})
    config = _write_config(tmp_path, stage26_8_root=stage26_8_root, run_horizon_chain=False)

    summary = s26.run_xunce_stage26_8a_expand_seed_or_horizon_budget(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8a_required_inputs"
    assert expected_reason in summary["input_rejections"]


def test_stage26_8a_runs_horizon_ladder_and_routes_stage26_9_on_majority_positive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import scripts.run_xunce_stage26_8a_expand_seed_or_horizon_budget as s26

    stage26_8_root = _make_stage26_8_root(tmp_path)
    base_config = _write_stage26_8_base_config(tmp_path)
    config = _write_config(tmp_path, stage26_8_root=stage26_8_root, stage26_8_base_config=base_config)
    calls: list[tuple[int, Path]] = []

    def fake_stage26_8(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        horizon = int(cfg["collector_rollout_steps"])
        calls.append((horizon, output_root))
        assert cfg["collector_rollout_steps"] == cfg["eval_rollout_steps"]
        assert cfg["seeds"] == [260801, 260802, 260803]
        assert cfg["hybrid_astar_candidate_eval_workers"] == 4
        assert cfg["stop_on_first_seed_blocker"] is True
        if horizon == 16:
            return _stage26_8_summary(positive=2, nonnegative=2, mean_per100=0.4, status="passed", route="run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot")
        return _stage26_8_summary(positive=0, nonnegative=3, mean_per100=0.0)

    monkeypatch.setattr(s26.stage26_8, "run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot", fake_stage26_8)

    summary = s26.run_xunce_stage26_8a_expand_seed_or_horizon_budget(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot"
    assert summary["majority_positive_horizon_count"] == 1
    assert summary["best_horizon_steps"] == 16
    assert calls == [
        (12, tmp_path / "out" / "h12" / "s26_8"),
        (16, tmp_path / "out" / "h16" / "s26_8"),
        (20, tmp_path / "out" / "h20" / "s26_8"),
    ]
    rows = _read_jsonl(tmp_path / "out" / "xunce-stage26-8a-horizon-results.jsonl")
    assert [row["horizon_steps"] for row in rows] == [12, 16, 20]


def test_stage26_8a_can_reuse_existing_horizon_outputs_without_rerun(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import scripts.run_xunce_stage26_8a_expand_seed_or_horizon_budget as s26

    stage26_8_root = _make_stage26_8_root(tmp_path)
    base_config = _write_stage26_8_base_config(tmp_path)
    config = _write_config(
        tmp_path,
        stage26_8_root=stage26_8_root,
        stage26_8_base_config=base_config,
        reuse_existing_horizon_outputs=True,
    )
    for horizon in (12, 16, 20):
        existing_root = tmp_path / "out" / f"h{horizon}" / "s26_8"
        existing_root.mkdir(parents=True)
        _write_json(existing_root / "xunce-stage26-8-summary.json", _stage26_8_summary(positive=0, nonnegative=3, mean_per100=0.0))

    def fail_if_called(**_: object) -> dict:
        raise AssertionError("existing horizon output should be reused")

    monkeypatch.setattr(s26.stage26_8, "run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot", fail_if_called)

    summary = s26.run_xunce_stage26_8a_expand_seed_or_horizon_budget(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["horizon_count"] == 3
    assert summary["next_required_change"] == "repair_stage26_synthetic_policy_update_signal_strength"


def test_stage26_8a_reclassifies_nested_collector_failure_as_binding() -> None:
    import scripts.run_xunce_stage26_8a_expand_seed_or_horizon_budget as s26

    summary = _stage26_8_summary(positive=0, nonnegative=0, mean_per100=0.0, route="repair_stage26_8_update_stability")
    summary.update({"seed_count": 1, "completed_seed_count": 0, "clean_seed_count": 0, "update_failure_seed_count": 1})
    # Build an in-memory row by writing the seed result fixture through a tmp path.
    # The real runner sees this pattern when an older nested Stage26.8 summary
    # called a Stage26.1 collector failure an update failure.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        horizon_root = Path(tmp) / "h16"
        stage26_8_root = horizon_root / "s26_8"
        stage26_8_root.mkdir(parents=True)
        _write_jsonl(
            stage26_8_root / "xunce-stage26-8-seed-results.jsonl",
            [
                {
                    "stage26_1_status": "failed",
                    "stage26_1_next_required_change": "repair_stage26_1_collector_synthetic_map_binding",
                    "stage26_2_status": None,
                    "update_failure": True,
                    "binding_or_safety_failure": False,
                }
            ],
        )
        row = s26._horizon_row(
            horizon_steps=16,
            horizon_root=horizon_root,
            stage26_8_output_root=stage26_8_root,
            stage26_8_config_path=horizon_root / "config.json",
            summary=summary,
        )

    assert row["collector_failure_seed_count"] == 1
    assert row["binding_or_safety_failure_seed_count"] == 1
    assert row["update_failure_seed_count"] == 0


def test_stage26_8a_routes_expand_seed_budget_when_single_horizon_has_single_positive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import scripts.run_xunce_stage26_8a_expand_seed_or_horizon_budget as s26

    stage26_8_root = _make_stage26_8_root(tmp_path)
    base_config = _write_stage26_8_base_config(tmp_path)
    config = _write_config(tmp_path, stage26_8_root=stage26_8_root, stage26_8_base_config=base_config)

    def fake_stage26_8(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        horizon = json.loads(config_path.read_text(encoding="utf-8"))["collector_rollout_steps"]
        if horizon == 20:
            return _stage26_8_summary(positive=1, nonnegative=3, mean_per100=0.2)
        return _stage26_8_summary(positive=0, nonnegative=3, mean_per100=0.0)

    monkeypatch.setattr(s26.stage26_8, "run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot", fake_stage26_8)

    summary = s26.run_xunce_stage26_8a_expand_seed_or_horizon_budget(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    recommended = json.loads((tmp_path / "out" / "xunce-stage26-8a-recommended-next-config.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "expand_stage26_8a_seed_budget_at_best_horizon"
    assert summary["best_horizon_steps"] == 20
    assert recommended["recommended_horizon_steps"] == 20
    assert recommended["recommended_new_seeds"] == [260804, 260805]


def test_stage26_8a_routes_policy_signal_when_all_horizons_clean_zero(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import scripts.run_xunce_stage26_8a_expand_seed_or_horizon_budget as s26

    stage26_8_root = _make_stage26_8_root(tmp_path)
    base_config = _write_stage26_8_base_config(tmp_path)
    config = _write_config(tmp_path, stage26_8_root=stage26_8_root, stage26_8_base_config=base_config)

    monkeypatch.setattr(
        s26.stage26_8,
        "run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot",
        lambda **_: _stage26_8_summary(positive=0, nonnegative=3, mean_per100=0.0),
    )

    summary = s26.run_xunce_stage26_8a_expand_seed_or_horizon_budget(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_synthetic_policy_update_signal_strength"
    assert summary["zero_delta_horizon_count"] == 3


def test_stage26_8a_routes_credit_repair_when_majority_negative(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8a_expand_seed_or_horizon_budget as s26

    rows = [
        _horizon_row(12, positive=0, negative=2, mean_per100=-0.3),
        _horizon_row(16, positive=0, negative=0, mean_per100=0.0),
    ]
    aggregate = s26._horizon_aggregate(rows)
    route = s26._route(boundary_rejections=[], input_rejections=[], horizon_rows=rows, aggregate=aggregate)

    assert route == "repair_stage26_synthetic_credit_assignment"


def test_stage26_8a_routes_binding_or_update_repairs() -> None:
    import scripts.run_xunce_stage26_8a_expand_seed_or_horizon_budget as s26

    binding_rows = [_horizon_row(12, positive=2, binding=1, mean_per100=0.4)]
    update_rows = [_horizon_row(12, positive=2, update=1, mean_per100=0.4)]

    assert s26._route(boundary_rejections=[], input_rejections=[], horizon_rows=binding_rows, aggregate=s26._horizon_aggregate(binding_rows)) == "repair_stage26_8a_horizon_eval_binding_or_safety"
    assert s26._route(boundary_rejections=[], input_rejections=[], horizon_rows=update_rows, aggregate=s26._horizon_aggregate(update_rows)) == "repair_stage26_8a_update_stability"


def test_stage26_8a_best_horizon_prefers_clean_horizon_over_blocked_higher_horizon() -> None:
    import scripts.run_xunce_stage26_8a_expand_seed_or_horizon_budget as s26

    rows = [
        _horizon_row(12, positive=0, mean_per100=0.0),
        _horizon_row(16, positive=0, binding=1, mean_per100=0.0),
    ]

    aggregate = s26._horizon_aggregate(rows)

    assert aggregate["best_horizon_steps"] == 12
    assert aggregate["clean_horizon_count"] == 1


def test_stage26_8a_routes_eval_repair_on_stage26_8_execution_failure() -> None:
    import scripts.run_xunce_stage26_8a_expand_seed_or_horizon_budget as s26

    rows = [_horizon_row(12, positive=0, execution=1, mean_per100=0.0)]
    aggregate = s26._horizon_aggregate(rows)
    route = s26._route(boundary_rejections=[], input_rejections=[], horizon_rows=rows, aggregate=aggregate)

    assert aggregate["clean_horizon_count"] == 0
    assert route == "repair_stage26_8a_horizon_eval_binding_or_safety"


def test_stage26_8a_boundary_rejection_has_priority(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8a_expand_seed_or_horizon_budget as s26

    stage26_8_root = _make_stage26_8_root(tmp_path)
    config = _write_config(tmp_path, stage26_8_root=stage26_8_root, run_horizon_chain=False, starts_online_canary=True)

    summary = s26.run_xunce_stage26_8a_expand_seed_or_horizon_budget(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage26_8a_boundary_rejections"


def _make_stage26_8_root(tmp_path: Path, **overrides: object) -> Path:
    root = tmp_path / "stage26_8"
    summary = _stage26_8_summary(positive=0, nonnegative=3, mean_per100=0.0)
    summary.update(
        {
            "schema_version": "xunce-stage26-8-summary/v1",
            "stage_id": "xunce-stage26-8-synthetic-terrain-multi-seed-coverage-efficiency-pilot",
            "status": "failed",
            "next_required_change": "expand_stage26_8_seed_or_horizon_budget",
            "input_rejections": [],
            "boundary_rejections": [],
        }
    )
    summary.update(overrides)
    root.mkdir(parents=True)
    _write_json(root / "xunce-stage26-8-summary.json", summary)
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


def _write_config(tmp_path: Path, *, stage26_8_root: Path, **overrides: object) -> Path:
    payload = {
        "schema_version": "xunce-stage26-8a-expand-seed-or-horizon-budget-config/v1",
        "stage_id": "xunce-stage26-8a-expand-seed-or-horizon-budget",
        "stage26_8_root": str(stage26_8_root),
        "stage26_8_base_config": str(overrides.pop("stage26_8_base_config", _write_stage26_8_base_config(tmp_path))),
        "horizons": [12, 16, 20],
        "seeds": [260801, 260802, 260803],
        "run_horizon_chain": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage26_8a_config.json"
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
    seed_count = 3
    majority = 2
    return {
        "horizon_steps": horizon,
        "horizon_label": f"h{horizon}",
        "seed_count": seed_count,
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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
