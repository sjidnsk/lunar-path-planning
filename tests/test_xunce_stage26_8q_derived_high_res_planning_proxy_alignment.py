import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
PLANNER_SRC = str(REPO_ROOT / "path-planner" / "src")
for item in (SCRIPTS, PLANNER_SRC):
    if item not in sys.path:
        sys.path.insert(0, item)


def test_derived_proxy_grid_conservatively_upsamples_passable_mask() -> None:
    from path_planner.core import CostGrid, GridSpec
    from scripts.xunce_hybrid_astar_candidate_path_cost import build_derived_high_res_planning_proxy_grid

    source = CostGrid(
        GridSpec(width=2, height=1, resolution=4.0),
        np.asarray([[1.0, 7.0]], dtype=float),
        np.asarray([[True, False]], dtype=bool),
    )

    proxy = build_derived_high_res_planning_proxy_grid(source, 1.0)

    assert proxy.spec.width == 8
    assert proxy.spec.height == 4
    assert proxy.spec.resolution == 1.0
    assert proxy.passable_mask[:, :4].all()
    assert not proxy.passable_mask[:, 4:].any()
    assert proxy.metadata["planning_grid_source"] == "derived_high_res_planning_proxy/v1"
    assert proxy.metadata["source_grid_resolution_m"] == 4.0
    assert proxy.metadata["planner_grid_resolution_m"] == 1.0
    assert proxy.metadata["source_grid_hash"]
    assert proxy.metadata["planning_proxy_hash"]
    assert source.spec.resolution == 4.0

    redistributed = CostGrid(
        GridSpec(width=2, height=1, resolution=4.0),
        np.asarray([[7.0, 1.0]], dtype=float),
        np.asarray([[True, False]], dtype=bool),
    )
    redistributed_proxy = build_derived_high_res_planning_proxy_grid(redistributed, 1.0)
    assert redistributed_proxy.metadata["planning_proxy_hash"] != proxy.metadata["planning_proxy_hash"]


def test_closed_key_resolution_is_opt_in_and_default_stays_grid_cell_based() -> None:
    from path_planner.core import GridSpec
    from path_planner.search.hybrid_astar import HybridAStarPlanner, Pose2D

    spec = GridSpec(width=2, height=2, resolution=4.0)
    planner = HybridAStarPlanner()
    first = Pose2D(0.25, 0.25, 0.0)
    second = Pose2D(1.25, 0.25, 0.0)

    assert planner._key(spec, first, 72) == planner._key(spec, second, 72)
    assert planner._key(spec, first, 72, 1.0) != planner._key(spec, second, 72, 1.0)


def test_candidate_goal_world_pose_is_used_without_rewriting_candidate_viewpoint() -> None:
    from scripts.xunce_hybrid_astar_candidate_path_cost import (
        build_cost_grid_from_config,
        evaluate_hybrid_astar_candidate_path_cost,
    )

    grid = build_cost_grid_from_config({"width": 12, "height": 3, "resolution_m": 1.0})
    candidate = {
        "scenario_id": "s",
        "step_index": 0,
        "candidate_index": 0,
        "candidate_set_hash": "hash",
        "candidate_viewpoint": [1, 0, 0.0],
        "candidate_theta_deg": 0.0,
        "candidate_goal_world_pose": [5.5, 0.5],
    }

    row = evaluate_hybrid_astar_candidate_path_cost(
        grid=grid,
        current_pose=[0.5, 0.5, 0.0],
        candidate=candidate,
        platform_contract_hash="platform",
        max_traversable_slope_deg=30.0,
        goal_position_tolerance_m=1.0,
        goal_theta_tolerance_deg=45.0,
        primitive_duration_s=1.0,
        closed_key_xy_resolution_m=1.0,
    )

    assert row["candidate_viewpoint"] == [1, 0, 0.0]
    assert row["candidate_goal_world_pose"] == [5.5, 0.5]
    assert row["closed_key_xy_resolution_m"] == 1.0
    assert row["hybrid_astar_dominance_key_policy"] == "xy_resolution_m_theta_bin/v1"


def test_continuous_theta_probe_metadata_preserves_proxy_rejection_provenance() -> None:
    from scripts.run_xunce_stage21_1_on_policy_ppo_rollout_collector import (
        _aggregate_continuous_theta_probe_metadata,
        _planning_proxy_rejection_fields,
    )

    metadata = _aggregate_continuous_theta_probe_metadata(
        {
            "path_cost_sources": ["hybrid_astar_pose_path/v1", "hybrid_astar_pose_path/v1"],
            "hybrid_astar_path_costs": [2.0, 3.0],
            "hybrid_astar_pose_path_hashes": ["a", "b"],
            "hybrid_astar_trajectory_kinds": ["hybrid_astar_pose_path", "hybrid_astar_pose_path"],
            "hybrid_astar_reachable_flags": [True, True],
            "hybrid_astar_failure_reasons": [None, None],
            "legacy_grid_astar_path_costs": [None, None],
            "hybrid_vs_grid_path_cost_deltas": [None, None],
            "default_astar_replaced_flags": [False, False],
            "hybrid_astar_ackermann_feasible_claimed_flags": [False, False],
            "hybrid_astar_planning_grid_source": "derived_high_res_planning_proxy/v1",
            "planner_grid_resolution_m": 1.0,
            "source_grid_resolution_m": 4.0,
            "planning_proxy_hash": "proxy-hash",
            "closed_key_xy_resolution_m": 1.0,
            "planning_proxy_candidate_binding": "coarse_candidate_cell_center_world_pose/v1",
        },
        [[0.0, 45.0]],
    )

    fields = _planning_proxy_rejection_fields(metadata)

    assert fields["hybrid_astar_planning_grid_source"] == "derived_high_res_planning_proxy/v1"
    assert fields["planner_grid_resolution_m"] == 1.0
    assert fields["source_grid_resolution_m"] == 4.0
    assert fields["closed_key_xy_resolution_m"] == 1.0
    assert fields["planning_proxy_hash"] == "proxy-hash"
    assert fields["planning_proxy_candidate_binding"] == "coarse_candidate_cell_center_world_pose/v1"


def test_stage26_8q_rejects_wrong_stage26_8p_route(tmp_path: Path) -> None:
    from scripts import run_xunce_stage26_8q_derived_high_res_planning_proxy_alignment as runner

    summary = runner.run_xunce_stage26_8q_derived_high_res_planning_proxy_alignment(
        config_path=_write_config(tmp_path, stage26_8p_route="unexpected"),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8q_required_inputs"


def test_stage26_8q_run_next_executes_one_combo_and_proxy_config_is_opt_in(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage26_8q_derived_high_res_planning_proxy_alignment as runner

    calls: list[dict] = []
    monkeypatch.setattr(runner.stage26_1, "run_xunce_stage26_1_synthetic_terrain_collector_smoke", _fake_stage26_1(calls, {}))

    summary = runner.run_xunce_stage26_8q_derived_high_res_planning_proxy_alignment(
        config_path=_write_config(tmp_path),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["next_required_change"] == "continue_stage26_8q_proxy_alignment"
    assert len(calls) == 1
    cfg = json.loads(Path(calls[0]["config_path"]).read_text(encoding="utf-8"))
    assert cfg["stage26_8q_combo_id"] == "current_baseline"
    assert cfg["hybrid_astar_primitive_duration_s"] == 4.0
    assert cfg.get("hybrid_astar_planning_grid_source") is None
    assert cfg["selected_continuous_theta_reachability_guard_enabled"] is False
    assert cfg["selected_continuous_theta_unreachable_resample_policy"] == "terminal/v1"
    assert cfg["runs_new_ppo_update"] is False
    assert cfg["publishes_checkpoint"] is False
    assert cfg["replaces_default_policy"] is False
    assert cfg["connects_real_executor"] is False
    assert cfg["starts_online_canary"] is False


def test_stage26_8q_aggregate_only_does_not_run_stage26_1(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage26_8q_derived_high_res_planning_proxy_alignment as runner

    def fail_if_called(**_kwargs):
        raise AssertionError("Stage26.1 should not run in aggregate_only")

    monkeypatch.setattr(runner.stage26_1, "run_xunce_stage26_1_synthetic_terrain_collector_smoke", fail_if_called)

    summary = runner.run_xunce_stage26_8q_derived_high_res_planning_proxy_alignment(
        config_path=_write_config(tmp_path, run_mode="aggregate_only"),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["next_required_change"] == "continue_stage26_8q_proxy_alignment"
    assert summary["completed_combo_count"] == 0


def test_stage26_8q_boundary_rejection_does_not_run_stage26_1(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage26_8q_derived_high_res_planning_proxy_alignment as runner

    def fail_if_called(**_kwargs):
        raise AssertionError("Stage26.1 should not run when boundary is open")

    monkeypatch.setattr(runner.stage26_1, "run_xunce_stage26_1_synthetic_terrain_collector_smoke", fail_if_called)

    summary = runner.run_xunce_stage26_8q_derived_high_res_planning_proxy_alignment(
        config_path=_write_config(tmp_path, release_or_training_authorized=True),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage26_8q_boundary_rejections"
    assert "release_or_training_authorized" in summary["boundary_rejections"]


def test_stage26_8q_routes_to_rerun_8n_when_aligned_proxy_hits_target(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage26_8q_derived_high_res_planning_proxy_alignment as runner

    calls: list[dict] = []
    monkeypatch.setattr(
        runner.stage26_1,
        "run_xunce_stage26_1_synthetic_terrain_collector_smoke",
        _fake_stage26_1(calls, {"current_baseline": 63, "aligned_1m_proxy": 120}),
    )
    config = _write_config(tmp_path)
    out = tmp_path / "out"

    for _ in range(2):
        runner.run_xunce_stage26_8q_derived_high_res_planning_proxy_alignment(
            config_path=config,
            output_root=out,
            repo_root=REPO_ROOT,
        )

    summary = runner.run_xunce_stage26_8q_derived_high_res_planning_proxy_alignment(
        config_path=config,
        output_root=out,
        repo_root=REPO_ROOT,
        run_mode_override="aggregate_only",
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "rerun_stage26_8n_aggressive_update_sweep_with_aligned_planning_proxy"
    recommended = json.loads((out / runner.RECOMMENDED_STAGE26_8N_CONFIG_FILE).read_text(encoding="utf-8"))
    assert recommended["hybrid_astar_planning_grid_source"] == "derived_high_res_planning_proxy/v1"
    assert recommended["planner_grid_resolution_m"] == 1.0
    assert recommended["hybrid_astar_closed_key_xy_resolution_m"] == 1.0


def test_stage26_8q_routes_to_start_pool_when_aligned_proxy_has_no_material_gain(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage26_8q_derived_high_res_planning_proxy_alignment as runner

    calls: list[dict] = []
    monkeypatch.setattr(
        runner.stage26_1,
        "run_xunce_stage26_1_synthetic_terrain_collector_smoke",
        _fake_stage26_1(calls, {"current_baseline": 63, "aligned_1m_proxy": 70}),
    )
    config = _write_config(tmp_path)
    out = tmp_path / "out"

    for _ in range(2):
        runner.run_xunce_stage26_8q_derived_high_res_planning_proxy_alignment(
            config_path=config,
            output_root=out,
            repo_root=REPO_ROOT,
        )

    summary = runner.run_xunce_stage26_8q_derived_high_res_planning_proxy_alignment(
        config_path=config,
        output_root=out,
        repo_root=REPO_ROOT,
        run_mode_override="aggregate_only",
    )

    assert summary["next_required_change"] == "repair_stage26_synthetic_start_pool_or_candidate_reachability"


def test_stage26_8q_routes_to_expand_when_aligned_proxy_has_material_short_gain(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage26_8q_derived_high_res_planning_proxy_alignment as runner

    calls: list[dict] = []
    monkeypatch.setattr(
        runner.stage26_1,
        "run_xunce_stage26_1_synthetic_terrain_collector_smoke",
        _fake_stage26_1(calls, {"current_baseline": 63, "aligned_1m_proxy": 92}),
    )
    config = _write_config(tmp_path)
    out = tmp_path / "out"

    for _ in range(2):
        runner.run_xunce_stage26_8q_derived_high_res_planning_proxy_alignment(
            config_path=config,
            output_root=out,
            repo_root=REPO_ROOT,
        )

    summary = runner.run_xunce_stage26_8q_derived_high_res_planning_proxy_alignment(
        config_path=config,
        output_root=out,
        repo_root=REPO_ROOT,
        run_mode_override="aggregate_only",
    )

    assert summary["next_required_change"] == "expand_stage26_8q_sample_budget_with_aligned_proxy"


def _write_config(
    tmp_path: Path,
    *,
    stage26_8p_route: str = "repair_stage26_synthetic_start_pool_or_candidate_reachability",
    run_mode: str = "run_next",
    release_or_training_authorized: bool = False,
) -> Path:
    stage26_8p_root = tmp_path / "stage26_8p"
    fixture_root = tmp_path / "fixture"
    stage26_8p_root.mkdir()
    fixture_root.mkdir()
    (stage26_8p_root / "xunce-stage26-8p-summary.json").write_text(
        json.dumps({"status": "passed", "next_required_change": stage26_8p_route}),
        encoding="utf-8",
    )
    payload = {
        "schema_version": "xunce-stage26-8q-derived-high-res-planning-proxy-alignment-config/v1",
        "stage_id": "xunce-stage26-8q-derived-high-res-planning-proxy-alignment",
        "run_mode": run_mode,
        "stage26_8p_root": str(stage26_8p_root),
        "source_scenario_fixture_root": str(fixture_root),
        "base_stage26_1_config": "configs/xunce_stage26_1_synthetic_terrain_collector_smoke_v1.json",
        "required_scenario_count": 6,
        "rollout_steps": 20,
        "min_trainable_transition_count": 100,
        "material_trainable_improvement_threshold": 20,
        "sampling_seed": 260801,
        "scenario_seed_base": 260801,
        "baseline_combo_id": "current_baseline",
        "aligned_combo_id": "aligned_1m_proxy",
        "aligned_planner_grid_resolution_m": 1.0,
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
        "hybrid_astar_candidate_eval_workers": 4,
        "max_traversable_slope_deg": 30.0,
        "alignment_combos": [
            {
                "combo_id": "current_baseline",
                "hybrid_astar_primitive_duration_s": 4.0,
                "hybrid_astar_goal_position_tolerance_m": 4.0,
                "hybrid_astar_goal_theta_tolerance_deg": 10.0,
                "hybrid_astar_integration_dt_s": 0.25,
                "hybrid_astar_max_speed_mps": 1.0,
                "hybrid_astar_max_angular_speed_degps": 45.0,
            },
            {
                "combo_id": "aligned_1m_proxy",
                "hybrid_astar_planning_grid_source": "derived_high_res_planning_proxy/v1",
                "planner_grid_resolution_m": 1.0,
                "hybrid_astar_closed_key_xy_resolution_m": 1.0,
                "hybrid_astar_primitive_duration_s": 1.0,
                "hybrid_astar_goal_position_tolerance_m": 1.0,
                "hybrid_astar_goal_theta_tolerance_deg": 45.0,
                "hybrid_astar_integration_dt_s": 0.25,
                "hybrid_astar_max_speed_mps": 1.0,
                "hybrid_astar_max_angular_speed_degps": 45.0,
            },
        ],
        "release_or_training_authorized": release_or_training_authorized,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    path = tmp_path / "stage26_8q_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fake_stage26_1(calls: list[dict], counts: dict[str, int]):
    def run_xunce_stage26_1_synthetic_terrain_collector_smoke(**kwargs):
        config_path = Path(kwargs["config_path"])
        output_root = Path(kwargs["output_root"])
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        combo_id = cfg["stage26_8q_combo_id"]
        count = counts.get(combo_id, 63)
        calls.append({"combo_id": combo_id, "config_path": str(config_path), "output_root": str(output_root)})
        s21_1 = output_root / "s21_1"
        s21_1.mkdir(parents=True, exist_ok=True)
        proxy = cfg.get("hybrid_astar_planning_grid_source") == "derived_high_res_planning_proxy/v1"
        flags = [True] * min(36, max(1, count // 20)) + [False] * (36 - min(36, max(1, count // 20)))
        info = {
            "hybrid_astar_reachable_flags": flags,
            "hybrid_astar_planning_grid_source": cfg.get("hybrid_astar_planning_grid_source"),
            "planner_grid_resolution_m": cfg.get("planner_grid_resolution_m"),
            "source_grid_resolution_m": 4.0 if proxy else None,
            "planning_proxy_hash": "proxy-hash" if proxy else None,
            "planning_proxy_candidate_binding": "coarse_candidate_cell_center_world_pose/v1" if proxy else None,
        }
        (s21_1 / "xunce-stage21-1-ppo-trainable-batch.jsonl").write_text(
            "\n".join(json.dumps({"info": info}) for _ in range(max(1, count // 10))) + "\n",
            encoding="utf-8",
        )
        terminal_count = 0 if count >= 100 else 6
        (s21_1 / "xunce-stage21-1-rejection-report.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "reason": "no_hybrid_reachable_candidate_terminal",
                        "scenario_id": f"stage26_synthetic_{idx:03d}",
                        "step_index": idx + 1,
                        "action_mask_true_count": 36,
                        "hard_risk_clean_mask_true_count": 36,
                        "hybrid_astar_reachable_count": 0,
                        "sampling_mask_true_count": 0,
                        "hybrid_astar_planning_grid_source": cfg.get("hybrid_astar_planning_grid_source"),
                        "planner_grid_resolution_m": cfg.get("planner_grid_resolution_m"),
                        "source_grid_resolution_m": 4.0 if proxy else None,
                        "closed_key_xy_resolution_m": cfg.get("hybrid_astar_closed_key_xy_resolution_m"),
                        "planning_proxy_hash": "proxy-hash" if proxy else None,
                        "planning_proxy_candidate_binding": (
                            "coarse_candidate_cell_center_world_pose/v1" if proxy else None
                        ),
                    }
                )
                for idx in range(terminal_count)
            ),
            encoding="utf-8",
        )
        return {
            "status": "passed" if count >= 100 else "failed",
            "next_required_change": "run_stage26_2_synthetic_terrain_ppo_update_smoke" if count >= 100 else "expand_stage26_8o_sample_budget_or_start_pool",
            "transition_count": count,
            "batch_row_count": count,
            "reward_row_count": count,
            "stage21_1_status": "passed" if count >= 100 else "failed",
            "stage21_1_source_roi_expansion_root_match": True,
            "stage21_1_no_hybrid_reachable_candidate_terminal_count": terminal_count,
            "stage21_1_selected_pose_unreachable_terminal_count": 0,
            "hard_risk_violation_count": 0,
            "mask_violation_count": 0,
            "path_planning_failure_count": 0,
            "open_grid_fallback_count": 0,
            "synthetic_physical_obstacle_pollution_count": 0,
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
            "path_cost_source": "hybrid_astar_pose_path/v1",
            "platform_contract_hash": "platform-hash",
            "synthetic_terrain_hash": "synthetic-hash",
        }

    return run_xunce_stage26_1_synthetic_terrain_collector_smoke
