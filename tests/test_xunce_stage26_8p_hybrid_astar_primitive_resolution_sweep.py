import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_8p_rejects_wrong_stage26_8o_route(tmp_path: Path) -> None:
    from scripts import run_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep as runner

    config = _write_config(tmp_path, stage26_8o_route="not_expected")
    summary = runner.run_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8p_required_inputs"


def test_stage26_8p_run_next_executes_one_combo_and_passes_only_primitive_duration(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep as runner

    calls: list[dict] = []
    monkeypatch.setattr(runner.stage26_1, "run_xunce_stage26_1_synthetic_terrain_collector_smoke", _fake_stage26_1(calls, {}))

    summary = runner.run_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep(
        config_path=_write_config(tmp_path),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["next_required_change"] == "continue_stage26_8p_primitive_resolution_sweep"
    assert len(calls) == 1
    cfg = json.loads(Path(calls[0]["config_path"]).read_text(encoding="utf-8"))
    assert cfg["stage26_8p_combo_id"] == "p4_0_baseline"
    assert cfg["hybrid_astar_primitive_duration_s"] == 4.0
    assert cfg["hybrid_astar_integration_dt_s"] == 0.25
    assert cfg["hybrid_astar_max_speed_mps"] == 1.0
    assert cfg["hybrid_astar_max_angular_speed_degps"] == 45.0
    assert cfg["selected_continuous_theta_reachability_guard_enabled"] is True


def test_stage26_8p_aggregate_only_does_not_run_stage26_1(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep as runner

    def fail_if_called(**_kwargs):
        raise AssertionError("Stage26.1 should not run in aggregate_only")

    monkeypatch.setattr(runner.stage26_1, "run_xunce_stage26_1_synthetic_terrain_collector_smoke", fail_if_called)
    summary = runner.run_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep(
        config_path=_write_config(tmp_path, run_mode="aggregate_only"),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["next_required_change"] == "continue_stage26_8p_primitive_resolution_sweep"
    assert summary["completed_combo_count"] == 0


def test_stage26_8p_recommends_fine_primitive_when_trainable_target_met(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep as runner

    counts = {"p4_0_baseline": 63, "p2_0_medium": 88, "p1_0_fine": 120}
    calls: list[dict] = []
    monkeypatch.setattr(runner.stage26_1, "run_xunce_stage26_1_synthetic_terrain_collector_smoke", _fake_stage26_1(calls, counts))
    config = _write_config(tmp_path)
    out = tmp_path / "out"

    for _ in range(3):
        runner.run_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep(
            config_path=config,
            output_root=out,
            repo_root=REPO_ROOT,
        )

    summary = runner.run_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep(
        config_path=config,
        output_root=out,
        repo_root=REPO_ROOT,
        run_mode_override="aggregate_only",
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "rerun_stage26_8n_aggressive_update_sweep_with_repaired_primitive_resolution"
    assert summary["best_combo_id"] == "p1_0_fine"
    recommended = json.loads((out / runner.RECOMMENDED_STAGE26_8N_CONFIG_FILE).read_text(encoding="utf-8"))
    assert recommended["hybrid_astar_primitive_duration_s"] == 1.0
    assert recommended["publishes_checkpoint"] is False


def test_stage26_8p_routes_to_start_pool_when_no_material_improvement(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep as runner

    counts = {"p4_0_baseline": 63, "p2_0_medium": 65, "p1_0_fine": 70}
    calls: list[dict] = []
    monkeypatch.setattr(runner.stage26_1, "run_xunce_stage26_1_synthetic_terrain_collector_smoke", _fake_stage26_1(calls, counts))
    config = _write_config(tmp_path)
    out = tmp_path / "out"

    for _ in range(3):
        runner.run_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep(
            config_path=config,
            output_root=out,
            repo_root=REPO_ROOT,
        )

    summary = runner.run_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep(
        config_path=config,
        output_root=out,
        repo_root=REPO_ROOT,
        run_mode_override="aggregate_only",
    )

    assert summary["next_required_change"] == "repair_stage26_synthetic_start_pool_or_candidate_reachability"


def test_stage26_8p_routes_to_expand_with_best_primitive_on_material_short_improvement(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep as runner

    counts = {"p4_0_baseline": 63, "p2_0_medium": 92, "p1_0_fine": 95}
    calls: list[dict] = []
    monkeypatch.setattr(runner.stage26_1, "run_xunce_stage26_1_synthetic_terrain_collector_smoke", _fake_stage26_1(calls, counts))
    config = _write_config(tmp_path)
    out = tmp_path / "out"

    for _ in range(3):
        runner.run_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep(
            config_path=config,
            output_root=out,
            repo_root=REPO_ROOT,
        )

    summary = runner.run_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep(
        config_path=config,
        output_root=out,
        repo_root=REPO_ROOT,
        run_mode_override="aggregate_only",
    )

    assert summary["next_required_change"] == "expand_stage26_8o_sample_budget_with_repaired_primitive_resolution"
    audit = json.loads((out / runner.REACHABILITY_AUDIT_FILE).read_text(encoding="utf-8"))
    assert audit["material_reachability_improvement"] is True


def _write_config(
    tmp_path: Path,
    *,
    stage26_8o_route: str = "expand_stage26_8o_sample_budget_or_start_pool",
    run_mode: str = "run_next",
) -> Path:
    stage26_8o_root = tmp_path / "stage26_8o"
    fixture_root = tmp_path / "fixture"
    stage26_8o_root.mkdir()
    fixture_root.mkdir()
    (stage26_8o_root / "xunce-stage26-8o-summary.json").write_text(
        json.dumps({"status": "failed", "next_required_change": stage26_8o_route}),
        encoding="utf-8",
    )
    payload = {
        "schema_version": "xunce-stage26-8p-hybrid-astar-primitive-resolution-sweep-config/v1",
        "stage_id": "xunce-stage26-8p-hybrid-astar-primitive-resolution-sweep",
        "run_mode": run_mode,
        "stage26_8o_root": str(stage26_8o_root),
        "source_scenario_fixture_root": str(fixture_root),
        "base_stage26_1_config": "configs/xunce_stage26_1_synthetic_terrain_collector_smoke_v1.json",
        "required_scenario_count": 6,
        "rollout_steps": 20,
        "min_trainable_transition_count": 100,
        "sampling_seed": 260801,
        "scenario_seed_base": 260801,
        "baseline_combo_id": "p4_0_baseline",
        "material_trainable_improvement_threshold": 20,
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
        "hybrid_astar_candidate_eval_workers": 4,
        "max_traversable_slope_deg": 30.0,
        "primitive_sweep_combos": [
            {
                "combo_id": "p4_0_baseline",
                "hybrid_astar_primitive_duration_s": 4.0,
                "hybrid_astar_integration_dt_s": 0.25,
                "hybrid_astar_max_speed_mps": 1.0,
                "hybrid_astar_max_angular_speed_degps": 45.0,
            },
            {
                "combo_id": "p2_0_medium",
                "hybrid_astar_primitive_duration_s": 2.0,
                "hybrid_astar_integration_dt_s": 0.25,
                "hybrid_astar_max_speed_mps": 1.0,
                "hybrid_astar_max_angular_speed_degps": 45.0,
            },
            {
                "combo_id": "p1_0_fine",
                "hybrid_astar_primitive_duration_s": 1.0,
                "hybrid_astar_integration_dt_s": 0.25,
                "hybrid_astar_max_speed_mps": 1.0,
                "hybrid_astar_max_angular_speed_degps": 45.0,
            },
        ],
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    path = tmp_path / "stage26_8p_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fake_stage26_1(calls: list[dict], counts: dict[str, int]):
    def run_xunce_stage26_1_synthetic_terrain_collector_smoke(**kwargs):
        config_path = Path(kwargs["config_path"])
        output_root = Path(kwargs["output_root"])
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        combo_id = cfg["stage26_8p_combo_id"]
        count = counts.get(combo_id, 63)
        calls.append({"combo_id": combo_id, "config_path": str(config_path), "output_root": str(output_root)})
        s21_1 = output_root / "s21_1"
        s21_1.mkdir(parents=True, exist_ok=True)
        reachable = max(0, min(36, count // 20))
        (s21_1 / "xunce-stage21-1-ppo-trainable-batch.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "info": {
                            "hybrid_astar_reachable_flags": [True] * reachable + [False] * (36 - reachable),
                        }
                    }
                )
                for _ in range(max(1, count // 10))
            )
            + "\n",
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
            "batch_row_count": count if count >= 100 else 0,
            "reward_row_count": count if count >= 100 else 0,
            "stage21_1_status": "passed" if count >= 100 else "failed",
            "stage21_1_source_roi_expansion_root_match": True,
            "stage21_1_no_hybrid_reachable_candidate_terminal_count": terminal_count,
            "stage21_1_trainable_transition_count_by_scenario": {"stage26_synthetic_000": count},
            "stage21_1_selected_continuous_theta_unreachable_attempt_count": 10,
            "stage21_1_selected_continuous_theta_resample_success_count": 10,
            "stage21_1_selected_pose_unreachable_terminal_count": 0,
            "hard_risk_violation_count": 0,
            "mask_violation_count": 0,
            "path_planning_failure_count": 0,
            "open_grid_fallback_count": 0,
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
            "path_cost_source": "hybrid_astar_pose_path/v1",
            "platform_contract_hash": "platform-hash",
            "synthetic_terrain_hash": "synthetic-hash",
        }

    return run_xunce_stage26_1_synthetic_terrain_collector_smoke
