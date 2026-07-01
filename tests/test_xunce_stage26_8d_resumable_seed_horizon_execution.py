from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_8d_aggregate_only_carries_over_completed_eval_and_update(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8d_resumable_seed_horizon_execution as s26

    stage26_8b_root = _make_stage26_8b_root(tmp_path)
    stage26_8c_root = _make_stage26_8c_root(tmp_path)
    _write_stage26_1_summary(stage26_8c_root / "h16" / "s26_8" / "s0" / "s26_1")
    _write_stage26_2_summary(stage26_8c_root / "h16" / "s26_8" / "s0" / "s26_2")
    _write_stage26_3_summary(stage26_8c_root / "h16" / "s26_8" / "s0" / "s26_3", status="failed", per100m=0.0)
    _write_stage26_1_summary(stage26_8c_root / "h16" / "s26_8" / "s1" / "s26_1")
    _write_stage26_2_summary(stage26_8c_root / "h16" / "s26_8" / "s1" / "s26_2")
    config = _write_config(tmp_path, stage26_8b_root=stage26_8b_root, stage26_8c_root=stage26_8c_root, run_mode="aggregate_only")

    summary = s26.run_xunce_stage26_8d_resumable_seed_horizon_execution(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "continue_stage26_8d_seed_horizon_jobs"
    assert summary["completed_job_count"] == 1
    assert summary["pending_job_count"] == 5
    assert summary["next_job_id"] == "h16_s260802"
    assert summary["next_phase"] == "stage26_3"
    carryover = _read_json(tmp_path / "out" / "xunce-stage26-8d-carryover-audit.json")
    assert carryover["carried_over_completed_eval_count"] == 1
    assert carryover["carried_over_update_complete_next_eval_count"] == 1


def test_stage26_8d_run_next_executes_one_missing_phase(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8d_resumable_seed_horizon_execution as s26

    stage26_8b_root = _make_stage26_8b_root(tmp_path)
    stage26_8c_root = _make_stage26_8c_root(tmp_path)
    _write_stage26_1_summary(stage26_8c_root / "h16" / "s26_8" / "s0" / "s26_1")
    _write_stage26_2_summary(stage26_8c_root / "h16" / "s26_8" / "s0" / "s26_2")
    _write_stage26_3_summary(stage26_8c_root / "h16" / "s26_8" / "s0" / "s26_3", status="failed", per100m=0.0)
    _write_stage26_1_summary(stage26_8c_root / "h16" / "s26_8" / "s1" / "s26_1")
    _write_stage26_2_summary(stage26_8c_root / "h16" / "s26_8" / "s1" / "s26_2")
    config = _write_config(tmp_path, stage26_8b_root=stage26_8b_root, stage26_8c_root=stage26_8c_root)
    calls: list[tuple[str, str]] = []

    def fake_run_phase(item: dict, *, config: dict, output_root: Path, repo_root: Path, started_at: str) -> dict:
        calls.append((item["job_id"], item["phase"]))
        assert item["job_id"] == "h16_s260802"
        assert item["phase"] == "stage26_3"
        assert item["stage26_2_root"].endswith("h16\\s26_8\\s1\\s26_2") or item["stage26_2_root"].endswith("h16/s26_8/s1/s26_2")
        root = output_root / "h16" / "s260802" / "s26_3"
        _write_stage26_3_summary(root, status="failed", per100m=0.0)
        return {
            "job_id": item["job_id"],
            "phase": item["phase"],
            "started_at": started_at,
            "finished_at": started_at,
            "summary_path": str(root / "xunce-stage26-3-summary.json"),
            "status": "failed",
            "next_required_change": "repair_stage26_synthetic_policy_update_signal_strength",
        }

    monkeypatch.setattr(s26, "_run_phase", fake_run_phase)

    summary = s26.run_xunce_stage26_8d_resumable_seed_horizon_execution(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert calls == [("h16_s260802", "stage26_3")]
    assert summary["phase_execution_count"] == 1
    assert summary["completed_job_count"] == 2
    assert summary["next_required_change"] == "continue_stage26_8d_seed_horizon_jobs"
    rows = _read_jsonl(tmp_path / "out" / "xunce-stage26-8d-job-state.jsonl")
    assert [row for row in rows if row["job_id"] == "h16_s260802"][0]["stage26_3_source"] == "stage26_8d_output"


def test_stage26_8d_failed_carryover_is_not_success(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8d_resumable_seed_horizon_execution as s26

    stage26_8b_root = _make_stage26_8b_root(tmp_path)
    stage26_8c_root = _make_stage26_8c_root(tmp_path)
    _write_stage26_1_summary(stage26_8c_root / "h16" / "s26_8" / "s0" / "s26_1")
    _write_stage26_2_summary(stage26_8c_root / "h16" / "s26_8" / "s0" / "s26_2")
    _write_stage26_3_summary(stage26_8c_root / "h16" / "s26_8" / "s0" / "s26_3", status="failed", per100m=0.0, lineage=False)
    config = _write_config(tmp_path, stage26_8b_root=stage26_8b_root, stage26_8c_root=stage26_8c_root, run_mode="aggregate_only")

    summary = s26.run_xunce_stage26_8d_resumable_seed_horizon_execution(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_8d_job_binding_or_safety"
    assert summary["completed_job_count"] == 0
    assert summary["failed_job_count"] == 1


def test_stage26_8d_routes_stage26_9_when_horizon_majority_positive(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8d_resumable_seed_horizon_execution as s26

    stage26_8b_root = _make_stage26_8b_root(tmp_path)
    stage26_8c_root = _make_stage26_8c_root(tmp_path)
    for horizon in (16, 20):
        for index, seed in enumerate((260801, 260802, 260803)):
            root = stage26_8c_root / f"h{horizon}" / "s26_8" / f"s{index}"
            _write_stage26_1_summary(root / "s26_1")
            _write_stage26_2_summary(root / "s26_2")
            _write_stage26_3_summary(
                root / "s26_3",
                status="passed" if horizon == 16 and index < 2 else "failed",
                route="run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot" if horizon == 16 and index < 2 else "repair_stage26_synthetic_policy_update_signal_strength",
                per100m=1.0 if horizon == 16 and index < 2 else 0.0,
                final_delta=0.01 if horizon == 16 and index < 2 else 0.0,
            )
    config = _write_config(tmp_path, stage26_8b_root=stage26_8b_root, stage26_8c_root=stage26_8c_root, run_mode="aggregate_only")

    summary = s26.run_xunce_stage26_8d_resumable_seed_horizon_execution(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot"
    assert summary["pending_job_count"] == 0
    assert summary["horizon_results"][0]["majority_positive"] is True


def test_stage26_8d_boundary_and_stage_runner_registration(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8d_resumable_seed_horizon_execution as s26

    stage26_8b_root = _make_stage26_8b_root(tmp_path)
    stage26_8c_root = _make_stage26_8c_root(tmp_path)
    config = _write_config(
        tmp_path,
        stage26_8b_root=stage26_8b_root,
        stage26_8c_root=stage26_8c_root,
        run_mode="aggregate_only",
        starts_online_canary=True,
    )

    summary = s26.run_xunce_stage26_8d_resumable_seed_horizon_execution(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage26_8d_boundary_rejections"
    registry = _read_json(REPO_ROOT / "configs" / "stage_registry.json")
    assert "xunce-stage26-8d-resumable-seed-horizon-execution" in registry["stages"]


def _make_stage26_8b_root(tmp_path: Path) -> Path:
    root = tmp_path / "stage26_8b"
    _write_json(
        root / "xunce-stage26-8b-summary.json",
        {
            "schema_version": "xunce-stage26-8b-summary/v1",
            "stage_id": "xunce-stage26-8b-repair-horizon-collector-terminal-reachability",
            "status": "passed",
            "next_required_change": "resume_stage26_8a_from_h16_h20",
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
            "path_cost_source": "hybrid_astar_pose_path/v1",
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    return root


def _make_stage26_8c_root(tmp_path: Path) -> Path:
    root = tmp_path / "stage26_8c"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_config(tmp_path: Path, *, stage26_8b_root: Path, stage26_8c_root: Path, **overrides: object) -> Path:
    path = tmp_path / "stage26_8d_config.json"
    payload = {
        "schema_version": "xunce-stage26-8d-resumable-seed-horizon-execution-config/v1",
        "stage_id": "xunce-stage26-8d-resumable-seed-horizon-execution",
        "stage26_8b_root": str(stage26_8b_root),
        "stage26_8c_root": str(stage26_8c_root),
        "stage26_8_base_config": str(tmp_path / "stage26_8_base.json"),
        "horizons": [16, 20],
        "seeds": [260801, 260802, 260803],
        "run_mode": "run_next",
        "max_jobs_per_invocation": 1,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    _write_json(path, payload)
    _write_json(
        tmp_path / "stage26_8_base.json",
        {
            "schema_version": "xunce-stage26-8-synthetic-terrain-multi-seed-coverage-efficiency-pilot-config/v1",
            "stage_id": "xunce-stage26-8-synthetic-terrain-multi-seed-coverage-efficiency-pilot",
            "stage26_7h_root": str(tmp_path / "stage26_7h"),
            "stage26_0_root": str(tmp_path / "stage26_0"),
            "stage26_1_base_config": str(tmp_path / "stage26_1.json"),
            "stage26_2_base_config": str(tmp_path / "stage26_2.json"),
            "stage26_3_base_config": str(tmp_path / "stage26_3.json"),
            "seeds": [260801, 260802, 260803],
        },
    )
    return path


def _write_stage26_1_summary(root: Path) -> None:
    _write_json(root / "xunce-stage26-1-summary.json", {"status": "passed", "next_required_change": "run_stage26_2_synthetic_terrain_ppo_update_smoke"})


def _write_stage26_2_summary(root: Path) -> None:
    _write_json(root / "xunce-stage26-2-summary.json", {"status": "passed", "next_required_change": "run_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke"})


def _write_stage26_3_summary(
    root: Path,
    *,
    status: str,
    route: str = "repair_stage26_synthetic_policy_update_signal_strength",
    per100m: float,
    final_delta: float = 0.0,
    lineage: bool = True,
) -> None:
    payload = {
        "status": status,
        "next_required_change": route,
        "post_update_success_metric": "main_coverable_coverage_efficiency/v1",
        "coverage_denominator_source": "main_coverable_cells/v1",
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
        "synthetic_terrain_hash": "synthetic-hash",
        "platform_contract_hash": "platform-hash",
        "max_traversable_slope_deg": 30.0,
        "final_coverage_delta": final_delta,
        "coverage_auc_delta": 0.0,
        "coverage_per_100m_delta": per100m,
        "hybrid_astar_path_cost_delta": 0.0,
        "stage21_5_runtime_blocker": False,
        "stage21_5_execution_reason_codes": [],
    }
    if not lineage:
        payload["coverage_source"] = "wrong"
    _write_json(root / "xunce-stage26-3-summary.json", payload)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
