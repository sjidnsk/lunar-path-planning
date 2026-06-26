import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage24_3_runs_hybrid_path_cost_collector_reward_batch_smoke(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke as s24

    _patch_stage21_runs(monkeypatch, s24)
    config = _write_config(tmp_path, _write_stage24_2_root(tmp_path))

    summary = s24.run_xunce_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    audit = json.loads((tmp_path / "out" / "xunce-stage24-3-hybrid-path-transition-contract-audit.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage24_4_hybrid_astar_path_cost_ppo_update_smoke"
    assert summary["coverage_source"] == "endpoint_theta_slope_obstacle_los/v1"
    assert summary["path_cost_source"] == "hybrid_astar_pose_path/v1"
    assert summary["hybrid_path_transition_contract_missing_count"] == 0
    assert summary["hybrid_path_reward_contract_missing_count"] == 0
    assert summary["hybrid_path_batch_contract_missing_count"] == 0
    assert summary["point_grid_path_cost_fallback_used_count"] == 0
    assert audit["stage21_3_rejects_grid_only_path_cost_batch"] is True
    assert summary["default_astar_replaced"] is False
    assert summary["ackermann_feasible_claimed"] is False
    assert summary["runs_new_ppo_update"] is False
    stage21_1_config = json.loads((tmp_path / "out" / "xunce-stage24-3-stage21-1-config.json").read_text(encoding="utf-8"))
    stage21_2_config = json.loads((tmp_path / "out" / "xunce-stage24-3-stage21-2-config.json").read_text(encoding="utf-8"))
    stage21_3_config = json.loads((tmp_path / "out" / "xunce-stage24-3-stage21-3-config.json").read_text(encoding="utf-8"))
    assert stage21_1_config["hybrid_astar_pose_path_cost_enabled"] is True
    assert stage21_1_config["source_roi_expansion_root"].endswith("high_res_roi_expansion")
    assert stage21_2_config["require_hybrid_astar_path_cost_contract"] is True
    assert stage21_3_config["require_hybrid_astar_path_cost_contract"] is True


def test_stage24_3_routes_collector_repair_when_hybrid_fields_missing(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke as s24

    _patch_stage21_runs(monkeypatch, s24, transition_mode="missing_hybrid")
    config = _write_config(tmp_path, _write_stage24_2_root(tmp_path))

    summary = s24.run_xunce_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage24_3_collector_hybrid_path_cost_contract"
    assert "hybrid_path_transition_contract_missing" in summary["blocking_reason_codes"]


def test_stage24_3_routes_reward_repair_when_reward_uses_grid_fallback(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke as s24

    _patch_stage21_runs(monkeypatch, s24, reward_mode="grid_fallback")
    config = _write_config(tmp_path, _write_stage24_2_root(tmp_path))

    summary = s24.run_xunce_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage24_3_reward_path_cost_provenance"
    assert "point_grid_path_cost_fallback_used" in summary["blocking_reason_codes"]


def test_stage24_3_routes_batch_repair_when_grid_only_batch_is_allowed(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke as s24

    _patch_stage21_runs(monkeypatch, s24, batch_mode="missing_hybrid")
    config = _write_config(tmp_path, _write_stage24_2_root(tmp_path))

    summary = s24.run_xunce_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage24_3_batch_gate"
    assert "hybrid_path_batch_contract_missing" in summary["blocking_reason_codes"]


def test_stage24_3_rejects_unready_stage24_2_without_running_substages(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke as s24

    _patch_stage21_runs(monkeypatch, s24)
    config = _write_config(
        tmp_path,
        _write_stage24_2_root(tmp_path, next_required_change="repair_stage24_2_reward_path_cost_source"),
    )

    summary = s24.run_xunce_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage24_3_required_inputs"
    assert "stage24_2_route_not_stage24_3" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s21_1").exists()


def _patch_stage21_runs(
    monkeypatch,
    s24,
    *,
    transition_mode: str = "valid",
    reward_mode: str = "valid",
    batch_mode: str = "valid",
) -> None:
    def fake_stage21_1(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        output_root.mkdir(parents=True, exist_ok=True)
        rows = [_transition(mode=transition_mode)]
        _write_jsonl(output_root / s24.stage21_1.TRAINABLE_BATCH_FILE, rows)
        _write_jsonl(output_root / s24.stage21_1.TRANSITIONS_FILE, rows)
        _write_jsonl(output_root / s24.stage21_1.REJECTION_FILE, [])
        summary = {"status": "passed", "next_required_change": "implement_stage21_2_coverage_first_ppo_reward_contract", "trainable_transition_count": 1}
        (output_root / s24.stage21_1.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        manifest = {"model_audit": {"source_roi_expansion_root": cfg.get("source_roi_expansion_root")}}
        (output_root / s24.stage21_1.MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
        return summary

    def fake_stage21_2(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        output_root.mkdir(parents=True, exist_ok=True)
        rows = [_reward(mode=reward_mode)]
        _write_jsonl(output_root / s24.stage21_2.EVALUATION_FILE, rows)
        summary = {
            "status": "passed",
            "next_required_change": "implement_stage21_3_ppo_batch_validation",
            "hybrid_astar_path_cost_contract_missing_count": 0 if reward_mode == "valid" else 1,
            "point_grid_path_cost_fallback_used_count": 0 if reward_mode == "valid" else 1,
        }
        (output_root / s24.stage21_2.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        return summary

    def fake_stage21_3(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        output_root.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_root / s24.stage21_3.BATCH_FILE, [_batch_row(mode=batch_mode)])
        summary = {"status": "passed", "next_required_change": "implement_stage21_4_tiny_ppo_update_smoke", "trainable_transition_count": 1}
        (output_root / s24.stage21_3.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(s24.stage21_1, "run_xunce_stage21_1_on_policy_ppo_rollout_collector", fake_stage21_1)
    monkeypatch.setattr(s24.stage21_2, "run_xunce_stage21_2_coverage_first_ppo_reward_contract", fake_stage21_2)
    monkeypatch.setattr(s24.stage21_3, "run_xunce_stage21_3_ppo_batch_validation", fake_stage21_3)


def _transition(*, mode: str) -> dict:
    info = _info()
    if mode == "missing_hybrid":
        info = {key: value for key, value in info.items() if not key.startswith("hybrid_astar") and not key.startswith("legacy_grid") and not key.startswith("hybrid_vs_grid") and key != "path_cost_source"}
    return {
        "transition_id": "s1:step-0:sample-1",
        "scenario_id": "s1",
        "step_index": 0,
        "action_index": 1,
        "trainable": True,
        "info": info,
    }


def _reward(*, mode: str) -> dict:
    row = _batch_row(mode="valid")
    if mode == "grid_fallback":
        row["hybrid_astar_path_cost_reward_contract"] = False
        row["path_cost_source"] = "legacy_grid_astar_path/v1"
        row["point_grid_path_cost_fallback_used"] = True
        row["reward_metrics"]["path_cost_m"] = row["legacy_grid_astar_path_cost"]
    return row


def _batch_row(*, mode: str) -> dict:
    row = {
        "transition_id": "s1:step-0:sample-1",
        "action_index": 1,
        "theta_aware_reward_contract": True,
        "slope_obstacle_aware_theta_reward_contract": True,
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "candidate_viewpoint": [1, 2, 45],
        "candidate_theta_deg": 45,
        "obstacle_aware_new_visible_cell_count": 4,
        "obstacle_aware_theta_coverage_hash": "obstacle-hash-45",
        "obstacle_aware_theta_coverage_gain_per_path_cost": 2.0,
        "slope_obstacle_source_hash": "slope-source-hash",
        "platform_contract_hash": "platform-hash",
        "max_traversable_slope_deg": 30.0,
        "slope_blocked_source_kind": "slope_blocked_as_obstacle_proxy",
        "point_only_reward_fallback_used": False,
        "unobstructed_theta_reward_fallback_used": False,
        "hybrid_astar_path_cost_reward_contract": True,
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "hybrid_astar_path_cost": 5.5,
        "hybrid_astar_pose_path_hash": "pose-path-hash",
        "hybrid_astar_trajectory_kind": "hybrid_astar_pose_path",
        "legacy_grid_astar_path_cost": 4.0,
        "hybrid_vs_grid_path_cost_delta": 1.5,
        "default_astar_replaced": False,
        "hybrid_astar_ackermann_feasible_claimed": False,
        "point_grid_path_cost_fallback_used": False,
        "reward_metrics": {"path_cost_m": 5.5},
        "info": _info(),
    }
    if mode == "missing_hybrid":
        row["hybrid_astar_path_cost_reward_contract"] = False
        row["path_cost_source"] = "legacy_grid_astar_path/v1"
        row["point_grid_path_cost_fallback_used"] = True
        row["reward_metrics"] = {"path_cost_m": 4.0}
    return row


def _info() -> dict:
    return {
        "candidate_viewpoints": [[1, 2, 0], [1, 2, 45]],
        "candidate_theta_deg": [0, 45],
        "selected_viewpoint": [1, 2, 45],
        "selected_theta_deg": 45,
        "action_mask": [True, True],
        "sampling_mask": [True, True],
        "hard_risk_clean_mask": [True, True],
        "obstacle_aware_new_visible_cell_counts": [2, 4],
        "obstacle_aware_theta_coverage_hashes": ["obstacle-hash-0", "obstacle-hash-45"],
        "obstacle_aware_theta_coverage_gain_per_path_costs": [1.0, 2.0],
        "slope_obstacle_source_hash": "slope-source-hash",
        "slope_obstacle_source_hashes": ["slope-source-hash", "slope-source-hash"],
        "platform_contract_hash": "platform-hash",
        "platform_contract_hashes": ["platform-hash", "platform-hash"],
        "max_traversable_slope_deg": 30.0,
        "slope_blocked_source_kind": "slope_blocked_as_obstacle_proxy",
        "strict_obstacle_aware_new_visible_cell_count": True,
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "path_cost_sources": ["hybrid_astar_pose_path/v1", "hybrid_astar_pose_path/v1"],
        "hybrid_astar_path_costs": [4.5, 5.5],
        "hybrid_astar_pose_path_hashes": ["pose-path-hash-0", "pose-path-hash"],
        "hybrid_astar_trajectory_kinds": ["hybrid_astar_pose_path", "hybrid_astar_pose_path"],
        "legacy_grid_astar_path_costs": [3.0, 4.0],
        "hybrid_vs_grid_path_cost_deltas": [1.5, 1.5],
        "default_astar_replaced": False,
        "default_astar_replaced_flags": [False, False],
        "hybrid_astar_ackermann_feasible_claimed": False,
        "hybrid_astar_ackermann_feasible_claimed_flags": [False, False],
        "path_cost": 4.0,
        "hard_risk_violation": False,
    }


def _write_config(tmp_path: Path, stage24_2_root: Path, **overrides) -> Path:
    payload = {
        "schema_version": "xunce-stage24-3-hybrid-astar-path-cost-ppo-collector-smoke-config/v1",
        "stage24_2_root": str(stage24_2_root),
        "required_scenario_count": 2,
        "rollout_steps": 4,
        "dynamic_max_candidates_per_step": 36,
        "dynamic_proposal_pool_limit_per_step": 288,
        "min_trainable_transition_count": 1,
        "stage24_3_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage24_3_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_stage24_2_root(
    tmp_path: Path,
    *,
    status: str = "passed",
    next_required_change: str = "run_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke",
) -> Path:
    stage23_2b = tmp_path / "stage23_2b"
    high_res = tmp_path / "high_res_roi_expansion"
    stage23_2b.mkdir()
    high_res.mkdir()
    (stage23_2b / "xunce-stage23-2b-rerun-stage23-2a-summary.json").write_text(
        json.dumps({"status": "passed", "high_res_root": str(high_res)}),
        encoding="utf-8",
    )
    root = tmp_path / "stage24_2"
    root.mkdir()
    summary = {
        "schema_version": "xunce-stage24-2-summary/v1",
        "status": status,
        "next_required_change": next_required_change,
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "max_traversable_slope_deg": 30.0,
        "platform_contract_id": "agilex_scout_mini_piper",
        "platform_contract_hash": "platform-hash",
        "stage23_2b_root": str(stage23_2b),
        "default_astar_replaced": False,
        "ackermann_feasible_claimed": False,
    }
    (root / "xunce-stage24-2-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return root


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
