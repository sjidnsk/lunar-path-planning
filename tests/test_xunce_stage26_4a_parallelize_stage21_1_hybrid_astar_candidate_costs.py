import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_4a_passes_when_serial_and_parallel_collector_outputs_match(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_4a_parallelize_stage21_1_hybrid_astar_candidate_costs as s26

    calls: list[int] = []

    def fake_stage26_1(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        workers = int(cfg["hybrid_astar_candidate_eval_workers"])
        calls.append(workers)
        _write_stage26_1_artifacts(s26, output_root, workers=workers, variant="same")
        return {
            "status": "passed",
            "next_required_change": "run_stage26_2_synthetic_terrain_ppo_update_smoke",
            "trainable_transition_count": 1,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }

    monkeypatch.setattr(s26.stage26_1, "run_xunce_stage26_1_synthetic_terrain_collector_smoke", fake_stage26_1)
    config = _write_config(tmp_path, _write_stage26_0_root(tmp_path))

    summary = s26.run_xunce_stage26_4a_parallelize_stage21_1_hybrid_astar_candidate_costs(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert calls == [1, 4]
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "rerun_stage26_4_synthetic_policy_update_signal_strength_with_parallel_collector"
    assert summary["parallel_equivalence_mismatch_count"] == 0
    assert summary["serial_worker_count"] == 1
    assert summary["parallel_worker_count"] == 4
    assert summary["release_or_boundary_clean"] is True


def test_stage26_4a_routes_equivalence_repair_when_parallel_output_differs(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_4a_parallelize_stage21_1_hybrid_astar_candidate_costs as s26

    def fake_stage26_1(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        workers = int(cfg["hybrid_astar_candidate_eval_workers"])
        _write_stage26_1_artifacts(s26, output_root, workers=workers, variant="parallel-diff" if workers > 1 else "same")
        return {
            "status": "passed",
            "next_required_change": "run_stage26_2_synthetic_terrain_ppo_update_smoke",
            "trainable_transition_count": 1,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }

    monkeypatch.setattr(s26.stage26_1, "run_xunce_stage26_1_synthetic_terrain_collector_smoke", fake_stage26_1)
    config = _write_config(tmp_path, _write_stage26_0_root(tmp_path))

    summary = s26.run_xunce_stage26_4a_parallelize_stage21_1_hybrid_astar_candidate_costs(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_4a_parallel_equivalence"
    assert summary["parallel_equivalence_mismatch_count"] > 0


def _write_stage26_1_artifacts(module, output_root: Path, *, workers: int, variant: str) -> None:
    s21_1 = output_root / "s21_1"
    s21_1.mkdir(parents=True, exist_ok=True)
    pose_hash = "pose-path-hash"
    if variant == "parallel-diff":
        pose_hash = "pose-path-hash-different"
    row = {
        "transition_id": "scenario-a:step-0:sample-0",
        "scenario_id": "scenario-a",
        "step_index": 0,
        "action_index": 0,
        "trainable": True,
        "info": {
            "candidate_set_hash": "candidate-set-hash",
            "selected_viewpoint": [1, 1, 45],
            "path_cost_sources": ["hybrid_astar_pose_path/v1", "hybrid_astar_pose_path/v1"],
            "hybrid_astar_path_costs": [5.0, 7.0],
            "hybrid_astar_pose_path_hashes": [pose_hash, "pose-path-hash-2"],
            "hybrid_astar_reachable_flags": [True, True],
            "synthetic_terrain_hash": "synthetic-hash",
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "platform_contract_hash": "platform-hash",
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
            "path_cost_source": "hybrid_astar_pose_path/v1",
            "hybrid_astar_candidate_eval_parallel_enabled": workers > 1,
            "hybrid_astar_candidate_eval_workers_requested": workers,
            "hybrid_astar_candidate_eval_workers_effective": workers,
            "hybrid_astar_candidate_eval_submitted_count": 2,
            "hybrid_astar_candidate_eval_failed_count": 0,
            "hybrid_astar_candidate_eval_duration_s": 0.1 if workers == 1 else 0.05,
        },
    }
    _write_jsonl(s21_1 / module.stage21_1.TRAINABLE_BATCH_FILE, [row])
    _write_jsonl(s21_1 / module.stage21_1.TRANSITIONS_FILE, [row])
    (s21_1 / module.stage21_1.SUMMARY_FILE).write_text(
        json.dumps({"status": "passed", "trainable_transition_count": 1}, ensure_ascii=False),
        encoding="utf-8",
    )
    (s21_1 / module.stage21_1.MANIFEST_FILE).write_text(
        json.dumps({"model_audit": {"source_roi_expansion_root": "synthetic-root"}}, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_stage26_0_root(tmp_path: Path) -> Path:
    root = tmp_path / "stage26_0"
    root.mkdir()
    summary = {
        "schema_version": "xunce-stage26-0-summary/v1",
        "status": "passed",
        "next_required_change": "run_stage26_1_synthetic_terrain_collector_smoke",
        "synthetic_terrain_model_id": "synthetic_rock_pit_terrain/v1",
        "synthetic_terrain_hash": "synthetic-hash",
        "source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "physical_obstacle_cells_written": False,
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "max_traversable_slope_deg": 30.0,
    }
    (root / "xunce-stage26-0-summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    return root


def _write_config(tmp_path: Path, stage26_0_root: Path) -> Path:
    payload = {
        "schema_version": "xunce-stage26-4a-parallelize-stage21-1-hybrid-astar-candidate-costs-config/v1",
        "stage26_0_root": str(stage26_0_root),
        "stage26_1_base_config": "configs/xunce_stage26_1_synthetic_terrain_collector_smoke_v1.json",
        "required_scenario_count": 2,
        "rollout_steps": 4,
        "dynamic_max_candidates_per_step": 36,
        "dynamic_proposal_pool_limit_per_step": 288,
        "serial_hybrid_astar_candidate_eval_workers": 1,
        "parallel_hybrid_astar_candidate_eval_workers": 4,
        "stage26_4a_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    path = tmp_path / "stage26_4a_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
