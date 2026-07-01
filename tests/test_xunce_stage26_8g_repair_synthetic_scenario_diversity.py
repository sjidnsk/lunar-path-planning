import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_8g_rejects_non_scenario_diversity_route(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8g_repair_synthetic_scenario_diversity as s26

    root = _write_stage26_8f_root(tmp_path, route="continue_stage26_8d_seed_horizon_jobs")
    summary = s26.run_xunce_stage26_8g_repair_synthetic_scenario_diversity(
        config_path=_write_config(tmp_path, root),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8g_required_inputs"
    assert "stage26_8f_route_mismatch" in summary["input_rejections"]


def test_stage26_8g_routes_policy_signal_after_diversity_repair(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8g_repair_synthetic_scenario_diversity as s26

    root = _write_stage26_8f_root(tmp_path)
    _patch_chain(monkeypatch, s26, tmp_path, diversity_repaired=True, per100m_delta=0.0)

    summary = s26.run_xunce_stage26_8g_repair_synthetic_scenario_diversity(
        config_path=_write_config(tmp_path, root),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    recheck = _read_json(tmp_path / "out" / "xunce-stage26-8g-scenario-diversity-recheck-audit.json")
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "repair_stage26_synthetic_policy_update_signal_strength"
    assert summary["scenario_fixture_count"] == 3
    assert summary["scenario_fixture_unique_start_count"] == 3
    assert recheck["scenario_diversity_repaired"] is True


def test_stage26_8g_routes_resume_when_diverse_and_efficiency_positive(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8g_repair_synthetic_scenario_diversity as s26

    root = _write_stage26_8f_root(tmp_path)
    _patch_chain(monkeypatch, s26, tmp_path, diversity_repaired=True, per100m_delta=1.25)

    summary = s26.run_xunce_stage26_8g_repair_synthetic_scenario_diversity(
        config_path=_write_config(tmp_path, root),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "resume_stage26_8d_seed_horizon_jobs_with_diverse_scenarios"
    assert summary["main_coverage_per_100m_delta"] == 1.25


def test_stage26_8g_routes_eval_diversity_when_recheck_still_duplicate(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8g_repair_synthetic_scenario_diversity as s26

    root = _write_stage26_8f_root(tmp_path)
    _patch_chain(monkeypatch, s26, tmp_path, diversity_repaired=False, per100m_delta=0.0)

    summary = s26.run_xunce_stage26_8g_repair_synthetic_scenario_diversity(
        config_path=_write_config(tmp_path, root),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_8g_eval_scenario_diversity_binding"


def test_stage26_8g_registry_entry_exists() -> None:
    registry = _read_json(REPO_ROOT / "configs" / "stage_registry.json")
    assert "xunce-stage26-8g-repair-synthetic-scenario-diversity" in registry["stages"]


def _patch_chain(monkeypatch, s26, tmp_path: Path, *, diversity_repaired: bool, per100m_delta: float) -> None:
    fixture_path = tmp_path / "fixtures.jsonl"
    fixture_rows = [
        {
            "scenario_id": f"stage26_synthetic_{idx:03d}",
            "scenario_start_cell": [idx, idx + 1],
            "scenario_diversity_signature_hash": f"fixture-{idx}",
            "scenario_diversity_content_hash": f"content-{idx}",
        }
        for idx in range(3)
    ]
    _write_jsonl(fixture_path, fixture_rows)

    def fake_stage26_1(config, run_root, repo_root):
        return {
            "status": "passed",
            "next_required_change": "run_stage26_2_synthetic_terrain_ppo_update_smoke",
            "scenario_fixture_catalog": str(fixture_path),
        }

    def fake_stage26_2(config, run_root, repo_root):
        return {"status": "passed", "next_required_change": "run_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke"}

    def fake_stage26_3(config, run_root, repo_root):
        return {
            "status": "failed",
            "next_required_change": "repair_stage26_synthetic_policy_update_signal_strength",
            "coverage_denominator_source": "main_coverable_cells/v1",
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
            "path_cost_source": "hybrid_astar_pose_path/v1",
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
            "main_coverage_per_100m_delta": per100m_delta,
            "selected_action_changed_count": 0,
        }

    def fake_recheck(stage26_3_root, config):
        return {
            "schema_version": "xunce-stage26-8g-scenario-diversity-recheck-audit/v1",
            "scenario_signature_count": 3,
            "scenario_signature_duplicate_group_count": 0 if diversity_repaired else 1,
            "scenario_signature_duplicate_ratio": 0.0 if diversity_repaired else 1.0,
            "scenario_diversity_repaired": diversity_repaired,
        }

    monkeypatch.setattr(s26, "_run_stage26_1", fake_stage26_1)
    monkeypatch.setattr(s26, "_run_stage26_2", fake_stage26_2)
    monkeypatch.setattr(s26, "_run_stage26_3", fake_stage26_3)
    monkeypatch.setattr(s26, "_recheck_stage26_3", fake_recheck)


def _write_stage26_8f_root(
    tmp_path: Path,
    *,
    status: str = "passed",
    route: str = "repair_stage26_synthetic_scenario_diversity",
) -> Path:
    root = tmp_path / "stage26_8f"
    _write_json(
        root / "xunce-stage26-8f-summary.json",
        {
            "schema_version": "xunce-stage26-8f-summary/v1",
            "stage_id": "xunce-stage26-8f-scenario-diversity-and-policy-margin-audit",
            "status": status,
            "next_required_change": route,
        },
    )
    _write_json(
        root / "xunce-stage26-8f-scenario-diversity-audit.json",
        {
            "scenario_signature_count": 3,
            "scenario_signature_duplicate_group_count": 1,
            "scenario_signature_duplicate_ratio": 1.0,
            "scenario_signatures": [
                {
                    "first_current_cell": [0, 0],
                    "candidate_set_sequence_hash": "same",
                    "covered_cells_sequence_hash": "same",
                    "selected_action_sequence_hash": "same",
                    "episode_metrics_hash": "same",
                }
                for _ in range(3)
            ],
        },
    )
    return root


def _write_config(tmp_path: Path, stage26_8f_root: Path) -> Path:
    path = tmp_path / "stage26_8g_config.json"
    _write_json(
        path,
        {
            "schema_version": "xunce-stage26-8g-repair-synthetic-scenario-diversity-config/v1",
            "stage_id": "xunce-stage26-8g-repair-synthetic-scenario-diversity",
            "stage26_8f_root": str(stage26_8f_root),
            "stage26_0_root": "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/outputs/path_feedback_batch_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract_v1",
            "stage26_1_base_config": "configs/xunce_stage26_1_synthetic_terrain_collector_smoke_v1.json",
            "stage26_2_base_config": "configs/xunce_stage26_2_synthetic_terrain_ppo_update_smoke_v1.json",
            "stage26_3_base_config": "configs/xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke_v1.json",
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
