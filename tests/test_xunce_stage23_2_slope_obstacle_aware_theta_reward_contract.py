import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage23_2_passes_with_platform_30deg_slope_los_reward_contract(tmp_path: Path) -> None:
    from scripts.run_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract import (
        run_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract,
    )

    stage23_2b = _write_stage23_2b_root(tmp_path)
    config = _write_config(tmp_path, stage23_2b)

    summary = run_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    replay_rows = _read_jsonl(tmp_path / "out" / "xunce-stage23-2-reward-replay.jsonl")
    manifest = json.loads((tmp_path / "out" / "xunce-stage23-2-manifest.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke"
    assert summary["coverage_source"] == "endpoint_theta_slope_obstacle_los/v1"
    assert summary["max_traversable_slope_deg"] == 30.0
    assert summary["slope_obstacle_reward_contract_missing_count"] == 0
    assert summary["slope_los_changed_row_count"] > 0
    assert summary["los_visible_count_proxy_used_count"] == 2
    assert summary["does_not_claim_collector_new_visible_provenance"] is True
    assert summary["stage23_3_collector_new_visible_required"] is True
    assert summary["stage21_2_slope_reward_supported"] is True
    assert summary["stage21_3_accepts_slope_reward_batch"] is True
    assert summary["stage21_3_rejects_unobstructed_theta_reward_batch"] is True
    assert summary["stage21_3_rejects_point_only_reward_batch"] is True
    assert replay_rows[0]["coverage_source"] == "endpoint_theta_slope_obstacle_los/v1"
    assert (
        replay_rows[0]["obstacle_aware_new_visible_cell_count_source"]
        == "stage23_0_obstacle_aware_visible_cell_count_replay_proxy"
    )
    assert replay_rows[0]["strict_obstacle_aware_new_visible_cell_count"] is False
    assert replay_rows[0]["slope_blocked_source_kind"] == "slope_blocked_as_obstacle_proxy"
    assert replay_rows[0]["point_only_reward_fallback_used"] is False
    assert replay_rows[0]["unobstructed_theta_reward_fallback_used"] is False
    assert manifest["los_audit_path"].endswith("xunce-stage23-0-obstacle-los-audit.jsonl")


def test_stage23_2_rejects_missing_slope_source_provenance(tmp_path: Path) -> None:
    from scripts.run_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract import (
        run_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract,
    )

    stage23_2b = _write_stage23_2b_root(tmp_path, obstacle_source_hash="")
    config = _write_config(tmp_path, stage23_2b)

    summary = run_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage23_2_reward_provenance"
    assert "slope_obstacle_reward_contract_missing" in summary["blocking_reason_codes"]


def test_stage23_2_discovers_short_stage23_2b_los_audit_path(tmp_path: Path) -> None:
    from scripts.run_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract import (
        run_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract,
    )

    stage23_2b = _write_stage23_2b_root(tmp_path, layout="short")
    config = _write_config(tmp_path, stage23_2b)

    summary = run_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    manifest = json.loads((tmp_path / "out" / "xunce-stage23-2-manifest.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert "\\a30\\" in manifest["los_audit_path"] or "/a30/" in manifest["los_audit_path"]


def test_stage23_2_requires_passed_stage23_2b_lineage(tmp_path: Path) -> None:
    from scripts.run_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract import (
        run_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract,
    )

    stage23_2b = _write_stage23_2b_root(tmp_path, status="failed")
    config = _write_config(tmp_path, stage23_2b)

    summary = run_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage23_2_required_inputs"
    assert "stage23_2b_not_passed" in summary["blocking_reason_codes"]


def _write_stage23_2b_root(
    tmp_path: Path,
    *,
    status: str = "passed",
    obstacle_source_hash: str = "slope-source-hash",
    layout: str = "legacy",
) -> Path:
    root = tmp_path / "stage23_2b"
    if layout == "short":
        los_root = root / "a30" / "s1" / "a" / "s23_0"
    else:
        los_root = root / "s23_2a_platform_30" / "s23_1" / "s23_1_a" / "s23_0"
    los_root.mkdir(parents=True)
    (root / "xunce-stage23-2b-summary.json").write_text(
        json.dumps(
            {
                "schema_version": "xunce-stage23-2b-summary/v1",
                "status": status,
                "next_required_change": "implement_stage23_2_slope_obstacle_aware_theta_reward_contract",
                "platform_contract_id": "agilex_scout_mini_piper",
                "platform_contract_hash": "platform-hash",
                "max_traversable_slope_deg": 30.0,
                "default_slope_blocked_cell_count_total": 743,
                "default_slope_material_occlusion_viewpoint_count": 493,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    los_rows = [
        {
            "schema_version": "xunce-stage23-0-obstacle-los-audit-row/v1",
            "scenario_id": "s1",
            "step_index": 0,
            "candidate_index": 0,
            "candidate_cell": [4, 5],
            "candidate_viewpoint": [4, 5, 90],
            "candidate_theta_deg": 90,
            "sensor_fov_deg": 90.0,
            "sensor_range_cells": 3,
            "path_cost": 2.0,
            "clear_visible_cell_count": 12,
            "clear_theta_coverage_hash": "clear-hash",
            "obstacle_aware_visible_cell_count": 7,
            "obstacle_aware_theta_coverage_hash": "slope-los-hash",
            "visibility_removed_by_obstacle_count": 5,
            "obstacle_source_hash": obstacle_source_hash,
            "obstacle_source_kind": "slope_blocked_as_obstacle_proxy",
        },
        {
            "schema_version": "xunce-stage23-0-obstacle-los-audit-row/v1",
            "scenario_id": "s1",
            "step_index": 1,
            "candidate_index": 0,
            "candidate_cell": [4, 6],
            "candidate_viewpoint": [4, 6, 45],
            "candidate_theta_deg": 45,
            "sensor_fov_deg": 90.0,
            "sensor_range_cells": 3,
            "path_cost": 4.0,
            "clear_visible_cell_count": 10,
            "clear_theta_coverage_hash": "clear-hash-2",
            "obstacle_aware_visible_cell_count": 10,
            "obstacle_aware_theta_coverage_hash": "slope-los-hash-2",
            "visibility_removed_by_obstacle_count": 0,
            "obstacle_source_hash": obstacle_source_hash,
            "obstacle_source_kind": "slope_blocked_as_obstacle_proxy",
        },
    ]
    _write_jsonl(los_root / "xunce-stage23-0-obstacle-los-audit.jsonl", los_rows)
    return root


def _write_config(tmp_path: Path, stage23_2b: Path) -> Path:
    payload = {
        "schema_version": "xunce-stage23-2-slope-obstacle-aware-theta-reward-contract-config/v1",
        "stage23_2b_root": str(stage23_2b),
        "coverage_first_reward_profile": "configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json",
        "coverage_denominator_cells": 100.0,
        "max_replay_rows": 100,
        "max_traversable_slope_deg": 30.0,
        "stage23_2_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    path = tmp_path / "stage23_2_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
