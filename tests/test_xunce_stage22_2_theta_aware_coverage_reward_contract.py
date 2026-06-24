import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
MODEL_EXPLORER_SRC = str(REPO_ROOT / "model-explorer" / "src")
if MODEL_EXPLORER_SRC not in sys.path:
    sys.path.insert(0, MODEL_EXPLORER_SRC)


def test_stage22_2_replays_theta_reward_and_routes_stage22_3(tmp_path: Path) -> None:
    import scripts.run_xunce_stage22_2_theta_aware_coverage_reward_contract as s22

    stage22_1 = _write_stage22_1_root(tmp_path)
    config = _write_config(tmp_path, stage22_1)

    summary = s22.run_xunce_stage22_2_theta_aware_coverage_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    replay_rows = _read_jsonl(tmp_path / "out" / "xunce-stage22-2-reward-replay.jsonl")
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage22_3_theta_aware_ppo_collector_smoke"
    assert summary["theta_reward_contract_missing_count"] == 0
    assert summary["point_only_reward_fallback_used_count"] == 0
    assert summary["coverage_diff_group_count"] == 1
    assert summary["reward_discriminated_coverage_diff_group_count"] == 1
    assert summary["stage21_3_rejects_mismatched_theta_reward_batch"] is True
    assert {row["coverage_source"] for row in replay_rows} == {"theta_aware_sensor_footprint/v1"}
    assert replay_rows[0]["theta_coverage_denominator_cells"] == 100.0
    assert replay_rows[0]["reward"] != replay_rows[1]["reward"]
    assert summary["runs_new_ppo_update"] is False
    assert summary["publishes_checkpoint"] is False


def test_stage22_2_fails_when_reward_provenance_missing(tmp_path: Path) -> None:
    import scripts.run_xunce_stage22_2_theta_aware_coverage_reward_contract as s22

    stage22_1 = _write_stage22_1_root(tmp_path, missing_theta_hash=True)
    config = _write_config(tmp_path, stage22_1)

    summary = s22.run_xunce_stage22_2_theta_aware_coverage_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage22_2_theta_reward_provenance"
    assert "theta_reward_contract_missing" in summary["blocking_reason_codes"]


def test_stage22_2_requires_stage22_1_passed(tmp_path: Path) -> None:
    import scripts.run_xunce_stage22_2_theta_aware_coverage_reward_contract as s22

    stage22_1 = _write_stage22_1_root(tmp_path, status="failed")
    config = _write_config(tmp_path, stage22_1)

    summary = s22.run_xunce_stage22_2_theta_aware_coverage_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage22_2_required_inputs"
    assert "stage22_1_not_passed" in summary["blocking_reason_codes"]


def _write_stage22_1_root(tmp_path: Path, *, status: str = "passed", missing_theta_hash: bool = False) -> Path:
    root = tmp_path / "stage22_1"
    root.mkdir()
    summary = {
        "schema_version": "xunce-stage22-1-summary/v1",
        "status": status,
        "next_required_change": "run_stage22_2_theta_aware_coverage_reward_contract",
        "hash_missing_theta_count": 0,
        "mask_length_mismatch_count": 0,
        "stage21_batch_theta_incompatible_after_expansion": False,
    }
    (root / "xunce-stage22-1-summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    rows = [
        _viewpoint_row(theta=0, new_visible=5, theta_hash=None if missing_theta_hash else "hash-0"),
        _viewpoint_row(theta=45, new_visible=6, theta_hash="hash-45"),
    ]
    _write_jsonl(root / "xunce-stage22-1-viewpoint-candidate-audit.jsonl", rows)
    return root


def _viewpoint_row(*, theta: int, new_visible: int, theta_hash: str | None) -> dict:
    return {
        "schema_version": "xunce-stage22-1-viewpoint-candidate-audit-row/v1",
        "scenario_id": "s1",
        "step_index": 0,
        "candidate_cell": [2, 3],
        "candidate_theta_deg": theta,
        "candidate_viewpoint": [2, 3, theta],
        "base_candidate_index": 0,
        "viewpoint_index": theta // 45,
        "candidate_set_hash": "theta-set",
        "sensor_model_id": "theta-fov-90-range-radius/v1",
        "sensor_fov_deg": 90.0,
        "sensor_range_cells": 2,
        "theta_visible_cell_count": new_visible,
        "theta_new_visible_cell_count": new_visible,
        "theta_coverage_hash": theta_hash,
        "theta_coverage_gain_per_path_cost": float(new_visible) / 2.0,
        "action_mask_valid": True,
    }


def _write_config(tmp_path: Path, stage22_1: Path) -> Path:
    config = {
        "schema_version": "xunce-stage22-2-theta-aware-coverage-reward-contract-config/v1",
        "stage22_1_root": str(stage22_1),
        "coverage_first_reward_profile": "configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json",
        "coverage_denominator_cells": 100.0,
        "stage22_2_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    path = tmp_path / "stage22_2_config.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
