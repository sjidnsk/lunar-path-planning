import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_10_generates_terminal_aware_nested_configs_and_runs_stage26_9(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import scripts.run_xunce_stage26_10_terminal_aware_reward_shaping as s26

    captured: dict = {}

    def fake_stage26_9(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        captured["config"] = payload
        captured["output_root"] = str(output_root)
        captured["repo_root"] = str(repo_root)
        return {
            "status": "passed",
            "next_required_change": "continue_stage26_9_long_horizon_jobs",
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }

    monkeypatch.setattr(s26.stage26_9, "run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot", fake_stage26_9)

    summary = s26.run_xunce_stage26_10_terminal_aware_reward_shaping(
        config_path=_write_config(tmp_path),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )
    derived_stage26_1 = Path(captured["config"]["base_stage26_1_config"])
    derived_stage26_1_payload = json.loads(derived_stage26_1.read_text(encoding="utf-8"))

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "continue_stage26_9_long_horizon_jobs"
    assert summary["terminal_aware_reward_profile"] == "configs/xunce_stage26_10_terminal_aware_ppo_reward_profile_v3.json"
    assert derived_stage26_1_payload["coverage_first_reward_profile"] == (
        "configs/xunce_stage26_10_terminal_aware_ppo_reward_profile_v3.json"
    )
    assert captured["config"]["coverage_source"] == "endpoint_theta_slope_obstacle_los/v1"
    assert captured["config"]["path_cost_source"] == "hybrid_astar_pose_path/v1"
    assert captured["config"]["action_space_type"] == "hybrid_discrete_xy_continuous_theta/v1"
    assert captured["config"]["candidate_reachability_gate_source"] == "hybrid_astar_pose_reachability/v1"
    for field in (
        "release_or_training_authorized",
        "publishes_checkpoint",
        "replaces_default_policy",
        "connects_real_executor",
        "starts_online_canary",
    ):
        assert summary[field] is False
        assert captured["config"][field] is False
    assert summary["canary_traffic_fraction"] == 0.0
    assert captured["config"]["canary_traffic_fraction"] == 0.0


def test_stage26_10_registry_entry_exists() -> None:
    registry = json.loads((REPO_ROOT / "configs" / "stage_registry.json").read_text(encoding="utf-8"))
    stage = registry["stages"]["xunce-stage26-10-terminal-aware-reward-shaping"]

    assert stage["script"] == "scripts/run_xunce_stage26_10_terminal_aware_reward_shaping.py"
    assert stage["default_config"] == "configs/xunce_stage26_10_terminal_aware_reward_shaping_v1.json"
    assert stage["default_output_root"] == "D:/xunce/out/s26_10"


def test_stage26_10_missing_config_fails_fast(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_10_terminal_aware_reward_shaping as s26

    with pytest.raises(FileNotFoundError):
        s26.run_xunce_stage26_10_terminal_aware_reward_shaping(
            config_path=tmp_path / "missing-stage26-10.json",
            output_root=tmp_path / "out",
            repo_root=REPO_ROOT,
        )


def _write_config(tmp_path: Path) -> Path:
    payload = {
        "schema_version": "xunce-stage26-10-terminal-aware-reward-shaping-config/v1",
        "stage_id": "xunce-stage26-10-terminal-aware-reward-shaping",
        "base_stage26_1_config": "configs/xunce_stage26_1_synthetic_terrain_collector_smoke_v1.json",
        "base_stage26_9_config": "configs/xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot_v1.json",
        "terminal_aware_reward_profile": "configs/xunce_stage26_10_terminal_aware_ppo_reward_profile_v3.json",
        "run_stage26_9": True,
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    path = tmp_path / "stage26_10_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
