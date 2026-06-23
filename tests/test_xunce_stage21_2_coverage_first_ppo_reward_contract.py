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


def test_stage21_2_evaluates_stage21_1_batch_and_routes_to_stage21_3(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract import (
        run_xunce_stage21_2_coverage_first_ppo_reward_contract,
    )

    collector_root = _write_stage21_1_root(tmp_path, final_coverage=0.995, rollout_steps=4)
    config = _write_config(tmp_path, collector_root)

    summary = run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "implement_stage21_3_ppo_batch_validation"
    assert summary["reward_evaluation_row_count"] == 1
    assert summary["hard_risk_positive_reward_count"] == 0
    assert summary["stage21_2_authorized"] is False
    assert (tmp_path / "out" / "xunce-stage21-2-reward-contract-evaluation.jsonl").is_file()


def test_stage21_2_routes_40_step_below_99pct_to_horizon_scaling(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract import (
        run_xunce_stage21_2_coverage_first_ppo_reward_contract,
    )

    collector_root = _write_stage21_1_root(tmp_path, final_coverage=0.55, rollout_steps=40)
    config = _write_config(tmp_path, collector_root)

    summary = run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "stage21_12_rollout_horizon_or_mission_budget_scaling_for_99pct_coverage"


def test_stage21_2_boundary_flag_hard_fails(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract import (
        run_xunce_stage21_2_coverage_first_ppo_reward_contract,
    )

    collector_root = _write_stage21_1_root(tmp_path)
    config = _write_config(tmp_path, collector_root, runs_new_ppo_update=True)

    summary = run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage21_2_coverage_first_reward_boundary_rejections"
    assert "runs_new_ppo_update" in summary["blocking_reason_codes"]


def test_stage21_2_requires_stage21_1_passed(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract import (
        run_xunce_stage21_2_coverage_first_ppo_reward_contract,
    )

    collector_root = _write_stage21_1_root(tmp_path, status="failed")
    config = _write_config(tmp_path, collector_root)

    summary = run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage21_2_required_inputs"
    assert "stage21_1_collector_not_passed" in summary["blocking_reason_codes"]


def _write_stage21_1_root(
    tmp_path: Path,
    *,
    final_coverage: float = 0.995,
    rollout_steps: int = 4,
    status: str = "passed",
) -> Path:
    root = tmp_path / "stage21_1"
    root.mkdir()
    (root / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json").write_text(
        json.dumps(
            {
                "schema_version": "xunce-stage21-1-on-policy-ppo-rollout-collector-summary/v1",
                "status": status,
                "next_required_change": "implement_stage21_2_coverage_first_ppo_reward_contract",
                "profile_hash": "interim-v3",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    transition = {
        "schema_version": "xunce-stage21-1-ppo-transition/v1",
        "transition_id": "s1:0",
        "scenario_id": "s1",
        "step_index": 0,
        "done": True,
        "trainable": True,
        "reward": 0.1,
        "info": {
            "coverage_rate_delta": 0.1,
            "final_coverage_rate_after_step": final_coverage,
            "path_cost": 10.0,
            "soft_risk_exposure": 1.0,
            "hard_risk_violation": False,
        },
    }
    (root / "xunce-stage21-1-ppo-trainable-batch.jsonl").write_text(
        json.dumps(transition, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    episode = {
        "schema_version": "xunce-stage21-1-ppo-rollout-episode/v1",
        "scenario_id": "s1",
        "rollout_steps": rollout_steps,
        "final_coverage_rate": final_coverage,
    }
    (root / "xunce-stage21-1-ppo-rollout-episodes.jsonl").write_text(
        json.dumps(episode, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return root


def _write_config(tmp_path: Path, collector_root: Path, **overrides) -> Path:
    payload = {
        "schema_version": "xunce-stage21-2-coverage-first-ppo-reward-contract-config/v1",
        "stage21_1_collector_root": str(collector_root),
        "coverage_first_reward_profile": "configs/xunce_stage21_coverage_first_ppo_reward_profile_v1.json",
        "stage21_2_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage21_2_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
