import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_10a_generates_variants_runs_offline_and_topk_eval(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_10a_terminal_aware_reward_weight_sweep as s26

    captured_stage26_10_configs: list[dict] = []

    def fake_stage26_10(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        captured_stage26_10_configs.append(
            {
                "payload": payload,
                "config_path": str(config_path),
                "output_root": str(output_root),
                "repo_root": str(repo_root),
            }
        )
        return {
            "status": "passed",
            "next_required_change": "review_stage26_9_long_horizon_efficiency_readiness",
            "mean_main_coverage_per_100m_delta": 0.25,
            "main_coverage_per_100m_delta": 0.25,
            "selected_action_changed_count": 2,
            "binding_or_safety_failure": False,
            "release_or_training_authorized": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }

    monkeypatch.setattr(
        s26.stage26_10,
        "run_xunce_stage26_10_terminal_aware_reward_shaping",
        fake_stage26_10,
    )

    config_path = _write_config(tmp_path, _write_stage21_1_root(tmp_path))
    summary = s26.run_xunce_stage26_10a_terminal_aware_reward_weight_sweep(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    matrix = _read_jsonl(tmp_path / "out" / "xunce-stage26-10a-reward-weight-sweep-matrix.jsonl")
    profiles = sorted((tmp_path / "out" / "profiles").glob("*.json"))

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "review_stage26_10a_terminal_aware_weight_readiness"
    assert summary["profile_count"] == 9
    assert summary["offline_passed_count"] == 9
    assert len(summary["top_k_profile_ids"]) == 3
    assert summary["best_profile_id"] == summary["top_k_profile_ids"][0]
    from model_explorer.policy.coverage_first_reward import load_coverage_first_reward_profile

    assert len(profiles) == 9
    loaded_profiles = [load_coverage_first_reward_profile(path) for path in profiles]
    assert len({profile.profile_hash for profile in loaded_profiles}) == 9
    assert {profile.schema_version for profile in loaded_profiles} == {"xunce-stage21-coverage-first-ppo-reward-profile/v3"}
    assert {profile.profile_version for profile in loaded_profiles} == {"stage26-10-terminal-aware-v3"}
    assert {row["offline_status"] for row in matrix} == {"passed"}
    assert all(row["stage21_2_status"] == "passed" for row in matrix)
    assert all(row["stage21_3_status"] == "passed" for row in matrix)
    assert all(row["hard_risk_positive_reward_count"] == 0 for row in matrix)
    assert all(row["reward_trainable_false_count"] == 0 for row in matrix)
    assert all(row["all_reward_reason_codes_nonblocking"] is True for row in matrix)
    assert all(
        row["reward_reason_code_total_count"] == row["nonblocking_reward_reason_code_count"]
        for row in matrix
    )
    assert all(row["unknown_or_blocking_reward_reason_codes"] == [] for row in matrix)
    assert captured_stage26_10_configs
    assert captured_stage26_10_configs[0]["payload"]["terminal_aware_reward_profile"].startswith(
        str((tmp_path / "out" / "profiles").resolve())
    )
    assert captured_stage26_10_configs[0]["payload"]["terminal_aware_reward_profile"] != (
        "configs/xunce_stage26_10_terminal_aware_ppo_reward_profile_v3.json"
    )
    assert summary["release_or_training_authorized"] is False
    assert summary["publishes_checkpoint"] is False
    assert summary["replaces_default_policy"] is False
    assert summary["connects_real_executor"] is False
    assert summary["starts_online_canary"] is False
    assert summary["canary_traffic_fraction"] == 0.0


def test_stage26_10a_registry_entry_exists() -> None:
    registry = json.loads((REPO_ROOT / "configs" / "stage_registry.json").read_text(encoding="utf-8"))
    stage = registry["stages"]["xunce-stage26-10a-terminal-aware-reward-weight-sweep"]

    assert stage["script"] == "scripts/run_xunce_stage26_10a_terminal_aware_reward_weight_sweep.py"
    assert stage["default_config"] == "configs/xunce_stage26_10a_terminal_aware_reward_weight_sweep_v1.json"
    assert stage["default_output_root"] == "D:/xunce/out/s26_10a"


def test_stage26_10a_continues_same_topk_profile_when_stage26_9_is_pending(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.run_xunce_stage26_10a_terminal_aware_reward_weight_sweep as s26

    captured_profiles: list[str] = []
    responses = [
        {
            "status": "passed",
            "next_required_change": "continue_stage26_9_long_horizon_jobs",
            "binding_or_safety_failure": False,
            "release_or_training_authorized": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
        {
            "status": "passed",
            "next_required_change": "review_stage26_9_long_horizon_efficiency_readiness",
            "main_coverage_per_100m_delta": 0.1,
            "selected_action_changed_count": 1,
            "binding_or_safety_failure": False,
            "release_or_training_authorized": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    ]

    def fake_stage26_10(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        captured_profiles.append(payload["terminal_aware_reward_profile"])
        return dict(responses.pop(0))

    monkeypatch.setattr(
        s26.stage26_10,
        "run_xunce_stage26_10_terminal_aware_reward_shaping",
        fake_stage26_10,
    )

    config_path = _write_config(tmp_path, _write_stage21_1_root(tmp_path))

    first = s26.run_xunce_stage26_10a_terminal_aware_reward_weight_sweep(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )
    second = s26.run_xunce_stage26_10a_terminal_aware_reward_weight_sweep(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert first["next_required_change"] == "continue_stage26_10a_topk_bounded_eval"
    assert second["next_required_change"] == "review_stage26_10a_terminal_aware_weight_readiness"
    assert len(captured_profiles) == 2
    assert captured_profiles[0] == captured_profiles[1]


def test_stage26_10a_maps_downstream_credit_route_to_terminal_credit_assignment() -> None:
    import scripts.run_xunce_stage26_10a_terminal_aware_reward_weight_sweep as s26

    top_k = [
        "xunce-stage26-10a-balanced_high",
        "xunce-stage26-10a-terminal_heavy",
        "xunce-stage26-10a-dead_end_high",
    ]
    rows = [
        {
            "profile_id": top_k[0],
            "offline_status": "passed",
            "stage26_10_status": "failed",
            "stage26_10_next_required_change": "repair_stage26_synthetic_credit_assignment",
            "binding_or_safety_failure": False,
            "main_coverage_per_100m_delta": -2.2294907984308523,
            "selected_action_changed_count": 6,
        },
        {"profile_id": top_k[1], "offline_status": "passed", "stage26_10_status": "pending"},
        {"profile_id": top_k[2], "offline_status": "passed", "stage26_10_status": "pending"},
    ]

    assert s26._route([], [], rows, top_k) == "repair_stage26_terminal_credit_assignment"


def test_stage26_10a_resume_does_not_start_next_topk_after_credit_route(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import scripts.run_xunce_stage26_10a_terminal_aware_reward_weight_sweep as s26

    calls: list[str] = []

    def fake_stage26_10(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        calls.append(payload["terminal_aware_reward_profile"])
        return {
            "status": "failed",
            "next_required_change": "repair_stage26_synthetic_credit_assignment",
            "binding_or_safety_failure": False,
            "release_or_training_authorized": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }

    monkeypatch.setattr(
        s26.stage26_10,
        "run_xunce_stage26_10_terminal_aware_reward_shaping",
        fake_stage26_10,
    )

    config_path = _write_config(tmp_path, _write_stage21_1_root(tmp_path))
    first = s26.run_xunce_stage26_10a_terminal_aware_reward_weight_sweep(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )
    second = s26.run_xunce_stage26_10a_terminal_aware_reward_weight_sweep(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert first["next_required_change"] == "repair_stage26_terminal_credit_assignment"
    assert second["next_required_change"] == "repair_stage26_terminal_credit_assignment"
    assert len(calls) == 1


def _write_config(tmp_path: Path, collector_root: Path) -> Path:
    payload = {
        "schema_version": "xunce-stage26-10a-terminal-aware-reward-weight-sweep-config/v1",
        "stage_id": "xunce-stage26-10a-terminal-aware-reward-weight-sweep",
        "stage21_1_collector_root": str(collector_root),
        "baseline_reward_profile": "configs/xunce_stage26_10_terminal_aware_ppo_reward_profile_v3.json",
        "base_stage21_2_config": "configs/xunce_stage21_2_coverage_first_ppo_reward_contract_v1.json",
        "base_stage21_3_config": "configs/xunce_stage21_3_ppo_batch_validation_v1.json",
        "base_stage26_10_config": "configs/xunce_stage26_10_terminal_aware_reward_shaping_v1.json",
        "top_k": 3,
        "run_topk_stage26_10": True,
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    path = tmp_path / "stage26_10a_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_stage21_1_root(tmp_path: Path) -> Path:
    root = tmp_path / "stage21_1"
    root.mkdir()
    (root / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json").write_text(
        json.dumps(
            {
                "schema_version": "xunce-stage21-1-on-policy-ppo-rollout-collector-summary/v1",
                "status": "passed",
                "next_required_change": "implement_stage21_2_coverage_first_ppo_reward_contract",
                "profile_hash": "collector-canonical-profile-hash",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    transitions = [
        _transition("s1:0", step=0, done=False, final_coverage=0.45, next_action_mask_zero=False),
        _transition("s1:1", step=1, done=True, final_coverage=0.5, next_action_mask_zero=True),
    ]
    _write_jsonl(root / "xunce-stage21-1-ppo-trainable-batch.jsonl", transitions)
    _write_jsonl(
        root / "xunce-stage21-1-ppo-rollout-episodes.jsonl",
        [
            {
                "schema_version": "xunce-stage21-1-ppo-rollout-episode/v1",
                "scenario_id": "s1",
                "rollout_steps": 2,
                "final_coverage_rate": 0.5,
            }
        ],
    )
    return root


def _transition(transition_id: str, *, step: int, done: bool, final_coverage: float, next_action_mask_zero: bool) -> dict:
    info = {
        "coverage_rate_delta": 0.08 if not done else 0.02,
        "final_coverage_rate_after_step": final_coverage,
        "path_cost": 8.0,
        "soft_risk_exposure": 1.0,
        "hard_risk_violation": False,
        "action_mask": [True],
        "sampling_mask": [True],
        "hard_risk_clean_mask": [True],
        "candidate_cells": [[1, step]],
        "old_log_prob_recompute_abs_error": 0.0,
    }
    if next_action_mask_zero:
        info.update(
            {
                "terminal_reason": "no_hybrid_astar_pose_reachable_candidate_for_action_mask",
                "next_action_mask_true_count": 0,
                "next_action_mask_zero": True,
                "next_grid_action_mask_true_count": 1,
                "dead_end_attribution_source": "next_state_action_mask_all_false/v1",
            }
        )
    return {
        "schema_version": "xunce-stage21-1-ppo-transition/v1",
        "transition_id": transition_id,
        "scenario_id": "s1",
        "step_index": step,
        "observation": {"action_mask": [True], "candidate_cells": [[1, step]]},
        "xunce_batch": {"action_mask": {"shape": [1, 1], "dtype": "bool", "values": [[True]]}},
        "action_index": 0,
        "old_log_prob": -0.1,
        "old_value": 0.5,
        "reward": 0.0,
        "next_observation": None if done else {"action_mask": [True]},
        "next_xunce_batch": None if done else {"action_mask": {"shape": [1, 1], "dtype": "bool", "values": [[True]]}},
        "done": done,
        "trainable": True,
        "info": info,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
