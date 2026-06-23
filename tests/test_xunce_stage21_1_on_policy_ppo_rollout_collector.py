import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
MODEL_EXPLORER_SRC = str(REPO_ROOT / "model-explorer" / "src")
if MODEL_EXPLORER_SRC not in sys.path:
    sys.path.insert(0, MODEL_EXPLORER_SRC)


def test_stage21_1_writes_passed_collector_artifacts(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    config = _write_config(tmp_path)
    transition = _transition()

    def fake_collect(**_kwargs):
        return runner.CollectionResult(
            episodes=[
                {
                    "schema_version": "xunce-stage21-1-ppo-rollout-episode/v1",
                    "scenario_id": "s1",
                    "trainable_transition_count": 1,
                    "hard_risk_violation_count": 0,
                    "reason_codes": [],
                }
            ],
            transitions=[transition],
            trainable_batch=[transition],
            rejections=[],
            reward_audit=[],
            sampling_audit=[],
            model_audit={"xunce_checkpoint_audit": {"checkpoint_loaded": True}},
            reason_codes=[],
        )

    monkeypatch.setattr(runner, "_collect_rollouts", fake_collect)

    summary = runner.run_xunce_stage21_1_on_policy_ppo_rollout_collector(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "implement_stage21_2_coverage_first_ppo_reward_contract"
    assert summary["trainable_transition_count"] == 1
    assert summary["mask_violation_count"] == 0
    assert summary["hard_risk_violation_count"] == 0
    assert summary["runs_new_ppo_update"] is False
    assert summary["publishes_checkpoint"] is False
    assert (tmp_path / "out" / "xunce-stage21-1-ppo-trainable-batch.jsonl").is_file()
    assert (tmp_path / "out" / "xunce-stage21-1-report.md").is_file()


def test_stage21_1_boundary_flag_hard_fails(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    config = _write_config(tmp_path, publishes_checkpoint=True)
    monkeypatch.setattr(
        runner,
        "_collect_rollouts",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("collector should not run")),
    )

    summary = runner.run_xunce_stage21_1_on_policy_ppo_rollout_collector(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage21_1_on_policy_collector_boundary_rejections"
    assert "publishes_checkpoint" in summary["blocking_reason_codes"]


def test_stage21_1_requires_stage21_0_passed(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    config = _write_config(tmp_path, stage21_0_status="failed")
    monkeypatch.setattr(
        runner,
        "_collect_rollouts",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("collector should not run")),
    )

    summary = runner.run_xunce_stage21_1_on_policy_ppo_rollout_collector(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_1_on_policy_collector_inputs"
    assert "stage21_0_readiness_not_passed" in summary["blocking_reason_codes"]


def test_hard_risk_clean_sampling_mask_excludes_hard_risk_candidates() -> None:
    from scripts.run_xunce_stage21_1_on_policy_ppo_rollout_collector import (
        _hard_risk_clean_mask,
        _hard_risk_clean_sampling_mask,
    )

    candidates = [
        {"reachable": True, "path_cost": 1.0, "path_allowed_by_risk": True},
        {"reachable": True, "path_cost": 1.0, "path_allowed_by_risk": False},
        {"reachable": True, "path_cost": 1.0, "hard_risk_flags": ["slope_platform_limit_violation"]},
        {"reachable": True, "path_cost": 1.0, "open_grid_fallback_used": True},
    ]

    assert _hard_risk_clean_mask(candidates, allow_open_grid_fallback=False) == (
        True,
        False,
        False,
        False,
    )
    assert _hard_risk_clean_sampling_mask(candidates, (True, True, True, True), allow_open_grid_fallback=False) == (
        True,
        False,
        False,
        False,
    )
    assert _hard_risk_clean_sampling_mask(candidates, (False, True, True, True), allow_open_grid_fallback=False) == (
        False,
        False,
        False,
        False,
    )


def test_log_prob_recompute_matches_categorical_distribution() -> None:
    from scripts.run_xunce_stage21_1_on_policy_ppo_rollout_collector import _recompute_log_prob

    logits = [0.1, 0.2, -1.0]
    expected = float(torch.distributions.Categorical(logits=torch.tensor(logits)).log_prob(torch.tensor(1)).item())

    assert abs(_recompute_log_prob(logits, 1) - expected) < 1.0e-8


def _write_config(tmp_path: Path, **overrides) -> Path:
    from model_explorer.policy.canonical_reward import load_canonical_reward_profile

    profile = load_canonical_reward_profile(REPO_ROOT / "configs" / "xunce_canonical_reward_guard_profile_v3.json")
    stage21_0 = tmp_path / "stage21_0"
    stage21_0.mkdir()
    (stage21_0 / "xunce-stage21-0-pure-ppo-readiness-summary.json").write_text(
        json.dumps(
            {
                "schema_version": "xunce-stage21-0-pure-ppo-readiness-summary/v1",
                "status": overrides.pop("stage21_0_status", "passed"),
                "next_required_change": "implement_stage21_1_xunce_on_policy_ppo_rollout_collector",
                "profile_hash": profile.profile_hash,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    payload = {
        "schema_version": "xunce-stage21-1-on-policy-ppo-rollout-collector-config/v1",
        "stage21_0_readiness_root": str(stage21_0),
        "high_fidelity_config": "configs/xunce_high_fidelity_exploration_coverage_comparison_stage18_9_strict_v3.json",
        "canonical_reward_profile": "configs/xunce_canonical_reward_guard_profile_v3.json",
        "dynamic_validation_work_root": str(tmp_path / "dynamic_validation"),
        "required_scenario_count": 1,
        "rollout_steps": 1,
        "dynamic_max_candidates_per_step": 2,
        "dynamic_proposal_pool_limit_per_step": 8,
        "sampling_seed": 1,
        "sampling_temperature": 1.0,
        "min_trainable_transition_count": 1,
        "max_log_prob_recompute_abs_error": 1.0e-6,
        "stage21_1_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    config = tmp_path / "stage21_1_config.json"
    config.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def _transition() -> dict:
    return {
        "schema_version": "xunce-stage21-1-ppo-transition/v1",
        "transition_id": "s1:step-0:sample-0",
        "scenario_id": "s1",
        "step_index": 0,
        "observation": {"action_mask": [True]},
        "xunce_batch": {"action_mask": {"shape": [1, 1], "dtype": "bool", "values": [[True]]}},
        "action_index": 0,
        "old_log_prob": 0.0,
        "old_value": 0.25,
        "reward": 0.1,
        "next_observation": None,
        "next_xunce_batch": None,
        "done": True,
        "trainable": True,
        "info": {
            "action_mask": [True],
            "sampling_mask": [True],
            "hard_risk_clean_mask": [True],
            "argmax_action_index": 0,
            "old_log_prob_recompute_abs_error": 0.0,
            "hard_risk_violation": False,
        },
    }
