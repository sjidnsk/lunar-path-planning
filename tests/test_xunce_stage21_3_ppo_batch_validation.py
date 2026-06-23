import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage21_3_validates_batch_and_routes_to_stage21_4(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path)
    config = _write_config(tmp_path, stage21_1, stage21_2)

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "implement_stage21_4_tiny_ppo_update_smoke"
    assert summary["trainable_transition_count"] == 2
    assert summary["return_finite_count"] == 2
    assert summary["advantage_finite_count"] == 2
    assert summary["invalid_action_count"] == 0
    assert summary["hard_risk_violation_count"] == 0
    assert summary["duplicate_transition_id_count"] == 0
    assert summary["duplicate_reward_transition_id_count"] == 0
    assert summary["reward_trainable_false_count"] == 0
    assert summary["runs_new_ppo_update"] is False
    batch_path = tmp_path / "out" / "xunce-stage21-3-ppo-trainable-batch.jsonl"
    rows = _read_jsonl(batch_path)
    assert {row["stage21_3_split"] for row in rows} <= {"train", "validation"}
    assert all("advantage_normalization_scope" in row for row in rows)


def test_stage21_3_boundary_flag_hard_fails(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path)
    config = _write_config(tmp_path, stage21_1, stage21_2, starts_online_canary=True)

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage21_3_ppo_batch_validation_boundary_rejections"
    assert "starts_online_canary" in summary["blocking_reason_codes"]


def test_stage21_3_detects_reward_lineage_mismatch(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path, omit_second_reward=True)
    config = _write_config(tmp_path, stage21_1, stage21_2)

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_3_batch_lineage"
    assert "missing_reward_transition_rows" in summary["blocking_reason_codes"]


def test_stage21_3_detects_duplicate_transition_ids(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path, duplicate_transition=True)
    config = _write_config(tmp_path, stage21_1, stage21_2)

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_3_duplicate_lineage_ids"
    assert "duplicate_transition_ids" in summary["blocking_reason_codes"]


def test_stage21_3_detects_episode_without_terminal(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path, no_terminal=True)
    config = _write_config(tmp_path, stage21_1, stage21_2)

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_3_episode_boundary_contract"
    assert "episode_terminal_count_mismatch" in summary["blocking_reason_codes"]


def test_stage21_3_detects_nontrainable_reward(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path, reward_trainable=False)
    config = _write_config(tmp_path, stage21_1, stage21_2)

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_3_reward_trainability_contract"
    assert "reward_trainable_false" in summary["blocking_reason_codes"]


def test_stage21_3_detects_invalid_action_mask_and_negative_action(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path, negative_action=True)
    config = _write_config(tmp_path, stage21_1, stage21_2)

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_3_ppo_batch_contract"
    assert "invalid_action_detected" in summary["blocking_reason_codes"]


def test_stage21_3_gates_old_log_prob_recompute_error(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_3_ppo_batch_validation import run_xunce_stage21_3_ppo_batch_validation

    stage21_1, stage21_2 = _write_roots(tmp_path, old_log_error=0.5)
    config = _write_config(tmp_path, stage21_1, stage21_2, max_old_log_prob_recompute_abs_error=1.0e-6)

    summary = run_xunce_stage21_3_ppo_batch_validation(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_3_ppo_batch_contract"
    assert "old_log_prob_recompute_error_exceeds_threshold" in summary["blocking_reason_codes"]


def _write_roots(
    tmp_path: Path,
    *,
    omit_second_reward: bool = False,
    duplicate_transition: bool = False,
    no_terminal: bool = False,
    reward_trainable: bool = True,
    negative_action: bool = False,
    old_log_error: float = 0.0,
) -> tuple[Path, Path]:
    stage21_1 = tmp_path / "stage21_1"
    stage21_2 = tmp_path / "stage21_2"
    stage21_1.mkdir()
    stage21_2.mkdir()
    (stage21_1 / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json").write_text(
        json.dumps(
            {
                "schema_version": "xunce-stage21-1-on-policy-ppo-rollout-collector-summary/v1",
                "status": "passed",
                "next_required_change": "implement_stage21_2_coverage_first_ppo_reward_contract",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (stage21_2 / "xunce-stage21-2-coverage-first-reward-summary.json").write_text(
        json.dumps(
            {
                "schema_version": "xunce-stage21-2-coverage-first-reward-summary/v1",
                "status": "passed",
                "next_required_change": "implement_stage21_3_ppo_batch_validation",
                "profile_hash": "reward-hash",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    transitions = [
        _transition("s1:0", step=0, done=False, reward=1.0, next_observation={"action_mask": [True]}, old_log_error=old_log_error),
        _transition(
            "s1:1" if not duplicate_transition else "s1:0",
            step=1,
            done=not no_terminal,
            reward=2.0,
            next_observation=None if not no_terminal else {"action_mask": [True]},
            negative_action=negative_action,
        ),
    ]
    _write_jsonl(stage21_1 / "xunce-stage21-1-ppo-trainable-batch.jsonl", transitions)
    rewards = [_reward("s1:0", step=0, done=False, reward=1.0, trainable=reward_trainable)]
    if not omit_second_reward:
        rewards.append(_reward("s1:1", step=1, done=not no_terminal, reward=2.0, trainable=reward_trainable))
    _write_jsonl(stage21_2 / "xunce-stage21-2-reward-contract-evaluation.jsonl", rewards)
    return stage21_1, stage21_2


def _transition(
    transition_id: str,
    *,
    step: int,
    done: bool,
    reward: float,
    next_observation,
    negative_action: bool = False,
    old_log_error: float = 0.0,
) -> dict:
    action_index = -1 if negative_action else 0
    return {
        "schema_version": "xunce-stage21-1-ppo-transition/v1",
        "transition_id": transition_id,
        "scenario_id": "s1",
        "step_index": step,
        "observation": {"action_mask": [True]},
        "xunce_batch": {"action_mask": {"shape": [1, 1], "dtype": "bool", "values": [[True]]}},
        "action_index": action_index,
        "old_log_prob": -0.1,
        "old_value": 0.5,
        "reward": reward,
        "next_observation": next_observation,
        "next_xunce_batch": None if done else {"action_mask": {"shape": [1, 1], "dtype": "bool", "values": [[True]]}},
        "done": done,
        "trainable": True,
        "info": {
            "action_mask": [True],
            "sampling_mask": [True],
            "hard_risk_clean_mask": [True],
            "hard_risk_violation": False,
            "old_log_prob_recompute_abs_error": old_log_error,
        },
    }


def _reward(transition_id: str, *, step: int, done: bool, reward: float, trainable: bool) -> dict:
    return {
        "schema_version": "xunce-stage21-2-reward-contract-evaluation/v1",
        "transition_id": transition_id,
        "scenario_id": "s1",
        "step_index": step,
        "done": done,
        "reward": reward,
        "components": {},
        "trainable": trainable,
        "reason_codes": [] if trainable else ["hard_risk_rejected"],
        "profile_id": "profile",
        "profile_version": "stage21-coverage-first-v1",
        "profile_hash": "reward-hash",
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_config(tmp_path: Path, stage21_1: Path, stage21_2: Path, **overrides) -> Path:
    payload = {
        "schema_version": "xunce-stage21-3-ppo-batch-validation-config/v1",
        "stage21_1_collector_root": str(stage21_1),
        "stage21_2_reward_contract_root": str(stage21_2),
        "discount_factor": 0.99,
        "normalize_advantages": True,
        "validation_fraction": 0.25,
        "min_trainable_transition_count": 2,
        "max_old_log_prob_recompute_abs_error": 1.0e-6,
        "require_validation_split": False,
        "stage21_3_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage21_3_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
