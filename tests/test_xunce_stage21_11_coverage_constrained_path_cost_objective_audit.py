from __future__ import annotations

import json
from pathlib import Path

from scripts.run_xunce_stage21_11_coverage_constrained_path_cost_objective_audit import (
    ROUTE_INPUTS,
    ROUTE_ADVANTAGE,
    ROUTE_BINDING,
    ROUTE_BOUNDARY,
    ROUTE_REWARD,
    ROUTE_SIGNAL,
    ROUTE_STAGE21_12,
    run_xunce_stage21_11_coverage_constrained_path_cost_objective_audit,
)


def test_stage21_11_routes_to_reward_when_reward_best_differs_from_coverage_per_cost(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path)

    summary = run_xunce_stage21_11_coverage_constrained_path_cost_objective_audit(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_REWARD
    audit = json.loads(Path(summary["reward_separation_audit"]).read_text(encoding="utf-8"))
    assert "reward_proxy_best_candidate_not_aligned_with_coverage_per_cost" in audit["reason_codes"]
    assert summary["runs_new_ppo_update"] is False
    assert summary["publishes_checkpoint"] is False


def test_stage21_11_routes_to_advantage_when_advantage_not_aligned(tmp_path: Path) -> None:
    root, config = _fixture(
        tmp_path,
        reward_min_match=0.0,
        selected_min_match=0.0,
        advantages=[2.0, 1.0],
        probability_delta=0.001,
        aggregate_delta=0.01,
    )

    summary = run_xunce_stage21_11_coverage_constrained_path_cost_objective_audit(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["next_required_change"] == ROUTE_ADVANTAGE
    audit = json.loads(Path(summary["advantage_separation_audit"]).read_text(encoding="utf-8"))
    assert "advantage_not_positive_for_coverage_per_cost" in audit["reason_codes"]


def test_stage21_11_routes_to_signal_when_probability_shift_too_small(tmp_path: Path) -> None:
    root, config = _fixture(
        tmp_path,
        reward_min_match=0.0,
        selected_min_match=0.0,
        min_adv_corr=-1.0,
        advantages=[1.0, 2.0],
        probability_delta=0.0,
        aggregate_delta=0.01,
    )

    summary = run_xunce_stage21_11_coverage_constrained_path_cost_objective_audit(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["next_required_change"] == ROUTE_SIGNAL
    audit = json.loads(Path(summary["action_probability_shift_audit"]).read_text(encoding="utf-8"))
    assert "post_update_action_probabilities_nearly_unchanged" in audit["reason_codes"]


def test_stage21_11_routes_to_binding_when_probabilities_change_but_trajectory_delta_zero(tmp_path: Path) -> None:
    root, config = _fixture(
        tmp_path,
        reward_min_match=0.0,
        selected_min_match=0.0,
        min_adv_corr=-1.0,
        advantages=[1.0, 2.0],
        probability_delta=0.001,
        aggregate_delta=0.0,
    )

    summary = run_xunce_stage21_11_coverage_constrained_path_cost_objective_audit(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["next_required_change"] == ROUTE_BINDING


def test_stage21_11_passes_to_stage21_12_when_audits_align(tmp_path: Path) -> None:
    root, config = _fixture(
        tmp_path,
        reward_min_match=0.0,
        selected_min_match=0.0,
        min_adv_corr=-1.0,
        advantages=[1.0, 2.0],
        probability_delta=0.001,
        aggregate_delta=0.01,
    )

    summary = run_xunce_stage21_11_coverage_constrained_path_cost_objective_audit(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == ROUTE_STAGE21_12
    recommended = json.loads(Path(summary["recommended_stage21_6_config"]).read_text(encoding="utf-8"))
    assert recommended["objective_mode"] == "coverage_constrained_path_cost"
    assert recommended["publishes_checkpoint"] is False


def test_stage21_11_boundary_flags_hard_fail(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path, starts_online_canary=True)

    summary = run_xunce_stage21_11_coverage_constrained_path_cost_objective_audit(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_BOUNDARY
    assert "config_starts_online_canary_true" in summary["boundary_reason_codes"]


def test_stage21_11_missing_seed_artifact_routes_to_inputs(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path)
    missing = root / "stage21_10" / "s6" / "seed_2101" / "stage21_3" / "xunce-stage21-3-return-advantage-audit.jsonl"
    missing.unlink()

    summary = run_xunce_stage21_11_coverage_constrained_path_cost_objective_audit(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_INPUTS
    assert "seed_2101_missing_stage21_3_return_advantage_audit" in summary["input_reason_codes"]


def test_stage21_11_weak_probability_binding_cannot_pass(tmp_path: Path) -> None:
    root, config = _fixture(
        tmp_path,
        reward_min_match=0.0,
        selected_min_match=0.0,
        min_adv_corr=-1.0,
        advantages=[1.0, 2.0],
        probability_delta=0.2,
        aggregate_delta=0.01,
    )
    for subdir in ("pre_ppo_xunce", "post_ppo_xunce"):
        path = root / "stage21_10" / "s6" / "seed_2101" / "stage21_5" / subdir / "xunce-exploration-coverage-model-inference.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            row.pop("covered_cells_hash", None)
            row.pop("current_cell", None)
        _jsonl(path, rows)

    summary = run_xunce_stage21_11_coverage_constrained_path_cost_objective_audit(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["next_required_change"] == ROUTE_SIGNAL
    audit = json.loads(Path(summary["action_probability_shift_audit"]).read_text(encoding="utf-8"))
    assert audit["strong_state_join_available_count"] == 0
    assert "strong_state_probability_binding_unavailable" in audit["reason_codes"]


def _fixture(
    tmp_path: Path,
    *,
    reward_min_match: float = 0.9,
    selected_min_match: float = 0.9,
    min_adv_corr: float = 0.0,
    advantages: list[float] | None = None,
    probability_delta: float = 0.0,
    aggregate_delta: float = 0.0,
    starts_online_canary: bool = False,
) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    stage21_10 = root / "stage21_10"
    stage21_6 = stage21_10 / "s6"
    seed = stage21_6 / "seed_2101"
    for directory in (
        seed / "stage21_3",
        seed / "stage21_4",
        seed / "stage21_5",
        seed / "stage21_5" / "pre_ppo_xunce",
        seed / "stage21_5" / "post_ppo_xunce",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    advantages = advantages or [1.0, 2.0]
    _json(stage21_10 / "xunce-stage21-10-summary.json", _stage21_10_summary(stage21_6))
    _json(stage21_10 / "xunce-stage21-10-repaired-stage21-6-config.json", {"schema_version": "xunce-stage21-6-multi-seed-ppo-pilot-config/v1"})
    _json(stage21_6 / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json", {"status": "failed"})
    _json(stage21_6 / "xunce-stage21-6-aggregate-metrics.json", _aggregate(aggregate_delta))
    _json(stage21_6 / "xunce-stage21-6-lineage-audit.json", {"passed": True})
    _jsonl(
        stage21_6 / "xunce-stage21-6-seed-results.jsonl",
        [
            {
                "schema_version": "xunce-stage21-6-seed-result/v1",
                "seed": 2101,
                "stage21_3_root": str(seed / "stage21_3"),
                "stage21_4_root": str(seed / "stage21_4"),
                "stage21_5_root": str(seed / "stage21_5"),
            }
        ],
    )
    _jsonl(seed / "stage21_3" / "xunce-stage21-3-ppo-trainable-batch.jsonl", _batch_rows(advantages))
    _jsonl(seed / "stage21_3" / "xunce-stage21-3-return-advantage-audit.jsonl", [{"advantage": value} for value in advantages])
    _jsonl(seed / "stage21_4" / "xunce-stage21-4-ppo-loss-audit.jsonl", [{"policy_loss": 0.1}])
    _json(seed / "stage21_4" / "xunce-stage21-4-gradient-audit.json", {"policy_loss_grad_norm": 0.1})
    _json(
        seed / "stage21_5" / "xunce-stage21-5-post-update-evaluation-summary.json",
        {
            "pre_evaluation_root": str(seed / "stage21_5" / "pre_ppo_xunce"),
            "post_evaluation_root": str(seed / "stage21_5" / "post_ppo_xunce"),
        },
    )
    _jsonl(seed / "stage21_5" / "pre_ppo_xunce" / "xunce-exploration-coverage-model-inference.jsonl", _inference_rows(0.0))
    _jsonl(seed / "stage21_5" / "post_ppo_xunce" / "xunce-exploration-coverage-model-inference.jsonl", _inference_rows(probability_delta))
    profile = root / "configs" / "xunce_stage21_coverage_first_ppo_reward_profile_v1.json"
    profile.parent.mkdir(parents=True, exist_ok=True)
    _json(profile, {"schema_version": "xunce-stage21-coverage-first-ppo-reward-profile/v1"})
    config = root / "config.json"
    _json(
        config,
        {
            "schema_version": "xunce-stage21-11-coverage-constrained-path-cost-objective-audit-config/v1",
            "stage21_10_root": str(stage21_10),
            "coverage_first_reward_profile": str(profile),
            "min_reward_best_matches_coverage_per_cost_rate": reward_min_match,
            "min_selected_best_coverage_per_cost_rate": selected_min_match,
            "min_advantage_coverage_per_cost_correlation": min_adv_corr,
            "starts_online_canary": starts_online_canary,
        },
    )
    return root, config


def _stage21_10_summary(stage21_6: Path) -> dict[str, object]:
    return {
        "status": "failed",
        "next_required_change": "repair_stage21_reward_signal_or_advantage_separation",
        "stage21_6_root": str(stage21_6),
        "repaired_stage21_6_config": str(stage21_6.parent / "xunce-stage21-10-repaired-stage21-6-config.json"),
    }


def _aggregate(delta: float) -> dict[str, object]:
    return {
        "trainable_transition_count_total": 2,
        "final_coverage_delta_mean": delta,
        "coverage_auc_delta_mean": delta,
        "pre_clip_grad_norm_max": 2.0,
        "hard_risk_violation_total": 0,
        "model_inference_mask_violation_total": 0,
        "model_inference_failure_total": 0,
        "path_planning_failure_total": 0,
        "open_grid_fallback_total": 0,
        "safety_boundary_violation_total": 0,
        "execution_boundary_violation_total": 0,
    }


def _batch_rows(advantages: list[float]) -> list[dict[str, object]]:
    return [
        _batch_row(0, advantages[0], "hash-a", [0.10, 0.08], [0.10, 0.01]),
        _batch_row(1, advantages[1], "hash-b", [0.08, 0.10], [0.10, 0.01]),
    ]


def _batch_row(action_index: int, advantage: float, hash_value: str, coverage: list[float], path_cost: list[float]) -> dict[str, object]:
    features = [
        [0, 0, 0, 0, 0, 0, 1, coverage[0], 0, 0, 0, 0, 0, path_cost[0], 0],
        [0, 0, 0, 0, 0, 0, 1, coverage[1], 0, 0, 0, 0, 0, path_cost[1], 0],
    ]
    return {
        "transition_id": f"scenario:step-{action_index}",
        "scenario_id": "scenario",
        "step_index": action_index,
        "action_index": action_index,
        "advantage": advantage,
        "raw_advantage": advantage,
        "reward": 1.0,
        "info": {
            "scenario_id": "scenario",
            "step_index": action_index,
            "candidate_set_hash": hash_value,
            "covered_cells_hash": f"covered-{action_index}",
            "current_cell_before": [action_index, 0],
            "action_mask": [True, True],
            "candidate_cells": [[0, 0], [1, 1]],
        },
        "observation": {
            "action_mask": [True, True],
            "candidate_cells": [[0, 0], [1, 1]],
            "candidate_feature_names": [
                "cell_x",
                "cell_y",
                "relative_dx",
                "relative_dy",
                "relative_distance",
                "utility",
                "reachable",
                "expected_coverage_rate_delta",
                "expected_new_coverage_area",
                "information_gain",
                "confidence_gain",
                "value",
                "risk",
                "path_cost",
                "energy_cost",
            ],
            "candidate_features": features,
        },
        "reward_components": {},
    }


def _inference_rows(delta: float) -> list[dict[str, object]]:
    return [
        _inference_row(0, "hash-a", "covered-0", [0, 0], [0.60 - delta, 0.40 + delta]),
        _inference_row(1, "hash-b", "covered-1", [1, 0], [0.40 - delta, 0.60 + delta]),
    ]


def _inference_row(step: int, hash_value: str, covered_hash: str, current_cell: list[int], probs: list[float]) -> dict[str, object]:
    selected = 0 if probs[0] >= probs[1] else 1
    return {
        "schema_version": "xunce-exploration-coverage-model-inference/v1",
        "scenario_id": "scenario",
        "step_index": step,
        "candidate_set_hash": hash_value,
        "covered_cells_hash": covered_hash,
        "current_cell": current_cell,
        "detail": {
            "action_probs": probs,
            "selected_action_index": selected,
            "selected_rank": 1,
        },
        "selected_action_index": selected,
    }


def _json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
