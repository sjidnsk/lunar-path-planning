from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_EXPLORER_SRC = str(REPO_ROOT / "model-explorer" / "src")
if MODEL_EXPLORER_SRC not in sys.path:
    sys.path.insert(0, MODEL_EXPLORER_SRC)

from model_explorer.policy.coverage_first_reward import compute_coverage_first_reward_components, load_coverage_first_reward_profile

from scripts.run_xunce_stage21_12_coverage_constrained_reward_profile_repair import (
    ROUTE_ADVANTAGE,
    ROUTE_BOUNDARY,
    ROUTE_CONTINUE,
    ROUTE_INPUTS,
    ROUTE_STAGE21_13,
    run_xunce_stage21_12_coverage_constrained_reward_profile_repair,
)


def test_stage21_12_passes_when_v2_reward_aligns_with_coverage_per_cost(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path)

    summary = run_xunce_stage21_12_coverage_constrained_reward_profile_repair(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == ROUTE_STAGE21_13
    assert summary["v2_reward_best_matches_coverage_per_cost_rate"] > summary["v1_reward_best_matches_coverage_per_cost_rate"]
    recommended = json.loads(Path(summary["recommended_stage21_6_config"]).read_text(encoding="utf-8"))
    assert recommended["stage21_12_recommended"] is True
    assert recommended["runs_new_ppo_update"] is False
    stage21_2 = json.loads(Path(summary["recommended_stage21_2_config"]).read_text(encoding="utf-8"))
    assert stage21_2["coverage_first_reward_profile"].endswith("xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json")


def test_stage21_12_routes_to_continue_when_v2_alignment_still_too_low(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path, min_v2_rate=1.1)

    summary = run_xunce_stage21_12_coverage_constrained_reward_profile_repair(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_CONTINUE
    assert "v2_reward_best_still_not_aligned_with_coverage_per_cost" in summary["reward_reason_codes"]


def test_stage21_12_routes_to_advantage_when_v2_advantage_is_flat(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path, min_advantage_std=100.0)

    summary = run_xunce_stage21_12_coverage_constrained_reward_profile_repair(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_ADVANTAGE
    assert "v2_advantage_replay_flat" in summary["advantage_reason_codes"]


def test_stage21_12_does_not_mark_hard_risk_transition_trainable(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path, hard_risk_selected=True, min_advantage_std=10.0)

    summary = run_xunce_stage21_12_coverage_constrained_reward_profile_repair(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )
    comparison = json.loads(Path(summary["reward_profile_comparison"]).read_text(encoding="utf-8"))

    assert comparison["v2_hard_risk_trainable_count"] == 0
    replay_rows = [json.loads(line) for line in Path(summary["reward_replay"]).read_text(encoding="utf-8").splitlines()]
    assert any(row["hard_risk_rejected"] and row["selected_v2_trainable"] is False for row in replay_rows)


def test_stage21_12_uses_selected_hard_risk_clean_mask_for_trainability(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path, min_advantage_std=10.0)
    batch_path = root / "stage21_10" / "s6" / "seed_2101" / "stage21_3" / "xunce-stage21-3-ppo-trainable-batch.jsonl"
    rows = [json.loads(line) for line in batch_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["info"]["path_allowed_by_risk"] = True
    rows[0]["info"]["hard_risk_flags"] = []
    rows[0]["info"]["hard_risk_clean_mask"] = [False, True]
    rows[0]["observation"]["hard_risk_clean_mask"] = [False, True]
    _jsonl(batch_path, rows)
    _write_stage21_2_reward_artifacts(root / "stage21_10" / "s6" / "seed_2101" / "stage21_2", rows)

    summary = run_xunce_stage21_12_coverage_constrained_reward_profile_repair(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )
    replay_rows = [json.loads(line) for line in Path(summary["reward_replay"]).read_text(encoding="utf-8").splitlines()]

    assert any(row["hard_risk_rejected"] and row["selected_v2_trainable"] is False for row in replay_rows)


def test_stage21_12_boundary_flags_hard_fail(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path, starts_online_canary=True)

    summary = run_xunce_stage21_12_coverage_constrained_reward_profile_repair(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_BOUNDARY
    assert "config_starts_online_canary_true" in summary["boundary_reason_codes"]


def test_stage21_12_missing_seed_batch_routes_to_inputs(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path)
    missing = root / "stage21_10" / "s6" / "seed_2101" / "stage21_3" / "xunce-stage21-3-ppo-trainable-batch.jsonl"
    missing.unlink()

    summary = run_xunce_stage21_12_coverage_constrained_reward_profile_repair(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_INPUTS
    assert "seed_2101_missing_stage21_3_trainable_batch" in summary["input_reason_codes"]


def test_stage21_12_missing_stage21_2_reward_row_routes_to_inputs(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path)
    reward_path = root / "stage21_10" / "s6" / "seed_2101" / "stage21_2" / "xunce-stage21-2-reward-contract-evaluation.jsonl"
    rows = [json.loads(line) for line in reward_path.read_text(encoding="utf-8").splitlines()]
    _jsonl(reward_path, rows[:-1])

    summary = run_xunce_stage21_12_coverage_constrained_reward_profile_repair(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_INPUTS
    assert "seed_2101_stage21_2_missing_reward_rows_for_batch" in summary["input_reason_codes"]


def test_stage21_12_duplicate_or_extra_stage21_2_reward_rows_route_to_inputs(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path)
    reward_path = root / "stage21_10" / "s6" / "seed_2101" / "stage21_2" / "xunce-stage21-2-reward-contract-evaluation.jsonl"
    rows = [json.loads(line) for line in reward_path.read_text(encoding="utf-8").splitlines()]
    extra = dict(rows[0])
    extra["transition_id"] = "extra-transition"
    _jsonl(reward_path, rows + [rows[0], extra])

    summary = run_xunce_stage21_12_coverage_constrained_reward_profile_repair(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_INPUTS
    assert "seed_2101_stage21_2_duplicate_transition_id" in summary["input_reason_codes"]
    assert "seed_2101_stage21_2_extra_reward_rows" in summary["input_reason_codes"]


def test_stage21_12_missing_stage21_3_split_routes_to_inputs(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path)
    batch_path = root / "stage21_10" / "s6" / "seed_2101" / "stage21_3" / "xunce-stage21-3-ppo-trainable-batch.jsonl"
    rows = [json.loads(line) for line in batch_path.read_text(encoding="utf-8").splitlines()]
    rows[0].pop("stage21_3_split", None)
    _jsonl(batch_path, rows)

    summary = run_xunce_stage21_12_coverage_constrained_reward_profile_repair(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_INPUTS
    assert "seed_2101_stage21_3_missing_split" in summary["input_reason_codes"]


def test_stage21_12_uses_stage21_11_coverage_per_cost_definition(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path)

    summary = run_xunce_stage21_12_coverage_constrained_reward_profile_repair(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )
    rows = [json.loads(line) for line in Path(summary["reward_replay"]).read_text(encoding="utf-8").splitlines()]
    first_selected = rows[0]["selected"]

    assert first_selected["coverage_gain_rate"] == 0.1
    assert first_selected["path_cost_proxy"] == 0.1
    assert first_selected["coverage_per_cost"] == 1.0


def test_stage21_12_advantage_replay_groups_by_seed_and_scenario(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path)
    stage21_6 = root / "stage21_10" / "s6"
    seed2 = stage21_6 / "seed_2102"
    (seed2 / "stage21_2").mkdir(parents=True, exist_ok=True)
    (seed2 / "stage21_3").mkdir(parents=True, exist_ok=True)
    batch_rows = [
        _batch_row(0, [0.080, 0.079], [0.10, 0.01], "hash-c", False),
        _batch_row(0, [0.100, 0.098], [0.10, 0.01], "hash-d", False),
    ]
    _jsonl(seed2 / "stage21_3" / "xunce-stage21-3-ppo-trainable-batch.jsonl", batch_rows)
    _write_stage21_2_reward_artifacts(seed2 / "stage21_2", batch_rows)
    seed_rows_path = stage21_6 / "xunce-stage21-6-seed-results.jsonl"
    seed_rows = [json.loads(line) for line in seed_rows_path.read_text(encoding="utf-8").splitlines()]
    seed_rows.append(
        {
            "schema_version": "xunce-stage21-6-seed-result/v1",
            "seed": 2102,
            "seed_root": str(seed2),
            "stage21_2_root": str(seed2 / "stage21_2"),
            "stage21_3_root": str(seed2 / "stage21_3"),
        }
    )
    _jsonl(seed_rows_path, seed_rows)

    summary = run_xunce_stage21_12_coverage_constrained_reward_profile_repair(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "passed"
    audit = json.loads(Path(summary["advantage_replay_audit"]).read_text(encoding="utf-8"))
    assert audit["v2_advantage_coverage_per_cost_correlation"] > 0.0


def _fixture(
    tmp_path: Path,
    *,
    min_v2_rate: float = 0.5,
    min_advantage_std: float = 1e-9,
    hard_risk_selected: bool = False,
    starts_online_canary: bool = False,
) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    stage21_11 = root / "stage21_11"
    stage21_6 = root / "stage21_10" / "s6"
    seed = stage21_6 / "seed_2101"
    (seed / "stage21_3").mkdir(parents=True, exist_ok=True)
    (seed / "stage21_2").mkdir(parents=True, exist_ok=True)
    batch_rows = _batch_rows(hard_risk_selected)
    _json(stage21_11 / "xunce-stage21-11-summary.json", _stage21_11_summary(stage21_6, stage21_11))
    _json(stage21_11 / "xunce-stage21-11-recommended-stage21-6-config.json", _recommended_stage21_6())
    _json(stage21_6 / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json", {"status": "failed"})
    _json(stage21_6 / "xunce-stage21-6-aggregate-metrics.json", _aggregate())
    _json(stage21_6 / "xunce-stage21-6-lineage-audit.json", {"passed": True})
    _jsonl(
        stage21_6 / "xunce-stage21-6-seed-results.jsonl",
        [
            {
                "schema_version": "xunce-stage21-6-seed-result/v1",
                "seed": 2101,
                "seed_root": str(seed),
                "stage21_2_root": str(seed / "stage21_2"),
                "stage21_3_root": str(seed / "stage21_3"),
            }
        ],
    )
    _jsonl(seed / "stage21_3" / "xunce-stage21-3-ppo-trainable-batch.jsonl", batch_rows)
    _write_stage21_2_reward_artifacts(seed / "stage21_2", batch_rows)
    stage21_2_base = root / "stage21_2_base.json"
    _json(
        stage21_2_base,
        {
            "schema_version": "xunce-stage21-2-coverage-first-ppo-reward-contract-config/v1",
            "stage21_1_collector_root": "unused",
            "coverage_first_reward_profile": "configs/xunce_stage21_coverage_first_ppo_reward_profile_v1.json",
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    config = root / "config.json"
    _json(
        config,
        {
            "schema_version": "xunce-stage21-12-coverage-constrained-reward-profile-repair-config/v1",
            "stage21_11_root": str(stage21_11),
            "stage21_10_stage21_6_root": str(stage21_6),
            "reward_profile_v1": str(REPO_ROOT / "configs" / "xunce_stage21_coverage_first_ppo_reward_profile_v1.json"),
            "reward_profile_v2": str(REPO_ROOT / "configs" / "xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json"),
            "stage21_2_base_config": str(stage21_2_base),
            "min_v2_reward_best_matches_coverage_per_cost_rate": min_v2_rate,
            "min_v2_alignment_improvement": 0.1,
            "min_v2_advantage_std": min_advantage_std,
            "starts_online_canary": starts_online_canary,
        },
    )
    return root, config


def _stage21_11_summary(stage21_6: Path, stage21_11: Path) -> dict[str, object]:
    return {
        "status": "failed",
        "next_required_change": "repair_stage21_coverage_constrained_reward_profile",
        "stage21_6_root": str(stage21_6),
        "reward_best_matches_coverage_per_cost_rate": 0.05,
        "recommended_stage21_6_config": str(stage21_11 / "xunce-stage21-11-recommended-stage21-6-config.json"),
    }


def _recommended_stage21_6() -> dict[str, object]:
    return {
        "schema_version": "xunce-stage21-6-multi-seed-ppo-pilot-config/v1",
        "stage_id": "xunce-stage21-6-multi-seed-ppo-pilot",
        "stage21_2_base_config": "configs/xunce_stage21_2_coverage_first_ppo_reward_contract_v1.json",
        "stage21_4_base_config": "stage21_9_repaired.json",
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _aggregate() -> dict[str, object]:
    return {
        "trainable_transition_count_total": 2,
        "final_coverage_delta_mean": 0.0,
        "coverage_auc_delta_mean": 0.0,
        "hard_risk_violation_total": 0,
        "model_inference_mask_violation_total": 0,
        "model_inference_failure_total": 0,
        "path_planning_failure_total": 0,
        "open_grid_fallback_total": 0,
        "safety_boundary_violation_total": 0,
        "execution_boundary_violation_total": 0,
    }


def _batch_rows(hard_risk_selected: bool) -> list[dict[str, object]]:
    rows = [
        _batch_row(0, [0.100, 0.098], [0.10, 0.01], "hash-a", hard_risk_selected),
        _batch_row(0, [0.080, 0.079], [0.10, 0.01], "hash-b", False),
    ]
    return rows


def _batch_row(action_index: int, coverage: list[float], path_cost: list[float], hash_value: str, hard_risk: bool) -> dict[str, object]:
    features = [
        [0, 0, 0, 0, 0, 0, 1, coverage[0], 0, 0, 0, 0, 0, path_cost[0], 0],
        [0, 0, 0, 0, 0, 0, 1, coverage[1], 0, 0, 0, 0, 0, path_cost[1], 0],
    ]
    return {
        "transition_id": f"scenario:{hash_value}",
        "scenario_id": "scenario",
        "step_index": 0 if hash_value == "hash-a" else 1,
        "action_index": action_index,
        "trainable": True,
        "info": {
            "scenario_id": "scenario",
            "step_index": 0 if hash_value == "hash-a" else 1,
            "candidate_set_hash": hash_value,
            "action_mask": [True, True],
            "candidate_cells": [[0, 0], [1, 1]],
            "coverage_rate_delta": coverage[action_index],
            "final_coverage_rate_after_step": 0.10 + coverage[action_index],
            "path_cost": path_cost[action_index],
            "soft_risk_exposure": 0.0,
            "path_allowed_by_risk": not hard_risk,
            "hard_risk_flags": ["obstacle_crossing"] if hard_risk else [],
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
        "value": 0.0,
        "stage21_3_split": "train",
    }


def _write_stage21_2_reward_artifacts(root: Path, batch_rows: list[dict[str, object]]) -> None:
    profile = load_coverage_first_reward_profile(REPO_ROOT / "configs" / "xunce_stage21_coverage_first_ppo_reward_profile_v1.json")
    rows: list[dict[str, object]] = []
    for row in batch_rows:
        info = row["info"]  # type: ignore[index]
        action_index = int(row["action_index"])  # type: ignore[arg-type]
        clean_mask = info.get("hard_risk_clean_mask")  # type: ignore[attr-defined]
        selected_hard_risk_clean = True
        if isinstance(clean_mask, list) and 0 <= action_index < len(clean_mask):
            selected_hard_risk_clean = bool(clean_mask[action_index])
        path_allowed = bool(info["path_allowed_by_risk"]) and selected_hard_risk_clean  # type: ignore[index]
        hard_risk_flags = list(info["hard_risk_flags"])  # type: ignore[index]
        if not selected_hard_risk_clean and "hard_risk_clean_mask_false" not in hard_risk_flags:
            hard_risk_flags.append("hard_risk_clean_mask_false")
        result = compute_coverage_first_reward_components(
            {
                "coverage_rate_delta": info["coverage_rate_delta"],  # type: ignore[index]
                "coverage_progress_rate": info["final_coverage_rate_after_step"],  # type: ignore[index]
                "final_coverage_rate": info["final_coverage_rate_after_step"],  # type: ignore[index]
                "path_cost_m": info["path_cost"],  # type: ignore[index]
                "soft_risk_exposure": info["soft_risk_exposure"],  # type: ignore[index]
                "done": False,
                "path_allowed_by_risk": path_allowed,
                "hard_risk_flags": hard_risk_flags,
                "hard_risk_violation_count": 1 if hard_risk_flags else 0,
            },
            profile,
        )
        rows.append(
            {
                "transition_id": row["transition_id"],
                "reward": result.reward,
                "components": result.components,
                "profile_hash": result.profile_hash,
                "profile_id": result.profile_id,
                "profile_version": result.profile_version,
            }
        )
    _json(
        root / "xunce-stage21-2-coverage-first-reward-summary.json",
        {"status": "passed", "profile_hash": profile.profile_hash},
    )
    _jsonl(root / "xunce-stage21-2-reward-contract-evaluation.jsonl", rows)


def _json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
