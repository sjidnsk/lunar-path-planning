from __future__ import annotations

import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_path_efficiency_v2_penalizes_high_hybrid_path_cost() -> None:
    from scripts.xunce_synthetic_exploration_credit import select_synthetic_credit_target

    rows = [
        [0.1, 1.0, 0.6, 0.85, 1.0, 0.0, 0.0, 0.0],
        [0.1, 1.0, 0.5, 0.75, 0.2, 0.0, 0.0, 0.0],
    ]

    v1 = select_synthetic_credit_target(
        rows,
        action_mask=[True, True],
        sampling_mask=[True, True],
        hard_risk_clean_mask=[True, True],
        hybrid_reachable_flags=[True, True],
        score_version="coverage_proxy_v1",
    )
    v2 = select_synthetic_credit_target(
        rows,
        action_mask=[True, True],
        sampling_mask=[True, True],
        hard_risk_clean_mask=[True, True],
        hybrid_reachable_flags=[True, True],
        score_version="path_efficiency_v2",
    )

    assert v1["synthetic_credit_target_index"] == 0
    assert v2["synthetic_credit_target_index"] == 1
    assert v2["synthetic_credit_score_version"] == "path_efficiency_v2"
    assert v2["selected_target_hybrid_cost_norm"] == 0.2


def test_path_efficiency_v2_filters_high_cost_candidates_and_reports_relaxation() -> None:
    from scripts.xunce_synthetic_exploration_credit import select_synthetic_credit_target

    rows = [
        [0.1, 1.0, 0.8, 0.9, 0.9, 0.0, 0.0, 0.0],
        [0.1, 1.0, 0.6, 0.7, 0.6, 0.0, 0.0, 0.0],
    ]
    strict = select_synthetic_credit_target(
        rows,
        action_mask=[True, True],
        sampling_mask=[True, True],
        hard_risk_clean_mask=[True, True],
        hybrid_reachable_flags=[True, True],
        score_version="path_efficiency_v2",
        path_efficiency_max_cost_norm=0.7,
    )
    relaxed = select_synthetic_credit_target(
        rows,
        action_mask=[True, True],
        sampling_mask=[True, True],
        hard_risk_clean_mask=[True, True],
        hybrid_reachable_flags=[True, True],
        score_version="path_efficiency_v2",
        path_efficiency_max_cost_norm=0.1,
    )

    assert strict["synthetic_credit_target_index"] == 1
    assert strict["path_efficiency_filter_relaxed"] is False
    assert relaxed["path_efficiency_filter_relaxed"] is True
    assert relaxed["synthetic_credit_target_index"] in {0, 1}


def test_stage21_1_synthetic_credit_metadata_writes_path_efficiency_fields() -> None:
    import torch
    import scripts.run_xunce_stage21_1_on_policy_ppo_rollout_collector as s21_1

    result = s21_1._synthetic_credit_step_metadata(
        xunce_batch={"candidate_features": torch.zeros((1, 2, 8), dtype=torch.float32)},
        observation_payload={
            "candidate_feature_names": ["relative_distance", "risk"],
            "candidate_features": [[1.0, 0.1], [2.0, 0.1]],
        },
        slope_theta_metadata={
            "obstacle_aware_new_visible_cell_counts": [8, 6],
            "obstacle_aware_theta_coverage_gain_per_path_costs": [0.9, 0.75],
        },
        hybrid_path_metadata={
            "hybrid_astar_reachable_flags": [True, True],
            "hybrid_astar_path_costs": [100.0, 20.0],
            "hybrid_astar_reachability_probe_theta_deg": 30.0,
        },
        synthetic_terrain_metadata={
            "synthetic_los_blocker_candidate_counts": [0, 0],
            "synthetic_hard_obstacle_candidate_counts": [0, 0],
        },
        action_mask=(True, True),
        sampling_mask=(True, True),
        hard_risk_clean_mask=(True, True),
        enabled=True,
        score_version="path_efficiency_v2",
        path_efficiency_max_cost_norm=0.7,
    )

    assert result["synthetic_credit_target_index"] == 1
    assert result["synthetic_credit_score_version"] == "path_efficiency_v2"
    assert result["path_efficiency_filter_relaxed"] is False
    assert result["selected_target_hybrid_cost_norm"] < 1.0
    assert result["selected_target_gain_per_cost_norm"] > 0.0


def test_stage26_7_rejects_untrusted_stage26_6_root(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair as s26

    stage26_6 = tmp_path / "stage26_6"
    stage26_6.mkdir()
    _write_json(stage26_6 / "xunce-stage26-6-summary.json", {"status": "passed"})
    config = _stage26_7_config(tmp_path, stage26_6)

    summary = s26.run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_7_required_inputs"


def test_stage26_7_routes_to_multi_seed_when_v2_improves_efficiency() -> None:
    import scripts.run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair as s26

    route = s26._route(
        boundary_rejections=[],
        input_rejections=[],
        target_audit={"v2_target_selected_count": 4},
        behavior_audit={"behavior_logprob_recomputable": True},
        eval_comparison={
            "best_combo": {
                "stage26_1_status": "passed",
                "stage26_2_status": "passed",
                "stage26_3_status": "passed",
                "selected_action_changed_count": 1,
                "final_coverage_delta": 0.1,
                "coverage_auc_delta": 0.1,
                "hybrid_astar_path_cost_delta": 12.0,
                "coverage_per_100m_delta": 0.1,
                "scenario_regression_count": 0,
                "hard_risk_violation_count": 0,
                "mask_violation_count": 0,
                "path_planning_failure_count": 0,
                "open_grid_fallback_count": 0,
            }
        },
        sweep_rows=[],
    )

    assert route == "run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot"


def test_stage26_7_does_not_route_to_multi_seed_without_passed_substages() -> None:
    import scripts.run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair as s26

    route = s26._route(
        boundary_rejections=[],
        input_rejections=[],
        target_audit={"v2_target_selected_count": 4},
        behavior_audit={"behavior_logprob_recomputable": True},
        eval_comparison={
            "best_combo": {
                "stage26_1_status": "passed",
                "stage26_2_status": "passed",
                "stage26_3_status": "failed",
                "selected_action_changed_count": 1,
                "final_coverage_delta": 0.1,
                "coverage_auc_delta": 0.1,
                "hybrid_astar_path_cost_delta": -1.0,
                "coverage_per_100m_delta": 0.1,
                "scenario_regression_count": 0,
                "hard_risk_violation_count": 0,
                "mask_violation_count": 0,
                "path_planning_failure_count": 0,
                "open_grid_fallback_count": 0,
            }
        },
        sweep_rows=[],
    )

    assert route != "run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot"


def test_stage26_7_reads_v2_target_fields_from_trainable_batch(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair as s26

    root = tmp_path / "s26_1" / "s21_1"
    root.mkdir(parents=True)
    row = {
        "transition_id": "t0",
        "action_index": 2,
        "behavior_policy_id": "synthetic_credit_mixture_policy/v1",
        "old_log_prob": -1.0,
        "old_behavior_log_prob": -1.0,
        "synthetic_credit_target_index": 2,
        "synthetic_credit_target_selected": True,
        "synthetic_credit_score": 0.42,
        "synthetic_credit_score_version": "path_efficiency_v2",
        "path_efficiency_filter_relaxed": False,
        "selected_target_hybrid_cost_norm": 0.33,
        "selected_target_gain_per_cost_norm": 0.77,
    }
    _write_jsonl(root / "xunce-stage21-1-ppo-trainable-batch.jsonl", [row])

    rows = s26._target_rows(tmp_path / "s26_1")
    audit = s26._combo_target_audit(
        {
            "combo_id": "path_efficiency_v2_default",
            "synthetic_credit_score_version": "path_efficiency_v2",
            "path_efficiency_max_cost_norm": 0.7,
        },
        rows,
    )

    assert rows[0]["synthetic_credit_score_version"] == "path_efficiency_v2"
    assert rows[0]["synthetic_credit_target_selected"] is True
    assert audit["synthetic_credit_target_selected_count"] == 1
    assert audit["synthetic_credit_score_version_count"] == 1
    assert audit["mean_selected_target_hybrid_cost_norm"] == 0.33
    assert audit["mean_selected_target_gain_per_cost_norm"] == 0.77


def test_stage26_7_routes_to_target_score_when_unit_distance_coverage_declines() -> None:
    import scripts.run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair as s26

    route = s26._route(
        boundary_rejections=[],
        input_rejections=[],
        target_audit={"v2_target_selected_count": 4},
        behavior_audit={"behavior_logprob_recomputable": True},
        eval_comparison={
            "best_combo": {
                "selected_action_changed_count": 1,
                "final_coverage_delta": 0.1,
                "coverage_auc_delta": 0.1,
                "hybrid_astar_path_cost_delta": 12.0,
                "coverage_per_100m_delta": -0.1,
                "scenario_regression_count": 0,
            }
        },
        sweep_rows=[],
    )

    assert route == "repair_stage26_7_path_efficiency_target_score"


def test_stage26_7_score_sweep_uses_short_work_dirs() -> None:
    import scripts.run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair as s26

    combos = s26._score_sweep(
        [
            {
                "combo_id": "path_efficiency_v2_default",
                "synthetic_credit_score_version": "path_efficiency_v2",
                "path_efficiency_max_cost_norm": 0.7,
            }
        ]
    )

    assert combos[0]["combo_id"] == "path_efficiency_v2_default"
    assert combos[0]["work_dir"] == "c0"


def test_stage26_7_reuses_completed_combo_outputs(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair as s26

    stage26_6 = tmp_path / "stage26_6"
    stage26_6.mkdir()
    _write_json(
        stage26_6 / "xunce-stage26-6-summary.json",
        {
            "status": "failed",
            "next_required_change": "repair_stage26_synthetic_credit_assignment",
            "feature_exposure_fixed": True,
            "behavior_logprob_contract_fixed": True,
            "synthetic_credit_target_selected_count": 12,
        },
    )
    config_path = _stage26_7_config(tmp_path, stage26_6)
    config = s26._load_config(config_path, REPO_ROOT)
    combo = config["score_sweep"][0]
    combo_root = tmp_path / "out" / combo["work_dir"]
    _write_json(combo_root / "s26_1" / "xunce-stage26-1-summary.json", {"status": "passed"})
    _write_json(combo_root / "s26_2" / "xunce-stage26-2-summary.json", {"status": "passed"})
    _write_json(
        combo_root / "s26_3" / "xunce-stage26-3-summary.json",
        {"status": "failed", "next_required_change": "repair_stage26_synthetic_credit_assignment"},
    )

    monkeypatch.setattr(
        s26.stage26_1,
        "run_xunce_stage26_1_synthetic_terrain_collector_smoke",
        lambda **_: (_ for _ in ()).throw(AssertionError("stage26_1 should be reused")),
    )
    monkeypatch.setattr(
        s26.stage26_2,
        "run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke",
        lambda **_: (_ for _ in ()).throw(AssertionError("stage26_2 should be reused")),
    )
    monkeypatch.setattr(
        s26.stage26_3,
        "run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke",
        lambda **_: (_ for _ in ()).throw(AssertionError("stage26_3 should be reused")),
    )

    detail = s26._run_combo(config=config, combo=combo, output_root=tmp_path / "out", repo_root=REPO_ROOT)

    assert detail["sweep_row"]["stage26_1_reused_existing_output"] is True
    assert detail["sweep_row"]["stage26_2_reused_existing_output"] is True
    assert detail["sweep_row"]["stage26_3_reused_existing_output"] is True


def test_stage26_7_summary_compares_v1_and_v2_combos(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair as s26

    stage26_6 = tmp_path / "stage26_6"
    stage26_6.mkdir()
    _write_json(
        stage26_6 / "xunce-stage26-6-summary.json",
        {
            "status": "failed",
            "next_required_change": "repair_stage26_synthetic_credit_assignment",
            "feature_exposure_fixed": True,
            "behavior_logprob_contract_fixed": True,
            "synthetic_credit_target_selected_count": 12,
        },
    )
    config = _stage26_7_config(tmp_path, stage26_6)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["run_stage26_chain"] = True
    _write_json(config, payload)

    def fake_run_combo(*, config, combo, output_root, repo_root):
        path_delta = 10.0 if combo["synthetic_credit_score_version"] == "coverage_proxy_v1" else -2.0
        coverage_per_100m = -1.0 if path_delta > 0.0 else 0.2
        row = {
            "schema_version": s26.SWEEP_SCHEMA_VERSION,
            "combo_id": combo["combo_id"],
            "combo_work_dir": combo["work_dir"],
            "synthetic_credit_score_version": combo["synthetic_credit_score_version"],
            "path_efficiency_max_cost_norm": combo["path_efficiency_max_cost_norm"],
            "stage26_1_status": "passed",
            "stage26_2_status": "passed",
            "stage26_3_status": "passed",
            "stage26_3_next_required_change": "run_stage26_4_synthetic_terrain_multi_seed_ppo_pilot",
            "synthetic_credit_target_selected_count": 4,
            "synthetic_credit_score_version_count": 4,
            "path_efficiency_filter_relaxed_count": 0,
            "mean_selected_target_hybrid_cost_norm": 0.3,
            "mean_selected_target_gain_per_cost_norm": 0.8,
            "behavior_policy_row_count": 4,
            "behavior_logprob_mismatch_count": 0,
            "stage21_3_behavior_logprob_missing_count": 0,
            "selected_action_changed_count": 1,
            "selected_viewpoint_changed_count": 1,
            "selected_theta_changed_count": 1,
            "mean_abs_probability_delta": 0.2,
            "final_coverage_delta": 0.1,
            "coverage_auc_delta": 0.1,
            "hybrid_astar_path_cost_delta": path_delta,
            "coverage_per_100m_delta": coverage_per_100m,
            "scenario_regression_count": 0,
            "hard_risk_violation_count": 0,
            "mask_violation_count": 0,
            "path_planning_failure_count": 0,
            "open_grid_fallback_count": 0,
        }
        return {
            "combo": combo,
            "combo_work_dir": combo["work_dir"],
            "combo_root": output_root / combo["work_dir"],
            "stage26_1_config": {"synthetic_credit_score_version": combo["synthetic_credit_score_version"]},
            "stage26_2_config": {"epochs": 4},
            "sweep_row": row,
        }

    monkeypatch.setattr(s26, "_run_combo", fake_run_combo)
    summary = s26.run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["best_combo_score_version"] == "path_efficiency_v2"
    assert summary["next_required_change"] == "run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot"
    sweep_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "xunce-stage26-7-score-sweep-results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["synthetic_credit_score_version"] for row in sweep_rows} == {"coverage_proxy_v1", "path_efficiency_v2"}


def _stage26_7_config(tmp_path: Path, stage26_6_root: Path) -> Path:
    config = tmp_path / "config.json"
    _write_json(
        config,
        {
            "schema_version": "xunce-stage26-7-synthetic-credit-assignment-path-efficiency-repair-config/v1",
            "stage_id": "xunce-stage26-7-synthetic-credit-assignment-path-efficiency-repair",
            "stage26_6_root": str(stage26_6_root),
            "run_stage26_chain": False,
            "score_sweep": [
                {
                    "combo_id": "v1_current_baseline",
                    "work_dir": "c0",
                    "synthetic_credit_score_version": "coverage_proxy_v1",
                    "path_efficiency_max_cost_norm": 0.7,
                },
                {
                    "combo_id": "path_efficiency_v2_default",
                    "work_dir": "c1",
                    "synthetic_credit_score_version": "path_efficiency_v2",
                    "path_efficiency_max_cost_norm": 0.7,
                },
            ],
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    return config


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
