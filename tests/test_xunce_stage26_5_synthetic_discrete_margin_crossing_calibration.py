from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


SYNTHETIC_HASH = "e628d24c6fb0f961c50294ea0a018a2558a79a0b30ad21dd31b6e34fc0867e85"


def test_stage26_5_routes_iterative_probe_when_best_rank_closes_but_not_crossed(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_5_synthetic_discrete_margin_crossing_calibration as s26

    stage26_4 = _write_stage26_4_root(tmp_path, mode="rank_closing")
    config = _write_config(tmp_path, stage26_4)

    summary = s26.run_xunce_stage26_5_synthetic_discrete_margin_crossing_calibration(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    margin = json.loads((tmp_path / "out" / s26.MARGIN_AUDIT_FILE).read_text(encoding="utf-8"))
    credit = json.loads((tmp_path / "out" / s26.CREDIT_AUDIT_FILE).read_text(encoding="utf-8"))
    feature = json.loads((tmp_path / "out" / s26.FEATURE_AUDIT_FILE).read_text(encoding="utf-8"))

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s26.ROUTE_ITERATIVE
    assert summary["stage26_4_hybrid_astar_candidate_eval_workers"] == 4
    assert margin["strong_state_join_available_count"] == 1
    assert margin["mean_best_rank_gap_closure"] > 0.0
    assert margin["mean_estimated_updates_to_cross_margin"] > 1.0
    assert credit["counterfactual_best_not_directly_credited"] is False
    assert feature["candidate_feature_signal_missing"] is False
    assert summary["publishes_checkpoint"] is False


def test_stage26_5_routes_exploration_credit_when_best_candidate_was_never_selected(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_5_synthetic_discrete_margin_crossing_calibration as s26

    stage26_4 = _write_stage26_4_root(tmp_path, mode="best_never_selected")
    config = _write_config(tmp_path, stage26_4)

    summary = s26.run_xunce_stage26_5_synthetic_discrete_margin_crossing_calibration(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    credit = json.loads((tmp_path / "out" / s26.CREDIT_AUDIT_FILE).read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s26.ROUTE_EXPLORATION_CREDIT
    assert credit["counterfactual_best_not_directly_credited"] is True
    assert credit["best_candidate_selected_in_train_count"] == 0


def test_stage26_5_routes_feature_exposure_when_candidate_features_lack_synthetic_signal(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_5_synthetic_discrete_margin_crossing_calibration as s26

    stage26_4 = _write_stage26_4_root(tmp_path, mode="feature_missing")
    config = _write_config(tmp_path, stage26_4)

    summary = s26.run_xunce_stage26_5_synthetic_discrete_margin_crossing_calibration(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    feature = json.loads((tmp_path / "out" / s26.FEATURE_AUDIT_FILE).read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s26.ROUTE_FEATURE_EXPOSURE
    assert feature["candidate_feature_signal_missing"] is True
    assert "hybrid_astar_path_costs" in feature["missing_candidate_feature_fields"]


def test_stage26_5_rejects_stage26_4_without_parallel_collector(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_5_synthetic_discrete_margin_crossing_calibration as s26

    stage26_4 = _write_stage26_4_root(tmp_path, worker_count=1)
    config = _write_config(tmp_path, stage26_4)

    summary = s26.run_xunce_stage26_5_synthetic_discrete_margin_crossing_calibration(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s26.ROUTE_PARALLEL_COLLECTOR
    assert "stage26_4_worker_count_not_4" in summary["blocking_reason_codes"]


def test_stage26_5_uses_stage26_4_default_config_when_artifact_omits_worker(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_5_synthetic_discrete_margin_crossing_calibration as s26

    stage26_4 = _write_stage26_4_root(tmp_path, mode="rank_closing", worker_count=0)
    (stage26_4 / "xunce-stage26-4-stage26-1-config.json").write_text("{}", encoding="utf-8")
    config = _write_config(tmp_path, stage26_4)
    default_config = tmp_path / "configs" / "xunce_stage26_4_synthetic_policy_update_signal_strength_repair_v1.json"
    _write_json(default_config, {"hybrid_astar_candidate_eval_workers": 4})
    monkeypatch.setattr(s26, "_read_json_if_exists", _read_json_if_exists_with_default(default_config))

    summary = s26.run_xunce_stage26_5_synthetic_discrete_margin_crossing_calibration(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["stage26_4_hybrid_astar_candidate_eval_workers"] == 4
    assert "stage26_4_worker_count_not_4" not in summary["blocking_reason_codes"]


def test_stage26_5_rejects_unexpected_stage26_4_route(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_5_synthetic_discrete_margin_crossing_calibration as s26

    stage26_4 = _write_stage26_4_root(tmp_path, next_required_change="repair_stage26_policy_value_loss_balance")
    config = _write_config(tmp_path, stage26_4)

    summary = s26.run_xunce_stage26_5_synthetic_discrete_margin_crossing_calibration(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s26.ROUTE_INPUTS
    assert "stage26_4_route_not_discrete_margin" in summary["blocking_reason_codes"]


def _write_config(tmp_path: Path, stage26_4_root: Path) -> Path:
    path = tmp_path / "stage26_5_config.json"
    _write_json(
        path,
        {
            "schema_version": "xunce-stage26-5-synthetic-discrete-margin-crossing-calibration-config/v1",
            "stage_id": "xunce-stage26-5-synthetic-discrete-margin-crossing-calibration",
            "stage26_4_root": str(stage26_4_root),
            "required_hybrid_astar_candidate_eval_workers": 4,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    return path


def _write_stage26_4_root(
    tmp_path: Path,
    *,
    mode: str = "rank_closing",
    worker_count: int = 4,
    next_required_change: str = "calibrate_stage26_synthetic_discrete_margin_crossing",
) -> Path:
    root = tmp_path / "stage26_4"
    c1 = root / "c" / "c1"
    c1_s3 = c1 / "s3"
    c1_s2 = c1 / "s2"
    s26_1 = root / "s26_1"
    for path in (c1_s3 / "pre", c1_s3 / "post", c1_s2, s26_1 / "s21_3", s26_1 / "s21_2"):
        path.mkdir(parents=True, exist_ok=True)
    direct_selected = mode != "best_never_selected"
    include_features = mode != "feature_missing"
    pre = _inference_row(
        probs=[0.56, 0.16, 0.28],
        logits=[1.2, 0.0, 0.5],
        selected_index=0,
        best_index=1,
        include_features=include_features,
    )
    post = _inference_row(
        probs=[0.52, 0.30, 0.18],
        logits=[1.0, 0.7, -0.1],
        selected_index=0,
        best_index=1,
        include_features=include_features,
    )
    _write_jsonl(c1_s3 / "pre" / "xunce-exploration-coverage-model-inference.jsonl", [pre])
    _write_jsonl(c1_s3 / "post" / "xunce-exploration-coverage-model-inference.jsonl", [post])
    _write_json(
        c1_s3 / "xunce-stage26-3-summary.json",
        {
            "schema_version": "xunce-stage26-3-summary/v1",
            "status": "failed",
            "next_required_change": "calibrate_stage26_synthetic_discrete_margin_crossing",
            "strong_state_join_available_count": 1,
            "synthetic_terrain_hash": SYNTHETIC_HASH,
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
            "path_cost_source": "hybrid_astar_pose_path/v1",
        },
    )
    _write_json(
        c1_s2 / "xunce-stage26-2-summary.json",
        {
            "schema_version": "xunce-stage26-2-summary/v1",
            "status": "passed",
            "batch_row_count": 16,
        },
    )
    _write_json(
        c1_s2 / "xunce-stage26-2-loss-gradient-audit.json",
        {
            "component_grad_norms": {
                "policy_loss_grad_norm": 1.0,
                "value_loss_grad_norm": 0.1,
                "entropy_loss_grad_norm": 0.01,
            },
            "final_post_update_approx_kl": 0.001,
            "pre_clip_grad_norm": 1.0,
            "post_clip_grad_norm": 1.0,
        },
    )
    _write_jsonl(
        s26_1 / "s21_3" / "xunce-stage21-3-trainable-ppo-batch.jsonl",
        [
            {
                "scenario_id": "scenario-1",
                "step_index": 0,
                "selected_action_index": 1 if direct_selected else 0,
                "action_index": 1 if direct_selected else 0,
                "advantage": 1.2,
            }
        ],
    )
    _write_jsonl(
        s26_1 / "s21_2" / "xunce-stage21-2-coverage-first-reward-evaluation.jsonl",
        [{"reward": 1.0, "coverage_gain_per_path_cost": 3.0}],
    )
    _write_json(
        root / "xunce-stage26-4-summary.json",
        {
            "schema_version": "xunce-stage26-4-summary/v1",
            "stage_id": "xunce-stage26-4-synthetic-policy-update-signal-strength-repair",
            "status": "failed",
            "next_required_change": next_required_change,
            "stage26_1_root": str(s26_1),
            "best_combo_id": "policy_amp_depth",
            "trainable_transition_count": 16,
            "stable_combo_count": 1,
            "synthetic_terrain_hash": SYNTHETIC_HASH,
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
            "path_cost_source": "hybrid_astar_pose_path/v1",
            "max_traversable_slope_deg": 30.0,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    _write_json(
        root / "xunce-stage26-4-stage26-1-config.json",
        {"hybrid_astar_candidate_eval_workers": worker_count},
    )
    _write_jsonl(
        root / "xunce-stage26-4-sweep-results.jsonl",
        [
            {
                "combo_id": "policy_amp_depth",
                "combo_work_id": "c1",
                "combo_root": str(c1),
                "stage26_2_status": "passed",
                "stage26_3_status": "failed",
                "stage26_3_next_required_change": "calibrate_stage26_synthetic_discrete_margin_crossing",
                "strong_state_join_available_count": 1,
                "mean_abs_probability_delta": 1.0e-5,
                "selected_action_changed_count": 0,
                "selected_viewpoint_changed_count": 0,
                "selected_theta_changed_count": 0,
                "synthetic_contract_mismatch_count": 0,
                "physical_obstacle_payload_count": 0,
                "grid_fallback_count": 0,
            }
        ],
    )
    return root


def _inference_row(
    *,
    probs: list[float],
    logits: list[float],
    selected_index: int,
    best_index: int,
    include_features: bool,
) -> dict:
    row = {
        "policy": "xunce",
        "scenario_id": "scenario-1",
        "step_index": 0,
        "current_cell": [0, 0],
        "covered_cells_hash": "covered-a",
        "candidate_set_hash": "candidate-a",
        "synthetic_terrain_hash": SYNTHETIC_HASH,
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "selected_action_index": selected_index,
        "selected_viewpoint": [selected_index, 1, 0],
        "selected_theta_deg": 0,
        "candidate_viewpoints": [[0, 1, 0], [1, 1, 45], [2, 1, 90]],
        "action_probs": probs,
        "logits": logits,
        "action_mask": [True, True, True],
        "theta_coverage_gain_per_path_costs": [1.0, 4.0 if best_index == 1 else 0.5, 2.0],
        "synthetic_los_blocker_cells_used": True,
        "synthetic_hard_obstacle_cells_used": True,
        "physical_obstacle_cells_written": False,
        "default_astar_replaced": False,
        "hybrid_astar_ackermann_feasible_claimed": False,
    }
    if include_features:
        row["hybrid_astar_path_costs"] = [6.0, 5.0, 7.0]
        row["synthetic_los_blocker_candidate_counts"] = [0, 3, 1]
        row["synthetic_hard_obstacle_candidate_counts"] = [0, 2, 1]
    return row


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_json_if_exists_with_default(default_config: Path):
    def reader(path: Path) -> dict:
        if path.as_posix().endswith("configs/xunce_stage26_4_synthetic_policy_update_signal_strength_repair_v1.json"):
            return json.loads(default_config.read_text(encoding="utf-8"))
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    return reader
