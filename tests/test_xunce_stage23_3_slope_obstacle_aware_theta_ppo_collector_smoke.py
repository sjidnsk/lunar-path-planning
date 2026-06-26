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


def test_stage23_3_runs_slope_theta_collector_reward_batch_smoke(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke as s23

    _patch_stage21_runs(monkeypatch, s23)
    config = _write_config(tmp_path, _write_stage23_2_root(tmp_path))

    summary = s23.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    audit = json.loads((tmp_path / "out" / "xunce-stage23-3-slope-theta-transition-contract-audit.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke"
    assert summary["coverage_source"] == "endpoint_theta_slope_obstacle_los/v1"
    assert summary["trainable_transition_count"] == 1
    assert summary["slope_theta_transition_contract_missing_count"] == 0
    assert summary["strict_obstacle_aware_new_visible_false_count"] == 0
    assert summary["slope_theta_reward_contract_missing_count"] == 0
    assert summary["slope_theta_batch_reward_contract_missing_count"] == 0
    assert summary["stage21_1_source_roi_expansion_root_match"] is True
    assert summary["transition_id_count"] == 1
    assert summary["reward_transition_id_count"] == 1
    assert summary["batch_transition_id_count"] == 1
    assert summary["transition_missing_reward_count"] == 0
    assert summary["reward_transition_id_not_in_stage21_1_count"] == 0
    assert summary["batch_transition_id_not_in_stage21_1_count"] == 0
    assert summary["batch_transition_id_not_in_stage21_2_count"] == 0
    assert audit["stage21_3_rejects_old_reward_batch"] is True
    assert summary["runs_new_ppo_update"] is False
    assert summary["publishes_checkpoint"] is False
    stage21_1_config = json.loads((tmp_path / "out" / "xunce-stage23-3-stage21-1-config.json").read_text(encoding="utf-8"))
    stage21_2_config = json.loads((tmp_path / "out" / "xunce-stage23-3-stage21-2-config.json").read_text(encoding="utf-8"))
    stage21_3_config = json.loads((tmp_path / "out" / "xunce-stage23-3-stage21-3-config.json").read_text(encoding="utf-8"))
    assert stage21_1_config["theta_aware_candidate_viewpoints_enabled"] is True
    assert stage21_1_config["slope_obstacle_aware_theta_reward_enabled"] is True
    assert stage21_1_config["obstacle_occlusion_enabled"] is True
    assert stage21_1_config["derive_slope_blocked_cells_from_sidecar_dem"] is True
    assert stage21_1_config["platform_contract_hash"] == "platform-hash"
    assert stage21_1_config["max_traversable_slope_deg"] == 30.0
    assert stage21_1_config["source_roi_expansion_root"].endswith("high_res_roi_expansion")
    assert stage21_2_config["slope_obstacle_aware_theta_reward_enabled"] is True
    assert stage21_3_config["require_slope_obstacle_aware_theta_reward_contract"] is True


def test_stage23_3_routes_collector_repair_when_slope_fields_missing(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke as s23

    _patch_stage21_runs(monkeypatch, s23, transition_mode="missing_slope")
    config = _write_config(tmp_path, _write_stage23_2_root(tmp_path))

    summary = s23.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage23_3_collector_slope_theta_transition_contract"
    assert "slope_theta_transition_contract_missing" in summary["blocking_reason_codes"]


def test_stage23_3_routes_reward_repair_when_reward_uses_unobstructed_fallback(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke as s23

    _patch_stage21_runs(monkeypatch, s23, reward_mode="fallback")
    config = _write_config(tmp_path, _write_stage23_2_root(tmp_path))

    summary = s23.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage23_3_slope_theta_reward_provenance"
    assert "reward_fallback_used" in summary["blocking_reason_codes"]


def test_stage23_3_routes_reward_repair_when_reward_footprint_mismatches_transition(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke as s23

    _patch_stage21_runs(monkeypatch, s23, reward_mode="bad_footprint")
    config = _write_config(tmp_path, _write_stage23_2_root(tmp_path))

    summary = s23.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage23_3_slope_theta_reward_provenance"
    assert "reward_slope_footprint_binding_mismatch" in summary["blocking_reason_codes"]


def test_stage23_3_routes_reward_repair_when_reward_transition_is_orphan(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke as s23

    _patch_stage21_runs(monkeypatch, s23, reward_transition_id="orphan-transition")
    config = _write_config(tmp_path, _write_stage23_2_root(tmp_path))

    summary = s23.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage23_3_slope_theta_reward_provenance"
    assert "transition_missing_reward" in summary["blocking_reason_codes"]
    assert "reward_transition_id_not_in_stage21_1" in summary["blocking_reason_codes"]


def test_stage23_3_routes_batch_repair_when_batch_transition_is_orphan(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke as s23

    _patch_stage21_runs(monkeypatch, s23, batch_transition_id="orphan-transition")
    config = _write_config(tmp_path, _write_stage23_2_root(tmp_path))

    summary = s23.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage23_3_batch_gate"
    assert "batch_transition_id_not_in_stage21_1" in summary["blocking_reason_codes"]
    assert "batch_transition_id_not_in_stage21_2" in summary["blocking_reason_codes"]


def test_stage23_3_routes_batch_repair_when_transition_reward_missing_from_batch(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke as s23

    _patch_stage21_runs(
        monkeypatch,
        s23,
        extra_transition_id="s1:step-1:sample-2",
        extra_reward_transition_id="s1:step-1:sample-2",
    )
    config = _write_config(tmp_path, _write_stage23_2_root(tmp_path))

    summary = s23.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage23_3_batch_gate"
    assert "transition_missing_batch" in summary["blocking_reason_codes"]
    assert "reward_missing_batch" in summary["blocking_reason_codes"]


def test_stage23_3_routes_collector_repair_when_stage21_1_loaded_wrong_high_res_root(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke as s23

    wrong_root = tmp_path / "wrong_high_res_root"
    wrong_root.mkdir()
    _patch_stage21_runs(monkeypatch, s23, manifest_source_roi_expansion_root=str(wrong_root))
    config = _write_config(tmp_path, _write_stage23_2_root(tmp_path))

    summary = s23.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage23_3_collector_slope_theta_transition_contract"
    assert summary["stage21_1_source_roi_expansion_root_match"] is False
    assert "stage21_1_source_roi_expansion_root_mismatch" in summary["blocking_reason_codes"]


def test_stage23_3_rejects_unready_stage23_2_without_running_substages(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke as s23

    _patch_stage21_runs(monkeypatch, s23)
    stage23_2 = _write_stage23_2_root(tmp_path, next_required_change="repair_stage23_2_reward_provenance")
    config = _write_config(tmp_path, stage23_2)

    summary = s23.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage23_3_required_inputs"
    assert "stage23_2_route_not_stage23_3" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s21_1").exists()


def test_stage23_3_boundary_flags_hard_fail(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke as s23

    _patch_stage21_runs(monkeypatch, s23)
    config = _write_config(tmp_path, _write_stage23_2_root(tmp_path), publishes_checkpoint=True)

    summary = s23.run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage23_3_boundary_rejections"
    assert "publishes_checkpoint" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s21_1").exists()


def _patch_stage21_runs(
    monkeypatch,
    s23,
    *,
    transition_mode: str = "valid",
    reward_mode: str = "valid",
    reward_transition_id: str = "s1:step-0:sample-1",
    batch_transition_id: str = "s1:step-0:sample-1",
    extra_transition_id: str | None = None,
    extra_reward_transition_id: str | None = None,
    manifest_source_roi_expansion_root: str | None = None,
) -> None:
    def fake_stage21_1(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        output_root.mkdir(parents=True, exist_ok=True)
        rows = [_transition(mode=transition_mode)]
        if extra_transition_id is not None:
            rows.append(_transition(mode=transition_mode, transition_id=extra_transition_id))
        _write_jsonl(output_root / s23.stage21_1.TRAINABLE_BATCH_FILE, rows)
        _write_jsonl(output_root / s23.stage21_1.TRANSITIONS_FILE, rows)
        _write_jsonl(output_root / s23.stage21_1.REJECTION_FILE, [])
        summary = {
            "schema_version": "xunce-stage21-1-on-policy-ppo-rollout-collector-summary/v1",
            "status": "passed",
            "next_required_change": "implement_stage21_2_coverage_first_ppo_reward_contract",
            "trainable_transition_count": len(rows),
            "mask_violation_count": 0,
            "hard_risk_violation_count": 0,
            "runs_new_ppo_update": False,
        }
        (output_root / s23.stage21_1.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        stage21_1_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        manifest = {
            "schema_version": "xunce-stage21-1-manifest/v1",
            "model_audit": {
                "source_roi_expansion_root": manifest_source_roi_expansion_root
                if manifest_source_roi_expansion_root is not None
                else stage21_1_config.get("source_roi_expansion_root")
            },
        }
        (output_root / s23.stage21_1.MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
        return summary

    def fake_stage21_2(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        output_root.mkdir(parents=True, exist_ok=True)
        rows = [_reward(mode=reward_mode, transition_id=reward_transition_id)]
        if extra_reward_transition_id is not None:
            rows.append(_reward(mode=reward_mode, transition_id=extra_reward_transition_id))
        _write_jsonl(output_root / s23.stage21_2.EVALUATION_FILE, rows)
        summary = {
            "schema_version": "xunce-stage21-2-coverage-first-reward-summary/v1",
            "status": "passed",
            "next_required_change": "implement_stage21_3_ppo_batch_validation",
            "slope_obstacle_aware_theta_reward_enabled": True,
            "slope_obstacle_reward_contract_missing_count": 0 if reward_mode == "valid" else 1,
        }
        (output_root / s23.stage21_2.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        return summary

    def fake_stage21_3(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        output_root.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_root / s23.stage21_3.BATCH_FILE, [_batch_row(transition_id=batch_transition_id)])
        summary = {
            "schema_version": "xunce-stage21-3-ppo-batch-validation-summary/v1",
            "status": "passed",
            "next_required_change": "implement_stage21_4_tiny_ppo_update_smoke",
            "trainable_transition_count": 1,
            "slope_obstacle_aware_theta_reward_contract_missing_count": 0,
        }
        (output_root / s23.stage21_3.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(s23.stage21_1, "run_xunce_stage21_1_on_policy_ppo_rollout_collector", fake_stage21_1)
    monkeypatch.setattr(s23.stage21_2, "run_xunce_stage21_2_coverage_first_ppo_reward_contract", fake_stage21_2)
    monkeypatch.setattr(s23.stage21_3, "run_xunce_stage21_3_ppo_batch_validation", fake_stage21_3)


def _transition(*, mode: str, transition_id: str = "s1:step-0:sample-1") -> dict:
    info = _slope_info()
    if mode == "missing_slope":
        info = {key: value for key, value in info.items() if not key.startswith("obstacle_aware") and key not in {"slope_obstacle_source_hash", "slope_obstacle_source_hashes", "strict_obstacle_aware_new_visible_cell_count"}}
    return {
        "schema_version": "xunce-stage21-1-ppo-transition/v1",
        "transition_id": transition_id,
        "scenario_id": "s1",
        "step_index": 0,
        "observation": {},
        "xunce_batch": {},
        "action_index": 1,
        "old_log_prob": 0.0,
        "old_value": 0.0,
        "reward": 0.1,
        "done": False,
        "trainable": True,
        "info": info,
    }


def _reward(*, mode: str, transition_id: str = "s1:step-0:sample-1") -> dict:
    row = _batch_row(transition_id=transition_id)
    if mode == "fallback":
        row["coverage_source"] = "theta_aware_sensor_footprint/v1"
        row["slope_obstacle_aware_theta_reward_contract"] = False
        row["unobstructed_theta_reward_fallback_used"] = True
    if mode == "bad_footprint":
        row["obstacle_aware_new_visible_cell_count"] = 9
        row["reward_metrics"] = {"coverage_rate_delta": 0.09}
    return row


def _batch_row(*, transition_id: str = "s1:step-0:sample-1") -> dict:
    return {
        "transition_id": transition_id,
        "action_index": 1,
        "theta_aware_reward_contract": True,
        "slope_obstacle_aware_theta_reward_contract": True,
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "candidate_viewpoint": [1, 2, 45],
        "candidate_theta_deg": 45,
        "theta_new_visible_cell_count": 7,
        "theta_coverage_hash": "clear-hash",
        "theta_coverage_gain_per_path_cost": 3.5,
        "theta_coverage_denominator_cells": 100.0,
        "obstacle_aware_new_visible_cell_count": 4,
        "obstacle_aware_theta_coverage_hash": "obstacle-hash-45",
        "obstacle_aware_theta_coverage_gain_per_path_cost": 2.0,
        "obstacle_aware_theta_coverage_denominator_cells": 100.0,
        "slope_obstacle_source_hash": "slope-source-hash",
        "platform_contract_hash": "platform-hash",
        "max_traversable_slope_deg": 30.0,
        "slope_blocked_source_kind": "slope_blocked_as_obstacle_proxy",
        "point_only_reward_fallback_used": False,
        "unobstructed_theta_reward_fallback_used": False,
        "reward_metrics": {"coverage_rate_delta": 0.04},
        "info": _slope_info(),
    }


def _slope_info() -> dict:
    return {
        "candidate_viewpoints": [[1, 2, 0], [1, 2, 45]],
        "candidate_theta_deg": [0, 45],
        "theta_new_visible_cell_counts": [3, 7],
        "theta_coverage_hashes": ["clear-hash-0", "clear-hash"],
        "theta_coverage_gain_per_path_costs": [1.5, 3.5],
        "obstacle_aware_new_visible_cell_counts": [2, 4],
        "obstacle_aware_theta_coverage_hashes": ["obstacle-hash-0", "obstacle-hash-45"],
        "obstacle_aware_theta_coverage_gain_per_path_costs": [1.0, 2.0],
        "slope_obstacle_source_hash": "slope-source-hash",
        "slope_obstacle_source_hashes": ["slope-source-hash", "slope-source-hash"],
        "platform_contract_hash": "platform-hash",
        "platform_contract_hashes": ["platform-hash", "platform-hash"],
        "max_traversable_slope_deg": 30.0,
        "slope_blocked_source_kind": "slope_blocked_as_obstacle_proxy",
        "strict_obstacle_aware_new_visible_cell_count": True,
        "selected_viewpoint": [1, 2, 45],
        "selected_theta_deg": 45,
        "selected_cell": [1, 2],
        "action_mask": [True, True],
        "sampling_mask": [True, True],
        "hard_risk_clean_mask": [True, True],
        "hard_risk_violation": False,
    }


def _write_config(tmp_path: Path, stage23_2_root: Path, **overrides) -> Path:
    payload = {
        "schema_version": "xunce-stage23-3-slope-obstacle-aware-theta-ppo-collector-smoke-config/v1",
        "stage23_2_root": str(stage23_2_root),
        "stage21_1_base_config": "configs/xunce_stage21_1_on_policy_ppo_rollout_collector_v1.json",
        "stage21_2_base_config": "configs/xunce_stage21_2_coverage_first_ppo_reward_contract_v1.json",
        "stage21_3_base_config": "configs/xunce_stage21_3_ppo_batch_validation_v1.json",
        "coverage_first_reward_profile": "configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json",
        "required_scenario_count": 2,
        "rollout_steps": 4,
        "dynamic_max_candidates_per_step": 36,
        "dynamic_proposal_pool_limit_per_step": 288,
        "min_trainable_transition_count": 1,
        "theta_bin_count": 8,
        "theta_step_deg": 45,
        "sensor_fov_deg": 90.0,
        "sensor_range_cells": 3,
        "theta_coverage_denominator_cells": 100.0,
        "max_traversable_slope_deg": 30.0,
        "stage23_3_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage23_3_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_stage23_2_root(
    tmp_path: Path,
    *,
    status: str = "passed",
    next_required_change: str = "run_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke",
) -> Path:
    stage23_2b = tmp_path / "stage23_2b"
    high_res = tmp_path / "high_res_roi_expansion"
    stage23_2b.mkdir()
    high_res.mkdir()
    (stage23_2b / "xunce-stage23-2b-rerun-stage23-2a-summary.json").write_text(
        json.dumps(
            {
                "schema_version": "xunce-stage23-2a-summary/v1",
                "status": "passed",
                "high_res_root": str(high_res),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    root = tmp_path / "stage23_2"
    root.mkdir()
    summary = {
        "schema_version": "xunce-stage23-2-summary/v1",
        "status": status,
        "next_required_change": next_required_change,
        "max_traversable_slope_deg": 30.0,
        "platform_contract_id": "agilex_scout_mini_piper",
        "platform_contract_hash": "platform-hash",
        "stage23_2b_root": str(stage23_2b),
        "slope_obstacle_reward_contract_missing_count": 0,
        "point_only_reward_fallback_used_count": 0,
        "unobstructed_theta_reward_fallback_used_count": 0,
    }
    (root / "xunce-stage23-2-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return root


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
