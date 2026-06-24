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


def test_stage22_3_runs_theta_collector_reward_batch_smoke(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage22_3_theta_aware_ppo_collector_smoke as s22

    _patch_stage21_runs(monkeypatch, s22)
    config = _write_config(tmp_path, _write_stage22_2_root(tmp_path))

    summary = s22.run_xunce_stage22_3_theta_aware_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    audit = json.loads((tmp_path / "out" / "xunce-stage22-3-theta-transition-contract-audit.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage22_4_theta_aware_ppo_update_smoke"
    assert summary["trainable_transition_count"] == 1
    assert summary["theta_transition_contract_missing_count"] == 0
    assert summary["theta_reward_contract_missing_count"] == 0
    assert summary["theta_batch_reward_contract_missing_count"] == 0
    assert audit["stage21_3_rejects_point_only_reward_batch"] is True
    assert summary["runs_new_ppo_update"] is False
    assert summary["publishes_checkpoint"] is False
    assert (tmp_path / "out" / "s21_1" / "xunce-stage21-1-ppo-trainable-batch.jsonl").is_file()
    stage21_1_config = json.loads((tmp_path / "out" / "xunce-stage22-3-stage21-1-config.json").read_text(encoding="utf-8"))
    stage21_2_config = json.loads((tmp_path / "out" / "xunce-stage22-3-stage21-2-config.json").read_text(encoding="utf-8"))
    stage21_3_config = json.loads((tmp_path / "out" / "xunce-stage22-3-stage21-3-config.json").read_text(encoding="utf-8"))
    assert stage21_1_config["theta_aware_candidate_viewpoints_enabled"] is True
    assert stage21_1_config["theta_bin_count"] == 8
    assert stage21_1_config["sensor_fov_deg"] == 90.0
    assert stage21_2_config["require_theta_aware_reward_contract"] is True
    assert stage21_2_config["theta_coverage_denominator_cells"] == 1.0
    assert stage21_3_config["require_theta_aware_viewpoint_contract"] is True
    assert stage21_3_config["require_theta_aware_reward_contract"] is True


def test_stage22_3_routes_collector_repair_when_theta_fields_missing(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage22_3_theta_aware_ppo_collector_smoke as s22

    _patch_stage21_runs(monkeypatch, s22, transition_mode="missing_theta")
    config = _write_config(tmp_path, _write_stage22_2_root(tmp_path))

    summary = s22.run_xunce_stage22_3_theta_aware_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage22_3_collector_theta_transition_contract"
    assert "theta_transition_contract_missing" in summary["blocking_reason_codes"]


def test_stage22_3_routes_binding_repair_when_action_viewpoint_mismatches(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage22_3_theta_aware_ppo_collector_smoke as s22

    _patch_stage21_runs(monkeypatch, s22, transition_mode="bad_binding")
    config = _write_config(tmp_path, _write_stage22_2_root(tmp_path))

    summary = s22.run_xunce_stage22_3_theta_aware_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage22_3_action_viewpoint_binding"
    assert "action_viewpoint_binding_mismatch" in summary["blocking_reason_codes"]


def test_stage22_3_routes_binding_repair_when_sampling_mask_contract_mismatches(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage22_3_theta_aware_ppo_collector_smoke as s22

    _patch_stage21_runs(monkeypatch, s22, transition_mode="bad_sampling_contract")
    config = _write_config(tmp_path, _write_stage22_2_root(tmp_path))

    summary = s22.run_xunce_stage22_3_theta_aware_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage22_3_action_viewpoint_binding"
    assert summary["sampling_mask_contract_mismatch_count"] == 1
    assert "sampling_mask_contract_mismatch" in summary["blocking_reason_codes"]


def test_stage22_3_routes_binding_repair_when_selected_mask_index_is_out_of_range(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage22_3_theta_aware_ppo_collector_smoke as s22

    _patch_stage21_runs(monkeypatch, s22, transition_mode="short_selected_mask")
    config = _write_config(tmp_path, _write_stage22_2_root(tmp_path))

    summary = s22.run_xunce_stage22_3_theta_aware_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage22_3_action_viewpoint_binding"
    assert summary["mask_length_mismatch_count"] == 1
    assert summary["mask_violation_count"] == 3


def test_stage22_3_routes_reward_repair_when_point_only_fallback_used(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage22_3_theta_aware_ppo_collector_smoke as s22

    _patch_stage21_runs(monkeypatch, s22, reward_mode="point_only_fallback")
    config = _write_config(tmp_path, _write_stage22_2_root(tmp_path))

    summary = s22.run_xunce_stage22_3_theta_aware_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage22_3_theta_reward_provenance"
    assert "point_only_reward_fallback_used" in summary["blocking_reason_codes"]


def test_stage22_3_routes_reward_repair_when_reward_viewpoint_mismatches(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage22_3_theta_aware_ppo_collector_smoke as s22

    _patch_stage21_runs(monkeypatch, s22, reward_mode="bad_viewpoint")
    config = _write_config(tmp_path, _write_stage22_2_root(tmp_path))

    summary = s22.run_xunce_stage22_3_theta_aware_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage22_3_theta_reward_provenance"
    assert "reward_viewpoint_binding_mismatch" in summary["blocking_reason_codes"]


def test_stage22_3_routes_reward_repair_when_reward_footprint_mismatches_transition(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage22_3_theta_aware_ppo_collector_smoke as s22

    _patch_stage21_runs(monkeypatch, s22, reward_mode="bad_footprint")
    config = _write_config(tmp_path, _write_stage22_2_root(tmp_path))

    summary = s22.run_xunce_stage22_3_theta_aware_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage22_3_theta_reward_provenance"
    assert summary["reward_footprint_binding_mismatch_count"] == 3
    assert "reward_footprint_binding_mismatch" in summary["blocking_reason_codes"]


def test_stage22_3_routes_reward_repair_when_reward_delta_is_inconsistent(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage22_3_theta_aware_ppo_collector_smoke as s22

    _patch_stage21_runs(monkeypatch, s22, reward_mode="bad_delta")
    config = _write_config(tmp_path, _write_stage22_2_root(tmp_path))

    summary = s22.run_xunce_stage22_3_theta_aware_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage22_3_theta_reward_provenance"
    assert "theta_reward_contract_missing" in summary["blocking_reason_codes"]


def test_stage22_3_routes_collector_repair_when_substage_passes_with_zero_rows(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage22_3_theta_aware_ppo_collector_smoke as s22

    _patch_stage21_runs(monkeypatch, s22, empty_stage21_1=True)
    config = _write_config(tmp_path, _write_stage22_2_root(tmp_path))

    summary = s22.run_xunce_stage22_3_theta_aware_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage22_3_collector_theta_transition_contract"
    assert "stage21_1_transition_rows_missing" in summary["blocking_reason_codes"]


def test_stage22_3_rejects_unready_stage22_2_without_running_substages(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage22_3_theta_aware_ppo_collector_smoke as s22

    _patch_stage21_runs(monkeypatch, s22)
    stage22_2 = _write_stage22_2_root(tmp_path, next_required_change="repair_stage22_2_theta_reward_provenance")
    config = _write_config(tmp_path, stage22_2)

    summary = s22.run_xunce_stage22_3_theta_aware_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage22_3_required_inputs"
    assert "stage22_2_route_not_stage22_3" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s21_1").exists()


def test_stage22_3_boundary_flags_hard_fail(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage22_3_theta_aware_ppo_collector_smoke as s22

    _patch_stage21_runs(monkeypatch, s22)
    config = _write_config(tmp_path, _write_stage22_2_root(tmp_path), publishes_checkpoint=True)

    summary = s22.run_xunce_stage22_3_theta_aware_ppo_collector_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage22_3_boundary_rejections"
    assert "publishes_checkpoint" in summary["blocking_reason_codes"]
    assert not (tmp_path / "out" / "s21_1").exists()


def _patch_stage21_runs(
    monkeypatch,
    s22,
    *,
    transition_mode: str = "valid",
    reward_mode: str = "valid",
    empty_stage21_1: bool = False,
) -> None:
    def fake_stage21_1(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        output_root.mkdir(parents=True, exist_ok=True)
        transition = _transition(mode=transition_mode)
        rows = [] if empty_stage21_1 else [transition]
        _write_jsonl(output_root / s22.stage21_1.TRAINABLE_BATCH_FILE, rows)
        _write_jsonl(output_root / s22.stage21_1.TRANSITIONS_FILE, rows)
        _write_jsonl(output_root / s22.stage21_1.REJECTION_FILE, [])
        summary = {
            "schema_version": "xunce-stage21-1-on-policy-ppo-rollout-collector-summary/v1",
            "status": "passed",
            "next_required_change": "implement_stage21_2_coverage_first_ppo_reward_contract",
            "trainable_transition_count": 1,
            "mask_violation_count": 0,
            "hard_risk_violation_count": 0,
            "runs_new_ppo_update": False,
        }
        (output_root / s22.stage21_1.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        return summary

    def fake_stage21_2(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        output_root.mkdir(parents=True, exist_ok=True)
        reward = _reward_row(
            point_only_fallback=reward_mode == "point_only_fallback",
            bad_viewpoint=reward_mode == "bad_viewpoint",
            bad_delta=reward_mode == "bad_delta",
            bad_footprint=reward_mode == "bad_footprint",
        )
        _write_jsonl(output_root / s22.stage21_2.EVALUATION_FILE, [reward])
        summary = {
            "schema_version": "xunce-stage21-2-coverage-first-reward-summary/v1",
            "status": "passed",
            "next_required_change": "implement_stage21_3_ppo_batch_validation",
            "theta_aware_reward_contract_required": True,
            "theta_aware_reward_contract_count": 0 if reward_mode == "point_only_fallback" else 1,
            "point_only_reward_fallback_used_count": 1 if reward_mode == "point_only_fallback" else 0,
        }
        (output_root / s22.stage21_2.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        return summary

    def fake_stage21_3(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        output_root.mkdir(parents=True, exist_ok=True)
        batch = _batch_row()
        _write_jsonl(output_root / s22.stage21_3.BATCH_FILE, [batch])
        summary = {
            "schema_version": "xunce-stage21-3-ppo-batch-validation-summary/v1",
            "status": "passed",
            "next_required_change": "implement_stage21_4_tiny_ppo_update_smoke",
            "trainable_transition_count": 1,
            "theta_aware_viewpoint_contract_required": True,
            "theta_aware_reward_contract_required": True,
            "theta_aware_viewpoint_contract_missing_count": 0,
            "theta_aware_reward_contract_missing_count": 0,
        }
        (output_root / s22.stage21_3.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(s22.stage21_1, "run_xunce_stage21_1_on_policy_ppo_rollout_collector", fake_stage21_1)
    monkeypatch.setattr(s22.stage21_2, "run_xunce_stage21_2_coverage_first_ppo_reward_contract", fake_stage21_2)
    monkeypatch.setattr(s22.stage21_3, "run_xunce_stage21_3_ppo_batch_validation", fake_stage21_3)


def _transition(*, mode: str) -> dict:
    info = {
        "candidate_viewpoints": [[1, 2, 0], [1, 2, 45]],
        "candidate_theta_deg": [0, 45],
        "theta_new_visible_cell_counts": [3, 4],
        "theta_coverage_hashes": ["h0", "h45"],
        "theta_coverage_gain_per_path_costs": [1.5, 2.0],
        "selected_viewpoint": [1, 2, 45],
        "selected_theta_deg": 45,
        "selected_cell": [1, 2],
        "action_mask": [True, True],
        "sampling_mask": [True, True],
        "hard_risk_clean_mask": [True, True],
        "hard_risk_violation": False,
    }
    if mode == "missing_theta":
        info.pop("candidate_viewpoints")
    if mode == "bad_binding":
        info["selected_viewpoint"] = [9, 9, 90]
    if mode == "bad_sampling_contract":
        info["hard_risk_clean_mask"] = [False, True]
        info["sampling_mask"] = [True, True]
    if mode == "short_selected_mask":
        info["action_mask"] = [True]
        info["sampling_mask"] = [True]
        info["hard_risk_clean_mask"] = [True]
    return {
        "schema_version": "xunce-stage21-1-ppo-transition/v1",
        "transition_id": "s1:step-0:sample-1",
        "scenario_id": "s1",
        "step_index": 0,
        "action_index": 1,
        "old_log_prob": -0.1,
        "old_value": 0.0,
        "reward": 1.0,
        "done": True,
        "trainable": True,
        "info": info,
    }


def _reward_row(
    *,
    point_only_fallback: bool = False,
    bad_viewpoint: bool = False,
    bad_delta: bool = False,
    bad_footprint: bool = False,
) -> dict:
    viewpoint = [9, 9, 90] if bad_viewpoint else [1, 2, 45]
    theta = 90 if bad_viewpoint else 45
    theta_new_visible = 5 if bad_footprint else 4
    theta_hash = "wrong-hash" if bad_footprint else "h45"
    theta_gain = 9.0 if bad_footprint else 2.0
    metrics = {"coverage_rate_delta": 0.99 if bad_delta else theta_new_visible / 10.0}
    return {
        "transition_id": "s1:step-0:sample-1",
        "scenario_id": "s1",
        "step_index": 0,
        "reward": 1.0,
        "trainable": True,
        "theta_aware_reward_contract": not point_only_fallback,
        "coverage_source": None if point_only_fallback else "theta_aware_sensor_footprint/v1",
        "candidate_viewpoint": viewpoint,
        "candidate_theta_deg": theta,
        "theta_new_visible_cell_count": theta_new_visible,
        "theta_coverage_hash": theta_hash,
        "theta_coverage_gain_per_path_cost": theta_gain,
        "theta_coverage_denominator_cells": 10.0,
        "point_only_reward_fallback_used": point_only_fallback,
        "metrics": metrics,
    }


def _batch_row() -> dict:
    row = _reward_row()
    row.update(
        {
            "action_index": 1,
            "info": _transition(mode="valid")["info"],
            "reward_metrics": {"coverage_rate_delta": 0.4},
        }
    )
    return row


def _write_stage22_2_root(
    tmp_path: Path,
    *,
    status: str = "passed",
    next_required_change: str = "run_stage22_3_theta_aware_ppo_collector_smoke",
) -> Path:
    root = tmp_path / "stage22_2"
    root.mkdir()
    summary = {
        "schema_version": "xunce-stage22-2-summary/v1",
        "status": status,
        "next_required_change": next_required_change,
        "theta_reward_contract_missing_count": 0,
        "point_only_reward_fallback_used_count": 0,
        "coverage_diff_but_reward_equal_group_count": 0,
    }
    (root / "xunce-stage22-2-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return root


def _write_config(tmp_path: Path, stage22_2: Path, **overrides) -> Path:
    payload = {
        "schema_version": "xunce-stage22-3-theta-aware-ppo-collector-smoke-config/v1",
        "stage22_2_root": str(stage22_2),
        "stage21_1_base_config": "configs/xunce_stage21_1_on_policy_ppo_rollout_collector_v1.json",
        "stage21_2_base_config": "configs/xunce_stage21_2_coverage_first_ppo_reward_contract_v1.json",
        "stage21_3_base_config": "configs/xunce_stage21_3_ppo_batch_validation_v1.json",
        "coverage_first_reward_profile": "configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json",
        "stage22_3_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage22_3_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
