import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
MODEL_EXPLORER_SRC = str(REPO_ROOT / "model-explorer" / "src")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
if MODEL_EXPLORER_SRC not in sys.path:
    sys.path.insert(0, MODEL_EXPLORER_SRC)


def _load_profile():
    from model_explorer.policy.canonical_reward import load_canonical_reward_profile

    return load_canonical_reward_profile(REPO_ROOT / "configs" / "xunce_canonical_reward_guard_profile_v3.json")


def _run_audit(config_path: Path, output_root: Path):
    from scripts.run_xunce_stage18_9_trajectory_risk_boundary_reward_audit import (
        run_xunce_stage18_9_trajectory_risk_boundary_reward_audit,
    )

    return run_xunce_stage18_9_trajectory_risk_boundary_reward_audit(
        config_path=config_path,
        output_root=output_root,
        repo_root=REPO_ROOT,
    )


def test_hard_risk_violation_blocks(tmp_path: Path) -> None:
    config = _write_stage18_9_fixture(tmp_path, hard_risk_violation_count=1)

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["trajectory_gate_passed"] is False
    assert summary["next_required_change"] == "repair_path_risk_boundary_filtering"
    assert "hard_risk_violation" in summary["blocking_reason_codes"]
    assert summary["stage19_authorized"] is False


def test_coverage_win_cost_fail_routes_reward_weights(tmp_path: Path) -> None:
    config = _write_stage18_9_fixture(tmp_path, path_cost_delta_m=25.0)

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["trajectory_gate_passed"] is False
    assert summary["next_required_change"] == "refine_coverage_cost_reward_weights"
    assert "path_cost_budget_exceeded" in summary["blocking_reason_codes"]
    assert summary["stage19_authorized"] is False


def test_soft_risk_exposure_fail_routes_calibration(tmp_path: Path) -> None:
    config = _write_stage18_9_fixture(tmp_path, soft_risk_exposure_delta=26.0)

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["trajectory_gate_passed"] is False
    assert summary["next_required_change"] == "calibrate_soft_risk_exposure_weight"
    assert "soft_risk_exposure_budget_exceeded" in summary["blocking_reason_codes"]
    assert summary["stage19_authorized"] is False


def test_clean_trajectory_preflight_but_authorized_false(tmp_path: Path) -> None:
    config = _write_stage18_9_fixture(tmp_path)

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "passed"
    assert summary["trajectory_gate_passed"] is True
    assert summary["next_required_change"] == "prepare_stage19_evaluator_critic_preflight"
    assert summary["stage19_authorized"] is False
    assert summary["next_stage_routing"]["stage19_authorized"] is False
    assert (tmp_path / "out" / "xunce-stage18-9-trajectory-risk-reward-summary.json").is_file()
    assert (tmp_path / "out" / "xunce-stage18-9-scenario-trajectory-audit.jsonl").is_file()
    assert (tmp_path / "out" / "xunce-stage18-9-path-risk-boundary-summary.json").is_file()
    assert (tmp_path / "out" / "xunce-stage18-9-reward-profile-v3-evaluation.json").is_file()
    assert (tmp_path / "out" / "xunce-stage18-9-next-stage-routing.json").is_file()
    assert (tmp_path / "out" / "xunce-stage18-9-report.md").is_file()
    assert (tmp_path / "out" / "xunce-stage18-9-manifest.json").is_file()


def test_candidate_diagnostic_cannot_override_trajectory_fail(tmp_path: Path) -> None:
    config = _write_stage18_9_fixture(
        tmp_path,
        path_cost_delta_m=30.0,
        stage18_7_overrides={"best_xunce_selected_guard_clean_rate": 1.0},
    )

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "refine_coverage_cost_reward_weights"
    assert summary["candidate_diagnostics"]["best_xunce_selected_guard_clean_rate"] == 1.0
    assert summary["candidate_diagnostics"]["diagnostic_only"] is True


def test_profile_hash_mismatch_fails(tmp_path: Path) -> None:
    config = _write_stage18_9_fixture(tmp_path, profile_hash="different-profile-hash")

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_required_stage18_9_inputs"
    assert "profile_hash_mismatch" in summary["blocking_reason_codes"]
    assert summary["stage19_authorized"] is False


def test_missing_profile_lineage_fails_instead_of_rebranding_as_v3(tmp_path: Path) -> None:
    config = _write_stage18_9_fixture(tmp_path)
    coverage_root = tmp_path / "stage18_4e"
    for filename in (
        "xunce-exploration-coverage-comparison-summary.json",
        "xunce-exploration-coverage-comparison-aggregate.json",
        "xunce-exploration-coverage-comparison-manifest.json",
    ):
        path = coverage_root / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        for field in ("profile_id", "profile_version", "profile_hash"):
            payload.pop(field, None)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pairs_path = coverage_root / "xunce-exploration-coverage-comparison-pairs.jsonl"
    pair = json.loads(pairs_path.read_text(encoding="utf-8").splitlines()[0])
    for field in ("profile_id", "profile_version", "profile_hash"):
        pair.pop(field, None)
    pairs_path.write_text(json.dumps(pair, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = _run_audit(config, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_required_stage18_9_inputs"
    assert "missing_profile_lineage" in summary["blocking_reason_codes"]


def _write_stage18_9_fixture(
    base: Path,
    *,
    coverage_delta_cells: float = 12.0,
    path_cost_delta_m: float = 4.0,
    coverage_per_100m_delta: float = 0.2,
    soft_risk_exposure_delta: float = 0.2,
    hard_risk_violation_count: int = 0,
    profile_hash: str | None = None,
    stage18_7_overrides: dict | None = None,
) -> Path:
    profile = _load_profile()
    coverage_root = base / "stage18_4e"
    stage18_7_root = base / "stage18_7"
    coverage_root.mkdir(parents=True, exist_ok=True)
    stage18_7_root.mkdir(parents=True, exist_ok=True)
    actual_profile_hash = profile_hash or profile.profile_hash
    common = {
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": actual_profile_hash,
        "canary_traffic_fraction": 0.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "runs_new_ppo_update": False,
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
        "default_policy_replacement_approved": False,
    }
    summary = {
        "schema_version": "xunce-exploration-coverage-comparison-summary/v1",
        "status": "passed",
        "coverage_delta_cells": coverage_delta_cells,
        "path_cost_delta_m": path_cost_delta_m,
        "coverage_per_100m_delta": coverage_per_100m_delta,
        "soft_risk_exposure_delta": soft_risk_exposure_delta,
        "hard_risk_violation_count": hard_risk_violation_count,
        **common,
    }
    aggregate = {
        "schema_version": "xunce-exploration-coverage-comparison-aggregate/v1",
        "coverage_delta_cells_mean": coverage_delta_cells,
        "path_cost_delta_m_mean": path_cost_delta_m,
        "coverage_per_100m_delta_mean": coverage_per_100m_delta,
        "soft_risk_exposure_delta_mean": soft_risk_exposure_delta,
        "hard_risk_violation_count": hard_risk_violation_count,
        **common,
    }
    pair = {
        "schema_version": "xunce-exploration-coverage-comparison-pair/v1",
        "scenario_id": "s0",
        "coverage_delta_cells": coverage_delta_cells,
        "path_cost_delta_m": path_cost_delta_m,
        "coverage_per_100m_delta": coverage_per_100m_delta,
        "soft_risk_exposure_delta": soft_risk_exposure_delta,
        "hard_risk_violation_count": hard_risk_violation_count,
        **common,
    }
    episode = {
        "schema_version": "xunce-exploration-coverage-episode/v1",
        "scenario_id": "s0",
        "hard_risk_violation_count": hard_risk_violation_count,
        "soft_risk_exposure_delta": soft_risk_exposure_delta,
        **common,
    }
    manifest = {
        "schema_version": "xunce-exploration-coverage-comparison-manifest/v1",
        "summary_status": "passed",
        "normalized_config": {"emit_candidate_metric_audit": True},
        **common,
    }
    _write_json(coverage_root / "xunce-exploration-coverage-comparison-summary.json", summary)
    _write_json(coverage_root / "xunce-exploration-coverage-comparison-aggregate.json", aggregate)
    _write_jsonl(coverage_root / "xunce-exploration-coverage-comparison-pairs.jsonl", [pair])
    _write_jsonl(coverage_root / "xunce-exploration-coverage-episodes.jsonl", [episode])
    _write_json(coverage_root / "xunce-exploration-coverage-comparison-manifest.json", manifest)
    stage18_7 = {
        "schema_version": "xunce-stage18-7-candidate-count-scaling-summary/v1",
        "status": "passed",
        "best_xunce_selected_guard_clean_rate": 0.5,
        "stage19_authorized": False,
        "next_required_change": "prepare_stage19_evaluator_critic_preflight",
        **common,
    }
    stage18_7.update(stage18_7_overrides or {})
    _write_json(stage18_7_root / "xunce-stage18-7-candidate-count-scaling-summary.json", stage18_7)
    config = {
        "schema_version": "xunce-stage18-9-trajectory-risk-boundary-reward-audit-config/v1",
        "coverage_comparison_root": str(coverage_root),
        "stage18_7_candidate_count_scaling_root": str(stage18_7_root),
        "canonical_reward_profile": str(REPO_ROOT / "configs" / "xunce_canonical_reward_guard_profile_v3.json"),
        "canary_traffic_fraction": 0.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "runs_new_ppo_update": False,
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
        "default_policy_replacement_approved": False,
    }
    config_path = base / "stage18_9_config.json"
    _write_json(config_path, config)
    return config_path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
