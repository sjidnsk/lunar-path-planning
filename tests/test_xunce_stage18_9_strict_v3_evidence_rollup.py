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


SWEEPS = ((6, 48), (12, 96), (24, 192), (36, 288))


def _profile():
    from model_explorer.policy.canonical_reward import load_canonical_reward_profile

    return load_canonical_reward_profile(REPO_ROOT / "configs" / "xunce_canonical_reward_guard_profile_v3.json")


def _run(config_path: Path, output_root: Path):
    from scripts.run_xunce_stage18_9_strict_v3_evidence_rollup import (
        run_xunce_stage18_9_strict_v3_evidence_rollup,
    )

    return run_xunce_stage18_9_strict_v3_evidence_rollup(
        config_path=config_path,
        output_root=output_root,
        repo_root=REPO_ROOT,
    )


def test_rollup_routes_to_preflight_when_any_count_trajectory_guard_passes(tmp_path: Path) -> None:
    profile = _profile()
    config_path = _write_config(tmp_path, profile)
    _write_stage18_7(tmp_path, profile)
    for count, pool in SWEEPS:
        _write_stage18_9(
            tmp_path,
            profile,
            count=count,
            pool=pool,
            trajectory_guard_passed=count == 36,
            coverage_delta_cells=12.0 if count == 36 else 5.0,
            path_cost_delta_m=8.0 if count == 36 else 24.0,
            coverage_per_100m_delta=1.0 if count == 36 else -0.5,
            soft_risk_exposure_delta=4.0,
        )

    summary = _run(config_path, tmp_path / "out")

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "prepare_stage19_evaluator_critic_preflight"
    assert summary["stage19_authorized"] is False
    assert summary["best_candidate_count"] == 36
    assert summary["trajectory_guard_passed_count"] == 1
    assert (tmp_path / "out" / "xunce-stage18-9-strict-v3-evidence-summary.json").is_file()
    rows = (tmp_path / "out" / "xunce-stage18-9-strict-v3-count-results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 4


def test_rollup_rejects_missing_or_non_v3_inputs(tmp_path: Path) -> None:
    profile = _profile()
    config_path = _write_config(tmp_path, profile)
    _write_stage18_7(tmp_path, profile)
    for count, pool in SWEEPS[:-1]:
        _write_stage18_9(tmp_path, profile, count=count, pool=pool)
    _write_stage18_9(tmp_path, profile, count=36, pool=288, profile_version="v2")

    summary = _run(config_path, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage18_9_strict_v3_required_inputs"
    assert "profile_lineage_mismatch" in summary["blocking_reason_codes"]


def test_rollup_routes_hard_risk_before_reward_refinement(tmp_path: Path) -> None:
    profile = _profile()
    config_path = _write_config(tmp_path, profile)
    _write_stage18_7(tmp_path, profile)
    for count, pool in SWEEPS:
        _write_stage18_9(
            tmp_path,
            profile,
            count=count,
            pool=pool,
            hard_risk_violation_count=1 if count == 24 else 0,
            trajectory_guard_passed=False,
        )

    summary = _run(config_path, tmp_path / "out")

    assert summary["next_required_change"] == "repair_path_risk_boundary_filtering"
    assert "hard_risk_violation" in summary["blocking_reason_codes"]


def test_rollup_boundary_flags_hard_fail(tmp_path: Path) -> None:
    profile = _profile()
    config_path = _write_config(tmp_path, profile)
    _write_stage18_7(tmp_path, profile)
    for count, pool in SWEEPS:
        _write_stage18_9(tmp_path, profile, count=count, pool=pool, trajectory_guard_passed=True)
    summary_path = tmp_path / "count_006" / "stage18_9" / "xunce-stage18-9-trajectory-risk-reward-summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["starts_online_canary"] = True
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = _run(config_path, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage18_9_strict_v3_boundary_rejections"
    assert "starts_online_canary" in summary["blocking_reason_codes"]


def test_rollup_rejects_nested_stage19_authorization(tmp_path: Path) -> None:
    profile = _profile()
    config_path = _write_config(tmp_path, profile)
    _write_stage18_7(tmp_path, profile)
    for count, pool in SWEEPS:
        _write_stage18_9(tmp_path, profile, count=count, pool=pool, trajectory_guard_passed=True)
    summary_path = tmp_path / "count_012" / "stage18_9" / "xunce-stage18-9-trajectory-risk-reward-summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["stage19_readiness"]["authorized"] = True
    payload["next_stage_routing"]["stage19_authorized"] = True
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = _run(config_path, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage18_9_strict_v3_boundary_rejections"
    assert "stage19_authorized_not_false" in summary["blocking_reason_codes"]


def test_rollup_rejects_preflight_with_missing_observed_metrics(tmp_path: Path) -> None:
    profile = _profile()
    config_path = _write_config(tmp_path, profile)
    _write_stage18_7(tmp_path, profile)
    for count, pool in SWEEPS:
        _write_stage18_9(tmp_path, profile, count=count, pool=pool, trajectory_guard_passed=count == 36)
    summary_path = tmp_path / "count_036" / "stage18_9" / "xunce-stage18-9-trajectory-risk-reward-summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["trajectory_guard_summary"].pop("observed")
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = _run(config_path, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage18_9_strict_v3_required_inputs"
    assert "missing_observed_metrics" in summary["blocking_reason_codes"]
    assert summary["trajectory_guard_passed_count"] == 0


def _write_config(tmp_path: Path, profile) -> Path:
    config = {
        "schema_version": "xunce-stage18-9-strict-v3-evidence-rollup-config/v1",
        "artifact_workspace_root": str(tmp_path),
        "canonical_reward_profile": str(REPO_ROOT / "configs" / "xunce_canonical_reward_guard_profile_v3.json"),
        "stage18_7_candidate_count_scaling_root": str(tmp_path / "stage18_7"),
        "counts": [
            {
                "candidate_count": count,
                "proposal_pool_limit": pool,
                "stage18_9_root": str(tmp_path / f"count_{count:03d}" / "stage18_9"),
            }
            for count, pool in SWEEPS
        ],
        "canary_traffic_fraction": 0.0,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "default_policy_replacement_approved": False,
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
    }
    path = tmp_path / "rollup-config.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_stage18_7(tmp_path: Path, profile) -> None:
    root = tmp_path / "stage18_7"
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "xunce-stage18-7-candidate-count-scaling-summary/v1",
        "status": "passed",
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
        "stage19_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
    }
    (root / "xunce-stage18-7-candidate-count-scaling-summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_stage18_9(
    tmp_path: Path,
    profile,
    *,
    count: int,
    pool: int,
    trajectory_guard_passed: bool = False,
    hard_risk_violation_count: int = 0,
    coverage_delta_cells: float = 5.0,
    path_cost_delta_m: float = 25.0,
    coverage_per_100m_delta: float = -1.0,
    soft_risk_exposure_delta: float = 2.0,
    profile_version: str | None = None,
) -> None:
    root = tmp_path / f"count_{count:03d}" / "stage18_9"
    root.mkdir(parents=True, exist_ok=True)
    route = "prepare_stage19_evaluator_critic_preflight" if trajectory_guard_passed else "refine_coverage_cost_reward_weights"
    payload = {
        "schema_version": "xunce-stage18-9-trajectory-risk-reward-summary/v1",
        "status": "passed" if trajectory_guard_passed else "failed",
        "profile_id": profile.profile_id,
        "profile_version": profile_version or profile.profile_version,
        "profile_hash": profile.profile_hash,
        "trajectory_guard_passed": trajectory_guard_passed,
        "path_risk_boundary_summary": {
            "path_risk_boundary_passed": hard_risk_violation_count == 0,
            "hard_risk_violation_count": hard_risk_violation_count,
        },
        "trajectory_guard_summary": {
            "observed": {
                "coverage_delta_cells": coverage_delta_cells,
                "path_cost_delta_m": path_cost_delta_m,
                "coverage_per_100m_delta": coverage_per_100m_delta,
                "soft_risk_exposure_delta": soft_risk_exposure_delta,
            },
            "coverage_advantage_established": coverage_delta_cells >= 1.0,
            "path_cost_budget_passed": path_cost_delta_m <= 20.0,
            "coverage_efficiency_passed": coverage_per_100m_delta >= 0.0,
            "soft_risk_exposure_passed": soft_risk_exposure_delta <= 25.0,
        },
        "next_required_change": route,
        "next_stage_routing": {
            "schema_version": "xunce-stage18-9-next-stage-routing/v1",
            "primary_route": route,
            "stage19_authorized": False,
        },
        "stage19_authorized": False,
        "stage19_readiness": {
            "schema_version": "xunce-stage18-9-stage19-readiness/v1",
            "readiness": "ready_for_stage19_preflight_human_review_only" if trajectory_guard_passed else "not_authorized",
            "authorized": False,
            "trajectory_guard_passed": trajectory_guard_passed,
        },
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    (root / "xunce-stage18-9-trajectory-risk-reward-summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
