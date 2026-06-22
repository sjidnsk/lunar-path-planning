import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def _run(config_path: Path, output_root: Path):
    from scripts.run_xunce_stage19_evaluator_critic_preflight import run_xunce_stage19_evaluator_critic_preflight

    return run_xunce_stage19_evaluator_critic_preflight(
        config_path=config_path,
        output_root=output_root,
        repo_root=REPO_ROOT,
        overrides={},
    )


def test_stage19_selects_36_w010_practical_target_with_capped_coverage(tmp_path: Path) -> None:
    stage18_11_root = _write_stage18_11_fixture(tmp_path)
    config = _write_config(tmp_path, stage18_11_root)

    summary = _run(config, tmp_path / "out")

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "stage20_reward_rerank_oracle_preference_dataset_preparation"
    assert summary["oracle_target_feasible"] is True
    assert summary["primary_target_selected"] is True
    assert summary["selected_candidate_count"] == 36
    assert summary["selected_path_cost_weight"] == 0.1
    assert summary["xunce_checkpoint_advantage_established"] is False
    assert summary["stage20_authorized"] is False
    assert summary["runs_new_ppo_update"] is False
    target = json.loads((tmp_path / "out" / "xunce-stage19-practical-target-selection.json").read_text(encoding="utf-8"))
    assert target["primary"]["oracle"]["final_coverage_rate_raw_mean"] > 1.0
    assert target["primary"]["oracle"]["final_coverage_rate_capped_min"] >= 0.99


def test_stage19_routes_to_continue_when_only_upper_bound_is_feasible(tmp_path: Path) -> None:
    stage18_11_root = _write_stage18_11_fixture(tmp_path, primary_coverage=0.80)
    config = _write_config(tmp_path, stage18_11_root)

    summary = _run(config, tmp_path / "out")

    assert summary["next_required_change"] == "continue_path_cost_weight_calibration_at_99pct_coverage"
    assert summary["practical_target_selection"]["upper_bound_target_feasible"] is True
    assert summary["practical_target_selection"]["primary_target_feasible"] is False


def test_stage19_uses_capped_coverage_and_routes_to_horizon_when_no_rollout_reaches_99(tmp_path: Path) -> None:
    stage18_11_root = _write_stage18_11_fixture(tmp_path, upper_coverage=0.90, primary_coverage=0.89, six_coverage=0.88)
    config = _write_config(tmp_path, stage18_11_root)

    summary = _run(config, tmp_path / "out")

    assert summary["next_required_change"] == "stage18_12_rollout_horizon_or_mission_budget_scaling_for_99pct_coverage"
    assert summary["oracle_target_feasible"] is False


def test_stage19_hard_risk_routes_to_boundary_repair(tmp_path: Path) -> None:
    stage18_11_root = _write_stage18_11_fixture(tmp_path, primary_hard_risk=1)
    config = _write_config(tmp_path, stage18_11_root)

    summary = _run(config, tmp_path / "out")

    assert summary["next_required_change"] == "repair_path_risk_boundary_filtering"


def test_stage19_missing_preference_pairs_routes_to_collect_more_evidence(tmp_path: Path) -> None:
    stage18_11_root = _write_stage18_11_fixture(tmp_path, write_steps=False)
    config = _write_config(tmp_path, stage18_11_root)

    summary = _run(config, tmp_path / "out")

    assert summary["next_required_change"] == "collect_more_reward_rerank_preference_evidence"
    assert summary["critic_target_readiness"]["critic_target_ready"] is False


def test_stage19_boundary_flag_hard_fails(tmp_path: Path) -> None:
    stage18_11_root = _write_stage18_11_fixture(tmp_path)
    config = _write_config(tmp_path, stage18_11_root)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["starts_online_canary"] = True
    config.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = _run(config, tmp_path / "out")

    assert summary["next_required_change"] == "resolve_stage19_evaluator_critic_preflight_boundary_rejections"
    assert "starts_online_canary" in summary["blocking_reason_codes"]


def _write_config(tmp_path: Path, stage18_11_root: Path) -> Path:
    config = {
        "schema_version": "xunce-stage19-evaluator-critic-preflight-config/v1",
        "stage18_11_path_cost_weight_calibration_root": str(stage18_11_root),
        "profile_id": "xunce-coverage-cost-risk-boundary-v3",
        "profile_version": "v3",
        "target_final_coverage_rate_capped": 0.99,
        "primary_candidate_count": 36,
        "primary_path_cost_weight": 0.1,
        "upper_bound_candidate_count": 36,
        "upper_bound_path_cost_weight": 0.0,
        "max_primary_path_cost_total_m_mean": 700.0,
        "max_primary_soft_risk_exposure_total_mean": 650.0,
        "max_hard_risk_violation_count": 0.0,
        "min_preference_pair_count": 1,
        "canary_traffic_fraction": 0.0,
        "stage20_authorized": False,
        "training_or_release_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "runs_new_ppo_update": False,
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
    }
    path = tmp_path / "stage19_config.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_stage18_11_fixture(
    tmp_path: Path,
    *,
    six_coverage: float = 1.05,
    upper_coverage: float = 1.08,
    primary_coverage: float = 1.04,
    primary_hard_risk: int = 0,
    write_steps: bool = True,
) -> Path:
    root = tmp_path / "stage18_11"
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        _write_rollout(tmp_path, "count_006_w000", 6, 0.0, six_coverage, 942.0, 840.0, write_steps=write_steps),
        _write_rollout(tmp_path, "count_024_w020", 24, 0.2, 0.68, 331.0, 327.0, write_steps=write_steps),
        _write_rollout(tmp_path, "count_036_w000", 36, 0.0, upper_coverage, 945.0, 840.0, write_steps=write_steps),
        _write_rollout(
            tmp_path,
            "count_036_w010",
            36,
            0.1,
            primary_coverage,
            674.0,
            634.0,
            hard_risk=primary_hard_risk,
            write_steps=write_steps,
        ),
    ]
    summary = {
        "schema_version": "xunce-stage18-11-path-cost-weight-calibration-summary/v1",
        "status": "passed",
        "next_required_change": "prepare_stage19_evaluator_critic_preflight",
        "profile_id": "xunce-coverage-cost-risk-boundary-v3",
        "profile_version": "v3",
        "profile_hash": "fixture-v3-hash",
        "stage19_authorized": False,
        "diagnostic_rollout_summary": {"rows": rows},
    }
    manifest = {"schema_version": "xunce-stage18-11-manifest/v1", "summary": str(root / "xunce-stage18-11-path-cost-weight-calibration-summary.json")}
    (root / "xunce-stage18-11-path-cost-weight-calibration-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "xunce-stage18-11-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return root


def _write_rollout(
    tmp_path: Path,
    name: str,
    count: int,
    weight: float,
    coverage: float,
    path_cost: float,
    soft_risk: float,
    *,
    hard_risk: int = 0,
    write_steps: bool = True,
) -> dict:
    root = tmp_path / "diagnostic_rollouts" / name / "stage18_4e"
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "xunce-exploration-coverage-comparison-summary/v1",
        "profile_id": "xunce-coverage-cost-risk-boundary-v3",
        "profile_version": "v3",
        "profile_hash": "fixture-v3-hash",
        "include_canonical_reward_rerank_oracle": True,
        "canonical_reward_rerank_profile_id": f"xunce-coverage-cost-risk-boundary-v3-path-cost-w{int(weight * 100):03d}",
        "canonical_reward_rerank_profile_hash": f"fixture-profile-{weight}",
        "xunce_coverage_advantage_established": False,
    }
    episodes = [
        _episode("canonical_reward_rerank_oracle", coverage, path_cost, soft_risk, hard_risk),
        _episode("canonical_reward_rerank_oracle", max(0.99, coverage - 0.01), path_cost + 5.0, soft_risk + 2.0, hard_risk),
        _episode("xunce", 0.35, 170.0, 210.0, 0),
        _episode("incumbent", 0.23, 112.0, 164.0, 0),
    ]
    (root / "xunce-exploration-coverage-comparison-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_jsonl(root / "xunce-exploration-coverage-episodes.jsonl", episodes)
    if write_steps:
        steps = [
            _step("canonical_reward_rerank_oracle", "s1", 0, 3, "set-shared", 20, 50.0),
            _step("xunce", "s1", 0, 1, "set-shared", 5, 20.0),
            _step("incumbent", "s1", 0, 0, "set-shared", 3, 10.0),
        ]
        _write_jsonl(root / "xunce-exploration-coverage-steps.jsonl", steps)
    _write_jsonl(root / "xunce-exploration-coverage-paired-decision-audit.jsonl", [{"scenario_id": "s1", "step_index": 0}])
    _write_jsonl(root / "xunce-exploration-coverage-candidate-metric-audit.jsonl", [{"scenario_id": "s1", "step_index": 0}])
    return {
        "name": name,
        "candidate_count": count,
        "path_cost_weight": weight,
        "proposal_pool_limit": count * 8,
        "root": str(root),
        "complete": True,
    }


def _episode(policy: str, coverage: float, path_cost: float, soft_risk: float, hard_risk: int) -> dict:
    return {
        "policy": policy,
        "final_coverage_rate": coverage,
        "final_coverage_rate_capped": min(1.0, coverage),
        "coverage_rate_capped": min(1.0, coverage),
        "path_cost_total_m": path_cost,
        "soft_risk_exposure_total": soft_risk,
        "hard_risk_violation_count": hard_risk,
        "coverage_saturation_exceeded": coverage > 1.0,
    }


def _step(policy: str, scenario_id: str, step: int, selected: int, candidate_hash: str, coverage: int, cost: float) -> dict:
    return {
        "policy": policy,
        "scenario_id": scenario_id,
        "step_index": step,
        "selected_action_index": selected,
        "candidate_set_hash": candidate_hash,
        "new_covered_cell_count": coverage,
        "path_cost": cost,
        "soft_risk_exposure": 1.0,
        "hard_risk_violation_count": 0,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
