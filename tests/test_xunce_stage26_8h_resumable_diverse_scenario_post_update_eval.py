import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_8h_accepts_partial_stage26_8g_and_routes_to_post_eval(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval as s26

    root = _write_stage26_8g_partial_root(tmp_path, pre_complete=True, post_complete=False)
    summary = s26.run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval(
        config_path=_write_config(tmp_path, root, run_mode="aggregate_only"),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "continue_stage26_8h_post_eval"
    assert summary["pre_eval_complete"] is True
    assert summary["post_eval_complete"] is False
    assert summary["next_phase"] == "post_eval"


def test_stage26_8h_run_next_advances_only_post_when_pre_is_carried_over(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval as s26

    root = _write_stage26_8g_partial_root(tmp_path, pre_complete=True, post_complete=False)

    def fake_eval(config, stage26_8g_root, repo_root, output_root, *, label):
        assert label == "post"
        _write_eval_root(output_root)

    monkeypatch.setattr(s26, "_run_eval_phase", fake_eval)

    summary = s26.run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval(
        config_path=_write_config(tmp_path, root),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["phase_execution"]["phase"] == "post_eval"
    assert summary["pre_eval_complete"] is True
    assert summary["post_eval_complete"] is True
    assert summary["next_required_change"] == "repair_stage26_8h_stage26_3_aggregate_contract"


def test_stage26_8h_phase_exception_is_failed_but_resumable(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval as s26

    root = _write_stage26_8g_partial_root(tmp_path, pre_complete=True, post_complete=False)

    def boom(*_args, **_kwargs):
        raise RuntimeError("post eval interrupted")

    monkeypatch.setattr(s26, "_run_eval_phase", boom)

    output_root = tmp_path / "out"
    summary = s26.run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval(
        config_path=_write_config(tmp_path, root),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "continue_stage26_8h_post_eval"
    assert summary["phase_execution_failed"] is True
    rows = [json.loads(line) for line in (output_root / s26.PHASE_STATE_FILE).read_text(encoding="utf-8").splitlines()]
    post_row = next(row for row in rows if row["phase"] == "post_eval")
    assert "RuntimeError:post eval interrupted" in post_row["blocking_reason"]


def test_stage26_8h_stage26_3_aggregate_uses_artifact_only_config(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval as s26

    root = _write_stage26_8g_partial_root(tmp_path, pre_complete=True, post_complete=True)
    captured = {}

    def fake_stage26_3(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        captured.update(cfg)
        summary = {
            "schema_version": "xunce-stage26-3-summary/v1",
            "stage_id": "xunce-stage26-3-synthetic-terrain-post-update-trajectory-eval-smoke",
            "status": "failed",
            "next_required_change": "repair_stage26_synthetic_policy_update_signal_strength",
            "coverage_denominator_source": "main_coverable_cells/v1",
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
            "path_cost_source": "hybrid_astar_pose_path/v1",
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
            "post_update_success_metric": "main_coverable_coverage_efficiency/v1",
            "synthetic_terrain_hash": "synthetic-hash",
            "platform_contract_hash": "platform-hash",
            "max_traversable_slope_deg": 30.0,
            "main_coverage_per_100m_delta": 0.0,
            "strong_state_join_available_count": 48,
        }
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / s26.stage26_3.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(s26.stage26_3, "run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke", fake_stage26_3)
    monkeypatch.setattr(
        s26.stage26_8g,
        "_recheck_stage26_3",
        lambda *_args, **_kwargs: {
            "scenario_signature_duplicate_group_count": 0,
            "scenario_diversity_repaired": True,
        },
    )

    summary = s26.run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval(
        config_path=_write_config(tmp_path, root, run_mode="run_phase", phase="stage26_3_aggregate"),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert captured["execute_high_fidelity_evaluations"] is False
    assert captured["pre_ppo_evaluation_root"].endswith("pre")
    assert captured["post_ppo_evaluation_root"].endswith("post")
    assert summary["next_required_change"] == "repair_stage26_synthetic_policy_update_signal_strength"


def test_stage26_8h_routes_resume_when_efficiency_positive(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval as s26

    root = _write_stage26_8g_partial_root(tmp_path, pre_complete=True, post_complete=True)

    def fake_stage26_3(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        summary = {
            "schema_version": "xunce-stage26-3-summary/v1",
            "stage_id": "xunce-stage26-3-synthetic-terrain-post-update-trajectory-eval-smoke",
            "status": "passed",
            "next_required_change": "run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot",
            "coverage_denominator_source": "main_coverable_cells/v1",
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
            "path_cost_source": "hybrid_astar_pose_path/v1",
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
            "post_update_success_metric": "main_coverable_coverage_efficiency/v1",
            "synthetic_terrain_hash": "synthetic-hash",
            "platform_contract_hash": "platform-hash",
            "max_traversable_slope_deg": 30.0,
            "main_coverage_per_100m_delta": 1.0,
            "strong_state_join_available_count": 48,
        }
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / s26.stage26_3.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(s26.stage26_3, "run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke", fake_stage26_3)
    monkeypatch.setattr(
        s26.stage26_8g,
        "_recheck_stage26_3",
        lambda *_args, **_kwargs: {
            "scenario_signature_duplicate_group_count": 0,
            "scenario_diversity_repaired": True,
        },
    )

    summary = s26.run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval(
        config_path=_write_config(tmp_path, root, run_mode="run_phase", phase="stage26_3_aggregate"),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "resume_stage26_8d_seed_horizon_jobs_with_diverse_scenarios"
    assert summary["main_coverage_per_100m_delta"] == 1.0


def test_stage26_8h_rejects_fixture_missing_start_cell(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval as s26

    root = _write_stage26_8g_partial_root(tmp_path, pre_complete=True, post_complete=False)
    _write_jsonl(
        root / "xunce-stage26-8g-scenario-fixture-catalog.jsonl",
        [
            {
                "scenario_id": "scenario-0",
                "scenario_diversity_content_hash": "content-0",
                "scenario_diversity_signature_hash": "signature-0",
            },
            {
                "scenario_id": "scenario-1",
                "scenario_start_cell": [1, 2],
                "scenario_diversity_content_hash": "content-1",
                "scenario_diversity_signature_hash": "signature-1",
            },
            {
                "scenario_id": "scenario-2",
                "scenario_start_cell": [2, 3],
                "scenario_diversity_content_hash": "content-2",
                "scenario_diversity_signature_hash": "signature-2",
            },
        ],
    )

    summary = s26.run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval(
        config_path=_write_config(tmp_path, root, run_mode="aggregate_only"),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_8g_collector_scenario_diversity_binding"
    assert "scenario_start_cell_missing_or_invalid" in summary["fixture_rejections"]


def test_stage26_8h_registry_entry_exists() -> None:
    registry = json.loads((REPO_ROOT / "configs" / "stage_registry.json").read_text(encoding="utf-8"))
    assert "xunce-stage26-8h-resumable-diverse-scenario-post-update-eval" in registry["stages"]


def _write_stage26_8g_partial_root(tmp_path: Path, *, pre_complete: bool, post_complete: bool) -> Path:
    root = tmp_path / "stage26_8g"
    job = root / "h16_seed260801"
    _write_json(job / "s26_1" / "xunce-stage26-1-summary.json", {"status": "passed", "scenario_fixture_catalog": str(root / "xunce-stage26-8g-scenario-fixture-catalog.jsonl")})
    _write_json(job / "s26_2" / "xunce-stage26-2-summary.json", {"status": "passed", "stage26_1_root": str(job / "s26_1")})
    _write_json(job / "xunce-stage26-8g-stage26-3-config.json", _stage26_3_config(job))
    _write_json(job / "s26_3" / "xunce-stage26-3-stage21-5-config.json", _stage21_5_config(job))
    _write_json(job / "s26_3" / "xunce-stage26-3-high-fidelity-config.json", {"schema_version": "fake-high-fidelity-config/v1"})
    _write_jsonl(
        root / "xunce-stage26-8g-scenario-fixture-catalog.jsonl",
        [
            {
                "scenario_id": f"scenario-{idx}",
                "scenario_start_cell": [idx, idx + 1],
                "scenario_diversity_content_hash": f"content-{idx}",
                "scenario_diversity_signature_hash": f"signature-{idx}",
            }
            for idx in range(3)
        ],
    )
    if pre_complete:
        _write_eval_root(job / "s26_3" / "pre")
    if post_complete:
        _write_eval_root(job / "s26_3" / "post")
    return root


def _stage26_3_config(job: Path) -> dict:
    return {
        "schema_version": "xunce-stage26-3-synthetic-terrain-post-update-trajectory-eval-smoke-config/v1",
        "stage26_2_root": str(job / "s26_2"),
        "stage21_5_base_config": "configs/xunce_stage21_5_post_update_offline_trajectory_evaluation_v1.json",
        "high_fidelity_config": "configs/xunce_high_fidelity_exploration_coverage_comparison_stage18_9_strict_v3.json",
        "required_scenario_count": 3,
        "rollout_steps": 16,
        "dynamic_max_candidates_per_step": 36,
        "dynamic_proposal_pool_limit_per_step": 288,
        "theta_bin_count": 8,
        "theta_step_deg": 45,
        "sensor_fov_deg": 90.0,
        "sensor_range_cells": 2,
        "coverage_denominator_source": "main_coverable_cells/v1",
        "post_update_success_metric": "main_coverable_coverage_efficiency/v1",
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
    }


def _stage21_5_config(job: Path) -> dict:
    return {
        "schema_version": "xunce-stage21-5-post-update-offline-trajectory-evaluation-config/v1",
        "stage21_4_tiny_ppo_update_smoke_root": str(job / "s26_2" / "s21_4"),
        "high_fidelity_config": str(job / "s26_3" / "xunce-stage26-3-high-fidelity-config.json"),
        "execute_high_fidelity_evaluations": True,
        "pre_ppo_evaluation_root": str(job / "s26_3" / "pre"),
        "post_ppo_evaluation_root": str(job / "s26_3" / "post"),
        "required_scenario_count": 3,
        "rollout_steps": 16,
        "dynamic_max_candidates_per_step": 36,
        "dynamic_proposal_pool_limit_per_step": 288,
        "include_oracle_baselines": False,
        "include_canonical_reward_rerank_oracle": False,
        "xunce_only_evaluation": True,
        "emit_candidate_metric_audit": True,
        "hybrid_astar_candidate_eval_workers": 4,
    }


def _write_eval_root(root: Path) -> None:
    _write_json(root / "xunce-exploration-coverage-comparison-summary.json", {"status": "passed", "next_required_change": "review_xunce_incumbent_comparison_metrics"})
    _write_jsonl(root / "xunce-exploration-coverage-episodes.jsonl", [{"policy": "xunce", "scenario_id": "s0"}])
    _write_jsonl(root / "xunce-exploration-coverage-model-inference.jsonl", [{"policy": "xunce", "scenario_id": "s0"}])
    _write_jsonl(root / "xunce-exploration-coverage-steps.jsonl", [{"policy": "xunce", "scenario_id": "s0"}])


def _write_config(tmp_path: Path, root: Path, *, run_mode: str = "run_next", phase: str | None = None) -> Path:
    path = tmp_path / "stage26_8h_config.json"
    payload = {
        "schema_version": "xunce-stage26-8h-resumable-diverse-scenario-post-update-eval-config/v1",
        "stage_id": "xunce-stage26-8h-resumable-diverse-scenario-post-update-eval",
        "stage26_8g_root": str(root),
        "run_mode": run_mode,
        "seed": 260801,
        "rollout_steps": 16,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    if phase:
        payload["phase"] = phase
    _write_json(path, payload)
    return path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
