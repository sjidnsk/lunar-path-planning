import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_8i_rejects_non_policy_signal_stage26_8h_root(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair as s26

    root = _write_stage26_8h_root(tmp_path, next_required_change="resume_stage26_8d_seed_horizon_jobs_with_diverse_scenarios")
    summary = s26.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair(
        config_path=_write_config(tmp_path, root, run_stage26_chain=False),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8i_required_inputs"
    assert "stage26_8h_route_mismatch" in summary["input_rejections"]


def test_stage26_8i_rejects_stage26_8h_phase_execution_failure(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair as s26

    root = _write_stage26_8h_root(tmp_path)
    summary_path = root / "xunce-stage26-8h-summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["phase_execution_failed"] = True
    _write_json(summary_path, payload)

    summary = s26.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair(
        config_path=_write_config(tmp_path, root, run_stage26_chain=False),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8i_required_inputs"
    assert "stage26_8h_phase_execution_failed" in summary["input_rejections"]


def test_stage26_8i_rejects_failed_stage26_8h_status(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair as s26

    root = _write_stage26_8h_root(tmp_path)
    summary_path = root / "xunce-stage26-8h-summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["status"] = "failed"
    _write_json(summary_path, payload)

    summary = s26.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair(
        config_path=_write_config(tmp_path, root, run_stage26_chain=False),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8i_required_inputs"
    assert "stage26_8h_status_not_passed_clean_policy_signal_repair" in summary["input_rejections"]


def test_stage26_8i_builds_update_config_from_diverse_stage26_1_root(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair as s26

    root = _write_stage26_8h_root(tmp_path)
    captured = {}

    def fake_stage26_2(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        captured.update(cfg)
        return _write_stage26_2_result(output_root, policy_kl=0.1)

    monkeypatch.setattr(s26.stage26_2, "run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke", fake_stage26_2)
    monkeypatch.setattr(s26.stage26_8h, "run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval", _fake_stage26_8h(action_changed=False))

    summary = s26.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair(
        config_path=_write_config(tmp_path, root, update_sweep=[_combo("probe", epochs=8, lr=2.0e-5, policy=2.0)]),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert Path(captured["stage26_1_root"]).as_posix().endswith("stage26_8g/h16_seed260801/s26_1")
    assert captured["epochs"] == 8
    assert captured["learning_rate"] == 2.0e-5
    assert captured["policy_loss_coefficient"] == 2.0
    assert summary["combo_count"] == 1


def test_stage26_8i_policy_kl_unstable_does_not_run_eval(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair as s26

    root = _write_stage26_8h_root(tmp_path)

    def fake_stage26_2(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        return _write_stage26_2_result(output_root, policy_kl=2.0)

    def fail_eval(**_kwargs):
        raise AssertionError("unstable combo must not run Stage26.8H eval")

    monkeypatch.setattr(s26.stage26_2, "run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke", fake_stage26_2)
    monkeypatch.setattr(s26.stage26_8h, "run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval", fail_eval)

    summary = s26.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair(
        config_path=_write_config(tmp_path, root, update_sweep=[_combo("unstable")]),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["next_required_change"] == "repair_stage26_8i_update_stability"
    assert summary["stable_combo_count"] == 0


def test_stage26_8i_action_change_with_nonnegative_efficiency_routes_resume(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair as s26

    root = _write_stage26_8h_root(tmp_path)

    def fake_stage26_2(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        return _write_stage26_2_result(output_root, policy_kl=0.1)

    monkeypatch.setattr(s26.stage26_2, "run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke", fake_stage26_2)
    monkeypatch.setattr(s26.stage26_8h, "run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval", _fake_stage26_8h(action_changed=True, per100m=1.0))

    summary = s26.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair(
        config_path=_write_config(tmp_path, root, update_sweep=[_combo("changed")]),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "resume_stage26_8d_seed_horizon_jobs_with_diverse_scenarios"
    assert summary["selected_action_changed_count"] == 1
    assert summary["main_coverage_per_100m_delta"] == 1.0


def test_stage26_8i_multiple_good_combos_route_to_stage26_9(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair as s26

    root = _write_stage26_8h_root(tmp_path)

    def fake_stage26_2(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        return _write_stage26_2_result(output_root, policy_kl=0.1)

    monkeypatch.setattr(s26.stage26_2, "run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke", fake_stage26_2)
    monkeypatch.setattr(s26.stage26_8h, "run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval", _fake_stage26_8h(action_changed=True, per100m=0.5))

    summary = s26.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair(
        config_path=_write_config(
            tmp_path,
            root,
            update_sweep=[_combo("changed_a"), _combo("changed_b")],
            max_stable_eval_count=2,
        ),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot"
    rows = (tmp_path / "out" / "xunce-stage26-8i-update-sweep-results.jsonl").read_text(encoding="utf-8").splitlines()
    assert sum(1 for line in rows if json.loads(line)["stage26_8h_status"] == "passed") == 2


def test_stage26_8i_action_unchanged_routes_to_update_strength_when_delta_too_small(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair as s26

    root = _write_stage26_8h_root(tmp_path)

    def fake_stage26_2(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        return _write_stage26_2_result(output_root, policy_kl=0.1)

    monkeypatch.setattr(s26.stage26_2, "run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke", fake_stage26_2)
    monkeypatch.setattr(s26.stage26_8h, "run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval", _fake_stage26_8h(action_changed=False))

    summary = s26.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair(
        config_path=_write_config(tmp_path, root, update_sweep=[_combo("unchanged")]),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["next_required_change"] == "increase_stage26_synthetic_update_strength_or_sample_count"
    assert summary["selected_action_changed_count"] == 0
    assert summary["policy_delta_too_small_for_margin"] is True


def test_stage26_8i_routes_to_continue_when_stable_combo_eval_pending(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair as s26

    root = _write_stage26_8h_root(tmp_path)

    def fake_stage26_2(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        return _write_stage26_2_result(output_root, policy_kl=0.1)

    monkeypatch.setattr(s26.stage26_2, "run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke", fake_stage26_2)
    monkeypatch.setattr(s26.stage26_8h, "run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval", _fake_stage26_8h_pending_post())

    summary = s26.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair(
        config_path=_write_config(tmp_path, root, update_sweep=[_combo("pending")]),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "continue_stage26_8h_post_eval"


def test_stage26_8i_routes_to_post_eval_execution_repair_on_broken_process_pool(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair as s26

    root = _write_stage26_8h_root(tmp_path)

    def fake_stage26_2(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        return _write_stage26_2_result(output_root, policy_kl=0.1)

    monkeypatch.setattr(s26.stage26_2, "run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke", fake_stage26_2)
    monkeypatch.setattr(s26.stage26_8h, "run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval", _fake_stage26_8h_pending_post(broken=True))

    summary = s26.run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair(
        config_path=_write_config(tmp_path, root, update_sweep=[_combo("broken")]),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_8i_stable_combo_post_eval_execution"


def test_stage26_8i_registry_entry_exists() -> None:
    registry = json.loads((REPO_ROOT / "configs" / "stage_registry.json").read_text(encoding="utf-8"))
    assert "xunce-stage26-8i-diverse-scenario-policy-signal-strength-repair" in registry["stages"]


def _write_stage26_8h_root(tmp_path: Path, *, next_required_change: str = "repair_stage26_synthetic_policy_update_signal_strength") -> Path:
    root = tmp_path / "stage26_8h"
    stage26_8g = tmp_path / "stage26_8g"
    job = stage26_8g / "h16_seed260801"
    _write_json(job / "s26_1" / "xunce-stage26-1-summary.json", {"status": "passed"})
    _write_jsonl(
        stage26_8g / "xunce-stage26-8g-scenario-fixture-catalog.jsonl",
        [
            {"scenario_start_cell": [0, 1], "scenario_diversity_content_hash": "c0", "scenario_diversity_signature_hash": "s0"},
            {"scenario_start_cell": [2, 3], "scenario_diversity_content_hash": "c1", "scenario_diversity_signature_hash": "s1"},
            {"scenario_start_cell": [4, 5], "scenario_diversity_content_hash": "c2", "scenario_diversity_signature_hash": "s2"},
        ],
    )
    _write_json(job / "xunce-stage26-8g-stage26-3-config.json", {"schema_version": "fake-stage26-3-config/v1", "stage26_2_root": str(job / "s26_2")})
    _write_json(job / "s26_3" / "xunce-stage26-3-stage21-5-config.json", {"stage21_4_tiny_ppo_update_smoke_root": str(job / "s26_2" / "s21_4"), "high_fidelity_config": str(job / "s26_3" / "xunce-stage26-3-high-fidelity-config.json")})
    _write_json(job / "s26_3" / "xunce-stage26-3-high-fidelity-config.json", {"schema_version": "fake-high-fidelity/v1"})
    stage26_3 = {
        "schema_version": "xunce-stage26-3-summary/v1",
        "stage_id": "xunce-stage26-3-synthetic-terrain-post-update-trajectory-eval-smoke",
        "status": "failed",
        "next_required_change": "repair_stage26_synthetic_policy_update_signal_strength",
        "strong_state_join_available_count": 48,
        "selected_action_changed_count": 0,
        "synthetic_inference_required_field_missing_count": 0,
        "hybrid_path_contract_mismatch_count": 0,
        "hard_risk_violation_count": 0,
        "mask_violation_count": 0,
        "path_planning_failure_count": 0,
        "open_grid_fallback_count": 0,
    }
    _write_json(root / "s26_3_aggregate" / "xunce-stage26-3-summary.json", stage26_3)
    _write_json(
        root / "xunce-stage26-8h-summary.json",
        {
            "schema_version": "xunce-stage26-8h-summary/v1",
            "stage_id": "xunce-stage26-8h-resumable-diverse-scenario-post-update-eval",
            "status": "passed",
            "next_required_change": next_required_change,
            "pre_eval_complete": True,
            "post_eval_complete": True,
            "stage26_3_aggregate_complete": True,
            "scenario_diversity_repaired": True,
            "scenario_signature_duplicate_group_count": 0,
            "stage26_8g_root": str(stage26_8g),
            "stage26_3_root": str(root / "s26_3_aggregate"),
        },
    )
    return root


def _write_stage26_2_result(output_root: Path, *, policy_kl: float) -> dict:
    s21_4 = output_root / "s21_4"
    _write_json(
        s21_4 / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json",
        {
            "status": "passed",
            "final_post_update_policy_approx_kl": policy_kl,
            "final_post_update_behavior_approx_kl": 2.5,
            "ppo_ratio_old_log_prob_source": "behavior_policy_when_present",
            "kl_gate_source": "policy_old_logprob_when_available/v1",
            "behavior_policy_kl_diagnostic_only": True,
            "source_xunce_candidate_checkpoint": "source.pt",
            "experimental_checkpoint_path": "experimental.pt",
        },
    )
    return {
        "schema_version": "xunce-stage26-2-summary/v1",
        "stage_id": "xunce-stage26-2-synthetic-terrain-ppo-update-smoke",
        "status": "passed" if policy_kl <= 1.5 else "failed",
        "next_required_change": "run_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke" if policy_kl <= 1.5 else "repair_stage26_2_synthetic_ppo_update_stability",
        "stage21_4_root": str(s21_4),
        "stage21_4_status": "passed" if policy_kl <= 1.5 else "failed",
        "final_post_update_approx_kl": policy_kl,
        "loss_finite": True,
        "gradient_finite": True,
        "checkpoint_exists": True,
        "checkpoint_reload_passed": True,
        "experimental_only": True,
        "checkpoint_boundary_passed": True,
        "stage21_3_lineage_passed": True,
        "source_checkpoint_sha_consistent": True,
        "collector_source_checkpoint_match": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "parameter_delta_l2": 0.01,
    }


def _fake_stage26_8h(*, action_changed: bool, per100m: float = 0.0):
    def fake(config_path: Path, output_root: Path, repo_root: Path) -> dict:
        pre = output_root / "pre"
        post = output_root / "post"
        _write_inference(pre, selected=0, probs=[0.8, 0.2], logits=[2.0, 0.0])
        if action_changed:
            _write_inference(post, selected=1, probs=[0.4, 0.6], logits=[0.0, 1.0])
        else:
            _write_inference(post, selected=0, probs=[0.800001, 0.199999], logits=[2.000001, 0.0])
        _write_jsonl(
            output_root / "xunce-stage26-8h-phase-state.jsonl",
            [
                {"phase": "pre_eval", "status": "complete", "source_root": str(pre), "summary_path": str(pre / "xunce-exploration-coverage-comparison-summary.json"), "blocking_reason": ""},
                {"phase": "post_eval", "status": "complete", "source_root": str(post), "summary_path": str(post / "xunce-exploration-coverage-comparison-summary.json"), "blocking_reason": ""},
                {"phase": "stage26_3_aggregate", "status": "complete", "source_root": str(output_root / "s26_3_aggregate"), "summary_path": str(output_root / "s26_3_aggregate" / "xunce-stage26-3-summary.json"), "blocking_reason": ""},
            ],
        )
        summary = {
            "schema_version": "xunce-stage26-8h-summary/v1",
            "stage_id": "xunce-stage26-8h-resumable-diverse-scenario-post-update-eval",
            "status": "passed",
            "next_required_change": "resume_stage26_8d_seed_horizon_jobs_with_diverse_scenarios" if action_changed else "repair_stage26_synthetic_policy_update_signal_strength",
            "stage26_3_status": "passed" if action_changed else "failed",
            "stage26_3_next_required_change": "run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot" if action_changed else "repair_stage26_synthetic_policy_update_signal_strength",
            "strong_state_join_available_count": 1,
            "selected_action_changed_count": 1 if action_changed else 0,
            "selected_viewpoint_changed_count": 1 if action_changed else 0,
            "selected_theta_changed_count": 1 if action_changed else 0,
            "main_coverage_per_100m_delta": per100m,
            "main_final_coverage_delta": per100m / 100.0,
            "main_coverage_auc_delta": 0.0,
            "summary": str(output_root / "xunce-stage26-8h-summary.json"),
        }
        _write_json(output_root / "xunce-stage26-8h-summary.json", summary)
        return summary

    return fake


def _fake_stage26_8h_pending_post(*, broken: bool = False):
    def fake(config_path: Path, output_root: Path, repo_root: Path) -> dict:
        pre = output_root / "pre_ppo_xunce"
        _write_inference(pre, selected=0, probs=[0.8, 0.2], logits=[2.0, 0.0])
        blocking = "BrokenProcessPool:A child process terminated abruptly" if broken else ""
        _write_jsonl(
            output_root / "xunce-stage26-8h-phase-state.jsonl",
            [
                {"phase": "pre_eval", "status": "complete", "source_root": str(pre), "summary_path": str(pre / "xunce-exploration-coverage-comparison-summary.json"), "blocking_reason": ""},
                {"phase": "post_eval", "status": "pending", "source_root": str(output_root / "post_ppo_xunce"), "summary_path": str(output_root / "post_ppo_xunce" / "xunce-exploration-coverage-comparison-summary.json"), "blocking_reason": blocking, "execution": {"blocking_reason": blocking}},
                {"phase": "stage26_3_aggregate", "status": "pending", "source_root": str(output_root / "s26_3_aggregate"), "summary_path": str(output_root / "s26_3_aggregate" / "xunce-stage26-3-summary.json"), "blocking_reason": "summary_missing"},
            ],
        )
        summary = {
            "schema_version": "xunce-stage26-8h-summary/v1",
            "stage_id": "xunce-stage26-8h-resumable-diverse-scenario-post-update-eval",
            "status": "failed",
            "next_required_change": "continue_stage26_8h_post_eval",
            "stage26_3_status": None,
            "stage26_3_next_required_change": None,
            "strong_state_join_available_count": 0,
            "selected_action_changed_count": 0,
            "selected_viewpoint_changed_count": 0,
            "selected_theta_changed_count": 0,
            "main_coverage_per_100m_delta": 0.0,
            "summary": str(output_root / "xunce-stage26-8h-summary.json"),
        }
        _write_json(output_root / "xunce-stage26-8h-summary.json", summary)
        return summary

    return fake


def _write_inference(root: Path, *, selected: int, probs: list[float], logits: list[float]) -> None:
    _write_json(root / "xunce-exploration-coverage-comparison-summary.json", {"status": "passed", "next_required_change": "review_xunce_incumbent_comparison_metrics"})
    row = {
        "scenario_id": "s0",
        "step_index": 0,
        "current_cell": [1, 1],
        "covered_cells_hash": "covered",
        "candidate_set_hash": "candidates",
        "synthetic_terrain_hash": "synthetic",
        "selected_action_index": selected,
        "selected_viewpoint": [selected, 0, 0],
        "action_probs": probs,
        "logits": logits,
    }
    _write_jsonl(root / "xunce-exploration-coverage-model-inference.jsonl", [row])
    _write_jsonl(root / "xunce-exploration-coverage-episodes.jsonl", [{"policy": "xunce"}])
    _write_jsonl(root / "xunce-exploration-coverage-steps.jsonl", [{"policy": "xunce"}])


def _write_config(
    tmp_path: Path,
    root: Path,
    *,
    run_stage26_chain: bool = True,
    update_sweep: list[dict] | None = None,
    max_stable_eval_count: int = 1,
) -> Path:
    path = tmp_path / "stage26_8i_config.json"
    payload = {
        "schema_version": "xunce-stage26-8i-diverse-scenario-policy-signal-strength-repair-config/v1",
        "stage_id": "xunce-stage26-8i-diverse-scenario-policy-signal-strength-repair",
        "stage26_8h_root": str(root),
        "stage26_2_base_config": str(tmp_path / "stage26_2_base.json"),
        "run_stage26_chain": run_stage26_chain,
        "max_stable_eval_count": max_stable_eval_count,
        "max_abs_approx_kl": 1.5,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    if update_sweep is not None:
        payload["update_sweep"] = update_sweep
    _write_json(path, payload)
    _write_json(tmp_path / "stage26_2_base.json", {"schema_version": "xunce-stage26-2-synthetic-terrain-ppo-update-smoke-config/v1", "stage_id": "xunce-stage26-2-synthetic-terrain-ppo-update-smoke"})
    return path


def _combo(combo_id: str, *, epochs: int = 4, lr: float = 1.0e-5, policy: float = 1.0) -> dict:
    return {
        "combo_id": combo_id,
        "work_dir": combo_id,
        "epochs": epochs,
        "learning_rate": lr,
        "policy_loss_coefficient": policy,
        "value_loss_coefficient": 0.02,
        "entropy_coefficient": 0.01,
        "loss_scale": 0.25,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
