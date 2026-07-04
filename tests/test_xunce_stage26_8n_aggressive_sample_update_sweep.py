import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_8n_rejects_non_update_strength_stage26_8i_root(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    root = _write_stage26_8i_root(tmp_path, next_required_change="run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot")
    output_root = tmp_path / "out"

    summary = s26.run_xunce_stage26_8n_aggressive_sample_update_sweep(
        config_path=_write_config(tmp_path, root),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )
    routing = json.loads((output_root / "xunce-stage26-8n-next-stage-routing.json").read_text(encoding="utf-8"))

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8n_required_inputs"
    assert "stage26_8i_route_mismatch" in summary["input_rejections"]
    assert routing["release_or_training_authorized"] is False
    assert routing["publishes_checkpoint"] is False
    assert routing["replaces_default_policy"] is False
    assert routing["connects_real_executor"] is False
    assert routing["starts_online_canary"] is False
    assert routing["canary_traffic_fraction"] == 0.0


def test_stage26_8n_writes_aggressive_8m_config_with_collector_reuse(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    root = _write_stage26_8i_root(tmp_path)
    captured: dict = {}

    def fake_8m(*, config_path: Path, output_root: Path, repo_root: Path, **kwargs) -> dict:
        captured.update(json.loads(config_path.read_text(encoding="utf-8")))
        return _write_8m_pending_summary(output_root)

    monkeypatch.setattr(s26.stage26_8m, "run_xunce_stage26_8m_generalized_resumable_training_pipeline", fake_8m)

    summary = s26.run_xunce_stage26_8n_aggressive_sample_update_sweep(
        config_path=_write_config(tmp_path, root),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["next_required_change"] == "continue_stage26_8n_aggressive_jobs"
    assert captured["collector_reuse_policy"] == "by_horizon_seed_scenario_rollout/v1"
    assert captured["scenario_counts"] == [6]
    assert captured["collector_rollout_steps"] == [20]
    assert captured["eval_rollout_steps"] == [20]
    assert [combo["combo_id"] for combo in captured["update_combos"]] == [
        "policy_amp_repro",
        "depth16_lr3e5_policy3",
        "depth24_lr3e5_policy3",
        "lr5e5_policy3",
        "value_off_policy4_probe",
    ]
    assert captured["max_abs_approx_kl"] == 1.5


def test_stage26_8n_routes_to_expand_sample_budget_when_shared_collector_under_target(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    root = _write_stage26_8i_root(tmp_path)
    output_root = tmp_path / "out"
    collector_root = output_root / "m" / "cabc123"
    _write_json(
        collector_root / "xunce-stage26-1-summary.json",
        {"schema_version": "xunce-stage26-1-summary/v1", "status": "passed", "trainable_transition_count": 80},
    )
    _write_8m_state(output_root / "m", collector_root=collector_root, collector_status="complete")

    summary = s26.run_xunce_stage26_8n_aggressive_sample_update_sweep(
        config_path=_write_config(tmp_path, root, run_stage26_8m=False),
        output_root=output_root,
        repo_root=REPO_ROOT,
    )

    assert summary["next_required_change"] == "expand_stage26_8n_aggressive_sample_budget"
    assert summary["trainable_transition_count"] == 80


def test_stage26_8n_routes_collector_failure_before_expand_sample_budget(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    rows = [
        {
            "job_id": "shared",
            "phase": "collector",
            "status": "failed",
            "blocking_reason": "input_hash_mismatch",
            "binding_or_safety_failure": False,
        }
    ]

    route = s26._route(
        boundary_rejections=[],
        input_rejections=[],
        collector_audit={"collector_complete": True, "trainable_transition_count": 3},
        update_rows=[],
        selected_eval_job_ids=[],
        job_rows=rows,
        stage26_8m_summary={},
        min_trainable_transition_count=100,
        max_abs_approx_kl=1.5,
    )

    assert route == "repair_stage26_8n_collector_binding_or_safety"


def test_stage26_8n_selects_only_two_stable_combos_for_eval(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    rows = [
        _update_row("a", stable=True, parameter_delta_l2=0.2, policy_kl=1.4),
        _update_row("b", stable=True, parameter_delta_l2=0.5, policy_kl=1.2),
        _update_row("c", stable=True, parameter_delta_l2=0.3, policy_kl=1.0),
        _update_row("d", stable=False, parameter_delta_l2=0.9, policy_kl=2.0),
    ]

    selected = s26._selected_eval_job_ids(rows, max_abs_approx_kl=1.5, max_eval_count=2)

    assert selected == ["b", "c"]


def test_stage26_8n_ignores_nonselected_eval_pending_when_selected_eval_complete(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    selected = ["selected_job"]
    rows = [
        {"job_id": "selected_job", "phase": "aggregate", "status": "complete", "selected_action_changed_count": 1, "main_coverage_per_100m_delta": 0.0},
        {"job_id": "skipped_job", "phase": "eval_pre", "status": "pending"},
        {"job_id": "skipped_job", "phase": "eval_post", "status": "pending"},
        {"job_id": "skipped_job", "phase": "aggregate", "status": "pending"},
    ]

    route = s26._route(
        boundary_rejections=[],
        input_rejections=[],
        collector_audit={"collector_complete": True, "trainable_transition_count": 120},
        update_rows=[_update_row("selected_job", stable=True, parameter_delta_l2=0.5, policy_kl=0.3)],
        selected_eval_job_ids=selected,
        job_rows=rows,
        stage26_8m_summary={},
        min_trainable_transition_count=100,
        max_abs_approx_kl=1.5,
    )

    assert route == "resume_stage26_8d_seed_horizon_jobs_with_diverse_scenarios"


def test_stage26_8n_routes_eval_repair_for_lineage_mismatched_positive_aggregate(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8n_aggressive_sample_update_sweep as s26

    rows = [
        {
            "job_id": "selected_job",
            "phase": "aggregate",
            "status": "complete",
            "selected_action_changed_count": 1,
            "main_coverage_per_100m_delta": 1.0,
            "lineage_mismatch": True,
        }
    ]

    route = s26._route(
        boundary_rejections=[],
        input_rejections=[],
        collector_audit={"collector_complete": True, "trainable_transition_count": 120},
        update_rows=[_update_row("selected_job", stable=True, parameter_delta_l2=0.5, policy_kl=0.3)],
        selected_eval_job_ids=["selected_job"],
        job_rows=rows,
        stage26_8m_summary={},
        min_trainable_transition_count=100,
        max_abs_approx_kl=1.5,
    )

    assert route == "repair_stage26_8n_eval_binding_or_safety"


def test_stage26_8n_registry_entry_exists() -> None:
    registry = json.loads((REPO_ROOT / "configs" / "stage_registry.json").read_text(encoding="utf-8"))
    assert "xunce-stage26-8n-aggressive-sample-update-sweep" in registry["stages"]


def _write_config(tmp_path: Path, stage26_8i_root: Path, **overrides) -> Path:
    (tmp_path / "stage26_8g").mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "xunce-stage26-8n-aggressive-sample-update-sweep-config/v1",
        "stage_id": "xunce-stage26-8n-aggressive-sample-update-sweep",
        "stage26_8i_root": str(stage26_8i_root),
        "source_scenario_fixture_root": str(tmp_path / "stage26_8g"),
        "run_stage26_8m": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    } | overrides
    path = tmp_path / "config.json"
    _write_json(path, payload)
    return path


def _write_stage26_8i_root(
    tmp_path: Path,
    *,
    next_required_change: str = "increase_stage26_synthetic_update_strength_or_sample_count",
) -> Path:
    root = tmp_path / "stage26_8i"
    _write_json(
        root / "xunce-stage26-8i-summary.json",
        {
            "schema_version": "xunce-stage26-8i-summary/v1",
            "stage_id": "xunce-stage26-8i-diverse-scenario-policy-signal-strength-repair",
            "status": "failed",
            "next_required_change": next_required_change,
            "stable_combo_count": 5,
            "selected_action_changed_count": 0,
            "strong_state_join_available_count": 48,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    return root


def _write_8m_pending_summary(root: Path) -> dict:
    summary = {
        "schema_version": "xunce-stage26-8m-summary/v1",
        "stage_id": "xunce-stage26-8m-generalized-resumable-training-pipeline",
        "status": "passed",
        "next_required_change": "continue_stage26_8m_jobs",
        "pending_job_count": 5,
        "completed_job_count": 0,
        "failed_job_count": 0,
    }
    _write_json(root / "xunce-stage26-8m-summary.json", summary)
    return summary


def _write_8m_state(root: Path, *, collector_root: Path, collector_status: str) -> None:
    rows = []
    for combo in ("policy_amp_repro", "depth16_lr3e5_policy3"):
        rows.append(
            {
                "schema_version": "xunce-stage26-8m-job-state/v1",
                "job_id": f"h16_s260801_sc6_cr20_er20_u_{combo}",
                "phase": "collector",
                "status": collector_status,
                "output_root": str(collector_root),
                "summary_path": str(collector_root / "xunce-stage26-1-summary.json"),
                "blocking_reason": "",
            }
        )
        rows.append(
            {
                "schema_version": "xunce-stage26-8m-job-state/v1",
                "job_id": f"h16_s260801_sc6_cr20_er20_u_{combo}",
                "phase": "update",
                "status": "pending",
                "output_root": str(root / "jobs" / f"h16_s260801_sc6_cr20_er20_u_{combo}" / "update"),
                "summary_path": str(root / "jobs" / f"h16_s260801_sc6_cr20_er20_u_{combo}" / "update" / "xunce-stage26-2-summary.json"),
                "blocking_reason": "summary_missing",
            }
        )
    _write_jsonl(root / "xunce-stage26-8m-job-state.jsonl", rows)


def _update_row(combo_id: str, *, stable: bool, parameter_delta_l2: float, policy_kl: float) -> dict:
    return {
        "job_id": combo_id,
        "combo_id": combo_id,
        "phase": "update",
        "status": "complete" if stable else "failed",
        "stage21_4_status": "passed" if stable else "failed",
        "loss_finite": stable,
        "gradient_finite": stable,
        "final_post_update_policy_approx_kl": policy_kl,
        "parameter_delta_l2": parameter_delta_l2,
        "checkpoint_exists": stable,
        "checkpoint_reload_passed": stable,
        "experimental_only": stable,
        "checkpoint_boundary_passed": stable,
        "stage21_3_lineage_passed": stable,
        "source_checkpoint_sha_consistent": stable,
        "collector_source_checkpoint_match": stable,
        "ppo_ratio_old_log_prob_source": "behavior_policy_when_present",
        "kl_gate_source": "policy_old_logprob_when_available/v1",
        "behavior_policy_kl_diagnostic_only": True,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
