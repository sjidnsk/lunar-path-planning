import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


STAGE_ID = "xunce-stage26-10b-completion-capable-shadow-trial"
CONFIG_SCHEMA_VERSION = "xunce-stage26-10b-completion-capable-shadow-trial-config/v1"


def test_stage26_10b_registry_default_config_and_root_are_correct() -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    registry = json.loads((REPO_ROOT / "configs" / "stage_registry.json").read_text(encoding="utf-8"))
    stage = registry["stages"][STAGE_ID]
    config = json.loads(
        (REPO_ROOT / "configs" / "xunce_stage26_10b_completion_capable_shadow_trial_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert s26.STAGE_ID == STAGE_ID
    assert s26.CONFIG_SCHEMA_VERSION == CONFIG_SCHEMA_VERSION
    assert s26.DEFAULT_OUTPUT_ROOT == "D:/xunce/out/s26_10b"
    assert s26.SUMMARY_FILE == "xunce-stage26-10b-summary.json"
    assert s26.ROUTING_FILE == "xunce-stage26-10b-routing.json"
    assert s26.MANIFEST_FILE == "xunce-stage26-10b-manifest.json"
    assert s26.MATRIX_FILE == "xunce-stage26-10b-matrix.jsonl"
    assert s26.JOB_STATE_FILE == "xunce-stage26-10b-job-state.jsonl"
    assert s26.REPORT_FILE == "xunce-stage26-10b-report.md"
    assert stage["script"] == "scripts/run_xunce_stage26_10b_completion_capable_shadow_trial.py"
    assert stage["default_config"] == "configs/xunce_stage26_10b_completion_capable_shadow_trial_v1.json"
    assert stage["default_output_root"] == "D:/xunce/out/s26_10b"
    assert config["schema_version"] == CONFIG_SCHEMA_VERSION
    assert config["stage_id"] == STAGE_ID
    assert config["default_output_root"] == "D:/xunce/out/s26_10b"
    assert config["horizons"] == [1024]
    assert config["seeds"] == [260801, 260802, 260803]
    assert config["scenario_counts"] == [11]
    assert config["collector_rollout_steps"] == [1024]
    assert config["eval_rollout_steps"] == [1024]
    assert config["l0_oracle_proxy_scenario_count"] == 11
    assert config["l0_target_coverage_rate"] == 0.99
    assert [combo["combo_id"] for combo in config["update_combos"]] == [
        "depth24_lr3e5_policy3",
        "lr5e5_policy3",
    ]


def test_stage26_10b_boundary_true_rejects(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    config_path = _write_config(tmp_path, release_or_training_authorized=True)
    summary = s26.run_xunce_stage26_10b_completion_capable_shadow_trial(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage26_10b_boundary_rejections"
    assert summary["boundary_rejections"] == ["release_or_training_authorized"]
    assert summary["release_or_training_authorized"] is False
    assert summary["publishes_checkpoint"] is False
    assert summary["replaces_default_policy"] is False
    assert summary["connects_real_executor"] is False
    assert summary["starts_online_canary"] is False
    assert summary["canary_traffic_fraction"] == 0.0


def test_stage26_10b_hard_boundary_true_rejects_without_propagating_flag(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    config_path = _write_config(tmp_path, modifies_ppo_loss=True)
    summary = s26.run_xunce_stage26_10b_completion_capable_shadow_trial(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage26_10b_boundary_rejections"
    assert summary["boundary_rejections"] == ["modifies_ppo_loss"]
    assert summary["modifies_ppo_loss"] is False


def test_stage26_10b_terminal_off_control_zeroes_weights_hashes_and_preserves_baseline(
    tmp_path: Path,
) -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    baseline_path = REPO_ROOT / "configs" / "xunce_stage26_10_terminal_aware_ppo_reward_profile_v3.json"
    baseline_before = json.loads(baseline_path.read_text(encoding="utf-8"))
    config_path = _write_config(tmp_path)

    first = s26.run_xunce_stage26_10b_completion_capable_shadow_trial(
        config_path=config_path,
        output_root=tmp_path / "out1",
        repo_root=REPO_ROOT,
    )
    second = s26.run_xunce_stage26_10b_completion_capable_shadow_trial(
        config_path=config_path,
        output_root=tmp_path / "out2",
        repo_root=REPO_ROOT,
    )
    terminal_off = json.loads((tmp_path / "out1" / "profiles" / "terminal_off_control.json").read_text(encoding="utf-8"))
    baseline_after = json.loads(baseline_path.read_text(encoding="utf-8"))

    assert terminal_off["profile_id"] == "xunce-stage26-10b-terminal-off-control"
    assert terminal_off["weights"]["dead_end"] == 0.0
    assert terminal_off["weights"]["incomplete_terminal"] == 0.0
    assert terminal_off["weights"]["terminal_final_coverage"] == 0.0
    assert terminal_off["weights"]["success_99pct"] == 0.0
    assert terminal_off["weights"]["coverage_gain"] == baseline_before["weights"]["coverage_gain"]
    assert first["profile_hashes"]["terminal_off_control"] == second["profile_hashes"]["terminal_off_control"]
    assert baseline_after == baseline_before


def test_stage26_10b_l0_oracle_unreachable_routes_completion_reachability(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    monkeypatch.setattr(s26, "_load_l0_oracle_summary", lambda config, repo_root: {"completion_reachable": False})
    config_path = _write_config(tmp_path, stage26_10a_root=_write_stage26_10a_root(tmp_path))

    summary = s26.run_xunce_stage26_10b_completion_capable_shadow_trial(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_10b_completion_reachability"


def test_stage26_10b_missing_l0_fixture_catalog_routes_required_inputs(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    config_path = _write_config(
        tmp_path,
        stage26_10a_root=_write_stage26_10a_root(tmp_path),
        l0_fixture_catalog=str(tmp_path / "missing-fixture-catalog.jsonl"),
    )

    summary = s26.run_xunce_stage26_10b_completion_capable_shadow_trial(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_10b_required_inputs"
    assert "missing_l0_fixture_catalog" in summary["input_rejections"]


def test_stage26_10b_missing_explicit_source_oracle_summary_routes_required_inputs(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    config_path = _write_config(
        tmp_path,
        stage26_10a_root=_write_stage26_10a_root(tmp_path),
        source_oracle_summary=str(tmp_path / "missing-l0-oracle-summary.json"),
    )

    summary = s26.run_xunce_stage26_10b_completion_capable_shadow_trial(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_10b_required_inputs"
    assert "missing_source_oracle_summary" in summary["input_rejections"]


def test_stage26_10b_missing_baseline_profile_routes_required_inputs(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    config_path = _write_config(
        tmp_path,
        baseline_reward_profile=str(tmp_path / "missing-profile.json"),
        stage26_10a_root=_write_stage26_10a_root(tmp_path),
        l0_fixture_catalog=_write_l0_fixture_catalog(tmp_path),
    )

    summary = s26.run_xunce_stage26_10b_completion_capable_shadow_trial(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_10b_required_inputs"
    assert "missing_baseline_reward_profile" in summary["input_rejections"]


def test_stage26_10b_l0_oracle_proxy_runs_from_fixture_catalog(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    fixture_catalog = _write_l0_fixture_catalog(tmp_path)
    config = s26._load_config(
        _write_config(
            tmp_path,
            l0_fixture_catalog=fixture_catalog,
            l0_coverage_radius_cells=1,
            l0_replanning_cycle_limit=32,
            l0_segment_step_limit=8,
        ),
        REPO_ROOT,
    )

    summary = s26._run_l0_oracle_proxy(config, REPO_ROOT, tmp_path / "l0")

    assert summary["completion_reachable"] is True
    assert summary["source"] == "coverage_memory_replanning_loop_oracle_proxy/v1"
    assert summary["scenario_count"] == 1
    assert (tmp_path / "l0" / "l0-global-99-source-config.json").is_file()
    source_config = json.loads((tmp_path / "l0" / "l0-global-99-source-config.json").read_text(encoding="utf-8"))
    assert source_config["scenarios"][0]["l0_proxy_source"] == "stage26_8g_fixture_sidecar_passable_mask/v1"


def test_stage26_10b_l0_oracle_proxy_reuses_cached_summary(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    cached = {"completion_reachable": True, "source": "cached-l0"}
    l0_root = tmp_path / "l0"
    l0_root.mkdir()
    (l0_root / "l0-oracle-summary.json").write_text(json.dumps(cached), encoding="utf-8")

    def fail_if_called(**kwargs):
        raise AssertionError("coverage memory proxy should not rerun when cached summary exists")

    monkeypatch.setattr(s26.coverage_memory_loop, "run_coverage_memory_replanning_loop", fail_if_called)

    summary = s26._run_l0_oracle_proxy({"l0_fixture_catalog": _write_l0_fixture_catalog(tmp_path)}, REPO_ROOT, l0_root)

    assert summary == cached


def test_stage26_10b_l0_oracle_reachable_writes_paired_8m_configs_and_advances_one_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    calls: list[dict] = []

    def fake_stage26_8m(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        batch_dir = output_root / "job" / "collector" / "stage21_3"
        batch_dir.mkdir(parents=True)
        batch_row = {
            "transition_id": "terminal-success",
            "done": True,
            "transition_trainable": True,
            "reward_trainable": True,
            "return": 5.0,
            "advantage": 1.0,
            "reward_components": {"success_99pct_bonus_component": 5.0},
        }
        (batch_dir / "xunce-stage21-3-ppo-trainable-batch.jsonl").write_text(
            json.dumps(batch_row) + "\n",
            encoding="utf-8",
        )
        return_row = {"transition_id": "terminal-success", "done": True, "return": 5.0, "advantage": 1.0}
        (batch_dir / "xunce-stage21-3-return-advantage-audit.jsonl").write_text(
            json.dumps(return_row) + "\n",
            encoding="utf-8",
        )
        calls.append({"config_path": str(config_path), "output_root": str(output_root), "payload": payload})
        return {
            "status": "passed",
            "next_required_change": "continue_stage26_8m_jobs",
            "pending_job_count": 1,
            "completed_job_count": 0,
            "main_coverage_per_100m_delta": 0.0,
            "final_coverage_delta": 0.0,
            "success_99pct_count_delta": 0,
            "binding_or_safety_failure": False,
        }

    monkeypatch.setattr(s26, "_load_l0_oracle_summary", lambda config, repo_root: {"completion_reachable": True})
    monkeypatch.setattr(s26, "_invoke_stage26_8m", fake_stage26_8m)
    config_path = _write_config(tmp_path, stage26_10a_root=_write_stage26_10a_root(tmp_path))

    summary = s26.run_xunce_stage26_10b_completion_capable_shadow_trial(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    configs = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((tmp_path / "out" / "stage26_8m_configs").glob("*.json"))
    }
    assert summary["next_required_change"] == "continue_stage26_10b_completion_shadow_trial"
    assert len(calls) == 1
    assert set(configs) >= {"balanced_high.json", "terminal_off_control.json", "baseline_v3.json"}
    balanced = configs["balanced_high.json"]
    terminal_off = configs["terminal_off_control.json"]
    baseline = configs["baseline_v3.json"]
    for key in ("horizons", "seeds", "scenario_counts", "collector_rollout_steps", "eval_rollout_steps", "update_combos"):
        assert balanced[key] == terminal_off[key]
        assert balanced[key] == baseline[key]
    assert "balanced_high.json" in balanced["coverage_first_reward_profile"]
    assert "terminal_off_control.json" in terminal_off["coverage_first_reward_profile"]
    assert "baseline_v3.json" in baseline["coverage_first_reward_profile"]
    for filename, profile_filename in {
        "balanced_high-stage26-1-base.json": "balanced_high.json",
        "terminal_off_control-stage26-1-base.json": "terminal_off_control.json",
        "baseline_v3-stage26-1-base.json": "baseline_v3.json",
    }.items():
        derived_base = configs[filename]
        assert profile_filename in derived_base["coverage_first_reward_profile"]
        assert filename in configs[filename.replace("-stage26-1-base", "")]["base_stage26_1_config"]
    assert balanced["max_jobs_per_invocation"] == 1
    assert terminal_off["max_jobs_per_invocation"] == 1
    assert baseline["modifies_ppo_loss"] is False
    assert baseline["modifies_network"] is False
    assert baseline["modifies_action_space"] is False
    assert baseline["modifies_hybrid_astar_pose_gate"] is False
    assert baseline["modifies_candidate_generation"] is False
    assert balanced["candidate_reachability_gate_source"] == "hybrid_astar_pose_reachability/v1"
    assert balanced["candidate_reachability_max_theta_proposals_per_candidate"] == 5
    assert balanced["candidate_reachability_theta_proposal_policy"] == "candidate_current_bearing_sweep/v1"
    assert balanced["hybrid_astar_planning_grid_source"] == "derived_high_res_planning_proxy/v1"
    assert balanced["planner_grid_resolution_m"] == 1.0
    assert balanced["hybrid_astar_max_iterations"] == 200


def test_stage26_10b_existing_matrix_update_is_not_lost_after_one_job(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    def fake_stage26_8m(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        batch_dir = output_root / "job" / "collector" / "stage21_3"
        batch_dir.mkdir(parents=True)
        batch_row = {
            "transition_id": "terminal-success",
            "done": True,
            "transition_trainable": True,
            "reward_trainable": True,
            "return": 5.0,
            "advantage": 1.0,
            "reward_components": {"success_99pct_bonus_component": 5.0},
        }
        (batch_dir / "xunce-stage21-3-ppo-trainable-batch.jsonl").write_text(json.dumps(batch_row) + "\n", encoding="utf-8")
        (batch_dir / "xunce-stage21-3-return-advantage-audit.jsonl").write_text(
            json.dumps({"transition_id": "terminal-success", "done": True, "return": 5.0, "advantage": 1.0}) + "\n",
            encoding="utf-8",
        )
        return {
            "status": "passed",
            "next_required_change": "continue_stage26_8m_jobs",
            "pending_job_count": 1,
            "completed_job_count": 0,
            "binding_or_safety_failure": False,
        }

    monkeypatch.setattr(s26, "_load_l0_oracle_summary", lambda config, repo_root: {"completion_reachable": True})
    monkeypatch.setattr(s26, "_invoke_stage26_8m", fake_stage26_8m)
    output_root = tmp_path / "out"
    output_root.mkdir()
    rows = [
        _matrix_row("l0_oracle", level="L0", status="complete"),
        _matrix_row("balanced_high", level="L1", status="pending"),
        _matrix_row("terminal_heavy", level="L2", status="pending"),
    ]
    (output_root / "xunce-stage26-10b-matrix.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, stage26_10a_root=_write_stage26_10a_root(tmp_path))

    s26.run_xunce_stage26_10b_completion_capable_shadow_trial(
        config_path=config_path,
        output_root=output_root,
        repo_root=REPO_ROOT,
    )
    persisted = [row for row in _read_jsonl(output_root / "xunce-stage26-10b-matrix.jsonl") if row["profile_id"] == "balanced_high"][0]

    assert persisted["stage26_8m_status"] == "passed"
    assert persisted["status"] == "pending"


def test_stage26_10b_l2_incomplete_cannot_route_readiness() -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    rows = [
        _matrix_row("l0_oracle", level="L0", status="complete"),
        _matrix_row("balanced_high", level="L1", status="complete"),
        _matrix_row("terminal_off_control", level="L1", status="complete"),
        _matrix_row("baseline_v3", level="L2", status="complete"),
        _matrix_row("terminal_heavy", level="L2", status="pending"),
    ]

    assert s26._route([], [], {"completion_reachable": True}, rows) == "continue_stage26_10b_completion_shadow_trial"


def test_stage26_10b_l1_control_pending_or_missing_cannot_route_readiness() -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    pending_control = [
        _matrix_row("l0_oracle", level="L0", status="complete"),
        _completion_row("terminal_off_control", level="L1", success=0.20, steps=900.0, path=300.0),
        _completion_row("balanced_high", level="L1", success=0.35, steps=820.0, path=310.0),
        _completion_row("baseline_v3", level="L2", success=0.22, steps=890.0, path=300.0),
        _completion_row("terminal_heavy", level="L2", success=0.30, steps=850.0, path=305.0),
        _completion_row("dead_end_high", level="L2", success=0.26, steps=870.0, path=307.0),
    ]
    pending_control[1]["status"] = "pending"
    missing_control = [row for row in pending_control if row["profile_id"] != "terminal_off_control"]

    assert s26._route([], [], {"completion_reachable": True}, pending_control) == "continue_stage26_10b_completion_shadow_trial"
    assert s26._route([], [], {"completion_reachable": True}, missing_control) == "continue_stage26_10b_completion_shadow_trial"


def test_stage26_10b_failed_stage26_8m_row_routes_eval_repair() -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    rows = [_matrix_row("balanced_high", level="L1", status="failed", binding_or_safety_failure=False)]

    assert s26._route([], [], {"completion_reachable": True}, rows) == "repair_stage26_10b_eval_binding_or_safety"


def test_stage26_10b_stage26_8m_phase_execution_failure_routes_eval_not_terminal_credit(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    summary = {
        "status": "passed",
        "next_required_change": "continue_stage26_8m_jobs",
        "pending_job_count": 6,
        "completed_job_count": 0,
        "binding_or_safety_failure": False,
        "phase_executions": [
            {
                "phase": "collector",
                "status": "failed",
                "blocking_reason": "ConfigError:hybrid_astar_planning_grid_source must be a non-empty string",
            }
        ],
    }
    row = s26._row_from_stage26_8m_summary(
        _matrix_row("terminal_off_control", level="L1", status="pending"),
        summary,
        {"missing": True, "terminal_credit_missing_flag_count": 3},
        tmp_path,
    )

    assert row["status"] == "failed"
    assert row["binding_or_safety_failure"] is True
    assert s26._route([], [], {"completion_reachable": True}, [row]) == "repair_stage26_10b_eval_binding_or_safety"


def test_stage26_10b_l2_complete_missing_completion_metrics_routes_metric_binding() -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    rows = [
        _matrix_row("l0_oracle", level="L0", status="complete"),
        _matrix_row("terminal_off_control", level="L1", status="complete"),
        _matrix_row("balanced_high", level="L1", status="complete"),
        _matrix_row("baseline_v3", level="L2", status="complete"),
        _matrix_row("terminal_heavy", level="L2", status="complete"),
        _matrix_row("dead_end_high", level="L2", status="complete"),
    ]

    assert s26._route([], [], {"completion_reachable": True}, rows) == "repair_stage26_10b_completion_metric_binding"


def test_stage26_10b_l2_complete_success_improvement_routes_readiness() -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    rows = [
        _matrix_row("l0_oracle", level="L0", status="complete"),
        _completion_row("terminal_off_control", level="L1", success=0.20, steps=900.0, path=300.0),
        _completion_row("balanced_high", level="L1", success=0.35, steps=820.0, path=310.0),
        _completion_row("baseline_v3", level="L2", success=0.22, steps=890.0, path=300.0),
        _completion_row("terminal_heavy", level="L2", success=0.30, steps=850.0, path=305.0),
        _completion_row("dead_end_high", level="L2", success=0.26, steps=870.0, path=307.0),
    ]

    assert s26._route([], [], {"completion_reachable": True}, rows) == "review_stage26_10b_completion_shadow_readiness"


def test_stage26_10b_success_improvement_with_negative_final_coverage_routes_credit_repair() -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    rows = [
        _matrix_row("l0_oracle", level="L0", status="complete"),
        _completion_row("terminal_off_control", level="L1", success=0.20, steps=900.0, path=300.0),
        _completion_row("balanced_high", level="L1", success=0.35, steps=820.0, path=310.0, final_delta=-0.005),
        _completion_row("baseline_v3", level="L2", success=0.22, steps=890.0, path=300.0),
        _completion_row("terminal_heavy", level="L2", success=0.30, steps=850.0, path=305.0),
        _completion_row("dead_end_high", level="L2", success=0.26, steps=870.0, path=307.0),
    ]

    assert s26._route([], [], {"completion_reachable": True}, rows) == "repair_stage26_terminal_credit_assignment"


def test_stage26_10b_l2_complete_flat_success_but_positive_per100m_routes_short_sighted() -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    rows = [
        _matrix_row("l0_oracle", level="L0", status="complete"),
        _completion_row("terminal_off_control", level="L1", success=0.20, steps=900.0, path=300.0),
        _completion_row("balanced_high", level="L1", success=0.20, steps=890.0, path=300.0, per100=0.2, final_delta=0.0),
        _completion_row("baseline_v3", level="L2", success=0.20, steps=900.0, path=300.0),
        _completion_row("terminal_heavy", level="L2", success=0.20, steps=900.0, path=300.0),
        _completion_row("dead_end_high", level="L2", success=0.20, steps=900.0, path=300.0),
    ]

    assert s26._route([], [], {"completion_reachable": True}, rows) == "repair_stage26_10b_short_sighted_efficiency"


def test_stage26_10b_terminal_credit_audit_missing_or_nonzero_flags_routes_credit_assignment() -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    rows = [_matrix_row("balanced_high", level="L1", status="complete", terminal_credit_missing_flag_count=1)]

    assert s26._route([], [], {"completion_reachable": True}, rows) == "repair_stage26_terminal_credit_assignment"
    assert (
        s26._route([], [], {"completion_reachable": True}, [_matrix_row("balanced_high", level="L1", status="complete", terminal_credit_audit_missing=True)])
        == "repair_stage26_terminal_credit_assignment"
    )


def test_stage26_10b_terminal_credit_audit_scans_stage21_3_batch_and_return_fields(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    batch_dir = tmp_path / "job" / "collector" / "stage21_3"
    batch_dir.mkdir(parents=True)
    batch_rows = [
        {
            "transition_id": "terminal-success",
            "done": True,
            "transition_trainable": True,
            "reward_trainable": True,
            "return": 7.5,
            "advantage": 1.25,
            "reward_components": {
                "success_99pct_bonus_component": 5.0,
                "incomplete_terminal_penalty_component": 0.0,
                "dead_end_penalty_component": 0.0,
            },
        }
    ]
    (batch_dir / "xunce-stage21-3-ppo-trainable-batch.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in batch_rows),
        encoding="utf-8",
    )
    (batch_dir / "xunce-stage21-3-return-advantage-audit.jsonl").write_text(
        json.dumps({"transition_id": "terminal-success", "done": True, "return": 7.5, "advantage": 1.25}) + "\n",
        encoding="utf-8",
    )

    audit = s26._terminal_credit_audit_from_stage26_8m_output(tmp_path)

    assert audit["missing"] is False
    assert audit["stage21_3_batch_file_count"] == 1
    assert audit["done_terminal_transition_count"] == 1
    assert audit["terminal_trainable_row_count"] == 1
    assert audit["success_99pct_bonus_nonzero_count"] == 1
    assert audit["terminal_return_advantage_row_count"] == 1
    assert audit["terminal_credit_missing_flag_count"] == 0


def test_stage26_10b_terminal_off_control_does_not_require_nonzero_terminal_component(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    batch_dir = tmp_path / "job" / "collector" / "stage21_3"
    batch_dir.mkdir(parents=True)
    batch_row = {
        "transition_id": "terminal-control",
        "done": True,
        "transition_trainable": True,
        "reward_trainable": True,
        "return": 1.0,
        "advantage": 0.5,
        "reward_components": {
            "success_99pct_bonus_component": 0.0,
            "incomplete_terminal_penalty_component": 0.0,
            "dead_end_penalty_component": 0.0,
        },
    }
    (batch_dir / "xunce-stage21-3-ppo-trainable-batch.jsonl").write_text(json.dumps(batch_row) + "\n", encoding="utf-8")
    (batch_dir / "xunce-stage21-3-return-advantage-audit.jsonl").write_text(
        json.dumps({"transition_id": "terminal-control", "done": True, "return": 1.0, "advantage": 0.5}) + "\n",
        encoding="utf-8",
    )

    audit = s26._terminal_credit_audit_from_stage26_8m_output(tmp_path, expect_terminal_reward_components=False)

    assert audit["missing"] is False
    assert audit["terminal_credit_missing_flag_count"] == 0


def test_stage26_10b_terminal_credit_audit_requires_return_advantage_file(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    batch_dir = tmp_path / "job" / "collector" / "stage21_3"
    batch_dir.mkdir(parents=True)
    batch_row = {
        "transition_id": "terminal-success",
        "done": True,
        "transition_trainable": True,
        "reward_trainable": True,
        "return": 7.5,
        "advantage": 1.25,
        "reward_components": {"success_99pct_bonus_component": 5.0},
    }
    (batch_dir / "xunce-stage21-3-ppo-trainable-batch.jsonl").write_text(json.dumps(batch_row) + "\n", encoding="utf-8")

    audit = s26._terminal_credit_audit_from_stage26_8m_output(tmp_path)

    assert audit["missing"] is True
    assert audit["stage21_3_return_advantage_audit_file_count"] == 0
    assert audit["terminal_credit_missing_flag_count"] > 0


def test_stage26_10b_terminal_credit_audit_requires_each_trainable_terminal_in_return_audit(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    batch_dir = tmp_path / "job" / "collector" / "stage21_3"
    batch_dir.mkdir(parents=True)
    rows = [
        {
            "transition_id": "nontrainable-with-return",
            "done": True,
            "transition_trainable": False,
            "reward_trainable": True,
            "return": 1.0,
            "advantage": 0.1,
            "reward_components": {"success_99pct_bonus_component": 5.0},
        },
        {
            "transition_id": "trainable-missing-return-audit",
            "done": True,
            "transition_trainable": True,
            "reward_trainable": True,
            "return": 2.0,
            "advantage": 0.2,
            "reward_components": {"success_99pct_bonus_component": 5.0},
        },
    ]
    (batch_dir / "xunce-stage21-3-ppo-trainable-batch.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (batch_dir / "xunce-stage21-3-return-advantage-audit.jsonl").write_text(
        json.dumps({"transition_id": "nontrainable-with-return", "done": True, "return": 1.0, "advantage": 0.1}) + "\n",
        encoding="utf-8",
    )

    audit = s26._terminal_credit_audit_from_stage26_8m_output(tmp_path)

    assert audit["terminal_trainable_row_count"] == 1
    assert audit["terminal_return_advantage_row_count"] == 0
    assert audit["missing"] is True


def test_stage26_10b_terminal_credit_audit_flags_missing_stage21_3_batch(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    audit = s26._terminal_credit_audit_from_stage26_8m_output(tmp_path)

    assert audit["missing"] is True
    assert audit["stage21_3_batch_file_count"] == 0
    assert audit["terminal_credit_missing_flag_count"] > 0


def test_stage26_10b_per100m_positive_but_final_and_success_flat_routes_short_sighted_efficiency() -> None:
    import scripts.run_xunce_stage26_10b_completion_capable_shadow_trial as s26

    rows = [
        _matrix_row(
            "balanced_high",
            level="L1",
            status="complete",
            main_coverage_per_100m_delta=0.25,
            final_coverage_delta=0.0,
            success_99pct_count_delta=0,
        )
    ]

    assert s26._route([], [], {"completion_reachable": True}, rows) == "repair_stage26_10b_short_sighted_efficiency"


def _write_config(tmp_path: Path, **overrides) -> Path:
    payload = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "base_stage26_8m_config": "configs/xunce_stage26_8m_generalized_resumable_training_pipeline_v1.json",
        "baseline_reward_profile": "configs/xunce_stage26_10_terminal_aware_ppo_reward_profile_v3.json",
        "stage26_10a_root": "",
        "source_oracle_summary": "",
        "source_policy_checkpoint_path": "outputs/path_feedback_batch_xunce_sandbox_candidate_preflight_v1/sandbox_package/xunce-controlled-training-candidate.pt",
        "source_policy_sha256": "366bc8f004879813a3500205ac048e4db73a77a68a7e326862098641880ecf3b",
        "source_policy_is_random_untrained": False,
        "source_policy_network_architecture": "xunce_full_network_v1",
        "horizons": [1024],
        "seeds": [261001],
        "scenario_counts": [3],
        "collector_rollout_steps": [1024],
        "eval_rollout_steps": [1024],
        "update_combos": [{"combo_id": "completion_shadow", "epochs": 1, "learning_rate": 1.0e-5}],
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage26_10b_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_stage26_10a_root(tmp_path: Path) -> str:
    root = tmp_path / "stage26_10a"
    profiles = root / "profiles"
    profiles.mkdir(parents=True)
    baseline = json.loads(
        (REPO_ROOT / "configs" / "xunce_stage26_10_terminal_aware_ppo_reward_profile_v3.json").read_text(
            encoding="utf-8"
        )
    )
    rows = []
    for variant, weights in {
        "balanced_high": {"dead_end": 4.0, "incomplete_terminal": 2.0, "terminal_final_coverage": 3.0, "success_99pct": 8.0},
        "terminal_heavy": {"dead_end": 2.0, "incomplete_terminal": 2.0, "terminal_final_coverage": 3.0, "success_99pct": 8.0},
        "dead_end_high": {"dead_end": 4.0, "incomplete_terminal": 1.0, "terminal_final_coverage": 2.0, "success_99pct": 5.0},
    }.items():
        payload = json.loads(json.dumps(baseline))
        payload["profile_id"] = f"xunce-stage26-10a-{variant}"
        payload["weights"].update(weights)
        path = profiles / f"{variant}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({"variant_id": variant, "profile_id": payload["profile_id"], "profile_path": str(path)})
    (root / "xunce-stage26-10a-reward-weight-sweep-matrix.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return str(root)


def _write_l0_fixture_catalog(tmp_path: Path) -> str:
    root = tmp_path / "l0_fixture"
    root.mkdir()
    sidecar = {
        "schema_version": "path-planner-sidecar/v1",
        "cost": [[1.0 for _ in range(4)] for _ in range(4)],
        "passable_mask": [[True for _ in range(4)] for _ in range(4)],
        "blocked_cells": [],
    }
    sidecar_path = root / "sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    row = {
        "schema_version": "xunce-stage26-scenario-fixture/v1",
        "scenario_id": "unit_l0",
        "scenario_start_cell": [0, 0],
        "source_sidecar": str(sidecar_path),
    }
    catalog_path = root / "xunce-stage26-8g-scenario-fixture-catalog.jsonl"
    catalog_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return str(catalog_path)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _matrix_row(profile_id: str, *, level: str, status: str, **overrides) -> dict:
    row = {
        "profile_id": profile_id,
        "level": level,
        "status": status,
        "binding_or_safety_failure": False,
        "terminal_credit_audit_missing": False,
        "terminal_credit_missing_flag_count": 0,
        "main_coverage_per_100m_delta": 0.0,
        "final_coverage_delta": 0.1,
        "success_99pct_count_delta": 1,
    }
    row.update(overrides)
    return row


def _completion_row(
    profile_id: str,
    *,
    level: str,
    success: float,
    steps: float,
    path: float,
    per100: float = 0.0,
    final_delta: float = 0.02,
) -> dict:
    return _matrix_row(
        profile_id,
        level=level,
        status="complete",
        success_99pct_rate=success,
        median_steps_to_99pct=steps,
        median_path_cost_to_99pct=path,
        final_coverage_rate_delta=final_delta,
        worst_seed_final_coverage_rate_delta=max(final_delta, -0.005),
        scenario_regression_rate=0.0,
        main_coverage_per_100m_delta=per100,
    )
