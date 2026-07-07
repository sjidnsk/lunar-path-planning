import json
from pathlib import Path

import pytest

import scripts.run_xunce_stage26_8m_generalized_resumable_training_pipeline as stage26_8m


def test_config_rejects_bad_stage_id(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"stage_id": "wrong-stage"})

    with pytest.raises(ValueError, match="stage_id"):
        stage26_8m._load_config(config_path, _repo_root())


def test_config_rejects_boundary_true(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"publishes_checkpoint": True})
    config = stage26_8m._load_config(config_path, _repo_root())

    assert stage26_8m._boundary_rejections(config) == ["publishes_checkpoint"]


def test_config_rejects_kl_above_limit(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"max_abs_approx_kl": 1.5001})

    with pytest.raises(ValueError, match="max_abs_approx_kl"):
        stage26_8m._load_config(config_path, _repo_root())


def test_job_expansion_is_deterministic_and_includes_combo_id(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "horizons": [12, 16],
            "seeds": [260801, 260802],
            "update_combos": [
                {
                    "combo_id": "policy_amp_x2",
                    "epochs": 16,
                    "learning_rate": 0.00003,
                    "policy_loss_coefficient": 3.0,
                    "value_loss_coefficient": 0.01,
                    "entropy_coefficient": 0.005,
                    "loss_scale": 0.5,
                },
                {
                    "combo_id": "lr_x3",
                    "epochs": 8,
                    "learning_rate": 0.00003,
                    "policy_loss_coefficient": 1.0,
                    "value_loss_coefficient": 0.02,
                    "entropy_coefficient": 0.01,
                    "loss_scale": 0.25,
                },
            ],
        },
    )
    config = stage26_8m._load_config(config_path, _repo_root())

    jobs1 = stage26_8m._expand_jobs(config, tmp_path / "out", _repo_root())
    jobs2 = stage26_8m._expand_jobs(config, tmp_path / "out", _repo_root())

    assert len(jobs1) == 8
    assert [job["job_id"] for job in jobs1] == [job["job_id"] for job in jobs2]
    assert "h16_s260802_sc3_cr16_er16_u_lr_x3" in {job["job_id"] for job in jobs1}


def test_aggregate_only_missing_summaries_routes_continue(tmp_path: Path) -> None:
    summary = stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=_write_config(tmp_path, {"run_mode": "aggregate_only"}),
        output_root=tmp_path / "out",
        repo_root=_repo_root(),
    )
    rows = _read_jsonl(tmp_path / "out" / stage26_8m.JOB_STATE_FILE)

    assert summary["next_required_change"] == stage26_8m.ROUTE_CONTINUE
    assert summary["next_phase"] == "collector"
    assert rows[0]["phase"] == "collector"
    assert rows[0]["status"] == "pending"


def test_run_next_executes_only_one_phase(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_collector(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        calls.append("collector")
        summary = {"schema_version": "xunce-stage26-1-summary/v1", "status": "passed", "next_required_change": "run_stage26_2_synthetic_terrain_ppo_update_smoke"}
        _write_json(output_root / stage26_8m.stage26_1.SUMMARY_FILE, summary)
        return summary

    monkeypatch.setattr(stage26_8m.stage26_1, "run_xunce_stage26_1_synthetic_terrain_collector_smoke", fake_collector)

    summary = stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=_write_config(tmp_path, {}),
        output_root=tmp_path / "out",
        repo_root=_repo_root(),
    )
    rows = _read_jsonl(tmp_path / "out" / stage26_8m.JOB_STATE_FILE)

    assert calls == ["collector"]
    assert summary["phase_execution_count"] == 1
    assert _state(rows, "collector")["status"] == "complete"
    assert _state(rows, "update")["status"] == "pending"


def test_collector_reuse_policy_shares_collector_across_update_combos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[Path] = []

    def fake_collector(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        calls.append(output_root)
        summary = {
            "schema_version": "xunce-stage26-1-summary/v1",
            "status": "passed",
            "next_required_change": "run_stage26_2_synthetic_terrain_ppo_update_smoke",
            "trainable_transition_count": 120,
        }
        _write_json(output_root / stage26_8m.stage26_1.SUMMARY_FILE, summary)
        return summary

    monkeypatch.setattr(stage26_8m.stage26_1, "run_xunce_stage26_1_synthetic_terrain_collector_smoke", fake_collector)
    config_path = _write_config(
        tmp_path,
        {
            "collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1",
            "max_jobs_per_invocation": 1,
            "update_combos": [_combo("a"), _combo("b")],
        },
    )
    output_root = tmp_path / "out"

    first = stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=config_path,
        output_root=output_root,
        repo_root=_repo_root(),
    )
    second = stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=config_path,
        output_root=output_root,
        repo_root=_repo_root(),
    )
    rows = _read_jsonl(output_root / stage26_8m.JOB_STATE_FILE)

    assert len(calls) == 1
    assert calls[0].parent == output_root
    assert calls[0].name.startswith("c")
    assert first["phase_execution_count"] == 1
    assert second["next_phase"] == "update"
    assert {row["output_root"] for row in rows if row["phase"] == "collector"} == {str(calls[0])}


def test_collector_reuse_policy_update_config_points_to_shared_collector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_update_config: dict = {}

    def fake_update(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        captured_update_config.update(json.loads(config_path.read_text(encoding="utf-8")))
        summary = _update_summary(output_root)
        _write_json(output_root / stage26_8m.stage26_2.SUMMARY_FILE, summary)
        return summary

    monkeypatch.setattr(stage26_8m.stage26_2, "run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke", fake_update)
    config_path = _write_config(
        tmp_path,
        {
            "collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1",
            "run_mode": "run_phase",
            "phase": "update",
        },
    )
    config = stage26_8m._load_config(config_path, _repo_root())
    job = stage26_8m._expand_jobs(config, tmp_path / "out", _repo_root())[0]
    _write_collector_complete(Path(job["collector_root"]))
    _write_collector_reuse_marker(Path(job["collector_root"]), job)

    stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=_repo_root(),
        job_id_override=job["job_id"],
    )

    assert Path(captured_update_config["stage26_1_root"]) == Path(job["collector_root"])


def test_collector_reuse_policy_ignores_stale_input_hash_and_reruns_current_root(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {"collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1"},
    )
    output_root = tmp_path / "out"
    config = stage26_8m._load_config(config_path, _repo_root())
    job = stage26_8m._expand_jobs(config, output_root, _repo_root())[0]
    _write_collector_complete(Path(job["collector_root"]))
    _write_collector_reuse_marker(Path(job["collector_root"]), job)
    _write_jsonl(
        output_root / stage26_8m.JOB_STATE_FILE,
        [
            {
                "job_id": job["job_id"],
                "phase": "collector",
                "status": "complete",
                "config_hash": job["phase_config_hashes"]["collector"],
                "input_hash": "stale-input-hash",
                "output_root": job["collector_root"],
                "summary_path": str(Path(job["collector_root"]) / stage26_8m.stage26_1.SUMMARY_FILE),
            }
        ],
    )

    summary = stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=config_path,
        output_root=output_root,
        repo_root=_repo_root(),
        run_mode_override="aggregate_only",
    )
    rows = _read_jsonl(output_root / stage26_8m.JOB_STATE_FILE)

    assert summary["next_required_change"] == stage26_8m.ROUTE_CONTINUE
    assert summary["resume_state_rejections"] == []
    assert _state(rows, "collector")["status"] == "pending"
    assert _state(rows, "collector")["blocking_reason"] == "stale_resume_state_ignored"


def test_collector_reuse_policy_ignores_stale_state_for_old_root(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {"collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1"},
    )
    output_root = tmp_path / "out"
    config = stage26_8m._load_config(config_path, _repo_root())
    job = stage26_8m._expand_jobs(config, output_root, _repo_root())[0]
    old_root = output_root / "shared_collectors" / "h16_s260801_sc3_cr16_oldhash"
    _write_jsonl(
        output_root / stage26_8m.JOB_STATE_FILE,
        [
            {
                "job_id": job["job_id"],
                "phase": "collector",
                "status": "complete",
                "config_hash": job["phase_config_hashes"]["collector"],
                "input_hash": "stale-input-hash",
                "output_root": str(old_root),
                "summary_path": str(old_root / stage26_8m.stage26_1.SUMMARY_FILE),
            }
        ],
    )

    summary = stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=config_path,
        output_root=output_root,
        repo_root=_repo_root(),
        run_mode_override="aggregate_only",
    )

    assert summary["next_required_change"] == stage26_8m.ROUTE_CONTINUE
    assert summary["resume_state_rejections"] == []


def test_collector_reuse_policy_collector_hash_ignores_update_combo_change(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    config_path = _write_config(
        tmp_path,
        {
            "collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1",
            "update_combos": [_combo("a") | {"epochs": 4}],
        },
    )
    old_config = stage26_8m._load_config(config_path, _repo_root())
    old_job = stage26_8m._expand_jobs(old_config, output_root, _repo_root())[0]
    _write_collector_complete(Path(old_job["collector_root"]))
    _write_collector_reuse_marker(Path(old_job["collector_root"]), old_job)
    _write_jsonl(
        output_root / stage26_8m.JOB_STATE_FILE,
        [
            {
                "job_id": old_job["job_id"],
                "phase": "collector",
                "status": "complete",
                "config_hash": old_job["phase_config_hashes"]["collector"],
                "input_hash": old_job["input_hash"],
                "output_root": old_job["collector_root"],
                "summary_path": str(Path(old_job["collector_root"]) / stage26_8m.stage26_1.SUMMARY_FILE),
            }
        ],
    )
    new_config_path = _write_config(
        tmp_path,
        {
            "collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1",
            "update_combos": [_combo("a") | {"epochs": 16}],
        },
    )

    summary = stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=new_config_path,
        output_root=output_root,
        repo_root=_repo_root(),
        run_mode_override="aggregate_only",
    )
    rows = _read_jsonl(output_root / stage26_8m.JOB_STATE_FILE)

    assert summary["resume_state_rejections"] == []
    assert _state(rows, "collector")["status"] == "complete"


def test_planner_overrides_enter_generated_stage26_1_collector_config(tmp_path: Path) -> None:
    overrides = _planner_overrides()
    config = stage26_8m._load_config(_write_config(tmp_path, overrides), _repo_root())
    job = stage26_8m._expand_jobs(config, tmp_path / "out", _repo_root())[0]

    generated = stage26_8m._build_collector_config(config, job, tmp_path / "cfg", _repo_root())

    assert tuple(stage26_8m.PLANNER_OVERRIDE_FIELDS) == tuple(overrides)
    for field, expected in overrides.items():
        assert generated[field] == expected


def test_planner_overrides_enter_stage26_3_eval_base_config(tmp_path: Path) -> None:
    overrides = _planner_overrides()
    config = stage26_8m._load_config(_write_config(tmp_path, overrides), _repo_root())
    job = stage26_8m._expand_jobs(config, tmp_path / "out", _repo_root())[0]

    generated = stage26_8m._build_stage26_3_base_config(config, job, tmp_path / "out" / "job", _repo_root())

    for field, expected in overrides.items():
        assert generated[field] == expected


def test_planner_proxy_resolution_requires_planning_grid_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hybrid_astar_planning_grid_source"):
        stage26_8m._load_config(_write_config(tmp_path, {"planner_grid_resolution_m": 1.0}), _repo_root())


def test_planner_override_rejects_invalid_planning_grid_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hybrid_astar_planning_grid_source"):
        stage26_8m._load_config(
            _write_config(tmp_path, {"hybrid_astar_planning_grid_source": "sidecar_cost_grid/v1"}),
            _repo_root(),
        )


def test_update_config_does_not_consume_planner_overrides(tmp_path: Path) -> None:
    config = stage26_8m._load_config(_write_config(tmp_path, _planner_overrides()), _repo_root())
    job = stage26_8m._expand_jobs(config, tmp_path / "out", _repo_root())[0]

    generated = stage26_8m._build_update_config(config, job, tmp_path / "out" / "job", _repo_root())

    assert not any(field in generated for field in stage26_8m.PLANNER_OVERRIDE_FIELDS)
    assert "candidate_reachability_gate_source" not in generated
    assert "candidate_reachability_theta_proposal_policy" not in generated


def test_candidate_reachability_gate_source_enters_generated_collector_and_eval_configs(tmp_path: Path) -> None:
    config = stage26_8m._load_config(
        _write_config(
            tmp_path,
            {
                "candidate_reachability_gate_source": "hybrid_astar_pose_reachability/v1",
                "candidate_reachability_max_theta_proposals_per_candidate": 1,
                "candidate_reachability_theta_proposal_policy": "candidate_current_bearing_sweep/v1",
                **_planner_overrides(),
            },
        ),
        _repo_root(),
    )
    output_root = tmp_path / "out"
    job = stage26_8m._expand_jobs(config, output_root, _repo_root())[0]
    job_root = output_root / "job"
    config_root = job_root / "configs"
    config_root.mkdir(parents=True)

    collector = stage26_8m._build_collector_config(config, job, config_root, _repo_root())
    stage26_3_config = stage26_8m._build_stage26_3_base_config(config, job, job_root, _repo_root())
    stage21_5_config = stage26_8m._build_stage21_5_eval_config(config, job, job_root, config_root, _repo_root())

    assert collector["candidate_reachability_gate_source"] == "hybrid_astar_pose_reachability/v1"
    assert collector["candidate_reachability_max_theta_proposals_per_candidate"] == 1
    assert collector["candidate_reachability_theta_proposal_policy"] == "candidate_current_bearing_sweep/v1"
    assert collector["selected_continuous_theta_reachability_guard_enabled"] is True
    assert collector["selected_continuous_theta_unreachable_resample_policy"] == "reachable_theta_proposal/v1"
    assert stage26_3_config["candidate_reachability_gate_source"] == "hybrid_astar_pose_reachability/v1"
    assert stage26_3_config["candidate_reachability_theta_proposal_policy"] == "candidate_current_bearing_sweep/v1"
    assert stage21_5_config["candidate_reachability_gate_source"] == "hybrid_astar_pose_reachability/v1"
    assert stage21_5_config["candidate_reachability_theta_proposal_policy"] == "candidate_current_bearing_sweep/v1"


def test_input_hash_changes_when_candidate_reachability_gate_source_changes(tmp_path: Path) -> None:
    legacy_config = stage26_8m._load_config(
        _write_config(tmp_path, {"collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1"}),
        _repo_root(),
    )
    hard_gate_config = stage26_8m._load_config(
        _write_config(
            tmp_path,
            {
                "collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1",
                "candidate_reachability_gate_source": "hybrid_astar_pose_reachability/v1",
            },
        ),
        _repo_root(),
    )
    output_root = tmp_path / "out"
    legacy_job = stage26_8m._expand_jobs(legacy_config, output_root, _repo_root())[0]
    hard_gate_job = stage26_8m._expand_jobs(hard_gate_config, output_root, _repo_root())[0]

    assert stage26_8m._input_hash(legacy_config, _repo_root()) != stage26_8m._input_hash(
        hard_gate_config,
        _repo_root(),
    )
    assert legacy_job["phase_config_hashes"]["collector"] != hard_gate_job["phase_config_hashes"]["collector"]


def test_input_hash_changes_when_candidate_theta_proposal_policy_changes(tmp_path: Path) -> None:
    legacy_config = stage26_8m._load_config(
        _write_config(
            tmp_path,
            {
                "collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1",
                "candidate_reachability_gate_source": "hybrid_astar_pose_reachability/v1",
                "candidate_reachability_max_theta_proposals_per_candidate": 3,
                "candidate_reachability_theta_proposal_policy": "candidate_viewpoint_current_step/v1",
            },
        ),
        _repo_root(),
    )
    repair_config = stage26_8m._load_config(
        _write_config(
            tmp_path,
            {
                "collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1",
                "candidate_reachability_gate_source": "hybrid_astar_pose_reachability/v1",
                "candidate_reachability_max_theta_proposals_per_candidate": 3,
                "candidate_reachability_theta_proposal_policy": "candidate_current_bearing_sweep/v1",
            },
        ),
        _repo_root(),
    )
    output_root = tmp_path / "out"
    legacy_job = stage26_8m._expand_jobs(legacy_config, output_root, _repo_root())[0]
    repair_job = stage26_8m._expand_jobs(repair_config, output_root, _repo_root())[0]

    assert stage26_8m._input_hash(legacy_config, _repo_root()) != stage26_8m._input_hash(
        repair_config,
        _repo_root(),
    )
    assert legacy_job["phase_config_hashes"]["collector"] != repair_job["phase_config_hashes"]["collector"]


def test_input_hash_changes_when_planner_proxy_override_changes(tmp_path: Path) -> None:
    base_config = stage26_8m._load_config(
        _write_config(
            tmp_path,
            {
                "collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1",
                **_planner_overrides(planner_grid_resolution_m=1.0),
            },
        ),
        _repo_root(),
    )
    changed_config = stage26_8m._load_config(
        _write_config(
            tmp_path,
            {
                "collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1",
                **_planner_overrides(planner_grid_resolution_m=0.5),
            },
        ),
        _repo_root(),
    )
    output_root = tmp_path / "out"
    base_job = stage26_8m._expand_jobs(base_config, output_root, _repo_root())[0]
    changed_job = stage26_8m._expand_jobs(changed_config, output_root, _repo_root())[0]

    assert stage26_8m._input_hash(base_config, _repo_root()) != stage26_8m._input_hash(changed_config, _repo_root())
    assert base_job["phase_config_hashes"]["collector"] != changed_job["phase_config_hashes"]["collector"]


def test_eval_binding_implementation_fingerprint_changes_only_eval_phase_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = stage26_8m._load_config(_write_config(tmp_path, {}), _repo_root())
    job = stage26_8m._expand_jobs(config, tmp_path / "out", _repo_root())[0]
    original_hashes = dict(job["phase_config_hashes"])

    monkeypatch.setattr(
        stage26_8m,
        "_eval_binding_implementation_fingerprint",
        lambda: {"test_eval_binding_implementation_fingerprint": "changed"},
    )
    changed_job = stage26_8m._expand_jobs(config, tmp_path / "out", _repo_root())[0]
    changed_hashes = changed_job["phase_config_hashes"]

    assert changed_hashes["collector"] == original_hashes["collector"]
    assert changed_hashes["update"] == original_hashes["update"]
    assert changed_hashes["eval_pre"] != original_hashes["eval_pre"]
    assert changed_hashes["eval_post"] != original_hashes["eval_post"]
    assert changed_hashes["aggregate"] != original_hashes["aggregate"]


def test_eval_summary_without_phase_hash_marker_stays_pending_when_resume_state_exists(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"run_mode": "aggregate_only"})
    config = stage26_8m._load_config(config_path, _repo_root())
    output_root = tmp_path / "out"
    job = stage26_8m._expand_jobs(config, output_root, _repo_root())[0]
    job_root = stage26_8m._job_output_root(output_root, job)
    _write_collector_complete(job_root / "collector")
    _write_update_complete(job_root / "update")
    _write_eval_complete(job_root / "eval" / "pre")
    _write_jsonl(
        output_root / stage26_8m.JOB_STATE_FILE,
        [
            {
                "schema_version": stage26_8m.JOB_STATE_SCHEMA_VERSION,
                "job_id": job["job_id"],
                "phase": "eval_pre",
                "status": "pending",
                "output_root": str(job_root / "eval" / "pre"),
                "summary_path": str(job_root / "eval" / "pre" / stage26_8m.hf.SUMMARY_FILE),
                "config_hash": job["phase_config_hashes"]["eval_pre"],
                "input_hash": job["input_hash"],
            }
        ],
    )

    stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=config_path,
        output_root=output_root,
        repo_root=_repo_root(),
    )
    rows = _read_jsonl(output_root / stage26_8m.JOB_STATE_FILE)

    assert _state(rows, "eval_pre")["status"] == "pending"
    assert _state(rows, "eval_pre")["blocking_reason"] == "stale_resume_state_ignored"


def test_collector_reuse_marker_rejects_planner_proxy_override_change(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    old_config = stage26_8m._load_config(
        _write_config(
            tmp_path,
            {
                "collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1",
                **_planner_overrides(planner_grid_resolution_m=1.0),
            },
        ),
        _repo_root(),
    )
    old_job = stage26_8m._expand_jobs(old_config, output_root, _repo_root())[0]
    new_config_path = _write_config(
        tmp_path,
        {
            "collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1",
            **_planner_overrides(planner_grid_resolution_m=0.5),
        },
    )
    new_config = stage26_8m._load_config(new_config_path, _repo_root())
    new_job = stage26_8m._expand_jobs(new_config, output_root, _repo_root())[0]
    collector_root = Path(new_job["collector_root"])
    _write_collector_complete(collector_root)
    _write_collector_reuse_marker(collector_root, old_job)

    summary = stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=new_config_path,
        output_root=output_root,
        repo_root=_repo_root(),
        run_mode_override="aggregate_only",
    )
    rows = _read_jsonl(output_root / stage26_8m.JOB_STATE_FILE)

    assert _state(rows, "collector")["status"] == "failed"
    assert _state(rows, "collector")["blocking_reason"] == "collector_reuse_key_mismatch"
    assert summary["next_required_change"] == stage26_8m.ROUTE_RESUME_STATE


def test_input_hash_changes_when_fixture_catalog_changes(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {"collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1"},
    )
    config = stage26_8m._load_config(config_path, _repo_root())
    fixture = Path(config["source_scenario_fixture_root"]) / "xunce-stage26-scenario-fixtures.jsonl"
    fixture.write_text('{"scenario_id":"a"}\n', encoding="utf-8")
    first = stage26_8m._input_hash(config, _repo_root())
    fixture.write_text('{"scenario_id":"b"}\n', encoding="utf-8")
    second = stage26_8m._input_hash(config, _repo_root())

    assert first != second


def test_input_hash_changes_when_update_nested_config_changes(tmp_path: Path) -> None:
    stage21_4_base = tmp_path / "stage21-4.json"
    stage21_4_base.write_text('{"learning_rate": 0.000001}\n', encoding="utf-8")
    stage26_2_base = tmp_path / "stage26-2.json"
    _write_json(
        stage26_2_base,
        {
            "stage21_4_base_config": str(stage21_4_base),
            "high_fidelity_config": "configs/xunce_high_fidelity_exploration_coverage_comparison_stage18_9_strict_v3.json",
        },
    )
    config_path = _write_config(
        tmp_path,
        {
            "base_stage26_2_config": str(stage26_2_base),
            "collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1",
        },
    )
    config = stage26_8m._load_config(config_path, _repo_root())

    first = stage26_8m._input_hash(config, _repo_root())
    stage21_4_base.write_text('{"learning_rate": 0.000002}\n', encoding="utf-8")
    second = stage26_8m._input_hash(config, _repo_root())

    assert first != second


def test_input_hash_changes_when_stage26_1_reward_profile_changes(tmp_path: Path) -> None:
    reward_profile = tmp_path / "reward-profile.json"
    reward_profile.write_text('{"coverage_weight": 1.0}\n', encoding="utf-8")
    stage26_1_base = tmp_path / "stage26-1.json"
    _write_json(
        stage26_1_base,
        {
            "stage21_1_base_config": "configs/xunce_stage21_1_on_policy_ppo_rollout_collector_v1.json",
            "stage21_2_base_config": "configs/xunce_stage21_2_coverage_first_ppo_reward_contract_v1.json",
            "stage21_3_base_config": "configs/xunce_stage21_3_ppo_batch_validation_v1.json",
            "coverage_first_reward_profile": str(reward_profile),
        },
    )
    config_path = _write_config(
        tmp_path,
        {
            "base_stage26_1_config": str(stage26_1_base),
            "collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1",
        },
    )
    config = stage26_8m._load_config(config_path, _repo_root())

    first = stage26_8m._input_hash(config, _repo_root())
    reward_profile.write_text('{"coverage_weight": 2.0}\n', encoding="utf-8")
    second = stage26_8m._input_hash(config, _repo_root())

    assert first != second


def test_collector_reuse_root_requires_marker_when_no_matching_state(tmp_path: Path) -> None:
    config = stage26_8m._load_config(
        _write_config(tmp_path, {"collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1"}),
        _repo_root(),
    )
    output_root = tmp_path / "out"
    job = stage26_8m._expand_jobs(config, output_root, _repo_root())[0]
    collector_root = Path(job["collector_root"])
    _write_json(
        collector_root / stage26_8m.stage26_1.SUMMARY_FILE,
        {
            "schema_version": "xunce-stage26-1-summary/v1",
            "status": "passed",
            "trainable_transition_count": 120,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )

    summary = stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=_write_config(tmp_path, {"collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1"}),
        output_root=output_root,
        repo_root=_repo_root(),
        run_mode_override="aggregate_only",
    )
    rows = _read_jsonl(output_root / stage26_8m.JOB_STATE_FILE)

    assert _state(rows, "collector")["status"] == "failed"
    assert _state(rows, "collector")["blocking_reason"] == "collector_reuse_key_missing"
    assert summary["next_required_change"] == stage26_8m.ROUTE_RESUME_STATE


def test_physical_paths_are_short_for_windows_nested_artifacts(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {"collector_reuse_policy": "by_horizon_seed_scenario_rollout/v1"},
    )
    output_root = (
        Path("D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/outputs/")
        / "path_feedback_batch_xunce_stage26_8n_aggressive_sample_update_sweep_v1"
        / "m"
    )
    config = stage26_8m._load_config(config_path, _repo_root())
    job = stage26_8m._expand_jobs(config, output_root, _repo_root())[0]
    collector_nested = Path(job["collector_root"]) / "_dynamic_validation_work_stage26_1" / "xunce-exploration-coverage-dynamic-validation-results.jsonl"
    eval_nested = stage26_8m._job_output_root(output_root, job) / "w" / "xunce-exploration-coverage-dynamic-validation-results.jsonl"

    assert len(str(collector_nested)) < 260
    assert len(str(eval_nested)) < 260


def test_lineage_mismatch_complete_aggregate_routes_to_eval_binding(tmp_path: Path) -> None:
    config = stage26_8m._load_config(_write_config(tmp_path, {}), _repo_root())
    job = stage26_8m._expand_jobs(config, tmp_path / "out", _repo_root())[0]
    job_root = stage26_8m._job_output_root(tmp_path / "out", job)
    _write_complete_job(job_root, changed=1, per100=1.0)
    summary_path = job_root / "aggregate" / stage26_8m.stage26_3.SUMMARY_FILE
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["coverage_denominator_source"] = "legacy_roi_valid_cells/v1"
    _write_json(summary_path, payload)

    summary = stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=_write_config(tmp_path, {"run_mode": "aggregate_only"}),
        output_root=tmp_path / "out",
        repo_root=_repo_root(),
    )

    assert summary["next_required_change"] == stage26_8m.ROUTE_EVAL


def test_run_phase_refuses_incomplete_dependency(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"run_mode": "run_phase", "phase": "update"})
    config = stage26_8m._load_config(config_path, _repo_root())
    job_id = stage26_8m._expand_jobs(config, tmp_path / "out", _repo_root())[0]["job_id"]

    summary = stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=_repo_root(),
        job_id_override=job_id,
    )

    assert summary["phase_execution_count"] == 0
    assert summary["next_phase"] == "collector"


def test_aggregate_uses_explicit_pre_post_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"run_mode": "run_phase", "phase": "aggregate"})
    config = stage26_8m._load_config(config_path, _repo_root())
    job = stage26_8m._expand_jobs(config, tmp_path / "out", _repo_root())[0]
    job_root = stage26_8m._job_output_root(tmp_path / "out", job)
    _write_collector_complete(job_root / "collector")
    _write_update_complete(job_root / "update")
    _write_eval_complete(job_root / "eval" / "pre")
    _write_eval_complete(job_root / "eval" / "post")

    captured: dict[str, Path] = {}

    def fake_aggregate(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        captured["config_path"] = config_path
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        assert config_payload["execute_high_fidelity_evaluations"] is False
        assert Path(config_payload["pre_ppo_evaluation_root"]) == job_root / "eval" / "pre"
        assert Path(config_payload["post_ppo_evaluation_root"]) == job_root / "eval" / "post"
        summary = _aggregate_summary(changed=0, per100=0.0)
        _write_json(output_root / stage26_8m.stage26_3.SUMMARY_FILE, summary)
        return summary

    monkeypatch.setattr(stage26_8m.stage26_3, "run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke", fake_aggregate)

    summary = stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=_repo_root(),
        job_id_override=job["job_id"],
    )

    assert captured["config_path"].is_file()
    assert summary["next_required_change"] == stage26_8m.ROUTE_INCREASE


def test_update_failure_routes_to_update_stability(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"run_mode": "aggregate_only"})
    config = stage26_8m._load_config(config_path, _repo_root())
    job = stage26_8m._expand_jobs(config, tmp_path / "out", _repo_root())[0]
    job_root = stage26_8m._job_output_root(tmp_path / "out", job)
    _write_collector_complete(job_root / "collector")
    _write_json(job_root / "update" / stage26_8m.stage26_2.SUMMARY_FILE, {"status": "failed"})

    summary = stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=_repo_root(),
    )

    assert summary["next_required_change"] == stage26_8m.ROUTE_UPDATE


def test_action_changed_nonnegative_efficiency_routes_resume_queue(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"run_mode": "aggregate_only"})
    config = stage26_8m._load_config(config_path, _repo_root())
    job = stage26_8m._expand_jobs(config, tmp_path / "out", _repo_root())[0]
    _write_complete_job(stage26_8m._job_output_root(tmp_path / "out", job), changed=1, per100=0.0)

    summary = stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=_repo_root(),
    )

    assert summary["next_required_change"] == stage26_8m.ROUTE_RESUME_DIVERSE


def test_legacy_aggregate_coverage_per_100m_delta_falls_back_to_main_metric(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"run_mode": "aggregate_only"})
    config = stage26_8m._load_config(config_path, _repo_root())
    job = stage26_8m._expand_jobs(config, tmp_path / "out", _repo_root())[0]
    job_root = stage26_8m._job_output_root(tmp_path / "out", job)
    _write_complete_job(job_root, changed=1, per100=2.5)
    aggregate_path = job_root / "aggregate" / stage26_8m.stage26_3.SUMMARY_FILE
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate.pop("main_coverage_per_100m_delta")
    _write_json(aggregate_path, aggregate)

    summary = stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=_repo_root(),
    )

    assert summary["mean_main_coverage_per_100m_delta"] == 2.5
    assert summary["positive_efficiency_job_count"] == 1
    assert summary["next_required_change"] == stage26_8m.ROUTE_STAGE26_9


def test_majority_positive_routes_stage26_9(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"run_mode": "aggregate_only", "seeds": [260801, 260802, 260803]})
    config = stage26_8m._load_config(config_path, _repo_root())
    jobs = stage26_8m._expand_jobs(config, tmp_path / "out", _repo_root())
    _write_complete_job(stage26_8m._job_output_root(tmp_path / "out", jobs[0]), changed=1, per100=0.1)
    _write_complete_job(stage26_8m._job_output_root(tmp_path / "out", jobs[1]), changed=1, per100=0.2)
    _write_complete_job(stage26_8m._job_output_root(tmp_path / "out", jobs[2]), changed=0, per100=0.0)

    summary = stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=_repo_root(),
    )

    assert summary["next_required_change"] == stage26_8m.ROUTE_STAGE26_9


def test_registry_contains_stage26_8m() -> None:
    registry = json.loads((_repo_root() / "configs" / "stage_registry.json").read_text(encoding="utf-8"))

    assert stage26_8m.STAGE_ID in registry["stages"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_config(tmp_path: Path, overrides: dict) -> Path:
    source_root = tmp_path / "source_fixture"
    source_root.mkdir(exist_ok=True)
    payload = {
        "schema_version": stage26_8m.CONFIG_SCHEMA_VERSION,
        "stage_id": stage26_8m.STAGE_ID,
        "source_scenario_fixture_root": str(source_root),
        "horizons": [16],
        "seeds": [260801],
        "scenario_counts": [3],
        "collector_rollout_steps": [16],
        "eval_rollout_steps": [16],
        "update_combos": [
            {
                "combo_id": "policy_amp_x2",
                "epochs": 16,
                "learning_rate": 0.00003,
                "policy_loss_coefficient": 3.0,
                "value_loss_coefficient": 0.01,
                "entropy_coefficient": 0.005,
                "loss_scale": 0.5,
            }
        ],
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    } | overrides
    path = tmp_path / "config.json"
    _write_json(path, payload)
    return path


def _combo(combo_id: str) -> dict:
    return {
        "combo_id": combo_id,
        "epochs": 1,
        "learning_rate": 1.0e-6,
        "policy_loss_coefficient": 1.0,
        "value_loss_coefficient": 0.01,
        "entropy_coefficient": 0.005,
        "loss_scale": 0.25,
    }


def _planner_overrides(**updates: object) -> dict:
    values = {
        "hybrid_astar_planning_grid_source": "derived_high_res_planning_proxy/v1",
        "planner_grid_resolution_m": 1.0,
        "hybrid_astar_closed_key_xy_resolution_m": 1.0,
        "hybrid_astar_primitive_duration_s": 1.25,
        "hybrid_astar_goal_position_tolerance_m": 1.5,
        "hybrid_astar_goal_theta_tolerance_deg": 4.0,
        "hybrid_astar_max_iterations": 5000,
        "hybrid_astar_integration_dt_s": 0.2,
        "hybrid_astar_max_speed_mps": 2.5,
        "hybrid_astar_max_angular_speed_degps": 35.0,
    }
    values.update(updates)
    return values


def _write_complete_job(job_root: Path, *, changed: int, per100: float) -> None:
    _write_collector_complete(job_root / "collector")
    _write_update_complete(job_root / "update")
    _write_eval_complete(job_root / "eval" / "pre")
    _write_eval_complete(job_root / "eval" / "post")
    _write_json(job_root / "aggregate" / stage26_8m.stage26_3.SUMMARY_FILE, _aggregate_summary(changed=changed, per100=per100))


def _write_collector_complete(root: Path) -> None:
    _write_json(root / stage26_8m.stage26_1.SUMMARY_FILE, {"schema_version": "xunce-stage26-1-summary/v1", "status": "passed"})


def _write_collector_reuse_marker(root: Path, job: dict) -> None:
    _write_json(root / stage26_8m.COLLECTOR_REUSE_MARKER_FILE, stage26_8m._collector_reuse_marker(job))


def _write_update_complete(root: Path) -> None:
    stage21_4_root = root / "s21_4"
    _write_json(
        stage21_4_root / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json",
        {
            "ppo_ratio_old_log_prob_source": "behavior_policy_when_present",
            "kl_gate_source": "policy_old_logprob_when_available/v1",
            "behavior_policy_kl_diagnostic_only": True,
            "final_post_update_policy_approx_kl": 0.1,
            "source_xunce_candidate_checkpoint": str(root / "source.pt"),
            "experimental_checkpoint_path": str(root / "experimental.pt"),
        },
    )
    _write_json(
        root / stage26_8m.stage26_2.SUMMARY_FILE,
        {
            "schema_version": "xunce-stage26-2-summary/v1",
            "status": "passed",
            "stage21_4_status": "passed",
            "stage21_4_next_required_change": "run_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke",
            "stage21_4_root": str(stage21_4_root),
            "final_post_update_policy_approx_kl": 0.1,
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
        },
    )


def _write_eval_complete(root: Path) -> None:
    _write_json(root / stage26_8m.hf.SUMMARY_FILE, {"status": "passed"})
    _write_jsonl(root / stage26_8m.hf.MODEL_INFERENCE_FILE, [{"row": 1}])
    _write_jsonl(root / stage26_8m.hf.EPISODES_FILE, [{"episode": 1}])


def _aggregate_summary(*, changed: int, per100: float) -> dict:
    return {
        "schema_version": stage26_8m.stage26_3.SUMMARY_SCHEMA_VERSION,
        "status": "failed" if changed == 0 else "passed",
        "next_required_change": "repair_stage26_synthetic_policy_update_signal_strength" if changed == 0 else "resume_stage26_8d_seed_horizon_jobs_with_diverse_scenarios",
        "selected_action_changed_count": changed,
        "coverage_per_100m_delta": per100,
        "main_coverage_per_100m_delta": per100,
        "final_coverage_delta": 0.0,
        "main_final_coverage_delta": 0.0,
        "coverage_denominator_source": stage26_8m.stage26_8.COVERAGE_DENOMINATOR_SOURCE,
        "post_update_success_metric": stage26_8m.stage26_8.SUCCESS_METRIC,
        "coverage_source": stage26_8m.stage26_8.COVERAGE_SOURCE,
        "path_cost_source": stage26_8m.stage26_8.PATH_COST_SOURCE,
        "synthetic_source_kind": stage26_8m.stage26_8.SYNTHETIC_SOURCE_KIND,
        "action_space_type": stage26_8m.stage26_8.ACTION_SPACE_TYPE,
        "max_traversable_slope_deg": 30.0,
        "stage21_5_execution_incomplete": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _state(rows: list[dict], phase: str) -> dict:
    return next(row for row in rows if row["phase"] == phase)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
