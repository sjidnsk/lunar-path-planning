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
    job_root = tmp_path / "out" / "jobs" / job["job_id"]
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
    job_root = tmp_path / "out" / "jobs" / job["job_id"]
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
    _write_complete_job(tmp_path / "out" / "jobs" / job["job_id"], changed=1, per100=0.0)

    summary = stage26_8m.run_xunce_stage26_8m_generalized_resumable_training_pipeline(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=_repo_root(),
    )

    assert summary["next_required_change"] == stage26_8m.ROUTE_RESUME_DIVERSE


def test_majority_positive_routes_stage26_9(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"run_mode": "aggregate_only", "seeds": [260801, 260802, 260803]})
    config = stage26_8m._load_config(config_path, _repo_root())
    jobs = stage26_8m._expand_jobs(config, tmp_path / "out", _repo_root())
    _write_complete_job(tmp_path / "out" / "jobs" / jobs[0]["job_id"], changed=1, per100=0.1)
    _write_complete_job(tmp_path / "out" / "jobs" / jobs[1]["job_id"], changed=1, per100=0.2)
    _write_complete_job(tmp_path / "out" / "jobs" / jobs[2]["job_id"], changed=0, per100=0.0)

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


def _write_complete_job(job_root: Path, *, changed: int, per100: float) -> None:
    _write_collector_complete(job_root / "collector")
    _write_update_complete(job_root / "update")
    _write_eval_complete(job_root / "eval" / "pre")
    _write_eval_complete(job_root / "eval" / "post")
    _write_json(job_root / "aggregate" / stage26_8m.stage26_3.SUMMARY_FILE, _aggregate_summary(changed=changed, per100=per100))


def _write_collector_complete(root: Path) -> None:
    _write_json(root / stage26_8m.stage26_1.SUMMARY_FILE, {"schema_version": "xunce-stage26-1-summary/v1", "status": "passed"})


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
