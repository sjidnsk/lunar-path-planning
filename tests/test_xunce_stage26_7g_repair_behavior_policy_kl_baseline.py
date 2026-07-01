from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_7g_rejects_untrusted_stage26_7f_root(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_7g_repair_behavior_policy_kl_baseline as s26

    stage26_7f = _make_stage26_7f_root(tmp_path, route="some_other_route")
    config = _write_config(tmp_path, stage26_7f_root=stage26_7f, run_stage26_chain=False)

    summary = s26.run_xunce_stage26_7g_repair_behavior_policy_kl_baseline(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_7g_required_inputs"


def test_stage26_7g_rejects_relaxed_policy_kl_limit(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_7g_repair_behavior_policy_kl_baseline as s26

    stage26_7f = _make_stage26_7f_root(tmp_path)
    config = _write_config(tmp_path, stage26_7f_root=stage26_7f, max_abs_approx_kl=1.5001)

    with pytest.raises(ValueError, match="max_abs_approx_kl"):
        s26.run_xunce_stage26_7g_repair_behavior_policy_kl_baseline(
            config_path=config,
            output_root=tmp_path / "out",
            repo_root=REPO_ROOT,
        )


def test_stage26_7g_routes_zero_update_policy_kl_capture_failure() -> None:
    import scripts.run_xunce_stage26_7g_repair_behavior_policy_kl_baseline as s26

    route = s26._route(
        boundary_rejections=[],
        input_rejections=[],
        zero_audit={"status": "passed", "policy_approx_kl": 0.02, "policy_kl_abs_tolerance": 0.001},
        sweep_rows=[],
        stable_rows=[],
        best_combo={},
        kl_audit={},
        post_eval_audit={},
    )

    assert route == "repair_stage26_7_policy_old_logprob_capture"


def test_stage26_7g_routes_bad_behavior_ratio_source() -> None:
    import scripts.run_xunce_stage26_7g_repair_behavior_policy_kl_baseline as s26

    row = _stable_row("bad_ratio")
    row["ppo_ratio_old_log_prob_source"] = "policy_only"
    route = s26._route(
        boundary_rejections=[],
        input_rejections=[],
        zero_audit=_zero_ok(),
        sweep_rows=[row],
        stable_rows=[row],
        best_combo=row,
        kl_audit={},
        post_eval_audit={"selected_action_changed_count": 1, "main_coverage_per_100m_delta": 0.1},
    )

    assert route == "repair_stage26_7_behavior_ratio_contract"


def test_stage26_7g_routes_bad_kl_gate_source() -> None:
    import scripts.run_xunce_stage26_7g_repair_behavior_policy_kl_baseline as s26

    row = _stable_row("bad_kl_source")
    row["kl_gate_source"] = "behavior_old_logprob"
    route = s26._route(
        boundary_rejections=[],
        input_rejections=[],
        zero_audit=_zero_ok(),
        sweep_rows=[row],
        stable_rows=[row],
        best_combo=row,
        kl_audit={},
        post_eval_audit={"selected_action_changed_count": 1, "main_coverage_per_100m_delta": 0.1},
    )

    assert route == "repair_stage26_7_kl_gate_source"


def test_stage26_7g_fake_sweep_uses_policy_kl_sources_and_routes_to_margin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.run_xunce_stage26_7g_repair_behavior_policy_kl_baseline as s26

    stage26_7f = _make_stage26_7f_root(tmp_path)
    config = _write_config(
        tmp_path,
        stage26_7f_root=stage26_7f,
        update_sweep=[_combo("stable", "u0")],
    )

    monkeypatch.setattr(
        s26,
        "_zero_update_kl_baseline_audit",
        lambda config, stage26_1_root, repo_root: {
            "schema_version": s26.ZERO_AUDIT_SCHEMA_VERSION,
            "status": "passed",
            "policy_approx_kl": 0.0,
            "behavior_approx_kl": 2.7,
            "policy_kl_abs_tolerance": 0.001,
            "behavior_policy_kl_diagnostic_only": True,
        },
    )

    def fake_run_combo(*, config, combo, primary_stage26_1_root, output_root, repo_root):
        row = _stable_row(combo["combo_id"])
        row.update(
            {
                "stage26_3_status": "failed",
                "stage26_3_next_required_change": "calibrate_stage26_synthetic_discrete_margin_crossing_after_credit",
                "selected_action_changed_count": 0,
                "main_coverage_per_100m_delta": 0.0,
            }
        )
        return {
            "combo": combo,
            "combo_root": output_root / str(combo["work_dir"]),
            "stage26_2_config": {},
            "stage26_3_config": {},
            "stage26_2_summary": {},
            "stage26_3_summary": {},
            "sweep_row": row,
        }

    monkeypatch.setattr(s26.stage26_7f, "_run_combo", fake_run_combo)
    monkeypatch.setattr(
        s26,
        "_combo_policy_behavior_kl_fields",
        lambda detail: {
            "pre_update_behavior_approx_kl": 2.7,
            "post_update_behavior_approx_kl": 2.9,
            "pre_update_policy_approx_kl": 0.0,
            "post_update_policy_approx_kl": 0.72,
            "final_post_update_policy_approx_kl": 0.72,
            "final_post_update_behavior_approx_kl": 2.9,
            "ppo_ratio_old_log_prob_source": "behavior_policy_when_present",
            "kl_gate_source": "policy_old_logprob_when_available/v1",
            "behavior_policy_kl_diagnostic_only": True,
        },
    )

    summary = s26.run_xunce_stage26_7g_repair_behavior_policy_kl_baseline(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["stable_combo_count"] == 1
    assert summary["best_stable_combo_id"] == "stable"
    assert summary["zero_update_policy_approx_kl"] == 0.0
    assert summary["zero_update_behavior_approx_kl"] == 2.7
    assert summary["best_stable_final_post_update_policy_approx_kl"] == 0.72
    assert summary["next_required_change"] == "calibrate_stage26_synthetic_discrete_margin_crossing_after_credit"


def test_stage26_7g_routes_failed_stage26_3_binding_before_credit_assignment() -> None:
    import scripts.run_xunce_stage26_7g_repair_behavior_policy_kl_baseline as s26

    row = _stable_row("binding_bad")
    row.update(
        {
            "stage26_3_status": "failed",
            "stage26_3_next_required_change": "rerun_stage26_3_required_inputs",
            "selected_action_changed_count": 2,
            "main_coverage_per_100m_delta": 0.0,
        }
    )

    route = s26._route(
        boundary_rejections=[],
        input_rejections=[],
        zero_audit=_zero_ok(),
        sweep_rows=[row],
        stable_rows=[row],
        best_combo=row,
        kl_audit={},
        post_eval_audit={
            "stage26_3_status": "failed",
            "stage26_3_next_required_change": "rerun_stage26_3_required_inputs",
            "selected_action_changed_count": 2,
            "main_coverage_per_100m_delta": 0.0,
        },
    )

    assert route == "repair_stage26_7_credit_post_update_eval_binding"


def _make_stage26_7f_root(
    tmp_path: Path,
    *,
    route: str = "repair_stage26_7_behavior_policy_kl_baseline",
) -> Path:
    root = tmp_path / "s26_7f"
    primary = root / "primary" / "s26_1"
    primary.mkdir(parents=True)
    _write_json(primary / "xunce-stage26-1-summary.json", {"status": "passed"})
    _write_json(
        root / "xunce-stage26-7f-summary.json",
        {
            "stage_id": "xunce-stage26-7f-synthetic-credit-ppo-update-stability-sweep",
            "status": "failed",
            "next_required_change": route,
            "pre_update_kl_baseline_exceeded": True,
            "primary_stage26_1_root": str(primary),
        },
    )
    return root


def _write_config(
    tmp_path: Path,
    *,
    stage26_7f_root: Path,
    max_abs_approx_kl: float = 1.5,
    run_stage26_chain: bool = True,
    update_sweep: list[dict[str, object]] | None = None,
) -> Path:
    stage26_2_base = tmp_path / "stage26_2_base.json"
    stage26_3_base = tmp_path / "stage26_3_base.json"
    checkpoint = tmp_path / "candidate.pt"
    high_fidelity_config = tmp_path / "high_fidelity.json"
    checkpoint.write_bytes(b"not-a-real-checkpoint")
    _write_json(high_fidelity_config, {})
    _write_json(
        stage26_2_base,
        {
            "stage26_1_root": "unused",
            "xunce_candidate_checkpoint": str(checkpoint),
            "high_fidelity_config": str(high_fidelity_config),
        },
    )
    _write_json(stage26_3_base, {"stage26_2_root": "unused"})
    config = tmp_path / "config.json"
    _write_json(
        config,
        {
            "schema_version": "xunce-stage26-7g-repair-behavior-policy-kl-baseline-config/v1",
            "stage_id": "xunce-stage26-7g-repair-behavior-policy-kl-baseline",
            "stage26_7f_root": str(stage26_7f_root),
            "stage26_2_base_config": str(stage26_2_base),
            "stage26_3_base_config": str(stage26_3_base),
            "run_stage26_chain": run_stage26_chain,
            "required_scenario_count": 2,
            "eval_rollout_steps": 4,
            "hybrid_astar_candidate_eval_workers": 4,
            "max_abs_approx_kl": max_abs_approx_kl,
            "zero_update_policy_kl_abs_tolerance": 0.001,
            "update_sweep": update_sweep or [_combo("default", "u0")],
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    return config


def _combo(combo_id: str, work_dir: str) -> dict[str, object]:
    return {
        "combo_id": combo_id,
        "work_dir": work_dir,
        "epochs": 1,
        "learning_rate": 5.0e-6,
        "policy_loss_coefficient": 1.0,
        "value_loss_coefficient": 0.02,
        "loss_scale": 0.25,
    }


def _stable_row(combo_id: str) -> dict[str, object]:
    return {
        "combo_id": combo_id,
        "stage26_2_status": "passed",
        "stage21_4_status": "passed",
        "checkpoint_reload_passed": True,
        "experimental_only": True,
        "checkpoint_boundary_passed": True,
        "final_post_update_approx_kl": 0.72,
        "final_post_update_policy_approx_kl": 0.72,
        "final_post_update_behavior_approx_kl": 2.9,
        "loss_finite": True,
        "gradient_finite": True,
        "parameter_delta_l2": 0.004,
        "ppo_ratio_old_log_prob_source": "behavior_policy_when_present",
        "kl_gate_source": "policy_old_logprob_when_available/v1",
        "behavior_policy_kl_diagnostic_only": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _zero_ok() -> dict[str, object]:
    return {"status": "passed", "policy_approx_kl": 0.0, "policy_kl_abs_tolerance": 0.001}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
