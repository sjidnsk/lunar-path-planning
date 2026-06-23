from __future__ import annotations

import json
from pathlib import Path

from scripts.run_xunce_stage21_9_gradient_normalization_loss_scaling_repair import (
    ROUTE_ADVANTAGE,
    ROUTE_CONTINUE,
    ROUTE_INPUTS,
    ROUTE_LOSS,
    ROUTE_RERUN_21_6,
    run_xunce_stage21_9_gradient_normalization_loss_scaling_repair,
)


def test_stage21_9_writes_repaired_configs_with_loss_scaling(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    config = _write_config(root)

    summary = run_xunce_stage21_9_gradient_normalization_loss_scaling_repair(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "partial"
    assert summary["next_required_change"] == ROUTE_LOSS
    repaired4 = json.loads(Path(summary["repaired_stage21_4_config"]).read_text(encoding="utf-8"))
    assert repaired4["advantage_clip_abs"] == 5.0
    assert repaired4["normalize_minibatch_advantages"] is True
    assert repaired4["loss_scale"] == 0.25
    assert repaired4["value_loss_coefficient"] == 0.1
    assert repaired4["publishes_checkpoint"] is False
    repaired8 = json.loads(Path(summary["repaired_stage21_8_config"]).read_text(encoding="utf-8"))
    assert repaired8["priority_combinations"][0]["combo_id"] == "stage21_9_repaired_loss_scale_smoke"
    assert repaired8["sweep_work_root"] == str(root / "sweep_runs")


def test_stage21_9_routes_to_advantage_contract_when_train_advantages_not_normalized(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path, normalized=False)
    config = _write_config(root)

    summary = run_xunce_stage21_9_gradient_normalization_loss_scaling_repair(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_ADVANTAGE


def test_stage21_9_routes_to_rerun_when_repaired_smoke_is_stable(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_repaired_smoke(root / "repaired_smoke", pre_clip=12.0, policy_shift=True, stable=True)
    config = _write_config(root)

    summary = run_xunce_stage21_9_gradient_normalization_loss_scaling_repair(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == ROUTE_RERUN_21_6
    assert summary["repaired_pre_clip_grad_norm_max"] == 12.0
    assert summary["stage21_9_authorized"] is False


def test_stage21_9_routes_to_continue_when_repaired_smoke_still_exceeds_gate(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_repaired_smoke(root / "repaired_smoke", pre_clip=31.0, policy_shift=True, stable=False)
    config = _write_config(root)

    summary = run_xunce_stage21_9_gradient_normalization_loss_scaling_repair(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_CONTINUE


def test_stage21_9_boundary_flags_hard_fail(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    config = _write_config(root, starts_online_canary=True)

    summary = run_xunce_stage21_9_gradient_normalization_loss_scaling_repair(
        config_path=config,
        output_root=root / "out",
        repo_root=root,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_INPUTS
    assert "stage21_9_config_starts_online_canary_true" in summary["boundary_reason_codes"]


def _fixture_root(tmp_path: Path, *, normalized: bool = True) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "configs").mkdir()
    _json(root / "configs" / "stage21_4.json", {"schema_version": "xunce-stage21-4-tiny-ppo-update-smoke-config/v1"})
    _json(root / "configs" / "stage21_8.json", {"schema_version": "xunce-stage21-8-ppo-update-strength-calibration-config/v1"})
    stage21_8 = root / "stage21_8"
    stage21_8.mkdir()
    _json(
        stage21_8 / "xunce-stage21-8-calibration-summary.json",
        {"status": "failed", "next_required_change": "repair_stage21_4_gradient_normalization_or_loss_scaling"},
    )
    combo = root / "completed_combo" / "stage21_6"
    combo.mkdir(parents=True)
    _json(combo / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json", {"status": "failed"})
    seed_rows = []
    for seed in (2101, 2102, 2103):
        stage21_3 = combo / f"seed_{seed}" / "stage21_3"
        stage21_4 = combo / f"seed_{seed}" / "stage21_4"
        stage21_3.mkdir(parents=True)
        stage21_4.mkdir(parents=True)
        _jsonl(
            stage21_3 / "xunce-stage21-3-ppo-trainable-batch.jsonl",
            [
                {
                    "stage21_3_split": "train",
                    "raw_advantage": 10.0,
                    "advantage": 1.0,
                    "advantage_normalization_applied": normalized,
                },
                {
                    "stage21_3_split": "train",
                    "raw_advantage": -4.0,
                    "advantage": -1.0,
                    "advantage_normalization_applied": normalized,
                },
            ],
        )
        _jsonl(
            stage21_3 / "xunce-stage21-3-return-advantage-audit.jsonl",
            [{"stage21_3_split": "train"}],
        )
        _jsonl(
            stage21_4 / "xunce-stage21-4-ppo-loss-audit.jsonl",
            [
                {
                    "total_loss": 82.0,
                    "policy_loss": 0.0,
                    "value_loss": 164.0,
                    "entropy": 3.0,
                    "total_loss_grad_norm": 45.0,
                    "policy_loss_grad_norm": 0.0,
                    "value_loss_grad_norm": 45.0,
                    "entropy_loss_grad_norm": 0.01,
                }
            ],
        )
        _json(stage21_4 / "xunce-stage21-4-gradient-audit.json", {"component_grad_norms": {"value_loss_grad_norm": 45.0}})
        seed_rows.append(
            {
                "seed": seed,
                "stage21_3_root": str(stage21_3),
                "stage21_4_root": str(stage21_4),
                "stage21_5_root": str(combo / f"seed_{seed}" / "stage21_5"),
                "trainable_transition_count": 80,
                "pre_clip_grad_norm": 45.0,
                "post_clip_grad_norm": 1.0,
                "grad_norm_finite": True,
            }
        )
    _jsonl(combo / "xunce-stage21-6-seed-results.jsonl", seed_rows)
    return root


def _write_config(root: Path, **overrides: object) -> Path:
    payload = {
        "schema_version": "xunce-stage21-9-gradient-normalization-loss-scaling-repair-config/v1",
        "stage21_8_root": str(root / "stage21_8"),
        "stage21_8_completed_combo_root": str(root / "completed_combo" / "stage21_6"),
        "stage21_4_base_config": str(root / "configs" / "stage21_4.json"),
        "stage21_8_base_config": str(root / "configs" / "stage21_8.json"),
        "repaired_stage21_8_output_root": str(root / "repaired_smoke"),
        "repaired_stage21_8_sweep_work_root": str(root / "sweep_runs"),
        "execute_repaired_smoke": False,
        "repaired_smoke_timeout_seconds": 7200,
        "learning_rate": 0.000002,
        "epochs": 1,
        "clip_ratio": 0.2,
        "stage21_4_max_grad_norm": 1.0,
        "stage21_6_pre_clip_grad_norm_gate": 25.0,
        "advantage_clip_abs": 5.0,
        "normalize_minibatch_advantages": True,
        "loss_scale": 0.25,
        "value_loss_coefficient": 0.1,
        "entropy_coefficient": 0.01,
        "stage21_9_authorized": False,
        "training_or_release_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = root / "stage21_9_config.json"
    _json(path, payload)
    return path


def _write_repaired_smoke(root: Path, *, pre_clip: float, policy_shift: bool, stable: bool) -> None:
    root.mkdir(parents=True)
    _json(
        root / "xunce-stage21-8-calibration-summary.json",
        {
            "status": "passed" if stable else "failed",
            "next_required_change": (
                "rerun_stage21_6_multi_seed_pilot_with_stage21_8_recommended_config"
                if stable
                else "repair_stage21_4_gradient_normalization_or_loss_scaling"
            ),
        },
    )
    _jsonl(
        root / "xunce-stage21-8-sweep-results.jsonl",
        [
            {
                "combo_id": "stage21_9_repaired_loss_scale_smoke",
                "status": "passed" if stable else "failed",
                "source_type": "stage21_8_sweep",
                "pre_clip_grad_norm_max": pre_clip,
                "stage21_6_pre_clip_grad_norm_gate": 25.0,
                "post_clip_grad_norm_max": 1.0,
                "post_clip_within_stage21_4_max_grad_norm": True,
                "post_update_approx_kl_mean": 0.0001,
                "min_entropy_mean": 3.0,
                "parameter_delta_l2_mean": 0.01,
                "policy_shift_observable": policy_shift,
                "coverage_auc_not_regressed": True,
                "numerically_stable": stable,
                "boundary_reason_codes": [],
                "stage21_6_next_required_change": "prepare_stage22_formal_pure_ppo_training_run" if stable else "repair",
            }
        ],
    )


def _json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
