from __future__ import annotations

import json
import hashlib
from pathlib import Path

import scripts.run_xunce_stage21_18_iterative_ppo_learning_curve_audit as stage21_18
from scripts.run_xunce_stage21_18_iterative_ppo_learning_curve_audit import (
    ROUTE_MARGIN,
    ROUTE_POLICY_SOURCE,
    ROUTE_SCALE,
    run_xunce_stage21_18_iterative_ppo_learning_curve_audit,
)


def test_stage21_18_uses_previous_round_checkpoint_for_next_round(monkeypatch, tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    expected_sources = set()
    for seed in (2101, 2102, 2103):
        round1_checkpoint = root / f"round1-{seed}.pt"
        round1_checkpoint.write_text(f"round1-{seed}", encoding="utf-8")
        expected_sources.add(str(round1_checkpoint))
        _write_stage21_6_round(_chain_root(root, seed, 1) / "stage21_6", seed=seed, checkpoint_path=round1_checkpoint)

    observed: list[str] = []

    def fake_stage21_6(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, object]:
        cfg = _read_json(config_path)
        stage21_4 = _read_json(Path(cfg["stage21_4_base_config"]))
        observed.append(stage21_4["xunce_candidate_checkpoint"])
        seed = int(cfg["seed_list"][0])
        next_checkpoint = root / f"round2-{seed}.pt"
        next_checkpoint.write_text(f"round2-{seed}", encoding="utf-8")
        _write_stage21_6_round(output_root, seed=seed, checkpoint_path=next_checkpoint, probability_delta=0.0002)
        return _read_json(output_root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json")

    monkeypatch.setattr(stage21_18, "run_xunce_stage21_6_multi_seed_ppo_pilot", fake_stage21_6)
    summary = run_xunce_stage21_18_iterative_ppo_learning_curve_audit(
        config_path=_write_config(root, execute_rounds=True, max_new_rounds_to_execute=1),
        output_root=root / "out",
        repo_root=root,
    )

    assert set(observed) == expected_sources
    assert summary["completed_round_count"] == 2
    assert summary["stage21_18_authorized"] is False


def test_stage21_18_missing_collector_checkpoint_audit_blocks_lineage(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    checkpoint = root / "round1-2101.pt"
    checkpoint.write_text("round1-2101", encoding="utf-8")
    _write_stage21_6_round(
        _chain_root(root, 2101, 1) / "stage21_6",
        seed=2101,
        checkpoint_path=checkpoint,
        write_collector_audit=False,
    )
    config = stage21_18._load_config(_write_config(root), root)
    row = stage21_18._read_seed_round_result(config, 2101, 1, _chain_root(root, 2101, 1) / "stage21_6")

    assert row["checkpoint_lineage_passed"] is False
    assert row["selected_experimental_checkpoint_path"] is None


def test_stage21_18_collector_source_mismatch_blocks_lineage(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    checkpoint = root / "round1-2101.pt"
    checkpoint.write_text("round1-2101", encoding="utf-8")
    other = root / "other-source.pt"
    other.write_text("other-source", encoding="utf-8")
    _write_stage21_6_round(
        _chain_root(root, 2101, 1) / "stage21_6",
        seed=2101,
        checkpoint_path=checkpoint,
        collector_source_checkpoint_path=other,
    )
    config = stage21_18._load_config(_write_config(root), root)
    row = stage21_18._read_seed_round_result(config, 2101, 1, _chain_root(root, 2101, 1) / "stage21_6")

    assert row["checkpoint_lineage_passed"] is False


def test_stage21_18_missing_loaded_checkpoint_path_blocks_lineage(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    checkpoint = root / "round1-2101.pt"
    checkpoint.write_text("round1-2101", encoding="utf-8")
    _write_stage21_6_round(
        _chain_root(root, 2101, 1) / "stage21_6",
        seed=2101,
        checkpoint_path=checkpoint,
        omit_collector_checkpoint_path=True,
    )
    config = stage21_18._load_config(_write_config(root), root)
    row = stage21_18._read_seed_round_result(config, 2101, 1, _chain_root(root, 2101, 1) / "stage21_6")

    assert row["checkpoint_lineage_passed"] is False


def test_stage21_18_reload_checkpoint_path_mismatch_blocks_lineage(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    checkpoint = root / "round1-2101.pt"
    checkpoint.write_text("round1-2101", encoding="utf-8")
    other = root / "other-reload.pt"
    other.write_text("other-reload", encoding="utf-8")
    _write_stage21_6_round(
        _chain_root(root, 2101, 1) / "stage21_6",
        seed=2101,
        checkpoint_path=checkpoint,
        reload_checkpoint_path=other,
    )
    config = stage21_18._load_config(_write_config(root), root)
    row = stage21_18._read_seed_round_result(config, 2101, 1, _chain_root(root, 2101, 1) / "stage21_6")

    assert row["checkpoint_lineage_passed"] is False


def test_stage21_18_reload_or_non_experimental_checkpoint_blocks_lineage(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    config = stage21_18._load_config(_write_config(root), root)
    reload_failed = root / "reload-failed.pt"
    reload_failed.write_text("reload-failed", encoding="utf-8")
    not_experimental = root / "not-experimental.pt"
    not_experimental.write_text("not-experimental", encoding="utf-8")
    _write_stage21_6_round(
        _chain_root(root, 2101, 1) / "stage21_6",
        seed=2101,
        checkpoint_path=reload_failed,
        checkpoint_reload_passed=False,
    )
    _write_stage21_6_round(
        _chain_root(root, 2102, 1) / "stage21_6",
        seed=2102,
        checkpoint_path=not_experimental,
        experimental_only=False,
    )

    row_reload = stage21_18._read_seed_round_result(config, 2101, 1, _chain_root(root, 2101, 1) / "stage21_6")
    row_experimental = stage21_18._read_seed_round_result(config, 2102, 1, _chain_root(root, 2102, 1) / "stage21_6")

    assert row_reload["checkpoint_lineage_passed"] is False
    assert row_experimental["checkpoint_lineage_passed"] is False


def test_stage21_18_routes_to_policy_source_when_multi_round_probability_stays_tiny(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    previous_by_seed: dict[int, Path] = {}
    for index in range(1, 4):
        for seed in (2101, 2102, 2103):
            checkpoint = root / f"round{index}-{seed}.pt"
            checkpoint.write_text(f"{index}-{seed}", encoding="utf-8")
            _write_stage21_6_round(
                _chain_root(root, seed, index) / "stage21_6",
                seed=seed,
                checkpoint_path=checkpoint,
                source_checkpoint_path=previous_by_seed.get(seed),
                probability_delta=0.0001,
            )
            previous_by_seed[seed] = checkpoint

    summary = _run(root)

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_POLICY_SOURCE
    assert summary["completed_round_count"] == 3
    assert summary["mean_abs_probability_delta_max"] < 0.005


def test_stage21_18_routes_to_margin_when_probability_accumulates_but_rank_does_not(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    previous_by_seed: dict[int, Path] = {}
    for index in range(1, 4):
        for seed in (2101, 2102, 2103):
            checkpoint = root / f"round{index}-{seed}.pt"
            checkpoint.write_text(f"{index}-{seed}", encoding="utf-8")
            _write_stage21_6_round(
                _chain_root(root, seed, index) / "stage21_6",
                seed=seed,
                checkpoint_path=checkpoint,
                source_checkpoint_path=previous_by_seed.get(seed),
                probability_delta=0.02,
            )
            previous_by_seed[seed] = checkpoint

    summary = _run(root)

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == ROUTE_MARGIN
    assert summary["argmax_changed_count"] == 0


def test_stage21_18_partial_strong_join_gap_is_diagnostic_not_lineage_blocker(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    config = stage21_18._load_config(_write_config(root), root)
    rows = [{"numerically_stable": True} for _ in range(3)]
    lineage = {"checkpoint_lineage_passed": True}
    action_trend = {
        "stable_round_count": 3,
        "strong_state_join_available_count": 600,
        "strong_state_binding_unavailable_count": 12,
        "mean_abs_probability_delta_max": 0.02,
        "best_coverage_per_cost_probability_delta_max": 0.02,
        "argmax_changed_count": 0,
        "selected_action_changed_count": 0,
        "selected_rank_changed_count": 0,
    }
    coverage_trend = {"coverage_auc_improved_round_count": 0}

    status, route, reason = stage21_18._route(
        config=config,
        rows=rows,
        lineage=lineage,
        action_trend=action_trend,
        coverage_trend=coverage_trend,
        input_reasons=[],
        boundary_reasons=[],
    )

    assert status == "failed"
    assert route == ROUTE_MARGIN
    assert reason == "probability_accumulated_without_rank_crossing"


def test_stage21_18_routes_to_scale_when_coverage_trend_improves(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    previous_by_seed: dict[int, Path] = {}
    for index in range(1, 4):
        for seed in (2101, 2102, 2103):
            checkpoint = root / f"round{index}-{seed}.pt"
            checkpoint.write_text(f"{index}-{seed}", encoding="utf-8")
            _write_stage21_6_round(
                _chain_root(root, seed, index) / "stage21_6",
                seed=seed,
                checkpoint_path=checkpoint,
                source_checkpoint_path=previous_by_seed.get(seed),
                probability_delta=0.02,
                coverage_delta=0.01,
                action_shift=True,
                status="passed",
            )
            previous_by_seed[seed] = checkpoint

    summary = _run(root)

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == ROUTE_SCALE
    assert summary["coverage_auc_delta_max"] > 0.0


def test_stage21_18_seed_regression_prevents_scale_route(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    config = stage21_18._load_config(_write_config(root), root)
    rows = [{"numerically_stable": True} for _ in range(3)]
    lineage = {"checkpoint_lineage_passed": True}
    action_trend = {
        "stable_round_count": 3,
        "strong_state_join_available_count": 600,
        "strong_state_binding_unavailable_count": 0,
        "mean_abs_probability_delta_max": 0.02,
        "best_coverage_per_cost_probability_delta_max": 0.02,
        "argmax_changed_count": 1,
        "selected_action_changed_count": 1,
        "selected_rank_changed_count": 1,
    }
    coverage_trend = {
        "coverage_auc_improved_round_count": 0,
        "coverage_auc_not_regressed_round_count": 2,
        "final_stable_round_improved": False,
    }

    status, route, reason = stage21_18._route(
        config=config,
        rows=rows,
        lineage=lineage,
        action_trend=action_trend,
        coverage_trend=coverage_trend,
        input_reasons=[],
        boundary_reasons=[],
    )

    assert status == "failed"
    assert route != ROUTE_SCALE
    assert reason == "rank_or_argmax_changed_without_coverage_improvement"


def test_stage21_18_early_improvement_does_not_scale_when_final_round_not_improved(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    config = stage21_18._load_config(_write_config(root), root)
    rows = [{"numerically_stable": True} for _ in range(3)]
    lineage = {"checkpoint_lineage_passed": True}
    action_trend = {
        "stable_round_count": 3,
        "strong_state_join_available_count": 600,
        "strong_state_binding_unavailable_count": 0,
        "mean_abs_probability_delta_max": 0.02,
        "best_coverage_per_cost_probability_delta_max": 0.02,
        "argmax_changed_count": 1,
        "selected_action_changed_count": 1,
        "selected_rank_changed_count": 1,
    }
    coverage_trend = {
        "coverage_auc_improved_round_count": 1,
        "coverage_auc_not_regressed_round_count": 2,
        "final_stable_round_improved": False,
    }

    status, route, reason = stage21_18._route(
        config=config,
        rows=rows,
        lineage=lineage,
        action_trend=action_trend,
        coverage_trend=coverage_trend,
        input_reasons=[],
        boundary_reasons=[],
    )

    assert status == "failed"
    assert route != ROUTE_SCALE
    assert reason == "rank_or_argmax_changed_without_coverage_improvement"


def test_stage21_18_parent_checkpoint_path_mismatch_blocks_lineage(tmp_path: Path) -> None:
    rows = [
        {
            "round_index": 1,
            "checkpoint_lineage_passed": True,
            "selected_checkpoints": [{"seed": 2101, "path": "same-sha-a.pt", "sha256": "same"}],
        },
        {
            "round_index": 2,
            "checkpoint_lineage_passed": True,
            "selected_checkpoints": [{"seed": 2101, "source_path": "same-sha-b.pt", "source_sha256": "same"}],
        },
    ]

    lineage = stage21_18._checkpoint_lineage(rows)

    assert lineage["checkpoint_lineage_passed"] is False
    assert lineage["parent_checkpoint_mismatch_count"] == 1


def _run(root: Path) -> dict[str, object]:
    return run_xunce_stage21_18_iterative_ppo_learning_curve_audit(
        config_path=_write_config(root, execute_rounds=False),
        output_root=root / "out",
        repo_root=root,
    )


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    configs = root / "configs"
    configs.mkdir()
    initial_checkpoint = root / "initial.pt"
    initial_checkpoint.write_text("initial", encoding="utf-8")
    high_fidelity = configs / "hf.json"
    _json(
        high_fidelity,
        {
            "schema_version": "xunce-high-fidelity-exploration-coverage-comparison-config/v1",
            "xunce_candidate_checkpoint": str(initial_checkpoint),
        },
    )
    stage21_1 = configs / "stage21_1.json"
    _json(
        stage21_1,
        {
            "schema_version": "xunce-stage21-1-on-policy-ppo-rollout-collector-config/v1",
            "high_fidelity_config": str(high_fidelity),
            "stage21_0_readiness_root": str(root / "stage21_0"),
            "canonical_reward_profile": str(configs / "profile.json"),
            "dynamic_validation_work_root": str(root / "work"),
        },
    )
    stage21_4 = configs / "stage21_4.json"
    _json(
        stage21_4,
        {
            "schema_version": "xunce-stage21-4-tiny-ppo-update-smoke-config/v1",
            "high_fidelity_config": str(high_fidelity),
            "xunce_candidate_checkpoint": str(initial_checkpoint),
            "epochs": 8,
            "learning_rate": 0.00002,
            "clip_ratio": 0.2,
            "loss_scale": 0.5,
            "value_loss_coefficient": 0.02,
            "policy_loss_coefficient": 2.0,
        },
    )
    stage21_5 = configs / "stage21_5.json"
    _json(
        stage21_5,
        {
            "schema_version": "xunce-stage21-5-post-update-offline-trajectory-evaluation-config/v1",
            "high_fidelity_config": str(high_fidelity),
            "canonical_reward_rerank_profile": str(configs / "profile.json"),
        },
    )
    stage21_6 = configs / "stage21_6.json"
    _json(
        stage21_6,
        {
            "schema_version": "xunce-stage21-6-multi-seed-ppo-pilot-config/v1",
            "stage21_1_base_config": str(stage21_1),
            "stage21_2_base_config": str(configs / "stage21_2.json"),
            "stage21_3_base_config": str(configs / "stage21_3.json"),
            "stage21_4_base_config": str(stage21_4),
            "stage21_5_base_config": str(stage21_5),
            "stage21_5_prerequisite_root": str(root / "stage21_5_prereq"),
            "seed_list": [2101, 2102, 2103],
            "required_scenario_count": 8,
            "rollout_steps": 10,
            "dynamic_max_candidates_per_step": 36,
            "dynamic_proposal_pool_limit_per_step": 288,
            "execute_seed_pipeline": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    stage21_17 = root / "stage21_17"
    stage21_17.mkdir()
    _json(
        stage21_17 / "xunce-stage21-17-summary.json",
        {
            "status": "failed",
            "next_required_change": "increase_stage21_policy_update_signal_strength_with_policy_loss_multiplier",
            "best_balanced_combo_id": "v0p00_p1p0_loss0p25",
        },
    )
    _json(stage21_17 / "xunce-stage21-17-recommended-stage21-6-config.json", _read_json(stage21_6))
    _json(stage21_17 / "xunce-stage21-17-recommended-stage21-4-config.json", _read_json(stage21_4))
    return root


def _write_config(root: Path, **overrides: object) -> Path:
    payload = {
        "schema_version": "xunce-stage21-18-iterative-ppo-learning-curve-audit-config/v1",
        "stage21_17_root": str(root / "stage21_17"),
        "stage21_6_base_config": str(root / "stage21_17" / "xunce-stage21-17-recommended-stage21-6-config.json"),
        "stage21_4_base_config": str(root / "stage21_17" / "xunce-stage21-17-recommended-stage21-4-config.json"),
        "round_work_root": str(root / "rounds"),
        "iterative_round_count": 5,
        "min_completed_round_count": 3,
        "max_new_rounds_to_execute": 0,
        "execute_rounds": False,
        "stage21_6_execution_mode": "in_process",
        "per_round_timeout_seconds": 7200,
        "probability_delta_threshold": 0.005,
        "seed_list": [2101, 2102, 2103],
        "required_scenario_count": 8,
        "rollout_steps": 10,
        "dynamic_max_candidates_per_step": 36,
        "dynamic_proposal_pool_limit_per_step": 288,
        "checkpoint_selection_policy": "best_capped_auc_then_lowest_seed",
        "stage21_18_authorized": False,
        "training_or_release_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = root / "stage21_18_config.json"
    _json(path, payload)
    return path


def _write_stage21_6_round(
    root: Path,
    *,
    seed: int,
    checkpoint_path: Path,
    source_checkpoint_path: Path | None = None,
    probability_delta: float = 0.0001,
    coverage_delta: float = 0.0,
    action_shift: bool = False,
    status: str = "failed",
    write_collector_audit: bool = True,
    collector_source_checkpoint_path: Path | None = None,
    checkpoint_reload_passed: bool = True,
    experimental_only: bool = True,
    omit_collector_checkpoint_path: bool = False,
    reload_checkpoint_path: Path | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    generated = root.parent / "generated_configs"
    generated.mkdir(parents=True, exist_ok=True)
    source_checkpoint_path = source_checkpoint_path or checkpoint_path
    collector_source_checkpoint_path = collector_source_checkpoint_path or source_checkpoint_path
    _json(generated / "high_fidelity_config.json", {"xunce_candidate_checkpoint": str(source_checkpoint_path)})
    _json(
        root / "xunce-stage21-6-multi-seed-ppo-pilot-summary.json",
        {
            "status": status,
            "next_required_change": "repair_stage21_return_advantage_credit_assignment",
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    _json(
        root / "xunce-stage21-6-aggregate-metrics.json",
        {
            "trainable_transition_count_total": 240,
            "final_coverage_delta_mean": coverage_delta,
            "final_coverage_delta_min": coverage_delta,
            "coverage_auc_delta_mean": coverage_delta,
            "coverage_auc_delta_min": coverage_delta,
            "pre_clip_grad_norm_max": 3.0,
            "max_post_update_approx_kl_mean": 0.001,
            "min_entropy_mean": 3.0,
            "hard_risk_violation_total": 0,
            "safety_boundary_violation_total": 0,
            "execution_boundary_violation_total": 0,
            "model_inference_failure_total": 0,
            "model_inference_mask_violation_total": 0,
            "open_grid_fallback_total": 0,
            "path_planning_failure_total": 0,
            "unreachable_selected_total": 0,
            "checkpoint_reload_failed_count": 0,
            "checkpoint_not_experimental_only_count": 0,
            "seed_release_boundary_violation_total": 0,
        },
    )
    _json(root / "xunce-stage21-6-lineage-audit.json", {"passed": True})
    seed_rows = []
    source_sha = _sha256(source_checkpoint_path)
    checkpoint_sha = _sha256(checkpoint_path)
    for seed in (seed,):
        seed_root = root / f"seed_{seed}"
        stage21_1 = seed_root / "stage21_1"
        stage21_4 = seed_root / "stage21_4"
        stage21_5 = seed_root / "stage21_5"
        pre = stage21_5 / "pre_ppo_xunce"
        post = stage21_5 / "post_ppo_xunce"
        pre.mkdir(parents=True, exist_ok=True)
        post.mkdir(parents=True, exist_ok=True)
        _json(
            stage21_1 / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json",
            {
                "status": "passed",
                "summary": str(stage21_1 / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json"),
            },
        )
        if write_collector_audit:
            _json(
                stage21_1 / "xunce-stage21-1-manifest.json",
                {
                    "model_audit": {
                        "xunce_candidate_checkpoint": str(collector_source_checkpoint_path),
                        "xunce_checkpoint_audit": {
                            "checkpoint_loaded": True,
                            "checkpoint_sha256": _sha256(collector_source_checkpoint_path),
                        },
                    }
                },
            )
            if not omit_collector_checkpoint_path:
                manifest = _read_json(stage21_1 / "xunce-stage21-1-manifest.json")
                manifest["model_audit"]["xunce_checkpoint_audit"]["checkpoint_path"] = str(collector_source_checkpoint_path)
                _json(stage21_1 / "xunce-stage21-1-manifest.json", manifest)
        _json(
            stage21_4 / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json",
            {
                "experimental_checkpoint_path": str(checkpoint_path),
                "experimental_checkpoint_sha256": checkpoint_sha,
                "source_xunce_checkpoint_sha256": source_sha,
                "source_xunce_candidate_checkpoint": str(source_checkpoint_path),
                "checkpoint_reload_passed": checkpoint_reload_passed,
                "experimental_checkpoint": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
                "training_or_release_authorized": False,
            },
        )
        _json(
            stage21_4 / "xunce-stage21-4-checkpoint-audit.json",
            {
                "checkpoint_reload_passed": checkpoint_reload_passed,
                "experimental_checkpoint_path": str(checkpoint_path),
                "experimental_checkpoint_sha256": checkpoint_sha,
                "metadata": {
                    "experimental_only": experimental_only,
                    "source_checkpoint_path": str(source_checkpoint_path),
                    "source_checkpoint_sha256": source_sha,
                },
                "reload_audit": {
                    "checkpoint_loaded": True,
                    "checkpoint_path": str(reload_checkpoint_path or checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha,
                },
            },
        )
        _json(
            stage21_4 / "xunce-stage21-4-gradient-audit.json",
            {
                "component_grad_norms": {
                    "policy_loss_grad_norm": 1.0,
                    "value_loss_grad_norm": 1.0,
                    "entropy_loss_grad_norm": 0.01,
                    "total_loss_grad_norm": 2.0,
                },
                "pre_clip_grad_norm": 3.0,
                "post_clip_grad_norm": 1.0,
                "grad_norm_finite": True,
            },
        )
        _json(
            stage21_5 / "xunce-stage21-5-post-update-evaluation-summary.json",
            {
                "pre_evaluation_root": str(pre),
                "post_evaluation_root": str(post),
                "final_coverage_rate_capped_delta": coverage_delta,
                "coverage_curve_auc_capped_delta": coverage_delta,
            },
        )
        _jsonl(pre / "xunce-exploration-coverage-model-inference.jsonl", [_inference_row(0, 1, 0.60)])
        _jsonl(
            post / "xunce-exploration-coverage-model-inference.jsonl",
            [_inference_row(1 if action_shift else 0, 2 if action_shift else 1, 0.60 + probability_delta, best_delta=probability_delta)],
        )
        _jsonl(pre / "xunce-exploration-coverage-candidate-metric-audit.jsonl", _candidate_rows())
        seed_rows.append(
            {
                "seed": seed,
                "seed_root": str(seed_root),
                "stage21_1_root": str(stage21_1),
                "stage21_4_root": str(stage21_4),
                "stage21_5_root": str(stage21_5),
                "pre_clip_grad_norm": 3.0,
                "post_clip_grad_norm": 1.0,
                "max_post_update_approx_kl": 0.001,
                "min_entropy": 3.0,
                "grad_norm_finite": True,
                "parameter_delta_l2": 0.01,
                "final_coverage_delta": coverage_delta,
                "coverage_auc_delta": coverage_delta,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
                "training_or_release_authorized": False,
                "checkpoint_reload_passed": True,
                "experimental_only": True,
                "experimental_checkpoint": True,
            }
        )
    _jsonl(root / "xunce-stage21-6-seed-results.jsonl", seed_rows)


def _chain_root(root: Path, seed: int, round_index: int) -> Path:
    return root / "rounds" / f"seed_{seed}" / f"round_{round_index:03d}"


def _inference_row(selected: int, rank: int, top_probability: float, *, best_delta: float = 0.0) -> dict[str, object]:
    probs = [top_probability, 0.30 + best_delta, max(0.0, 0.10 - best_delta)]
    if selected == 1:
        probs = [0.35, top_probability, 0.05]
    return {
        "scenario_id": "scenario-a",
        "step_index": 0,
        "current_cell": [0, 0],
        "covered_cells_hash": "covered-hash",
        "candidate_set_hash": "candidate-hash",
        "candidate_cells": [[0, 1], [1, 0], [1, 1]],
        "action_mask": [True, True, True],
        "selected_action_index": selected,
        "selected_rank": rank,
        "selected_probability": probs[selected],
        "action_probs": probs,
        "logits": [2.0, 1.0, 0.0],
        "masked_logits": [2.0, 1.0, 0.0],
        "value": 0.0,
        "finite_outputs": True,
    }


def _candidate_rows() -> list[dict[str, object]]:
    base = {
        "scenario_id": "scenario-a",
        "step_index": 0,
        "current_cell": [0, 0],
        "covered_cells_hash": "covered-hash",
        "candidate_set_hash": "candidate-hash",
        "action_mask_valid": True,
    }
    return [
        {**base, "candidate_index": 0, "coverage_gain_per_path_cost": 1.0},
        {**base, "candidate_index": 1, "coverage_gain_per_path_cost": 3.0},
        {**base, "candidate_index": 2, "coverage_gain_per_path_cost": 0.5},
    ]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
