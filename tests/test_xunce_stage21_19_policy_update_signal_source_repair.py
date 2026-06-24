import json
import math
from pathlib import Path

import scripts.run_xunce_stage21_19_policy_update_signal_source_repair as s19


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _log_prob(logits: list[float], action: int) -> float:
    max_logit = max(logits)
    denom = max_logit + math.log(sum(math.exp(value - max_logit) for value in logits))
    return logits[action] - denom


def _batch_row(*, action_index: int = 1, old_log_prob: float | None = None) -> dict:
    logits = [0.1, 0.4, -0.2]
    return {
        "transition_id": "s0:step-0:sample-0",
        "scenario_id": "s0",
        "step_index": 0,
        "action_index": action_index,
        "old_log_prob": _log_prob(logits, action_index) if old_log_prob is None else old_log_prob,
        "advantage": 1.0,
        "xunce_batch": {"action_mask": {"values": [[True, True, True]]}},
        "info": {
            "current_cell_before": [0, 0],
            "covered_cells_hash": "covered",
            "candidate_set_hash": "candidates",
            "old_sampling_logits": logits,
            "sampling_mask": [True, True, True],
            "action_mask": [True, True, True],
            "hard_risk_clean_mask": [True, True, True],
        },
    }


def test_action_logprob_binding_recomputes_old_log_prob(tmp_path: Path) -> None:
    stage21_3 = tmp_path / "stage21_3"
    _write_jsonl(stage21_3 / "xunce-stage21-3-ppo-trainable-batch.jsonl", [_batch_row()])

    audit = s19._audit_action_logprob_binding(
        [{"stage21_3_root": str(stage21_3), "round_index": 1, "seed": 2101}],
        {"old_log_prob_recompute_tolerance": 1e-6, "max_audit_sample_rows": 10},
    )

    assert audit["binding_passed"] is True
    assert audit["old_log_prob_recompute_max_abs_error"] == 0.0
    assert audit["invalid_action_mask_count"] == 0


def test_action_logprob_binding_rejects_mask_and_logprob_mismatch(tmp_path: Path) -> None:
    stage21_3 = tmp_path / "stage21_3"
    row = _batch_row(action_index=2, old_log_prob=0.0)
    row["info"]["sampling_mask"][2] = False
    _write_jsonl(stage21_3 / "xunce-stage21-3-ppo-trainable-batch.jsonl", [row])

    audit = s19._audit_action_logprob_binding(
        [{"stage21_3_root": str(stage21_3), "round_index": 1, "seed": 2101}],
        {"old_log_prob_recompute_tolerance": 1e-6, "max_audit_sample_rows": 10},
    )

    assert audit["binding_passed"] is False
    assert audit["invalid_action_mask_count"] == 1
    assert audit["nonfinite_log_prob_count"] == 1


def test_action_logprob_binding_requires_old_sampling_logits(tmp_path: Path) -> None:
    stage21_3 = tmp_path / "stage21_3"
    row = _batch_row()
    row["info"].pop("old_sampling_logits")
    row["info"]["old_logits"] = [0.1, 0.4, -0.2]
    _write_jsonl(stage21_3 / "xunce-stage21-3-ppo-trainable-batch.jsonl", [row])

    audit = s19._audit_action_logprob_binding(
        [{"stage21_3_root": str(stage21_3), "round_index": 1, "seed": 2101}],
        {"old_log_prob_recompute_tolerance": 1e-6, "max_audit_sample_rows": 10},
    )

    assert audit["binding_passed"] is False
    assert audit["missing_required_field_count"] == 1


def test_action_logprob_binding_requires_explicit_masks(tmp_path: Path) -> None:
    stage21_3 = tmp_path / "stage21_3"
    row = _batch_row()
    row["info"].pop("sampling_mask")
    _write_jsonl(stage21_3 / "xunce-stage21-3-ppo-trainable-batch.jsonl", [row])

    audit = s19._audit_action_logprob_binding(
        [{"stage21_3_root": str(stage21_3), "round_index": 1, "seed": 2101}],
        {"old_log_prob_recompute_tolerance": 1e-6, "max_audit_sample_rows": 10},
    )

    assert audit["binding_passed"] is False
    assert audit["missing_required_field_count"] == 1


def test_candidate_logit_sensitivity_filters_xunce_and_uses_candidate_metric_best_cpc(tmp_path: Path) -> None:
    stage21_5 = tmp_path / "stage21_5"
    pre_root = tmp_path / "pre"
    post_root = tmp_path / "post"
    summary = {"pre_evaluation_root": str(pre_root), "post_evaluation_root": str(post_root)}
    (stage21_5 / "xunce-stage21-5-post-update-evaluation-summary.json").parent.mkdir(parents=True, exist_ok=True)
    (stage21_5 / "xunce-stage21-5-post-update-evaluation-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    base = {
        "scenario_id": "s0",
        "step_index": 0,
        "current_cell": [0, 0],
        "covered_cells_hash": "covered",
        "candidate_set_hash": "candidates",
        "candidate_cells": [[0, 1], [1, 1]],
        "action_mask": [True, True],
        "logits": [0.0, 0.0],
        "masked_logits": [0.0, 0.0],
        "selected_action_index": 0,
        "selected_rank": 1,
    }
    pre_xunce = base | {"policy": "xunce", "action_probs": [0.7, 0.3]}
    post_xunce = base | {"policy": "xunce", "action_probs": [0.6, 0.4]}
    pre_incumbent = base | {"policy": "incumbent", "action_probs": [0.0, 1.0]}
    post_incumbent = base | {"policy": "incumbent", "action_probs": [1.0, 0.0]}
    _write_jsonl(pre_root / "xunce-exploration-coverage-model-inference.jsonl", [pre_xunce, pre_incumbent])
    _write_jsonl(post_root / "xunce-exploration-coverage-model-inference.jsonl", [post_xunce, post_incumbent])
    _write_jsonl(
        pre_root / "xunce-exploration-coverage-candidate-metric-audit.jsonl",
        [
            base | {"policy": "xunce", "candidate_index": 0, "coverage_gain_per_path_cost": 1.0, "action_mask_valid": True},
            base | {"policy": "xunce", "candidate_index": 1, "coverage_gain_per_path_cost": 2.0, "action_mask_valid": True},
        ],
    )

    audit = s19._audit_candidate_logit_sensitivity(
        [{"stage21_5_root": str(stage21_5)}],
        {},
    )

    assert audit["inference_binding_passed"] is True
    assert audit["duplicate_xunce_strong_key_count"] == 0
    assert audit["strong_state_join_available_count"] == 1
    assert audit["candidate_metric_join_count"] == 1
    assert audit["best_coverage_per_cost_probability_delta_mean"] == 0.10000000000000003
    assert audit["mean_abs_probability_delta"] == 0.1


def test_ratio_clip_advantage_routes_flat_advantage(tmp_path: Path) -> None:
    stage21_3 = tmp_path / "stage21_3"
    stage21_4 = tmp_path / "stage21_4"
    rows = [_batch_row(), _batch_row()]
    for row in rows:
        row["advantage"] = 0.0
    _write_jsonl(stage21_3 / "xunce-stage21-3-ppo-trainable-batch.jsonl", rows)
    _write_jsonl(
        stage21_4 / "xunce-stage21-4-ppo-loss-audit.jsonl",
        [{"policy_loss_grad_norm": 0.0, "post_update_clip_fraction": 0.0, "ratio_min": 1.0, "ratio_max": 1.0}],
    )

    audit = s19._audit_ratio_clip_advantage(
        [{"stage21_3_root": str(stage21_3), "stage21_4_root": str(stage21_4)}],
        {"min_advantage_std": 1e-6, "min_abs_advantage_mean": 0.0, "min_policy_grad_norm": 1e-8},
    )

    assert audit["ratio_clip_advantage_passed"] is False
    assert audit["weak_or_flat_advantage"] is True


def test_route_prioritizes_action_binding_failure() -> None:
    status, route, reason = s19._route(
        input_reasons=[],
        boundary_reasons=[],
        action_logprob={"binding_passed": False},
        ratio_clip={"ratio_clip_advantage_passed": True},
        policy_gradient={"policy_gradient_path_passed": True},
        candidate_logit={"mean_abs_probability_delta": 0.0, "candidate_logit_delta_max": 0.0, "selected_rank_changed_count": 0},
        smoke={"status": "skipped", "mean_abs_probability_delta": 0.0},
        config={"policy_signal_enhanced_probability_delta_threshold": 0.005, "candidate_logit_delta_margin_threshold": 0.0001},
    )

    assert status == "failed"
    assert route == s19.ROUTE_ACTION_BINDING
    assert reason == "action_logprob_mask_or_candidate_binding_failed"


def test_route_detects_logit_margin_without_rank_crossing() -> None:
    status, route, reason = s19._route(
        input_reasons=[],
        boundary_reasons=[],
        action_logprob={"binding_passed": True},
        ratio_clip={"ratio_clip_advantage_passed": True},
        policy_gradient={"policy_gradient_path_passed": True},
        candidate_logit={"mean_abs_probability_delta": 0.0001, "candidate_logit_delta_max": 0.01, "selected_rank_changed_count": 0},
        smoke={"status": "skipped", "mean_abs_probability_delta": 0.0},
        config={"policy_signal_enhanced_probability_delta_threshold": 0.005, "candidate_logit_delta_margin_threshold": 0.0001},
    )

    assert status == "failed"
    assert route == s19.ROUTE_MARGIN
    assert reason == "logits_changed_without_rank_crossing"
