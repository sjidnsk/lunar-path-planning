import json
import math
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage21_4_runs_tiny_ppo_update_and_writes_experimental_checkpoint(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_4_tiny_ppo_update_smoke import run_xunce_stage21_4_tiny_ppo_update_smoke

    stage21_3, source_checkpoint, high_fidelity_config = _write_stage21_4_inputs(tmp_path)
    config = _write_config(tmp_path, stage21_3, source_checkpoint, high_fidelity_config)
    source_hash_before = _sha256_file(source_checkpoint)

    summary = run_xunce_stage21_4_tiny_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "D_drive_like" / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "implement_stage21_5_single_seed_ppo_pilot"
    assert summary["runs_new_ppo_update"] is True
    assert summary["publishes_checkpoint"] is False
    assert summary["replaces_default_policy"] is False
    assert summary["connects_real_executor"] is False
    assert summary["starts_online_canary"] is False
    assert summary["train_transition_count"] == 2
    assert summary["sample_count_too_low_for_performance_claim"] is True
    assert summary["checkpoint_reload_passed"] is True
    assert summary["experimental_checkpoint"] is True
    assert summary["parameter_delta_l2"] > 0.0
    assert _sha256_file(source_checkpoint) == source_hash_before

    checkpoint_path = Path(summary["experimental_checkpoint_path"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["metadata"]["experimental_only"] is True
    assert payload["metadata"]["publishes_checkpoint"] is False
    assert payload["metadata"]["replaces_default_policy"] is False
    assert payload["metadata"]["training_or_release_authorized"] is False
    assert payload["metadata"]["canary_traffic_fraction"] == 0.0
    for field in (
        "candidate_feature_count",
        "edge_feature_count",
        "memory_feature_count",
        "context_feature_count",
        "missing_indicator_count",
        "hidden_dim",
        "message_passing_layers",
    ):
        assert field in payload["metadata"]

    loss_rows = _read_jsonl(Path(summary["loss_audit"]))
    assert len(loss_rows) == 1
    assert math.isfinite(loss_rows[0]["total_loss"])
    assert math.isfinite(loss_rows[0]["post_update_approx_kl"])


def test_stage21_4_consumes_only_train_split_rows(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_4_tiny_ppo_update_smoke import run_xunce_stage21_4_tiny_ppo_update_smoke

    stage21_3, source_checkpoint, high_fidelity_config = _write_stage21_4_inputs(tmp_path, include_validation=True)
    config = _write_config(tmp_path, stage21_3, source_checkpoint, high_fidelity_config)

    summary = run_xunce_stage21_4_tiny_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["train_transition_count"] == 2
    loss_rows = _read_jsonl(Path(summary["loss_audit"]))
    assert loss_rows[0]["transition_count"] == 2


def test_stage21_4_writes_loss_scaling_and_component_gradient_audit(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_4_tiny_ppo_update_smoke import run_xunce_stage21_4_tiny_ppo_update_smoke

    stage21_3, source_checkpoint, high_fidelity_config = _write_stage21_4_inputs(tmp_path)
    config = _write_config(
        tmp_path,
        stage21_3,
        source_checkpoint,
        high_fidelity_config,
        advantage_clip_abs=0.5,
        normalize_minibatch_advantages=True,
        loss_scale=0.25,
        value_loss_coefficient=0.1,
        policy_loss_coefficient=2.0,
    )

    summary = run_xunce_stage21_4_tiny_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["advantage_clip_abs"] == 0.5
    assert summary["normalize_minibatch_advantages"] is True
    assert summary["loss_scale"] == 0.25
    assert summary["policy_loss_coefficient"] == 2.0
    assert math.isfinite(summary["policy_loss_grad_norm"])
    assert math.isfinite(summary["value_loss_grad_norm"])
    assert math.isfinite(summary["entropy_loss_grad_norm"])
    assert math.isfinite(summary["total_loss_grad_norm"])
    loss_row = _read_jsonl(Path(summary["loss_audit"]))[0]
    assert loss_row["loss_scale"] == 0.25
    assert loss_row["value_loss_coefficient"] == 0.1
    assert loss_row["policy_loss_coefficient"] == 2.0
    assert "effective_advantage_std" in loss_row
    assert "total_loss_grad_norm" in loss_row
    assert "policy_loss_grad_norm" in loss_row
    gradient = json.loads(Path(summary["gradient_audit"]).read_text(encoding="utf-8"))
    assert gradient["loss_scale"] == 0.25
    assert gradient["policy_loss_coefficient"] == 2.0
    assert gradient["advantage_clip_abs"] == 0.5
    assert "total_loss_grad_norm" in gradient["component_grad_norms"]


def test_stage21_4_gates_policy_kl_not_behavior_kl_for_synthetic_credit_rows(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_4_tiny_ppo_update_smoke import run_xunce_stage21_4_tiny_ppo_update_smoke

    stage21_3, source_checkpoint, high_fidelity_config = _write_stage21_4_inputs(tmp_path)
    rows_path = stage21_3 / "xunce-stage21-3-ppo-trainable-batch.jsonl"
    rows = _read_jsonl(rows_path)
    _mark_synthetic_behavior_rows(rows, behavior_offset=2.7, policy_offset=0.0)
    _write_jsonl(rows_path, rows)
    config = _write_config(tmp_path, stage21_3, source_checkpoint, high_fidelity_config, max_abs_approx_kl=1.5)

    summary = run_xunce_stage21_4_tiny_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["ppo_ratio_old_log_prob_source"] == "behavior_policy_when_present"
    assert summary["kl_gate_source"] == "policy_old_logprob_when_available/v1"
    assert summary["behavior_policy_kl_diagnostic_only"] is True
    assert summary["final_post_update_behavior_approx_kl"] > 1.5
    assert abs(summary["final_pre_update_policy_approx_kl"]) < 1.0e-4
    assert abs(summary["final_post_update_policy_approx_kl"]) <= 1.5
    loss_row = _read_jsonl(Path(summary["loss_audit"]))[0]
    assert loss_row["ratio_max"] < math.exp(-2.6)
    assert loss_row["ratio_min"] > math.exp(-2.8)
    assert loss_row["clip_fraction"] == 1.0


def test_stage21_4_still_rejects_policy_kl_over_threshold(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_4_tiny_ppo_update_smoke import run_xunce_stage21_4_tiny_ppo_update_smoke

    stage21_3, source_checkpoint, high_fidelity_config = _write_stage21_4_inputs(tmp_path)
    rows_path = stage21_3 / "xunce-stage21-3-ppo-trainable-batch.jsonl"
    rows = _read_jsonl(rows_path)
    _mark_synthetic_behavior_rows(rows, behavior_offset=0.0, policy_offset=2.7)
    _write_jsonl(rows_path, rows)
    config = _write_config(tmp_path, stage21_3, source_checkpoint, high_fidelity_config, max_abs_approx_kl=1.5)

    summary = run_xunce_stage21_4_tiny_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_4_ppo_update_numerics"
    assert "post_update_approx_kl_exceeded" in summary["blocking_reason_codes"]
    assert summary["final_post_update_policy_approx_kl"] > 1.5


def test_stage21_4_boundary_flag_hard_fails(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_4_tiny_ppo_update_smoke import run_xunce_stage21_4_tiny_ppo_update_smoke

    stage21_3, source_checkpoint, high_fidelity_config = _write_stage21_4_inputs(tmp_path)
    config = _write_config(
        tmp_path,
        stage21_3,
        source_checkpoint,
        high_fidelity_config,
        publishes_checkpoint=True,
    )

    summary = run_xunce_stage21_4_tiny_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage21_4_tiny_ppo_boundary_rejections"
    assert "publishes_checkpoint" in summary["blocking_reason_codes"]
    assert summary["runs_new_ppo_update"] is False


def test_stage21_4_rejects_missing_train_rows(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_4_tiny_ppo_update_smoke import run_xunce_stage21_4_tiny_ppo_update_smoke

    stage21_3, source_checkpoint, high_fidelity_config = _write_stage21_4_inputs(tmp_path, all_validation=True)
    config = _write_config(tmp_path, stage21_3, source_checkpoint, high_fidelity_config)

    summary = run_xunce_stage21_4_tiny_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_4_train_split_batch_contract"
    assert "stage21_4_train_transition_count_short" in summary["blocking_reason_codes"]


def test_stage21_4_rejects_invalid_sampling_mask_action(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_4_tiny_ppo_update_smoke import run_xunce_stage21_4_tiny_ppo_update_smoke

    stage21_3, source_checkpoint, high_fidelity_config = _write_stage21_4_inputs(tmp_path, invalid_sampling_mask=True)
    config = _write_config(tmp_path, stage21_3, source_checkpoint, high_fidelity_config)

    summary = run_xunce_stage21_4_tiny_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_4_ppo_update_numerics"
    assert "action_not_allowed_by_sampling_mask" in summary["blocking_reason_codes"]


def test_stage21_4_rejects_missing_sampling_mask(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_4_tiny_ppo_update_smoke import run_xunce_stage21_4_tiny_ppo_update_smoke

    stage21_3, source_checkpoint, high_fidelity_config = _write_stage21_4_inputs(tmp_path, omit_sampling_mask=True)
    config = _write_config(tmp_path, stage21_3, source_checkpoint, high_fidelity_config)

    summary = run_xunce_stage21_4_tiny_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_4_ppo_update_numerics"
    assert "sampling_mask_missing" in summary["blocking_reason_codes"]


def test_stage21_4_rejects_sampling_mask_not_subset_of_hard_risk_mask(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_4_tiny_ppo_update_smoke import run_xunce_stage21_4_tiny_ppo_update_smoke

    stage21_3, source_checkpoint, high_fidelity_config = _write_stage21_4_inputs(tmp_path, hard_risk_mask_mismatch=True)
    config = _write_config(tmp_path, stage21_3, source_checkpoint, high_fidelity_config)

    summary = run_xunce_stage21_4_tiny_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_4_ppo_update_numerics"
    assert "sampling_mask_not_subset_of_hard_risk_clean_mask" in summary["blocking_reason_codes"]


def test_stage21_4_rejects_sampling_temperature_mismatch(tmp_path: Path) -> None:
    from scripts.run_xunce_stage21_4_tiny_ppo_update_smoke import run_xunce_stage21_4_tiny_ppo_update_smoke

    stage21_3, source_checkpoint, high_fidelity_config = _write_stage21_4_inputs(tmp_path, row_sampling_temperature=0.5)
    config = _write_config(tmp_path, stage21_3, source_checkpoint, high_fidelity_config)

    summary = run_xunce_stage21_4_tiny_ppo_update_smoke(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_4_ppo_update_numerics"
    assert "row_sampling_temperature_mismatch" in summary["blocking_reason_codes"]


def _write_stage21_4_inputs(
    tmp_path: Path,
    *,
    include_validation: bool = False,
    all_validation: bool = False,
    invalid_sampling_mask: bool = False,
    omit_sampling_mask: bool = False,
    hard_risk_mask_mismatch: bool = False,
    row_sampling_temperature: float | None = None,
) -> tuple[Path, Path, Path]:
    from scripts.xunce_full_network_common import XunceFullNetworkV1

    torch.manual_seed(214)
    xunce_config = {
        "candidate_feature_count": 8,
        "edge_feature_count": 5,
        "memory_feature_count": 6,
        "context_feature_count": 7,
        "missing_indicator_count": 3,
        "hidden_dim": 16,
        "message_passing_layers": 1,
        "candidate_count": 3,
    }
    model = XunceFullNetworkV1(dropout=0.0, **{key: xunce_config[key] for key in xunce_config if key != "candidate_count"})
    source_checkpoint = tmp_path / "source-xunce.pt"
    torch.save({"model_state_dict": model.state_dict(), "metadata": xunce_config}, source_checkpoint)
    high_fidelity_config = tmp_path / "high_fidelity.json"
    high_fidelity_config.write_text(json.dumps({"schema_version": "test/v1"}, ensure_ascii=False), encoding="utf-8")

    stage21_3 = tmp_path / "stage21_3"
    stage21_3.mkdir()
    (stage21_3 / "xunce-stage21-3-ppo-batch-validation-summary.json").write_text(
        json.dumps(
            {
                "schema_version": "xunce-stage21-3-ppo-batch-validation-summary/v1",
                "status": "passed",
                "next_required_change": "implement_stage21_4_tiny_ppo_update_smoke",
                "reward_profile_hash": "reward-hash",
                "stage21_1_collector_root": str(tmp_path / "stage21_1"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (stage21_3 / "xunce-stage21-3-lineage-audit.json").write_text(
        json.dumps(
            {
                "schema_version": "xunce-stage21-3-lineage-audit/v1",
                "reward_profile_hash_count": 1,
                "reward_profile_hash": "reward-hash",
                "reward_profile_hashes": ["reward-hash"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    stage21_1 = tmp_path / "stage21_1"
    stage21_1.mkdir()
    (stage21_1 / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json").write_text(
        json.dumps({"sampling_temperature": 1.0}, ensure_ascii=False),
        encoding="utf-8",
    )
    rows = [
        _batch_row(
            model,
            "t0",
            step=0,
            split="validation" if all_validation else "train",
            invalid_sampling_mask=invalid_sampling_mask,
            omit_sampling_mask=omit_sampling_mask,
            hard_risk_mask_mismatch=hard_risk_mask_mismatch,
            row_sampling_temperature=row_sampling_temperature,
        ),
        _batch_row(model, "t1", step=1, split="validation" if all_validation else "train", action_index=2),
    ]
    if include_validation:
        rows.append(_batch_row(model, "v0", step=2, split="validation", action_index=0))
    _write_jsonl(stage21_3 / "xunce-stage21-3-ppo-trainable-batch.jsonl", rows)
    return stage21_3, source_checkpoint, high_fidelity_config


def _batch_row(
    model,
    transition_id: str,
    *,
    step: int,
    split: str,
    action_index: int = 1,
    invalid_sampling_mask: bool = False,
    omit_sampling_mask: bool = False,
    hard_risk_mask_mismatch: bool = False,
    row_sampling_temperature: float | None = None,
) -> dict:
    xunce_batch = _xunce_batch()
    tensors = {key: _tensor_from_payload(value) for key, value in xunce_batch.items()}
    with torch.no_grad():
        output = model(**tensors)
    sampling_mask = [True, True, True]
    hard_risk_clean_mask = [True, True, True]
    if invalid_sampling_mask:
        sampling_mask[action_index] = False
    if hard_risk_mask_mismatch:
        hard_risk_clean_mask[action_index] = False
    logits = output.masked_logits[0].masked_fill(~torch.tensor(sampling_mask, dtype=torch.bool), -1.0e9)
    if invalid_sampling_mask:
        old_log_prob = -0.1
    else:
        old_log_prob = float(torch.distributions.Categorical(logits=logits).log_prob(torch.tensor(action_index)))
    old_value = float(output.value[0])
    row = {
        "schema_version": "xunce-stage21-3-ppo-batch/v1",
        "transition_id": transition_id,
        "scenario_id": "s1",
        "step_index": step,
        "xunce_batch": xunce_batch,
        "action_index": action_index,
        "old_log_prob": old_log_prob,
        "old_value": old_value,
        "reward": 1.0,
        "return": old_value + 1.0,
        "advantage": 1.0,
        "done": step == 1,
        "transition_trainable": True,
        "reward_trainable": True,
        "stage21_3_split": split,
        "info": {
            "action_mask": [True, True, True],
            "hard_risk_clean_mask": hard_risk_clean_mask,
            "hard_risk_violation": False,
        },
    }
    if not omit_sampling_mask:
        row["info"]["sampling_mask"] = sampling_mask
    if row_sampling_temperature is not None:
        row["info"]["sampling_temperature"] = row_sampling_temperature
    return row


def _mark_synthetic_behavior_rows(rows: list[dict], *, behavior_offset: float, policy_offset: float) -> None:
    for row in rows:
        info = row.setdefault("info", {})
        original = float(row["old_log_prob"])
        behavior = original + behavior_offset
        policy = original + policy_offset
        row["behavior_policy_id"] = "synthetic_credit_mixture_policy/v1"
        row["synthetic_credit_mixture_probability"] = 1.0
        row["synthetic_credit_target_index"] = row["action_index"]
        row["synthetic_credit_target_selected"] = True
        row["old_policy_log_prob"] = policy
        row["old_behavior_log_prob"] = behavior
        row["old_log_prob"] = behavior
        info["behavior_policy_id"] = "synthetic_credit_mixture_policy/v1"
        info["old_policy_log_prob"] = policy
        info["old_behavior_log_prob"] = behavior


def _xunce_batch() -> dict:
    return {
        "candidate_features": _payload(
            [
                [
                    [0.1, 0.2, 0.1, 0.2, 0.2, 0.0, 1.0, 0.1],
                    [0.2, 0.3, 0.2, 0.3, 0.3, 0.0, 1.0, 0.2],
                    [0.3, 0.4, 0.3, 0.4, 0.4, 0.0, 1.0, 0.3],
                ]
            ]
        ),
        "edge_features": _payload([[0.5, 0.1, 0.2, 1.0, 1.0], [0.5, 0.2, 0.3, 1.0, 1.0]]),
        "edge_index": {"dtype": "int64", "shape": [2, 2], "values": [[0, 1], [1, 2]]},
        "memory_features": _payload([[0.1, 0.2, 1.0, 1.0, 0.0, 0.0]]),
        "context_features": _payload(
            [
                [
                    [1.0, 1.0, 0.0, 0.1, 0.2, 0.3, 0.0],
                    [1.0, 1.0, 0.0, 0.2, 0.3, 0.4, 0.0],
                    [1.0, 1.0, 0.0, 0.3, 0.4, 0.5, 0.0],
                ]
            ]
        ),
        "action_mask": {"dtype": "bool", "shape": [1, 3], "values": [[True, True, True]]},
        "candidate_missing_indicators": _payload([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
    }


def _payload(values) -> dict:
    return {"dtype": "float32", "shape": _shape(values), "values": values}


def _shape(values) -> list[int]:
    shape: list[int] = []
    current = values
    while isinstance(current, list):
        shape.append(len(current))
        current = current[0] if current else []
    return shape


def _tensor_from_payload(payload: dict) -> torch.Tensor:
    dtype = torch.bool if payload["dtype"] == "bool" else torch.long if payload["dtype"] == "int64" else torch.float32
    return torch.tensor(payload["values"], dtype=dtype)


def _write_config(tmp_path: Path, stage21_3: Path, source_checkpoint: Path, high_fidelity_config: Path, **overrides) -> Path:
    config = {
        "schema_version": "xunce-stage21-4-tiny-ppo-update-smoke-config/v1",
        "stage21_3_ppo_batch_validation_root": str(stage21_3),
        "xunce_candidate_checkpoint": str(source_checkpoint),
        "high_fidelity_config": str(high_fidelity_config),
        "epochs": 1,
        "learning_rate": 1.0e-3,
        "clip_ratio": 0.2,
        "value_loss_coefficient": 0.5,
        "entropy_coefficient": 0.01,
        "max_grad_norm": 1.0,
        "sampling_temperature": 1.0,
        "max_abs_approx_kl": 10.0,
        "min_train_transition_count": 1,
        "min_transition_count_for_performance_claim": 200,
        "require_d_drive_output_root": False,
        "offline_ppo_update_smoke_authorized": True,
        "stage21_4_authorized": False,
        "runs_new_ppo_update": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config.update(overrides)
    path = tmp_path / "stage21_4_config.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
