"""Stage 6 frozen config 与 Stage 5 authority 的 fail-closed 合同。"""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, fields
import hashlib
import importlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/ppo_highres_frontier_stage6_v1.json"
STAGE5_GATE = Path(
    "D:/xunce/out/ppo_frontier/"
    "s5-task6-fix-r1-20260714T223702Z/s5/gate.json"
)


def _config_module():
    return importlib.import_module("lunar_exploration_ppo.configs.stage6")


def _workflow_module():
    return importlib.import_module("lunar_exploration_ppo.workflows.stage6")


def test_stage6_config_freezes_standard_training_eval_contract() -> None:
    module = _config_module()
    config = module.load_stage6_config(CONFIG)

    assert config.schema_version == "ppo_highres_frontier_stage6_config/v1"
    assert config.stage_id == "ppo_highres_frontier_stage6_standard_training_eval/v1"
    assert config.stage5_authority.model_dump() == {
        "commit_sha256": "b635740ee021258ef31811ec87c60add839fc5f9",
        "commit_tree": "edec4c4dbed7f91efa9856a9df899c594bbcf04c",
        "gate_path": STAGE5_GATE.as_posix(),
        "gate_sha256": "5fc93a7fb0d1d23c1c2ee99db14e15f6880c422d1ce85a83946542ebbe238bdf",
        "review_sha256": "3c69d565f27bdabf64cf1ca4728be5da420ffa5057444f29af9b717bd2d9d9db",
        "manifest_sha256": "edee94cf2a0f39f89cf040d850af154e060b07c6a66d0bfc47386d5b05ee5672",
        "checkpoint_sha256": "d1d80e6478262a68d01208ddc1e8721768b56109e8dd009e862dd668cb39dc4c",
        "policy_state_sha256": "51fecca55af92838f302951ae1272063cd945ef481579af5f6600469d9a7fe86",
        "authorized_stage": "ppo_highres_frontier_stage6_standard_training_eval/v1",
        "performance_advantage_established": False,
    }
    assert config.scale.model_dump() == {
        "profile": "Standard v1",
        "roi_size_m": 128,
        "highres_shape": (256, 256),
        "highres_cell_size_m": 0.5,
        "lowres_shape": (32, 32),
        "lowres_cell_size_m": 4.0,
        "local_crop_shape": (96, 96),
        "frontier_top_m": 1024,
        "max_steps": 128,
        "stagnation_no_gain_steps": 16,
        "success_threshold": 0.99,
        "coverage_denominator": "reachable_observable_free_highres_cells/v1",
        "coverable_mask_exact": True,
        "truth_source_kind": "procedural_lunar_rock_crater_proxy/v1",
    }
    assert config.rollout.model_dump() == {
        "num_envs": 8,
        "steps_per_env": 128,
        "batch_size": 1024,
        "multiprocessing_start_method": "spawn",
        "diagnostic_empty_reset_counts_toward_quota": False,
    }
    assert config.ppo.model_dump() == {
        "ppo_epochs": 4,
        "effective_minibatch_size": 256,
        "physical_microbatch_size": 32,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "value_clip_eps": 0.2,
        "value_loss_coef": 0.5,
        "frontier_entropy_coef": 0.01,
        "theta_entropy_enabled": False,
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "adam_eps": 1e-5,
        "weight_decay": 1e-4,
        "max_grad_norm": 0.5,
        "target_kl": 0.03,
        "compute_dtype": "float32",
        "amp_enabled": False,
    }
    assert config.training.model_dump() == {
        "initialization_source": "stage4_smoke_policy_weights_fresh_adamw/v1",
        "seeds": (20260716,),
        "updates_per_seed": 100,
        "validation_every_updates": 10,
        "validation_episodes": 16,
        "validation_policy": "deterministic_argmax_frontier_mean_theta/v1",
        "periodic_updates": (50, 100),
        "periodic_keep_count": 5,
    }
    assert config.training.seeds == (20260716,)
    assert config.training.updates_per_seed == 100
    forbidden_extension_fields = {
        "auto_expand_seeds",
        "potential_threshold",
        "stage6b_runner",
    }
    assert forbidden_extension_fields.isdisjoint(module.Stage6Config.model_fields)
    assert forbidden_extension_fields.isdisjoint(
        module.Stage6TrainingConfig.model_fields
    )
    assert config.evaluation.model_dump() == {
        "test_episodes": 64,
        "unseen_episodes": 64,
        "baseline_methods": (
            "random_valid_frontier",
            "nearest_frontier",
            "max_potential_gain_frontier",
            "gain_over_cost_frontier",
        ),
        "baseline_episodes_per_method_split": 64,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20260716,
        "ppo_theta_source": "policy_theta_mu/v1",
        "baseline_theta_source": "candidate_recommended_theta/v1",
    }
    assert config.catalog_split_counts == {
        "train": 700,
        "validation": 150,
        "test": 150,
        "unseen": 64,
    }
    assert config.device == "cuda"
    assert config.allow_cpu_fallback is False
    assert config.output_root == "D:/xunce/out/ppo_frontier"
    assert config.safety.model_dump() == {
        "vehicle_radius_m": 0.4215874761,
        "safety_margin_m": 0.10,
        "min_clearance_m": 0.5215874761,
        "traversability_threshold": 0.50,
        "max_traversable_slope_deg": 30.0,
    }
    assert config.resources.model_dump() == {
        "preflight_d_free_gib_min": 100.0,
        "runtime_d_free_gib_hard_stop_below": 50.0,
        "rss_warning_gib": 16.0,
        "rss_hard_stop_at_or_above_gib": 20.0,
        "vram_warning_gib": 9.0,
        "vram_hard_stop_at_or_above_gib": 10.1,
    }


def test_stage6_config_exact_bytes_parser_matches_path_api_and_fails_closed() -> None:
    module = _config_module()
    payload = CONFIG.read_bytes()

    assert module.parse_stage6_config_bytes(payload) == module.load_stage6_config(CONFIG)
    for invalid in (b"", bytearray(payload), memoryview(payload)):
        with pytest.raises(ValueError, match="snapshot bytes"):
            module.parse_stage6_config_bytes(invalid)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vehicle_radius_m", 0.4215874762),
        ("safety_margin_m", 0.11),
        ("min_clearance_m", 0.5215874762),
        ("traversability_threshold", 0.51),
        ("max_traversable_slope_deg", 29.0),
    ],
)
def test_stage6_config_rejects_each_safety_component_drift(
    field: str,
    value: float,
) -> None:
    module = _config_module()
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["safety"] = {
        "vehicle_radius_m": 0.4215874761,
        "safety_margin_m": 0.10,
        "min_clearance_m": 0.5215874761,
        "traversability_threshold": 0.50,
        "max_traversable_slope_deg": 30.0,
    }
    payload["safety"][field] = value

    with pytest.raises(ValidationError, match="safety contract drifted"):
        module.Stage6Config.model_validate(payload)


def test_safety_contract_is_five_field_frozen_serializable_config_lineage() -> None:
    module = _config_module()
    config = module.load_stage6_config(CONFIG)
    config_sha256 = hashlib.sha256(CONFIG.read_bytes()).hexdigest()

    contract = module.SafetyContract.from_stage6_config(config)
    serialized = contract.to_dict()

    assert tuple(field.name for field in fields(contract)) == (
        "vehicle_radius_m",
        "safety_margin_m",
        "min_clearance_m",
        "traversability_threshold",
        "max_traversable_slope_deg",
    )
    assert serialized == config.safety.model_dump(mode="json")
    assert json.loads(json.dumps(serialized)) == serialized
    assert module.SafetyContract.from_dict(serialized) == contract
    with pytest.raises(FrozenInstanceError):
        contract.traversability_threshold = 0.51

    binding = contract.binding(config_sha256=config_sha256)
    assert binding == {
        "safety_contract": serialized,
        "safety_contract_sha256": hashlib.sha256(
            module._canonical_json_bytes(serialized)
        ).hexdigest(),
        "safety_contract_source": "Stage6Config.safety/v1",
        "safety_contract_config_sha256": config_sha256,
    }
    assert module.SafetyContract.from_binding(
        binding,
        expected_config_sha256=config_sha256,
    ) == contract


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vehicle_radius_m", 0.4215874762),
        ("safety_margin_m", 0.11),
        ("min_clearance_m", 0.5215874762),
        ("traversability_threshold", 0.51),
        ("max_traversable_slope_deg", 29.0),
    ],
)
def test_safety_contract_deserialization_rejects_each_schema_drift(
    field: str,
    value: float,
) -> None:
    module = _config_module()
    payload = module.load_stage6_config(CONFIG).safety.model_dump(mode="json")
    payload[field] = value

    with pytest.raises(ValueError, match="safety contract drifted"):
        module.SafetyContract.from_dict(payload)


def test_safety_contract_binding_rejects_config_sha_and_source_drift() -> None:
    module = _config_module()
    contract = module.SafetyContract.from_stage6_config(
        module.load_stage6_config(CONFIG)
    )
    config_sha256 = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    binding = contract.binding(config_sha256=config_sha256)

    with pytest.raises(ValueError, match="config"):
        module.SafetyContract.from_binding(
            binding,
            expected_config_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="source"):
        module.SafetyContract.from_binding(
            {**binding, "safety_contract_source": "hardcoded_defaults/v1"},
            expected_config_sha256=config_sha256,
        )


def test_stage6_config_rejects_unknown_fields_and_contract_drift() -> None:
    module = _config_module()
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        module.Stage6Config.model_validate(payload)

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["ppo"]["gamma"] = 0.99
    with pytest.raises(ValidationError, match="contract drifted"):
        module.Stage6Config.model_validate(payload)


def test_stage5_gate_is_canonical_hash_chained_and_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _workflow_module()
    monkeypatch.setattr(workflow, "_verify_git_identity", lambda _root: None)
    before = hashlib.sha256(STAGE5_GATE.read_bytes()).hexdigest()

    handle = workflow.verify_frozen_stage5_authority(
        gate_path=STAGE5_GATE,
        repo_root=ROOT,
    )

    assert handle.identity["verified"] is True
    assert handle.identity["authorized_stage"] == (
        "ppo_highres_frontier_stage6_standard_training_eval/v1"
    )
    assert handle.identity["performance_advantage_established"] is False
    assert handle.identity["history_states"] == [
        "machine_passed",
        "awaiting_independent_review",
        "awaiting_human_approval",
        "approved",
        "next_stage",
    ]
    handle.require_current()
    assert hashlib.sha256(STAGE5_GATE.read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.__setitem__("authorized_next_stage", "wrong/v1"), "authorized"),
        (
            lambda value: value["history"][2].__setitem__("previous_record_hash", "0" * 64),
            "hash-chain",
        ),
        (
            lambda value: value["bindings"].__setitem__(
                "performance_advantage_established", True
            ),
            "performance",
        ),
    ],
)
def test_stage5_gate_bytes_fail_closed_on_tamper(mutation, match: str) -> None:
    workflow = _workflow_module()
    value = copy.deepcopy(json.loads(STAGE5_GATE.read_text(encoding="utf-8")))
    mutation(value)
    tampered = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    with pytest.raises(workflow.Stage6WorkflowError, match=match):
        workflow.verify_stage5_gate_bytes(tampered, repo_root=ROOT)


def test_stage5_gate_bytes_reject_noncanonical_json() -> None:
    workflow = _workflow_module()
    value = json.loads(STAGE5_GATE.read_text(encoding="utf-8"))
    noncanonical = json.dumps(value, ensure_ascii=False).encode("utf-8")

    with pytest.raises(workflow.Stage6WorkflowError, match="canonical"):
        workflow.verify_stage5_gate_bytes(noncanonical, repo_root=ROOT)
