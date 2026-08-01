"""Stage 6 Standard catalog、sampler、env 与 spawn spec 合同。"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import importlib
import pickle
from pathlib import Path

import numpy as np
import pytest

from lunar_exploration_ppo.env.env import EnvAction, StepResult


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/ppo_highres_frontier_stage6_v1.json"


@lru_cache(maxsize=1)
def _module_and_catalog():
    module = importlib.import_module("lunar_exploration_ppo.env.standard_training")
    return module, module.build_standard_catalog(verify_hashes=False)


@lru_cache(maxsize=1)
def _safety_lineage():
    config_module = importlib.import_module("lunar_exploration_ppo.configs.stage6")
    config = config_module.load_stage6_config(CONFIG)
    return (
        config_module.SafetyContract.from_stage6_config(config),
        hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    )


def test_standard_catalog_split_isolation_and_sampler_round_trip() -> None:
    module, catalog = _module_and_catalog()

    assert Counter(record.split for record in catalog.records) == {
        "train": 700,
        "validation": 150,
        "test": 150,
        "unseen": 64,
    }
    assert catalog.spatial_audit() == {
        "schema_version": "catalog_spatial_leakage_audit/v1",
        "parent_cross_split_count": 0,
        "child_overlap_pair_count": 0,
    }
    train_parents = {record.parent_roi for record in catalog.records if record.split == "train"}
    assert all(
        record.parent_roi not in train_parents
        for record in catalog.records
        if record.split != "train"
    )

    first = module.StandardScenarioSampler(catalog, split="train", seed=20260716)
    prefix = [first.next_record().scenario_id for _ in range(5)]
    state = first.capture_state()
    expected_next = first.next_record().scenario_id
    restored = module.StandardScenarioSampler(catalog, split="train", seed=20260716)
    restored.restore_state(state)
    assert restored.next_record().scenario_id == expected_next
    assert all(value.startswith("train/") for value in prefix)


def test_real_standard_env_reset_step_has_proxy_labels_and_no_truth_observation() -> None:
    module, catalog = _module_and_catalog()
    safety_contract, config_sha256 = _safety_lineage()
    env = module.StandardTrainingEnv(
        catalog,
        split="train",
        sampler_seed=20260716,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )

    observation = env.reset()

    assert observation.prior_channels.shape == (7, 32, 32)
    assert observation.local_crop.shape[1:] == (96, 96)
    assert observation.frontier_features.shape[0] == 1024
    assert observation.candidate_mask.shape == (1024,)
    assert not hasattr(observation, "truth")
    assert env.scenario_record.split == "train"
    assert env.scenario_bundle.truth.geometry.shape == (256, 256)
    assert env.scenario_bundle.truth.geometry.resolution_m == 0.5
    assert env.scenario_bundle.truth.provenance["source"] == (
        "procedural_lunar_rock_crater_proxy/v1"
    )
    assert env.scenario_bundle.truth.provenance["synthetic_source_kind"] == (
        "synthetic_terrain_obstacle_proxy/v1"
    )
    assert env.scenario_bundle.truth.provenance["physical_obstacle_cells_written"] is False
    assert env.scenario_bundle.prior.provenance["slope_product_pixels_used"] is False
    assert env.scenario_bundle.prior.provenance["physical_slope_derivation"] == (
        "numpy_gradient_metric_spacing_4m/v1"
    )
    assert np.isfinite(observation.local_crop).all()

    assert env.needs_policy is True
    result = env.step(env.select_rule_action(observation))
    assert isinstance(result, StepResult)
    assert result.trainable is True
    assert result.observation.local_crop.shape[1:] == (96, 96)


def test_real_standard_env_uses_standard_step_and_stagnation_limits() -> None:
    module, catalog = _module_and_catalog()
    safety_contract, config_sha256 = _safety_lineage()
    env = module.StandardTrainingEnv(
        catalog,
        split="train",
        sampler_seed=20260716,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )
    invalid_action = EnvAction(candidate_index=1024, target_theta=0.0)

    env.reset()
    env._inner.step_count = 63
    step_64 = env.step(invalid_action)
    assert step_64.reason == "none"
    assert step_64.done is False

    env._inner.step_count = 126
    env._inner.consecutive_no_gain_steps = 0
    step_127 = env.step(invalid_action)
    assert step_127.reason == "none"
    assert step_127.done is False

    step_128 = env.step(invalid_action)
    assert step_128.reason == "failure_done"
    assert step_128.done is True

    env.reset()
    env._inner.consecutive_no_gain_steps = 7
    no_gain_8 = env.step(invalid_action)
    assert no_gain_8.reason == "none"
    assert no_gain_8.done is False

    env._inner.consecutive_no_gain_steps = 14
    no_gain_15 = env.step(invalid_action)
    assert no_gain_15.reason == "none"
    assert no_gain_15.done is False

    no_gain_16 = env.step(invalid_action)
    assert no_gain_16.reason == "stagnation_done"
    assert no_gain_16.done is True


def test_standard_spawn_specs_are_eight_json_bound_production_envs() -> None:
    module, catalog = _module_and_catalog()
    safety_contract, config_sha256 = _safety_lineage()

    specs = module.standard_env_specs(
        catalog,
        split="train",
        sampler_seeds=tuple(range(20260716, 20260724)),
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )

    assert len(specs) == 8
    assert all(spec.factory is module.build_standard_training_env for spec in specs)
    assert all(spec.kwargs["split"] == "train" for spec in specs)
    assert all(spec.kwargs["production"] is True for spec in specs)
    assert all("fixture" not in spec.kwargs and "adapter" not in spec.kwargs for spec in specs)
    assert len({spec.kwargs["sampler_seed"] for spec in specs}) == 8
    assert all(spec.kwargs["safety_contract"] == safety_contract.to_dict() for spec in specs)
    assert all(
        spec.kwargs["safety_contract_config_sha256"] == config_sha256
        and spec.kwargs["safety_contract_source"] == "Stage6Config.safety/v1"
        for spec in specs
    )

    round_tripped = pickle.loads(pickle.dumps(specs[0]))
    env = round_tripped.factory(**round_tripped.kwargs)
    assert env.safety_contract == safety_contract
    assert env.config_sha256 == config_sha256


def test_warm_start_spawn_specs_restore_sampler_only_before_fresh_reset() -> None:
    module, catalog = _module_and_catalog()
    safety_contract, config_sha256 = _safety_lineage()
    sampler_seeds = tuple(range(20260716, 20260724))
    sampler_states: list[dict[str, object]] = []
    expected_scenarios: list[str] = []
    for seed in sampler_seeds:
        sampler = module.StandardScenarioSampler(catalog, split="train", seed=seed)
        sampler.next_record()
        sampler_states.append(sampler.capture_state())
        expected_scenarios.append(sampler.next_record().scenario_id)

    specs = module.standard_env_specs(
        catalog,
        split="train",
        sampler_seeds=sampler_seeds,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
        initial_sampler_states=tuple(sampler_states),
    )

    assert all(
        spec.kwargs["initial_sampler_state"] == sampler_states[index]
        for index, spec in enumerate(specs)
    )
    env = specs[0].factory(**specs[0].kwargs)
    with pytest.raises(RuntimeError, match="not been reset"):
        _ = env.scenario_record
    env.reset()
    assert env.scenario_record.scenario_id == expected_scenarios[0]


def test_standard_env_settings_freeze_explicit_safety_decomposition() -> None:
    module, _ = _module_and_catalog()
    safety_contract, config_sha256 = _safety_lineage()
    settings = module.StandardEnvSettings(
        scenario_key="train/example",
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )

    assert settings.vehicle_radius_m == 0.4215874761
    assert settings.safety_margin_m == 0.10
    assert settings.min_clearance_m == 0.5215874761
    assert settings.vehicle_radius_m + settings.safety_margin_m == pytest.approx(
        settings.min_clearance_m,
        abs=1.0e-12,
    )
    assert settings.traversability_threshold == 0.50
    assert settings.max_traversable_slope_deg == 30.0


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
def test_standard_env_settings_reject_each_safety_component_drift(
    field: str,
    value: float,
) -> None:
    module, _ = _module_and_catalog()
    _, catalog = _module_and_catalog()
    safety_contract, config_sha256 = _safety_lineage()
    binding = safety_contract.binding(config_sha256=config_sha256)
    drifted = dict(binding["safety_contract"])
    drifted[field] = value
    binding["safety_contract"] = drifted
    binding["safety_contract_sha256"] = hashlib.sha256(
        importlib.import_module(
            "lunar_exploration_ppo.configs.stage6"
        )._canonical_json_bytes(drifted)
    ).hexdigest()

    with pytest.raises(ValueError, match="safety contract drifted"):
        module.build_standard_training_env(
            catalog_payload=catalog.to_dict(),
            split="train",
            sampler_seed=20260716,
            production=True,
            **binding,
        )


def test_standard_env_factory_rejects_safety_config_source_drift() -> None:
    module, catalog = _module_and_catalog()
    safety_contract, config_sha256 = _safety_lineage()
    binding = safety_contract.binding(config_sha256=config_sha256)

    with pytest.raises(ValueError, match="source"):
        module.build_standard_training_env(
            catalog_payload=catalog.to_dict(),
            split="train",
            sampler_seed=20260716,
            production=True,
            **{**binding, "safety_contract_source": "hardcoded_defaults/v1"},
        )


def test_scenario_0506_final_proxy_start_is_exact_safe() -> None:
    module, catalog = _module_and_catalog()
    record = next(
        record
        for record in catalog.records
        if record.scenario_id == "train/scenario-0506"
    )
    prior = catalog.read_standard_prior(record)
    preferred = np.argwhere(
        np.asarray(prior.channels[3], dtype=np.float64)[2:30, 2:30] >= 0.75
    )
    assert len(preferred) == 0

    bundle = module.StandardScenarioFactory(catalog).build(record)
    truth = bundle.truth
    finite = (
        np.isfinite(truth.height)
        & np.isfinite(truth.slope_deg)
        & np.isfinite(truth.traversability)
    )
    free = (
        finite
        & ~truth.hard_obstacle
        & (truth.slope_deg <= 30.0)
        & (truth.traversability >= 0.50)
    )
    final_safe = importlib.import_module(
        "lunar_exploration_ppo.env.map_state"
    ).apply_clearance_once(
        free,
        ~free,
        resolution_m=0.5,
        min_clearance_m=0.5215874761,
    )

    start = bundle.start_pose.cell
    assert final_safe[start.y, start.x], (
        record.scenario_id,
        start,
        float(truth.slope_deg[start.y, start.x]),
    )
    assert (start.x, start.y) == (124, 76)
