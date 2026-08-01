"""Stage 5 frozen config、authority、machine artifact 与 runner 合同。"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from lunar_exploration_ppo.configs.stage1 import load_stage1_config
from lunar_exploration_ppo.eval.baselines import ALL_METHODS
from lunar_exploration_ppo.eval.evaluator import (
    METHOD_ACTION_RULES,
    EvaluationSummary,
)
from lunar_exploration_ppo.eval.metrics import build_episode_result, episode_record, summarize_episodes
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/ppo_highres_frontier_stage5_v1.json"
STAGE4_GATE = Path(
    "D:/xunce/out/ppo_frontier/"
    "s4-task5-r2-final-20260714T151959Z/s4/gate.json"
)
STAGE4_CHECKPOINTS = STAGE4_GATE.parent / "checkpoints"
STAGE4_ACTION_FIXTURE_SHA256 = (
    "66dc513c9809174f193bcaa232ab8d21f125a08a167d400f427e9ca5da5876e6"
)
STAGE4_CHECKPOINT_LOAD_AUDIT_SHA256 = (
    "2a29e69c0b28069b7d63b8e74d677d7ecc0ee19e3fd8a58fec44f4f530052810"
)
STAGE4_LATEST_DETERMINISTIC_ACTION = {
    "selected_frontier_index": 11,
    "selected_theta_fp32_hex": "60cb1b3e",
    "log_prob_frontier_fp32_hex": "702767c0",
    "log_prob_theta_fp32_hex": "fa62dfbf",
    "log_prob_total_fp32_hex": "766cabc0",
    "value_fp32_hex": "9063453e",
}


def _config_module():
    return importlib.import_module("lunar_exploration_ppo.configs.stage5")


def _workflow_module():
    return importlib.import_module("lunar_exploration_ppo.workflows.stage5")


def test_stage5_config_freezes_authority_schedule_statistics_and_output() -> None:
    module = _config_module()
    config = module.load_stage5_config(CONFIG)

    assert config.stage_id == "ppo_highres_frontier_stage5_fair_baseline_evaluator/v1"
    assert config.stage4_authority.commit_sha256 == (
        "f94865c3b2984bbffdeefc764a921cae2d9a8089"
    )
    assert config.stage4_authority.commit_tree == (
        "b53b9cca692990acbe1c2aa860c5eeb57821a37b"
    )
    assert config.stage4_authority.gate_sha256 == (
        "a12bbe6bc993add5cd8329bb9e172602e1e2e853a60c374343881e8263c0603e"
    )
    assert config.methods == ALL_METHODS
    assert config.eval_episode_count == 16
    assert len(config.scenario_schedule) == 16
    assert len({row.evaluation_seed for row in config.scenario_schedule}) == 16
    assert all(row.scenario_key == "smoke-v1" for row in config.scenario_schedule)
    assert config.bootstrap_resamples == 2000
    assert config.bootstrap_seed == 20260715
    assert config.max_steps == 64
    assert config.success_threshold == 0.99
    assert config.zero_distance_policy == "zero_when_no_travel/v1"
    assert config.terminal_curve_policy == "terminal_carry_forward_through_max_steps/v1"
    assert config.claim_boundary == (
        "fair_baseline_evaluator_system_closure_no_task_advantage/v1"
    )
    assert config.device == "cuda"
    assert config.allow_cpu_fallback is False
    assert config.output_root == "D:/xunce/out/ppo_frontier"

    payload = config.model_dump(mode="json")
    payload["bootstrap_resamples"] = 1999
    with pytest.raises(ValidationError, match="contract drifted"):
        module.Stage5Config.model_validate(payload)


def _source_identity() -> dict[str, object]:
    return {
        "schema_version": "stage5_reviewed_source_set/v1",
        "source_set_sha256": "a" * 64,
        "paths": [{"path": "src/example.py", "sha256": "b" * 64, "size_bytes": 1}],
    }


def _environment_identity() -> dict[str, object]:
    return {
        "schema_version": "ppo_highres_frontier_stage5_environment/v1",
        "cuda_available": True,
        "compute_dtype": "float32",
        "amp_enabled": False,
        "cuda_device_name": "fixture CUDA",
    }


def _git_identity(workflow) -> dict[str, object]:
    return {
        "schema_version": "ppo_highres_frontier_stage5_prospective_git_tree/v1",
        "head_commit": "f94865c3b2984bbffdeefc764a921cae2d9a8089",
        "base_commit": "f94865c3b2984bbffdeefc764a921cae2d9a8089",
        "prospective_git_tree": "c" * 40,
        "changed_paths": list(workflow.STAGE5_CHANGED_PATHS),
        "changed_path_set_sha256": "d" * 64,
        "real_index_empty": True,
    }


def _authority(config) -> dict[str, object]:
    return {
        "schema_version": "stage5_stage4_authority_audit/v1",
        "verified": True,
        "authorized_stage": config.stage_id,
        "run_id": config.stage4_authority.run_id,
        "commit_sha256": config.stage4_authority.commit_sha256,
        "commit_tree": config.stage4_authority.commit_tree,
        "gate_path": config.stage4_authority.gate_path,
        "gate_sha256": config.stage4_authority.gate_sha256,
        "config_sha256": config.stage4_authority.config_sha256,
        "source_set_sha256": config.stage4_authority.source_set_sha256,
        "environment_sha256": config.stage4_authority.environment_sha256,
        "manifest_sha256": config.stage4_authority.manifest_sha256,
        "checkpoint_root": config.stage4_authority.checkpoint_root,
        "checkpoint_sha256": config.stage4_authority.checkpoint_sha256,
        "checkpoint_manifest_sha256": config.stage4_authority.checkpoint_manifest_sha256,
        "policy_state_sha256": config.stage4_authority.policy_state_sha256,
        "update_step": 3,
    }


def _checkpoint_audit(config) -> dict[str, object]:
    return {
        "schema_version": "stage5_frozen_checkpoint_audit/v1",
        "checkpoint_root": config.stage4_authority.checkpoint_root,
        "checkpoint_directory": "update-00000003",
        "checkpoint_sha256": config.stage4_authority.checkpoint_sha256,
        "checkpoint_manifest_sha256": config.stage4_authority.checkpoint_manifest_sha256,
        "policy_state_sha256": config.stage4_authority.policy_state_sha256,
        "update_step": 3,
        "device": "cuda",
        "read_only": True,
        "deterministic_action_replay": {
            "schema_version": "stage5_checkpoint_action_replay/v1",
            "fixture_sha256": STAGE4_ACTION_FIXTURE_SHA256,
            "checkpoint_load_audit_sha256": STAGE4_CHECKPOINT_LOAD_AUDIT_SHA256,
            "expected_action": STAGE4_LATEST_DETERMINISTIC_ACTION,
            "actual_action": STAGE4_LATEST_DETERMINISTIC_ACTION,
            "bit_exact": True,
        },
    }


def _fixture_leakage_audit() -> dict[str, object]:
    return {
        "schema_version": "stage5_runtime_leakage_audit/v1",
        "mutation_source": "paired_unobserved_truth_and_coverable_mask/v1",
        "observed_truth_cells_byte_equal": True,
        "unobserved_truth_mutated": True,
        "coverable_mask_mutated": True,
        "all_observation_arrays_byte_equal": True,
        "candidate_set_byte_equal": True,
        "baseline_action_replay": {
            method: {"byte_equal": True} for method in ALL_METHODS[:-1]
        },
        "all_baseline_decisions_invariant": True,
        "baseline_selection_parameters": ["method", "observation", "rng"],
        "evaluator_selection_parameters": ["self", "method", "observation", "rng"],
        "truth_or_coverable_parameter_present": False,
        "selection_source_forbidden_token_present": False,
        "python_internal_inaccessibility_claimed": False,
    }


def test_stage5_runtime_leakage_audit_mutates_truth_and_denominator_only() -> None:
    workflow = _workflow_module()
    stage1_config = load_stage1_config(ROOT / "configs/ppo_highres_frontier_smoke_v1.json")

    audit = workflow.stage5_runtime_leakage_audit(stage1_config)

    assert audit["observed_truth_cells_byte_equal"] is True
    assert audit["unobserved_truth_mutated"] is True
    assert audit["coverable_mask_mutated"] is True
    assert audit["all_observation_arrays_byte_equal"] is True
    assert audit["candidate_set_byte_equal"] is True
    assert audit["all_baseline_decisions_invariant"] is True
    assert set(audit["baseline_action_replay"]) == set(ALL_METHODS[:-1])
    assert all(
        row["byte_equal"] is True
        for row in audit["baseline_action_replay"].values()
    )
    assert audit["truth_or_coverable_parameter_present"] is False
    assert audit["selection_source_forbidden_token_present"] is False
    assert audit["python_internal_inaccessibility_claimed"] is False
    assert audit["baseline_coverable_mask_sha256"] != audit[
        "mutated_coverable_mask_sha256"
    ]


def test_stage5_leakage_validator_fails_closed_on_decision_invariance_false() -> None:
    workflow = _workflow_module()
    tampered = _fixture_leakage_audit()
    tampered["all_baseline_decisions_invariant"] = False

    with pytest.raises(workflow.Stage5WorkflowError, match="leakage audit"):
        workflow._validated_leakage_audit(tampered)


def _fixture_summaries(config) -> tuple[EvaluationSummary, ...]:
    summaries = []
    episode_contracts = [
        {
            "scenario": row.model_dump(mode="json"),
            "scenario_id": "fixture",
            "scenario_hash": "e" * 64,
            "environment_class": "lunar_exploration_ppo.env.env.LunarExplorationEnv",
            "environment_config_sha256": "1" * 64,
            "sensor_model_id": "path-tangent-plus-endpoint-theta-fov-90-range-20m-los/v1",
            "initial_scan": "free_reset_scan_20m_90deg/v1",
            "frontier_generator_class": "lunar_exploration_ppo.env.frontier.FrontierGenerator",
            "candidate_feature_schema": "frontier_features_22/v1",
            "reachability_prefilter": "observed_safe_connected_component/v1",
            "planner_class": "fixture.Planner",
            "planner_validation_source": "stage1_path_planner_adapter/v1",
            "path_execution_sensor_updates": "shared_environment_step/v1",
            "coverage_denominator_source": "coverable_mask_exact",
            "coverable_mask_exact": True,
            "coverable_mask_hash": "2" * 64,
            "coverable_mask_algorithm_id": "exact_reachable_safe_pose_range_los/v1",
            "coverable_mask_precompute_scope": "scenario_reset/v1",
            "coverable_cell_count": 1,
            "max_steps": 64,
            "stagnation_no_gain_steps": 8,
            "success_threshold": 0.99,
            "safety_constants": {"fixture": True},
        }
        for row in config.scenario_schedule
    ]
    shared_contract = {
        "schema_version": "stage5_shared_environment_contract/v1",
        "scale_profile": "Smoke v1",
        "max_steps": 64,
        "success_threshold": 0.99,
        "scenario_schedule": [row.model_dump(mode="json") for row in config.scenario_schedule],
        "episodes": episode_contracts,
    }
    contract_sha256 = hashlib.sha256(
        ArtifactStore.canonical_json_bytes(shared_contract)
    ).hexdigest()
    for method in ALL_METHODS:
        episodes = []
        episode_audits = []
        for index, schedule in enumerate(config.scenario_schedule):
            success = index % 2 == 0
            if success:
                coverage = (0.1, 0.6, 0.99, *([0.99] * 62))
            else:
                coverage = (0.1, 0.2, 0.3, *([0.3] * 62))
            path = (0.0, 1.0, 2.0, *([2.0] * 62))
            episodes.append(
                build_episode_result(
                    method=method,
                    scale_profile="Smoke v1",
                    scenario_key=schedule.scenario_key,
                    scenario_seed=schedule.scenario_seed,
                    terrain_seed=schedule.terrain_seed,
                    start_pose_seed=schedule.start_pose_seed,
                    evaluation_seed=schedule.evaluation_seed,
                    coverage_curve=coverage,
                    cumulative_path_length_curve=path,
                    steps_executed=2,
                    invalid_action_count=0,
                    planner_failure_count=0,
                    safety_violation_count=0,
                    termination_reason="success_done" if success else "stagnation_done",
                    max_steps=64,
                    success_threshold=0.99,
                    zero_distance_policy="zero_when_no_travel/v1",
                )
            )
            episode_audits.append(
                {
                    "scenario": schedule.model_dump(mode="json"),
                    "scenario_id": "fixture",
                    "scenario_hash": "e" * 64,
                    "initial_coverage": 0.1,
                    "steps_executed": 2,
                    "selected_actions": [
                        {
                            "candidate_mask_valid": True,
                            "observed_safe": True,
                            "reachable_from_current_pose": True,
                        }
                    ],
                    "termination_reason": "success_done" if success else "stagnation_done",
                }
            )
        metrics, bootstrap = summarize_episodes(
            episodes,
            bootstrap_resamples=2000,
            bootstrap_seed=20260715,
        )
        summaries.append(
            EvaluationSummary(
                method=method,
                scale_profile="Smoke v1",
                episodes=tuple(episodes),
                metrics=metrics,
                bootstrap_audit=bootstrap,
                fairness_audit={
                    "schema_version": "stage5_method_fairness_audit/v2",
                    "method": method,
                    "shared_environment_contract_identical": True,
                    "method_specific_action_rule_only_difference": True,
                    **METHOD_ACTION_RULES[method],
                    "decision_input_fields": ["observed_only"],
                    "forbidden_decision_inputs": ["hidden_truth", "coverable_mask"],
                    "coverage_denominator_access": "environment_metric_layer_only/v1",
                    "ppo_eval_policy_mode": "deterministic_argmax_frontier_mean_theta/v1",
                    "shared_contract": shared_contract,
                    "shared_contract_sha256": contract_sha256,
                    "selected_action_count": 16,
                    "invalid_selected_action_count": 0,
                    "all_selected_actions_reachable_observed_safe": True,
                    "all_baseline_thetas_recommended": (
                        True if method in ALL_METHODS[:-1] else None
                    ),
                    "episode_audits": episode_audits,
                },
            )
        )
    return tuple(summaries)


def _fixture_runtime(config, workflow):
    return workflow._build_stage5_runtime_binding(
        config=config,
        run_id="s5-machine-fixture",
        source_identity=_source_identity(),
        environment_identity=_environment_identity(),
        git_identity=_git_identity(workflow),
        stage4_authority=_authority(config),
        stage1_config_binding={
            "path": str(ROOT / config.stage1_config_path),
            "sha256": config.stage1_config_sha256,
        },
        checkpoint_audit=_checkpoint_audit(config),
    )


def test_stage5_machine_payload_is_byte_deterministic_manifest_bound_and_stops_review() -> None:
    config = _config_module().load_stage5_config(CONFIG)
    workflow = _workflow_module()
    runtime = _fixture_runtime(config, workflow)
    summaries = _fixture_summaries(config)

    first = workflow._build_stage5_machine_payload(
        config=config,
        runtime=runtime,
        summaries=summaries,
        leakage_audit=_fixture_leakage_audit(),
    )
    second = workflow._build_stage5_machine_payload(
        config=config,
        runtime=runtime,
        summaries=tuple(reversed(summaries)),
        leakage_audit=_fixture_leakage_audit(),
    )

    assert first.artifacts == second.artifacts
    assert first.manifest == second.manifest
    assert set(first.artifacts) == set(workflow.STAGE5_MANIFEST_BOUND_ARTIFACTS)
    assert set(workflow.STAGE5_ROOT_ARTIFACTS) == {
        *workflow.STAGE5_MANIFEST_BOUND_ARTIFACTS,
        "manifest.json",
    }
    assert len(first.artifacts["baseline_episode_traces.jsonl"].splitlines()) == 80
    assert first.summary["state"] == "machine_passed"
    assert first.summary["method_episode_count"] == 80
    assert len(first.summary["acceptance"]["items"]) == 22
    assert all(first.summary["acceptance"]["items"].values())
    routing = json.loads(first.artifacts["routing.json"])
    assert routing["route"] == "awaiting_independent_review"
    assert routing["next_stage_entered"] is False
    states = [
        json.loads(line)["state"]
        for line in first.artifacts["phase-state.jsonl"].splitlines()
    ]
    assert states == ["machine_passed", "awaiting_independent_review"]
    progress = [
        json.loads(line) for line in first.artifacts["metrics.jsonl"].splitlines()
    ]
    assert len(progress) == 80
    assert {row["step_count"] for row in progress} == {2}
    assert {row["max_steps"] for row in progress} == {64}
    assert not {"review.json", "approval.json", "gate.json"} & set(
        workflow.STAGE5_ROOT_ARTIFACTS
    )
    manifest_paths = [entry["path"] for entry in first.manifest["artifacts"]]
    assert manifest_paths == sorted(workflow.STAGE5_MANIFEST_BOUND_ARTIFACTS)
    for relative, payload in first.artifacts.items():
        record = next(item for item in first.manifest["artifacts"] if item["path"] == relative)
        assert record["sha256"] == hashlib.sha256(payload).hexdigest()
        assert record["size_bytes"] == len(payload)


def test_stage5_acceptance_probe_audit_is_manifest_bound_recomputed_and_drives_six_items() -> None:
    config = _config_module().load_stage5_config(CONFIG)
    workflow = _workflow_module()
    summaries = _fixture_summaries(config)

    payload = workflow._build_stage5_machine_payload(
        config=config,
        runtime=_fixture_runtime(config, workflow),
        summaries=summaries,
        leakage_audit=_fixture_leakage_audit(),
    )

    assert "acceptance_probe_audit.json" in workflow.STAGE5_MANIFEST_BOUND_ARTIFACTS
    audit = json.loads(payload.artifacts["acceptance_probe_audit.json"])
    assert audit["schema_version"] == "stage5_acceptance_probe_audit/v1"
    assert audit["probe_policy"] == "manifest_bound_current_code_recomputation/v1"
    assert audit["canonical_inputs_sha256"] == hashlib.sha256(
        ArtifactStore.canonical_json_bytes(audit["canonical_inputs"])
    ).hexdigest()
    assert audit["passed"] is True

    probes = audit["probes"]
    bindings = {
        "09_empty_candidate_uses_no_candidate_done": "empty_candidate",
        "10_deterministic_baselines_replayable": "deterministic_baselines",
        "11_random_baseline_fixed_seed_replayable": "random_baseline",
        "13_ties_lowest_index": "lowest_index_ties",
        "20_comparison_csv_written_deterministically": "comparison_csv",
        "21_coverage_curves_csv_written_deterministically": "coverage_curves_csv",
    }
    assert audit["acceptance_item_bindings"] == bindings
    for item, probe in bindings.items():
        assert payload.summary["acceptance"]["items"][item] is probes[probe]["passed"]

    empty = probes["empty_candidate"]
    assert set(empty["selector_results"]) == set(ALL_METHODS)
    assert all(row["action"] is None for row in empty["selector_results"].values())
    assert empty["environment_terminal_result"]["reason"] == "no_candidate_done"
    assert empty["fabricated_action"] is False

    deterministic = probes["deterministic_baselines"]
    assert set(deterministic["methods"]) == set(ALL_METHODS[1:4])
    assert all(row["byte_identical"] is True for row in deterministic["methods"].values())

    random_probe = probes["random_baseline"]
    assert len(random_probe["recorded_sequence"]) == random_probe["sequence_length"]
    assert random_probe["first_sequence_sha256"] == random_probe["second_sequence_sha256"]

    ties = probes["lowest_index_ties"]
    assert set(ties["methods"]) == {*ALL_METHODS[1:4], "ppo_policy"}
    assert all(row["selected_index"] == 1 for row in ties["methods"].values())

    comparison = probes["comparison_csv"]
    curves = probes["coverage_curves_csv"]
    assert comparison["published_sha256"] == hashlib.sha256(
        payload.artifacts["baseline_comparison_table.csv"]
    ).hexdigest()
    assert curves["published_sha256"] == hashlib.sha256(
        payload.artifacts["baseline_coverage_curves.csv"]
    ).hexdigest()
    assert comparison["byte_identical"] is True
    assert curves["byte_identical"] is True

    assert workflow.verify_stage5_acceptance_probe_audit(
        audit,
        summaries=summaries,
        comparison_csv=payload.artifacts["baseline_comparison_table.csv"],
        coverage_curves_csv=payload.artifacts["baseline_coverage_curves.csv"],
    ) == audit
    manifest_entry = next(
        row
        for row in payload.manifest["artifacts"]
        if row["path"] == "acceptance_probe_audit.json"
    )
    assert manifest_entry["sha256"] == hashlib.sha256(
        payload.artifacts["acceptance_probe_audit.json"]
    ).hexdigest()
    assert payload.summary["acceptance"]["probe_artifact"] == (
        "acceptance_probe_audit.json"
    )
    assert payload.summary["acceptance"]["probe_sha256"] == manifest_entry["sha256"]


@pytest.mark.parametrize(
    "tamper",
    ("canonical_input_hash", "random_sequence", "published_csv_hash"),
)
def test_stage5_acceptance_probe_verifier_rejects_tamper_fail_closed(
    tamper: str,
) -> None:
    config = _config_module().load_stage5_config(CONFIG)
    workflow = _workflow_module()
    summaries = _fixture_summaries(config)
    payload = workflow._build_stage5_machine_payload(
        config=config,
        runtime=_fixture_runtime(config, workflow),
        summaries=summaries,
        leakage_audit=_fixture_leakage_audit(),
    )
    audit = copy.deepcopy(json.loads(payload.artifacts["acceptance_probe_audit.json"]))
    if tamper == "canonical_input_hash":
        audit["canonical_inputs_sha256"] = "0" * 64
    elif tamper == "random_sequence":
        audit["probes"]["random_baseline"]["recorded_sequence"][0] = 99
    else:
        audit["probes"]["comparison_csv"]["published_sha256"] = "0" * 64

    with pytest.raises(workflow.Stage5WorkflowError, match="acceptance probe"):
        workflow.verify_stage5_acceptance_probe_audit(
            audit,
            summaries=summaries,
            comparison_csv=payload.artifacts["baseline_comparison_table.csv"],
            coverage_curves_csv=payload.artifacts["baseline_coverage_curves.csv"],
        )
    assert "verify_stage5_acceptance_probe_audit(" in inspect.getsource(
        workflow.verify_stage5_machine_run
    )


def test_stage5_machine_payload_rejects_missing_exact_denominator_binding() -> None:
    config = _config_module().load_stage5_config(CONFIG)
    workflow = _workflow_module()
    broken_summaries = []
    for summary in _fixture_summaries(config):
        fairness = dict(summary.fairness_audit)
        contract = dict(fairness["shared_contract"])
        contracts = [dict(row) for row in contract["episodes"]]
        for row in contracts:
            del row["coverable_mask_hash"]
        contract["episodes"] = contracts
        fairness["shared_contract"] = contract
        fairness["shared_contract_sha256"] = hashlib.sha256(
            ArtifactStore.canonical_json_bytes(contract)
        ).hexdigest()
        broken_summaries.append(replace(summary, fairness_audit=fairness))

    with pytest.raises(workflow.Stage5WorkflowError, match="environment contract"):
        workflow._build_stage5_machine_payload(
            config=config,
            runtime=_fixture_runtime(config, workflow),
            summaries=broken_summaries,
            leakage_audit=_fixture_leakage_audit(),
        )


def test_stage5_workflow_result_carries_machine_routing_for_thin_cli() -> None:
    config = _config_module().load_stage5_config(CONFIG)
    workflow = _workflow_module()
    payload = workflow._build_stage5_machine_payload(
        config=config,
        runtime=_fixture_runtime(config, workflow),
        summaries=_fixture_summaries(config),
        leakage_audit=_fixture_leakage_audit(),
    )

    assert payload.routing["state"] == "machine_passed"
    assert payload.routing["route"] == "awaiting_independent_review"
    result = workflow.Stage5WorkflowResult(
        run_id="s5-cli-contract-test",
        stage_root=ROOT,
        summary=payload.summary,
        routing=payload.routing,
        manifest=payload.manifest,
    )
    assert result.routing == payload.routing
    assert "routing=payload.routing" in inspect.getsource(workflow.run_stage5_workflow)


def test_stage5_payload_writer_appends_jsonl_and_replays_exact_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config_module().load_stage5_config(CONFIG)
    workflow = _workflow_module()
    summaries = _fixture_summaries(config)
    payload = workflow._build_stage5_machine_payload(
        config=config,
        runtime=_fixture_runtime(config, workflow),
        summaries=summaries,
        leakage_audit=_fixture_leakage_audit(),
    )
    stage = tmp_path / "run" / "s5"
    stage.mkdir(parents=True)
    store = ArtifactStore(stage)
    appended: list[str] = []
    real_append = ArtifactStore.append_jsonl

    def recording_append(self, relative_path, value):
        appended.append(Path(relative_path).as_posix())
        return real_append(self, relative_path, value)

    monkeypatch.setattr(ArtifactStore, "append_jsonl", recording_append)

    workflow._write_stage5_machine_payload(store, payload)

    expected_jsonl_counts = {
        relative: len(payload.artifacts[relative].splitlines())
        for relative in (
            "metrics.jsonl",
            "phase-state.jsonl",
            "baseline_episode_traces.jsonl",
        )
    }
    assert {
        relative: appended.count(relative) for relative in expected_jsonl_counts
    } == expected_jsonl_counts
    for relative, body in payload.artifacts.items():
        assert (stage / relative).read_bytes() == body
    assert workflow._read_episode_traces(
        (stage / "baseline_episode_traces.jsonl").read_bytes()
    ) == tuple(episode for summary in summaries for episode in summary.episodes)
    assert (stage / "manifest.json").read_bytes() == ArtifactStore.canonical_json_bytes(
        payload.manifest
    )


def test_stage5_manifest_graph_rejects_tamper_and_extra_authority_artifact(
    tmp_path: Path,
) -> None:
    config = _config_module().load_stage5_config(CONFIG)
    workflow = _workflow_module()
    payload = workflow._build_stage5_machine_payload(
        config=config,
        runtime=_fixture_runtime(config, workflow),
        summaries=_fixture_summaries(config),
        leakage_audit=_fixture_leakage_audit(),
    )
    stage = tmp_path / "run" / "s5"
    stage.mkdir(parents=True)
    store = ArtifactStore(stage)
    for relative, body in payload.artifacts.items():
        store.write_bytes(relative, body)
    store.write_json("manifest.json", payload.manifest)

    workflow.verify_stage5_manifest_graph(stage).require_current()
    store.write_bytes("summary.json", b"tampered\n")
    with pytest.raises(workflow.Stage5WorkflowError, match="manifest"):
        workflow.verify_stage5_manifest_graph(stage)

    store.write_bytes("summary.json", payload.artifacts["summary.json"])
    store.write_json("approval.json", {"forbidden": True})
    with pytest.raises(workflow.Stage5WorkflowError, match="artifact set"):
        workflow.verify_stage5_manifest_graph(stage)


def test_stage5_ast_import_boundaries_runner_and_exports_are_exact() -> None:
    workflow = _workflow_module()
    del workflow
    forbidden_import_roots = {"model_explorer", "scripts"}
    for path in (ROOT / "src/lunar_exploration_ppo").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name.split(".")[0] for alias in node.names}
                assert not imported & forbidden_import_roots
                if "path_planner" in imported:
                    assert path.name in {
                        "path_planner_adapter.py",
                        "stage6_planning_child_source_repair.py",
                    }
            if isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                assert root not in forbidden_import_roots
                if root == "path_planner":
                    assert path.name in {
                        "path_planner_adapter.py",
                        "stage6_planning_child_source_repair.py",
                    }
        assert "sys.path" not in path.read_text(encoding="utf-8")

    runner = ROOT / "scripts/run_ppo_stage5_baselines.py"
    runner_source = runner.read_text(encoding="utf-8")
    assert "run_stage5_workflow" in runner_source
    assert 'if __name__ == "__main__"' in runner_source
    assert all(flag in runner_source for flag in ("--config", "--run-id", "--stage4-gate", "--checkpoint-root", "--output-root"))
    assert "--force" not in runner_source
    assert "LunarExplorationEnv" not in runner_source
    assert "CrossAttentionFrontierPolicy" not in runner_source

    eval_package = importlib.import_module("lunar_exploration_ppo.eval")
    assert eval_package.Evaluator.__name__ == "Evaluator"
    assert eval_package.EvaluationSummary.__name__ == "EvaluationSummary"


def test_stage5_workflow_uses_real_evaluator_and_cannot_issue_authority() -> None:
    workflow = _workflow_module()
    source = inspect.getsource(workflow)
    assert "Evaluator(" in source
    assert ".evaluate(" in source
    assert "load_frozen_stage4_policy(" in source
    assert "LunarExplorationEnv(" in source
    assert "ArtifactStore(" in source
    assert "Mock" not in source
    assert "monkeypatch" not in source
    lowered = source.lower()
    assert all(
        token not in lowered
        for token in (
            "issue_approval",
            "record_approval",
            "create_gate",
            "write_gate",
            "record_review",
            "publish_checkpoint",
            "start_canary",
            "connect_executor",
            "--force",
        )
    )

    doc = (ROOT / "docs/ppo-highres-frontier-stage5.md").read_text(encoding="utf-8")
    assert "fair_baseline_evaluator_system_closure_no_task_advantage/v1" in doc
    assert "不建立 PPO 性能优势" in doc
    assert all(f"`{method}`" in doc for method in ALL_METHODS)
    assert "no_candidate_done" in doc
    assert "no_reachable_candidate" not in doc
    assert "evaluator/" not in doc
