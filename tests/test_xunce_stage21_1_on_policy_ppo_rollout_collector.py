import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
MODEL_EXPLORER_SRC = str(REPO_ROOT / "model-explorer" / "src")
if MODEL_EXPLORER_SRC not in sys.path:
    sys.path.insert(0, MODEL_EXPLORER_SRC)


def test_stage21_1_candidate_reachability_gate_blocks_grid_only_candidate() -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    action_mask = runner._candidate_reachability_action_mask(
        (True, True),
        (True, False),
        enabled=True,
    )

    assert action_mask == (True, False)
    assert runner._no_sampling_candidate_reason(
        grid_action_mask=(True, True),
        action_mask=(False, False),
        hard_risk_clean_mask=(True, True),
        hybrid_reachable_mask=(False, False),
    ) == runner.NO_HYBRID_POSE_REACHABLE_ACTION_MASK_REASON


def test_stage21_1_candidate_reachability_gate_preserves_mask_length_when_provenance_short() -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    action_mask = runner._candidate_reachability_action_mask(
        (True, True, True),
        (True,),
        enabled=True,
    )

    assert action_mask == (True, False, False)


def test_stage21_1_candidate_reachability_provenance_mask_requires_hybrid_pose_source() -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    valid = _reachability_provenance()
    metadata = {
        "path_cost_sources": [runner.HYBRID_ASTAR_PATH_COST_SOURCE, runner.HYBRID_ASTAR_PATH_COST_SOURCE],
        "hybrid_astar_reachable_flags": [True, True],
        "hybrid_astar_path_costs": [3.0, 4.0],
        "hybrid_astar_pose_path_hashes": ["hash-a", "hash-b"],
        "hybrid_astar_trajectory_kinds": ["hybrid_astar_pose_path", "hybrid_astar_pose_path"],
        "candidate_reachability_provenances": [
            valid,
            {**valid, "source": "grid_astar_reachability/v1"},
        ],
    }

    assert runner._candidate_reachability_provenance_mask(metadata, candidate_count=2) == (True, False)


def test_stage21_1_replaces_xunce_batch_action_mask_on_same_device() -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    batch = {"action_mask": torch.tensor([[True, True]], dtype=torch.bool)}

    runner._replace_xunce_batch_action_mask(batch, (True, False))

    assert batch["action_mask"].tolist() == [[True, False]]
    assert batch["action_mask"].device.type == "cpu"


def test_stage21_1_replaces_observation_action_mask_after_reachability_gate() -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    observation = {"action_mask": [True, True], "candidate_cells": [[0, 0], [1, 0]]}

    runner._replace_observation_action_mask(observation, (True, False))

    assert observation["action_mask"] == [True, False]
    assert observation["candidate_cells"] == [[0, 0], [1, 0]]


def test_stage21_1_finalize_pending_attributes_next_action_mask_dead_end_to_previous_transition() -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    pending = {
        "transition_id": "s1:step-0:sample-0",
        "scenario_id": "s1",
        "done": False,
        "trainable": True,
        "info": {"action_mask": [True], "terminal_reason": None},
    }
    transitions: list[dict] = []
    trainable_batch: list[dict] = []

    runner._finalize_pending(
        pending,
        transitions,
        trainable_batch,
        done=True,
        next_observation={"action_mask": [False, False]},
        next_xunce_batch={"action_mask": {"shape": [1, 2], "dtype": "bool", "values": [[False, False]]}},
        terminal_reason=runner.NO_HYBRID_POSE_REACHABLE_ACTION_MASK_REASON,
        next_action_mask=(False, False),
        next_grid_action_mask=(True, True),
    )

    row = trainable_batch[0]
    assert row["done"] is True
    assert row["info"]["terminal_reason"] == runner.NO_HYBRID_POSE_REACHABLE_ACTION_MASK_REASON
    assert row["info"]["next_action_mask_true_count"] == 0
    assert row["info"]["next_action_mask_zero"] is True
    assert row["info"]["next_grid_action_mask_true_count"] == 2
    assert row["info"]["dead_end_attribution_source"] == "next_state_action_mask_all_false/v1"


def test_stage21_1_finalize_pending_does_not_mark_dead_end_when_next_action_mask_has_candidate() -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    pending = {
        "transition_id": "s1:step-0:sample-0",
        "scenario_id": "s1",
        "done": False,
        "trainable": True,
        "info": {"action_mask": [True]},
    }
    transitions: list[dict] = []
    trainable_batch: list[dict] = []

    runner._finalize_pending(
        pending,
        transitions,
        trainable_batch,
        done=True,
        next_observation={"action_mask": [True, False]},
        next_xunce_batch={"action_mask": {"shape": [1, 2], "dtype": "bool", "values": [[True, False]]}},
        terminal_reason="no_hard_risk_clean_candidate",
        next_action_mask=(True, False),
        next_grid_action_mask=(True, True),
    )

    info = trainable_batch[0]["info"]
    assert info["next_action_mask_true_count"] == 1
    assert info["next_action_mask_zero"] is False
    assert "dead_end_attribution_source" not in info


def test_stage21_1_continuous_theta_probe_sets_prefilter_grid_mask_and_cap() -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    probes, proposal_sets = runner._continuous_theta_probe_candidate_sets(
        [
            {"cell": [1, 0], "candidate_theta_deg": 10.0},
            {"cell": [2, 0], "candidate_theta_deg": 20.0},
        ],
        current_theta_deg=0.0,
        theta_step_deg=45.0,
        candidate_set_hash_value="candidate-set",
        action_mask=[True, False],
        max_proposals_per_candidate=1,
    )

    assert proposal_sets == [[10.0], []]
    assert len(probes) == 1
    assert probes[0]["candidate_viewpoint"] == [1, 0, 10.0]


def test_stage21_1_bearing_sweep_theta_policy_is_stable_and_capped() -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    probes, proposal_sets = runner._continuous_theta_probe_candidate_sets(
        [{"cell": [2, 1], "candidate_theta_deg": 0.0, "candidate_viewpoint": [2, 1, 90.0]}],
        current_cell=(0, 0),
        current_theta_deg=45.0,
        theta_step_deg=45.0,
        candidate_set_hash_value="candidate-set",
        max_proposals_per_candidate=5,
        proposal_policy=runner.CANDIDATE_REACHABILITY_THETA_PROPOSAL_POLICY_REPAIR,
    )

    assert proposal_sets == [[0.0, 90.0, 45.0, 26.56505117707799, 71.56505117707799]]
    assert [probe["candidate_viewpoint"][2] for probe in probes] == [
        0.0,
        90.0,
        45.0,
        26.56505117707799,
        71.56505117707799,
    ]


def test_stage21_1_rejection_evidence_includes_grid_allowed_probe_records() -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    evidence = runner._candidate_reachability_rejection_evidence(
        candidates=[
            {"cell": [1, 0], "candidate_theta_deg": 0.0},
            {"cell": [2, 0], "candidate_theta_deg": 90.0},
        ],
        grid_action_mask=[True, False],
        metadata={
            "candidate_reachability_theta_proposal_policy": runner.CANDIDATE_REACHABILITY_THETA_PROPOSAL_POLICY_REPAIR,
            "candidate_reachability_max_theta_proposals_per_candidate": 5,
            "candidate_reachability_probe_config_hash": "probe-config-hash",
            "hybrid_astar_max_iterations": 100,
            "hybrid_astar_theta_proposals_deg_by_candidate": [[0.0, 45.0], [90.0]],
            "hybrid_astar_reachable_theta_degs_by_candidate": [[], [90.0]],
            "hybrid_astar_theta_probe_records_by_candidate": [
                [
                    {
                        "theta_deg": 0.0,
                        "reachable": False,
                        "failure_reason": "search_exhausted",
                        "planner_config_hash": "planner-hash",
                    }
                ],
                [{"theta_deg": 90.0, "reachable": True}],
            ],
        },
        action_mask=[False, False],
        hard_risk_clean_mask=[True, True],
        sampling_mask=[False, False],
        candidate_reachability_provenance_mask=[False, True],
    )

    assert evidence["candidate_reachability_theta_proposal_policy"] == "candidate_current_bearing_sweep/v1"
    assert evidence["candidate_reachability_max_theta_proposals_per_candidate"] == 5
    assert evidence["candidate_reachability_probe_config_hash"] == "probe-config-hash"
    assert evidence["hybrid_astar_max_iterations"] == 100
    assert evidence["candidate_reachability_probe_evidence_schema"] == (
        "xunce-stage21-1-candidate-reachability-probe-evidence/v1"
    )
    assert len(evidence["candidate_reachability_probe_evidence"]) == 1
    row = evidence["candidate_reachability_probe_evidence"][0]
    assert row["candidate_index"] == 0
    assert row["candidate_cell"] == [1, 0]
    assert row["grid_action_allowed"] is True
    assert row["action_mask_allowed"] is False
    assert row["hard_risk_clean"] is True
    assert row["sampling_allowed"] is False
    assert row["provenance_pass"] is False
    assert row["theta_proposals_deg"] == [0.0, 45.0]
    assert row["theta_probe_records"][0]["failure_reason"] == "search_exhausted"


def test_stage21_1_writes_passed_collector_artifacts(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    config = _write_config(tmp_path)
    transition = _transition()

    def fake_collect(**_kwargs):
        return runner.CollectionResult(
            episodes=[
                {
                    "schema_version": "xunce-stage21-1-ppo-rollout-episode/v1",
                    "scenario_id": "s1",
                    "trainable_transition_count": 1,
                    "hard_risk_violation_count": 0,
                    "reason_codes": [],
                }
            ],
            transitions=[transition],
            trainable_batch=[transition],
            rejections=[],
            reward_audit=[],
            sampling_audit=[],
            model_audit={"xunce_checkpoint_audit": {"checkpoint_loaded": True}},
            reason_codes=[],
        )

    monkeypatch.setattr(runner, "_collect_rollouts", fake_collect)

    summary = runner.run_xunce_stage21_1_on_policy_ppo_rollout_collector(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "implement_stage21_2_coverage_first_ppo_reward_contract"
    assert summary["trainable_transition_count"] == 1
    assert summary["mask_violation_count"] == 0
    assert summary["hard_risk_violation_count"] == 0
    assert summary["runs_new_ppo_update"] is False
    assert summary["publishes_checkpoint"] is False
    assert (tmp_path / "out" / "xunce-stage21-1-ppo-trainable-batch.jsonl").is_file()
    assert (tmp_path / "out" / "xunce-stage21-1-report.md").is_file()


def test_stage21_1_boundary_flag_hard_fails(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    config = _write_config(tmp_path, publishes_checkpoint=True)
    monkeypatch.setattr(
        runner,
        "_collect_rollouts",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("collector should not run")),
    )

    summary = runner.run_xunce_stage21_1_on_policy_ppo_rollout_collector(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage21_1_on_policy_collector_boundary_rejections"
    assert "publishes_checkpoint" in summary["blocking_reason_codes"]


def test_stage21_1_treats_late_no_hybrid_reachable_as_terminal_when_trainable_enough(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    config = _write_config(tmp_path, min_trainable_transition_count=1, continuous_theta_action_space_enabled=True)
    transition = _transition()

    def fake_collect(**_kwargs):
        return runner.CollectionResult(
            episodes=[
                {
                    "schema_version": "xunce-stage21-1-ppo-rollout-episode/v1",
                    "scenario_id": "s1",
                    "trainable_transition_count": 1,
                    "hard_risk_violation_count": 0,
                    "reason_codes": ["no_hybrid_reachable_candidate_terminal"],
                }
            ],
            transitions=[transition],
            trainable_batch=[transition],
            rejections=[
                {
                    "reason": "no_hybrid_reachable_candidate_terminal",
                    "action_mask": [True, True],
                    "hard_risk_clean_mask": [True, True],
                    "hybrid_astar_reachable_mask": [False, False],
                    "sampling_mask": [False, False],
                    "action_mask_true_count": 2,
                    "hard_risk_clean_mask_true_count": 2,
                    "hybrid_astar_reachable_count": 0,
                    "sampling_mask_true_count": 0,
                }
            ],
            reward_audit=[],
            sampling_audit=[],
            model_audit={"xunce_checkpoint_audit": {"checkpoint_loaded": True}},
            reason_codes=["no_hybrid_reachable_candidate_terminal"],
        )

    monkeypatch.setattr(runner, "_collect_rollouts", fake_collect)

    summary = runner.run_xunce_stage21_1_on_policy_ppo_rollout_collector(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    rejections = [
        json.loads(line)
        for line in (tmp_path / "out" / runner.REJECTION_FILE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "implement_stage21_2_coverage_first_ppo_reward_contract"
    assert summary["no_hybrid_reachable_candidate_terminal_count"] == 1
    assert "no_hybrid_reachable_candidate_terminal" not in summary["blocking_reason_codes"]
    assert rejections[0]["hybrid_astar_reachable_count"] == 0


def test_stage21_1_keeps_early_no_hybrid_reachable_terminal_blocking(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    config = _write_config(tmp_path, min_trainable_transition_count=2, continuous_theta_action_space_enabled=True)
    transition = _transition()

    def fake_collect(**_kwargs):
        return runner.CollectionResult(
            episodes=[],
            transitions=[transition],
            trainable_batch=[transition],
            rejections=[{"reason": "no_hybrid_reachable_candidate_terminal"}],
            reward_audit=[],
            sampling_audit=[],
            model_audit={"xunce_checkpoint_audit": {"checkpoint_loaded": True}},
            reason_codes=["no_hybrid_reachable_candidate_terminal"],
        )

    monkeypatch.setattr(runner, "_collect_rollouts", fake_collect)

    summary = runner.run_xunce_stage21_1_on_policy_ppo_rollout_collector(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "expand_stage21_1_on_policy_rollout_collection"
    assert "trainable_transition_count_below_minimum" in summary["blocking_reason_codes"]


def test_no_sampling_candidate_reason_identifies_hybrid_reachability_terminal() -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    assert (
        runner._no_sampling_candidate_reason(
            action_mask=(True, True),
            hard_risk_clean_mask=(True, True),
            hybrid_reachable_mask=(False, False),
        )
        == "no_hybrid_reachable_candidate_terminal"
    )
    assert (
        runner._no_sampling_candidate_reason(
            action_mask=(False, False),
            hard_risk_clean_mask=(True, True),
            hybrid_reachable_mask=(True, True),
        )
        == "no_action_mask_candidate"
    )


def test_stage21_1_requires_stage21_0_passed(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    config = _write_config(tmp_path, stage21_0_status="failed")
    monkeypatch.setattr(
        runner,
        "_collect_rollouts",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("collector should not run")),
    )

    summary = runner.run_xunce_stage21_1_on_policy_ppo_rollout_collector(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage21_1_on_policy_collector_inputs"
    assert "stage21_0_readiness_not_passed" in summary["blocking_reason_codes"]


def test_hard_risk_clean_sampling_mask_excludes_hard_risk_candidates() -> None:
    from scripts.run_xunce_stage21_1_on_policy_ppo_rollout_collector import (
        _hard_risk_clean_mask,
        _hard_risk_clean_sampling_mask,
    )

    candidates = [
        {"reachable": True, "path_cost": 1.0, "path_allowed_by_risk": True},
        {"reachable": True, "path_cost": 1.0, "path_allowed_by_risk": False},
        {"reachable": True, "path_cost": 1.0, "hard_risk_flags": ["slope_platform_limit_violation"]},
        {"reachable": True, "path_cost": 1.0, "open_grid_fallback_used": True},
    ]

    assert _hard_risk_clean_mask(candidates, allow_open_grid_fallback=False) == (
        True,
        False,
        False,
        False,
    )
    assert _hard_risk_clean_sampling_mask(candidates, (True, True, True, True), allow_open_grid_fallback=False) == (
        True,
        False,
        False,
        False,
    )
    assert _hard_risk_clean_sampling_mask(candidates, (False, True, True, True), allow_open_grid_fallback=False) == (
        False,
        False,
        False,
        False,
    )


def test_log_prob_recompute_matches_categorical_distribution() -> None:
    from scripts.run_xunce_stage21_1_on_policy_ppo_rollout_collector import _recompute_log_prob

    logits = [0.1, 0.2, -1.0]
    expected = float(torch.distributions.Categorical(logits=torch.tensor(logits)).log_prob(torch.tensor(1)).item())

    assert abs(_recompute_log_prob(logits, 1) - expected) < 1.0e-8


def test_slope_obstacle_theta_metadata_records_strict_new_visible_counts(monkeypatch) -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    candidates = [
        {"cell": [1, 1], "candidate_theta_deg": 0, "path_cost": 2.0},
        {"cell": [2, 1], "candidate_theta_deg": 45, "path_cost": 4.0},
    ]
    covered = {(0, 0)}

    def fake_coverage(*, end, **_kwargs):
        return {(int(end[0]), int(end[1])), (0, 0)}

    monkeypatch.setattr(runner.hf, "_candidate_coverage_cells", fake_coverage)
    metadata = runner._slope_obstacle_theta_metadata(
        candidates,
        current_cell=(0, 0),
        covered_cells=covered,
        config={
            "slope_obstacle_aware_theta_reward_enabled": True,
            "obstacle_occlusion_enabled": True,
            "platform_contract_hash": "platform-hash",
            "platform_contract_id": "agilex_scout_mini_piper",
            "max_traversable_slope_deg": 30.0,
        },
        obstacle_source_linkage={
            "obstacle_source_hash": "slope-source-hash",
            "obstacle_source_kind": "slope_blocked_as_obstacle_proxy",
        },
    )

    assert metadata["obstacle_aware_new_visible_cell_counts"] == [1, 1]
    assert metadata["obstacle_aware_theta_coverage_gain_per_path_costs"] == [0.5, 0.25]
    assert metadata["slope_obstacle_source_hash"] == "slope-source-hash"
    assert metadata["platform_contract_hash"] == "platform-hash"
    assert metadata["max_traversable_slope_deg"] == 30.0
    assert metadata["slope_blocked_source_kind"] == "slope_blocked_as_obstacle_proxy"
    assert metadata["strict_obstacle_aware_new_visible_cell_count"] is True


def test_hybrid_astar_path_cost_metadata_records_candidate_level_pose_costs(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    sidecar = tmp_path / "scenario.path-planner-sidecar.json"
    sidecar.write_text(
        json.dumps(
            {
                "cost": [[1.0, 1.0], [1.0, 1.0]],
                "passable_mask": [[True, True], [True, True]],
                "metadata": {"map_source": {"resolution_m": 1.0}},
                "max_traversable_slope_deg": 30.0,
            }
        ),
        encoding="utf-8",
    )
    candidates = [
        {"candidate_viewpoint": [1, 0, 0], "candidate_theta_deg": 0, "path_cost": 1.0},
        {"candidate_viewpoint": [1, 1, 45], "candidate_theta_deg": 45, "path_cost": 2.0},
    ]

    def fake_evaluate(**kwargs):
        candidate = kwargs["candidate"]
        index = int(candidate["candidate_index"])
        assert candidate["candidate_set_hash"] == "candidate-set-hash"
        assert len(kwargs["current_pose"]) == 3
        return {
            "path_cost_source_recommendation": "hybrid_astar_pose_path/v1",
            "hybrid_astar_reachable": True,
            "hybrid_astar_trajectory_kind": "hybrid_astar_pose_path",
            "hybrid_astar_path_cost": 10.0 + index,
            "hybrid_astar_pose_path_hash": f"pose-path-{index}",
            "hybrid_astar_failure_reason": None,
            "legacy_grid_astar_path_cost": float(candidate["path_cost"]),
            "hybrid_vs_grid_path_cost_delta": 9.0,
            "default_astar_replaced": False,
            "hybrid_astar_ackermann_feasible_claimed": False,
        }

    monkeypatch.setattr(runner, "evaluate_hybrid_astar_candidate_path_cost", fake_evaluate)
    metadata = runner._hybrid_astar_path_cost_metadata(
        candidates,
        current_cell=(0, 0),
        current_theta_deg=45.0,
        candidate_set_hash_value="candidate-set-hash",
        config={"hybrid_astar_pose_path_cost_enabled": True, "max_traversable_slope_deg": 30.0},
        slice_row={"sidecar": str(sidecar)},
        platform_contract_hash="platform-hash",
    )

    assert metadata["path_cost_source"] == "hybrid_astar_pose_path/v1"
    assert metadata["path_cost_sources"] == ["hybrid_astar_pose_path/v1", "hybrid_astar_pose_path/v1"]
    assert metadata["hybrid_astar_path_costs"] == [10.0, 11.0]
    assert metadata["hybrid_astar_pose_path_hashes"] == ["pose-path-0", "pose-path-1"]
    assert metadata["hybrid_astar_trajectory_kinds"] == ["hybrid_astar_pose_path", "hybrid_astar_pose_path"]
    assert metadata["legacy_grid_astar_path_costs"] == [1.0, 2.0]
    assert metadata["default_astar_replaced"] is False
    assert metadata["hybrid_astar_ackermann_feasible_claimed"] is False
    assert metadata["hybrid_astar_current_pose_provenance"] == "stage21_1_current_cell_plus_previous_selected_theta/v1"


def test_hybrid_astar_path_cost_metadata_parallelizes_and_restores_candidate_order(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    sidecar = tmp_path / "scenario.path-planner-sidecar.json"
    sidecar.write_text(
        json.dumps(
            {
                "cost": [[1.0, 1.0], [1.0, 1.0]],
                "passable_mask": [[True, True], [True, True]],
                "metadata": {"map_source": {"resolution_m": 1.0}},
                "max_traversable_slope_deg": 30.0,
            }
        ),
        encoding="utf-8",
    )
    candidates = [
        {"candidate_viewpoint": [1, 0, 0], "candidate_theta_deg": 0, "path_cost": 1.0},
        {"candidate_viewpoint": [1, 1, 45], "candidate_theta_deg": 45, "path_cost": 2.0},
    ]
    submitted: list[int] = []

    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakeExecutor:
        def __init__(self, max_workers: int):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, args):
            submitted.append([int(index) for index, _payload in args[2]])
            return FakeFuture(fn(args))

    def fake_evaluate(**kwargs):
        candidate = kwargs["candidate"]
        index = int(candidate["candidate_index"])
        return {
            "path_cost_source_recommendation": "hybrid_astar_pose_path/v1",
            "hybrid_astar_reachable": True,
            "hybrid_astar_trajectory_kind": "hybrid_astar_pose_path",
            "hybrid_astar_path_cost": 20.0 + index,
            "hybrid_astar_pose_path_hash": f"parallel-pose-path-{index}",
            "hybrid_astar_failure_reason": None,
            "legacy_grid_astar_path_cost": float(candidate["path_cost"]),
            "hybrid_vs_grid_path_cost_delta": 19.0,
            "default_astar_replaced": False,
            "hybrid_astar_ackermann_feasible_claimed": False,
        }

    monkeypatch.setattr(runner, "ProcessPoolExecutor", FakeExecutor, raising=False)
    monkeypatch.setattr(runner, "as_completed", lambda futures: list(reversed(list(futures))), raising=False)
    monkeypatch.setattr(runner, "evaluate_hybrid_astar_candidate_path_cost", fake_evaluate)

    metadata = runner._hybrid_astar_path_cost_metadata(
        candidates,
        current_cell=(0, 0),
        current_theta_deg=45.0,
        candidate_set_hash_value="candidate-set-hash",
        config={
            "hybrid_astar_pose_path_cost_enabled": True,
            "hybrid_astar_candidate_eval_workers": 2,
            "max_traversable_slope_deg": 30.0,
        },
        slice_row={"sidecar": str(sidecar)},
        platform_contract_hash="platform-hash",
    )

    assert submitted == [[0], [1]]
    assert metadata["hybrid_astar_path_costs"] == [20.0, 21.0]
    assert metadata["hybrid_astar_pose_path_hashes"] == ["parallel-pose-path-0", "parallel-pose-path-1"]
    assert metadata["hybrid_astar_candidate_eval_parallel_enabled"] is True
    assert metadata["hybrid_astar_candidate_eval_workers_requested"] == 2
    assert metadata["hybrid_astar_candidate_eval_workers_effective"] == 2
    assert metadata["hybrid_astar_candidate_eval_submitted_count"] == 2
    assert metadata["hybrid_astar_candidate_eval_failed_count"] == 0
    assert metadata["hybrid_astar_candidate_eval_duration_s"] >= 0.0


def test_hybrid_astar_path_cost_metadata_parallel_failure_is_per_candidate(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    sidecar = tmp_path / "scenario.path-planner-sidecar.json"
    sidecar.write_text(
        json.dumps(
            {
                "cost": [[1.0, 1.0], [1.0, 1.0]],
                "passable_mask": [[True, True], [True, True]],
                "metadata": {"map_source": {"resolution_m": 1.0}},
                "max_traversable_slope_deg": 30.0,
            }
        ),
        encoding="utf-8",
    )
    candidates = [
        {"candidate_viewpoint": [1, 0, 0], "candidate_theta_deg": 0, "path_cost": 1.0},
        {"candidate_viewpoint": [1, 1, 45], "candidate_theta_deg": 45, "path_cost": 2.0},
    ]

    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakeExecutor:
        def __init__(self, max_workers: int):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, args):
            return FakeFuture(fn(args))

    def fake_evaluate(**kwargs):
        index = int(kwargs["candidate"]["candidate_index"])
        if index == 1:
            raise RuntimeError("boom")
        return {
            "path_cost_source_recommendation": "hybrid_astar_pose_path/v1",
            "hybrid_astar_reachable": True,
            "hybrid_astar_trajectory_kind": "hybrid_astar_pose_path",
            "hybrid_astar_path_cost": 30.0,
            "hybrid_astar_pose_path_hash": "parallel-pose-path-0",
            "hybrid_astar_failure_reason": None,
            "legacy_grid_astar_path_cost": 1.0,
            "hybrid_vs_grid_path_cost_delta": 29.0,
            "default_astar_replaced": False,
            "hybrid_astar_ackermann_feasible_claimed": False,
        }

    monkeypatch.setattr(runner, "ProcessPoolExecutor", FakeExecutor, raising=False)
    monkeypatch.setattr(runner, "as_completed", lambda futures: list(futures), raising=False)
    monkeypatch.setattr(runner, "evaluate_hybrid_astar_candidate_path_cost", fake_evaluate)

    metadata = runner._hybrid_astar_path_cost_metadata(
        candidates,
        current_cell=(0, 0),
        current_theta_deg=45.0,
        candidate_set_hash_value="candidate-set-hash",
        config={
            "hybrid_astar_pose_path_cost_enabled": True,
            "hybrid_astar_candidate_eval_workers": 2,
            "max_traversable_slope_deg": 30.0,
        },
        slice_row={"sidecar": str(sidecar)},
        platform_contract_hash="platform-hash",
    )

    assert metadata["hybrid_astar_path_costs"] == [30.0, None]
    assert metadata["hybrid_astar_pose_path_hashes"] == ["parallel-pose-path-0", None]
    assert metadata["hybrid_astar_reachable_flags"] == [True, False]
    assert metadata["hybrid_astar_failure_reasons"][1] == "hybrid_astar_evaluation_failed"
    assert metadata["hybrid_astar_candidate_eval_failed_count"] == 1


def _write_config(tmp_path: Path, **overrides) -> Path:
    from model_explorer.policy.canonical_reward import load_canonical_reward_profile

    profile = load_canonical_reward_profile(REPO_ROOT / "configs" / "xunce_canonical_reward_guard_profile_v3.json")
    stage21_0 = tmp_path / "stage21_0"
    stage21_0.mkdir()
    (stage21_0 / "xunce-stage21-0-pure-ppo-readiness-summary.json").write_text(
        json.dumps(
            {
                "schema_version": "xunce-stage21-0-pure-ppo-readiness-summary/v1",
                "status": overrides.pop("stage21_0_status", "passed"),
                "next_required_change": "implement_stage21_1_xunce_on_policy_ppo_rollout_collector",
                "profile_hash": profile.profile_hash,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    payload = {
        "schema_version": "xunce-stage21-1-on-policy-ppo-rollout-collector-config/v1",
        "stage21_0_readiness_root": str(stage21_0),
        "high_fidelity_config": "configs/xunce_high_fidelity_exploration_coverage_comparison_stage18_9_strict_v3.json",
        "canonical_reward_profile": "configs/xunce_canonical_reward_guard_profile_v3.json",
        "dynamic_validation_work_root": str(tmp_path / "dynamic_validation"),
        "required_scenario_count": 1,
        "rollout_steps": 1,
        "dynamic_max_candidates_per_step": 2,
        "dynamic_proposal_pool_limit_per_step": 8,
        "sampling_seed": 1,
        "sampling_temperature": 1.0,
        "min_trainable_transition_count": 1,
        "max_log_prob_recompute_abs_error": 1.0e-6,
        "stage21_1_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    config = tmp_path / "stage21_1_config.json"
    config.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def _transition() -> dict:
    return {
        "schema_version": "xunce-stage21-1-ppo-transition/v1",
        "transition_id": "s1:step-0:sample-0",
        "scenario_id": "s1",
        "step_index": 0,
        "observation": {"action_mask": [True]},
        "xunce_batch": {"action_mask": {"shape": [1, 1], "dtype": "bool", "values": [[True]]}},
        "action_index": 0,
        "old_log_prob": 0.0,
        "old_value": 0.25,
        "reward": 0.1,
        "next_observation": None,
        "next_xunce_batch": None,
        "done": True,
        "trainable": True,
        "info": {
            "action_mask": [True],
            "sampling_mask": [True],
            "hard_risk_clean_mask": [True],
            "argmax_action_index": 0,
            "old_log_prob_recompute_abs_error": 0.0,
            "hard_risk_violation": False,
        },
    }


def _reachability_provenance() -> dict:
    from scripts import run_xunce_stage21_1_on_policy_ppo_rollout_collector as runner

    return {
        "schema_version": runner.CANDIDATE_REACHABILITY_PROVENANCE_SCHEMA_VERSION,
        "source": runner.CANDIDATE_REACHABILITY_GATE_SOURCE,
        "backend": runner.HYBRID_ASTAR_PATH_COST_SOURCE,
        "candidate_index": 0,
        "candidate_set_hash": "candidate-set",
        "planner_config_hash": "planner-hash",
        "reachable": True,
        "path_cost": 3.0,
        "pose_path_hash": "pose-hash",
    }
