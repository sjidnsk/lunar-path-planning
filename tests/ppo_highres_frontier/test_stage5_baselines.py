"""Stage 5 基线选择与 deterministic PPO 动作合同。"""

from __future__ import annotations

import importlib
import inspect
import csv
import io
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from lunar_exploration_ppo.env.env import EnvAction
from lunar_exploration_ppo.configs.stage1 import load_stage1_config
from lunar_exploration_ppo.env.env import LunarExplorationEnv
from lunar_exploration_ppo.env.frontier import FrontierActionSet
from lunar_exploration_ppo.env.scenario import ScenarioBundle, ScenarioSource, TruthMap
from lunar_exploration_ppo.policy.cross_attention import PolicyBatch, PolicyForwardOutput
from lunar_exploration_ppo.policy.observation import PolicyObservation
from lunar_exploration_ppo.ppo.trainer import policy_state_sha256


def _baselines_module():
    return importlib.import_module("lunar_exploration_ppo.eval.baselines")


def _metrics_module():
    return importlib.import_module("lunar_exploration_ppo.eval.metrics")


def _reports_module():
    return importlib.import_module("lunar_exploration_ppo.eval.reports")


def _evaluator_module():
    return importlib.import_module("lunar_exploration_ppo.eval.evaluator")


def _observation(
    *,
    distances: tuple[float, ...] = (0.4, 0.1, 0.0, 0.2),
    gains: tuple[float, ...] = (0.1, 0.3, 1.0, 0.8),
    costs: tuple[float, ...] = (0.0, 0.2, 0.0, 2.0),
    mask: tuple[bool, ...] = (True, True, False, True),
) -> PolicyObservation:
    count = len(mask)
    features = np.zeros((count, 22), dtype=np.float32)
    features[:, 2] = np.asarray(distances, dtype=np.float32)
    features[:, 5] = np.asarray(gains, dtype=np.float32)
    features[:, 18] = np.asarray(costs, dtype=np.float32)
    for index in range(count):
        theta = 0.2 * (index + 1)
        features[index, 14] = math.sin(theta)
        features[index, 15] = math.cos(theta)
    return PolicyObservation(
        prior_channels=np.zeros((7, 2, 2), dtype=np.float32),
        coverage_summary=np.zeros((8, 2, 2), dtype=np.float32),
        local_crop=np.zeros((8, 2, 2), dtype=np.float32),
        frontier_features=features,
        pose_features=np.zeros((6,), dtype=np.float32),
        candidate_mask=np.asarray(mask, dtype=bool),
    )


def _policy_output(*, logits: tuple[float, ...], theta_mu: tuple[float, ...]) -> PolicyForwardOutput:
    logits_tensor = torch.tensor((logits,), dtype=torch.float32)
    theta_tensor = torch.tensor((theta_mu,), dtype=torch.float32)
    shape = logits_tensor.shape
    zeros = torch.zeros(shape, dtype=torch.float32)
    ones = torch.ones(shape, dtype=torch.float32)
    batch = shape[0]
    return PolicyForwardOutput(
        frontier_logits=logits_tensor,
        theta_mu_sin_raw=zeros,
        theta_mu_cos_raw=ones,
        theta_kappa_raw=zeros,
        theta_mu=theta_tensor,
        theta_kappa=ones,
        value=torch.zeros((batch,), dtype=torch.float32),
        global_map_tokens=torch.zeros((batch, 1, 1), dtype=torch.float32),
        local_map_tokens=torch.zeros((batch, 1, 1), dtype=torch.float32),
        pose_token=torch.zeros((batch, 1, 1), dtype=torch.float32),
        context_tokens=torch.zeros((batch, 1, 1), dtype=torch.float32),
        refined_frontier_tokens=torch.zeros((*shape, 1), dtype=torch.float32),
        action_hidden=torch.zeros((*shape, 1), dtype=torch.float32),
    )


def test_stage5_baseline_surface_and_exact_method_order() -> None:
    module = _baselines_module()
    assert module.BASELINE_METHODS == (
        "random_valid_frontier",
        "nearest_frontier",
        "max_potential_gain_frontier",
        "gain_over_cost_frontier",
    )
    assert module.ALL_METHODS == (*module.BASELINE_METHODS, "ppo_policy")
    assert callable(module.select_baseline_action)
    assert callable(module.select_ppo_action)


@pytest.mark.parametrize(
    ("method", "expected_index"),
    (
        ("nearest_frontier", 1),
        ("max_potential_gain_frontier", 3),
        ("gain_over_cost_frontier", 3),
    ),
)
def test_deterministic_baseline_score_formulas_ignore_masked_candidates(
    method: str,
    expected_index: int,
) -> None:
    module = _baselines_module()
    observation = _observation()

    action = module.select_baseline_action(
        method,
        observation,
        rng=np.random.Generator(np.random.PCG64(7)),
    )

    assert isinstance(action, EnvAction)
    assert action.candidate_index == expected_index
    assert action.target_theta == pytest.approx(0.2 * (expected_index + 1), abs=1e-7)


@pytest.mark.parametrize(
    "method",
    (
        "nearest_frontier",
        "max_potential_gain_frontier",
        "gain_over_cost_frontier",
    ),
)
def test_deterministic_baseline_ties_choose_lowest_valid_index(method: str) -> None:
    module = _baselines_module()
    observation = _observation(
        distances=(9.0, 0.2, 0.2, 0.2),
        gains=(9.0, 0.4, 0.4, 0.4),
        costs=(0.0, 0.0, 0.0, 0.0),
        mask=(False, True, True, True),
    )

    action = module.select_baseline_action(
        method,
        observation,
        rng=np.random.Generator(np.random.PCG64(7)),
    )

    assert action.candidate_index == 1


def test_random_baseline_is_uniform_over_valid_indices_and_fixed_seed_replays() -> None:
    module = _baselines_module()
    observation = _observation(mask=(True, False, True, True))
    first_rng = np.random.Generator(np.random.PCG64(20260715))
    second_rng = np.random.Generator(np.random.PCG64(20260715))

    first = [
        module.select_baseline_action(
            "random_valid_frontier", observation, rng=first_rng
        ).candidate_index
        for _ in range(3000)
    ]
    second = [
        module.select_baseline_action(
            "random_valid_frontier", observation, rng=second_rng
        ).candidate_index
        for _ in range(3000)
    ]

    assert first == second
    assert set(first) == {0, 2, 3}
    counts = np.bincount(first, minlength=4)
    assert all(abs(int(counts[index]) - 1000) < 100 for index in (0, 2, 3))
    assert counts[1] == 0


def test_recommended_theta_reconstruction_is_exact_and_fails_closed() -> None:
    module = _baselines_module()
    row = np.zeros((22,), dtype=np.float32)
    row[14] = math.sin(-2.4)
    row[15] = math.cos(-2.4)
    assert module.reconstruct_recommended_theta(row) == pytest.approx(-2.4, abs=1e-7)

    for sine, cosine in ((0.0, 0.0), (1.0e-14, -1.0e-14), (math.nan, 1.0), (1.0, math.inf)):
        invalid = row.copy()
        invalid[14] = sine
        invalid[15] = cosine
        with pytest.raises(module.BaselineSelectionError, match="recommended theta"):
            module.reconstruct_recommended_theta(invalid)


def test_negative_masked_out_of_range_and_empty_indices_fail_closed() -> None:
    module = _baselines_module()
    mask = np.asarray((True, False, True), dtype=bool)
    for index in (-1, 1, 3, True):
        with pytest.raises(module.BaselineSelectionError, match="index"):
            module.validate_selected_index(index, mask)

    empty = _observation(mask=(False, False, False, False))
    with pytest.raises(module.NoCandidateAction, match="no valid candidate"):
        module.select_baseline_action(
            "nearest_frontier",
            empty,
            rng=np.random.Generator(np.random.PCG64(1)),
        )


def test_ppo_eval_uses_lowest_masked_argmax_and_selected_theta_mu() -> None:
    module = _baselines_module()
    output = _policy_output(
        logits=(1.0, 5.0, 100.0, 5.0),
        theta_mu=(-0.4, 0.25, 1.7, -2.0),
    )
    mask = torch.tensor(((True, True, False, True),), dtype=torch.bool)

    action = module.select_ppo_action(output, mask)

    assert action == EnvAction(candidate_index=1, target_theta=0.25)


def test_selection_boundary_accepts_no_truth_or_coverable_mask_input() -> None:
    module = _baselines_module()
    parameters = tuple(inspect.signature(module.select_baseline_action).parameters)
    assert parameters == ("method", "observation", "rng")
    assert "truth" not in inspect.getsource(module.select_baseline_action).lower()
    assert "coverable" not in inspect.getsource(module.select_baseline_action).lower()

    observation = _observation()
    hidden_truth = np.zeros((128, 128), dtype=np.float64)
    before = module.select_baseline_action(
        "nearest_frontier",
        observation,
        rng=np.random.Generator(np.random.PCG64(4)),
    )
    hidden_truth[100:, 100:] = 999.0
    after = module.select_baseline_action(
        "nearest_frontier",
        observation,
        rng=np.random.Generator(np.random.PCG64(4)),
    )
    assert before == after


def _episode(
    episode_index: int,
    *,
    method: str = "nearest_frontier",
    success: bool | None = None,
):
    module = _metrics_module()
    reached = episode_index % 2 == 0 if success is None else success
    if reached:
        curve = (0.10, 0.60, 0.99, 0.99, 0.99)
        path = (0.0, 1.0, 2.0, 2.0, 2.0)
        steps_executed = 2
        reason = "success_done"
    else:
        curve = (0.10, 0.20, 0.30, 0.30, 0.30)
        path = (0.0, 1.0, 2.0, 2.0, 2.0)
        steps_executed = 2
        reason = "stagnation_done"
    return module.build_episode_result(
        method=method,
        scale_profile="Smoke v1",
        scenario_key=f"smoke-v1-{episode_index:02d}",
        scenario_seed=100,
        terrain_seed=200,
        start_pose_seed=300,
        evaluation_seed=20260715 + episode_index,
        coverage_curve=curve,
        cumulative_path_length_curve=path,
        steps_executed=steps_executed,
        invalid_action_count=episode_index % 3,
        planner_failure_count=episode_index % 2,
        safety_violation_count=0,
        termination_reason=reason,
        max_steps=4,
        success_threshold=0.99,
        zero_distance_policy="zero_when_no_travel/v1",
    )


def test_episode_schema_threshold_auc_success_only_and_zero_distance_policy() -> None:
    module = _metrics_module()
    result = _episode(0, success=True)

    assert tuple(module.episode_record(result)) == module.EPISODE_FIELDS
    assert result.success is True
    assert result.final_coverage == 0.99
    assert result.steps_to_99_success_only == 2
    assert result.path_length_to_99_success_only == 2.0
    assert result.coverage_auc_over_steps == pytest.approx(0.78125)
    assert result.coverage_per_meter == pytest.approx(0.495)

    failed = _episode(1, success=False)
    assert failed.success is False
    assert failed.steps_to_99_success_only is None
    assert failed.path_length_to_99_success_only is None

    zero_distance = module.build_episode_result(
        method="nearest_frontier",
        scale_profile="Smoke v1",
        scenario_key="no-candidate",
        scenario_seed=1,
        terrain_seed=2,
        start_pose_seed=3,
        evaluation_seed=4,
        coverage_curve=(0.25, 0.25, 0.25),
        cumulative_path_length_curve=(0.0, 0.0, 0.0),
        steps_executed=0,
        invalid_action_count=0,
        planner_failure_count=0,
        safety_violation_count=0,
        termination_reason="no_candidate_done",
        max_steps=2,
        success_threshold=0.99,
        zero_distance_policy="zero_when_no_travel/v1",
    )
    assert math.isfinite(zero_distance.coverage_per_meter)
    assert zero_distance.coverage_per_meter == 0.0


def test_episode_metrics_reject_nonfinite_curves_and_inconsistent_success() -> None:
    module = _metrics_module()
    kwargs = {
        "method": "nearest_frontier",
        "scale_profile": "Smoke v1",
        "scenario_key": "invalid",
        "scenario_seed": 1,
        "terrain_seed": 2,
        "start_pose_seed": 3,
        "evaluation_seed": 4,
        "cumulative_path_length_curve": (0.0, 0.0, 0.0),
        "steps_executed": 0,
        "invalid_action_count": 0,
        "planner_failure_count": 0,
        "safety_violation_count": 0,
        "termination_reason": "failure_done",
        "max_steps": 2,
        "success_threshold": 0.99,
        "zero_distance_policy": "zero_when_no_travel/v1",
    }
    with pytest.raises(module.MetricError, match="finite"):
        module.build_episode_result(coverage_curve=(0.2, math.nan, 0.2), **kwargs)

    valid = _episode(0, success=True)
    with pytest.raises(module.MetricError, match="success-only"):
        module.validate_episode_result(
            replace(valid, success=False, steps_to_99_success_only=2)
        )


def test_episode_bootstrap_ci_is_fixed_seed_deterministic_and_replayable() -> None:
    module = _metrics_module()
    episodes = tuple(_episode(index) for index in range(16))

    first_summary, first_audit = module.summarize_episodes(
        episodes,
        bootstrap_resamples=2000,
        bootstrap_seed=20260715,
    )
    second_summary, second_audit = module.summarize_episodes(
        tuple(reversed(episodes)),
        bootstrap_resamples=2000,
        bootstrap_seed=20260715,
    )

    assert first_summary == second_summary
    assert first_audit == second_audit
    assert first_summary["episode_count"] == 16
    assert first_summary["success_rate_under_fixed_step_budget"] == 0.5
    assert first_summary["steps_to_99_success_only"] == 2.0
    assert first_audit["sampling_unit"] == "episode"
    assert first_audit["resample_count"] == 2000
    assert first_audit["bootstrap_seed"] == 20260715
    assert first_audit["sample_index_shape"] == [2000, 16]
    replayed = module.bootstrap_episode_indices(16, 2000, 20260715)
    assert module.bootstrap_indices_sha256(replayed) == first_audit["sample_indices_sha256"]
    assert first_audit["sample_indices_sha256"] == (
        "dbde1be8f5002814bb579f1f6122a2e8a7d4818e9205c5e05d1b5bf6d99399d8"
    )
    for name in module.BOOTSTRAP_METRICS:
        interval = first_audit["metrics"][name]
        assert interval["valid_resample_count"] > 0
        assert interval["ci95_low"] <= interval["estimate"] <= interval["ci95_high"]


def test_bootstrap_success_only_metrics_are_null_when_all_episodes_fail() -> None:
    module = _metrics_module()
    episodes = tuple(_episode(index, success=False) for index in range(16))

    summary, audit = module.summarize_episodes(
        episodes,
        bootstrap_resamples=2000,
        bootstrap_seed=20260715,
    )

    assert summary["success_rate_under_fixed_step_budget"] == 0.0
    assert summary["steps_to_99_success_only"] is None
    assert summary["path_length_to_99_success_only"] is None
    for name in ("steps_to_99_success_only", "path_length_to_99_success_only"):
        assert audit["metrics"][name] == {
            "estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "valid_resample_count": 0,
        }


def test_bootstrap_singleton_and_all_equal_samples_have_degenerate_finite_ci() -> None:
    module = _metrics_module()
    singleton = (_episode(0, success=True),)

    _, singleton_audit = module.summarize_episodes(
        singleton,
        bootstrap_resamples=2000,
        bootstrap_seed=20260715,
    )

    assert singleton_audit["sample_indices_sha256"] == (
        "f85f2c34eb2843d2aa5951ee6e8e76985655b2e3ae2cbdd76bdfd654ecf19997"
    )
    for interval in singleton_audit["metrics"].values():
        assert interval["valid_resample_count"] == 2000
        assert interval["ci95_low"] == interval["estimate"] == interval["ci95_high"]

    all_equal = tuple(
        replace(
            _episode(index, success=True),
            invalid_action_count=0,
            planner_failure_count=0,
        )
        for index in range(16)
    )
    _, all_equal_audit = module.summarize_episodes(
        all_equal,
        bootstrap_resamples=2000,
        bootstrap_seed=20260715,
    )
    for interval in all_equal_audit["metrics"].values():
        assert interval["valid_resample_count"] == 2000
        assert interval["ci95_low"] == interval["estimate"] == interval["ci95_high"]


def _all_method_report_inputs():
    metrics = _metrics_module()
    baselines = _baselines_module()
    summaries = []
    audits = []
    episodes = []
    for method_index, method in enumerate(baselines.ALL_METHODS):
        method_episodes = tuple(
            replace(
                _episode(index, method=method),
                scale_profile="Smoke, v1",
                final_coverage=min(1.0, _episode(index, method=method).final_coverage),
            )
            for index in range(4)
        )
        summary, audit = metrics.summarize_episodes(
            method_episodes,
            bootstrap_resamples=40,
            bootstrap_seed=20260715 + method_index,
        )
        summaries.append(summary)
        audits.append(audit)
        episodes.extend(method_episodes)
    return tuple(summaries), tuple(audits), tuple(episodes)


def test_comparison_csv_is_deterministic_quoted_lf_and_method_ordered() -> None:
    module = _reports_module()
    baselines = _baselines_module()
    summaries, audits, _ = _all_method_report_inputs()

    first = module.comparison_table_csv(tuple(reversed(summaries)), tuple(reversed(audits)))
    second = module.comparison_table_csv(summaries, audits)

    assert first == second
    assert b"\r\n" not in first
    assert first.endswith(b"\n")
    assert b'"Smoke, v1"' in first
    rows = list(csv.DictReader(io.StringIO(first.decode("utf-8"), newline="")))
    assert [row["method"] for row in rows] == list(baselines.ALL_METHODS)
    assert tuple(rows[0]) == module.COMPARISON_COLUMNS
    assert all(row["success_rate_under_fixed_step_budget_ci95_low"] for row in rows)


def test_coverage_curve_csv_is_deterministic_and_includes_reset_step_zero() -> None:
    module = _reports_module()
    baselines = _baselines_module()
    _, _, episodes = _all_method_report_inputs()

    first = module.coverage_curves_csv(tuple(reversed(episodes)))
    second = module.coverage_curves_csv(episodes)

    assert first == second
    assert b"\r\n" not in first
    rows = list(csv.DictReader(io.StringIO(first.decode("utf-8"), newline="")))
    assert tuple(rows[0]) == module.COVERAGE_CURVE_COLUMNS
    assert len(rows) == len(baselines.ALL_METHODS) * 5
    first_method_rows = [row for row in rows if row["method"] == baselines.ALL_METHODS[0]]
    assert [int(row["step"]) for row in first_method_rows] == [0, 1, 2, 3, 4]
    assert float(first_method_rows[0]["mean_coverage"]) == pytest.approx(0.1)


def test_baseline_report_is_byte_stable_and_states_claim_boundary() -> None:
    module = _reports_module()
    summaries, _, _ = _all_method_report_inputs()
    claim = "fair_baseline_evaluator_system_closure_no_task_advantage/v1"

    first = module.baseline_eval_report(tuple(reversed(summaries)), claim_boundary=claim)
    second = module.baseline_eval_report(summaries, claim_boundary=claim)

    assert first == second
    assert claim.encode("ascii") in first
    assert b"does not establish PPO task advantage" in first
    assert b"timestamp" not in first.lower()
    assert b"ppo_policy" in first
    assert b"Methods differ only in candidate selection" not in first
    assert b"recommended_theta" in first
    assert b"theta_mu" in first

    doc = (ROOT / "docs/ppo-highres-frontier-stage5.md").read_text(
        encoding="utf-8"
    )
    assert "recommended_theta" in doc
    assert "theta_mu" in doc
    assert "baseline" in doc and "PPO" in doc


def test_report_generation_rejects_summary_schema_drift() -> None:
    module = _reports_module()
    summaries, audits, _ = _all_method_report_inputs()
    broken = dict(summaries[0])
    del broken["planner_failure_count_mean"]

    with pytest.raises(module.ReportError, match="schema"):
        module.comparison_table_csv((broken, *summaries[1:]), audits)


ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
STAGE1_CONFIG = ROOT / "configs/ppo_highres_frontier_smoke_v1.json"


def _evaluation_scenario(evaluation_seed: int = 20260715):
    module = _evaluator_module()
    return module.EvaluationScenario(
        scenario_key="smoke-v1",
        scenario_seed=20260710,
        terrain_seed=20260710,
        start_pose_seed=0,
        evaluation_seed=evaluation_seed,
    )


def _real_evaluator(*, policy: torch.nn.Module | None = None):
    module = _evaluator_module()
    config = load_stage1_config(STAGE1_CONFIG)
    return module.Evaluator(
        env_factory=lambda _scenario: LunarExplorationEnv(config),
        scale_profile=config.scale_profile,
        max_steps=config.max_steps,
        success_threshold=config.success_coverage_rate,
        zero_distance_policy="zero_when_no_travel/v1",
        bootstrap_resamples=40,
        bootstrap_seed=20260715,
        policy=policy,
        policy_device="cpu",
    )


class _EmptyFrontierGenerator:
    def extract(self, observed_state, prior, pose) -> FrontierActionSet:
        del prior, pose
        top_m = 512
        shape = observed_state.geometry.shape
        return FrontierActionSet(
            cells=(),
            frontier_features=np.zeros((top_m, 22), dtype=np.float32),
            candidate_mask=np.zeros((top_m,), dtype=bool),
            frontier_mask=np.zeros(shape, dtype=bool),
            observed_safe_frontier_mask=np.zeros(shape, dtype=bool),
            reachable_frontier_mask=np.zeros(shape, dtype=bool),
        )


class _FixedEvalPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, observation: PolicyBatch) -> PolicyForwardOutput:
        mask = observation.candidate_mask
        batch, candidates = mask.shape
        logits = torch.zeros((batch, candidates), dtype=torch.float32, device=mask.device)
        theta_mu = torch.atan2(
            observation.frontier_features[..., 14],
            observation.frontier_features[..., 15],
        )
        zeros = torch.zeros_like(logits)
        ones = torch.ones_like(logits)
        token = torch.zeros((batch, 1, 1), dtype=torch.float32, device=mask.device)
        candidate_token = torch.zeros(
            (batch, candidates, 1), dtype=torch.float32, device=mask.device
        )
        return PolicyForwardOutput(
            frontier_logits=logits,
            theta_mu_sin_raw=zeros,
            theta_mu_cos_raw=ones,
            theta_kappa_raw=zeros,
            theta_mu=theta_mu,
            theta_kappa=ones,
            value=torch.zeros((batch,), dtype=torch.float32, device=mask.device),
            global_map_tokens=token,
            local_map_tokens=token,
            pose_token=token,
            context_tokens=token,
            refined_frontier_tokens=candidate_token,
            action_hidden=candidate_token,
        )


class _DistinctThetaEvalPolicy(_FixedEvalPolicy):
    def forward(self, observation: PolicyBatch) -> PolicyForwardOutput:
        output = super().forward(observation)
        return replace(output, theta_mu=torch.full_like(output.theta_mu, -1.25))


def test_fairness_audit_preserves_method_specific_index_and_theta_contracts() -> None:
    scenario = (_evaluation_scenario(),)
    evaluator = _real_evaluator(policy=_DistinctThetaEvalPolicy())

    baseline = evaluator.evaluate("nearest_frontier", scenario)
    ppo = evaluator.evaluate("ppo_policy", scenario)

    expected = {
        "nearest_frontier": {
            "candidate_index_rule": (
                "min_distance_from_robot_norm_lowest_index_tie/v1"
            ),
            "theta_rule": "selected_candidate_recommended_theta/v1",
        },
        "ppo_policy": {
            "candidate_index_rule": (
                "masked_logit_argmax_lowest_index_tie/v1"
            ),
            "theta_rule": "selected_candidate_policy_theta_mu/v1",
        },
    }
    for summary in (baseline, ppo):
        audit = summary.fairness_audit
        assert audit["schema_version"] == "stage5_method_fairness_audit/v2"
        assert audit["shared_environment_contract_identical"] is True
        assert audit["method_specific_action_rule_only_difference"] is True
        assert audit["candidate_index_rule"] == expected[summary.method][
            "candidate_index_rule"
        ]
        assert audit["theta_rule"] == expected[summary.method]["theta_rule"]
        assert "candidate_selection_only_difference" not in audit

    baseline_actions = baseline.fairness_audit["episode_audits"][0][
        "selected_actions"
    ]
    ppo_actions = ppo.fairness_audit["episode_audits"][0]["selected_actions"]
    assert baseline_actions
    assert ppo_actions
    assert all(
        row["target_theta"] == row["recommended_theta"]
        for row in baseline_actions
    )
    assert all(
        row["target_theta"] != row["recommended_theta"] for row in ppo_actions
    )


def test_all_nonlearning_baselines_complete_real_smoke_under_one_shared_contract() -> None:
    baselines = _baselines_module()
    metrics = _metrics_module()
    scenario = (_evaluation_scenario(),)
    summaries = tuple(_real_evaluator().evaluate(method, scenario) for method in baselines.BASELINE_METHODS)

    assert [summary.method for summary in summaries] == list(baselines.BASELINE_METHODS)
    assert all(len(summary.episodes) == 1 for summary in summaries)
    assert all(len(summary.episodes[0].coverage_curve) == 65 for summary in summaries)
    assert all(
        summary.episodes[0].termination_reason
        in {"success_done", "failure_done", "stagnation_done", "no_candidate_done", "safety_done"}
        for summary in summaries
    )
    assert len({summary.fairness_audit["shared_contract_sha256"] for summary in summaries}) == 1
    assert all(
        summary.fairness_audit["shared_environment_contract_identical"]
        for summary in summaries
    )
    assert all(
        summary.fairness_audit["method_specific_action_rule_only_difference"]
        for summary in summaries
    )
    assert all(summary.fairness_audit["all_selected_actions_reachable_observed_safe"] for summary in summaries)
    assert len({tuple(metrics.episode_record(summary.episodes[0])) for summary in summaries}) == 1


def test_environment_contract_binds_exact_coverable_denominator_metadata() -> None:
    config = load_stage1_config(STAGE1_CONFIG)
    env = LunarExplorationEnv(config)
    evaluator = _real_evaluator()

    contract = evaluator._environment_contract(env, _evaluation_scenario())

    assert contract["coverage_denominator_source"] == "coverable_mask_exact"
    assert contract["coverable_mask_exact"] is True
    assert contract["coverable_mask_hash"] == env.coverage_metadata["sha256"]
    assert contract["coverable_mask_algorithm_id"] == env.coverage_metadata["algorithm_id"]
    assert contract["coverable_mask_precompute_scope"] == env.coverage_metadata[
        "precompute_scope"
    ]
    assert contract["coverable_cell_count"] == env.coverage_metadata[
        "coverable_cell_count"
    ]


def test_baseline_fairness_audit_proves_exact_recommended_theta() -> None:
    summary = _real_evaluator().evaluate(
        "gain_over_cost_frontier",
        (_evaluation_scenario(),),
    )

    assert summary.fairness_audit["all_baseline_thetas_recommended"] is True
    selected = summary.fairness_audit["episode_audits"][0]["selected_actions"]
    assert selected
    assert all(row["target_theta"] == row["recommended_theta"] for row in selected)


@pytest.mark.parametrize(
    "method",
    (
        "nearest_frontier",
        "max_potential_gain_frontier",
        "gain_over_cost_frontier",
        "random_valid_frontier",
    ),
)
def test_baseline_episode_replays_identically_with_fixed_episode_seed(method: str) -> None:
    scenario = (_evaluation_scenario(20260731),)
    evaluator = _real_evaluator()

    first = evaluator.evaluate(method, scenario)
    second = evaluator.evaluate(method, scenario)

    assert first.episodes == second.episodes
    assert first.metrics == second.metrics
    assert first.fairness_audit == second.fairness_audit


def test_empty_candidates_use_environment_no_candidate_done_without_action() -> None:
    module = _evaluator_module()
    config = load_stage1_config(STAGE1_CONFIG)
    evaluator = module.Evaluator(
        env_factory=lambda _scenario: LunarExplorationEnv(
            config,
            frontier_generator=_EmptyFrontierGenerator(),
        ),
        scale_profile=config.scale_profile,
        max_steps=config.max_steps,
        success_threshold=0.99,
        zero_distance_policy="zero_when_no_travel/v1",
        bootstrap_resamples=40,
        bootstrap_seed=20260715,
    )

    summary = evaluator.evaluate("nearest_frontier", (_evaluation_scenario(),))
    episode = summary.episodes[0]
    assert episode.termination_reason == "no_candidate_done"
    assert episode.steps_to_99_success_only is None
    assert episode.path_length_to_99_success_only is None
    assert episode.coverage_per_meter == 0.0
    assert len(set(episode.coverage_curve)) == 1
    assert summary.fairness_audit["selected_action_count"] == 0


def test_ppo_evaluator_is_argmax_mean_theta_eval_only_and_preserves_parameters() -> None:
    policy = _FixedEvalPolicy()
    before = policy_state_sha256(policy)
    evaluator = _real_evaluator(policy=policy)

    summary = evaluator.evaluate("ppo_policy", (_evaluation_scenario(),))

    assert policy_state_sha256(policy) == before
    assert summary.method == "ppo_policy"
    assert len(summary.episodes) == 1
    assert summary.fairness_audit["ppo_eval_policy_mode"] == (
        "deterministic_argmax_frontier_mean_theta/v1"
    )
    assert summary.fairness_audit["all_selected_actions_reachable_observed_safe"] is True


class _StaticScenarioSource:
    def __init__(self, bundle: ScenarioBundle) -> None:
        self.bundle = bundle

    def load(self, key: str) -> ScenarioBundle:
        assert key == "smoke-v1"
        return self.bundle


def test_mutating_unobserved_highres_truth_cannot_change_baseline_decision() -> None:
    baselines = _baselines_module()
    config = load_stage1_config(STAGE1_CONFIG)
    original = ScenarioSource().load("smoke-v1")
    mutated_height = original.truth.height.copy()
    mutated_height[100:, 100:] += 1000.0
    mutated_truth = TruthMap(
        geometry=original.truth.geometry,
        height=mutated_height,
        hard_obstacle=original.truth.hard_obstacle.copy(),
        slope_deg=original.truth.slope_deg.copy(),
        traversability=original.truth.traversability.copy(),
        provenance=dict(original.truth.provenance),
    )
    mutated = replace(original, scenario_hash="f" * 64, truth=mutated_truth)
    original_env = LunarExplorationEnv(config, scenario_source=_StaticScenarioSource(original))
    mutated_env = LunarExplorationEnv(config, scenario_source=_StaticScenarioSource(mutated))
    original_observation = original_env.reset()
    mutated_observation = mutated_env.reset()

    for left, right in zip(
        original_observation.array_fields(),
        mutated_observation.array_fields(),
        strict=True,
    ):
        assert np.array_equal(left, right)
    first = baselines.select_baseline_action(
        "gain_over_cost_frontier",
        original_observation,
        np.random.Generator(np.random.PCG64(9)),
    )
    second = baselines.select_baseline_action(
        "gain_over_cost_frontier",
        mutated_observation,
        np.random.Generator(np.random.PCG64(9)),
    )
    assert first == second


def test_evaluator_decision_boundary_has_no_truth_or_dense_coverable_argument() -> None:
    module = _evaluator_module()
    parameters = tuple(inspect.signature(module.Evaluator._select_action).parameters)
    assert parameters == ("self", "method", "observation", "rng")
    source = inspect.getsource(module.Evaluator._select_action).lower()
    assert "truth" not in source
    assert "coverable" not in source
