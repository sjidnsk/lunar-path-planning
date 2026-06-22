import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_EXPLORER_SRC = REPO_ROOT / "model-explorer" / "src"
if str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))


EXPECTED_COMPONENTS = {
    "coverage_component",
    "valuable_coverage_component",
    "information_component",
    "path_cost_component",
    "risk_component",
    "fallback_component",
    "failure_component",
}


FORBIDDEN_SOFT_REWARD_FIELDS = {
    "path_cost_excess",
    "risk_excess",
    "risk_cost_exposure_penalty",
    "path_cost_efficiency_penalty",
    "risk_efficiency_penalty",
    "efficiency_regression_penalty",
    "coverage_per_100m_delta",
    "coverage_gain_per_path_cost_delta",
    "path_cost_delta_m",
    "risk_delta",
    "risk_cost_weighted_delta",
    "controlled_regression_penalty",
}


def _load_default_profile():
    from model_explorer.policy.canonical_reward import load_canonical_reward_profile

    return load_canonical_reward_profile(REPO_ROOT / "configs" / "xunce_canonical_reward_guard_profile_v2.json")


def test_soft_reward_contains_only_seven_signed_components() -> None:
    from model_explorer.policy.canonical_reward import compute_canonical_reward_components

    result = compute_canonical_reward_components(
        {
            "coverage_gain_rate": 2.0,
            "valuable_coverage": 25000.0,
            "information_gain": 1.0,
            "path_cost_m": 100.0,
            "risk_proxy": 5.0,
            "fallback_used": True,
            "failure": True,
        },
        _load_default_profile(),
    )

    assert set(result.components) == EXPECTED_COMPONENTS
    assert result.components["coverage_component"] == 2.0
    assert result.components["valuable_coverage_component"] == 0.5
    assert result.components["information_component"] == 0.25
    assert result.components["path_cost_component"] == -0.2
    assert result.components["risk_component"] == -0.05
    assert result.components["fallback_component"] == -1.0
    assert result.components["failure_component"] == -1.0
    assert result.reward == sum(result.components.values())
    assert result.reward == 0.5
    assert result.profile_hash


def test_path_and_risk_are_penalized_once_and_forbidden_delta_fields_do_not_change_reward() -> None:
    from model_explorer.policy.canonical_reward import compute_canonical_reward_components

    profile = _load_default_profile()
    base_metrics = {
        "coverage_gain_rate": 1.0,
        "valuable_coverage": 0.0,
        "information_gain": 0.0,
        "path_cost_m": 50.0,
        "risk_proxy": 2.5,
        "fallback_used": False,
        "failure": False,
    }
    base = compute_canonical_reward_components(base_metrics, profile)
    with_forbidden = compute_canonical_reward_components(
        {
            **base_metrics,
            **{field: 9999.0 for field in FORBIDDEN_SOFT_REWARD_FIELDS},
        },
        profile,
    )

    assert base.components == with_forbidden.components
    assert base.components["path_cost_component"] == -0.1
    assert base.components["risk_component"] == -0.025
    assert not FORBIDDEN_SOFT_REWARD_FIELDS.intersection(base.components)


def test_reward_component_result_serializes_profile_identity() -> None:
    from model_explorer.policy.canonical_reward import compute_canonical_reward_components

    result = compute_canonical_reward_components({}, _load_default_profile())
    payload = result.to_dict()

    json.dumps(payload, sort_keys=True)
    assert payload["profile_id"] == "xunce-coverage-cost-risk-budget-v2"
    assert payload["profile_version"] == "v2"
    assert payload["profile_hash"] == result.profile_hash


def test_compute_step_reward_can_emit_canonical_components_without_breaking_legacy_call() -> None:
    from model_explorer.core.interfaces import GoalCandidate
    from model_explorer.policy.reward import compute_step_reward

    profile = _load_default_profile()
    selected_goal = GoalCandidate(
        cell=(1, 2),
        utility=1.0,
        reachable=True,
        experimental={"path_cost": 100.0, "risk": 5.0, "valuable_coverage": 25000.0},
    )
    canonical = compute_step_reward(
        selected_goal,
        {"coverage_rate_delta": 2.0, "information_gain": 1.0},
        canonical_profile=profile,
    )
    legacy = compute_step_reward(selected_goal, {"coverage_rate_delta": 2.0})

    assert set(canonical.reward_components) == EXPECTED_COMPONENTS
    assert canonical.reward == sum(canonical.reward_components.values())
    assert canonical.profile_id == profile.profile_id
    assert canonical.profile_hash == profile.profile_hash
    assert legacy.reward_components == {}
    assert legacy.profile_hash is None


def test_collector_and_evaluation_load_canonical_profile_from_reward_config_path() -> None:
    from model_explorer.core.interfaces import ConstraintSummary, GoalCandidate, GridSummary, ModelExplorerContract
    from model_explorer.policy.collector import collect_rollout_episode
    from model_explorer.policy.evaluation import evaluate_policy_baselines

    profile = _load_default_profile()
    profile_path = REPO_ROOT / "configs" / "xunce_canonical_reward_guard_profile_v2.json"
    contract = ModelExplorerContract(
        schema_version="model-explorer-contract/v1",
        grid=GridSummary(width=4, height=4, resolution=1.0, frame_id="map", origin=(0.0, 0.0), layers=()),
        constraints=ConstraintSummary(violation_count=0, passable_ratio=1.0, reason_counts={}),
        top_goals=(
            GoalCandidate(
                cell=(1, 1),
                utility=1.0,
                reachable=True,
                experimental={"path_cost": 10.0, "risk": 0.5, "valuable_coverage": 100.0},
            ),
        ),
        top_sequences=(),
        observation_update={"coverage_rate_delta": 1.0, "information_gain": 0.2},
    )

    episode = collect_rollout_episode(
        (contract,),
        reward_config={"canonical_profile_path": str(profile_path)},
        selection_strategy="utility",
    )
    report = evaluate_policy_baselines(
        (contract,),
        reward_config={"canonical_profile_path": str(profile_path)},
    )

    assert episode.transitions[0].info.extra["profile_hash"] == profile.profile_hash
    assert episode.transitions[0].info.extra["profile_id"] == profile.profile_id
    assert report["utility"]["profile_hash"] == profile.profile_hash
