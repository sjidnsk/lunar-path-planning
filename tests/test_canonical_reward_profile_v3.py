import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_EXPLORER_SRC = REPO_ROOT / "model-explorer" / "src"
if str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))


EXPECTED_V3_COMPONENTS = {
    "coverage_component",
    "roi_coverage_component",
    "information_component",
    "path_cost_component",
    "soft_risk_component",
    "fallback_component",
    "failure_component",
}


def _profile_path() -> Path:
    return REPO_ROOT / "configs" / "xunce_canonical_reward_guard_profile_v3.json"


def test_v3_profile_hash_is_stable_and_ignores_json_field_order(tmp_path: Path) -> None:
    from model_explorer.policy.canonical_reward import canonical_profile_hash, load_canonical_reward_profile

    profile_a = load_canonical_reward_profile(_profile_path())
    payload = json.loads(_profile_path().read_text(encoding="utf-8"))
    reordered = {
        "trajectory_guards": dict(reversed(list(payload["trajectory_guards"].items()))),
        "risk_policy": dict(reversed(list(payload["risk_policy"].items()))),
        "normalizers": dict(reversed(list(payload["normalizers"].items()))),
        "soft_reward_components": dict(reversed(list(payload["soft_reward_components"].items()))),
        "profile_version": payload["profile_version"],
        "profile_id": payload["profile_id"],
        "schema_version": payload["schema_version"],
    }
    path_b = tmp_path / "v3-reordered.json"
    path_b.write_text(json.dumps(reordered, ensure_ascii=False, indent=2), encoding="utf-8")

    profile_b = load_canonical_reward_profile(path_b)

    assert profile_a.profile_hash == profile_b.profile_hash
    assert canonical_profile_hash(profile_a) == canonical_profile_hash(profile_b)
    assert len(profile_a.profile_hash) == 64


def test_v3_component_set_uses_roi_and_soft_risk_names() -> None:
    from model_explorer.policy.canonical_reward import compute_canonical_reward_components, load_canonical_reward_profile

    result = compute_canonical_reward_components(
        {
            "coverage_gain_rate": 2.0,
            "roi_coverage": 25000.0,
            "information_gain": 1.0,
            "path_cost_m": 100.0,
            "soft_risk_exposure": 25.0,
            "fallback_used": True,
            "failure": True,
        },
        load_canonical_reward_profile(_profile_path()),
    )

    assert set(result.components) == EXPECTED_V3_COMPONENTS
    assert "risk_component" not in result.components
    assert "valuable_coverage_component" not in result.components
    assert result.components["roi_coverage_component"] > 0.0
    assert result.components["soft_risk_component"] < 0.0


def test_v3_soft_risk_is_tiny_when_path_cost_already_includes_risk_proxy() -> None:
    from model_explorer.policy.canonical_reward import compute_canonical_reward_components, load_canonical_reward_profile

    profile = load_canonical_reward_profile(_profile_path())
    result = compute_canonical_reward_components(
        {
            "path_cost_m": 100.0,
            "soft_risk_exposure": 25.0,
        },
        profile,
    )

    assert profile.risk_policy["path_cost_includes_risk_proxy"] is True
    assert profile.risk_policy["soft_risk_component_mode"] == "audit_weighted_tiny"
    assert abs(result.components["soft_risk_component"]) < abs(result.components["path_cost_component"]) / 10.0


def test_v3_hard_risk_violation_cannot_become_positive_reward() -> None:
    from model_explorer.policy.canonical_reward import compute_canonical_reward_components, load_canonical_reward_profile

    result = compute_canonical_reward_components(
        {
            "coverage_gain_rate": 10.0,
            "roi_coverage": 25000.0,
            "information_gain": 1.0,
            "path_cost_m": 0.0,
            "soft_risk_exposure": 0.0,
            "path_allowed_by_risk": False,
            "hard_risk_violation_count": 1,
        },
        load_canonical_reward_profile(_profile_path()),
    )

    assert "hard_risk_component" not in result.components
    assert result.components["failure_component"] < 0.0
    assert result.reward <= 0.0


def test_compute_step_reward_passes_v3_metrics_without_hard_risk_positive_reward() -> None:
    from model_explorer.core.interfaces import GoalCandidate
    from model_explorer.policy.canonical_reward import load_canonical_reward_profile
    from model_explorer.policy.reward import compute_step_reward

    goal = GoalCandidate(
        cell=(1, 2),
        utility=1.0,
        reachable=True,
        experimental={
            "path_cost": 100.0,
            "risk": 25.0,
            "roi_coverage": 25000.0,
            "soft_risk_exposure": 25.0,
            "path_allowed_by_risk": False,
            "hard_risk_violation_count": 1,
        },
    )

    reward = compute_step_reward(
        goal,
        {"coverage_rate_delta": 10.0, "information_gain": 1.0},
        canonical_profile=load_canonical_reward_profile(_profile_path()),
    )

    assert set(reward.reward_components or {}) == EXPECTED_V3_COMPONENTS
    assert reward.reward <= 0.0
    assert reward.reward_components["failure_component"] < 0.0
