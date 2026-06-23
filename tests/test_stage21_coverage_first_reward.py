import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_EXPLORER_SRC = str(REPO_ROOT / "model-explorer" / "src")
if MODEL_EXPLORER_SRC not in sys.path:
    sys.path.insert(0, MODEL_EXPLORER_SRC)

EXPECTED_COMPONENTS = {
    "step_coverage_gain_component",
    "coverage_progress_component",
    "final_coverage_bonus_component",
    "success_99pct_bonus_component",
    "path_cost_component",
    "soft_risk_component",
    "failure_component",
    "hard_risk_component",
}


def _profile_path() -> Path:
    return REPO_ROOT / "configs" / "xunce_stage21_coverage_first_ppo_reward_profile_v1.json"


def _profile_v2_path() -> Path:
    return REPO_ROOT / "configs" / "xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json"


def test_stage21_coverage_first_profile_hash_is_stable(tmp_path: Path) -> None:
    from model_explorer.policy.coverage_first_reward import coverage_first_profile_hash, load_coverage_first_reward_profile

    profile_a = load_coverage_first_reward_profile(_profile_path())
    payload = json.loads(_profile_path().read_text(encoding="utf-8"))
    reordered = {
        "component_source_map": dict(reversed(list(payload["component_source_map"].items()))),
        "hard_risk_policy": dict(reversed(list(payload["hard_risk_policy"].items()))),
        "risk_policy": dict(reversed(list(payload["risk_policy"].items()))),
        "normalizers": dict(reversed(list(payload["normalizers"].items()))),
        "weights": dict(reversed(list(payload["weights"].items()))),
        "mission_budget_route_when_below_target": payload["mission_budget_route_when_below_target"],
        "horizon_steps": payload["horizon_steps"],
        "coverage_cap_rate": payload["coverage_cap_rate"],
        "target_final_coverage_rate": payload["target_final_coverage_rate"],
        "profile_version": payload["profile_version"],
        "profile_id": payload["profile_id"],
        "schema_version": payload["schema_version"],
    }
    path = tmp_path / "reordered.json"
    path.write_text(json.dumps(reordered, ensure_ascii=False, indent=2), encoding="utf-8")

    profile_b = load_coverage_first_reward_profile(path)

    assert profile_a.profile_hash == profile_b.profile_hash
    assert coverage_first_profile_hash(profile_a) == profile_a.profile_hash
    assert len(profile_a.profile_hash) == 64


def test_stage21_coverage_first_component_set_and_terminal_bonus() -> None:
    from model_explorer.policy.coverage_first_reward import compute_coverage_first_reward_components, load_coverage_first_reward_profile

    profile = load_coverage_first_reward_profile(_profile_path())
    non_terminal = compute_coverage_first_reward_components(
        {
            "coverage_rate_delta": 0.02,
            "coverage_progress_rate": 0.5,
            "final_coverage_rate": 0.5,
            "path_cost_m": 10.0,
            "soft_risk_exposure": 5.0,
            "done": False,
        },
        profile,
    )
    terminal = compute_coverage_first_reward_components(
        {
            "coverage_rate_delta": 0.02,
            "coverage_progress_rate": 0.995,
            "final_coverage_rate": 0.995,
            "path_cost_m": 10.0,
            "soft_risk_exposure": 5.0,
            "done": True,
        },
        profile,
    )

    assert set(non_terminal.components) == EXPECTED_COMPONENTS
    assert non_terminal.components["final_coverage_bonus_component"] == 0.0
    assert non_terminal.components["success_99pct_bonus_component"] == 0.0
    assert terminal.components["final_coverage_bonus_component"] > 0.0
    assert terminal.components["success_99pct_bonus_component"] > 0.0


def test_stage21_coverage_first_no_success_bonus_below_99pct() -> None:
    from model_explorer.policy.coverage_first_reward import compute_coverage_first_reward_components, load_coverage_first_reward_profile

    result = compute_coverage_first_reward_components(
        {
            "coverage_rate_delta": 0.0,
            "coverage_progress_rate": 0.98,
            "final_coverage_rate": 0.98,
            "path_cost_m": 0.0,
            "soft_risk_exposure": 0.0,
            "done": True,
        },
        load_coverage_first_reward_profile(_profile_path()),
    )

    assert result.components["success_99pct_bonus_component"] == 0.0
    assert "final_coverage_below_99pct_target" in result.reason_codes


def test_stage21_coverage_first_hard_risk_cannot_be_positive_reward() -> None:
    from model_explorer.policy.coverage_first_reward import compute_coverage_first_reward_components, load_coverage_first_reward_profile

    result = compute_coverage_first_reward_components(
        {
            "coverage_rate_delta": 1.0,
            "coverage_progress_rate": 1.0,
            "final_coverage_rate": 1.0,
            "path_cost_m": 0.0,
            "soft_risk_exposure": 0.0,
            "done": True,
            "hard_risk_flags": ["obstacle_crossing"],
        },
        load_coverage_first_reward_profile(_profile_path()),
    )

    assert result.trainable is False
    assert result.reward <= 0.0
    assert result.components["step_coverage_gain_component"] == 0.0
    assert result.components["success_99pct_bonus_component"] == 0.0
    assert result.components["hard_risk_component"] < 0.0
    assert "hard_risk_rejected" in result.reason_codes


def test_stage21_coverage_first_soft_risk_is_capped_when_path_cost_includes_risk_proxy() -> None:
    from model_explorer.policy.coverage_first_reward import compute_coverage_first_reward_components, load_coverage_first_reward_profile

    result = compute_coverage_first_reward_components(
        {
            "coverage_rate_delta": 0.0,
            "coverage_progress_rate": 0.1,
            "final_coverage_rate": 0.1,
            "path_cost_m": 100.0,
            "soft_risk_exposure": 100000.0,
            "done": False,
        },
        load_coverage_first_reward_profile(_profile_path()),
    )

    assert result.risk_deduplication_applied is True
    assert abs(result.components["soft_risk_component"]) <= abs(result.components["path_cost_component"]) * 0.1 + 1e-12


def test_stage21_coverage_constrained_v2_component_set_and_cpc_signal() -> None:
    from model_explorer.policy.coverage_first_reward import compute_coverage_first_reward_components, load_coverage_first_reward_profile

    profile = load_coverage_first_reward_profile(_profile_v2_path())
    result = compute_coverage_first_reward_components(
        {
            "coverage_rate_delta": 0.05,
            "coverage_progress_rate": 0.2,
            "final_coverage_rate": 0.2,
            "coverage_per_cost": 0.04,
            "path_cost_m": 25.0,
            "soft_risk_exposure": 1.0,
            "done": False,
        },
        profile,
    )

    assert set(result.components) == {
        "coverage_gain_component",
        "coverage_progress_component",
        "coverage_per_cost_component",
        "path_cost_component",
        "soft_risk_component",
        "final_coverage_bonus_component",
        "success_99pct_bonus_component",
        "failure_component",
        "hard_risk_component",
    }
    assert result.components["coverage_gain_component"] > 0.0
    assert result.components["coverage_per_cost_component"] > 0.0
    assert result.trainable is True


def test_stage21_coverage_constrained_v2_path_cost_cannot_overpower_coverage_below_target() -> None:
    from model_explorer.policy.coverage_first_reward import compute_coverage_first_reward_components, load_coverage_first_reward_profile

    profile = load_coverage_first_reward_profile(_profile_v2_path())
    result = compute_coverage_first_reward_components(
        {
            "coverage_rate_delta": 0.02,
            "coverage_progress_rate": 0.1,
            "final_coverage_rate": 0.1,
            "coverage_per_cost": 0.01,
            "path_cost_m": 10000.0,
            "soft_risk_exposure": 0.0,
            "done": False,
        },
        profile,
    )

    assert abs(result.components["path_cost_component"]) <= abs(result.components["coverage_gain_component"]) * 0.5 + 1e-12
    assert result.reward > 0.0

    tiny_coverage_low_cost = compute_coverage_first_reward_components(
        {
            "coverage_rate_delta": 0.001,
            "coverage_progress_rate": 0.1,
            "final_coverage_rate": 0.1,
            "coverage_per_cost": 1.0,
            "path_cost_m": 0.001,
            "soft_risk_exposure": 0.0,
            "done": False,
        },
        profile,
    )
    higher_coverage = compute_coverage_first_reward_components(
        {
            "coverage_rate_delta": 0.05,
            "coverage_progress_rate": 0.1,
            "final_coverage_rate": 0.1,
            "coverage_per_cost": 0.05,
            "path_cost_m": 1.0,
            "soft_risk_exposure": 0.0,
            "done": False,
        },
        profile,
    )

    assert higher_coverage.reward > tiny_coverage_low_cost.reward


def test_stage21_coverage_constrained_v2_hard_risk_rejected_before_reward() -> None:
    from model_explorer.policy.coverage_first_reward import compute_coverage_first_reward_components, load_coverage_first_reward_profile

    result = compute_coverage_first_reward_components(
        {
            "coverage_rate_delta": 0.5,
            "coverage_progress_rate": 0.5,
            "final_coverage_rate": 0.5,
            "coverage_per_cost": 1.0,
            "path_cost_m": 1.0,
            "soft_risk_exposure": 0.0,
            "done": False,
            "path_allowed_by_risk": False,
        },
        load_coverage_first_reward_profile(_profile_v2_path()),
    )

    assert result.trainable is False
    assert result.components["coverage_gain_component"] == 0.0
    assert result.components["coverage_per_cost_component"] == 0.0
    assert result.reward <= 0.0
