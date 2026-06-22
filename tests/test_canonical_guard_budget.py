import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_EXPLORER_SRC = REPO_ROOT / "model-explorer" / "src"
if str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))


def _load_default_profile():
    from model_explorer.policy.canonical_reward import load_canonical_reward_profile

    return load_canonical_reward_profile(REPO_ROOT / "configs" / "xunce_canonical_reward_guard_profile_v2.json")


def _baseline() -> dict:
    return {
        "coverage_cells": 100.0,
        "path_cost_m": 100.0,
        "risk_proxy": 1.0,
        "risk_cost_weighted": 10.0,
        "coverage_per_100m": 50.0,
    }


def _candidate(**overrides) -> dict:
    payload = {
        "coverage_cells": 102.0,
        "path_cost_m": 119.0,
        "risk_proxy": 1.4,
        "risk_cost_weighted": 34.0,
        "coverage_per_100m": 50.0,
    }
    payload.update(overrides)
    return payload


def test_path_cost_and_risk_small_increases_pass_within_budget() -> None:
    from model_explorer.policy.canonical_reward import evaluate_canonical_guard

    result = evaluate_canonical_guard(_candidate(), _baseline(), _load_default_profile())

    assert result.passed is True
    assert result.failed_guards == []
    assert result.observed["path_cost_delta_m"] == 19.0
    assert result.observed["risk_delta"] == 0.4
    assert result.observed["risk_cost_weighted_delta"] == 24.0


def test_path_cost_over_budget_fails() -> None:
    from model_explorer.policy.canonical_reward import evaluate_canonical_guard

    result = evaluate_canonical_guard(_candidate(path_cost_m=121.0), _baseline(), _load_default_profile())

    assert result.passed is False
    assert "path_cost_budget_exceeded" in result.failed_guards


def test_risk_over_budget_fails() -> None:
    from model_explorer.policy.canonical_reward import evaluate_canonical_guard

    result = evaluate_canonical_guard(_candidate(risk_proxy=1.51), _baseline(), _load_default_profile())

    assert result.passed is False
    assert "risk_budget_exceeded" in result.failed_guards


def test_risk_cost_weighted_over_budget_fails() -> None:
    from model_explorer.policy.canonical_reward import evaluate_canonical_guard

    result = evaluate_canonical_guard(
        _candidate(risk_cost_weighted=36.0),
        _baseline(),
        _load_default_profile(),
    )

    assert result.passed is False
    assert "risk_cost_weighted_budget_exceeded" in result.failed_guards


def test_coverage_per_100m_regression_fails() -> None:
    from model_explorer.policy.canonical_reward import evaluate_canonical_guard

    result = evaluate_canonical_guard(_candidate(coverage_per_100m=49.9), _baseline(), _load_default_profile())

    assert result.passed is False
    assert "coverage_efficiency_regression" in result.failed_guards


def test_missing_guard_metrics_fail_instead_of_defaulting_to_zero() -> None:
    from model_explorer.policy.canonical_reward import evaluate_canonical_guard

    candidate = _candidate()
    candidate.pop("risk_proxy")
    result = evaluate_canonical_guard(candidate, _baseline(), _load_default_profile())

    assert result.passed is False
    assert "missing_guard_metric" in result.failed_guards
    assert "candidate.risk_proxy" in result.reason_codes
