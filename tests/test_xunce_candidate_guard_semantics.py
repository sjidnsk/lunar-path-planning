import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def _thresholds() -> dict:
    return {
        "min_coverage_delta_cells": 1.0,
        "max_path_cost_delta_m": 20.0,
        "max_risk_delta": 0.5,
        "max_risk_cost_weighted_delta": 25.0,
        "min_coverage_per_100m_delta": 0.0,
    }


def _paired(**overrides) -> dict:
    row = {
        "scenario_id": "s0",
        "step_index": 0,
        "candidate_set_hash": "hash-0",
        "covered_cells_hash": "covered-0",
        "incumbent_selected_action_index": 0,
        "incumbent_selected_expected_new_coverage_cell_count": 10.0,
        "incumbent_selected_path_cost": 10.0,
        "incumbent_selected_risk": 1.0,
    }
    row.update(overrides)
    return row


def _candidate(**overrides) -> dict:
    row = {
        "scenario_id": "s0",
        "step_index": 0,
        "candidate_set_hash": "hash-0",
        "covered_cells_hash": "covered-0",
        "candidate_index": 1,
        "action_mask_valid": True,
        "expected_new_coverage_cell_count": 12.0,
        "path_cost": 11.0,
        "risk": 1.2,
        "risk_cost_weighted": 13.2,
        "risk_source": "sidecar_path_cost_proxy/v1",
    }
    row.update(overrides)
    return row


def test_absolute_risk_above_threshold_passes_when_delta_is_inside_budget() -> None:
    from xunce_candidate_guard_semantics import candidate_guard_clean, candidate_guard_observation

    observation = candidate_guard_observation(_candidate(risk=1.2, risk_cost_weighted=13.2), _paired())

    assert observation["absolute_risk_proxy"] == 1.2
    assert observation["risk_delta"] == 0.19999999999999996
    assert candidate_guard_clean(observation, _thresholds()) is True


def test_risk_delta_over_budget_fails_even_when_absolute_threshold_is_not_used() -> None:
    from xunce_candidate_guard_semantics import (
        candidate_guard_clean,
        candidate_guard_failure_reasons,
        candidate_guard_observation,
    )

    observation = candidate_guard_observation(_candidate(risk=1.7, risk_cost_weighted=18.7), _paired())

    assert observation["risk_delta"] == 0.7
    assert candidate_guard_clean(observation, _thresholds()) is False
    assert "risk_budget_exceeded" in candidate_guard_failure_reasons(observation, _thresholds())


def test_risk_cost_weighted_delta_fails_independently() -> None:
    from xunce_candidate_guard_semantics import (
        candidate_guard_clean,
        candidate_guard_failure_reasons,
        candidate_guard_observation,
    )

    observation = candidate_guard_observation(_candidate(path_cost=30.0, risk=1.2, risk_cost_weighted=36.0), _paired())

    assert candidate_guard_clean(observation, _thresholds()) is False
    assert "risk_cost_weighted_budget_exceeded" in candidate_guard_failure_reasons(observation, _thresholds())


def test_coverage_per_100m_regression_fails() -> None:
    from xunce_candidate_guard_semantics import (
        candidate_guard_clean,
        candidate_guard_failure_reasons,
        candidate_guard_observation,
    )

    observation = candidate_guard_observation(
        _candidate(expected_new_coverage_cell_count=12.0, path_cost=20.0, risk=1.1, risk_cost_weighted=22.0),
        _paired(),
    )

    assert observation["coverage_delta_cells"] == 2.0
    assert observation["coverage_per_100m_delta"] < 0.0
    assert candidate_guard_clean(observation, _thresholds()) is False
    assert "coverage_per_100m_regression" in candidate_guard_failure_reasons(observation, _thresholds())


def test_missing_metrics_are_non_comparable_and_not_defaulted_to_zero() -> None:
    from xunce_candidate_guard_semantics import candidate_guard_clean, candidate_guard_observation

    candidate = _candidate()
    del candidate["risk"]

    observation = candidate_guard_observation(candidate, _paired())

    assert observation["comparable"] is False
    assert any(reason.startswith("missing_candidate_guard_metric") for reason in observation["reason_codes"])
    assert candidate_guard_clean(observation, _thresholds()) is False


def test_non_numeric_step_index_does_not_raise_in_paired_key() -> None:
    from xunce_candidate_guard_semantics import paired_decision_key

    assert paired_decision_key({"scenario_id": "s0", "step_index": "bad"}) == ("s0", -1, "", "")


def test_invalid_risk_proxy_alias_falls_back_to_legacy_risk_field() -> None:
    from xunce_candidate_guard_semantics import candidate_guard_clean, candidate_guard_observation

    observation = candidate_guard_observation(
        _candidate(risk_proxy=None, risk=1.2, risk_cost_weighted=13.2),
        _paired(),
    )

    assert observation["absolute_risk_proxy"] == 1.2
    assert candidate_guard_clean(observation, _thresholds()) is True


def test_missing_risk_cost_weighted_uses_path_cost_times_risk_fallback() -> None:
    from xunce_candidate_guard_semantics import candidate_guard_clean, candidate_guard_observation

    candidate = _candidate(risk=1.2)
    del candidate["risk_cost_weighted"]

    observation = candidate_guard_observation(candidate, _paired())

    assert observation["absolute_risk_cost_weighted"] == 13.2
    assert candidate_guard_clean(observation, _thresholds()) is True
