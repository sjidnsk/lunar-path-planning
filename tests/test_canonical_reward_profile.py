import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_EXPLORER_SRC = REPO_ROOT / "model-explorer" / "src"
if str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))


def _profile_payload() -> dict:
    return {
        "schema_version": "xunce-canonical-reward-guard-profile/v2",
        "profile_id": "xunce-coverage-cost-risk-budget-v2",
        "profile_version": "v2",
        "soft_reward_components": {
            "coverage_weight": 1.0,
            "valuable_coverage_weight": 0.5,
            "information_weight": 0.25,
            "path_cost_weight": 0.20,
            "risk_weight": 0.05,
            "fallback_penalty": 1.0,
            "failure_penalty": 1.0,
        },
        "normalizers": {
            "coverage_gain_rate": 1.0,
            "valuable_coverage": 25000.0,
            "information_gain": 1.0,
            "path_cost_m": 100.0,
            "risk_proxy": 5.0,
        },
        "guards": {
            "min_coverage_delta_cells": 1.0,
            "max_acceptable_path_cost_delta_m": 20.0,
            "min_coverage_per_100m_delta": 0.0,
            "max_acceptable_risk_delta": 0.5,
            "max_acceptable_risk_cost_weighted_delta": 25.0,
            "coverage_gain_per_path_cost_delta_mode": "audit_only",
            "aggregation_policy": "scenario_worst_case",
        },
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_profile_hash_is_stable_and_ignores_json_field_order(tmp_path: Path) -> None:
    from model_explorer.policy.canonical_reward import canonical_profile_hash, load_canonical_reward_profile

    payload = _profile_payload()
    path_a = _write(tmp_path / "profile-a.json", payload)
    reordered = {
        "guards": dict(reversed(list(payload["guards"].items()))),
        "normalizers": dict(reversed(list(payload["normalizers"].items()))),
        "soft_reward_components": dict(reversed(list(payload["soft_reward_components"].items()))),
        "profile_version": payload["profile_version"],
        "profile_id": payload["profile_id"],
        "schema_version": payload["schema_version"],
    }
    path_b = tmp_path / "nested" / "profile-b.json"
    path_b.parent.mkdir(parents=True, exist_ok=True)
    path_b = _write(path_b, reordered)

    profile_a = load_canonical_reward_profile(path_a)
    profile_b = load_canonical_reward_profile(path_b)

    assert canonical_profile_hash(profile_a) == canonical_profile_hash(profile_b)
    assert profile_a.profile_hash == profile_b.profile_hash
    assert len(profile_a.profile_hash) == 64


def test_profile_hash_uses_content_not_path_time_git_or_output_root(tmp_path: Path) -> None:
    from model_explorer.policy.canonical_reward import canonical_profile_hash, load_canonical_reward_profile

    payload = _profile_payload()
    left = load_canonical_reward_profile(_write(tmp_path / "left.json", payload))
    right_path = tmp_path / "different-output-root" / "right.json"
    right_path.parent.mkdir(parents=True)
    right = load_canonical_reward_profile(_write(right_path, payload))

    assert canonical_profile_hash(left) == canonical_profile_hash(right)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.pop("guards"),
        lambda payload: payload["soft_reward_components"].__setitem__("unknown_weight", 1.0),
        lambda payload: payload["normalizers"].__setitem__("path_cost_m", 0.0),
        lambda payload: payload.__setitem__("generated_at", "2026-06-20T00:00:00Z"),
    ],
)
def test_profile_validation_rejects_missing_unknown_and_invalid_fields(tmp_path: Path, mutate) -> None:
    from model_explorer.policy.canonical_reward import load_canonical_reward_profile

    payload = _profile_payload()
    mutate(payload)
    with pytest.raises(ValueError):
        load_canonical_reward_profile(_write(tmp_path / "bad-profile.json", payload))
