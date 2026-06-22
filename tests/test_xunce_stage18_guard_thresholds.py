import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
MODEL_EXPLORER_SRC = str(REPO_ROOT / "model-explorer" / "src")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
if MODEL_EXPLORER_SRC not in sys.path:
    sys.path.insert(0, MODEL_EXPLORER_SRC)


def test_v2_thresholds_preserve_candidate_risk_guard_semantics() -> None:
    from model_explorer.policy.canonical_reward import load_canonical_reward_profile
    from scripts.xunce_stage18_guard_thresholds import stage18_guard_thresholds

    profile = load_canonical_reward_profile(REPO_ROOT / "configs" / "xunce_canonical_reward_guard_profile_v2.json")

    thresholds = stage18_guard_thresholds(profile)

    assert thresholds["profile_version"] == "v2"
    assert thresholds["min_coverage_delta_cells"] == 1.0
    assert thresholds["max_path_cost_delta_m"] == 20.0
    assert thresholds["max_risk_delta"] == 0.5
    assert thresholds["max_risk_cost_weighted_delta"] == 25.0
    assert thresholds["max_soft_risk_exposure_delta"] == 25.0
    assert thresholds["max_hard_risk_violation_count"] == 0.0
    assert thresholds["risk_delta_hard_gate_enabled"] is True
    assert thresholds["candidate_level_risk_delta_guard_is_diagnostic_only"] is False
    assert thresholds["coverage_gain_per_path_cost_delta_mode"] == "audit_only"


def test_v3_thresholds_do_not_require_v2_risk_guard_keys() -> None:
    from model_explorer.policy.canonical_reward import load_canonical_reward_profile
    from scripts.xunce_stage18_guard_thresholds import stage18_guard_thresholds

    profile = load_canonical_reward_profile(REPO_ROOT / "configs" / "xunce_canonical_reward_guard_profile_v3.json")

    thresholds = stage18_guard_thresholds(profile)

    assert thresholds["profile_id"] == "xunce-coverage-cost-risk-boundary-v3"
    assert thresholds["profile_version"] == "v3"
    assert thresholds["min_coverage_delta_cells"] == 1.0
    assert thresholds["max_path_cost_delta_m"] == 20.0
    assert thresholds["min_coverage_per_100m_delta"] == 0.0
    assert thresholds["max_soft_risk_exposure_delta"] == 25.0
    assert thresholds["max_hard_risk_violation_count"] == 0.0
    assert thresholds["max_risk_delta"] is None
    assert thresholds["max_risk_cost_weighted_delta"] == 25.0
    assert thresholds["risk_delta_hard_gate_enabled"] is False
    assert thresholds["candidate_level_risk_delta_guard_is_diagnostic_only"] is True
    assert thresholds["coverage_gain_per_path_cost_delta_mode"] == "audit_only"


def test_threshold_payload_can_be_embedded_in_stage18_artifacts() -> None:
    from model_explorer.policy.canonical_reward import load_canonical_reward_profile
    from scripts.xunce_stage18_guard_thresholds import stage18_guard_thresholds

    profile = load_canonical_reward_profile(REPO_ROOT / "configs" / "xunce_canonical_reward_guard_profile_v3.json")
    thresholds = stage18_guard_thresholds(profile)

    artifact_payload = {
        "canonical_guard_thresholds": thresholds.to_artifact_dict(),
        "profile_id": thresholds.profile_id,
        "profile_version": thresholds.profile_version,
        "profile_hash": thresholds.profile_hash,
    }

    assert artifact_payload["canonical_guard_thresholds"]["risk_delta_hard_gate_enabled"] is False
    assert artifact_payload["canonical_guard_thresholds"]["candidate_level_risk_delta_guard_is_diagnostic_only"] is True
    assert artifact_payload["canonical_guard_thresholds"]["max_soft_risk_exposure_delta"] == 25.0
