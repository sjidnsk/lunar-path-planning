import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def _run(config_path: Path, output_root: Path):
    from scripts.run_xunce_stage21_0_pure_ppo_readiness_audit import (
        run_xunce_stage21_0_pure_ppo_readiness_audit,
    )

    return run_xunce_stage21_0_pure_ppo_readiness_audit(
        config_path=config_path,
        output_root=output_root,
        repo_root=REPO_ROOT,
    )


def test_stage21_0_audits_current_pure_ppo_readiness(tmp_path: Path) -> None:
    config = _write_config(tmp_path)

    summary = _run(config, tmp_path / "out")

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "implement_stage21_1_xunce_on_policy_ppo_rollout_collector"
    assert "xunce_full_network_masked_policy_value" in summary["existing_capabilities"]
    assert "generic_masked_ppo_loss" in summary["existing_capabilities"]
    assert "xunce_on_policy_stochastic_collector" in summary["missing_capabilities"]
    assert "coverage_first_ppo_reward_contract" in summary["missing_capabilities"]
    assert summary["stage20_oracle_imitation_role"] == "diagnostic_only_not_pure_ppo_gate"
    assert "missing_xunce_on_policy_stochastic_collector" in summary["stage21_1_blockers"]
    assert "missing_xunce_specific_ppo_batch_adapter" in summary["stage21_1_blockers"]
    assert summary["xunce_checkpoint_format_verdict"] == "xunce_checkpoint_format_readable_for_evaluation"
    assert (
        summary["high_fidelity_rollout_contract_verdict"]
        == "high_fidelity_rollout_contract_partial_requires_on_policy_stochastic_sampling"
    )
    assert summary["xunce_ppo_reuse_verdict"] == "generic_ppo_reusable_only_after_xunce_batch_adapter"
    assert summary["runs_new_ppo_update"] is False
    assert summary["publishes_checkpoint"] is False
    assert summary["capability_audit"].endswith("xunce-stage21-0-capability-audit.json")
    assert summary["routing"].endswith("xunce-stage21-0-next-stage-routing.json")
    assert summary["report"].endswith("xunce-stage21-0-report.md")
    assert summary["manifest"].endswith("xunce-stage21-0-manifest.json")
    assert (tmp_path / "out" / "xunce-stage21-0-capability-audit.json").is_file()
    assert (tmp_path / "out" / "xunce-stage21-0-report.md").is_file()
    report = (tmp_path / "out" / "xunce-stage21-0-report.md").read_text(encoding="utf-8")
    assert "Stage 21.1 Blockers" in report
    assert "high_fidelity_rollout_contract_verdict" in report


def test_stage21_0_boundary_flag_hard_fails(tmp_path: Path) -> None:
    config = _write_config(tmp_path, runs_new_ppo_update=True)

    summary = _run(config, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage21_0_pure_ppo_readiness_boundary_rejections"
    assert "runs_new_ppo_update" in summary["blocking_reason_codes"]
    assert summary["stage21_0_authorized"] is False


def test_stage21_0_boundary_field_must_be_boolean_false(tmp_path: Path) -> None:
    config = _write_config(tmp_path, publishes_checkpoint="false")

    summary = _run(config, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage21_0_pure_ppo_readiness_boundary_rejections"
    assert "publishes_checkpoint" in summary["blocking_reason_codes"]


def test_stage21_0_requires_v3_profile(tmp_path: Path) -> None:
    config = _write_config(tmp_path, canonical_reward_profile="configs/xunce_canonical_reward_guard_profile_v2.json")

    summary = _run(config, tmp_path / "out")

    assert summary["status"] == "failed"
    assert "stage21_0_requires_canonical_reward_profile_v3" in summary["blocking_reason_codes"]


def _write_config(tmp_path: Path, **overrides) -> Path:
    payload = {
        "schema_version": "xunce-stage21-0-pure-ppo-readiness-audit-config/v1",
        "stage_id": "xunce-stage21-0-pure-ppo-readiness-audit",
        "stage21_output_base": "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first",
        "canonical_reward_profile": "configs/xunce_canonical_reward_guard_profile_v3.json",
        "min_required_existing_capability_count": 6,
        "stage21_0_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage21_0_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
