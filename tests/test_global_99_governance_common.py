from scripts.global_99_governance_common import (
    GLOBAL_99_BOUNDARY_DEFAULTS,
    global_99_boundary_defaults,
    merge_global_99_boundary_defaults,
)


def test_global_99_boundary_defaults_include_online_canary_and_release_guards() -> None:
    defaults = global_99_boundary_defaults()

    assert defaults == GLOBAL_99_BOUNDARY_DEFAULTS
    assert defaults["publishes_checkpoint"] is False
    assert defaults["replaces_default_policy"] is False
    assert defaults["connects_real_executor"] is False
    assert defaults["starts_online_canary"] is False
    assert defaults["runs_new_ppo_update"] is False
    assert defaults["modifies_network"] is False
    assert defaults["modifies_action_space"] is False
    assert defaults["modifies_default_astar"] is False


def test_merge_global_99_boundary_defaults_preserves_explicit_values() -> None:
    summary = {"status": "failed", "publishes_checkpoint": True}

    merged = merge_global_99_boundary_defaults(summary)

    assert merged["publishes_checkpoint"] is True
    assert merged["starts_online_canary"] is False
    assert summary == {"status": "failed", "publishes_checkpoint": True}
