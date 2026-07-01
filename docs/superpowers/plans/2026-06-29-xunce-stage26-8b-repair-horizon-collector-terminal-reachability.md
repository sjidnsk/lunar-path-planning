# Stage26.8B Repair Horizon Collector Terminal Reachability Implementation Plan

## Summary

Stage26.8B repairs the H16 seed0 blocker found by Stage26.8A. The failure is not a synthetic sidecar load failure: the H16 collector has matching source roots and valid synthetic transition provenance. The real terminal condition is that, at the final attempted step, `action_mask` and `hard_risk_clean_mask` remain true while Hybrid A* reachability removes every candidate from `sampling_mask`.

This stage only repairs collector terminal semantics and Stage26.1 routing. It does not run PPO update, trajectory eval, checkpoint publication, default policy replacement, executor connection, or canary traffic.

## Implementation Tasks

- Update Stage21.1 collector so a no-sampling terminal row records action, hard-risk, Hybrid A*, and final sampling masks plus true-count diagnostics.
- Split no-sampling reasons into action-mask, hard-risk-clean, Hybrid-reachable, and unknown categories.
- Treat `no_hybrid_reachable_candidate_terminal` as non-blocking only when the collector already has enough trainable rows and has no safety/path fallback violations.
- Update Stage26.1 wrapper so a loaded synthetic source with Stage21.1 failure is not misreported as `collector_synthetic_map_binding`.
- Add Stage26.8B runner/config/tests/registry entry. The runner reuses the Stage26.8A H16 seed0 Stage26.1 config, reruns Stage26.1 only, and compares H12/H16 collector evidence.

## Verification

```powershell
python -m pytest tests\test_xunce_stage26_8b_repair_horizon_collector_terminal_reachability.py tests\test_xunce_stage26_8a_expand_seed_or_horizon_budget.py tests\test_xunce_stage26_1_synthetic_terrain_collector_smoke.py tests\test_xunce_stage21_1_on_policy_ppo_rollout_collector.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-8b
python -m py_compile scripts\run_xunce_stage26_8b_repair_horizon_collector_terminal_reachability.py scripts\run_xunce_stage26_1_synthetic_terrain_collector_smoke.py scripts\run_xunce_stage21_1_on_policy_ppo_rollout_collector.py
python scripts\run_stage.py --stage xunce-stage26-8b-repair-horizon-collector-terminal-reachability --dry-run
python scripts\run_stage.py --stage xunce-stage26-8b-repair-horizon-collector-terminal-reachability
```

## Acceptance

- Repaired H16 seed0 Stage26.1 passes and writes reward/batch rows.
- Stage21.1 rejection diagnostics show Hybrid A* reachability, not action mask or hard-risk, caused the terminal no-sampling condition.
- Stage26.1 no longer misclassifies loaded-source Stage21.1 terminal failures as synthetic sidecar binding failures.
- All boundary fields remain false/0.
- Next route is `resume_stage26_8a_from_h16_h20`.
