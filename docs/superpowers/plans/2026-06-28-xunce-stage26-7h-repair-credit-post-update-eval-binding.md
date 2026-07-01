# Stage26.7H Repair Credit Post-Update Eval Binding

## Goal

Repair the Stage26.7G post-update eval binding blocker. Stage26.7G proved that the behavior-policy KL gate is fixed and stable checkpoints exist, but the best combo still fails Stage26.3 because Stage21.5 reports pre-policy selected Hybrid A* poses as unreachable and Stage26.3 mixes that with missing inference fields.

## Contract

- Keep PPO, reward, network, Hybrid A*, synthetic terrain, and candidate generation semantics unchanged.
- Reuse the Stage26.7G best stable combo; do not rerun Stage26.2 update.
- Stage26.3 model inference rows must distinguish missing Hybrid A* provenance from explicit `hybrid_astar_reachable=false` selected poses.
- Stage21.5 must route pre selected unreachable separately from post selected unreachable.
- Coverage efficiency is judged by `main_coverage_per_100m_delta`; Hybrid A* path-cost delta is diagnostic only.

## Implementation

- Add `scripts/run_xunce_stage26_7h_repair_credit_post_update_eval_binding.py`.
- Add `configs/xunce_stage26_7h_repair_credit_post_update_eval_binding_v1.json`.
- Add `tests/test_xunce_stage26_7h_repair_credit_post_update_eval_binding.py`.
- Update Stage26.3 synthetic action audit with explicit unreachable counters.
- Update Stage21.5 routing for pre/post unreachable selected poses.
- Register `xunce-stage26-7h-repair-credit-post-update-eval-binding`.

## Review Gates

1. Binding contract review: explicit unreachable is not counted as missing provenance, and unreachable candidates are not marked reachable.
2. Stage21.5 semantics review: pre unreachable and post unreachable produce different routes without weakening safety gates.
3. Full rerun review: Stage26.7H reports the exact next route and does not claim performance from partial diagnostics.

## Verification

```powershell
python -m pytest tests\test_xunce_stage26_7h_repair_credit_post_update_eval_binding.py tests\test_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py tests\test_xunce_stage21_5_post_update_offline_trajectory_evaluation.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-7h
python -m py_compile scripts\run_xunce_stage26_7h_repair_credit_post_update_eval_binding.py scripts\run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py scripts\run_xunce_stage21_5_post_update_offline_trajectory_evaluation.py scripts\run_xunce_high_fidelity_exploration_coverage_comparison.py
python scripts\run_stage.py --stage xunce-stage26-7h-repair-credit-post-update-eval-binding --dry-run
```
