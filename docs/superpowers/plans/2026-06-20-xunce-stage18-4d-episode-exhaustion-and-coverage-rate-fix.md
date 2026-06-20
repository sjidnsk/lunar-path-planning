# Xunce Stage 18.4D Episode Exhaustion and Coverage Rate Fix

## Summary

This plan records the narrow Stage 18.4D fix for long dynamic rollouts:

- `no_valid_dynamic_candidates` is candidate generation exhaustion, not model inference failure.
- Dynamic exhaustion terminates an episode cleanly and does not count as mask violation, path planning failure, or true-inference failure.
- Coverage reporting now separates raw covered cells from capped percentage-style rates so long rollouts can exceed a legacy denominator without misleading percent interpretation.

## Scope

- Modify `scripts/run_xunce_high_fidelity_exploration_coverage_comparison.py`.
- Modify `scripts/xunce_stage18_pipeline.py`.
- Add tests for clean exhaustion, model inference failure split, and coverage saturation fields.
- Update README, architecture report, and topology-aware coverage policy spec.

## Non-Goals

- No dynamic NBV proposal changes.
- No A* batch validation changes.
- No model architecture, checkpoint, action space, default A*, executor, PPO, online canary, or release changes.

## Acceptance

- Dynamic candidate exhaustion writes a terminal step with `terminal_reason=candidate_generation_exhausted`.
- `candidate_generation_exhausted_count > 0` is diagnostic only.
- `model_inference_failure_count > 0` remains a hard blocker.
- `coverage_rate_raw`, `coverage_rate_capped`, `coverage_saturation_exceeded`, and `coverage_saturation_excess` are present in steps/episodes/summary.
- Pairwise comparison still uses raw `coverage_delta_cells` as the primary fact.

## Verification

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest `
  tests\test_xunce_high_fidelity_exploration_coverage_comparison.py `
  tests\test_xunce_stage18_research_evidence_pipeline.py `
  tests\test_xunce_dynamic_frontier_nbv.py `
  -q

D:\conda_envs\lunar-explorer\python.exe -m py_compile `
  scripts\run_xunce_high_fidelity_exploration_coverage_comparison.py `
  scripts\xunce_stage18_pipeline.py `
  scripts\xunce_dynamic_frontier_nbv.py

git diff --check
```
