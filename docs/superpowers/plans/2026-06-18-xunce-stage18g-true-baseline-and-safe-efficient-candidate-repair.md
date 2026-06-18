# Xunce Stage 18G True Baseline Binding and Safe-Efficient Candidate Repair

## Summary

Stage 18F.1 showed that current candidates contain higher-coverage actions, but
no action is both higher coverage and non-regressive on path cost, risk, and
cost-adjusted efficiency. It also found fallback incumbent selection. Stage 18G
therefore repairs evaluation prerequisites before any actor, critic, PPO, or
network-complexity work.

## Stages

- `xunce-true-incumbent-selection-binding`
  - Binds true incumbent and Xunce checkpoint selections from Stage 18B model
    inference JSONL to Stage 18E candidate scenarios.
  - Fails on missing inference rows, non-true inference, proxy selection, or
    candidate cell mismatch.
- `xunce-safe-efficient-opportunity-root-cause-audit`
  - Audits why safe-efficient opportunities are missing across cost, risk,
    efficiency, revisit, incumbent dominance, and ROI/map spread.
- `xunce-safe-efficient-candidate-repair`
  - Marks path-feedback-validated safe-efficient candidates and emits
    diagnostic-only interpolation proposals.
  - Unvalidated proposals are `proposal_only=true` and never count as positive
    safe-efficient opportunities.

## Decision Rules

- Binding missing: `run_true_incumbent_selection_binding`
- Candidate alignment mismatch: `repair_stage18e_candidate_alignment`
- No validated repaired candidates: `repair_path_feedback_candidate_validation`
- Safe-efficient opportunity still zero: `expand_roi_or_map_complexity`
- Repair passed: rerun Stage 18F.1, Stage 18F, then Stage 18C-v2
- Oracle separable but Xunce not better: `train_coverage_cost_critic_or_adapter_iteration`

## Validation

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest `
  tests\test_xunce_true_incumbent_selection_binding.py `
  tests\test_xunce_safe_efficient_opportunity_root_cause_audit.py `
  tests\test_xunce_safe_efficient_candidate_repair.py `
  tests\test_xunce_cost_efficient_coverage_opportunity_refinement.py `
  tests\test_xunce_oracle_separability_benchmark.py `
  tests\test_platform_stage_runner.py `
  -q

D:\conda_envs\lunar-explorer\python.exe scripts\run_stage.py --stage xunce-true-incumbent-selection-binding --dry-run
D:\conda_envs\lunar-explorer\python.exe scripts\run_stage.py --stage xunce-safe-efficient-opportunity-root-cause-audit --dry-run
D:\conda_envs\lunar-explorer\python.exe scripts\run_stage.py --stage xunce-safe-efficient-candidate-repair --dry-run
```

## Boundaries

Stage 18G does not train actor/critic models, run PPO, publish checkpoint,
replace default policy, connect executor, start online canary, or download new
map products.

