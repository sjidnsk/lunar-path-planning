# Windows + Ubuntu Platform Compatibility Phase 2-4

## Summary

This phase extends the Python-first execution layer beyond Xunce Stage 15-18 so
Windows and Ubuntu can share the same non-Drake offline validation profile.
Bash and PowerShell files remain wrappers; supported orchestration must run
through Python scripts and `sys.executable`.

## Implemented Scope

- Expand `configs/stage_registry.json` to include Xunce Stage 15-18,
  path-feedback validation, policy training readiness review, policy-gated
  sequential canary rollout, guarded PPO rollout pilot, iterative PPO
  mini-loop stability, and quasi-real guarded PPO stability replay.
- Replace high-priority Python-to-Bash calls in guarded rollout orchestration
  with direct Python script commands.
- Add `scripts/run_platform_validation_matrix.py` for Windows non-Drake,
  Ubuntu non-Drake, and optional Ubuntu Drake validation profiles.
- Keep real bootstrap installation opt-in; normal platform validation uses
  bootstrap dry-run unless `--real-bootstrap` is explicitly passed.
- Preserve all release and training boundaries: no checkpoint publication, no
  default policy replacement, no executor connection, and no online canary.

## Supported Entrypoints

```bash
python scripts/run_stage.py --list
python scripts/run_stage.py --stage xunce-high-fidelity-real-map-comparison --dry-run
python scripts/run_stage.py --stage guarded-ppo-rollout-pilot --dry-run
python scripts/run_platform_validation_matrix.py --profile ubuntu-non-drake --dry-run
```

Windows equivalent:

```powershell
python scripts\run_stage.py --stage guarded-ppo-rollout-pilot --dry-run
python scripts\run_platform_validation_matrix.py --profile windows-non-drake --dry-run
```

## Validation

```bash
python -m pytest \
  tests/test_platform_stage_runner.py \
  tests/test_path_feedback_windows_compat.py \
  tests/test_no_new_python_bash_dependencies.py \
  tests/test_policy_gated_sequential_canary_rollout.py \
  tests/test_guarded_ppo_rollout_pilot.py \
  tests/test_iterative_ppo_mini_loop_stability.py \
  tests/test_quasi_real_guarded_ppo_stability_replay.py \
  tests/test_platform_validation_matrix.py \
  -q
```

The platform matrix writes
`outputs/platform_validation/<profile>/platform-validation-matrix-summary.json`
on real execution. Dry-run mode does not create the output root.

## Legacy Boundary

The repository still contains many historical `.sh` wrappers. They are Ubuntu
convenience or legacy entrypoints, not the canonical implementation. New
supported stages must be Python-first, registered when governed, and validated
against adding direct Python-to-Bash dependencies.
