# Windows and Ubuntu Compatibility Audit

## Current State

- Core Python package requirements are already lightweight and Python 3.12 based.
- `model-explorer` requires Python `>=3.12,<3.13`; both Windows and Ubuntu should use Python 3.12.
- Drake imports are isolated under `path-planner/src/path_planner/drake_backend/` and tests use the `drake` marker or `pytest.importorskip("pydrake")`.
- The main historical blocker was execution orchestration: many `.sh` wrappers existed, while Windows had no root `.ps1` equivalent.

## Implemented Platform Boundary

- `scripts/bootstrap_env.py` is the canonical Conda bootstrap entrypoint.
- `scripts/run_stage.py` plus `configs/stage_registry.json` provide the retained stage-runner surface.
- `scripts/run_platform_smoke.py` defines lightweight smoke profiles.
- `scripts/run_platform_validation_matrix.py` defines full Windows/Ubuntu non-Drake and Ubuntu Drake validation profiles, writes a JSON summary, and keeps bootstrap real installation opt-in.
- `scripts/run_stage.ps1` and `scripts/bootstrap_windows_conda.ps1` are PowerShell convenience wrappers.

## Supported Registry Surface

- `xunce-shadow-replay-validation`
- `xunce-sandbox-candidate-preflight`
- `xunce-release-governance-gate`
- `xunce-high-fidelity-real-map-roi-expansion`
- `xunce-high-fidelity-real-map-comparison`
- `policy-training-readiness-review`
- `policy-gated-sequential-canary-rollout`
- `guarded-ppo-rollout-pilot`
- `iterative-ppo-mini-loop-stability`
- `quasi-real-guarded-ppo-stability-replay`

All supported registry entries point to Python scripts and run through the
current interpreter. They must dry-run without `bash`, `python3`, `.sh`, or
machine-specific `/home/kai` paths.

## Remaining Legacy Surface

- Many historical `.sh` files remain as Ubuntu convenience wrappers.
- Some legacy research orchestration scripts outside the supported registry may still chain shell-only wrappers and are not part of the Windows-supported profile.
- The compatibility test suite prevents platform-supported Python paths from adding new direct `subprocess.run(["bash", ...])` dependencies.

## Drake Policy

Windows support does not include Drake. This is intentional for the current
profile. Drake validation belongs to `ubuntu-drake` and only runs when
`pydrake` imports successfully.

## CI Matrix

`.github/workflows/platform-compatibility.yml` defines the repository CI
boundary:

- `windows-latest` non-Drake, Python 3.12.
- `ubuntu-latest` non-Drake, Python 3.12.
- Optional `ubuntu-drake` via manual `workflow_dispatch` only.

The automatic jobs run platform compatibility tests, Xunce Stage 15-18 unit
tests, supported stage dry-runs, platform matrix dry-run, and visual-workbench
frontend tests. CI must not download large LOLA products, run real PPO, publish
checkpoints, replace default policy, connect executors, or start online canary.

## Recommended Migration Rule

When adding a new runnable stage:

1. Put the implementation in a Python script or package module.
2. Add the stage to `configs/stage_registry.json` if it is part of a governed pipeline.
3. Keep `.sh` or `.ps1` files as thin wrappers only.
4. Use `sys.executable` and `subprocess.run([...], shell=False)`.
5. Keep large outputs under `outputs/` or external data roots such as `D:\CodexDownloads\lunar-path-planning`.
