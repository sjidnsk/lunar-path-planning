# Windows and Ubuntu Setup

This project now treats Python as the canonical cross-platform execution layer.
Bash and PowerShell scripts are convenience wrappers only.

## Supported Profiles

- Windows non-Drake: Conda Python 3.12, core offline pipelines, Xunce Stage 15-18, path-feedback batch, supported policy/rollout dry-run orchestration, model-explorer, and visual-workbench backend/frontend checks.
- Ubuntu non-Drake: same as Windows, using a named Conda environment by default.
- Ubuntu Drake optional: adds `pydrake` IRIS/GCS backend tests when the runtime provides Drake.

Windows does not need Drake for the supported profile. `pydrake` tests are an
Ubuntu/Linux optional profile and should not block Windows validation.

## Bootstrap

Windows PowerShell:

```powershell
python scripts\bootstrap_env.py --platform windows --dry-run
python scripts\bootstrap_env.py --platform windows --install-editable --with-training --with-visual-workbench --run-validation
```

Windows defaults:

- Conda environment: `D:\conda_envs\lunar-explorer`
- Download/data root: `D:\CodexDownloads\lunar-path-planning`

Ubuntu Bash:

```bash
python scripts/bootstrap_env.py --platform ubuntu --dry-run
python scripts/bootstrap_env.py --platform ubuntu --env-name lunar-explorer --install-editable --with-training --with-visual-workbench --run-validation
```

The legacy Ubuntu wrapper remains available:

```bash
bash scripts/bootstrap_ubuntu_conda.sh --run-validation
```

## Stage Runner

Use the Python stage runner first:

```bash
python scripts/run_stage.py --list
python scripts/run_stage.py --stage xunce-shadow-replay-validation --dry-run
python scripts/run_stage.py --stage xunce-high-fidelity-real-map-comparison --dry-run
python scripts/run_stage.py --stage guarded-ppo-rollout-pilot --dry-run
```

PowerShell wrapper:

```powershell
.\scripts\run_stage.ps1 --stage xunce-high-fidelity-real-map-comparison --dry-run
```

The runner reads `configs/stage_registry.json` and invokes registered Python
scripts with the current interpreter. It must not use Bash, `python3`, or
machine-specific `/home/kai/...` paths.

Supported registered stages currently include Xunce Stage 15-18,
path-feedback single and batch validation, policy training readiness review,
policy-gated sequential canary rollout, guarded PPO rollout pilot, iterative
PPO mini-loop stability, and quasi-real guarded PPO stability replay. The PPO
and rollout stages remain governed offline evidence paths; dry-run is the
recommended way to inspect their cross-platform command surface.

## Path Feedback

Use the Python entrypoint:

```bash
python scripts/run_path_feedback_validation.py --dry-run
python scripts/run_path_feedback_validation.py --scenario-set all --diagnostic-profile all --top-k 3
python scripts/run_batch_path_feedback_validation.py --matrix configs/path_feedback_batch_dataset_v1.json --dry-run
```

`scripts/run_path_feedback_validation.sh` remains a thin Ubuntu convenience
wrapper. It is not the canonical implementation.

## Tests

Windows non-Drake:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest tests/test_platform_stage_runner.py tests/test_bootstrap_env.py tests/test_path_feedback_windows_compat.py tests/test_no_new_python_bash_dependencies.py tests/test_platform_validation_matrix.py -q
python scripts\run_platform_validation_matrix.py --profile windows-non-drake --dry-run
```

Ubuntu non-Drake:

```bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest tests/test_platform_stage_runner.py tests/test_bootstrap_env.py tests/test_path_feedback_windows_compat.py tests/test_no_new_python_bash_dependencies.py tests/test_platform_validation_matrix.py -q
python scripts/run_platform_validation_matrix.py --profile ubuntu-non-drake --dry-run
```

Ubuntu Drake optional:

```bash
python -c "import pydrake; print('pydrake ok')"
python scripts/run_platform_validation_matrix.py --profile ubuntu-drake
```

Visual Workbench frontend checks:

```powershell
cd visual-workbench\web
npm test
npm run build
```

Ubuntu equivalent:

```bash
cd visual-workbench/web
npm test
npm run build
```

## Data and Caches

Large downloads, raw map products, model files, and generated exports should
stay outside tracked source. On Windows, default to
`D:\CodexDownloads\lunar-path-planning`. Do not write machine-specific absolute
paths into tracked config; use runtime arguments or local untracked overrides.
