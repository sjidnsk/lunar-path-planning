# Windows and Ubuntu Setup

This project now treats Python as the canonical cross-platform execution layer.
Bash and PowerShell scripts are convenience wrappers only.

## Supported Profiles

- Windows non-Drake: Conda Python 3.12, parent package, `path-planner` default grid A*, `dev-platform-constraints`, and Stage 6 configuration contracts.
- Ubuntu non-Drake: same as Windows, using a named Conda environment by default.
- Ubuntu Drake optional: adds `pydrake` IRIS/GCS backend tests when the runtime provides Drake.

Windows does not need Drake for the supported profile. `pydrake` tests are an
Ubuntu/Linux optional profile and should not block Windows validation.

## Bootstrap

Windows PowerShell:

```powershell
python scripts\bootstrap_env.py --platform windows --dry-run
python scripts\bootstrap_env.py --platform windows --install-editable --with-training --run-validation
```

Windows defaults:

- Conda environment: `D:\conda_envs\lunar-explorer`
- Download/data root: `D:\CodexDownloads\lunar-path-planning`

Ubuntu Bash:

```bash
python scripts/bootstrap_env.py --platform ubuntu --dry-run
python scripts/bootstrap_env.py --platform ubuntu --env-name lunar-explorer --install-editable --with-training --run-validation
```

The legacy Ubuntu wrapper remains available:

```bash
bash scripts/bootstrap_ubuntu_conda.sh --run-validation
```

## Retained Entry Points

Use the platform smoke or matrix dry-run to inspect the retained parent,
`path-planner`, and `dev-platform-constraints` command surface:

```bash
python scripts/run_platform_smoke.py --profile windows-non-drake --dry-run
python scripts/run_platform_validation_matrix.py --profile windows-non-drake --dry-run
```

Use `docs/ppo-highres-frontier-stage6.md` for Stage 6 and
`docs/xunce-midterm-dual-gate-runbook.md` for the G1/G2/G3 delivery chain.

## Tests

Windows non-Drake:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest tests/test_bootstrap_env.py tests/test_bootstrap_ubuntu_conda.py tests/test_platform_smoke.py tests/test_no_new_python_bash_dependencies.py tests/test_platform_validation_matrix.py tests/test_mainline_repository_surface.py tests/test_retired_routes_absent.py -q
python scripts\run_platform_validation_matrix.py --profile windows-non-drake --dry-run
```

Ubuntu non-Drake:

```bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest tests/test_bootstrap_env.py tests/test_bootstrap_ubuntu_conda.py tests/test_platform_smoke.py tests/test_no_new_python_bash_dependencies.py tests/test_platform_validation_matrix.py tests/test_mainline_repository_surface.py tests/test_retired_routes_absent.py -q
python scripts/run_platform_validation_matrix.py --profile ubuntu-non-drake --dry-run
```

Ubuntu Drake optional:

```bash
python -c "import pydrake; print('pydrake ok')"
python scripts/run_platform_validation_matrix.py --profile ubuntu-drake
```

## Data and Caches

Large downloads, raw map products, model files, and generated exports should
stay outside tracked source. On Windows, default to
`D:\CodexDownloads\lunar-path-planning`. Do not write machine-specific absolute
paths into tracked config; use runtime arguments or local untracked overrides.
