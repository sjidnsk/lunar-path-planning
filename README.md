# lunar-path-planning

System-level repository for lunar rover autonomous exploration and path planning.

This parent repository coordinates three Git submodules:

- `path-planner`
- `model-explorer`
- `dev-platform-constraints`

`a_gcs_ws-2.0.1` is intentionally excluded from the parent repository.

## Ubuntu One-Click Conda Setup

Target environment:

- Ubuntu 24.04
- Conda-compatible runtime, such as Miniconda, Mambaforge, or micromamba with a `conda` compatible command
- Python 3.12

Fresh clone:

```bash
git clone --recurse-submodules https://github.com/sjidnsk/lunar-path-planning.git
cd lunar-path-planning
bash scripts/bootstrap_ubuntu_conda.sh --run-validation
```

If the parent repository was cloned without submodules, the script runs:

```bash
git submodule update --init --recursive path-planner model-explorer dev-platform-constraints
```

After setup:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lunar-explorer
```

The default setup creates or updates a named Conda environment `lunar-explorer`,
checks Python 3.12, and runs import smoke checks for all three modules.
`--run-validation` additionally runs the module test suites and
`model_explorer verify`.

Useful options:

```bash
bash scripts/bootstrap_ubuntu_conda.sh --dry-run
bash scripts/bootstrap_ubuntu_conda.sh --env-prefix "$HOME/conda_envs/lunar-explorer"
bash scripts/bootstrap_ubuntu_conda.sh --conda mamba --run-validation
bash scripts/bootstrap_ubuntu_conda.sh --install-editable
bash scripts/bootstrap_ubuntu_conda.sh --with-training
```

By default, local packages are exposed through per-module `PYTHONPATH` during
validation and are not installed editable. Use `--install-editable` only when
you want package entry points installed into the Conda environment. Use
`--with-training` only when `model-explorer` training paths are needed, because
it installs PyTorch.
