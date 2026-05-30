#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${LUNAR_EXPLORER_ENV_NAME:-lunar-explorer}"
ENV_PREFIX="${LUNAR_EXPLORER_ENV_PREFIX:-}"
PYTHON_SPEC="python=3.12"
PYTHON_VERSION_CHECK="import sys; assert sys.version_info[:2] == (3, 12), sys.version; print(sys.version)"

DRY_RUN=0
RUN_VALIDATION=0
INSTALL_EDITABLE=0
WITH_TRAINING=0
SKIP_SUBMODULES=0

usage() {
  cat <<'USAGE'
Usage: bash scripts/bootstrap_ubuntu_conda.sh [options]

Create or update the shared Ubuntu Conda environment for the lunar-path-planning parent repository.

Options:
  --conda PATH          Conda-compatible executable to use. Default: conda
  --env-name NAME       Named Conda environment. Default: lunar-explorer
  --env-prefix PATH     Prefix-based Conda environment path. Overrides --env-name.
  --install-editable    Install the three submodules as editable Python packages.
  --with-training       Install model-explorer training dependency extras, including PyTorch.
  --run-validation      Run the three submodule test suites after setup.
  --skip-submodules     Do not run git submodule update.
  --dry-run             Print commands without executing them.
  -h, --help            Show this help.

Default behavior initializes submodules, creates or updates a Python 3.12 Conda environment,
checks Python version, and runs import smoke checks without installing local editable packages.
USAGE
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "$value" ]]; then
    echo "Missing value for $option" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda)
      require_value "$1" "${2:-}"
      CONDA_BIN="$2"
      shift 2
      ;;
    --env-name)
      require_value "$1" "${2:-}"
      ENV_NAME="$2"
      shift 2
      ;;
    --env-prefix)
      require_value "$1" "${2:-}"
      ENV_PREFIX="$2"
      shift 2
      ;;
    --install-editable)
      INSTALL_EDITABLE=1
      shift
      ;;
    --with-training)
      WITH_TRAINING=1
      shift
      ;;
    --run-validation)
      RUN_VALIDATION=1
      shift
      ;;
    --skip-submodules)
      SKIP_SUBMODULES=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENVIRONMENT_FILE="$REPO_ROOT/environment.yml"
MODULES=(path-planner model-explorer dev-platform-constraints)

if [[ -n "$ENV_PREFIX" ]]; then
  CONDA_TARGET_ARGS=(-p "$ENV_PREFIX")
  CONDA_RUN_ARGS=(-p "$ENV_PREFIX")
  ENV_TARGET_DISPLAY="-p \"$ENV_PREFIX\""
  ACTIVATE_TARGET="$ENV_PREFIX"
else
  CONDA_TARGET_ARGS=(-n "$ENV_NAME")
  CONDA_RUN_ARGS=(-n "$ENV_NAME")
  ENV_TARGET_DISPLAY="-n \"$ENV_NAME\""
  ACTIVATE_TARGET="$ENV_NAME"
fi

run_step() {
  local display="$1"
  shift
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY RUN] $display"
    return
  fi

  echo "==> $display"
  "$@"
}

quote_display_arg() {
  local arg="$1"
  if [[ "$arg" =~ ^[A-Za-z0-9_./:=+-]+$ ]]; then
    printf '%s' "$arg"
  else
    printf '"%s"' "${arg//\"/\\\"}"
  fi
}

format_display_command() {
  local formatted=""
  local arg
  for arg in "$@"; do
    if [[ -n "$formatted" ]]; then
      formatted+=" "
    fi
    formatted+="$(quote_display_arg "$arg")"
  done
  printf '%s' "$formatted"
}

run_in_module() {
  local module="$1"
  shift
  local module_root="$REPO_ROOT/$module"
  local source_path="$module_root/src"
  local command_display
  command_display="$(format_display_command "$@")"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY RUN] (cd $module && PYTHONPATH=src $CONDA_BIN run $ENV_TARGET_DISPLAY $command_display)"
    return
  fi

  (
    cd "$module_root"
    env PYTHONPATH="$source_path" "$CONDA_BIN" run "${CONDA_RUN_ARGS[@]}" "$@"
  )
}

conda_env_exists() {
  "$CONDA_BIN" run "${CONDA_RUN_ARGS[@]}" python --version >/dev/null 2>&1
}

create_or_update_env() {
  if conda_env_exists; then
    "$CONDA_BIN" env update "${CONDA_TARGET_ARGS[@]}" -f "$ENVIRONMENT_FILE" --prune
  else
    "$CONDA_BIN" env create "${CONDA_TARGET_ARGS[@]}" -f "$ENVIRONMENT_FILE"
  fi
}

ensure_submodules() {
  if [[ "$SKIP_SUBMODULES" -eq 1 ]]; then
    return
  fi

  if [[ -d "$REPO_ROOT/.git" ]]; then
    run_step \
      "git submodule update --init --recursive ${MODULES[*]}" \
      git -C "$REPO_ROOT" submodule update --init --recursive "${MODULES[@]}"
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    return
  fi

  for module in "${MODULES[@]}"; do
    if [[ ! -d "$REPO_ROOT/$module/src" ]]; then
      echo "Missing initialized submodule: $module" >&2
      echo "Run: git submodule update --init --recursive ${MODULES[*]}" >&2
      exit 1
    fi
  done
}

install_editable_packages() {
  local model_spec="$REPO_ROOT/model-explorer"
  if [[ "$WITH_TRAINING" -eq 1 ]]; then
    model_spec="$model_spec[training]"
  fi

  "$CONDA_BIN" run "${CONDA_RUN_ARGS[@]}" python -m pip install \
    -e "$REPO_ROOT/path-planner" \
    -e "$REPO_ROOT/dev-platform-constraints" \
    -e "$model_spec"
}

install_training_runtime_only() {
  "$CONDA_BIN" run "${CONDA_RUN_ARGS[@]}" python -m pip install "torch>=2.0"
}

run_import_smoke() {
  run_in_module path-planner python -c "import path_planner; print('path_planner import ok')"
  run_in_module model-explorer python -c "import model_explorer; print('model_explorer import ok')"
  run_in_module dev-platform-constraints python -c "import dev_platform_constraints; print('dev_platform_constraints import ok')"
}

run_validation_suites() {
  run_in_module path-planner python -m pytest
  run_in_module model-explorer python -m unittest discover -s tests -v
  run_in_module dev-platform-constraints python -m unittest discover -s tests
  run_in_module model-explorer python -m model_explorer verify
}

cat <<INFO
Repository: $REPO_ROOT
Conda executable: $CONDA_BIN
Conda environment target: $ENV_TARGET_DISPLAY
Environment file: $ENVIRONMENT_FILE
Submodules: ${MODULES[*]}
INFO

ensure_submodules

run_step \
  "$CONDA_BIN env create/update $ENV_TARGET_DISPLAY -f \"$ENVIRONMENT_FILE\"" \
  create_or_update_env

run_step \
  "$CONDA_BIN install $ENV_TARGET_DISPLAY -c conda-forge $PYTHON_SPEC --yes" \
  "$CONDA_BIN" install "${CONDA_TARGET_ARGS[@]}" -c conda-forge "$PYTHON_SPEC" --yes

run_step \
  "$CONDA_BIN run $ENV_TARGET_DISPLAY python -c \"$PYTHON_VERSION_CHECK\"" \
  "$CONDA_BIN" run "${CONDA_RUN_ARGS[@]}" python -c "$PYTHON_VERSION_CHECK"

if [[ "$INSTALL_EDITABLE" -eq 1 ]]; then
  editable_model_display="model-explorer"
  if [[ "$WITH_TRAINING" -eq 1 ]]; then
    editable_model_display="model-explorer[training]"
  fi

  run_step \
    "$CONDA_BIN run $ENV_TARGET_DISPLAY python -m pip install -e path-planner -e dev-platform-constraints -e $editable_model_display" \
    install_editable_packages
elif [[ "$WITH_TRAINING" -eq 1 ]]; then
  run_step \
    "$CONDA_BIN run $ENV_TARGET_DISPLAY python -m pip install \"torch>=2.0\"" \
    install_training_runtime_only
fi

run_import_smoke

if [[ "$RUN_VALIDATION" -eq 1 ]]; then
  run_validation_suites
fi

cat <<NEXT

Next commands:
  source "\$($CONDA_BIN info --base)/etc/profile.d/conda.sh"
  conda activate $ACTIVATE_TARGET

Source-tree commands use per-module PYTHONPATH, for example:
  cd path-planner && PYTHONPATH=src python -m pytest
  cd model-explorer && PYTHONPATH=src python -m model_explorer verify
  cd dev-platform-constraints && PYTHONPATH=src python scripts/run_minimal_closure.py
NEXT
