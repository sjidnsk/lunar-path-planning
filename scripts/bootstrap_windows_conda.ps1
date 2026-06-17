param(
    [string]$Conda = "conda",
    [string]$EnvPrefix = "D:\conda_envs\lunar-explorer",
    [switch]$InstallEditable,
    [switch]$WithTraining,
    [switch]$WithVisualWorkbench,
    [switch]$RunValidation,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Args = @(
    (Join-Path $ScriptDir "bootstrap_env.py"),
    "--platform", "windows",
    "--conda", $Conda,
    "--env-prefix", $EnvPrefix
)
if ($InstallEditable) { $Args += "--install-editable" }
if ($WithTraining) { $Args += "--with-training" }
if ($WithVisualWorkbench) { $Args += "--with-visual-workbench" }
if ($RunValidation) { $Args += "--run-validation" }
if ($DryRun) { $Args += "--dry-run" }

& $Python @Args
exit $LASTEXITCODE
