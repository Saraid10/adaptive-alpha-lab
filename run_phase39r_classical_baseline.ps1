param(
    [int]$MaxFolds = 0,
    [string]$RunName = "phase39r_classical_full",
    [string]$OutputDir = "models",
    [string]$OutputPrefix = "crypto20_repaired_classical_",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$PythonExe = Join-Path $PSScriptRoot "env\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$Arguments = @(
    "src\repaired_classical_baseline.py",
    "--run-name", $RunName,
    "--output-dir", $OutputDir,
    "--output-prefix", $OutputPrefix
)
if ($MaxFolds -gt 0) {
    $Arguments += @("--max-folds", $MaxFolds)
}
if ($Resume) {
    $Arguments += "--resume"
}

& $PythonExe @Arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
