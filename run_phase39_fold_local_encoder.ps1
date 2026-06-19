param(
    [int]$Epochs = 30,
    [int]$BatchSize = 128,
    [int]$MaxWindows = 0,
    [int]$MaxFolds = 0
)

$ErrorActionPreference = "Stop"

$PythonExe = Join-Path $PSScriptRoot "env\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$Arguments = @(
    "src\fold_local_encoder_walkforward.py",
    "--universe", "crypto20",
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--max-windows", $MaxWindows,
    "--output-prefix", "crypto20_fold_local_"
)

if ($MaxFolds -gt 0) {
    $Arguments += @("--max-folds", $MaxFolds)
}

& $PythonExe @Arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
