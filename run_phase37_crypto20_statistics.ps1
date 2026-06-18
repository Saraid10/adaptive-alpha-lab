param(
    [int]$BootstrapSamples = 10000,
    [int]$DmLag = 7
)

$ErrorActionPreference = "Stop"

$PythonExe = Join-Path $PSScriptRoot "env\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

& $PythonExe src\statistical_tests.py `
    --predictions models\crypto20_walkforward_alpha_oos_predictions.csv `
    --experiment-results models\crypto20_walkforward_experiment_results.csv `
    --output-prefix crypto20_ `
    --reference-methods global_lgbm regime_lgbm_hmm regime_lgbm_kmeans `
    --bootstrap-samples $BootstrapSamples `
    --dm-lag $DmLag
