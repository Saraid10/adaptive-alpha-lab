param(
  [switch]$SkipFeatureDiagnostics
)

$ErrorActionPreference = "Stop"

$python = ".\env\Scripts\python.exe"
if (-not (Test-Path $python)) {
  $python = "python"
}

$args = @("src\phase42_interpretation_execution.py", "--universe", "crypto20")
if ($SkipFeatureDiagnostics) {
  $args += "--skip-feature-diagnostics"
}

& $python @args
