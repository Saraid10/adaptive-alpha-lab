param(
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ArgsList = @("src\phase44_paper_readiness_package.py")
if ($DryRun) {
  $ArgsList += "--dry-run"
}

& ".\env\Scripts\python.exe" @ArgsList
