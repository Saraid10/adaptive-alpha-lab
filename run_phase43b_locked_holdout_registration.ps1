param(
  [string]$Config = "configs\phase43b_locked_holdout_registration_v1.json",
  [string]$DbPath = ""
)

$ErrorActionPreference = "Stop"

if ($DbPath -eq "") {
  .\env\Scripts\python.exe src\phase43b_locked_holdout_registration.py --config $Config
} else {
  .\env\Scripts\python.exe src\phase43b_locked_holdout_registration.py --config $Config --db-path $DbPath
}
