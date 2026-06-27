from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import BASE_DIR


PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

EXPECTED_REPAIRED_CLASSICAL = {
    "global_lgbm",
    "regime_lgbm_hmm",
    "regime_lgbm_kmeans",
    "regime_lgbm_vol_bucket",
}

EXPECTED_REPAIRED_NEURAL = {
    "global_lgbm",
    "regime_lgbm_contrastive",
    "regime_lgbm_contrastive_hmm",
    "regime_lgbm_hmm",
    "regime_lgbm_hmm_guided_gmm",
    "regime_lgbm_hmm_guided_hmm",
    "regime_lgbm_kmeans",
    "regime_lgbm_vol_bucket",
}

REQUIRED_METRIC_COLUMNS = {
    "method",
    "IC",
    "mean_asset_IC",
    "median_asset_IC",
    "cross_sectional_IC",
    "pooled_IC",
    "Sharpe",
    "drawdown",
    "turnover",
    "total_return",
    "n_return_observations",
    "n_test_rows",
}


@dataclass
class CheckResult:
    check: str
    status: str
    detail: str


def add(results: list[CheckResult], check: str, status: str, detail: str) -> None:
    results.append(CheckResult(check, status, detail))


def require_file(results: list[CheckResult], path: Path, check: str) -> bool:
    exists = path.exists()
    add(results, check, PASS if exists else FAIL, str(path.relative_to(BASE_DIR)))
    return exists


def read_csv_checked(results: list[CheckResult], path: Path, check: str) -> pd.DataFrame | None:
    if not require_file(results, path, f"{check}_exists"):
        return None
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        add(results, f"{check}_readable", FAIL, repr(exc))
        return None
    add(results, f"{check}_readable", PASS, f"rows={len(frame)} columns={len(frame.columns)}")
    return frame


def run_command(results: list[CheckResult], check: str, command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=BASE_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    detail = completed.stdout.strip().splitlines()[-1:] or [""]
    add(
        results,
        check,
        PASS if completed.returncode == 0 else FAIL,
        f"returncode={completed.returncode}; last_output={detail[0]}",
    )


def check_freeze_manifest(results: list[CheckResult]) -> None:
    path = BASE_DIR / "models" / "crypto20_development_freeze_manifest.json"
    if not require_file(results, path, "freeze_manifest_exists"):
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    if manifest.get("freeze_id") != "crypto20-development-v1":
        failures.append("freeze_id")
    if manifest.get("data_role") != "development_observed":
        failures.append("data_role")
    if len(manifest.get("symbols", [])) != 20:
        failures.append("symbol_count")
    if int(manifest.get("fold_count", -1)) != 16:
        failures.append("fold_count")
    if int(manifest.get("prediction_rows", -1)) != 321_380:
        failures.append("prediction_rows")
    add(
        results,
        "freeze_manifest_invariants",
        FAIL if failures else PASS,
        "failed=" + ",".join(failures) if failures else "crypto20-development-v1 invariants hold",
    )


def check_checkpoint_run(
    results: list[CheckResult],
    run_name: str,
    expected_folds: int,
    expected_methods: set[str],
) -> None:
    run_dir = BASE_DIR / ".tmp" / "phase39_fold_local" / run_name
    if not require_file(results, run_dir / "run_state.json", f"{run_name}_run_state_exists"):
        return
    checkpoint_root = run_dir / "checkpoints"
    fold_dirs = sorted(
        path
        for path in checkpoint_root.glob("fold_*")
        if (path / "checkpoint.json").exists()
    )
    add(
        results,
        f"{run_name}_checkpoint_count",
        PASS if len(fold_dirs) == expected_folds else FAIL,
        f"{len(fold_dirs)}/{expected_folds}",
    )
    if not fold_dirs:
        return
    first = pd.read_csv(fold_dirs[0] / "predictions.csv")
    methods = set(first["method"].astype(str))
    add(
        results,
        f"{run_name}_checkpoint_methods",
        PASS if methods == expected_methods else FAIL,
        f"observed={sorted(methods)}",
    )


def check_result_artifacts(
    results: list[CheckResult],
    prefix: str,
    expected_methods: set[str],
    expected_rows_per_method: int,
) -> None:
    models = BASE_DIR / "models"
    summary = read_csv_checked(
        results, models / f"{prefix}experiment_results.csv", f"{prefix}experiment_results"
    )
    coverage = read_csv_checked(
        results, models / f"{prefix}encoder_coverage.csv", f"{prefix}coverage"
    )
    fold_metrics = read_csv_checked(
        results, models / f"{prefix}fold_metrics.csv", f"{prefix}fold_metrics"
    )
    if summary is None:
        return

    missing_cols = sorted(REQUIRED_METRIC_COLUMNS - set(summary.columns))
    methods = set(summary["method"].astype(str)) if "method" in summary else set()
    row_counts = (
        summary.set_index("method")["n_test_rows"].to_dict()
        if {"method", "n_test_rows"}.issubset(summary.columns)
        else {}
    )
    bad_rows = {
        method: rows
        for method, rows in row_counts.items()
        if int(rows) != expected_rows_per_method
    }
    failures = bool(missing_cols or methods != expected_methods or bad_rows)
    detail = [
        f"methods={sorted(methods)}",
        f"expected_rows={expected_rows_per_method}",
    ]
    if missing_cols:
        detail.append(f"missing_cols={missing_cols}")
    if bad_rows:
        detail.append(f"bad_rows={bad_rows}")
    add(results, f"{prefix}summary_invariants", FAIL if failures else PASS, "; ".join(detail))

    if coverage is not None:
        method_counts = (
            coverage.groupby("method")["test_prediction_rows"].sum().to_dict()
            if {"method", "test_prediction_rows"}.issubset(coverage.columns)
            else {}
        )
        unequal = {k: v for k, v in method_counts.items() if int(v) != expected_rows_per_method}
        add(
            results,
            f"{prefix}coverage_equal_rows",
            FAIL if unequal or set(method_counts) != expected_methods else PASS,
            f"rows={method_counts}",
        )

    if fold_metrics is not None:
        folds_by_method = (
            fold_metrics.groupby("method")["fold"].nunique().to_dict()
            if {"method", "fold"}.issubset(fold_metrics.columns)
            else {}
        )
        bad_folds = {k: v for k, v in folds_by_method.items() if int(v) != 16}
        add(
            results,
            f"{prefix}fold_metric_coverage",
            FAIL if bad_folds or set(folds_by_method) != expected_methods else PASS,
            f"folds={folds_by_method}",
        )


def check_classical_artifacts(results: list[CheckResult]) -> None:
    models = BASE_DIR / "models"
    summary = read_csv_checked(
        results,
        models / "crypto20_repaired_classical_experiment_results.csv",
        "repaired_classical_experiment_results",
    )
    coverage = read_csv_checked(
        results,
        models / "crypto20_repaired_classical_coverage.csv",
        "repaired_classical_coverage",
    )
    if summary is not None:
        methods = set(summary["method"].astype(str))
        rows = summary.set_index("method")["n_test_rows"].to_dict()
        bad_rows = {k: v for k, v in rows.items() if int(v) != 230_400}
        add(
            results,
            "repaired_classical_summary_invariants",
            PASS if methods == EXPECTED_REPAIRED_CLASSICAL and not bad_rows else FAIL,
            f"methods={sorted(methods)}; bad_rows={bad_rows}",
        )
    if coverage is not None:
        rows = coverage.groupby("method")["test_rows"].sum().to_dict()
        bad_rows = {k: v for k, v in rows.items() if int(v) != 230_400}
        add(
            results,
            "repaired_classical_coverage_equal_rows",
            PASS if set(rows) == EXPECTED_REPAIRED_CLASSICAL and not bad_rows else FAIL,
            f"rows={rows}",
        )


def check_claim_control_docs(results: list[CheckResult]) -> None:
    required_phrases = {
        BASE_DIR / "reports" / "claim_registry.md": [
            "The repaired classical full run is complete",
            "does not support any claim that a neural/guided method is better",
            "Existing Crypto-20 results are an untouched final test",
        ],
        BASE_DIR / "reports" / "phase39r_neural_fold_local_results.md": [
            "full development-observed benchmark",
            "cannot be used as an untouched final test",
            "Outer-test metrics do not influence training or model selection",
        ],
    }
    for path, phrases in required_phrases.items():
        if not require_file(results, path, f"{path.stem}_exists"):
            continue
        text = path.read_text(encoding="utf-8")
        missing = [phrase for phrase in phrases if phrase not in text]
        add(
            results,
            f"{path.stem}_claim_control",
            FAIL if missing else PASS,
            f"missing={missing}" if missing else "required claim-control phrases present",
        )


def write_report(results: list[CheckResult], output_path: Path) -> None:
    frame = pd.DataFrame([result.__dict__ for result in results])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    failures = frame[frame["status"] == FAIL]
    warnings = frame[frame["status"] == WARN]
    markdown = [
        "# Research Grade Check Report",
        "",
        f"- Checks: {len(frame)}",
        f"- Failures: {len(failures)}",
        f"- Warnings: {len(warnings)}",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for row in frame.itertuples(index=False):
        detail = str(row.detail).replace("|", "\\|")
        markdown.append(f"| {row.check} | {row.status} | {detail} |")
    output_path.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the lightweight research-grade regression gate."
    )
    parser.add_argument(
        "--mode",
        choices=["artifact", "full"],
        default="artifact",
        help="artifact checks local outputs only; full also runs freeze verify, unit tests, and calendar audit.",
    )
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "models" / "research_grade_check_report.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results: list[CheckResult] = []
    check_freeze_manifest(results)
    check_classical_artifacts(results)
    check_result_artifacts(
        results,
        "crypto20_repaired_fold_local_",
        EXPECTED_REPAIRED_NEURAL,
        expected_rows_per_method=230_400,
    )
    check_checkpoint_run(
        results,
        "phase39r_neural_full_v1",
        expected_folds=16,
        expected_methods=EXPECTED_REPAIRED_NEURAL,
    )
    check_claim_control_docs(results)

    if args.mode == "full":
        run_command(
            results,
            "freeze_verify_command",
            [sys.executable, "src/freeze_development_dataset.py", "--verify-only"],
        )
        run_command(
            results,
            "unit_tests_command",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
        )
        run_command(
            results,
            "calendar_audit_command",
            [
                sys.executable,
                "src/fold_local_encoder_walkforward.py",
                "--universe",
                "crypto20",
                "--calendar-audit-only",
            ],
        )

    output_path = Path(args.output)
    write_report(results, output_path)
    failures = [result for result in results if result.status == FAIL]
    for result in results:
        print(f"{result.status:4} {result.check}: {result.detail}")
    print(f"\nSaved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.md')}")
    if failures:
        print(f"FAIL: {len(failures)} research-grade checks failed.")
        return 1
    print("OK: research-grade checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
