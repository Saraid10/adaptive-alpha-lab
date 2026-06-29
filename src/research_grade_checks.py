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


def check_statistical_artifacts(results: list[CheckResult]) -> None:
    models = BASE_DIR / "models"
    prefix = "crypto20_repaired_fold_local_statistical_"
    summary = read_csv_checked(
        results,
        models / f"{prefix}method_summary.csv",
        "phase40_repaired_statistical_method_summary",
    )
    claims = read_csv_checked(
        results,
        models / f"{prefix}claims.csv",
        "phase40_repaired_statistical_claims",
    )
    pairwise = read_csv_checked(
        results,
        models / f"{prefix}pairwise_tests.csv",
        "phase40_repaired_statistical_pairwise_tests",
    )

    for stem in [
        "fold_metrics",
        "asset_metrics",
        "asset_pairwise_tests",
        "test_summary",
        "multiple_testing",
        "sharpe_diagnostics",
    ]:
        read_csv_checked(results, models / f"{prefix}{stem}.csv", f"phase40_repaired_statistical_{stem}")

    for stem in [
        "ic_confidence_intervals",
        "multiple_testing",
        "sharpe_diagnostics",
    ]:
        require_file(results, models / f"{prefix}{stem}.png", f"phase40_repaired_statistical_{stem}_png")

    if summary is not None:
        methods = set(summary["method"].astype(str)) if "method" in summary else set()
        folds = summary.set_index("method")["n_folds"].to_dict() if {"method", "n_folds"}.issubset(summary.columns) else {}
        rows = (
            summary.set_index("method")["full_sample_n_test_rows"].to_dict()
            if {"method", "full_sample_n_test_rows"}.issubset(summary.columns)
            else {}
        )
        bad_folds = {method: value for method, value in folds.items() if int(value) != 16}
        bad_rows = {method: value for method, value in rows.items() if int(value) != 230_400}
        add(
            results,
            "phase40_repaired_statistical_summary_invariants",
            FAIL if methods != EXPECTED_REPAIRED_NEURAL or bad_folds or bad_rows else PASS,
            f"methods={sorted(methods)}; bad_folds={bad_folds}; bad_rows={bad_rows}",
        )

    if claims is not None:
        required_cols = {"metric", "claim_status", "comparison"}
        missing_cols = sorted(required_cols - set(claims.columns))
        positive_alpha_claims = claims[
            claims["metric"].isin(["IC", "Sharpe"])
            & claims["claim_status"].astype(str).str.contains("survives", case=False, na=False)
        ] if not missing_cols else pd.DataFrame()
        add(
            results,
            "phase40_repaired_statistical_no_corrected_alpha_claim",
            FAIL if missing_cols or not positive_alpha_claims.empty else PASS,
            f"missing_cols={missing_cols}; corrected_ic_sharpe_claims={len(positive_alpha_claims)}",
        )

    if pairwise is not None:
        references = set(pairwise["reference_method"].dropna().astype(str)) if "reference_method" in pairwise else set()
        expected_refs = {"global_lgbm", "regime_lgbm_hmm", "regime_lgbm_kmeans"}
        add(
            results,
            "phase40_repaired_statistical_reference_methods",
            PASS if expected_refs.issubset(references) else FAIL,
            f"references={sorted(references)}",
        )


def check_phase41_artifacts(results: list[CheckResult]) -> None:
    config_path = BASE_DIR / "configs" / "phase41_bounded_candidates_v1.json"
    registry_path = BASE_DIR / "models" / "phase41_candidate_registry.csv"
    rules_path = BASE_DIR / "models" / "phase41_selection_rules.csv"
    report_path = BASE_DIR / "reports" / "phase41_bounded_improvement_protocol.md"
    runner_path = BASE_DIR / "src" / "phase41_inner_validation_candidates.py"
    runner_test_path = BASE_DIR / "tests" / "test_phase41_inner_validation_candidates.py"
    runner_ps1_path = BASE_DIR / "run_phase41_inner_validation_candidates.ps1"
    runner_sh_path = BASE_DIR / "run_phase41_inner_validation_candidates.sh"
    full_summary_path = BASE_DIR / "models" / "phase41_classical_experiment_results.csv"
    full_fold_path = BASE_DIR / "models" / "phase41_classical_fold_metrics.csv"
    full_selected_path = BASE_DIR / "models" / "phase41_classical_selected_candidates.csv"
    full_candidates_path = BASE_DIR / "models" / "phase41_classical_inner_candidate_results.csv"
    full_claims_path = BASE_DIR / "models" / "phase41_classical_statistical_claims.csv"
    full_stats_path = BASE_DIR / "models" / "phase41_classical_statistical_method_summary.csv"
    full_report_path = BASE_DIR / "reports" / "phase41_inner_validation_candidate_run.md"

    if require_file(results, config_path, "phase41_config_exists"):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        failures = []
        if config.get("selection_boundary") != "inner_chronological_validation_only":
            failures.append("selection_boundary")
        if not config.get("hard_constraints", {}).get("outer_test_selection_forbidden", False):
            failures.append("outer_test_selection_forbidden")
        forbidden = set(config.get("forbidden_selection_inputs", []))
        if "models/crypto20_repaired_fold_local_alpha_oos_predictions.csv" not in forbidden:
            failures.append("forbidden_oos_predictions")
        add(
            results,
            "phase41_config_guardrails",
            FAIL if failures else PASS,
            f"failed={failures}" if failures else "inner-validation-only guardrails present",
        )

    registry = read_csv_checked(results, registry_path, "phase41_candidate_registry")
    rules = read_csv_checked(results, rules_path, "phase41_selection_rules")
    if registry is not None:
        families = set(registry["family"].astype(str)) if "family" in registry else set()
        scopes = set(registry["selection_scope"].astype(str)) if "selection_scope" in registry else set()
        expected = {"probability_calibration", "soft_regime_gating", "execution_control"}
        threshold = registry[registry["candidate_id"] == "p41_score_threshold"] if "candidate_id" in registry else pd.DataFrame()
        threshold_deferred = (
            not threshold.empty
            and "status" in threshold
            and set(threshold["status"].astype(str)) == {"registered_deferred"}
        )
        add(
            results,
            "phase41_candidate_registry_guardrails",
            PASS if expected.issubset(families) and scopes == {"inner_validation_only"} and threshold_deferred else FAIL,
            f"families={sorted(families)}; scopes={sorted(scopes)}; threshold_deferred={threshold_deferred}",
        )
    if rules is not None:
        rule_text = " ".join(rules.astype(str).agg(" ".join, axis=1).tolist())
        required = [
            "Do not select candidate parameters from repaired outer-test predictions",
            "Use inner_validation_nll as the primary candidate selector",
            "Reject candidates that increase turnover by more than 25%",
        ]
        missing = [phrase for phrase in required if phrase not in rule_text]
        add(
            results,
            "phase41_selection_rules_guardrails",
            FAIL if missing else PASS,
            f"missing={missing}" if missing else "mandatory Phase 41 rules present",
        )
    if require_file(results, report_path, "phase41_report_exists"):
        text = report_path.read_text(encoding="utf-8")
        required = [
            "does **not** tune against Phase 40 outer-test results",
            "Forbidden Selection Inputs",
            "not a performance claim",
        ]
        missing = [phrase for phrase in required if phrase not in text]
        add(
            results,
            "phase41_report_guardrails",
            FAIL if missing else PASS,
            f"missing={missing}" if missing else "Phase 41 report guardrails present",
        )
    for path, check in [
        (runner_path, "phase41b_runner_exists"),
        (runner_test_path, "phase41b_runner_tests_exist"),
        (runner_ps1_path, "phase41b_runner_ps1_exists"),
        (runner_sh_path, "phase41b_runner_sh_exists"),
    ]:
        require_file(results, path, check)

    summary = read_csv_checked(results, full_summary_path, "phase41b_full_experiment_results")
    fold_metrics = read_csv_checked(results, full_fold_path, "phase41b_full_fold_metrics")
    selected = read_csv_checked(results, full_selected_path, "phase41b_full_selected_candidates")
    candidates = read_csv_checked(results, full_candidates_path, "phase41b_full_inner_candidate_results")
    claims = read_csv_checked(results, full_claims_path, "phase41b_full_statistical_claims")
    stats = read_csv_checked(results, full_stats_path, "phase41b_full_statistical_method_summary")
    if summary is not None:
        methods = set(summary["method"].astype(str)) if "method" in summary else set()
        rows = summary.set_index("method")["n_test_rows"].to_dict() if {"method", "n_test_rows"}.issubset(summary.columns) else {}
        expected = {"global_lgbm", "regime_lgbm_hmm", "regime_lgbm_kmeans", "regime_lgbm_vol_bucket"}
        bad_rows = {method: value for method, value in rows.items() if int(value) != 230_400}
        add(
            results,
            "phase41b_full_summary_invariants",
            FAIL if methods != expected or bad_rows else PASS,
            f"methods={sorted(methods)}; bad_rows={bad_rows}",
        )
    if fold_metrics is not None:
        folds = fold_metrics.groupby("method")["fold"].nunique().to_dict() if {"method", "fold"}.issubset(fold_metrics.columns) else {}
        bad_folds = {method: value for method, value in folds.items() if int(value) != 16}
        add(
            results,
            "phase41b_full_fold_coverage",
            FAIL if bad_folds or len(folds) != 4 else PASS,
            f"folds={folds}",
        )
    if claims is not None:
        alpha_claims = claims[
            claims["metric"].isin(["IC", "Sharpe"])
            & claims["claim_status"].astype(str).str.contains("survives", case=False, na=False)
        ] if {"metric", "claim_status"}.issubset(claims.columns) else pd.DataFrame()
        add(
            results,
            "phase41b_no_corrected_alpha_claim",
            FAIL if not alpha_claims.empty else PASS,
            f"corrected_ic_sharpe_claims={len(alpha_claims)}",
        )
    if candidates is not None and selected is not None:
        expected_candidate_ids = {
            "baseline",
            "p41_prob_temperature",
            "p41_prior_blend",
            "p41_posterior_temperature",
            "p41_global_regime_shrinkage",
        }
        observed_candidates = set(candidates["candidate_id"].astype(str)) if "candidate_id" in candidates else set()
        observed_selected = set(selected["candidate_id"].astype(str)) if "candidate_id" in selected else set()
        deferred_present = "p41_score_threshold" in observed_candidates or "p41_score_threshold" in observed_selected
        unexpected = sorted((observed_candidates | observed_selected) - expected_candidate_ids)
        add(
            results,
            "phase41b_candidate_scope_guardrail",
            FAIL if deferred_present or unexpected else PASS,
            f"unexpected={unexpected}; deferred_present={deferred_present}",
        )
    if stats is not None:
        methods = set(stats["method"].astype(str)) if "method" in stats else set()
        add(
            results,
            "phase41b_statistical_methods",
            PASS if methods == {"global_lgbm", "regime_lgbm_hmm", "regime_lgbm_kmeans", "regime_lgbm_vol_bucket"} else FAIL,
            f"methods={sorted(methods)}",
        )
    if require_file(results, full_report_path, "phase41b_full_report_exists"):
        text = full_report_path.read_text(encoding="utf-8")
        required = [
            "full development-observed run",
            "Corrected IC/Sharpe claims",
            "Controlled negative result",
            "Score-threshold candidates are registered but deferred",
            "Forbidden wording",
        ]
        missing = [phrase for phrase in required if phrase not in text]
        add(
            results,
            "phase41b_full_report_guardrails",
            FAIL if missing else PASS,
            f"missing={missing}" if missing else "Phase 41B full report guardrails present",
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
            "The repaired classical full run, repaired neural/guided full run, and Phase 40 repaired statistical adjudication are complete",
            "no repaired method currently supports a robust positive-alpha or dominance claim",
            "must not tune directly against Phase 40 outer-test outcomes",
            "Existing Crypto-20 results are an untouched final test",
        ],
        BASE_DIR / "reports" / "phase39r_neural_fold_local_results.md": [
            "full development-observed benchmark",
            "cannot be used as an untouched final test",
            "Outer-test metrics do not influence training or model selection",
        ],
        BASE_DIR / "reports" / "phase40_repaired_statistical_adjudication.md": [
            "does not support a robust positive-alpha or method-dominance claim",
            "not an untouched final-test claim",
            "must not tune directly against Phase 40 outer-test outcomes",
        ],
        BASE_DIR / "reports" / "phase41_bounded_improvement_protocol.md": [
            "does **not** tune against Phase 40 outer-test results",
            "Forbidden Selection Inputs",
            "not a performance claim",
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
    check_statistical_artifacts(results)
    check_phase41_artifacts(results)
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
