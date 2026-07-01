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


def check_phase42_artifacts(results: list[CheckResult]) -> None:
    models = BASE_DIR / "models"
    stress_path = models / "phase42_execution_stress_results.csv"
    stress_summary_path = models / "phase42_execution_stress_summary.csv"
    transition_path = models / "phase42_regime_transition_diagnostics.csv"
    stable_path = models / "phase42_stable_transition_alpha.csv"
    alpha_path = models / "phase42_cross_asset_alpha_diagnostics.csv"
    feature_path = models / "phase42_feature_family_diagnostics.csv"
    report_path = BASE_DIR / "reports" / "phase42_interpretation_execution_hardening.md"
    runner_path = BASE_DIR / "src" / "phase42_interpretation_execution.py"
    test_path = BASE_DIR / "tests" / "test_phase42_interpretation_execution.py"
    ps1_path = BASE_DIR / "run_phase42_interpretation_execution.ps1"
    sh_path = BASE_DIR / "run_phase42_interpretation_execution.sh"

    for path, check in [
        (runner_path, "phase42_runner_exists"),
        (test_path, "phase42_tests_exist"),
        (ps1_path, "phase42_runner_ps1_exists"),
        (sh_path, "phase42_runner_sh_exists"),
    ]:
        require_file(results, path, check)

    stress = read_csv_checked(results, stress_path, "phase42_execution_stress_results")
    stress_summary = read_csv_checked(results, stress_summary_path, "phase42_execution_stress_summary")
    transition = read_csv_checked(results, transition_path, "phase42_regime_transition_diagnostics")
    stable = read_csv_checked(results, stable_path, "phase42_stable_transition_alpha")
    alpha = read_csv_checked(results, alpha_path, "phase42_cross_asset_alpha_diagnostics")
    features = read_csv_checked(results, feature_path, "phase42_feature_family_diagnostics")

    if stress is not None:
        expected_benchmarks = {"phase39r_repaired_neural", "phase41b_classical_candidates"}
        benchmarks = set(stress["benchmark"].astype(str)) if "benchmark" in stress else set()
        methods_by_benchmark = (
            stress.groupby("benchmark")["method"].nunique().to_dict()
            if {"benchmark", "method"}.issubset(stress.columns)
            else {}
        )
        cells = (
            stress.groupby(["benchmark", "method"]).size().to_dict()
            if {"benchmark", "method"}.issubset(stress.columns)
            else {}
        )
        bad_cells = {key: value for key, value in cells.items() if int(value) != 16}
        add(
            results,
            "phase42_execution_stress_coverage",
            PASS if benchmarks == expected_benchmarks and not bad_cells else FAIL,
            f"benchmarks={sorted(benchmarks)}; methods_by_benchmark={methods_by_benchmark}; bad_cells={bad_cells}",
        )
    if stress_summary is not None:
        positive = int((pd.to_numeric(stress_summary.get("positive_return_cells", pd.Series(dtype=float)), errors="coerce") > 0).sum())
        add(
            results,
            "phase42_execution_summary_guardrail",
            PASS if len(stress_summary) == 12 and positive >= 1 else FAIL,
            f"rows={len(stress_summary)}; methods_with_positive_stress_cells={positive}",
        )
    if transition is not None:
        expected_regimes = {"contrastive", "contrastive_hmm", "hmm", "hmm_guided_gmm", "hmm_guided_hmm", "kmeans", "vol_bucket"}
        methods = set(transition["regime_method"].astype(str)) if "regime_method" in transition else set()
        valid_rates = (
            pd.to_numeric(transition["switch_rate"], errors="coerce").between(0, 1).all()
            and pd.to_numeric(transition["regime_balance_entropy"], errors="coerce").between(0, 1).all()
            if {"switch_rate", "regime_balance_entropy"}.issubset(transition.columns)
            else False
        )
        add(
            results,
            "phase42_transition_diagnostics_guardrail",
            PASS if methods == expected_regimes and valid_rates else FAIL,
            f"methods={sorted(methods)}; valid_rates={valid_rates}",
        )
    if stable is not None:
        buckets = set(stable["state_bucket"].astype(str)) if "state_bucket" in stable else set()
        add(
            results,
            "phase42_stable_transition_alpha_guardrail",
            PASS if {"stable", "transition"}.issubset(buckets) and len(stable) >= 10 else FAIL,
            f"buckets={sorted(buckets)}; rows={len(stable)}",
        )
    if alpha is not None:
        benchmarks = set(alpha["benchmark"].astype(str)) if "benchmark" in alpha else set()
        add(
            results,
            "phase42_cross_asset_alpha_guardrail",
            PASS if benchmarks == {"phase39r_repaired_neural", "phase41b_classical_candidates"} and len(alpha) == 12 else FAIL,
            f"benchmarks={sorted(benchmarks)}; rows={len(alpha)}",
        )
    if features is not None:
        families = set(features["feature_family"].astype(str)) if "feature_family" in features else set()
        add(
            results,
            "phase42_feature_family_guardrail",
            PASS if len(families) >= 5 and len(features) >= 5 else FAIL,
            f"families={sorted(families)}",
        )
    if require_file(results, report_path, "phase42_report_exists"):
        text = report_path.read_text(encoding="utf-8")
        required = [
            "development-observed diagnostic phase",
            "does not tune models",
            "Execution Stress Summary",
            "Regime Transition Diagnostics",
            "Feature-Family Target Alignment",
            "Forbidden wording",
        ]
        missing = [phrase for phrase in required if phrase not in text]
        add(
            results,
            "phase42_report_guardrails",
            FAIL if missing else PASS,
            f"missing={missing}" if missing else "Phase 42 report guardrails present",
        )


def check_phase43a_artifacts(results: list[CheckResult]) -> None:
    models = BASE_DIR / "models"
    config_path = BASE_DIR / "configs" / "phase43_locked_holdout_freeze_v1.json"
    runner_path = BASE_DIR / "src" / "phase43_locked_holdout_freeze.py"
    test_path = BASE_DIR / "tests" / "test_phase43_locked_holdout_freeze.py"
    ps1_path = BASE_DIR / "run_phase43_locked_holdout_freeze.ps1"
    sh_path = BASE_DIR / "run_phase43_locked_holdout_freeze.sh"
    manifest_path = models / "phase43_locked_candidate_manifest.csv"
    claims_path = models / "phase43_locked_claim_rules.csv"
    holdout_path = models / "phase43_locked_holdout_rules.csv"
    report_path = BASE_DIR / "reports" / "phase43_locked_holdout_freeze.md"

    for path, check in [
        (config_path, "phase43a_config_exists"),
        (runner_path, "phase43a_runner_exists"),
        (test_path, "phase43a_tests_exist"),
        (ps1_path, "phase43a_runner_ps1_exists"),
        (sh_path, "phase43a_runner_sh_exists"),
    ]:
        require_file(results, path, check)

    manifest = read_csv_checked(results, manifest_path, "phase43a_locked_candidate_manifest")
    claims = read_csv_checked(results, claims_path, "phase43a_locked_claim_rules")
    holdout = read_csv_checked(results, holdout_path, "phase43a_locked_holdout_rules")

    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        failures = []
        if config.get("data_role") != "locked_unobserved_until_phase43b":
            failures.append("data_role")
        if config.get("final_candidate", {}).get("method") != "regime_lgbm_hmm_guided_hmm":
            failures.append("final_candidate")
        excluded = set(config.get("excluded_from_final_candidate", []))
        required_exclusions = {
            "phase41b_probability_calibration",
            "phase41b_soft_gating",
            "score_threshold_execution_control",
        }
        if not required_exclusions.issubset(excluded):
            failures.append("excluded_rescue_paths")
        add(
            results,
            "phase43a_config_guardrails",
            FAIL if failures else PASS,
            f"failed={failures}" if failures else "locked holdout freeze config guardrails present",
        )
    if manifest is not None:
        final = manifest[
            (manifest["item_type"] == "final_candidate")
            & (manifest["name"] == "regime_lgbm_hmm_guided_hmm")
            & (manifest["status"] == "frozen")
        ] if {"item_type", "name", "status"}.issubset(manifest.columns) else pd.DataFrame()
        exclusions = set(
            manifest.loc[manifest["item_type"] == "excluded_rescue_path", "name"].astype(str)
        ) if {"item_type", "name"}.issubset(manifest.columns) else set()
        hashes = int((manifest["item_type"] == "support_artifact_hash").sum()) if "item_type" in manifest else 0
        add(
            results,
            "phase43a_manifest_guardrails",
            PASS if len(final) == 1 and {"phase41b_probability_calibration", "score_threshold_execution_control"}.issubset(exclusions) and hashes >= 5 else FAIL,
            f"final_rows={len(final)}; exclusions={sorted(exclusions)}; support_hashes={hashes}",
        )
    if claims is not None:
        text = " ".join(claims.astype(str).agg(" ".join, axis=1).tolist())
        required = [
            "failed locked confirmation",
            "tradable strategy",
            "threshold tuning rescued the model",
        ]
        missing = [phrase for phrase in required if phrase not in text]
        add(
            results,
            "phase43a_claim_rules_guardrails",
            FAIL if missing else PASS,
            f"missing={missing}" if missing else "locked claim rules present",
        )
    if holdout is not None:
        rules = holdout.set_index("rule_id")["rule_value"].astype(str).to_dict() if {"rule_id", "rule_value"}.issubset(holdout.columns) else {}
        ok = (
            rules.get("preferred_holdout") == "external_asset_holdout"
            and rules.get("candidate_selection_on_holdout") == "forbidden"
            and rules.get("threshold_selection_on_holdout") == "forbidden"
            and rules.get("rerun_after_failure") == "forbidden"
        )
        add(
            results,
            "phase43a_holdout_rules_guardrails",
            PASS if ok else FAIL,
            f"rules={rules}",
        )
    if require_file(results, report_path, "phase43a_report_exists"):
        text = report_path.read_text(encoding="utf-8")
        required = [
            "before any locked-holdout outcome is inspected",
            "No locked/final data is evaluated in Phase 43A",
            "Phase 43A proves the model generalizes",
            "threshold tuning after holdout inspection",
        ]
        missing = [phrase for phrase in required if phrase not in text]
        add(
            results,
            "phase43a_report_guardrails",
            FAIL if missing else PASS,
            f"missing={missing}" if missing else "Phase 43A report guardrails present",
        )


def check_phase43b_registration_artifacts(results: list[CheckResult]) -> None:
    models = BASE_DIR / "models"
    config_path = BASE_DIR / "configs" / "phase43b_locked_holdout_registration_v1.json"
    runner_path = BASE_DIR / "src" / "phase43b_locked_holdout_registration.py"
    test_path = BASE_DIR / "tests" / "test_phase43b_locked_holdout_registration.py"
    ps1_path = BASE_DIR / "run_phase43b_locked_holdout_registration.ps1"
    sh_path = BASE_DIR / "run_phase43b_locked_holdout_registration.sh"
    quality_path = models / "phase43b_holdout_candidate_quality.csv"
    symbols_path = models / "phase43b_registered_holdout_symbols.csv"
    manifest_path = models / "phase43b_locked_holdout_registration_manifest.csv"
    report_path = BASE_DIR / "reports" / "phase43b_locked_holdout_registration.md"

    for path, check in [
        (config_path, "phase43b_registration_config_exists"),
        (runner_path, "phase43b_registration_runner_exists"),
        (test_path, "phase43b_registration_tests_exist"),
        (ps1_path, "phase43b_registration_runner_ps1_exists"),
        (sh_path, "phase43b_registration_runner_sh_exists"),
    ]:
        require_file(results, path, check)

    quality = read_csv_checked(results, quality_path, "phase43b_holdout_candidate_quality")
    symbols = read_csv_checked(results, symbols_path, "phase43b_registered_holdout_symbols")
    manifest = read_csv_checked(results, manifest_path, "phase43b_locked_holdout_registration_manifest")

    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        forbidden = set(config.get("forbidden_inputs", []))
        required_forbidden = {
            "model_predictions",
            "alpha_metrics",
            "threshold_search_on_holdout",
            "candidate_selection_on_holdout",
            "rerun_after_failure",
        }
        failures = []
        if config.get("data_role") != "locked_unobserved_registration_only":
            failures.append("data_role")
        if config.get("parent_freeze_id") != "phase43-locked-holdout-freeze-v1":
            failures.append("parent_freeze_id")
        if not required_forbidden.issubset(forbidden):
            failures.append("forbidden_inputs")
        add(
            results,
            "phase43b_registration_config_guardrails",
            FAIL if failures else PASS,
            f"failed={failures}" if failures else "registration-only guardrails present",
        )

    if quality is not None:
        required_cols = {
            "design_rank",
            "symbol",
            "external_candidate",
            "ohlcv_rows",
            "feature_rows",
            "target_rows",
            "max_gap_hours",
            "holdout_eligible",
            "failure_reason",
        }
        missing_cols = sorted(required_cols - set(quality.columns))
        development_selected = (
            quality[
                quality.get("in_development_universe", pd.Series(False, index=quality.index)).astype(bool)
                & quality.get("holdout_eligible", pd.Series(False, index=quality.index)).astype(bool)
            ]
            if not missing_cols
            else pd.DataFrame()
        )
        add(
            results,
            "phase43b_registration_quality_guardrails",
            FAIL if missing_cols or not development_selected.empty else PASS,
            f"missing_cols={missing_cols}; development_selected={len(development_selected)}",
        )

    if manifest is not None:
        manifest_items = (
            manifest.set_index("item")["value"].astype(str).to_dict()
            if {"item", "value"}.issubset(manifest.columns)
            else {}
        )
        status = manifest_items.get("registration_status")
        selected_count = int(manifest_items.get("selected_asset_count", "0"))
        final_candidate = manifest_items.get("final_candidate")
        forbidden_text = manifest_items.get("forbidden_inputs_confirmed", "")
        ok_status = status in {"registered_ready", "blocked_not_ready"}
        no_outcome_inputs = "model_predictions" in forbidden_text and "alpha_metrics" in forbidden_text
        status_consistent = (status == "blocked_not_ready" and selected_count < 10) or (
            status == "registered_ready" and selected_count >= 10
        )
        add(
            results,
            "phase43b_registration_manifest_guardrails",
            PASS if ok_status and status_consistent and final_candidate == "regime_lgbm_hmm_guided_hmm" and no_outcome_inputs else FAIL,
            f"status={status}; selected_count={selected_count}; final_candidate={final_candidate}",
        )

    if symbols is not None and manifest is not None:
        manifest_items = (
            manifest.set_index("item")["value"].astype(str).to_dict()
            if {"item", "value"}.issubset(manifest.columns)
            else {}
        )
        selected_count = int(manifest_items.get("selected_asset_count", "0"))
        symbol_count = len(symbols)
        add(
            results,
            "phase43b_registered_symbol_count",
            PASS if symbol_count == selected_count else FAIL,
            f"symbols={symbol_count}; manifest_selected_count={selected_count}",
        )

    if require_file(results, report_path, "phase43b_registration_report_exists"):
        text = report_path.read_text(encoding="utf-8")
        required = [
            "before any frozen-model outcome is evaluated",
            "No model predictions, alpha metrics, method rankings",
            "Phase 43B registration proves generalization",
            "allows retrying the locked evaluation after failure",
        ]
        missing = [phrase for phrase in required if phrase not in text]
        add(
            results,
            "phase43b_registration_report_guardrails",
            FAIL if missing else PASS,
            f"missing={missing}" if missing else "Phase 43B registration report guardrails present",
        )


def check_phase43b_locked_eval_artifacts(results: list[CheckResult]) -> None:
    models = BASE_DIR / "models"
    freeze_runner = BASE_DIR / "src" / "phase43b_locked_holdout_freeze.py"
    freeze_test = BASE_DIR / "tests" / "test_phase43b_locked_holdout_freeze.py"
    adjudication_runner = BASE_DIR / "src" / "phase43b_locked_holdout_adjudication.py"
    adjudication_test = BASE_DIR / "tests" / "test_phase43b_locked_holdout_adjudication.py"
    freeze_manifest_path = models / "phase43b_locked_holdout_freeze_manifest.json"
    symbol_manifest_path = models / "phase43b_locked_holdout_symbol_manifest.csv"
    fold_calendar_path = models / "phase43b_locked_holdout_fold_calendar.csv"
    universe_path = models / "phase43b_locked_holdout_universe_frozen.csv"
    eval_results_path = models / "phase43b_locked_external_experiment_results.csv"
    eval_fold_path = models / "phase43b_locked_external_fold_metrics.csv"
    eval_manifest_path = models / "phase43b_locked_external_encoder_manifest.csv"
    eval_coverage_path = models / "phase43b_locked_external_encoder_coverage.csv"
    primary_path = models / "phase43b_locked_external_primary_comparison.csv"
    claims_path = models / "phase43b_locked_external_claims.csv"
    freeze_report_path = BASE_DIR / "reports" / "phase43b_locked_holdout_data_freeze.md"
    eval_report_path = BASE_DIR / "reports" / "phase43b_locked_external_evaluation.md"
    adjudication_report_path = BASE_DIR / "reports" / "phase43b_locked_external_adjudication.md"

    for path, check in [
        (freeze_runner, "phase43b_freeze_runner_exists"),
        (freeze_test, "phase43b_freeze_tests_exist"),
        (adjudication_runner, "phase43b_adjudication_runner_exists"),
        (adjudication_test, "phase43b_adjudication_tests_exist"),
        (freeze_report_path, "phase43b_freeze_report_exists"),
        (eval_report_path, "phase43b_eval_report_exists"),
        (adjudication_report_path, "phase43b_adjudication_report_exists"),
    ]:
        require_file(results, path, check)

    symbol_manifest = read_csv_checked(results, symbol_manifest_path, "phase43b_locked_holdout_symbol_manifest")
    fold_calendar = read_csv_checked(results, fold_calendar_path, "phase43b_locked_holdout_fold_calendar")
    read_csv_checked(results, universe_path, "phase43b_locked_holdout_universe_frozen")
    eval_results = read_csv_checked(results, eval_results_path, "phase43b_locked_external_experiment_results")
    eval_folds = read_csv_checked(results, eval_fold_path, "phase43b_locked_external_fold_metrics")
    read_csv_checked(results, eval_manifest_path, "phase43b_locked_external_encoder_manifest")
    read_csv_checked(results, eval_coverage_path, "phase43b_locked_external_encoder_coverage")
    primary = read_csv_checked(results, primary_path, "phase43b_locked_external_primary_comparison")
    claims = read_csv_checked(results, claims_path, "phase43b_locked_external_claims")

    if require_file(results, freeze_manifest_path, "phase43b_locked_holdout_freeze_manifest_exists"):
        freeze = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
        failures = []
        if freeze.get("freeze_id") != "phase43b-locked-external-holdout-v1":
            failures.append("freeze_id")
        if freeze.get("data_role") != "locked_registered_unobserved":
            failures.append("data_role")
        if len(freeze.get("symbols", [])) != 10:
            failures.append("symbol_count")
        if int(freeze.get("fold_count", -1)) != 18:
            failures.append("fold_count")
        if int(freeze.get("prediction_rows", -1)) != 173_770:
            failures.append("prediction_rows")
        add(
            results,
            "phase43b_locked_holdout_freeze_guardrails",
            FAIL if failures else PASS,
            f"failed={failures}" if failures else "locked holdout freeze invariants hold",
        )

    if symbol_manifest is not None:
        rows_by_symbol = symbol_manifest.set_index("symbol")["prediction_rows"].to_dict() if {"symbol", "prediction_rows"}.issubset(symbol_manifest.columns) else {}
        add(
            results,
            "phase43b_symbol_manifest_guardrails",
            PASS if len(rows_by_symbol) == 10 and set(map(int, rows_by_symbol.values())) == {17_377} else FAIL,
            f"symbols={len(rows_by_symbol)}; prediction_rows={sorted(set(map(int, rows_by_symbol.values()))) if rows_by_symbol else []}",
        )
    if fold_calendar is not None:
        min_gap = pd.to_numeric(fold_calendar.get("calendar_gap_hours", pd.Series(dtype=float)), errors="coerce").min()
        add(
            results,
            "phase43b_fold_calendar_guardrails",
            PASS if len(fold_calendar) == 18 and min_gap > 0 else FAIL,
            f"folds={len(fold_calendar)}; min_gap={min_gap}",
        )
    if eval_results is not None:
        methods = set(eval_results["method"].astype(str)) if "method" in eval_results else set()
        expected = EXPECTED_REPAIRED_NEURAL
        rows = eval_results.set_index("method")["n_test_rows"].to_dict() if {"method", "n_test_rows"}.issubset(eval_results.columns) else {}
        equal_rows = set(map(int, rows.values())) == {129_600} if rows else False
        final_present = "regime_lgbm_hmm_guided_hmm" in methods
        add(
            results,
            "phase43b_locked_eval_result_guardrails",
            PASS if methods == expected and equal_rows and final_present else FAIL,
            f"methods={sorted(methods)}; row_counts={rows}",
        )
    if eval_folds is not None:
        folds = eval_folds.groupby("method")["fold"].nunique().to_dict() if {"method", "fold"}.issubset(eval_folds.columns) else {}
        add(
            results,
            "phase43b_locked_eval_fold_guardrails",
            PASS if set(folds) == EXPECTED_REPAIRED_NEURAL and set(map(int, folds.values())) == {18} else FAIL,
            f"folds={folds}",
        )
    if primary is not None:
        ok = (
            len(primary) == 2
            and set(primary.get("reference_method", pd.Series(dtype=str)).astype(str)) == {"global_lgbm", "regime_lgbm_hmm"}
            and primary.get("ic_improved", pd.Series(dtype=bool)).astype(bool).all()
            and primary.get("sharpe_non_worse", pd.Series(dtype=bool)).astype(bool).all()
            and primary.get("coverage_equal", pd.Series(dtype=bool)).astype(bool).all()
        )
        add(
            results,
            "phase43b_locked_primary_rule_guardrails",
            PASS if ok else FAIL,
            "primary frozen relative rule satisfied" if ok else "primary frozen relative rule not satisfied",
        )
    if claims is not None:
        claim_map = claims.set_index("claim_id")["claim_status"].astype(str).to_dict() if {"claim_id", "claim_status"}.issubset(claims.columns) else {}
        ok = (
            claim_map.get("locked_relative_success_rule") == "satisfied"
            and claim_map.get("positive_tradable_alpha") == "not_supported"
            and claim_map.get("same_holdout_retuning") == "forbidden"
        )
        add(
            results,
            "phase43b_locked_claim_guardrails",
            PASS if ok else FAIL,
            f"claims={claim_map}",
        )
    if adjudication_report_path.exists():
        text = adjudication_report_path.read_text(encoding="utf-8")
        required = [
            "not** a tradable-strategy claim",
            "cannot replace the final candidate after locked evaluation",
            "Forbidden wording",
        ]
        missing = [phrase for phrase in required if phrase not in text]
        add(
            results,
            "phase43b_locked_adjudication_report_guardrails",
            FAIL if missing else PASS,
            f"missing={missing}" if missing else "locked adjudication report guardrails present",
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
        BASE_DIR / "reports" / "phase42_interpretation_execution_hardening.md": [
            "development-observed diagnostic phase",
            "does not tune models",
            "Phase 42 proves the strategy is tradable",
        ],
        BASE_DIR / "reports" / "phase43_locked_holdout_freeze.md": [
            "before any locked-holdout outcome is inspected",
            "No locked/final data is evaluated in Phase 43A",
            "Phase 43A proves the model generalizes",
        ],
        BASE_DIR / "reports" / "phase43b_locked_holdout_registration.md": [
            "before any frozen-model outcome is evaluated",
            "No model predictions, alpha metrics, method rankings",
            "Phase 43B registration proves generalization",
        ],
        BASE_DIR / "reports" / "phase43b_locked_external_adjudication.md": [
            "not** a tradable-strategy claim",
            "cannot replace the final candidate after locked evaluation",
            "The locked holdout proves a tradable strategy",
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
    check_phase42_artifacts(results)
    check_phase43a_artifacts(results)
    check_phase43b_registration_artifacts(results)
    check_phase43b_locked_eval_artifacts(results)
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
