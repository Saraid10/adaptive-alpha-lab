import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from alpha_models import (
    EMBARGO,
    HORIZON_HOURS,
    PRIMARY_TARGET,
    REGIME_METHODS,
    STEP_SIZE,
    fold_ranges,
    load_assignments,
    load_model_frame,
    restrict_to_common_universe,
    row_ids_for_fold,
)
from config import BASE_DIR, DB_PATH, FEATURE_COLS, SAVE_DIR, SYMBOLS
from targets import HORIZONS


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


@dataclass
class AuditRecord:
    check: str
    status: str
    severity: str
    rows_checked: int
    rows_failed: int
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit leakage, embargo, target horizon, and benchmark coverage assumptions."
    )
    parser.add_argument("--symbols", nargs="*", default=SYMBOLS)
    return parser.parse_args()


def record(
    rows: list[AuditRecord],
    check: str,
    status: str,
    severity: str,
    detail: str,
    rows_checked: int = 0,
    rows_failed: int = 0,
) -> None:
    rows.append(
        AuditRecord(
            check=check,
            status=status,
            severity=severity,
            rows_checked=int(rows_checked),
            rows_failed=int(rows_failed),
            detail=detail,
        )
    )


def table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return bool(
        con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [name],
        ).fetchone()[0]
    )


def audit_database_tables(rows: list[AuditRecord], symbols: list[str]) -> None:
    if not os.path.exists(DB_PATH):
        record(rows, "database_exists", FAIL, "critical", f"Missing DuckDB database: {DB_PATH}")
        return

    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        required = ["ohlcv", "features", "targets"]
        missing = [table for table in required if not table_exists(con, table)]
        record(
            rows,
            "required_tables_exist",
            FAIL if missing else PASS,
            "critical",
            f"Missing tables: {missing}" if missing else "ohlcv, features, and targets tables exist.",
        )
        if missing:
            return

        for table in required:
            counts = con.execute(
                f"""
                SELECT symbol, COUNT(*) AS n_rows
                FROM {table}
                WHERE symbol IN ({",".join(["?"] * len(symbols))})
                GROUP BY symbol
                ORDER BY symbol
                """,
                symbols,
            ).df()
            absent = sorted(set(symbols) - set(counts["symbol"].tolist()))
            zero_symbols = counts[counts["n_rows"] <= 0]["symbol"].tolist()
            failed = len(absent) + len(zero_symbols)
            detail = (
                f"{table} counts: "
                + ", ".join(f"{r.symbol}={int(r.n_rows)}" for r in counts.itertuples())
            )
            if absent:
                detail += f"; absent={absent}"
            record(
                rows,
                f"{table}_symbol_rows",
                FAIL if failed else PASS,
                "critical",
                detail,
                rows_checked=len(symbols),
                rows_failed=failed,
            )
    finally:
        con.close()


def audit_feature_source(rows: list[AuditRecord]) -> None:
    path = Path(BASE_DIR) / "src" / "features.py"
    text = path.read_text(encoding="utf-8")
    future_patterns = [".shift(-", "center=True"]
    found = [pattern for pattern in future_patterns if pattern in text]
    record(
        rows,
        "feature_source_no_obvious_future_ops",
        FAIL if found else PASS,
        "critical",
        f"Found future-looking feature patterns: {found}" if found else "No shift(-h) or centered rolling windows found in features.py.",
    )


def audit_feature_target_schema(rows: list[AuditRecord], symbols: list[str]) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        feature_cols = con.execute("DESCRIBE features").df()["column_name"].tolist()
        missing_features = sorted(set(FEATURE_COLS) - set(feature_cols))
        record(
            rows,
            "feature_columns_present",
            FAIL if missing_features else PASS,
            "critical",
            f"Missing feature columns: {missing_features}" if missing_features else f"All {len(FEATURE_COLS)} configured feature columns exist.",
            rows_checked=len(FEATURE_COLS),
            rows_failed=len(missing_features),
        )

        target_cols = con.execute("DESCRIBE targets").df()["column_name"].tolist()
        required_targets = [PRIMARY_TARGET, f"forward_return_{HORIZON_HOURS}h"]
        missing_targets = sorted(set(required_targets) - set(target_cols))
        record(
            rows,
            "primary_target_columns_present",
            FAIL if missing_targets else PASS,
            "critical",
            f"Missing target columns: {missing_targets}" if missing_targets else f"Primary target columns exist: {required_targets}.",
            rows_checked=len(required_targets),
            rows_failed=len(missing_targets),
        )

        joined = con.execute(
            f"""
            SELECT
                f.symbol,
                f.open_time,
                {", ".join("f." + c for c in FEATURE_COLS)},
                t.{PRIMARY_TARGET} AS target_label,
                t.forward_return_{HORIZON_HOURS}h AS target_return
            FROM features f
            JOIN targets t
              ON f.symbol = t.symbol
             AND f.open_time = t.open_time
            WHERE f.symbol IN ({",".join(["?"] * len(symbols))})
            ORDER BY f.symbol, f.open_time
            """,
            symbols,
        ).df()
        joined["open_time"] = pd.to_datetime(joined["open_time"])
        joined = joined.reset_index(drop=True)
        joined["feat_idx"] = joined.groupby("symbol").cumcount().astype(int)
    finally:
        con.close()

    bad_feature_rows = joined[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    bad_target_rows = joined[["target_label", "target_return"]].replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    record(
        rows,
        "joined_model_rows_are_finite",
        FAIL if (bad_feature_rows.any() or bad_target_rows.any()) else PASS,
        "critical",
        "Joined feature/target rows contain no NaN/inf values."
        if not (bad_feature_rows.any() or bad_target_rows.any())
        else "NaN/inf detected in joined feature/target rows.",
        rows_checked=len(joined),
        rows_failed=int((bad_feature_rows | bad_target_rows).sum()),
    )
    return joined


def audit_target_horizon_loss(rows: list[AuditRecord], symbols: list[str]) -> None:
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        counts = con.execute(
            f"""
            SELECT
                f.symbol,
                COUNT(*) AS feature_rows,
                COUNT(t.open_time) AS target_rows,
                MAX(f.open_time) AS feature_latest,
                MAX(t.open_time) AS target_latest
            FROM features f
            LEFT JOIN targets t
              ON f.symbol = t.symbol
             AND f.open_time = t.open_time
            WHERE f.symbol IN ({",".join(["?"] * len(symbols))})
            GROUP BY f.symbol
            ORDER BY f.symbol
            """,
            symbols,
        ).df()
    finally:
        con.close()

    expected_loss = max(HORIZONS)
    failures = 0
    details = []
    for row in counts.itertuples(index=False):
        loss = int(row.feature_rows - row.target_rows)
        ok = loss == expected_loss and pd.Timestamp(row.target_latest) < pd.Timestamp(row.feature_latest)
        failures += int(not ok)
        details.append(
            f"{row.symbol}: feature_rows={int(row.feature_rows)}, target_rows={int(row.target_rows)}, "
            f"lost={loss}, expected_loss={expected_loss}, target_latest={row.target_latest}"
        )
    record(
        rows,
        "target_horizon_tail_loss",
        FAIL if failures else PASS,
        "critical",
        "; ".join(details),
        rows_checked=len(counts),
        rows_failed=failures,
    )


def audit_common_universe(rows: list[AuditRecord], symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[int, int, int]]]:
    df = load_model_frame(symbols)
    assignments = load_assignments(symbols)
    df, assignments = restrict_to_common_universe(df, assignments)
    folds = fold_ranges(df)

    method_counts = assignments.groupby("method").size().reindex(REGIME_METHODS, fill_value=0)
    missing = method_counts[method_counts == 0].index.tolist()
    unequal = int(method_counts.nunique() > 1)
    record(
        rows,
        "regime_assignment_common_coverage",
        FAIL if missing or unequal else PASS,
        "critical",
        "Regime assignment rows by method: "
        + ", ".join(f"{method}={int(count)}" for method, count in method_counts.items()),
        rows_checked=len(method_counts),
        rows_failed=len(missing) + unequal,
    )

    record(
        rows,
        "walk_forward_folds_exist",
        PASS if folds else FAIL,
        "critical",
        f"{len(folds)} folds generated with initial_train, step, and embargo settings.",
        rows_checked=len(folds),
        rows_failed=0 if folds else 1,
    )

    record(
        rows,
        "regime_discovery_protocol",
        WARN,
        "methodological",
        "Legacy regime_assignments.csv is an offline/global artifact. Use walkforward_experiment_results.csv for fold-local predictive regime claims.",
        rows_checked=len(assignments),
        rows_failed=0,
    )
    return df, assignments, folds


def audit_folds(
    rows: list[AuditRecord],
    df: pd.DataFrame,
    folds: list[tuple[int, int, int]],
) -> pd.DataFrame:
    df_by_row = df.set_index("row_id", drop=False)
    fold_rows = []
    hard_failures = 0

    for fold, (train_end, test_start, test_end) in enumerate(folds, start=1):
        train_ids, test_ids = row_ids_for_fold(df_by_row, train_end, test_start, test_end)
        train = df_by_row.loc[train_ids]
        test = df_by_row.loc[test_ids]
        train_max_idx = int(train["feat_idx"].max()) if not train.empty else -1
        test_min_idx = int(test["feat_idx"].min()) if not test.empty else -1
        test_max_idx = int(test["feat_idx"].max()) if not test.empty else -1
        embargo_bars = test_min_idx - train_max_idx - 1
        train_label_end = train_max_idx + HORIZON_HOURS
        label_gap_bars = test_min_idx - train_label_end
        row_overlap = len(set(train_ids).intersection(test_ids))

        embargo_ok = embargo_bars >= EMBARGO
        label_overlap_ok = train_label_end < test_min_idx
        row_overlap_ok = row_overlap == 0
        fold_ok = embargo_ok and label_overlap_ok and row_overlap_ok
        hard_failures += int(not fold_ok)

        fold_rows.append(
            {
                "fold": fold,
                "train_end": int(train_end),
                "test_start": int(test_start),
                "test_end": int(test_end),
                "n_train_rows": int(len(train_ids)),
                "n_test_rows": int(len(test_ids)),
                "train_max_feat_idx": train_max_idx,
                "test_min_feat_idx": test_min_idx,
                "test_max_feat_idx": test_max_idx,
                "embargo_required_bars": int(EMBARGO),
                "embargo_observed_bars": int(embargo_bars),
                "primary_label_horizon_bars": int(HORIZON_HOURS),
                "train_label_end_feat_idx": int(train_label_end),
                "label_gap_before_test_bars": int(label_gap_bars),
                "row_overlap_count": int(row_overlap),
                "status": PASS if fold_ok else FAIL,
            }
        )

    record(
        rows,
        "fold_embargo_and_label_purge",
        FAIL if hard_failures else PASS,
        "critical",
        f"{len(folds) - hard_failures}/{len(folds)} folds satisfy row separation, {EMBARGO}-bar embargo, and {HORIZON_HOURS}-bar label-horizon purge.",
        rows_checked=len(folds),
        rows_failed=hard_failures,
    )
    return pd.DataFrame(fold_rows)


def audit_predictions(rows: list[AuditRecord], fold_audit: pd.DataFrame) -> None:
    path = Path(SAVE_DIR) / "alpha_oos_predictions.csv"
    results_path = Path(SAVE_DIR) / "experiment_results.csv"
    if not path.exists():
        record(
            rows,
            "oos_prediction_alignment",
            WARN,
            "artifact",
            "alpha_oos_predictions.csv is missing locally; run alpha_models.py before full row-level prediction audit.",
        )
        return

    pred = pd.read_csv(path)
    required = {"method", "regime_method", "symbol", "feat_idx", "fold", "score", "target_return"}
    missing = sorted(required - set(pred.columns))
    if missing:
        record(rows, "oos_prediction_schema", FAIL, "critical", f"Missing prediction columns: {missing}")
        return
    record(rows, "oos_prediction_schema", PASS, "critical", "alpha_oos_predictions.csv has required audit columns.")

    fold_bounds = fold_audit.set_index("fold")[["test_start", "test_end"]].to_dict("index")
    aligned = []
    for row in pred.itertuples(index=False):
        bounds = fold_bounds.get(int(row.fold))
        ok = bool(bounds and int(row.feat_idx) >= bounds["test_start"] and int(row.feat_idx) < bounds["test_end"])
        aligned.append(ok)
    aligned = np.asarray(aligned, dtype=bool)
    record(
        rows,
        "oos_prediction_rows_match_test_folds",
        FAIL if (~aligned).any() else PASS,
        "critical",
        "All prediction rows fall inside their recorded fold test windows."
        if aligned.all()
        else "Some prediction rows fall outside their recorded fold test windows.",
        rows_checked=len(pred),
        rows_failed=int((~aligned).sum()),
    )

    dupes = pred.duplicated(["method", "symbol", "feat_idx"]).sum()
    record(
        rows,
        "oos_prediction_no_duplicate_method_rows",
        FAIL if dupes else PASS,
        "critical",
        "No duplicated method/symbol/feat_idx predictions."
        if dupes == 0
        else f"Found {int(dupes)} duplicated method/symbol/feat_idx prediction rows.",
        rows_checked=len(pred),
        rows_failed=int(dupes),
    )

    method_rows = pred.groupby("method").size()
    unequal = int(method_rows.nunique() > 1)
    record(
        rows,
        "oos_prediction_equal_method_coverage",
        FAIL if unequal else PASS,
        "critical",
        "Prediction rows by method: " + ", ".join(f"{m}={int(n)}" for m, n in method_rows.items()),
        rows_checked=len(method_rows),
        rows_failed=unequal,
    )

    if results_path.exists():
        results = pd.read_csv(results_path)
        expected = results.set_index("method")["n_test_rows"].to_dict()
        mismatches = []
        for method, observed in method_rows.items():
            if int(expected.get(method, -1)) != int(observed):
                mismatches.append(f"{method}: pred={int(observed)}, results={expected.get(method)}")
        record(
            rows,
            "experiment_results_match_predictions",
            FAIL if mismatches else PASS,
            "critical",
            "experiment_results.csv n_test_rows matches alpha_oos_predictions.csv."
            if not mismatches
            else "; ".join(mismatches),
            rows_checked=len(method_rows),
            rows_failed=len(mismatches),
        )


def audit_walkforward_artifacts(rows: list[AuditRecord]) -> None:
    results_path = Path(SAVE_DIR) / "walkforward_experiment_results.csv"
    comparison_path = Path(SAVE_DIR) / "walkforward_comparison.csv"
    summary_path = Path(SAVE_DIR) / "walkforward_regime_summary.csv"

    if not results_path.exists():
        record(
            rows,
            "fold_local_regime_refit_artifacts",
            WARN,
            "artifact",
            "walkforward_experiment_results.csv is missing; run walkforward_regimes.py for Phase 13 strict regime-refit results.",
        )
        return

    results = pd.read_csv(results_path)
    required_methods = ["global_lgbm"] + [f"regime_lgbm_{method}" for method in REGIME_METHODS]
    missing = sorted(set(required_methods) - set(results["method"]))
    row_counts = results.set_index("method")["n_test_rows"]
    unequal = int(row_counts.nunique() > 1)
    missing_files = [str(path.name) for path in [comparison_path, summary_path] if not path.exists()]
    failures = len(missing) + unequal + len(missing_files)

    detail = (
        "Fold-local rows by method: "
        + ", ".join(f"{method}={int(rows)}" for method, rows in row_counts.items())
    )
    if missing:
        detail += f"; missing_methods={missing}"
    if missing_files:
        detail += f"; missing_artifacts={missing_files}"

    record(
        rows,
        "fold_local_regime_refit_artifacts",
        FAIL if failures else PASS,
        "critical",
        detail,
        rows_checked=len(required_methods),
        rows_failed=failures,
    )


def audit_robustness_artifacts(rows: list[AuditRecord]) -> None:
    results_path = Path(SAVE_DIR) / "robustness_results.csv"
    summary_path = Path(SAVE_DIR) / "robustness_summary.csv"
    wins_path = Path(SAVE_DIR) / "robustness_wins.csv"

    if not results_path.exists():
        record(
            rows,
            "robustness_matrix_artifacts",
            WARN,
            "artifact",
            "robustness_results.csv is missing; run robustness.py for Phase 14A horizon/symbol robustness results.",
        )
        return

    results = pd.read_csv(results_path)
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    required_methods = ["global_lgbm"] + [f"regime_lgbm_{method}" for method in REGIME_METHODS]
    expected_cells = 9
    expected_rows = expected_cells * len(required_methods)
    missing_files = [str(path.name) for path in [summary_path, wins_path] if not path.exists()]
    missing_cols = sorted(
        {"symbol_scope", "target", "horizon", "method", "IC", "Sharpe", "drawdown", "n_test_rows"}
        - set(results.columns)
    )

    failures = len(missing_files) + len(missing_cols)
    detail_parts = [f"robustness rows={len(results)}, expected={expected_rows}"]

    if len(results) != expected_rows:
        failures += 1
        detail_parts.append("unexpected_result_row_count")

    if not missing_cols:
        cell_counts = results.groupby(["symbol_scope", "target", "horizon"])["method"].nunique()
        incomplete_cells = int((cell_counts != len(required_methods)).sum())
        failures += incomplete_cells
        detail_parts.append(f"grid_cells={len(cell_counts)}")
        if incomplete_cells:
            detail_parts.append(f"incomplete_cells={incomplete_cells}")

    if not summary.empty:
        detail_parts.append(f"summary_cells={len(summary)}")
        if len(summary) != expected_cells:
            failures += 1
            detail_parts.append("unexpected_summary_cell_count")

    if missing_files:
        detail_parts.append(f"missing_artifacts={missing_files}")
    if missing_cols:
        detail_parts.append(f"missing_columns={missing_cols}")

    record(
        rows,
        "robustness_matrix_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(detail_parts),
        rows_checked=expected_rows,
        rows_failed=failures,
    )


def audit_robustness_stress_artifacts(rows: list[AuditRecord]) -> None:
    results_path = Path(SAVE_DIR) / "robustness_stress_results.csv"
    summary_path = Path(SAVE_DIR) / "robustness_stress_summary.csv"
    wins_path = Path(SAVE_DIR) / "robustness_stress_wins.csv"

    if not results_path.exists():
        record(
            rows,
            "robustness_stress_artifacts",
            WARN,
            "artifact",
            "robustness_stress_results.csv is missing; run robustness_stress.py for Phase 14B cost/threshold/period stress results.",
        )
        return

    results = pd.read_csv(results_path)
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    required_methods = ["global_lgbm"] + [f"regime_lgbm_{method}" for method in REGIME_METHODS]
    expected_cells = 4 * 3 * 4
    expected_rows = expected_cells * len(required_methods)
    missing_files = [str(path.name) for path in [summary_path, wins_path] if not path.exists()]
    missing_cols = sorted(
        {
            "market_period",
            "threshold",
            "transaction_cost_bps",
            "method",
            "signal_IC",
            "Sharpe",
            "drawdown",
            "total_return",
            "n_test_rows",
        }
        - set(results.columns)
    )

    failures = len(missing_files) + len(missing_cols)
    detail_parts = [f"stress rows={len(results)}, expected={expected_rows}"]

    if len(results) != expected_rows:
        failures += 1
        detail_parts.append("unexpected_result_row_count")

    if not missing_cols:
        cell_counts = results.groupby(["market_period", "threshold", "transaction_cost_bps"])["method"].nunique()
        incomplete_cells = int((cell_counts != len(required_methods)).sum())
        failures += incomplete_cells
        detail_parts.append(f"stress_cells={len(cell_counts)}")
        if incomplete_cells:
            detail_parts.append(f"incomplete_cells={incomplete_cells}")

    if not summary.empty:
        detail_parts.append(f"summary_cells={len(summary)}")
        if len(summary) != expected_cells:
            failures += 1
            detail_parts.append("unexpected_summary_cell_count")

    if missing_files:
        detail_parts.append(f"missing_artifacts={missing_files}")
    if missing_cols:
        detail_parts.append(f"missing_columns={missing_cols}")

    record(
        rows,
        "robustness_stress_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(detail_parts),
        rows_checked=expected_rows,
        rows_failed=failures,
    )


def write_outputs(rows: list[AuditRecord], fold_audit: pd.DataFrame) -> pd.DataFrame:
    out_dir = Path(SAVE_DIR)
    out_dir.mkdir(exist_ok=True)
    audit = pd.DataFrame([row.__dict__ for row in rows])
    status_order = pd.Categorical(audit["status"], [FAIL, WARN, PASS], ordered=True)
    audit = audit.assign(_status_order=status_order).sort_values(["_status_order", "check"]).drop(columns="_status_order")
    audit.to_csv(out_dir / "validation_audit.csv", index=False)
    fold_audit.to_csv(out_dir / "fold_audit.csv", index=False)
    return audit


def main() -> None:
    args = parse_args()
    rows: list[AuditRecord] = []

    audit_database_tables(rows, args.symbols)
    if any(row.status == FAIL and row.severity == "critical" for row in rows):
        audit = write_outputs(rows, pd.DataFrame())
        print(audit.to_string(index=False))
        raise SystemExit("Validation audit failed before data-level checks.")

    audit_feature_source(rows)
    audit_feature_target_schema(rows, args.symbols)
    audit_target_horizon_loss(rows, args.symbols)
    df, _, folds = audit_common_universe(rows, args.symbols)
    fold_audit = audit_folds(rows, df, folds)
    audit_predictions(rows, fold_audit)
    audit_walkforward_artifacts(rows)
    audit_robustness_artifacts(rows)
    audit_robustness_stress_artifacts(rows)
    audit = write_outputs(rows, fold_audit)

    print("\nValidation audit summary:")
    print(audit[["check", "status", "severity", "rows_checked", "rows_failed", "detail"]].to_string(index=False))

    hard_fails = audit[(audit["status"] == FAIL) & (audit["severity"] == "critical")]
    if not hard_fails.empty:
        raise SystemExit(f"Validation audit failed: {len(hard_fails)} critical checks failed.")

    warn_count = int((audit["status"] == WARN).sum())
    print(f"\nOK: validation audit completed with {warn_count} warning(s).")
    print(f"Saved: {Path(SAVE_DIR) / 'validation_audit.csv'}")
    print(f"Saved: {Path(SAVE_DIR) / 'fold_audit.csv'}")


if __name__ == "__main__":
    main()
