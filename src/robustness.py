import argparse
import os
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha_models import (
    FEATURE_COLS,
    REGIME_METHODS,
    SAVE_DIR,
    aligned_proba,
    fit_lgbm,
    fold_ranges,
    row_ids_for_fold,
    signal_from_probs,
    validate_test_coverage,
)
from config import DB_PATH
from walkforward_regimes import (
    fit_fold_assignments,
    finite_matrix,
    load_dense_contrastive_universe,
)
from baselines import HMM_FEATURES


LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
HORIZONS = [4, 8, 24]
SYMBOL_SCOPES = [["BTCUSDT"], ["ETHUSDT"], ["BTCUSDT", "ETHUSDT"]]
TC_PER_TRADE = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 14A compact robustness matrix across symbol scopes and horizons."
    )
    parser.add_argument("--horizons", nargs="*", type=int, default=HORIZONS)
    parser.add_argument(
        "--symbol-scopes",
        nargs="*",
        default=["BTCUSDT", "ETHUSDT", "BTCUSDT+ETHUSDT"],
        help="Scopes as symbols joined by '+', e.g. BTCUSDT ETHUSDT BTCUSDT+ETHUSDT.",
    )
    return parser.parse_args()


def parse_symbol_scope(scope: str) -> list[str]:
    return [part.strip() for part in scope.split("+") if part.strip()]


def target_columns(horizon: int) -> tuple[str, str]:
    return f"tb_label_{horizon}h", f"forward_return_{horizon}h"


def load_target_frame(symbols: list[str], horizon: int) -> pd.DataFrame:
    target_col, return_col = target_columns(horizon)
    con = duckdb.connect(DB_PATH, read_only=True)
    frames = []
    try:
        for symbol in symbols:
            df = con.execute(
                f"""
                SELECT
                    f.open_time,
                    f.symbol,
                    {", ".join("f." + c for c in FEATURE_COLS)},
                    t.{target_col} AS target_label,
                    t.{return_col} AS target_return
                FROM features f
                JOIN targets t
                  ON f.symbol = t.symbol
                 AND f.open_time = t.open_time
                WHERE f.symbol = ?
                ORDER BY f.open_time
                """,
                [symbol],
            ).df()
            if df.empty:
                continue
            df["open_time"] = pd.to_datetime(df["open_time"])
            df = df.reset_index(drop=True)
            df["feat_idx"] = df.index.astype(int)
            frames.append(df)
    finally:
        con.close()

    if not frames:
        raise RuntimeError(f"No target rows for {symbols} horizon={horizon}h.")

    df = pd.concat(frames, ignore_index=True).sort_values(["symbol", "open_time"]).reset_index(drop=True)
    df["target_class"] = df["target_label"].map(LABEL_TO_CLASS).astype(int)
    return df


def align_horizon_frame(base_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    target_df = load_target_frame(sorted(base_df["symbol"].unique()), horizon)
    cols = ["symbol", "feat_idx", "target_label", "target_return", "target_class"]
    keep = base_df.drop(columns=[c for c in cols[2:] if c in base_df.columns]).copy()
    aligned = keep.merge(
        target_df[cols],
        on=["symbol", "feat_idx"],
        how="inner",
    )
    aligned = aligned.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    aligned["row_id"] = aligned["row_id"].astype(int)
    return aligned


def build_prediction_frame(
    df_by_row: pd.DataFrame,
    row_ids: list[int],
    probs: np.ndarray,
    method: str,
    regime_method: str,
    fold: int,
    target: str,
    horizon: int,
) -> pd.DataFrame:
    score, pred_label, signal = signal_from_probs(probs)
    out = df_by_row.loc[
        row_ids,
        ["row_id", "symbol", "feat_idx", "open_time", "target_label", "target_return"],
    ].copy()
    out["method"] = method
    out["regime_method"] = regime_method
    out["target"] = target
    out["horizon"] = f"{horizon}h"
    out["fold"] = fold
    out["prob_down"] = probs[:, LABEL_TO_CLASS[-1]]
    out["prob_neutral"] = probs[:, LABEL_TO_CLASS[0]]
    out["prob_up"] = probs[:, LABEL_TO_CLASS[1]]
    out["score"] = score
    out["pred_label"] = pred_label
    out["signal"] = signal
    return out


def train_regime_models(df_by_row: pd.DataFrame, assignment_df: pd.DataFrame, train_ids: list[int]) -> dict[int, object]:
    models = {}
    train_assign = assignment_df[assignment_df["row_id"].isin(train_ids)]
    for regime, group in train_assign.groupby("regime"):
        row_ids = group["row_id"].astype(int).tolist()
        if len(row_ids) < 40:
            continue
        model = fit_lgbm(
            df_by_row.loc[row_ids, FEATURE_COLS].values,
            df_by_row.loc[row_ids, "target_class"].values,
        )
        if model is not None:
            models[int(regime)] = model
    return models


def predict_regime_models(
    df_by_row: pd.DataFrame,
    assignment_df: pd.DataFrame,
    models: dict[int, object],
    test_ids: list[int],
    method: str,
    fold: int,
    target: str,
    horizon: int,
) -> pd.DataFrame | None:
    test_assign = assignment_df[assignment_df["row_id"].isin(test_ids)].copy()
    if test_assign.empty or not models:
        return None

    test_assign = test_assign.sort_values(["symbol", "feat_idx"])
    row_ids = test_assign["row_id"].astype(int).tolist()
    x_test = df_by_row.loc[row_ids, FEATURE_COLS].values
    combined = np.zeros((len(row_ids), 3), dtype=float)
    weight_sum = np.zeros((len(row_ids), 1), dtype=float)

    for regime, model in models.items():
        probs = aligned_proba(model, x_test)
        col = f"post_{regime}"
        weights = test_assign[col].to_numpy(dtype=float).reshape(-1, 1)
        combined += probs * weights
        weight_sum += weights

    combined = np.divide(
        combined,
        weight_sum,
        out=np.full_like(combined, 1.0 / 3.0),
        where=weight_sum != 0,
    )
    return build_prediction_frame(df_by_row, row_ids, combined, f"regime_lgbm_{method}", method, fold, target, horizon)


def run_global_model(
    df_by_row: pd.DataFrame,
    folds: list[tuple[int, int, int]],
    target: str,
    horizon: int,
) -> pd.DataFrame:
    predictions = []
    for fold, (train_end, test_start, test_end) in enumerate(folds, start=1):
        train_ids, test_ids = row_ids_for_fold(df_by_row, train_end, test_start, test_end)
        if not train_ids or not test_ids:
            continue
        model = fit_lgbm(
            df_by_row.loc[train_ids, FEATURE_COLS].values,
            df_by_row.loc[train_ids, "target_class"].values,
        )
        if model is None:
            continue
        probs = aligned_proba(model, df_by_row.loc[test_ids, FEATURE_COLS].values)
        predictions.append(build_prediction_frame(df_by_row, test_ids, probs, "global_lgbm", "none", fold, target, horizon))
    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def run_regime_model(
    df_by_row: pd.DataFrame,
    assignments: pd.DataFrame,
    method: str,
    folds: list[tuple[int, int, int]],
    target: str,
    horizon: int,
) -> pd.DataFrame:
    predictions = []
    method_assign = assignments[assignments["method"] == method].copy()
    for fold, (train_end, test_start, test_end) in enumerate(folds, start=1):
        train_ids, test_ids = row_ids_for_fold(df_by_row, train_end, test_start, test_end)
        if not train_ids or not test_ids:
            continue
        fold_assign = method_assign[method_assign["fold"] == fold]
        models = train_regime_models(df_by_row, fold_assign, train_ids)
        pred = predict_regime_models(df_by_row, fold_assign, models, test_ids, method, fold, target, horizon)
        if pred is not None:
            predictions.append(pred)
    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def apply_transaction_costs(signal: pd.Series, returns: pd.Series) -> pd.Series:
    trades = signal.diff().abs().fillna(signal.abs()) / 2.0
    return signal.shift(1).fillna(0) * returns - trades * TC_PER_TRADE


def add_net_returns(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in pred.sort_values(["symbol", "open_time"]).groupby("symbol", sort=False):
        group = group.copy()
        group["net_return"] = apply_transaction_costs(group["signal"], group["target_return"])
        rows.append(group)
    return pd.concat(rows, ignore_index=True).sort_values(["open_time", "symbol"]).reset_index(drop=True)


def summarize_predictions(pred: pd.DataFrame, method: str, regime_method: str, symbol_scope: str, target: str, horizon: int) -> dict:
    if pred.empty:
        return {
            "symbol_scope": symbol_scope,
            "target": target,
            "horizon": f"{horizon}h",
            "method": method,
            "regime_method": regime_method,
            "IC": np.nan,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "Sharpe": np.nan,
            "drawdown": np.nan,
            "turnover": np.nan,
            "total_return": np.nan,
            "n_trades": 0,
            "n_test_rows": 0,
        }

    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    pred = add_net_returns(pred)
    portfolio_returns = pred.groupby("open_time")["net_return"].mean().sort_index()
    cumulative = (1.0 + portfolio_returns).cumprod()
    drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
    annualize = np.sqrt(8760 / horizon)
    trades = pred.groupby("symbol")["signal"].diff().abs().fillna(pred["signal"].abs()) / 2.0
    ic = pred["score"].corr(pred["target_return"])

    return {
        "symbol_scope": symbol_scope,
        "target": target,
        "horizon": f"{horizon}h",
        "method": method,
        "regime_method": regime_method,
        "IC": float(ic) if pd.notna(ic) else np.nan,
        "accuracy": float(accuracy_score(pred["target_label"], pred["pred_label"])),
        "balanced_accuracy": float(balanced_accuracy_score(pred["target_label"], pred["pred_label"])),
        "Sharpe": float(portfolio_returns.mean() / (portfolio_returns.std() + 1e-8) * annualize),
        "drawdown": float(drawdown.min()),
        "turnover": float(trades.mean()),
        "total_return": float(cumulative.iloc[-1] - 1.0),
        "n_trades": int((trades > 0).sum()),
        "n_test_rows": int(len(pred)),
    }


def fit_scope_assignments(base_df: pd.DataFrame, embeddings: np.ndarray, folds: list[tuple[int, int, int]]) -> pd.DataFrame:
    df_by_row = base_df.set_index("row_id", drop=False)
    raw_matrix = finite_matrix(df_by_row.sort_index(), FEATURE_COLS)
    hmm_matrix = finite_matrix(df_by_row.sort_index(), HMM_FEATURES)
    assignments = []

    for fold, (train_end, test_start, test_end) in enumerate(folds, start=1):
        train_ids, test_ids = row_ids_for_fold(df_by_row, train_end, test_start, test_end)
        if not train_ids or not test_ids:
            continue
        print(f"    fold {fold:02d}: refitting regime assignments")
        fold_outputs = fit_fold_assignments(df_by_row, embeddings, raw_matrix, hmm_matrix, train_ids, test_ids, fold)
        assignments.extend(output.assignments for output in fold_outputs)

    return pd.concat(assignments, ignore_index=True)


def run_scope(scope: list[str], horizons: list[int]) -> pd.DataFrame:
    symbol_scope = "+".join(scope)
    print(f"\n=== Robustness scope: {symbol_scope} ===")
    base_df, embeddings = load_dense_contrastive_universe(scope)
    folds = fold_ranges(base_df)
    if not folds:
        raise RuntimeError(f"Not enough rows for folds in scope {symbol_scope}.")

    assignments = fit_scope_assignments(base_df, embeddings, folds)
    summaries = []
    for horizon in horizons:
        target, _ = target_columns(horizon)
        print(f"  horizon {horizon}h: training/evaluating alpha models")
        horizon_df = align_horizon_frame(base_df, horizon)
        df_by_row = horizon_df.set_index("row_id", drop=False)
        horizon_folds = fold_ranges(horizon_df)

        frames = {"global_lgbm": run_global_model(df_by_row, horizon_folds, target, horizon)}
        for method in REGIME_METHODS:
            frames[f"regime_lgbm_{method}"] = run_regime_model(df_by_row, assignments, method, horizon_folds, target, horizon)

        result = pd.DataFrame(
            [
                summarize_predictions(frames["global_lgbm"], "global_lgbm", "none", symbol_scope, target, horizon),
                *[
                    summarize_predictions(frames[f"regime_lgbm_{method}"], f"regime_lgbm_{method}", method, symbol_scope, target, horizon)
                    for method in REGIME_METHODS
                ],
            ]
        )
        validate_test_coverage(result)
        summaries.append(result)

    return pd.concat(summaries, ignore_index=True)


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["symbol_scope", "target", "horizon"]
    for key, group in results.groupby(group_cols, sort=False):
        best_ic = group.sort_values("IC", ascending=False).iloc[0]
        best_sharpe = group.sort_values("Sharpe", ascending=False).iloc[0]
        lowest_dd = group.sort_values("drawdown", ascending=False).iloc[0]
        rows.append(
            {
                "symbol_scope": key[0],
                "target": key[1],
                "horizon": key[2],
                "best_ic_method": best_ic["method"],
                "best_ic": best_ic["IC"],
                "best_sharpe_method": best_sharpe["method"],
                "best_sharpe": best_sharpe["Sharpe"],
                "lowest_drawdown_method": lowest_dd["method"],
                "lowest_drawdown": lowest_dd["drawdown"],
                "methods_tested": int(len(group)),
                "n_test_rows_per_method": int(group["n_test_rows"].max()),
            }
        )
    summary = pd.DataFrame(rows)

    wins = []
    for metric, method_col in [("IC", "best_ic_method"), ("Sharpe", "best_sharpe_method"), ("drawdown", "lowest_drawdown_method")]:
        counts = summary[method_col].value_counts()
        for method, count in counts.items():
            wins.append({"metric": metric, "method": method, "wins": int(count)})
    return summary, pd.DataFrame(wins)


def save_heatmap(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, column, title in [
        (axes[0], "best_ic_method", "Best IC Method"),
        (axes[1], "best_sharpe_method", "Best Sharpe Method"),
    ]:
        pivot = summary.pivot(index="symbol_scope", columns="horizon", values=column).sort_index()
        ax.imshow(np.arange(pivot.size).reshape(pivot.shape), cmap="Greys", alpha=0.08)
        ax.set_title(title)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i, scope in enumerate(pivot.index):
            for j, horizon in enumerate(pivot.columns):
                ax.text(j, i, str(pivot.loc[scope, horizon]).replace("regime_lgbm_", ""), ha="center", va="center", fontsize=8)

    fig.suptitle("Adaptive Alpha Lab - Phase 14A Robustness Winners", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "robustness_heatmap.png"), dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    scopes = [parse_symbol_scope(scope) for scope in args.symbol_scopes]
    all_results = []
    for scope in scopes:
        all_results.append(run_scope(scope, args.horizons))

    results = pd.concat(all_results, ignore_index=True)
    summary, wins = build_summary(results)
    results.to_csv(os.path.join(SAVE_DIR, "robustness_results.csv"), index=False)
    summary.to_csv(os.path.join(SAVE_DIR, "robustness_summary.csv"), index=False)
    wins.to_csv(os.path.join(SAVE_DIR, "robustness_wins.csv"), index=False)
    save_heatmap(summary)

    print("\nRobustness summary:")
    print(summary.to_string(index=False))
    print("\nWin counts:")
    print(wins.to_string(index=False))
    print("\nOK: robustness artifacts saved.")


if __name__ == "__main__":
    main()
