import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from alpha_models import (
    EMBARGO,
    FEATURE_COLS,
    N_REGIMES,
    REGIME_METHODS,
    RANDOM_STATE,
    SAVE_DIR,
    SYMBOLS,
    PredictionSet,
    predict_regime_models,
    portfolio_net_returns,
    row_ids_for_fold,
    run_global_model,
    summarize_predictions,
    train_regime_models,
    validate_test_coverage,
)
from alpha_models import fold_ranges, load_model_frame
from baselines import HMM_FEATURES


POST_COLS = [f"post_{k}" for k in range(N_REGIMES)]
ASSIGNMENT_COLS = ["fold", "split", "method", "symbol", "open_time", "feat_idx", "row_id", "regime"] + POST_COLS


@dataclass
class FoldAssignments:
    method: str
    implementation: str
    assignments: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a strict benchmark with regime models refit inside each walk-forward fold."
    )
    parser.add_argument("--symbols", nargs="*", default=SYMBOLS)
    return parser.parse_args()


def finite_matrix(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    out = np.zeros((len(probs), N_REGIMES), dtype=float)
    cols = min(probs.shape[1], N_REGIMES)
    out[:, :cols] = probs[:, :cols]
    row_sum = out.sum(axis=1, keepdims=True)
    return np.divide(out, row_sum, out=np.full_like(out, 1.0 / N_REGIMES), where=row_sum != 0)


def one_hot(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    probs = np.zeros((len(labels), N_REGIMES), dtype=float)
    valid = (labels >= 0) & (labels < N_REGIMES)
    probs[np.arange(len(labels))[valid], labels[valid]] = 1.0
    return probs


def build_assignment_frame(
    df_by_row: pd.DataFrame,
    row_ids: list[int],
    method: str,
    labels: np.ndarray,
    probs: np.ndarray,
    fold: int,
    split: str,
) -> pd.DataFrame:
    out = df_by_row.loc[row_ids, ["row_id", "symbol", "open_time", "feat_idx"]].copy()
    out["fold"] = fold
    out["split"] = split
    out["method"] = method
    out["regime"] = np.asarray(labels, dtype=int)
    probs = normalize_probs(probs)
    for k, col in enumerate(POST_COLS):
        out[col] = probs[:, k]
    return out[ASSIGNMENT_COLS]


def load_dense_contrastive_universe(symbols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    df = load_model_frame(symbols)
    original_rows = len(df)

    posterior_path = Path(SAVE_DIR) / "regime_posteriors.csv"
    embedding_path = Path(SAVE_DIR) / "embeddings.npy"
    if not posterior_path.exists() or not embedding_path.exists():
        raise RuntimeError("Missing regime_posteriors.csv or embeddings.npy. Run visualize_regimes.py first.")

    posteriors = pd.read_csv(posterior_path)
    if "embedding_idx" not in posteriors.columns:
        raise RuntimeError("regime_posteriors.csv is missing embedding_idx. Rerun visualize_regimes.py.")

    posteriors["feat_idx"] = posteriors["feat_idx"].astype(int)
    keys = posteriors[["symbol", "feat_idx", "embedding_idx"]].drop_duplicates(["symbol", "feat_idx"])
    df = df.merge(keys, on=["symbol", "feat_idx"], how="inner")
    df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=int)

    embeddings = np.load(embedding_path)
    aligned_embeddings = embeddings[df["embedding_idx"].astype(int).to_numpy()]
    coverage = len(df) / original_rows
    if coverage < 0.90:
        raise RuntimeError(f"Dense contrastive universe kept only {coverage:.1%} of rows.")
    print(f"Fold-local benchmark universe: {len(df):,}/{original_rows:,} rows ({coverage:.1%})")
    return df, aligned_embeddings


def fit_scaled(train_x: np.ndarray, full_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    full_scaled = scaler.transform(full_x)
    return train_scaled, full_scaled, scaler


def fit_gmm_assignments(
    df_by_row: pd.DataFrame,
    matrix: np.ndarray,
    train_ids: list[int],
    test_ids: list[int],
    fold: int,
    method: str,
) -> FoldAssignments:
    train_x, full_x, _ = fit_scaled(matrix[train_ids], matrix)
    model = GaussianMixture(
        n_components=N_REGIMES,
        covariance_type="full",
        n_init=5,
        random_state=RANDOM_STATE + fold,
    )
    model.fit(train_x)

    train_probs = normalize_probs(model.predict_proba(full_x[train_ids]))
    test_probs = normalize_probs(model.predict_proba(full_x[test_ids]))
    train_labels = train_probs.argmax(axis=1)
    test_labels = test_probs.argmax(axis=1)
    assignments = pd.concat(
        [
            build_assignment_frame(df_by_row, train_ids, method, train_labels, train_probs, fold, "train"),
            build_assignment_frame(df_by_row, test_ids, method, test_labels, test_probs, fold, "test"),
        ],
        ignore_index=True,
    )
    return FoldAssignments(method, "fold_local_embedding_gmm", assignments)


def fit_kmeans_assignments(
    df_by_row: pd.DataFrame,
    raw_matrix: np.ndarray,
    train_ids: list[int],
    test_ids: list[int],
    fold: int,
) -> FoldAssignments:
    train_x, full_x, _ = fit_scaled(raw_matrix[train_ids], raw_matrix)
    model = KMeans(n_clusters=N_REGIMES, n_init=20, random_state=RANDOM_STATE + fold)
    model.fit(train_x)

    train_labels = model.predict(full_x[train_ids]).astype(int)
    test_labels = model.predict(full_x[test_ids]).astype(int)
    assignments = pd.concat(
        [
            build_assignment_frame(df_by_row, train_ids, "kmeans", train_labels, one_hot(train_labels), fold, "train"),
            build_assignment_frame(df_by_row, test_ids, "kmeans", test_labels, one_hot(test_labels), fold, "test"),
        ],
        ignore_index=True,
    )
    return FoldAssignments("kmeans", "fold_local_sklearn_kmeans", assignments)


def emission_likelihood(model, x: np.ndarray) -> np.ndarray:
    log_likelihood = model._compute_log_likelihood(x)
    log_likelihood = log_likelihood - log_likelihood.max(axis=1, keepdims=True)
    return np.exp(log_likelihood)


def advance_state(state: np.ndarray, transmat: np.ndarray, steps: int) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    for _ in range(max(int(steps), 0)):
        state = state @ transmat
        total = state.sum()
        state = state / total if total > 0 else np.full_like(state, 1.0 / len(state))
    return state


def filter_hmm_posteriors(model, x: np.ndarray, initial: np.ndarray | None = None) -> np.ndarray:
    if len(x) == 0:
        return np.empty((0, N_REGIMES), dtype=float)

    likelihood = emission_likelihood(model, x)
    alpha = np.asarray(model.startprob_ if initial is None else initial, dtype=float)
    posts = []

    for i in range(len(x)):
        prior = alpha if i == 0 and initial is None else alpha @ model.transmat_
        alpha = prior * likelihood[i]
        total = alpha.sum()
        if not np.isfinite(total) or total <= 0:
            alpha = np.full(N_REGIMES, 1.0 / N_REGIMES)
        else:
            alpha = alpha / total
        posts.append(alpha.copy())

    return normalize_probs(np.vstack(posts))


def fit_hmm_assignments(
    df_by_row: pd.DataFrame,
    matrix: np.ndarray,
    train_ids: list[int],
    test_ids: list[int],
    fold: int,
    method: str,
    covariance_type: str,
    implementation: str,
) -> FoldAssignments:
    from hmmlearn.hmm import GaussianHMM

    train_frame = df_by_row.loc[train_ids].sort_values(["symbol", "feat_idx"])
    test_frame = df_by_row.loc[test_ids].sort_values(["symbol", "feat_idx"])
    train_ids_sorted = train_frame["row_id"].astype(int).tolist()
    test_ids_sorted = test_frame["row_id"].astype(int).tolist()

    train_x, full_x, _ = fit_scaled(matrix[train_ids_sorted], matrix)
    lengths = train_frame.groupby("symbol", sort=False).size().astype(int).tolist()
    model = GaussianHMM(
        n_components=N_REGIMES,
        covariance_type=covariance_type,
        n_iter=150,
        random_state=RANDOM_STATE + fold,
        min_covar=1e-4,
    )
    model.fit(train_x, lengths)

    train_assignments = []
    test_assignments = []
    train_terminal: dict[str, np.ndarray] = {}

    for symbol, symbol_train in train_frame.groupby("symbol", sort=False):
        row_ids = symbol_train["row_id"].astype(int).tolist()
        probs = filter_hmm_posteriors(model, full_x[row_ids])
        labels = probs.argmax(axis=1)
        train_terminal[symbol] = probs[-1]
        train_assignments.append(
            build_assignment_frame(df_by_row, row_ids, method, labels, probs, fold, "train")
        )

    for symbol, symbol_test in test_frame.groupby("symbol", sort=False):
        row_ids = symbol_test["row_id"].astype(int).tolist()
        symbol_train = train_frame[train_frame["symbol"] == symbol]
        if symbol in train_terminal and not symbol_train.empty:
            gap_bars = int(symbol_test["feat_idx"].min() - symbol_train["feat_idx"].max() - 1)
            initial = advance_state(train_terminal[symbol], model.transmat_, gap_bars)
        else:
            initial = None
        probs = filter_hmm_posteriors(model, full_x[row_ids], initial=initial)
        labels = probs.argmax(axis=1)
        test_assignments.append(
            build_assignment_frame(df_by_row, row_ids, method, labels, probs, fold, "test")
        )

    assignments = pd.concat(train_assignments + test_assignments, ignore_index=True)
    return FoldAssignments(method, implementation, assignments)


def fit_vol_bucket_assignments(
    df_by_row: pd.DataFrame,
    train_ids: list[int],
    test_ids: list[int],
    fold: int,
) -> FoldAssignments:
    train_frame = df_by_row.loc[train_ids].sort_values(["symbol", "feat_idx"])
    test_frame = df_by_row.loc[test_ids].sort_values(["symbol", "feat_idx"])
    train_parts = []
    test_parts = []

    for symbol, symbol_train in train_frame.groupby("symbol", sort=False):
        edges = np.quantile(symbol_train["vol_20h"].to_numpy(dtype=float), [0.25, 0.50, 0.75])
        edges = np.maximum.accumulate(edges)

        for split, frame, parts in [
            ("train", symbol_train, train_parts),
            ("test", test_frame[test_frame["symbol"] == symbol], test_parts),
        ]:
            if frame.empty:
                continue
            row_ids = frame["row_id"].astype(int).tolist()
            labels = np.searchsorted(edges, frame["vol_20h"].to_numpy(dtype=float), side="right")
            labels = np.clip(labels, 0, N_REGIMES - 1).astype(int)
            parts.append(build_assignment_frame(df_by_row, row_ids, "vol_bucket", labels, one_hot(labels), fold, split))

    assignments = pd.concat(train_parts + test_parts, ignore_index=True)
    return FoldAssignments("vol_bucket", "fold_local_volatility_quantiles", assignments)


def fit_fold_assignments(
    df_by_row: pd.DataFrame,
    embeddings: np.ndarray,
    raw_matrix: np.ndarray,
    hmm_matrix: np.ndarray,
    train_ids: list[int],
    test_ids: list[int],
    fold: int,
) -> list[FoldAssignments]:
    outputs = [
        fit_gmm_assignments(df_by_row, embeddings, train_ids, test_ids, fold, "contrastive"),
        fit_hmm_assignments(
            df_by_row,
            embeddings,
            train_ids,
            test_ids,
            fold,
            "contrastive_hmm",
            covariance_type="diag",
            implementation="fold_local_embedding_hmm_online_filter",
        ),
        fit_hmm_assignments(
            df_by_row,
            hmm_matrix,
            train_ids,
            test_ids,
            fold,
            "hmm",
            covariance_type="full",
            implementation="fold_local_raw_feature_hmm_online_filter",
        ),
        fit_kmeans_assignments(df_by_row, raw_matrix, train_ids, test_ids, fold),
        fit_vol_bucket_assignments(df_by_row, train_ids, test_ids, fold),
    ]
    return outputs


def run_fold_local_regime_models(
    df_by_row: pd.DataFrame,
    embeddings: np.ndarray,
    folds: list[tuple[int, int, int]],
) -> tuple[list[PredictionSet], pd.DataFrame, pd.DataFrame]:
    raw_matrix = finite_matrix(df_by_row.sort_index(), FEATURE_COLS)
    hmm_matrix = finite_matrix(df_by_row.sort_index(), HMM_FEATURES)
    predictions_by_method: dict[str, list[pd.DataFrame]] = {method: [] for method in REGIME_METHODS}
    test_assignments = []
    implementation_rows = []

    for fold, (train_end, test_start, test_end) in enumerate(folds, start=1):
        train_ids, test_ids = row_ids_for_fold(df_by_row, train_end, test_start, test_end)
        if not train_ids or not test_ids:
            continue
        print(f"Fold {fold:02d}: fitting fold-local regimes and alpha models...")
        fold_outputs = fit_fold_assignments(
            df_by_row,
            embeddings,
            raw_matrix,
            hmm_matrix,
            train_ids,
            test_ids,
            fold,
        )

        for output in fold_outputs:
            implementation_rows.append(
                {
                    "fold": fold,
                    "method": output.method,
                    "implementation": output.implementation,
                    "train_rows": int((output.assignments["split"] == "train").sum()),
                    "test_rows": int((output.assignments["split"] == "test").sum()),
                }
            )
            test_assignments.append(output.assignments[output.assignments["split"] == "test"])
            models = train_regime_models(df_by_row, output.assignments, train_ids)
            pred = predict_regime_models(df_by_row, output.assignments, models, test_ids, output.method, fold)
            if pred is not None:
                predictions_by_method[output.method].append(pred)

    outputs = []
    for method in REGIME_METHODS:
        frame = pd.concat(predictions_by_method[method], ignore_index=True) if predictions_by_method[method] else pd.DataFrame()
        outputs.append(PredictionSet(f"regime_lgbm_{method}", method, frame))

    assignments = pd.concat(test_assignments, ignore_index=True)
    implementations = pd.DataFrame(implementation_rows)
    return outputs, assignments, implementations


def avg_duration(assignments: pd.DataFrame) -> float:
    lengths = []
    for _, group in assignments.sort_values(["fold", "symbol", "feat_idx"]).groupby(["fold", "symbol"]):
        labels = group["regime"].to_numpy(dtype=int)
        if len(labels) == 0:
            continue
        current = labels[0]
        count = 1
        for label in labels[1:]:
            if label == current:
                count += 1
            else:
                lengths.append(count)
                current = label
                count = 1
        lengths.append(count)
    return float(np.mean(lengths)) if lengths else np.nan


def summarize_assignments(assignments: pd.DataFrame, implementations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in assignments.groupby("method", sort=False):
        counts = group["regime"].value_counts(normalize=True)
        confidence = group[POST_COLS].max(axis=1)
        impl = implementations[implementations["method"] == method]["implementation"].mode()
        rows.append(
            {
                "method": method,
                "protocol": "fold_local_regime_refit",
                "implementation": impl.iloc[0] if not impl.empty else "",
                "n_test_assignment_rows": int(len(group)),
                "n_folds": int(group["fold"].nunique()),
                "n_symbols": int(group["symbol"].nunique()),
                "n_regimes": int(group["regime"].nunique()),
                "avg_regime_duration": avg_duration(group),
                "min_regime_pct": float(counts.min()),
                "max_regime_pct": float(counts.max()),
                "mean_confidence": float(confidence.mean()),
            }
        )
    return pd.DataFrame(rows)


def save_walkforward_equity_plot(predictions: pd.DataFrame) -> None:
    if predictions.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    for method, group in predictions.groupby("method"):
        curve = (1.0 + portfolio_net_returns(group)).cumprod()
        ax.plot(curve.values, label=method, linewidth=1.2)
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--")
    ax.set_title("Adaptive Alpha Lab - Fold-Local Regime Refit Equity Curves")
    ax.set_ylabel("Portfolio value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "walkforward_equity_curve.png"), dpi=150)
    plt.close(fig)


def build_comparison(walkforward_results: pd.DataFrame) -> pd.DataFrame:
    offline_path = Path(SAVE_DIR) / "experiment_results.csv"
    if not offline_path.exists():
        return pd.DataFrame()

    offline = pd.read_csv(offline_path)
    merged = offline.merge(
        walkforward_results,
        on=["method", "target", "horizon", "regime_method", "symbol_scope"],
        how="inner",
        suffixes=("_offline", "_fold_local"),
    )
    if merged.empty:
        return merged

    for metric in ["IC", "Sharpe", "drawdown", "turnover", "total_return"]:
        merged[f"delta_{metric}"] = merged[f"{metric}_fold_local"] - merged[f"{metric}_offline"]
    cols = [
        "method",
        "regime_method",
        "IC_offline",
        "IC_fold_local",
        "delta_IC",
        "Sharpe_offline",
        "Sharpe_fold_local",
        "delta_Sharpe",
        "drawdown_offline",
        "drawdown_fold_local",
        "delta_drawdown",
        "n_test_rows_offline",
        "n_test_rows_fold_local",
    ]
    return merged[cols]


def main() -> None:
    args = parse_args()
    symbol_scope = "+".join(args.symbols)
    df, embeddings = load_dense_contrastive_universe(args.symbols)
    df_by_row = df.set_index("row_id", drop=False)
    folds = fold_ranges(df)
    if not folds:
        raise RuntimeError("Not enough rows for walk-forward validation.")

    print(
        f"Rows: {len(df):,} | symbols: {symbol_scope} | folds: {len(folds)} | "
        f"embargo: {EMBARGO} bars | protocol: fold-local regime refit"
    )

    global_output = run_global_model(df_by_row, folds)
    regime_outputs, assignments, implementations = run_fold_local_regime_models(df_by_row, embeddings, folds)
    outputs = [global_output] + regime_outputs

    predictions = pd.concat([output.frame for output in outputs if not output.frame.empty], ignore_index=True)
    predictions = predictions.sort_values(["method", "open_time", "symbol"]).reset_index(drop=True)
    predictions.to_csv(os.path.join(SAVE_DIR, "walkforward_alpha_oos_predictions.csv"), index=False)
    assignments.to_csv(os.path.join(SAVE_DIR, "walkforward_regime_assignments.csv"), index=False)

    summaries = [summarize_predictions(output.frame, output.method, output.regime_method, symbol_scope) for output in outputs]
    results = pd.DataFrame(summaries)
    validate_test_coverage(results)
    results.to_csv(os.path.join(SAVE_DIR, "walkforward_experiment_results.csv"), index=False)

    regime_summary = summarize_assignments(assignments, implementations)
    regime_summary.to_csv(os.path.join(SAVE_DIR, "walkforward_regime_summary.csv"), index=False)

    comparison = build_comparison(results)
    if not comparison.empty:
        comparison.to_csv(os.path.join(SAVE_DIR, "walkforward_comparison.csv"), index=False)
    save_walkforward_equity_plot(predictions)

    print("\nFold-local experiment results:")
    print(results.to_string(index=False))
    print("\nFold-local regime summary:")
    print(regime_summary.to_string(index=False))
    if not comparison.empty:
        print("\nOffline vs fold-local comparison:")
        print(comparison.to_string(index=False))
    print("\nOK: fold-local regime benchmark artifacts saved.")


if __name__ == "__main__":
    main()
