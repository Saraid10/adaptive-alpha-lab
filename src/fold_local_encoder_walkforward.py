from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from alpha_models import (
    EMBARGO,
    FEATURE_COLS,
    PredictionSet,
    build_prediction_frame,
    fit_lgbm,
    aligned_proba,
    fold_ranges,
    load_model_frame,
    predict_regime_models,
    row_ids_for_fold,
    summarize_predictions,
    train_regime_models,
    validate_test_coverage,
)
from baselines import HMM_FEATURES
from config import SAVE_DIR, WINDOW_SIZE
from dataset import load_feature_matrix
from fold_local_encoder import (
    FoldEncoderConfig,
    encode_causal_rows,
    fit_fold_encoder,
    make_fold_bounds,
)
from universe import add_symbol_args, resolve_symbols
from walkforward_regimes import (
    POST_COLS,
    build_guided_alpha_comparison,
    finite_matrix,
    fit_fold_assignments,
    summarize_assignments,
)


EXPECTED_METHODS = {
    "global_lgbm",
    "regime_lgbm_vol_bucket",
    "regime_lgbm_kmeans",
    "regime_lgbm_hmm",
    "regime_lgbm_contrastive",
    "regime_lgbm_contrastive_hmm",
    "regime_lgbm_hmm_guided_gmm",
    "regime_lgbm_hmm_guided_hmm",
}
EXPECTED_FULL_FOLDS = 16
FULL_PROTOCOL_EPOCHS = 30


def is_smoke_manifest(manifest: pd.DataFrame) -> bool:
    if manifest.empty:
        return True
    if "run_kind" in manifest and set(manifest["run_kind"].astype(str)) == {"full"}:
        return False
    folds = int(manifest["fold"].nunique())
    max_epochs = int(pd.to_numeric(manifest["max_epochs"], errors="coerce").min())
    window_cap = 0
    if "window_cap" in manifest:
        window_cap = int(pd.to_numeric(manifest["window_cap"], errors="coerce").fillna(0).max())
    return folds < EXPECTED_FULL_FOLDS or max_epochs < FULL_PROTOCOL_EPOCHS or window_cap > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fully fold-local vanilla/guided encoder benchmark."
    )
    add_symbol_args(parser)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--inner-validation-bars", type=int, default=720)
    parser.add_argument("--inner-embargo-bars", type=int, default=120)
    parser.add_argument("--label-purge-bars", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--minimum-selection-epochs", type=int, default=3)
    parser.add_argument("--output-prefix", default="crypto20_fold_local_")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild the compact Markdown report from existing compact CSV artifacts.",
    )
    parser.add_argument(
        "--heavy-dir",
        default=str(Path(".tmp") / "phase39_fold_local"),
        help="Ignored directory for fold weights and run metadata.",
    )
    return parser.parse_args()


def output_path(prefix: str, stem: str) -> Path:
    return Path(SAVE_DIR) / f"{prefix}{stem}.csv"


def load_raw_matrices(symbols: list[str]) -> dict[str, np.ndarray]:
    matrices = {symbol: load_feature_matrix(symbol) for symbol in symbols}
    short = {symbol: len(matrix) for symbol, matrix in matrices.items() if len(matrix) <= WINDOW_SIZE}
    if short:
        raise RuntimeError(f"Symbols with insufficient encoder history: {short}")
    return matrices


def build_common_frame(symbols: list[str]) -> pd.DataFrame:
    frame = load_model_frame(symbols)
    frame = frame[frame["feat_idx"] >= WINDOW_SIZE - 1].copy()
    counts = frame.groupby("symbol")["feat_idx"].agg(["min", "max", "count"])
    if counts.empty or len(counts) != len(symbols):
        raise RuntimeError("Fold-local encoder universe is missing requested symbols.")
    frame = frame.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    frame["row_id"] = np.arange(len(frame), dtype=int)
    return frame


def run_global_fold(
    df_by_row: pd.DataFrame,
    train_ids: list[int],
    test_ids: list[int],
    fold: int,
) -> pd.DataFrame:
    model = fit_lgbm(
        df_by_row.loc[train_ids, FEATURE_COLS].values,
        df_by_row.loc[train_ids, "target_class"].values,
    )
    if model is None:
        raise RuntimeError(f"Global LightGBM could not fit fold {fold}.")
    probabilities = aligned_proba(model, df_by_row.loc[test_ids, FEATURE_COLS].values)
    return build_prediction_frame(
        df_by_row, test_ids, probabilities, "global_lgbm", "none", fold
    )


def validate_complete_coverage(results: pd.DataFrame, predictions: pd.DataFrame, folds: int) -> None:
    methods = set(results["method"].astype(str))
    missing = sorted(EXPECTED_METHODS - methods)
    extra = sorted(methods - EXPECTED_METHODS)
    if missing or extra:
        raise RuntimeError(f"Phase 39 method mismatch: missing={missing}, extra={extra}")
    validate_test_coverage(results)
    counts = predictions.groupby("method").size()
    if counts.nunique() != 1:
        raise RuntimeError(f"Unequal row-level method coverage: {counts.to_dict()}")
    fold_counts = predictions.groupby("method")["fold"].nunique()
    if not (fold_counts == folds).all():
        raise RuntimeError(f"Method fold coverage mismatch: {fold_counts.to_dict()}")
    duplicated = predictions.duplicated(["method", "symbol", "feat_idx", "fold"]).sum()
    if duplicated:
        raise RuntimeError(f"Phase 39 produced {int(duplicated)} duplicate predictions.")


def build_historical_comparison(results: pd.DataFrame) -> pd.DataFrame:
    path = Path(SAVE_DIR) / "crypto20_walkforward_experiment_results.csv"
    if not path.exists():
        return pd.DataFrame()
    historical = pd.read_csv(path)
    metrics = ["IC", "Sharpe", "drawdown", "total_return", "turnover", "n_test_rows"]
    merged = results.merge(historical, on="method", how="left", suffixes=("_fold_local_encoder", "_phase36"))
    for metric in metrics:
        left = f"{metric}_fold_local_encoder"
        right = f"{metric}_phase36"
        if left in merged and right in merged:
            merged[f"delta_{metric}"] = merged[left] - merged[right]
    return merged


def write_results_report(
    path: Path,
    results: pd.DataFrame,
    manifest: pd.DataFrame,
    folds: int,
    symbols: list[str],
    smoke: bool,
) -> None:
    cols = ["method", "IC", "Sharpe", "drawdown", "total_return", "turnover", "n_test_rows"]
    def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
        display = frame.copy()
        for column in display.columns:
            if pd.api.types.is_float_dtype(display[column]):
                display[column] = display[column].map(lambda value: f"{value:.{digits}f}")
        headers = [str(column) for column in display.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines.extend(
            "| " + " | ".join(str(value) for value in row) + " |"
            for row in display.itertuples(index=False, name=None)
        )
        return "\n".join(lines)

    table = markdown_table(results[cols])
    selected_frame = (
        manifest.groupby("encoder_method")["selected_epoch"]
        .agg(["min", "median", "max"])
        .reset_index()
    )
    selected = markdown_table(selected_frame)
    status = "smoke validation only; no performance claim" if smoke else "full development-observed benchmark"
    text = f"""# Phase 39 Fully Fold-Local Encoder Results

## Run Status

This is a **{status}** over {folds} fold(s) and {len(symbols)} symbol(s). All outcomes remain development-observed and cannot be used as an untouched final test.

## Validity Contract

- Scalers, weak-supervision HMMs, contrastive pairs, encoder weights, assignment layers, and alpha models are fit inside each outer fold.
- Epochs are selected on an inner chronological validation block.
- The selected epoch count is refit on the full authorized outer-training interval.
- Outer-test metrics do not influence training or model selection.
- Vanilla and guided learned methods share the same outer folds and test coverage as classical and global baselines.

## Method Results

{table}

## Selected Epochs

{selected}

## Interpretation Rule

A smoke run validates code paths, leakage boundaries, artifacts, and coverage only. A full run is still development evidence. Model changes require a new registered candidate family, and confirmatory claims require a frozen configuration evaluated once on a locked holdout.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.report_only:
        results_path = output_path(args.output_prefix, "experiment_results")
        manifest_path = output_path(args.output_prefix, "encoder_manifest")
        if not results_path.exists() or not manifest_path.exists():
            raise RuntimeError("Report-only mode requires existing result and manifest CSV files.")
        results = pd.read_csv(results_path)
        manifest = pd.read_csv(manifest_path)
        folds = int(manifest["fold"].nunique())
        symbol_scope = str(results.iloc[0].get("symbol_scope", "")) if not results.empty else ""
        symbols = [symbol for symbol in symbol_scope.split("+") if symbol]
        write_results_report(
            Path("reports") / "phase39_fold_local_results.md",
            results,
            manifest,
            folds,
            symbols,
            smoke=is_smoke_manifest(manifest),
        )
        print("OK: Phase 39 compact report rebuilt from existing artifacts.")
        return
    symbols = resolve_symbols(args)
    config = FoldEncoderConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_windows=args.max_windows,
        inner_validation_bars=args.inner_validation_bars,
        inner_embargo_bars=args.inner_embargo_bars,
        label_purge_bars=args.label_purge_bars,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        minimum_selection_epochs=args.minimum_selection_epochs,
    )
    if config.epochs < 1 or config.batch_size < 2:
        raise ValueError("Epochs must be >=1 and batch size must be >=2.")

    frame = build_common_frame(symbols)
    df_by_row = frame.set_index("row_id", drop=False)
    raw_matrices = load_raw_matrices(symbols)
    folds = fold_ranges(frame)
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    if not folds:
        raise RuntimeError("No eligible walk-forward folds.")

    raw_matrix = finite_matrix(df_by_row.sort_index(), FEATURE_COLS)
    hmm_matrix = finite_matrix(df_by_row.sort_index(), HMM_FEATURES)
    heavy_dir = Path(args.heavy_dir)
    heavy_dir.mkdir(parents=True, exist_ok=True)

    prediction_parts: dict[str, list[pd.DataFrame]] = {
        method: [] for method in EXPECTED_METHODS
    }
    test_assignments = []
    implementation_rows = []
    manifest_rows = []
    loss_rows = []
    coverage_rows = []

    print(
        f"Phase 39 fold-local encoder benchmark: symbols={len(symbols)} folds={len(folds)} "
        f"epochs={config.epochs} max_windows={config.max_windows or 'all'} device={config.device}"
    )
    smoke_run = len(folds) < EXPECTED_FULL_FOLDS or config.epochs < FULL_PROTOCOL_EPOCHS or config.max_windows > 0
    run_kind = "smoke" if smoke_run else "full"
    for fold, (train_end, test_start, test_end) in enumerate(folds, start=1):
        print(f"\nFold {fold:02d}: train_end={train_end} test=[{test_start}, {test_end})")
        bounds = make_fold_bounds(fold, train_end, test_start, test_end, config)
        train_ids, test_ids = row_ids_for_fold(df_by_row, train_end, test_start, test_end)
        if not train_ids or not test_ids:
            raise RuntimeError(f"Fold {fold} has empty train or test rows.")

        vanilla = fit_fold_encoder("vanilla", raw_matrices, bounds, config)
        guided = fit_fold_encoder("guided", raw_matrices, bounds, config)
        fold_frame = frame[frame["row_id"].isin(train_ids + test_ids)].copy()
        vanilla_embeddings = encode_causal_rows(
            vanilla.model,
            raw_matrices,
            vanilla.scaler,
            fold_frame,
            config.device,
            total_rows=len(frame),
        )
        guided_embeddings = encode_causal_rows(
            guided.model,
            raw_matrices,
            guided.scaler,
            fold_frame,
            config.device,
            total_rows=len(frame),
        )

        fold_dir = heavy_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(vanilla.model.state_dict(), fold_dir / "vanilla_encoder.pt")
        torch.save(guided.model.state_dict(), fold_dir / "guided_encoder.pt")
        (fold_dir / "bounds.json").write_text(
            json.dumps(asdict(bounds), indent=2, sort_keys=True), encoding="utf-8"
        )

        for result in [vanilla, guided]:
            manifest_rows.append(
                {
                    "fold": fold,
                    "encoder_method": result.method,
                    **asdict(bounds),
                    "seed": config.seed,
                    "run_kind": run_kind,
                    "window_cap": config.max_windows,
                    "selected_epoch": result.selected_epoch,
                    "max_epochs": config.epochs,
                    "train_windows": result.train_windows,
                    "validation_windows": result.validation_windows,
                    "scaler_training_rows": result.scaler.training_rows,
                    "device": config.device,
                    "runtime_seconds": result.runtime_seconds,
                    "input_hash": result.input_hash,
                    "model_hash": result.model_hash,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "python_version": platform.python_version(),
                    "torch_version": torch.__version__,
                }
            )
            for stage, history in [
                ("inner_selection", result.selection_history),
                ("outer_refit", result.refit_history),
            ]:
                history = history.copy()
                history.insert(0, "stage", stage)
                history.insert(0, "encoder_method", result.method)
                history.insert(0, "fold", fold)
                loss_rows.extend(history.to_dict("records"))

        fold_outputs = fit_fold_assignments(
            df_by_row,
            vanilla_embeddings,
            raw_matrix,
            hmm_matrix,
            train_ids,
            test_ids,
            fold,
            guided_embeddings=guided_embeddings,
        )
        global_prediction = run_global_fold(df_by_row, train_ids, test_ids, fold)
        prediction_parts["global_lgbm"].append(global_prediction)
        coverage_rows.append(
            {
                "fold": fold,
                "method": "global_lgbm",
                "train_rows": len(train_ids),
                "test_assignment_rows": len(test_ids),
                "test_prediction_rows": len(global_prediction),
            }
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
            prediction = predict_regime_models(
                df_by_row, output.assignments, models, test_ids, output.method, fold
            )
            if prediction is None:
                raise RuntimeError(f"No prediction for fold {fold} method {output.method}.")
            method_name = f"regime_lgbm_{output.method}"
            prediction_parts[method_name].append(prediction)
            coverage_rows.append(
                {
                    "fold": fold,
                    "method": method_name,
                    "train_rows": int((output.assignments["split"] == "train").sum()),
                    "test_assignment_rows": int((output.assignments["split"] == "test").sum()),
                    "test_prediction_rows": len(prediction),
                }
            )

    outputs = []
    for method in sorted(EXPECTED_METHODS):
        prediction = pd.concat(prediction_parts[method], ignore_index=True)
        regime_method = "none" if method == "global_lgbm" else method.removeprefix("regime_lgbm_")
        outputs.append(PredictionSet(method, regime_method, prediction))
    predictions = pd.concat([output.frame for output in outputs], ignore_index=True)
    predictions = predictions.sort_values(["method", "fold", "open_time", "symbol"]).reset_index(drop=True)
    results = pd.DataFrame(
        [
            summarize_predictions(
                output.frame, output.method, output.regime_method, "+".join(symbols)
            )
            for output in outputs
        ]
    )
    validate_complete_coverage(results, predictions, len(folds))

    assignments = pd.concat(test_assignments, ignore_index=True)
    implementations = pd.DataFrame(implementation_rows)
    regime_summary = summarize_assignments(assignments, implementations)
    manifest = pd.DataFrame(manifest_rows)
    losses = pd.DataFrame(loss_rows)
    coverage = pd.DataFrame(coverage_rows)
    comparison = build_historical_comparison(results)
    guided_comparison = build_guided_alpha_comparison(results)

    manifest.to_csv(output_path(args.output_prefix, "encoder_manifest"), index=False)
    losses.to_csv(output_path(args.output_prefix, "encoder_loss"), index=False)
    coverage.to_csv(output_path(args.output_prefix, "encoder_coverage"), index=False)
    results.to_csv(output_path(args.output_prefix, "experiment_results"), index=False)
    comparison.to_csv(output_path(args.output_prefix, "method_comparison"), index=False)
    regime_summary.to_csv(output_path(args.output_prefix, "regime_summary"), index=False)
    guided_comparison.to_csv(output_path(args.output_prefix, "guided_comparison"), index=False)
    predictions.to_csv(output_path(args.output_prefix, "alpha_oos_predictions"), index=False)
    assignments.to_csv(output_path(args.output_prefix, "regime_assignments"), index=False)

    report_path = Path("reports") / "phase39_fold_local_results.md"
    write_results_report(
        report_path,
        results,
        manifest,
        len(folds),
        symbols,
        smoke=smoke_run,
    )
    run_metadata = {
        "args": vars(args),
        "config": asdict(config),
        "symbols": symbols,
        "folds": len(folds),
        "expected_methods": sorted(EXPECTED_METHODS),
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
    }
    (heavy_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nPhase 39 fold-local experiment results:")
    print(results.to_string(index=False))
    print(f"\nSaved compact report: {report_path}")
    print("OK: Phase 39 fold-local validity and equal-coverage checks passed.")


if __name__ == "__main__":
    main()
