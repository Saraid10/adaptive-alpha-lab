from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def read_csv(name: str) -> pd.DataFrame:
    path = MODELS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_repo_csv(relative_path: str) -> pd.DataFrame:
    path = BASE_DIR / relative_path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError:
        print("Streamlit is not installed. Run: pip install -r requirements.txt")
        return

    st.set_page_config(page_title="Adaptive Alpha Lab", layout="wide")
    st.title("Adaptive Alpha Lab")
    st.caption(
        "Regime-aware quant ML benchmark platform with financial labels, "
        "baselines, purged validation, and transaction-cost-aware evaluation."
    )
    st.info(
        "Key finding: contrastive-HMM improves the learned-regime path, "
        "but the raw-feature Gaussian HMM still leads on IC and Sharpe. "
        "The current evidence says HMM-style temporal state structure helps "
        "learned embeddings, while representation capacity alone is not enough."
    )

    results = read_csv("experiment_results.csv")
    walkforward_results = read_csv("walkforward_experiment_results.csv")
    walkforward_comparison = read_csv("walkforward_comparison.csv")
    walkforward_regime_summary = read_csv("walkforward_regime_summary.csv")
    robustness_results = read_csv("robustness_results.csv")
    robustness_summary = read_csv("robustness_summary.csv")
    robustness_wins = read_csv("robustness_wins.csv")
    stress_results = read_csv("robustness_stress_results.csv")
    stress_summary = read_csv("robustness_stress_summary.csv")
    stress_wins = read_csv("robustness_stress_wins.csv")
    regime_summary = read_csv("regime_benchmark_summary.csv")
    regime_stability = read_csv("regime_stability_summary.csv")
    per_regime = read_csv("per_regime_stats.csv")
    validation_audit = read_csv("validation_audit.csv")
    fold_audit = read_csv("fold_audit.csv")
    target_dist = read_csv("target_distribution.csv")
    target_quality = read_csv("target_quality.csv")
    run_index = read_repo_csv("runs/run_index.csv")

    st.header("Experiment Results")
    if results.empty:
        st.info("Run python src/alpha_models.py to generate experiment_results.csv.")
    else:
        st.dataframe(results, width="stretch")
        c1, c2, c3, c4 = st.columns(4)
        best_ic = results.sort_values("IC", ascending=False).iloc[0]
        best_dd = results.sort_values("drawdown", ascending=False).iloc[0]
        c1.metric("Best IC", f"{best_ic['IC']:.4f}", best_ic["method"])
        c2.metric("Best Sharpe", f"{results['Sharpe'].max():.3f}")
        c3.metric("Lowest Drawdown", f"{best_dd['drawdown']:.2%}", best_dd["method"])
        c4.metric("Methods Tested", len(results))

    st.header("Fold-Local Regime Refit")
    if walkforward_results.empty:
        st.info("Run python src/walkforward_regimes.py to generate fold-local regime results.")
    else:
        st.caption(
            "Strict benchmark where regime assignment models are refit inside each "
            "walk-forward fold using training history only."
        )
        st.dataframe(walkforward_results, width="stretch")
        c1, c2, c3, c4 = st.columns(4)
        best_ic = walkforward_results.sort_values("IC", ascending=False).iloc[0]
        best_sharpe = walkforward_results.sort_values("Sharpe", ascending=False).iloc[0]
        c1.metric("Best Fold-Local IC", f"{best_ic['IC']:.4f}", best_ic["method"])
        c2.metric("Best Fold-Local Sharpe", f"{best_sharpe['Sharpe']:.3f}", best_sharpe["method"])
        c3.metric("Fold-Local Methods", len(walkforward_results))
        c4.metric("Rows / Method", int(walkforward_results["n_test_rows"].max()))
        if not walkforward_comparison.empty:
            st.subheader("Offline vs Fold-Local")
            st.dataframe(walkforward_comparison, width="stretch")
        if not walkforward_regime_summary.empty:
            st.subheader("Fold-Local Regime Summary")
            st.dataframe(walkforward_regime_summary, width="stretch")
        walkforward_img = MODELS_DIR / "walkforward_equity_curve.png"
        if walkforward_img.exists():
            st.image(str(walkforward_img), width="stretch")

    st.header("Robustness Matrix")
    if robustness_summary.empty:
        st.info("Run python src/robustness.py to generate Phase 14A robustness artifacts.")
    else:
        st.caption(
            "Phase 14A repeats the fold-local benchmark across BTC-only, ETH-only, "
            "and BTC+ETH scopes at 4h, 8h, and 24h horizons."
        )
        st.dataframe(robustness_summary, width="stretch")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Grid Cells", len(robustness_summary))
        if robustness_wins.empty:
            c2.metric("Most IC Wins", "missing")
            c3.metric("Most Sharpe Wins", "missing")
            c4.metric("Most Drawdown Wins", "missing")
        else:
            ic_leader = robustness_wins[robustness_wins["metric"] == "IC"].sort_values("wins", ascending=False).iloc[0]
            sharpe_leader = robustness_wins[robustness_wins["metric"] == "Sharpe"].sort_values("wins", ascending=False).iloc[0]
            drawdown_leader = robustness_wins[robustness_wins["metric"] == "drawdown"].sort_values("wins", ascending=False).iloc[0]
            c2.metric("Most IC Wins", int(ic_leader["wins"]), ic_leader["method"])
            c3.metric("Most Sharpe Wins", int(sharpe_leader["wins"]), sharpe_leader["method"])
            c4.metric("Most Drawdown Wins", int(drawdown_leader["wins"]), drawdown_leader["method"])
        if not robustness_wins.empty:
            st.subheader("Robustness Win Counts")
            st.dataframe(robustness_wins, width="stretch")
        if not robustness_results.empty:
            st.subheader("Full Robustness Results")
            st.dataframe(robustness_results, width="stretch")
        robustness_img = MODELS_DIR / "robustness_heatmap.png"
        if robustness_img.exists():
            st.image(str(robustness_img), width="stretch")

    st.header("Stress Robustness")
    if stress_summary.empty:
        st.info("Run python src/robustness_stress.py to generate Phase 14B stress artifacts.")
    else:
        st.caption(
            "Phase 14B re-scores fold-local predictions across transaction costs, "
            "signal thresholds, and bull/sideways/bear market periods."
        )
        st.dataframe(stress_summary, width="stretch")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stress Cells", len(stress_summary))
        if stress_wins.empty:
            c2.metric("Signal IC Leader", "missing")
            c3.metric("Sharpe Leader", "missing")
            c4.metric("Return Leader", "missing")
        else:
            signal_leader = stress_wins[stress_wins["metric"] == "signal_IC"].sort_values("wins", ascending=False).iloc[0]
            sharpe_leader = stress_wins[stress_wins["metric"] == "Sharpe"].sort_values("wins", ascending=False).iloc[0]
            return_leader = stress_wins[stress_wins["metric"] == "total_return"].sort_values("wins", ascending=False).iloc[0]
            c2.metric("Signal IC Leader", int(signal_leader["wins"]), signal_leader["method"])
            c3.metric("Sharpe Leader", int(sharpe_leader["wins"]), sharpe_leader["method"])
            c4.metric("Return Leader", int(return_leader["wins"]), return_leader["method"])
        if not stress_wins.empty:
            st.subheader("Stress Win Counts")
            st.dataframe(stress_wins, width="stretch")
        if not stress_results.empty:
            st.subheader("Full Stress Results")
            st.dataframe(stress_results, width="stretch")
        stress_img = MODELS_DIR / "robustness_stress_heatmap.png"
        if stress_img.exists():
            st.image(str(stress_img), width="stretch")

    st.header("Run Registry")
    if run_index.empty:
        st.info("Run python src/archive_run.py to freeze a versioned research snapshot.")
    else:
        st.caption("Versioned snapshots keep baseline artifacts from being overwritten by later experiment runs.")
        latest_run = run_index.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registered Runs", len(run_index))
        c2.metric("Latest Run", latest_run["run_id"])
        c3.metric("Artifacts", int(latest_run["artifact_count"]))
        c4.metric("Missing", int(latest_run["missing_artifact_count"]))
        st.dataframe(run_index, width="stretch")

    st.header("Validation Audit")
    if validation_audit.empty:
        st.info("Run python src/validation_audit.py to generate validation_audit.csv.")
    else:
        pass_count = int((validation_audit["status"] == "PASS").sum())
        warn_count = int((validation_audit["status"] == "WARN").sum())
        fail_count = int((validation_audit["status"] == "FAIL").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Passed Checks", pass_count)
        c2.metric("Warnings", warn_count)
        c3.metric("Failures", fail_count)
        st.dataframe(validation_audit, width="stretch")
        if not fold_audit.empty:
            st.subheader("Fold Audit")
            st.dataframe(fold_audit, width="stretch")

    st.header("Targets")
    if target_dist.empty:
        st.info("Run python src/targets.py to generate target_distribution.csv.")
    else:
        st.dataframe(target_dist, width="stretch")
        if not target_quality.empty:
            st.subheader("Target Quality")
            st.dataframe(target_quality, width="stretch")
        img = MODELS_DIR / "target_distribution.png"
        if img.exists():
            st.image(str(img), caption="Target label distribution")

    st.header("Regime Benchmarking")
    if regime_summary.empty:
        st.info("Run python src/baselines.py to generate regime benchmark artifacts.")
    else:
        st.dataframe(regime_summary, width="stretch")
        if not regime_stability.empty:
            st.subheader("Regime Stability Diagnostics")
            st.dataframe(regime_stability, width="stretch")
            stability_img = MODELS_DIR / "regime_stability.png"
            if stability_img.exists():
                st.image(str(stability_img), width="stretch")
        st.subheader("Per-Regime Statistics")
        st.dataframe(per_regime, width="stretch")
        st.subheader("Transition Matrices")
        matrix_methods = ["contrastive", "contrastive_hmm", "hmm", "kmeans", "vol_bucket"]
        matrix_cols = st.columns(len(matrix_methods))
        for col, method in zip(matrix_cols, matrix_methods):
            path = MODELS_DIR / f"transition_matrix_{method}.png"
            with col:
                st.caption(method)
                if path.exists():
                    st.image(str(path))
                else:
                    st.info("missing")

    st.header("Visual Artifacts")
    images = [
        ("UMAP Regime Structure", "umap_improved.png"),
        ("Regime Timeline", "regime_timeline.png"),
        ("Backtest Dashboard", "phase4_dashboard.png"),
        ("Equity Curve", "equity_curve.png"),
    ]
    for title, name in images:
        path = MODELS_DIR / name
        if path.exists():
            st.subheader(title)
            st.image(str(path), width="stretch")


if __name__ == "__main__":
    main()
