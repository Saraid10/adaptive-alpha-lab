from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def read_csv(name: str) -> pd.DataFrame:
    path = MODELS_DIR / name
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
    regime_summary = read_csv("regime_benchmark_summary.csv")
    regime_stability = read_csv("regime_stability_summary.csv")
    per_regime = read_csv("per_regime_stats.csv")
    validation_audit = read_csv("validation_audit.csv")
    fold_audit = read_csv("fold_audit.csv")
    target_dist = read_csv("target_distribution.csv")
    target_quality = read_csv("target_quality.csv")

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
