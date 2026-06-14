import argparse
import csv
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
PAPER_DIR = BASE_DIR / "paper"

METHOD_LABELS = {
    "global_lgbm": "Global LightGBM",
    "regime_lgbm_hmm": "Raw-feature HMM + regime LightGBM",
    "regime_lgbm_kmeans": "KMeans + regime LightGBM",
    "regime_lgbm_vol_bucket": "Volatility buckets + regime LightGBM",
    "regime_lgbm_contrastive": "Vanilla contrastive-GMM + regime LightGBM",
    "regime_lgbm_contrastive_hmm": "Vanilla contrastive-HMM + regime LightGBM",
    "regime_lgbm_hmm_guided_gmm": "HMM-guided contrastive-GMM + regime LightGBM",
    "regime_lgbm_hmm_guided_hmm": "HMM-guided contrastive-HMM + regime LightGBM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Phase 27 manuscript skeleton and paper artifact map."
    )
    parser.add_argument("--output-dir", default=str(PAPER_DIR), help="Directory for paper/main.md.")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def pct(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except Exception:
        return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_Artifact not available._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df[columns].iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def build_main_result_table(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    preferred = [
        "regime_lgbm_hmm_guided_hmm",
        "regime_lgbm_hmm",
        "regime_lgbm_contrastive_hmm",
        "regime_lgbm_hmm_guided_gmm",
        "regime_lgbm_contrastive",
        "global_lgbm",
    ]
    ordered = results.copy()
    ordered["_order"] = ordered["method"].map({name: idx for idx, name in enumerate(preferred)}).fillna(99)
    ordered = ordered.sort_values(["_order", "method"]).drop(columns="_order")
    return pd.DataFrame(
        {
            "Method": ordered["method"].map(METHOD_LABELS).fillna(ordered["method"]),
            "IC": ordered["IC"].map(fmt),
            "Sharpe": ordered["Sharpe"].map(fmt),
            "Drawdown": ordered["drawdown"].map(lambda value: pct(value, 1)),
            "Total Return": ordered["total_return"].map(lambda value: pct(value, 1)),
            "Turnover": ordered["turnover"].map(fmt),
            "Rows": ordered["n_test_rows"].astype(str),
        }
    )


def build_claim_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    focus = summary[
        summary["comparison"].isin(
            [
                "guided_hmm_alpha_vs_raw_feature_hmm_alpha",
                "guided_hmm_alpha_vs_guided_gmm_alpha",
                "guided_hmm_alpha_vs_vanilla_contrastive_hmm_alpha",
                "time_frequency_guided_hmm_vs_time_only_guided_hmm",
            ]
        )
    ].copy()
    if focus.empty:
        focus = summary.copy()
    return pd.DataFrame(
        {
            "Comparison": focus["comparison"],
            "Status": focus["paper_status"],
            "IC Diff": focus["ic_mean_difference"].map(fmt),
            "IC p": focus["ic_p_value"].map(fmt),
            "Win Rate": focus["metric_win_rate"].map(lambda value: pct(value, 0)),
            "Allowed Language": focus["allowed_paper_language"],
        }
    )


def build_artifact_map() -> list[dict[str, str]]:
    return [
        {
            "paper_section": "Abstract and contributions",
            "artifact_type": "claim control",
            "artifact": "reports/claim_registry.md",
            "paper_role": "Prevents unsupported novelty, profitability, and generalization claims.",
        },
        {
            "paper_section": "Related work",
            "artifact_type": "literature",
            "artifact": "reports/related_work.md; reports/literature_matrix.csv",
            "paper_role": "Positions the project against regime models, time-series contrastive learning, validation, and alpha modeling.",
        },
        {
            "paper_section": "Data and labels",
            "artifact_type": "data diagnostic",
            "artifact": "models/target_distribution.csv; models/target_quality.csv",
            "paper_role": "Documents class balance, neutral share, and horizon-tail loss for financial labels.",
        },
        {
            "paper_section": "Validation",
            "artifact_type": "audit",
            "artifact": "models/validation_audit.csv; models/fold_audit.csv",
            "paper_role": "Documents embargo, label-horizon purge, common coverage, and artifact availability.",
        },
        {
            "paper_section": "Methods",
            "artifact_type": "model card",
            "artifact": "reports/model_card.md; reports/paper_protocol.md",
            "paper_role": "Records architecture, frozen protocol, methods, metrics, and forbidden claims.",
        },
        {
            "paper_section": "Main results",
            "artifact_type": "result table",
            "artifact": "models/walkforward_experiment_results.csv; models/guided_alpha_comparison.csv",
            "paper_role": "Primary fold-local alpha comparison for global, classical, vanilla learned, and guided learned regimes.",
        },
        {
            "paper_section": "Statistical evidence",
            "artifact_type": "statistical test",
            "artifact": "models/statistical_test_summary.csv; models/paper_statistical_summary.csv",
            "paper_role": "Separates point-estimate wins from fold-level significance and multiple-testing-safe claims.",
        },
        {
            "paper_section": "Mechanism",
            "artifact_type": "regime quality",
            "artifact": "models/regime_quality_summary.csv; models/guided_encoder_comparison.csv",
            "paper_role": "Measures structural alignment, persistence, entropy, NMI, ARI, and purity.",
        },
        {
            "paper_section": "Ablations",
            "artifact_type": "ablation",
            "artifact": "models/ablation_summary.csv; models/paper_claim_tests.csv",
            "paper_role": "Tests objective guidance, assignment layer, augmentation view, and classical-reference comparisons.",
        },
        {
            "paper_section": "Robustness",
            "artifact_type": "stress test",
            "artifact": "models/robustness_summary.csv; models/robustness_stress_summary.csv",
            "paper_role": "Checks symbol/horizon robustness and cost/threshold/period sensitivity.",
        },
        {
            "paper_section": "Interpretability",
            "artifact_type": "feature attribution",
            "artifact": "models/feature_importance_by_regime.csv; models/feature_family_summary.csv",
            "paper_role": "Shows fold-local feature drivers by method/regime without making causal claims.",
        },
    ]


def write_artifact_map(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["paper_section", "artifact_type", "artifact", "paper_role"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_checklist() -> str:
    return """# Paper Submission Checklist

## Phase 27 Status

This checklist tracks what remains before the project should be treated as a submission-ready paper package.

## Ready

- Central research question is frozen in `reports/paper_protocol.md`.
- Claim boundaries are frozen in `reports/claim_registry.md`.
- Literature positioning exists in `reports/related_work.md` and `reports/literature_matrix.csv`.
- Main fold-local benchmark artifacts exist.
- Phase 25 ablations and Phase 26 paper claim tests exist.
- Validation audit has no critical failures.

## Needs Human Writing

- Convert `paper/main.md` from scaffold to prose.
- Replace placeholder citations with the final venue citation style.
- Add final figure numbers after choosing the paper template.
- Tighten the abstract after the final submission venue is selected.
- Decide whether the appendix includes the Streamlit dashboard screenshots.

## Optional Before Submission

- Multi-asset generalization only if scope and compute allow it.
- Fold-local encoder retraining only if the paper needs a stronger leakage-resistance appendix.
- Full time-frequency encoder run only if Phase 25/26 gates are reopened.

## Must Not Claim

- Do not claim HMM states are ground truth.
- Do not claim a profitable or deployable trading strategy.
- Do not claim guided-HMM statistically dominates raw-feature HMM at 5%.
- Do not claim generalization outside BTC/ETH without new experiments.
"""


def build_paper(results: pd.DataFrame, claims: pd.DataFrame, artifact_rows: list[dict[str, str]]) -> str:
    main_results = build_main_result_table(results)
    claim_table = build_claim_table(claims)
    artifact_df = pd.DataFrame(artifact_rows)
    best_method = "not available"
    if not results.empty and "IC" in results.columns:
        row = results.sort_values("IC", ascending=False).iloc[0]
        best_method = f"{row['method']} (IC={fmt(row['IC'])}, Sharpe={fmt(row['Sharpe'])})"

    return f"""# HMM-Guided Contrastive Representations for Regime-Conditioned Financial Alpha Modeling

## Paper Status

Phase 27 manuscript skeleton. This is a structured draft generated from the current artifacts, not the final submission text.

## Abstract Draft

Financial market regimes are often modeled with classical sequential methods such as Hidden Markov Models, while recent deep time-series encoders promise richer learned representations. This project asks whether HMM-guided contrastive representations improve regime-conditioned financial alpha modeling compared with vanilla contrastive regimes, raw-feature HMM regimes, and global no-regime LightGBM baselines. The benchmark uses BTCUSDT and ETHUSDT hourly data, triple-barrier labels, expanding purged walk-forward validation, transaction costs, robustness grids, fold-level statistical tests, and fold-local interpretability. The strongest current point-estimate method is {best_method}. However, fold-level statistical dominance over the raw-feature HMM baseline remains inconclusive. The main contribution is therefore a controlled empirical finding: sequential assignment and HMM-guided weak supervision improve the learned-regime path, but the evidence supports cautious mechanism claims rather than a deployable trading claim.

## 1. Introduction

Market regimes change across time, and alpha models that work in one state can fail in another. A common solution is to detect regimes with an HMM and train or weight models by state. A separate line of work uses contrastive learning to discover time-series representations without hand labels. Adaptive Alpha Lab connects these two lines: it tests whether learned regime embeddings become more useful when guided by classical sequential structure.

### Contributions

1. A reproducible regime-conditioned alpha benchmark using common financial labels, common folds, common transaction costs, and common test rows.
2. A comparison between global LightGBM, classical regime baselines, vanilla contrastive regimes, and HMM-guided learned regimes.
3. A model-side intervention: HMM states are used as weak supervision for contrastive representation learning, while still being treated as proxy states rather than ground truth.
4. A paper-safe evidence stack with validation audit, robustness, statistical tests, ablations, and fold-local interpretability.
5. A cautious finding: guided-HMM produces the strongest current point estimates and stress robustness, but the fold-level IC edge over raw-feature HMM is not statistically significant at 5%.

## 2. Related Work

This paper should position itself across four literature clusters:

- Classical financial regime detection: HMMs, Gaussian mixtures, volatility regimes, and regime-switching models.
- Contrastive time-series representation learning: vanilla temporal contrastive objectives, TS2Vec/TNC/CoST-style representation learning, and recent time-frequency or hard-negative variants.
- Financial ML validation: triple-barrier labels, purging, embargoing, walk-forward evaluation, multiple-testing caution, and Probabilistic Sharpe diagnostics.
- Regime-conditioned alpha modeling: using state information to train, select, or weight predictive models.

The important limitation is philosophical: raw-feature HMM states are not true market-regime labels. They are a classical sequential reference used to test whether learned embeddings benefit from temporal state discipline.

## 3. Data and Labels

The current paper dataset contains BTCUSDT and ETHUSDT hourly bars from 2024-04-26 to 2026-04-26. The benchmark uses 22 engineered features and the primary target `tb_label_8h`. Secondary labels and horizons are kept as diagnostics and robustness artifacts.

Target and feature diagnostics should cite `models/target_distribution.csv`, `models/target_quality.csv`, and `src/features.py`.

## 4. Methods

### 4.1 Baselines

- `global_lgbm`: no-regime LightGBM baseline.
- `regime_lgbm_hmm`: raw-feature Gaussian HMM regime baseline.
- `regime_lgbm_kmeans`: non-sequential clustering baseline.
- `regime_lgbm_vol_bucket`: simple volatility-state baseline.
- `regime_lgbm_contrastive`: vanilla contrastive embedding with GMM assignment.
- `regime_lgbm_contrastive_hmm`: vanilla contrastive embedding with HMM assignment.

### 4.2 Proposed Method

The proposed method is `regime_lgbm_hmm_guided_hmm`: an HMM-guided contrastive encoder followed by sequential HMM assignment and regime-conditioned LightGBM alpha models. The HMM state sequence is used as weak supervision, not as ground truth.

### 4.3 Alpha Modeling

All alpha models use the same primary target, `tb_label_8h`, and the same walk-forward folds. The alpha score is derived from multiclass probabilities as `P(up) - P(down)`, with transaction costs applied in the evaluation layer.

## 5. Validation and Statistical Protocol

The predictive benchmark uses expanding walk-forward validation, a six-month initial training window, one-month test steps, a five-day embargo, and an eight-bar label-horizon purge. Paper claims should use fold-local regime assignments and `models/walkforward_experiment_results.csv`.

Statistical interpretation is fold-level first. Row-level diagnostics are useful for calibration or forecast-loss discussion but should not replace fold-level evidence because financial labels overlap across time.

## 6. Results

### 6.1 Main Fold-Local Alpha Results

{markdown_table(main_results, list(main_results.columns)) if not main_results.empty else "_Main result artifact missing._"}

### 6.2 Paper Claim Tests

{markdown_table(claim_table, list(claim_table.columns)) if not claim_table.empty else "_Paper claim artifact missing._"}

### 6.3 Current Interpretation

The current results support the mechanism that sequential assignment matters. HMM assignment improves the guided learned-regime path relative to guided-GMM on all focused point-estimate metrics and is raw-suggestive on fold-level IC, but not corrected significant. Guided-HMM also improves all focused point-estimate metrics versus raw-feature HMM, but the IC p-value remains too weak for a statistical dominance claim.

## 7. Robustness

The paper should discuss two robustness layers:

- Symbol/horizon robustness: useful for showing where the conclusion is stable or mixed.
- Cost/threshold/market-period stress robustness: stronger for the primary BTC+ETH 8h setup.

The safe wording is that guided-HMM is stress-robust on the primary BTC+ETH 8h benchmark, not universally dominant across all assets and horizons.

## 8. Interpretability

Fold-local feature attribution shows which feature families matter inside each method/regime. This should be framed as diagnostic interpretability, not causal explanation. The strongest paper use is to show whether the learned regimes rely on economically plausible features such as volatility state, momentum/autocorrelation, and distribution shape.

## 9. Ablations

The ablation suite tests objective guidance, assignment layer, augmentation view, and classical-reference comparisons. The current evidence says the assignment layer is the strongest mechanism. The time-frequency prototype is not strong enough to justify expansion yet.

## 10. Limitations

- HMM states are proxy states, not ground truth.
- The benchmark currently covers BTCUSDT and ETHUSDT only.
- The guided-HMM edge over raw-feature HMM is directionally supported but not statistically significant at 5%.
- Backtest results are research diagnostics, not live trading claims.
- Interpretability results are model-specific and not causal.
- The encoder is trained offline; fold-local encoder retraining remains a possible appendix experiment if compute allows.

## 11. Conclusion Draft

Adaptive Alpha Lab shows that learned market-regime representations need sequential discipline to become useful in this benchmark. Vanilla contrastive-GMM regimes are weak downstream, while HMM-guided embeddings paired with HMM assignment produce the strongest current point estimates and stress robustness. The central publishable finding is not a claim of profitable trading or statistical dominance, but a controlled empirical result: classical sequential structure can improve deep learned regime representations, and the assignment layer is a major driver of downstream usefulness.

## Figure and Table Plan

{markdown_table(artifact_df, ["paper_section", "artifact_type", "artifact", "paper_role"])}
"""


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = read_csv(MODELS_DIR / "walkforward_experiment_results.csv")
    claims = read_csv(MODELS_DIR / "paper_statistical_summary.csv")
    artifact_rows = build_artifact_map()

    paper_path = output_dir / "main.md"
    artifact_map_path = REPORTS_DIR / "paper_artifact_map.csv"
    checklist_path = REPORTS_DIR / "paper_submission_checklist.md"

    paper_path.write_text(build_paper(results, claims, artifact_rows), encoding="utf-8")
    write_artifact_map(artifact_map_path, artifact_rows)
    checklist_path.write_text(build_checklist(), encoding="utf-8")

    print(f"Saved paper skeleton: {paper_path}")
    print(f"Saved artifact map: {artifact_map_path}")
    print(f"Saved submission checklist: {checklist_path}")


if __name__ == "__main__":
    main()
