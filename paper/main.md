# HMM-Guided Contrastive Representations for Regime-Conditioned Financial Alpha Modeling

## Paper Status

Phase 27 manuscript skeleton. This is a structured draft generated from the current artifacts, not the final submission text.

## Abstract Draft

Financial market regimes are often modeled with classical sequential methods such as Hidden Markov Models, while recent deep time-series encoders promise richer learned representations. This project asks whether HMM-guided contrastive representations improve regime-conditioned financial alpha modeling compared with vanilla contrastive regimes, raw-feature HMM regimes, and global no-regime LightGBM baselines. The benchmark uses BTCUSDT and ETHUSDT hourly data, triple-barrier labels, expanding purged walk-forward validation, transaction costs, robustness grids, fold-level statistical tests, and fold-local interpretability. The strongest current point-estimate method is regime_lgbm_hmm_guided_hmm (IC=0.0094, Sharpe=0.0989). However, fold-level statistical dominance over the raw-feature HMM baseline remains inconclusive. The main contribution is therefore a controlled empirical finding: sequential assignment and HMM-guided weak supervision improve the learned-regime path, but the evidence supports cautious mechanism claims rather than a deployable trading claim.

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

| Method | IC | Sharpe | Drawdown | Total Return | Turnover | Rows |
| --- | --- | --- | --- | --- | --- | --- |
| HMM-guided contrastive-HMM + regime LightGBM | 0.0094 | 0.0989 | -61.4% | 3.1% | 0.0845 | 25920 |
| Raw-feature HMM + regime LightGBM | 0.0051 | -0.3405 | -71.0% | -53.6% | 0.0789 | 25920 |
| Vanilla contrastive-HMM + regime LightGBM | -0.0026 | -0.5476 | -77.8% | -68.5% | 0.0765 | 25920 |
| HMM-guided contrastive-GMM + regime LightGBM | -0.0092 | -0.9764 | -90.0% | -85.4% | 0.0787 | 25920 |
| Vanilla contrastive-GMM + regime LightGBM | -0.0110 | -0.8336 | -92.6% | -82.3% | 0.0737 | 25920 |
| Global LightGBM | 0.0024 | -0.5056 | -68.8% | -55.7% | 0.0503 | 25920 |
| KMeans + regime LightGBM | 0.0072 | -0.7282 | -86.0% | -79.7% | 0.0808 | 25920 |
| Volatility buckets + regime LightGBM | -0.0020 | -0.8203 | -85.4% | -82.0% | 0.0829 | 25920 |

### 6.2 Paper Claim Tests

| Comparison | Status | IC Diff | IC p | Win Rate | Allowed Language |
| --- | --- | --- | --- | --- | --- |
| guided_hmm_alpha_vs_guided_gmm_alpha | raw_suggestive | 0.0155 | 0.0755 | 100% | guided_hmm_alpha_vs_guided_gmm_alpha is suggestive before correction and should be framed as exploratory. |
| time_frequency_guided_hmm_vs_time_only_guided_hmm | do_not_expand_yet |  |  | 33% | time_frequency_guided_hmm_vs_time_only_guided_hmm is not strong enough to justify downstream expansion yet. |
| guided_hmm_alpha_vs_raw_feature_hmm_alpha | directionally_supported | 0.0022 | 0.8008 | 100% | guided_hmm_alpha_vs_raw_feature_hmm_alpha improves all focused point-estimate metrics but lacks statistical significance. |
| guided_hmm_alpha_vs_vanilla_contrastive_hmm_alpha | directionally_supported | 0.0143 | 0.2612 | 100% | guided_hmm_alpha_vs_vanilla_contrastive_hmm_alpha improves all focused point-estimate metrics but lacks statistical significance. |

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

## 11. Reproducibility

The public repository separates dashboard reproduction from full research reproduction. The dashboard runs from curated committed summary artifacts and minimal Streamlit dependencies. Full research reproduction uses `requirements-research.txt`, local data access, encoder training, fold-local validation, robustness checks, statistical tests, and validation audit.

The reproduction helper supports three modes: smoke, full, and dashboard. Raw data, DuckDB databases, model weights, embeddings, and row-level prediction files remain excluded from GitHub.

## 12. Conclusion Draft

Adaptive Alpha Lab shows that learned market-regime representations need sequential discipline to become useful in this benchmark. Vanilla contrastive-GMM regimes are weak downstream, while HMM-guided embeddings paired with HMM assignment produce the strongest current point estimates and stress robustness. The central publishable finding is not a claim of profitable trading or statistical dominance, but a controlled empirical result: classical sequential structure can improve deep learned regime representations, and the assignment layer is a major driver of downstream usefulness.

## Figure and Table Plan

| paper_section | artifact_type | artifact | paper_role |
| --- | --- | --- | --- |
| Abstract and contributions | claim control | reports/claim_registry.md | Prevents unsupported novelty, profitability, and generalization claims. |
| Related work | literature | reports/related_work.md; reports/literature_matrix.csv | Positions the project against regime models, time-series contrastive learning, validation, and alpha modeling. |
| Data and labels | data diagnostic | models/target_distribution.csv; models/target_quality.csv | Documents class balance, neutral share, and horizon-tail loss for financial labels. |
| Validation | audit | models/validation_audit.csv; models/fold_audit.csv | Documents embargo, label-horizon purge, common coverage, and artifact availability. |
| Methods | model card | reports/model_card.md; reports/paper_protocol.md | Records architecture, frozen protocol, methods, metrics, and forbidden claims. |
| Main results | result table | models/walkforward_experiment_results.csv; models/guided_alpha_comparison.csv | Primary fold-local alpha comparison for global, classical, vanilla learned, and guided learned regimes. |
| Statistical evidence | statistical test | models/statistical_test_summary.csv; models/paper_statistical_summary.csv | Separates point-estimate wins from fold-level significance and multiple-testing-safe claims. |
| Mechanism | regime quality | models/regime_quality_summary.csv; models/guided_encoder_comparison.csv | Measures structural alignment, persistence, entropy, NMI, ARI, and purity. |
| Ablations | ablation | models/ablation_summary.csv; models/paper_claim_tests.csv | Tests objective guidance, assignment layer, augmentation view, and classical-reference comparisons. |
| Robustness | stress test | models/robustness_summary.csv; models/robustness_stress_summary.csv | Checks symbol/horizon robustness and cost/threshold/period sensitivity. |
| Interpretability | feature attribution | models/feature_importance_by_regime.csv; models/feature_family_summary.csv | Shows fold-local feature drivers by method/regime without making causal claims. |
| Reproducibility | reproduction package | reproduce.ps1; reports/environment.md; reports/artifact_manifest.md; reports/reproduction_checklist.md | Documents smoke/full/dashboard reproduction paths, environment split, and artifact policy. |
