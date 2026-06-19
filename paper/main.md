# HMM-Guided Contrastive Representations for Regime-Conditioned Financial Alpha Modeling

## Paper Status

Phase 39 controlled working draft. The fully fold-local encoder pipeline has passed leakage/reproducibility tests and a one-fold smoke run. All inspected BTC/ETH and Crypto-20 outcomes remain development-observed; the full Phase 39 run and a later locked external evaluation are still pending.

## Abstract

Financial alpha models often behave differently across market regimes, yet regime labels are rarely observed and may be unstable during transitions. Classical Hidden Markov Models impose useful temporal state discipline, while contrastive time-series encoders can learn richer representations from raw features. This paper studies whether those ideas can be combined for regime-conditioned financial alpha modeling. Adaptive Alpha Lab benchmarks global LightGBM, raw-feature HMM regimes, clustering and volatility baselines, vanilla contrastive regimes, and an HMM-guided contrastive encoder. A controlled BTCUSDT/ETHUSDT pilot is followed by a pre-specified 20-asset crypto generalization study. The evaluation uses triple-barrier labels, expanding purged walk-forward validation, transaction costs, robustness grids, fold-level statistical tests, multiple-testing corrections, ablations, and fold-local interpretability. HMM-guided representations transfer structurally to Crypto-20 and produce the highest mean fold IC, but their IC advantage over raw-feature HMM is non-significant (`p=0.840`) and their calibrated multiclass loss is worse than global LightGBM after Holm correction. The contribution is therefore a reproducible empirical benchmark and a mechanism result: sequential guidance changes learned regime structure, but structural improvement does not guarantee statistically dominant or economically superior alpha.

## 1. Introduction

Market behavior is non-stationary. Momentum signals, volatility filters, and liquidity-sensitive features can all change their predictive value when the market moves from persistent trends into choppy or stressed periods. Regime-conditioned alpha modeling tries to address this by training or weighting predictors differently across market states. The difficulty is that financial regimes are latent: they are not directly labeled, and hard regime assignments can be unreliable near transitions.

The classical answer is to estimate regimes with a sequential model such as a Gaussian HMM. This gives persistent state assignments and transition probabilities, but it depends heavily on the chosen feature set and distributional assumptions. A newer answer is to learn time-series embeddings through contrastive objectives and then cluster the embedding space. This can discover nonlinear representations, but the original Adaptive Alpha Lab benchmark found that vanilla contrastive-GMM regimes were smooth yet weak for downstream alpha.

This paper studies a hybrid question: can a contrastive encoder become more useful if its training objective is guided by a classical sequential reference? The proposed path uses raw-feature HMM states as weak supervision for contrastive representation learning, then evaluates both GMM and HMM assignment layers on the learned embeddings. HMM states are treated as proxy states, not ground truth. The goal is to test whether sequential structure improves learned regimes under fair financial validation.

### Contributions

1. A reproducible regime-conditioned alpha benchmark using common labels, folds, transaction costs, and test rows across global, classical, vanilla learned, and HMM-guided learned methods.
2. An HMM-guided contrastive representation path that uses classical state sequences as weak supervision while explicitly rejecting the idea that HMM states are ground truth.
3. A fold-local downstream alpha test showing that the guided embedding path only becomes useful when paired with sequential HMM assignment rather than memoryless GMM assignment.
4. A validation and evidence layer that includes leakage audit, robustness checks, fold-level statistics, multiple-testing-aware claim language, ablations, and fold-local interpretability.
5. A cautious empirical finding: guided-HMM produces the strongest current point estimates and stress robustness, while statistical dominance over raw-feature HMM remains unproven.
6. A pre-specified Crypto-20 extension showing structural transfer, weak directional IC evidence, and a clear separation between ranking quality, probability calibration, and portfolio performance.

## 2. Related Work

This work sits between four research areas.

- Classical financial regime detection: HMMs, Gaussian mixtures, volatility regimes, and regime-switching models.
- Contrastive time-series representation learning: vanilla temporal contrastive objectives, TS2Vec/TNC/CoST-style representation learning, and recent time-frequency or hard-negative variants.
- Financial ML validation: triple-barrier labels, purging, embargoing, walk-forward evaluation, multiple-testing caution, and Probabilistic Sharpe diagnostics.
- Regime-conditioned alpha modeling: using state information to train, select, or weight predictive models.

The central distinction is that this paper does not treat HMM states as true labels. They are a classical sequential reference that can discipline learned embeddings and provide a competitive baseline. This framing allows the benchmark to ask whether learned regimes benefit from sequential structure without claiming that any fitted state sequence is the real market ontology.

## 3. Data and Labels

The controlled pilot contains BTCUSDT and ETHUSDT hourly bars from 2024-04-26 onward. A separate Crypto-20 universe is pre-specified before multi-asset model evaluation and contains 20 liquid crypto pairs selected through the documented universe and data-quality protocol. The two-asset pilot isolates regime-method behavior under relatively uniform microstructure, while Crypto-20 tests whether structural and predictive behavior survives a wider, more heterogeneous universe.

The feature store contains 22 engineered hourly features covering returns, realized volatility, volatility-of-volatility, liquidity proxies, order-flow proxy behavior, RSI/MACD/Bollinger-style technical state, distribution shape, and volume behavior. The primary target is `tb_label_8h`, an 8-hour triple-barrier label with down, neutral, and up classes. Secondary directional, triple-barrier, forward-return, and volatility-adjusted-return labels are retained for diagnostics and robustness, but the paper reports the primary target first.

The label diagnostics are intentionally part of the artifact set. They document class balance, neutral share, missing target rows, and the expected tail loss from horizon shifting. This prevents the benchmark from hiding label imbalance or silently comparing models on different prediction universes.

## 4. Methods

### 4.1 Baselines

- `global_lgbm`: no-regime LightGBM baseline.
- `regime_lgbm_hmm`: raw-feature Gaussian HMM regime baseline.
- `regime_lgbm_kmeans`: non-sequential clustering baseline.
- `regime_lgbm_vol_bucket`: simple volatility-state baseline.
- `regime_lgbm_contrastive`: vanilla contrastive embedding with GMM assignment.
- `regime_lgbm_contrastive_hmm`: vanilla contrastive embedding with HMM assignment.

### 4.2 Proposed Method

The proposed method is `regime_lgbm_hmm_guided_hmm`: an HMM-guided contrastive encoder followed by sequential HMM assignment and regime-conditioned LightGBM alpha models. The encoder uses the HMM state sequence to define weak same-state positives and boundary-aware negatives, but the downstream regime assignment is still evaluated as a model component rather than assumed correct.

### 4.3 Alpha Modeling

All alpha models use the same primary target, `tb_label_8h`, and the same walk-forward folds. LightGBM outputs multiclass probabilities over down, neutral, and up labels. The alpha score is `P(up) - P(down)`. A position is taken only when the score exceeds the threshold and the neutral class is not dominant. Transaction costs are applied in the evaluation layer rather than treated as an afterthought.

## 5. Validation and Statistical Protocol

The predictive benchmark uses expanding walk-forward validation, a six-month initial training window, one-month test steps, a five-day embargo, and an eight-bar label-horizon purge. This produces 18 pilot folds and 16 Crypto-20 folds. These are limited samples for fold-level significance testing, and the paper treats them as such. The tradeoff is deliberate: paired embargoed folds are statistically low-power, but they are more defensible than treating hundreds of thousands of overlapping hourly labels as independent observations.

Predictive paper claims use fold-local regime assignment artifacts, not the older offline/global regime files used for descriptive plots. The validation audit checks row separation, embargo spacing, target-horizon purge, coverage parity, duplicate predictions, and consistency between predictions and result summaries.

Statistical interpretation is fold-level first. Row-level diagnostics are useful for forecast-loss and calibration discussion, but adjacent financial labels overlap and should not be treated as independent evidence for IC or Sharpe claims. The paper therefore separates point estimates, fold-level tests, multiple-testing diagnostics, and Probabilistic Sharpe diagnostics.

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

The headline result is mechanistic rather than promotional: sequential structure in the assignment layer is the key driver, and HMM-guided representation learning makes that assignment layer more effective. HMM assignment improves the guided learned-regime path relative to guided-GMM on all focused point-estimate metrics and is raw-suggestive on fold-level IC, but it is not corrected significant. Guided-HMM also improves all focused point-estimate metrics versus raw-feature HMM. However, the main guided-HMM versus raw-feature HMM IC comparison has `p=0.801`, so it cannot be framed as a statistical win. This is still useful because it separates a promising mechanism from an overclaimed victory.

### 6.4 Crypto-20 Generalization and Statistical Adjudication

The Phase 36 Crypto-20 benchmark contains 20 assets, 16 paired folds, and `230,400` OOS rows per method. Guided-HMM has the highest mean fold IC (`0.0117`), but its bootstrap interval crosses zero. Its IC difference versus global LightGBM is `+0.00712` (`p=0.0939`, BH `q=0.3216`), versus raw-feature HMM is `+0.00065` (`p=0.8404`), and versus KMeans is `+0.00243` (`p=0.6311`). No fold-level Sharpe or total-return comparison establishes guided-HMM superiority.

A secondary time-block DM test finds worse multiclass negative log-likelihood for guided-HMM than global LightGBM (`p=1.51e-12`, surviving Holm correction) and raw-feature HMM (`p=0.0148`, surviving BH correction within the metric family). Across assets, guided-HMM improves IC over global LightGBM in 13 of 20 assets with mean difference `+0.00521`, but the assets are correlated and the sign test is non-significant. The generalization result is therefore structural and directional, not a statistically confirmed alpha or calibration win.

## 7. Robustness

The robustness evidence has two layers:

- Symbol/horizon robustness: useful for showing where the conclusion is stable or mixed.
- Cost/threshold/market-period stress robustness: stronger for the primary BTC+ETH 8h setup.

The safe wording is that guided-HMM is stress-robust on the primary BTC+ETH 8h benchmark, not universally dominant across all assets and horizons. This distinction matters because a method can be strong under transaction-cost and threshold stress while still showing mixed behavior across target horizons or single-symbol subsets.

## 8. Interpretability

Fold-local feature attribution shows which feature families matter inside each method and regime. This is diagnostic interpretability, not causal explanation. Its strongest use is plausibility: the guided-HMM regime-conditioned models rely heavily on volatility state, volatility-of-volatility, momentum/autocorrelation, and distribution-shape features, which is more convincing than a regime layer driven by arbitrary identifiers.

## 9. Ablations

The ablation suite tests objective guidance, assignment layer, augmentation view, and classical-reference comparisons. The strongest evidence is for the assignment layer: guided embeddings become useful when assignment respects temporal state persistence. The current time-frequency prototype is informative but negative; it should not receive a full downstream expansion until its structural evidence improves.

## 10. Limitations

- HMM states are proxy states, not ground truth.
- The pilot uses BTCUSDT and ETHUSDT as a controlled setting; Crypto-20 broadens the evidence within crypto but does not establish transfer to equities, FX, or other asset classes.
- The fold-level statistical tests use 18 pilot folds and 16 Crypto-20 folds, which is defensible but low-power.
- The guided-HMM edge over raw-feature HMM is directionally supported but not statistically significant at 5%; the main IC comparison has `p=0.801`.
- Backtest results are research diagnostics, not live trading claims.
- Interpretability results are model-specific and not causal.
- The current encoder is trained offline. Fully fold-local encoder training is a required validity experiment before the learned-regime path can support the final predictive claim; it is not delegated to an optional appendix.

## 11. Reproducibility

The public repository separates dashboard reproduction from full research reproduction. The dashboard runs from curated committed summary artifacts and minimal Streamlit dependencies. Full research reproduction uses `requirements-research.txt`, local data access, encoder training, fold-local validation, robustness checks, statistical tests, and validation audit.

The reproduction helper supports three modes: smoke, full, and dashboard. Raw data, DuckDB databases, model weights, embeddings, and row-level prediction files remain excluded from GitHub.

## 12. Conclusion

Adaptive Alpha Lab shows that learned market-regime representations need sequential discipline to become useful in this benchmark. Vanilla contrastive-GMM regimes are weak downstream, while HMM-guided embeddings paired with HMM assignment produce the strongest BTC/ETH point estimates and stress robustness. The pre-specified Crypto-20 study shows that the structural mechanism transfers and that guided-HMM produces weak directional ranking evidence, but it does not establish statistical dominance, superior calibration, or superior portfolio performance. The central publishable finding is therefore conditional rather than promotional: classical sequential structure can improve deep learned regime representations, while the translation from structural quality to economic alpha remains incomplete and measurable.

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
| Crypto-20 generalization | statistical test | models/crypto20_statistical_method_summary.csv; models/crypto20_statistical_claims.csv; reports/crypto20_statistical_protocol.md | Separates structural transfer and directional IC from unsupported alpha/calibration dominance. |
| Mechanism | regime quality | models/regime_quality_summary.csv; models/guided_encoder_comparison.csv | Measures structural alignment, persistence, entropy, NMI, ARI, and purity. |
| Ablations | ablation | models/ablation_summary.csv; models/paper_claim_tests.csv | Tests objective guidance, assignment layer, augmentation view, and classical-reference comparisons. |
| Robustness | stress test | models/robustness_summary.csv; models/robustness_stress_summary.csv | Checks symbol/horizon robustness and cost/threshold/period sensitivity. |
| Interpretability | feature attribution | models/feature_importance_by_regime.csv; models/feature_family_summary.csv | Shows fold-local feature drivers by method/regime without making causal claims. |
| Reproducibility | reproduction package | reproduce.ps1; reports/environment.md; reports/artifact_manifest.md; reports/reproduction_checklist.md | Documents smoke/full/dashboard reproduction paths, environment split, and artifact policy. |
