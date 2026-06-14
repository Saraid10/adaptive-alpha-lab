# Adaptive Alpha Lab Paper Protocol

## Protocol Status

This document freezes the paper-facing research protocol after Phase 23. Future experiments should change this file only through a new phase, not casually while running models.

Protocol version: Phase 24

Primary working title:

```text
HMM-Guided Contrastive Representations for Regime-Conditioned Financial Alpha Modeling
```

## Central Research Question

Do HMM-guided contrastive regime representations improve regime-conditioned financial alpha modeling compared with vanilla contrastive regimes, classical raw-feature HMM regimes, and a global no-regime LightGBM baseline under financial labels, purged walk-forward validation, transaction costs, robustness checks, statistical tests, and fold-local interpretability?

## Contribution Boundary

The project does not claim to invent HMMs, contrastive learning, LightGBM, triple-barrier labels, or market-regime modeling.

The contribution is a controlled empirical benchmark and model-side intervention:

1. A fair regime-conditioned alpha benchmark using common financial labels, common folds, common transaction costs, and common test rows.
2. Evidence that vanilla contrastive regimes are structurally stable but weakly aligned with classical sequential regimes and downstream alpha usefulness.
3. An HMM-guided contrastive training objective that uses classical state sequences as weak supervision while still treating them as proxy states, not ground truth.
4. A downstream fold-local test showing that guided embeddings plus HMM assignment become the strongest current point-estimate alpha method, while statistical dominance remains inconclusive.
5. Robustness, statistical, and interpretability artifacts that make the finding auditable rather than purely anecdotal.

## Dataset Freeze

Current paper dataset:

| Field | Value |
|---|---|
| Symbols | BTCUSDT, ETHUSDT |
| Bar frequency | 1 hour |
| Earliest OHLCV timestamp | 2024-04-26 16:30:00+05:30 |
| Latest OHLCV timestamp | 2026-04-26 15:30:00+05:30 |
| OHLCV rows per symbol | 17,520 |
| Feature rows per symbol | 17,460 |
| Primary label | `tb_label_8h` |
| Primary return column | `forward_return_8h` |
| Primary universe | BTCUSDT+ETHUSDT |

Raw data files and DuckDB databases are local research inputs and are not committed.

## Feature Freeze

The paper benchmark uses 22 engineered hourly features:

```text
ret_1h, ret_5h, ret_15h, ret_60h,
vol_5h, vol_20h, vol_of_vol,
amihud, volume_zscore,
ret_autocorr, spread_proxy, ofi_proxy,
rsi_14, gk_vol, skewness, kurtosis,
macd_signal, bband_pct_b, atr_14,
close_vs_vwap, log_vol_trend, ret_dispersion
```

Feature changes after this protocol should be treated as a new experiment family because they change both regime discovery and alpha modeling.

## Target Freeze

Primary paper target:

```text
tb_label_8h
```

Secondary diagnostic targets:

```text
dir_label_4h, dir_label_8h, dir_label_24h
tb_label_4h, tb_label_8h, tb_label_24h
forward_return_4h, forward_return_8h, forward_return_24h
vol_adj_return_4h, vol_adj_return_8h, vol_adj_return_24h
```

The paper should report the primary target first. Secondary horizons belong in robustness or appendix tables.

## Validation Freeze

Predictive claims must use:

| Field | Value |
|---|---|
| Validation style | Expanding walk-forward |
| Initial training | 6 months, 4,320 hourly bars |
| Test step | 1 month, 720 hourly bars |
| Embargo | 5 days, 120 hourly bars |
| Primary label-horizon purge | 8 bars |
| Main OOS rows | 25,920 per method |
| Transaction cost | 10 bps per trade unless stress-testing |

Offline/global regime files may be used for descriptive plots. They must not be used as predictive evidence unless explicitly labeled as offline/global.

## Methods Freeze

Primary comparison methods:

| Method | Role |
|---|---|
| `global_lgbm` | No-regime baseline |
| `regime_lgbm_hmm` | Classical raw-feature HMM baseline |
| `regime_lgbm_kmeans` | Non-sequential clustering baseline |
| `regime_lgbm_vol_bucket` | Simple volatility-state baseline |
| `regime_lgbm_contrastive` | Vanilla learned-regime baseline |
| `regime_lgbm_contrastive_hmm` | Vanilla embedding plus sequential assignment |
| `regime_lgbm_hmm_guided_gmm` | HMM-guided embedding plus GMM assignment |
| `regime_lgbm_hmm_guided_hmm` | Main proposed method |

The current proposed method is `regime_lgbm_hmm_guided_hmm`.

## Metrics Freeze

Primary predictive/economic metrics:

```text
IC, Sharpe, drawdown, total_return, turnover, n_trades, n_test_rows
```

Required statistical diagnostics:

```text
fold-level paired tests, bootstrap confidence intervals,
DM-style NLL forecast-loss diagnostics, multiple-testing correction,
Probabilistic Sharpe Ratio
```

Required regime-quality diagnostics:

```text
silhouette, average duration, transition diagonal, entropy,
NMI/ARI versus raw-feature HMM reference, purity versus raw-feature HMM reference
```

Required interpretability diagnostics:

```text
fold-local LightGBM gain/split importance, capped SHAP summary,
feature-family attribution by method/regime
```

## Permitted Claims

The paper may claim:

1. Vanilla contrastive regimes underperform raw-feature HMM regimes in the current benchmark.
2. Sequential assignment matters: HMM assignment is more useful than GMM assignment on learned embeddings.
3. HMM-guided weak supervision strongly improves structural agreement with the raw-feature HMM reference.
4. Guided embeddings plus HMM assignment produce the strongest current point estimates on the primary BTC+ETH 8h benchmark.
5. Guided-HMM is stress-robust on the primary BTC+ETH 8h prediction file.
6. Fold-local interpretability indicates economically plausible drivers dominated by volatility state, momentum/autocorrelation, and distribution shape.

## Forbidden Claims

The paper must not claim:

1. HMM states are ground-truth market regimes.
2. The strategy is profitable or deployable.
3. The guided-HMM method is statistically dominant over raw-feature HMM at 5% significance.
4. The result generalizes beyond BTC/ETH.
5. The interpretability results are causal explanations.
6. Offline/global regime artifacts prove predictive performance.

## Decision Gates

Phase 25 ablations and Phase 26 paper claim tests are mandatory before paper submission. Both are now complete in the current protocol.

Phase 27 drafts the paper skeleton before additional scope expansion. Multi-asset expansion is conditional. Proceed only if Phase 25/26 shows either:

1. a statistically meaningful guided-HMM improvement over raw-feature HMM, or
2. a robust enough stress/interpretable result that a generalization appendix is worth the compute.

The current Phase 26 read does not show statistically decisive guided-HMM dominance over raw-feature HMM. Therefore, the default paper is a crypto-market benchmark with multi-asset generalization as future work, unless a separate scope decision changes that plan.

Phase 27 is represented by `paper/main.md`, `reports/paper_artifact_map.csv`, and `reports/paper_submission_checklist.md`. Phase 28 is represented by `reproduce.ps1`, `reports/environment.md`, `reports/artifact_manifest.md`, and `reports/reproduction_checklist.md`.

Phase 29 turns the scaffold into manuscript-style prose while preserving the claim registry. The next default step is venue formatting and citation cleanup, not broad experiment expansion.

Reviewer-facing caveats must stay explicit in the paper draft:

1. BTC/ETH is a controlled crypto setting, not a broad asset-class generalization claim.
2. Eighteen walk-forward folds limit statistical power, but are more defensible than row-level independence over overlapping labels.
3. The `p=0.801` guided-HMM versus raw-feature HMM IC result prevents a statistical dominance claim; the main contribution is the sequential-assignment mechanism.

