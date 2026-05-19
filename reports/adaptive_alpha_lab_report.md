# Adaptive Alpha Lab Research Note

## Problem Statement

Financial markets are non-stationary: a signal that works in one market state can fail in another. Adaptive Alpha Lab tests whether learned market regimes improve alpha modeling compared with a global no-regime model and classical regime baselines under financial labels, purged walk-forward validation, and transaction costs.

The central question is:

> Do learned regimes improve IC, drawdown, Sharpe, or turnover versus global and classical-regime baselines?

## Data And Features

The current benchmark uses hourly Binance OHLCV data for BTCUSDT and ETHUSDT. Each symbol has 17,520 OHLCV bars and 17,460 feature rows after indicator warmup. The feature store contains 22 engineered technical and microstructure-inspired features, including multi-horizon returns, realized volatility, volatility of volatility, Amihud illiquidity, volume z-score, return autocorrelation, spread proxy, order-flow proxy, RSI, Garman-Klass volatility, ATR, close-vs-VWAP, log volume trend, and return dispersion.

## Target Labeling

The primary modeling target is `tb_label_8h`, an 8-hour triple-barrier label with classes `-1`, `0`, and `+1`. The neutral class captures periods where neither the profit barrier nor the stop barrier is hit before time expiry. Direction and volatility-adjusted labels are also generated at 4-hour, 8-hour, and 24-hour horizons.

The final target table contains 34,872 rows across BTCUSDT and ETHUSDT. For the primary 8-hour triple-barrier target, the neutral class is about 58.3% for both symbols, with up/down classes each near 20.6-21.0%.

## Regime Methods

The benchmark compares five regime methods on a common BTC+ETH universe:

| Method | Implementation | Rows | Silhouette | Avg Duration |
|---|---|---:|---:|---:|
| contrastive | contrastive encoder + GMM | 34,754 | 0.0959 | 30.51 |
| contrastive_hmm | contrastive embeddings + HMM | 34,754 | 0.1016 | 42.18 |
| hmm | hmmlearn Gaussian HMM | 34,754 | 0.0790 | 6.98 |
| kmeans | sklearn KMeans | 34,754 | 0.0967 | 4.05 |
| vol_bucket | realized-volatility quantiles | 34,754 | -0.0393 | 10.44 |

The contrastive method now uses dense stride-1 inference for every valid feature row after the 60-bar encoder window. This fixes the earlier sparse-coverage issue and makes downstream alpha comparisons fair.

Phase 11 adds a contrastive-HMM hybrid. Instead of clustering learned embeddings with GMM only, it fits a Gaussian HMM directly on the learned contrastive embedding sequence. This tests whether the weakness in contrastive regimes comes from representation learning itself or from the lack of temporal state dynamics in the assignment layer.

## Regime Stability Diagnostics

Phase 10 adds explicit stability diagnostics to separate regime persistence from downstream usefulness. This matters because a regime method can look visually smooth while still being weak for alpha conditioning.

| Method | Switches / 1k Bars | Avg Duration | Transition Diagonal | Stable IC | Transition IC |
|---|---:|---:|---:|---:|---:|
| contrastive | 32.72 | 30.51 | 0.967 | -0.0200 | 0.0482 |
| contrastive_hmm | 23.65 | 42.18 | 0.976 | 0.0033 | 0.0100 |
| hmm | 143.24 | 6.98 | 0.857 | 0.0084 | -0.0061 |
| kmeans | 246.60 | 4.05 | 0.753 | 0.0083 | -0.0137 |
| vol_bucket | 95.71 | 10.44 | 0.904 | 0.0001 | -0.0168 |

The important finding is not simply that more stable regimes are better. Contrastive-HMM produces the longest-lived regimes and repairs the negative stable-period IC seen in contrastive-GMM. However, the raw-feature HMM still has the strongest stable-period IC. This suggests that HMM-style temporal structure helps learned embeddings, but the current embedding space is still not as alpha-aligned as the simpler raw-feature HMM state space.

## Validation Setup

Alpha models use expanding walk-forward validation with:

- 6-month initial training window
- 1-month test step
- 5-day embargo between train and test
- multiclass LightGBM over labels `-1`, `0`, `+1`
- transaction cost of 10 bps per trade

The global baseline trains one LightGBM model across both symbols. Regime-aware models train separate LightGBM classifiers per regime and combine predictions through posterior weights where available. The alpha score is `P(+1) - P(-1)`, and trades are taken only when the neutral class is not dominant and the score clears the threshold.

## Validation Audit

Phase 12 adds an explicit validation audit so the benchmark can be evaluated as research evidence rather than a collection of backtest claims. The audit checks database tables, feature/target schema, finite joined rows, target horizon tail loss, common benchmark coverage, fold separation, embargo spacing, label-horizon purging, row-level prediction alignment, duplicate predictions, and consistency between `alpha_oos_predictions.csv` and `experiment_results.csv`.

The audit result is:

| Status | Count | Interpretation |
|---|---:|---|
| PASS | 17 | All critical data, fold, target, coverage, and prediction-alignment checks passed |
| WARN | 1 | Regime assignments are currently offline/global artifacts |
| FAIL | 0 | No critical validation failure was detected |

The most important positive result is that all 18 folds satisfy row separation, the 120-bar embargo, and the 8-bar primary label-horizon purge. All six alpha methods also have equal out-of-sample prediction coverage of 25,920 rows.

The warning is methodological rather than a code failure: current regime labels are generated as offline/global artifacts before alpha-model validation. This is acceptable for descriptive regime analysis and current benchmark exploration, but a peer-reviewed predictive claim should add Phase 13 fold-local regime refitting, where regime models are fitted using only training history inside each walk-forward fold.

## Model Comparison

| Method | IC | Accuracy | Balanced Accuracy | Sharpe | Drawdown | Turnover | Total Return |
|---|---:|---:|---:|---:|---:|---:|---:|
| global_lgbm | 0.0024 | 0.5736 | 0.3625 | -0.506 | -0.688 | 0.050 | -0.557 |
| regime_lgbm_contrastive | -0.0165 | 0.5616 | 0.3724 | -0.902 | -0.909 | 0.077 | -0.855 |
| regime_lgbm_contrastive_hmm | 0.0035 | 0.5585 | 0.3720 | -0.382 | -0.852 | 0.079 | -0.588 |
| regime_lgbm_hmm | 0.0051 | 0.5622 | 0.3727 | -0.229 | -0.825 | 0.081 | -0.446 |
| regime_lgbm_kmeans | -0.0013 | 0.5647 | 0.3670 | -1.180 | -0.920 | 0.079 | -0.912 |
| regime_lgbm_vol_bucket | -0.0030 | 0.5566 | 0.3670 | -0.988 | -0.896 | 0.083 | -0.872 |

All methods are evaluated on 25,920 out-of-sample rows. This equal test coverage is important: it prevents a regime method from looking better simply because it was tested on a smaller or easier subset.

## Results Interpretation

The real Gaussian HMM baseline produces the strongest IC in this run, improving from 0.0024 for the global baseline to 0.0051. It also has the least negative Sharpe among the tested methods, although drawdown remains large and total return remains negative after transaction costs.

The honest conclusion is mixed but stronger than before. The contrastive-HMM hybrid improves the learned-regime path substantially: IC moves from -0.0165 to 0.0035, and Sharpe improves from -0.902 to -0.382. That validates the Phase 11 hypothesis that temporal state dynamics help learned embeddings. However, the raw-feature HMM still beats the hybrid on IC and Sharpe, so the current learned representation is not yet superior to the simpler classical state model.

This is still a useful research result because it separates representation learning, regime diagnostics, and deployable alpha instead of overclaiming profitability. The project now has a clear scientific progression: dense contrastive regimes underperform, stability diagnostics identify the assignment-layer weakness, and contrastive-HMM partially fixes it.

## Limitations

- Hourly OHLCV is a noisy signal source.
- The current contrastive encoder is not a true Temporal Fusion Transformer.
- Regime assignments are currently offline/global artifacts; fold-local regime refitting is the next paper-grade validation upgrade.
- Backtest returns are research diagnostics, not deployable trading evidence.
- The project intentionally excludes live trading, RL, online retraining, and order-book data in this phase.

## Next Steps

1. Implement fold-local regime refitting so predictive regime labels are fitted only on training history.
2. Run robustness checks across thresholds, costs, horizons, and symbol splits.
3. Add block-bootstrap confidence intervals for IC and paired method differences.
4. Improve the learned encoder objective and retest the contrastive-HMM hybrid.
5. Add feature importance and SHAP summaries for global and regime-aware LightGBM models.
