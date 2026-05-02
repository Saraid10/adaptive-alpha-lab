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

The benchmark compares four regime methods on a common BTC+ETH universe:

| Method | Implementation | Rows | Silhouette | Avg Duration |
|---|---|---:|---:|---:|
| contrastive | contrastive encoder + GMM | 34,754 | 0.0959 | 30.51 |
| hmm | hmmlearn Gaussian HMM | 34,754 | 0.0790 | 6.98 |
| kmeans | sklearn KMeans | 34,754 | 0.0967 | 4.05 |
| vol_bucket | realized-volatility quantiles | 34,754 | -0.0393 | 10.44 |

The contrastive method now uses dense stride-1 inference for every valid feature row after the 60-bar encoder window. This fixes the earlier sparse-coverage issue and makes downstream alpha comparisons fair.

## Validation Setup

Alpha models use expanding walk-forward validation with:

- 6-month initial training window
- 1-month test step
- 5-day embargo between train and test
- multiclass LightGBM over labels `-1`, `0`, `+1`
- transaction cost of 10 bps per trade

The global baseline trains one LightGBM model across both symbols. Regime-aware models train separate LightGBM classifiers per regime and combine predictions through posterior weights where available. The alpha score is `P(+1) - P(-1)`, and trades are taken only when the neutral class is not dominant and the score clears the threshold.

## Model Comparison

| Method | IC | Accuracy | Balanced Accuracy | Sharpe | Drawdown | Turnover | Total Return |
|---|---:|---:|---:|---:|---:|---:|---:|
| global_lgbm | 0.0024 | 0.5736 | 0.3625 | -0.506 | -0.688 | 0.050 | -0.557 |
| regime_lgbm_contrastive | -0.0165 | 0.5616 | 0.3724 | -0.902 | -0.909 | 0.077 | -0.855 |
| regime_lgbm_hmm | 0.0079 | 0.5608 | 0.3742 | -0.182 | -0.829 | 0.085 | -0.403 |
| regime_lgbm_kmeans | -0.0013 | 0.5647 | 0.3670 | -1.180 | -0.920 | 0.079 | -0.912 |
| regime_lgbm_vol_bucket | -0.0030 | 0.5566 | 0.3670 | -0.988 | -0.896 | 0.083 | -0.872 |

All methods are evaluated on 25,920 out-of-sample rows. This equal test coverage is important: it prevents a regime method from looking better simply because it was tested on a smaller or easier subset.

## Results Interpretation

The real Gaussian HMM baseline produces the strongest IC in this run, improving from 0.0024 for the global baseline to 0.0079. It also has the least negative Sharpe among the tested methods, although drawdown remains large and total return remains negative after transaction costs.

The honest conclusion is mixed: regime awareness changes ranking behavior and risk characteristics, but the current learned contrastive regimes do not beat the classical HMM baseline on the primary out-of-sample alpha metrics. This is still a useful research result because it separates representation learning, regime diagnostics, and deployable alpha instead of overclaiming profitability.

## Limitations

- Hourly OHLCV is a noisy signal source.
- The current contrastive encoder is not a true Temporal Fusion Transformer.
- Backtest returns are research diagnostics, not deployable trading evidence.
- The project intentionally excludes live trading, RL, online retraining, and order-book data in this phase.

## Next Steps

1. Improve the contrastive encoder objective and compare it against the current HMM winner.
2. Add feature importance and SHAP summaries for global and regime-aware LightGBM models.
3. Tune score thresholds to separate predictive IC from executable turnover.
4. Add a short dashboard demo recording and final resume bullets.
