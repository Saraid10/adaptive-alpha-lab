# Adaptive Alpha Lab

> A research-grade quant ML benchmark platform that tests whether learned market regimes improve alpha modeling versus global baselines and classical regime methods, using proper financial labels, purged walk-forward validation, and transaction-cost-aware evaluation.

## Research Question

Does learning market regimes from raw financial time-series features improve alpha-model IC, drawdown, and Sharpe compared with:

- a global model with no regime awareness
- Gaussian HMM regimes
- KMeans regimes
- volatility-bucket regimes

The project is intentionally framed as a research benchmark, not a live trading bot. Honest weak or mixed results are part of the contribution.

## Architecture

```text
Binance OHLCV
    -> DuckDB feature store
    -> 22 engineered market/microstructure features
    -> Multi-horizon financial labels and triple-barrier targets
    -> Contrastive temporal regime encoder + classical baselines
    -> Global and regime-conditioned LightGBM models
    -> Purged walk-forward validation with transaction costs
    -> Research dashboard and report artifacts
```

## Current Capabilities

- Incremental Binance OHLCV ingestion for BTCUSDT and ETHUSDT.
- DuckDB feature store with 22 engineered features.
- Contrastive Transformer-style temporal encoder trained with NT-Xent loss.
- GMM regime posteriors and UMAP/timeline visualizations.
- Multi-horizon targets: 4h, 8h, and 24h.
- Triple-barrier labels with neutral class.
- Regime baselines: contrastive, Gaussian HMM, KMeans, volatility buckets.
- Global LightGBM baseline and regime-conditioned LightGBM models.
- 5-day embargoed walk-forward validation.
- Transaction-cost-aware experiment result table.
- Streamlit dashboard shell and research note.

## Run Order

Recommended local environment:

```powershell
py -3.11 -m venv env
.\env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```powershell
python src/check.py
python src/targets.py --symbols BTCUSDT ETHUSDT
python src/train_encoder.py --symbols BTCUSDT ETHUSDT
python src/visualize_regimes.py --symbols BTCUSDT ETHUSDT
python src/baselines.py --symbols BTCUSDT ETHUSDT
python src/alpha_models.py --symbols BTCUSDT ETHUSDT
python src/backtest.py
python -m compileall src dashboard.py
```

Optional dashboard:

```powershell
streamlit run dashboard.py
```

If Streamlit is not installed, run:

```powershell
pip install -r requirements.txt
```

## Key Artifacts

| Artifact | Purpose |
|---|---|
| `models/target_distribution.csv` | Label balance across horizons and label types |
| `models/target_quality.csv` | Missing-row, horizon-loss, and neutral-class checks |
| `models/regime_assignments.csv` | Aligned regime labels/posteriors for all methods |
| `models/regime_benchmark_summary.csv` | Regime-level comparison table |
| `models/per_regime_stats.csv` | Volatility, return, liquidity, and IC diagnostics by regime |
| `models/experiment_results.csv` | Master model comparison table |
| `models/alpha_oos_predictions.csv` | Out-of-sample model predictions |
| `models/phase4_dashboard.png` | Static research backtest dashboard |
| `reports/adaptive_alpha_lab_report.md` | Research note |

## Project Structure

```text
adaptive-alpha-engine/
├── dashboard.py
├── reports/
│   └── adaptive_alpha_lab_report.md
├── src/
│   ├── ingestion.py
│   ├── features.py
│   ├── targets.py
│   ├── dataset.py
│   ├── encoder.py
│   ├── train_encoder.py
│   ├── visualize_regimes.py
│   ├── baselines.py
│   ├── alpha_models.py
│   ├── backtest.py
│   └── check.py
└── models/
    └── generated research artifacts
```

## Methodology Notes

The primary target is `tb_label_8h`, an 8-hour triple-barrier label with classes `-1`, `0`, and `+1`. The alpha score is `P(+1) - P(-1)`. A prediction becomes a trade only when the neutral class is not dominant and the absolute score exceeds the threshold.

The validation scheme uses expanding walk-forward folds with a 5-day embargo gap between train and test windows. This reduces leakage from overlapping financial labels and makes the model comparison more defensible.

Dense contrastive regime inference uses stride 1 after the encoder window warmup, so the learned-regime method is compared on the same BTC+ETH row universe as HMM-style, KMeans, and volatility-bucket baselines.

## Latest Benchmark Snapshot

Latest run: BTCUSDT + ETHUSDT, `tb_label_8h`, 25,920 out-of-sample rows per method.

| Method | IC | Sharpe | Drawdown | Note |
|---|---:|---:|---:|---|
| global_lgbm | 0.0024 | -0.506 | -0.688 | no-regime baseline |
| regime_lgbm_contrastive | -0.0165 | -0.902 | -0.909 | learned-regime benchmark |
| regime_lgbm_hmm | 0.0079 | -0.182 | -0.829 | best IC and Sharpe; true hmmlearn HMM |
| regime_lgbm_kmeans | -0.0013 | -1.180 | -0.920 | classical clustering baseline |
| regime_lgbm_vol_bucket | -0.0030 | -0.988 | -0.896 | simple volatility baseline |

The current result is intentionally presented as research evidence, not a profitable trading claim. The true HMM baseline is strongest in this run, while learned contrastive regimes remain an important benchmark for testing whether representation learning can beat classical regime discovery.

## Current Status

The codebase now produces the full benchmark artifact set. The next important work is not live trading; it is improving the learned-regime encoder, adding feature importance analysis, and running threshold/turnover studies.

## Resume Bullets

- Built Adaptive Alpha Lab, a quant ML benchmark platform using DuckDB, PyTorch, LightGBM, and purged walk-forward validation to test whether learned market regimes improve alpha modeling under transaction costs.
- Benchmarked contrastive temporal regimes against HMM-style, KMeans, and volatility-bucket baselines using transition matrices, regime duration, per-regime diagnostics, IC, drawdown, and turnover.
- Implemented multi-horizon financial labels, 8-hour triple-barrier targets, posterior-weighted regime-conditioned models, and transaction-cost-aware out-of-sample evaluation.
